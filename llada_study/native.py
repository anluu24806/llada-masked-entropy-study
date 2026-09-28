"""Run the hash-pinned native sampler with passive pre-denoising observations.

No sampling operator is reimplemented. Function definitions are compiled from
vendor/generate.py unchanged; only unused imports and the CLI entry point are
excluded, so CPU tests do not require Transformers. A PyTorch forward hook reads
raw output logits before the native sampler applies noise or updates its canvas.
"""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SOURCE_SHA256 = "d4d2d3c015511e63fe9237cd3197346148fb0c795b6c89e0b5ef4846be25252e"
MODEL_REPOSITORY = "GSAI-ML/LLaDA-8B-Instruct"
MODEL_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
DEFAULT_CONFIG = {
    "steps": 128, "gen_length": 256, "block_length": 256,
    "temperature": 1.0, "cfg_scale": 0.0, "remasking": "low_confidence",
    "mask_id": 126336, "logits_eos_inf": False,
    "confidence_eos_eot_inf": False, "end_think_token_ids": None,
    "end_think_logit_boost": 0.0,
}
TRACE_MAX_BYTES = 9_000_000


def load_pinned_generate(source_file):
    """Load unchanged native function ASTs after checking the exact source hash."""
    source_file = Path(source_file)
    source = source_file.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"native source hash mismatch: {digest}")
    original = ast.parse(source, filename=str(source_file))
    functions = [node for node in original.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name != "main"]
    tree = ast.Module(body=functions, type_ignores=[])
    namespace = {"torch": torch, "np": np, "F": F, "__name__": "llada_native_pinned"}
    exec(compile(tree, str(source_file), "exec"), namespace)
    return namespace["generate"]


def _native_config(config):
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise TypeError("config must be a flat dictionary of native sampler arguments")
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unsupported native config keys: {sorted(unknown)}")
    settings = {**DEFAULT_CONFIG, **config}
    for name in ("steps", "gen_length", "block_length", "mask_id"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if value < (0 if name == "mask_id" else 1):
            raise ValueError(f"invalid {name}")
    if settings["gen_length"] % settings["block_length"]:
        raise ValueError("gen_length must be divisible by block_length")
    blocks = settings["gen_length"] // settings["block_length"]
    if settings["steps"] % blocks or settings["steps"] > settings["gen_length"]:
        raise ValueError("steps must be <= gen_length and divisible by the block count")
    if not math.isfinite(settings["temperature"]) or settings["temperature"] < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if settings["cfg_scale"] != 0 or settings["remasking"] != "low_confidence":
        raise ValueError("this study requires cfg_scale=0 and low_confidence remasking")
    if (settings["logits_eos_inf"] or settings["confidence_eos_eot_inf"]
            or settings["end_think_logit_boost"] != 0
            or settings["end_think_token_ids"]):
        raise ValueError("EOS and end-of-thinking interventions are prohibited")
    return settings


class _TraceObserver:
    def __init__(self, prompt_length, gen_length, mask_id):
        self.prompt_length = prompt_length
        self.gen_length = gen_length
        self.mask_id = mask_id
        self.entropy, self.proposals, self.masks, self.canvas = [], [], [], []

    @torch.no_grad()
    def observe(self, tokens, logits):
        if (tokens.ndim != 2 or tokens.shape[0] != 1 or logits.ndim != 3
                or tuple(logits.shape[:2]) != tuple(tokens.shape)):
            raise ValueError("observer requires one completion and matching [1,L,V] logits")
        vocab = logits.shape[-1]
        if not 1 < vocab < 2**31:
            raise ValueError("vocabulary size must be between 2 and int32 capacity")
        generated = tokens[0, self.prompt_length:]
        if generated.numel() != self.gen_length:
            raise ValueError("native forward canvas length differs from generation length")
        # Copy now: the native sampler subsequently updates tokens in place.
        self.canvas.append(generated.detach().cpu().numpy().astype(np.int32, copy=True))
        self.masks.append(generated.eq(self.mask_id).detach().cpu().numpy().copy())
        entropy, proposals = [], []
        for start in range(0, self.gen_length, 16):
            raw = logits[0, self.prompt_length + start:
                         self.prompt_length + min(start + 16, self.gen_length)]
            if not bool(torch.isfinite(raw).all().item()):
                raise ValueError("nonfinite raw logits: measurement is incomplete")
            # FP32 is explicit, independent of the model's BF16 inference dtype.
            p = torch.softmax(raw.float(), dim=-1)
            # xlogy defines the entropy contribution of p=0 as zero.
            h = -torch.special.xlogy(p, p).sum(dim=-1) / math.log(vocab)
            if not bool(torch.isfinite(h).all().item()):
                raise ValueError("nonfinite FP32 normalized entropy")
            entropy.append(h.detach().cpu().numpy().astype(np.float32, copy=True))
            proposals.append(raw.argmax(dim=-1).detach().cpu().numpy().astype(np.int32, copy=True))
        self.entropy.append(np.concatenate(entropy))
        self.proposals.append(np.concatenate(proposals))

    def finish(self):
        if not self.entropy:
            raise ValueError("native sampler produced no observations")
        trace = {name: np.stack(getattr(self, name))
                 for name in ("entropy", "proposals", "masks", "canvas")}
        if sum(value.nbytes for value in trace.values()) > TRACE_MAX_BYTES:
            raise ValueError("raw realization trace exceeds the 9 MB budget")
        return trace


def _rng_state():
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
    }


def _state_hashes(state):
    digest = lambda tensor: hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()
    return {"cpu": digest(state["cpu"]), "cuda": [digest(s) for s in state["cuda"]]}


class NativeRunner:
    """Pinned sampler adapter. ``run`` returns completion IDs and [T,G] arrays.

    ``config`` is a flat native configuration (see DEFAULT_CONFIG). Runs seed
    Torch once per single completion and restore the caller's RNG state. The
    source sampler determines all proposals used for actual sampling, transfer
    counts, selections, and final tokens. The observed argmax is diagnostic only.
    """
    def __init__(self, model, tokenizer, source_file):
        self.model = model
        self.tokenizer = tokenizer
        self.source_file = Path(source_file)
        self.generate = load_pinned_generate(self.source_file)
        self.last_run_metadata = None

    def _execute(self, prompt_ids, seed, config, observe):
        settings = _native_config(config)
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        device = self.model.device
        prompt = torch.as_tensor(prompt_ids, device=device)
        if prompt.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("prompt_ids must contain integer token IDs")
        if prompt.ndim == 1:
            prompt = prompt.unsqueeze(0)
        if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] == 0:
            raise ValueError("exactly one nonempty prompt is required per seed")
        prompt = prompt.to(torch.long)
        if bool((prompt < 0).any().item()) or bool((prompt == settings["mask_id"]).any().item()):
            raise ValueError("prompt IDs must be nonnegative and must not contain MASK")
        observer = _TraceObserver(prompt.shape[1], settings["gen_length"], settings["mask_id"]) if observe else None
        calls = 0

        def hook(module, args, output):
            nonlocal calls
            calls += 1
            if observer is not None:
                observer.observe(args[0], output.logits)
            # Returning None preserves the original output object.
            return None

        handle = self.model.register_forward_hook(hook)
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(seed))
                if devices:
                    torch.cuda.manual_seed_all(int(seed))
                with torch.no_grad():
                    output = self.generate(self.model, prompt, **settings)
                rng = _rng_state()
                final_ids = output[0, prompt.shape[1]:].detach().cpu().tolist()
        finally:
            handle.remove()
        if calls != settings["steps"]:
            raise AssertionError(f"expected {settings['steps']} forward calls, got {calls}")
        trace = observer.finish() if observer is not None else None
        metadata = {
            "seed": int(seed), "forward_calls": calls,
            "source_sha256": SOURCE_SHA256, "rng_after": _state_hashes(rng),
            "final_mask_count": sum(token == settings["mask_id"] for token in final_ids),
        }
        return final_ids, trace, metadata, rng

    def run(self, prompt_ids, seed, config=None):
        final_ids, trace, metadata, _ = self._execute(prompt_ids, seed, config, observe=True)
        self.last_run_metadata = metadata
        return final_ids, trace


def load_native_model(model_dir, source_file):
    """Load the locally staged pinned LLaDA model on CUDA, with no network fetch.

    The orchestration layer verifies staged model/tokenizer file hashes against
    model_pin.json before calling this function. The sampler hash is checked here.
    """
    load_pinned_generate(source_file)
    if not torch.cuda.is_available():
        raise RuntimeError("real LLaDA inference requires CUDA; CPU tests use NativeRunner with a dummy model")
    from transformers import AutoModel, AutoTokenizer
    model_dir = str(Path(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id == DEFAULT_CONFIG["mask_id"]:
        raise ValueError("native sampler requires padding token distinct from MASK")
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    return NativeRunner(model, tokenizer, source_file)


def equivalence_check(runner, config=None, seed=1729, prompt_ids=None):
    """Compare observed and unobserved native runs on one synthetic prompt.

    This is a mechanical check, never a task/answer evaluation. Exact completion
    IDs, post-run CPU and all CUDA RNG states, and forward counts must match.
    Raises on any mismatch and returns a JSON-serializable pass record otherwise.
    """
    if prompt_ids is None:
        prompt_ids = runner.tokenizer.encode("A synthetic sampler equivalence check.", add_special_tokens=True)
    plain_ids, _, plain_meta, plain_rng = runner._execute(prompt_ids, seed, config, observe=False)
    seen_ids, trace, seen_meta, seen_rng = runner._execute(prompt_ids, seed, config, observe=True)
    token_equal = plain_ids == seen_ids
    cpu_equal = torch.equal(plain_rng["cpu"], seen_rng["cpu"])
    cuda_equal = len(plain_rng["cuda"]) == len(seen_rng["cuda"]) and all(
        torch.equal(a, b) for a, b in zip(plain_rng["cuda"], seen_rng["cuda"])
    )
    forward_equal = plain_meta["forward_calls"] == seen_meta["forward_calls"] == trace["entropy"].shape[0]
    record = {
        "passed": bool(token_equal and cpu_equal and cuda_equal and forward_equal),
        "tokens_exact": token_equal, "cpu_rng_exact": cpu_equal,
        "cuda_rng_exact": cuda_equal, "forward_count_exact": forward_equal,
        "cuda_rng_states_checked": len(plain_rng["cuda"]),
        "plain_forward_calls": plain_meta["forward_calls"],
        "observed_forward_calls": seen_meta["forward_calls"],
        "seed": int(seed), "prompt_kind": "synthetic_no_task_answer",
        "device": str(runner.model.device),
        "source_sha256": SOURCE_SHA256,
        "plain_rng_after": plain_meta["rng_after"],
        "observed_rng_after": seen_meta["rng_after"],
        "trace_bytes": sum(array.nbytes for array in trace.values()),
    }
    if not record["passed"]:
        raise AssertionError(f"passive observer changed native sampling: {record}")
    return record
