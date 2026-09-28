import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from llada_study.measurement import summarize_trace
from llada_study.native import (
    NativeRunner, SOURCE_SHA256, _TraceObserver, equivalence_check, load_pinned_generate,
)

SOURCE = Path(__file__).resolve().parents[1] / "vendor" / "generate.py"
SMALL_CONFIG = {"steps": 4, "gen_length": 8, "block_length": 8,
                "temperature": 1.0, "mask_id": 11}


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("anchor", torch.zeros(()))
        self.config = SimpleNamespace(_name_or_path="dummy-llada")
        self.inputs = []
        self.outputs = []
        self.eval()

    @property
    def device(self):
        return self.anchor.device

    def forward(self, tokens, attention_mask=None):
        self.inputs.append(tokens.detach().clone())
        vocabulary = torch.arange(12, device=tokens.device, dtype=torch.float32)
        positions = torch.arange(tokens.shape[1], device=tokens.device, dtype=torch.float32)
        context = tokens.float().sum(dim=-1)[:, None, None]
        logits = torch.cos(vocabulary[None, None, :] * .37 + positions[None, :, None] * .11 + context * .01)
        logits[:, :, 11] = -100.
        self.outputs.append(logits.detach().clone())
        return SimpleNamespace(logits=logits)


class DummyTokenizer:
    def encode(self, text, add_special_tokens=True):
        return [1, 2, 3]


class NativeTests(unittest.TestCase):
    def make_runner(self):
        return NativeRunner(DummyModel(), DummyTokenizer(), SOURCE)

    def test_native_observed_unobserved_exact_equivalence(self):
        runner = self.make_runner()
        record = equivalence_check(runner, SMALL_CONFIG)
        self.assertTrue(record["passed"])
        self.assertTrue(record["tokens_exact"])
        self.assertTrue(record["cpu_rng_exact"])
        self.assertTrue(record["cuda_rng_exact"])
        self.assertEqual(record["plain_forward_calls"], 4)
        self.assertEqual(record["observed_forward_calls"], 4)
        self.assertEqual(record["source_sha256"], SOURCE_SHA256)
        json.dumps(record, allow_nan=False)

    def test_trace_is_raw_pre_update_and_rng_is_restored(self):
        runner = self.make_runner()
        torch.manual_seed(300)
        before = torch.get_rng_state().clone()
        final_ids, trace = runner.run([1, 2, 3], seed=42, config=SMALL_CONFIG)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(len(final_ids), 8)
        self.assertTrue(np.all(trace["masks"][0]))
        self.assertTrue(np.all(trace["canvas"][0] == 11))
        self.assertEqual(trace["entropy"].shape, (4, 8))
        self.assertEqual(trace["entropy"].dtype, np.float32)
        self.assertEqual(trace["proposals"].dtype, np.int32)
        self.assertEqual(trace["masks"].dtype, np.bool_)
        for i, (tokens, logits) in enumerate(zip(runner.model.inputs, runner.model.outputs)):
            np.testing.assert_array_equal(trace["canvas"][i], tokens[0, 3:].numpy())
            np.testing.assert_array_equal(trace["masks"][i], tokens[0, 3:].eq(11).numpy())
            raw = logits[0, 3:].float()
            probabilities = torch.softmax(raw, dim=-1)
            expected_entropy = -torch.special.xlogy(probabilities, probabilities).sum(-1) / math.log(12)
            np.testing.assert_array_equal(trace["entropy"][i], expected_entropy.numpy())
            np.testing.assert_array_equal(trace["proposals"][i], raw.argmax(-1).numpy())
        self.assertEqual(runner.last_run_metadata["forward_calls"], 4)
        self.assertEqual(runner.last_run_metadata["final_mask_count"], 0)
        summarized = summarize_trace(trace["entropy"], trace["proposals"], trace["masks"])
        self.assertEqual(summarized["diagnostics"]["n_eligible_intervals"], 3)

    def test_same_seed_independent_of_execution_order(self):
        runner = self.make_runner()
        ids_a, trace_a = runner.run([1, 2], 99, SMALL_CONFIG)
        runner.run([3, 4], 10001, SMALL_CONFIG)
        ids_b, trace_b = runner.run([1, 2], 99, SMALL_CONFIG)
        self.assertEqual(ids_a, ids_b)
        for name in trace_a:
            np.testing.assert_array_equal(trace_a[name], trace_b[name])

    def test_source_hash_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generate.py"
            path.write_bytes(SOURCE.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_pinned_generate(path)

    def test_rejects_batching_and_interventions(self):
        runner = self.make_runner()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            runner.run([[1, 2], [3, 4]], 1, SMALL_CONFIG)
        for change in ({"logits_eos_inf": True}, {"cfg_scale": 1.},
                       {"end_think_logit_boost": 1.}, {"remasking": "random"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.run([1], 1, {**SMALL_CONFIG, **change})

    def test_trace_npz_round_trip_and_standard_size_budget(self):
        runner = self.make_runner()
        _, trace = runner.run([1, 2], 16, SMALL_CONFIG)
        output = io.BytesIO()
        np.savez_compressed(output, **trace)
        output.seek(0)
        with np.load(output, allow_pickle=False) as restored:
            for name in trace:
                np.testing.assert_array_equal(restored[name], trace[name])
        # Full preregistered dimensions at these dtypes are 426 KB uncompressed.
        self.assertLess(128 * 256 * sum(value.dtype.itemsize for value in trace.values()), 9_000_000)

    def test_fp32_entropy_chunks_and_nonfinite_failure(self):
        observer = _TraceObserver(prompt_length=1, gen_length=33, mask_id=11)
        tokens = torch.full((1, 34), 11, dtype=torch.long)
        tokens[0, 0] = 1
        logits = torch.zeros((1, 34, 12), dtype=torch.bfloat16)
        before = torch.get_rng_state().clone()
        observer.observe(tokens, logits)
        trace = observer.finish()
        np.testing.assert_allclose(trace["entropy"], 1., atol=2e-6)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.all(logits == 0))
        logits[0, 17, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            observer.observe(tokens, logits)

    def test_forward_hook_removed_after_failure(self):
        runner = self.make_runner()
        original = runner.generate
        def broken(*args, **kwargs):
            raise RuntimeError("synthetic failure")
        runner.generate = broken
        before = len(runner.model._forward_hooks)
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            runner.run([1, 2], 17, SMALL_CONFIG)
        self.assertEqual(len(runner.model._forward_hooks), before)
        runner.generate = original


if __name__ == "__main__":
    unittest.main()
