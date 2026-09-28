# LLaDA masked-entropy study â€” 4 H100

An independent repeated-generation experiment. This is inference and offline statistics, not model training and not a rerun/redefinition of historical T17. Exactly 16 stochastic completions per distinct question, with the same pinned LLaDA checkpoint and native operating point. No SDAR, Gemma, TESS2, Validation or Confirmation outputs are used.

## Public repository scope

This repository publishes implementation, tests and configuration only. It contains no original prompts, gold labels, generated outputs, model weights or credentials.
GSM8K is downloaded from its pinned public source at runtime. The optional existing_discovery mode requires separately copying the private public.jsonl, gold.jsonl and PROVENANCE.json into data/existing_discovery/ on the server; this directory is ignored by Git.

## One command on the server

Clone/pull this entire folder. Linux x86_64, Bash, curl, sha256sum, flock, an NVIDIA driver compatible with CUDA 12.4, and four allocated H100 GPUs are required. No pre-existing Conda or Python environment is required. Allow approximately 30 GB for the environment/model cache plus about 1 MB per completion and 5 GB headroom for outputs. All paths should be on persistent storage.

Set a read-only Hugging Face token securely, once:

```bash
mkdir -p "$HOME/.config/huggingface"
chmod 700 "$HOME/.config/huggingface"
read -rsp "HF read token: " TOKEN_INPUT; echo
(umask 077; printf '%s' "$TOKEN_INPUT" > "$HOME/.config/huggingface/token")
unset TOKEN_INPUT
```

Choose the question population explicitly. The user has not yet selected between these two options; the launcher does not silently choose a benchmark.

**Existing questions:** 2,644 distinct prior LLaDA Discovery questions (from 2,700 UIDs), 42,304 new completions:

```bash
bash run_all.sh --dataset existing_discovery --run-dir /persistent/llada_entropy_existing
```

**GSM8K test:** pinned 1,319 questions, 21,104 new completions:

```bash
bash run_all.sh --dataset gsm8k --run-dir /persistent/llada_entropy_gsm8k
```

Run only one selected population for this experiment. The optional second adapter is not a recommendation to add a benchmark after observing results. For a custom population, provide public JSONL with `uid` (64 lowercase hex), `family`, `prompt`, and a separate gold JSONL with `uid`, `gold`; use `--dataset /path/public.jsonl --gold /path/gold.jsonl`.

The script creates an isolated Conda environment, verifies exact dependency versions, runs all CPU tests, fetches and byte-verifies the model once, prepares the selected dataset, launches four independent workers, performs a real-model passive-observer equivalence check on a synthetic non-task prompt on each GPU, generates all completions, verifies/seals process features, joins correctness, computes statistics and exports the return files. It does not need a notebook or a manual mid-run confirmation.

If your scheduler exposes more than four GPUs, set `CUDA_VISIBLE_DEVICES` to exactly the four allocated GPUs. Do not override a scheduler-provided four-GPU assignment. Batch size is one per GPU; questions are sharded by a deterministic UID hash. Shard assignment is engineering-only and is not a statistical unit.

## Background / scheduler execution

Use the attached `SERVER_JOB_DRAFT.md` as the job specification. On a shell server, run in tmux:

```bash
tmux new -s llada-entropy
bash run_all.sh --dataset gsm8k --run-dir /persistent/llada_entropy_gsm8k
# Detach: Ctrl-b, then d. Reattach: tmux attach -t llada-entropy
```

On Slurm, allocate one node, four H100 GPUs and sufficient CPU RAM (128 GB recommended), then use that same command as the job payload. GPU model setup can momentarily use several model-sized copies in CPU memory across workers.

Set `HF_HOME` to share/reuse a persistent model cache and `LLADA_RUNTIME_DIR` to choose the Conda installation location. These are caches/input weights; model weight files necessarily exceed 10 MB and are not result files to send back.

## Resume and completion

Run the exact same command, with the same code, config, dataset and `--run-dir`. Every completed realization is validated by identity, seed, byte count and SHA256 and skipped. An interrupted realization is archived and repeated using its original seed. No substitute questions, extra seeds or favorable-output retry. A residual MASK produced by native sampling is preserved and scored by the same fixed parser.

An active run directory is protected against a second orchestrator and duplicate worker writers. On one worker failure the orchestrator stops only its own remaining workers; completed artifacts remain resumable. No time limit or reduced population is applied by this code.

After all inference finishes, statistics can resume without GPUs or HF authentication:

```bash
bash run_all.sh --analyze-only --run-dir /persistent/llada_entropy_gsm8k
```

Successful completion requires `COMPLETE.json`, all 16 realizations per UID, a verified process seal, complete final tables, and size-checked exports. A partial population is never analyzed as complete.

The latest progress and empirical ETA are in `PROGRESS.json` and `STATUS_WORKER_0.json` through `STATUS_WORKER_3.json`. ETA uses recent measured seconds/completion. A rough scaling calculation after startup is
`remaining wall seconds â‰ˆ max(worker_remaining Ã— worker_recent_seconds_per_run)`.
No H100 timing has been measured during the local CPU build.

## Files to return

**Send every file in `<run-dir>/SEND_TO_CODEX/`.** It contains `INDEX.json` and standalone `RESULTS_001.zip`, `RESULTS_002.zip`, etc. Every file is strictly below **9,000,000 bytes**, hence below the requested decimal 10 MB. No multipart ZIP utility is needed; each ZIP opens independently.

The package includes:
- `REPORT.md`, `SUMMARY.json`, exact model/config/code/environment receipts and process seal;
- question-level accuracy/features, all within-question effects/intervals, mask-region summaries and Spearman results;
- every realization's features, score, parsed answer and response, split into bounded JSONL files;
- category-rate and within-question T2 plots;
- raw trace examples chosen deterministically: the first2 UID hashes per family, realizations r0/r1, irrespective of outcomes.

Keep `realizations/` on the server. It contains the complete per-position entropy, argmax, mask and canvas arrays, native completion IDs and completion seals. No raw realization file exceeds 9 MB. Full-vocabulary logits are deliberately reduced online to the exact requested statistics, not stored as enormous files.

All output file sizes are audited before COMPLETE. Raw data are not deleted during finalization. If a later raw reanalysis is needed, individual requested raw files can be sent separately under the same size bound.

## Scientific definition

See `PROTOCOL.md`. Primary association is within-question T2 versus final correctness; secondary four-category tests have Holm adjustment, and question-level Spearman is separate. All uncertainty resamples questions, retaining the 16 runs. There is no causal interpretation of entropy or mask count.

## Local verification

```bash
python -m pytest -q tests
bash -n run_all.sh
```

The CPU tests use synthetic tensors/dummy models, never scientific model inference. Actual H100 compatibility and native equivalence are checked by the server launch before scientific generation. Windows build verification is documented in `BUILD_VERIFICATION.json`.
