# LLaDA masked entropy / argmax — 4 H100 job

## Project
Repository: the Git repository containing this folder (user supplies URL).
Checkout: a persistent server directory. Execute this package's run_all.sh.
Scientific contract: PROTOCOL.md, config.json, model_pin.json.
Population: explicitly choose existing_discovery (2644 distinct questions) OR gsm8k (1319 questions) before launch.

## Schedule
One durable job on one node with 4 H100 GPUs; four independent batch-one workers.
No training, DDP, model-parallel GPU groups, adaptive pilot or wall-time cutoff in the code.
CPU RAM128GB recommended; storage approximately30GB for environment/checkpoint plus up to1MB/completion and5GB headroom.
Honor scheduler CUDA_VISIBLE_DEVICES. Never stop other users' processes.
Run CPU finalization after all16 completions of every question are sealed.

## Model
Pinned GSAI-ML/LLaDA-8B-Instruct checkpoint and native source exactly as model_pin.json.
BF16 model, FP32 raw entropy diagnostic, native stochastic temperature1,128steps,256tokens.
One shared prefetch/byte verification, then offline workers. No HF token in CLI arguments or logs.

## Data
gsm8k downloads only pinned main/test parquet; gold is separated from public prompts.
existing_discovery uses privately supplied old S1 Discovery prompts and independently recomputed deterministic answers; no old generated answers are read.
Neither option opens historical Validation/Confirmation traces.
16 deterministic UID-derived distinct seeds/question.

## Run
Each worker first checks exact passive-observer/native equivalence on a synthetic prompt.
Then save each completion's trace NPZ, process JSON, response JSON and SHA256 seal.
On interruption rerun same command/run directory. Preserve incomplete attempts; retry their original seed.
Process features are sealed before correctness is joined.
A changed code/config/dataset/environment cannot silently resume the same run.

## Commands
Create a private HF_TOKEN_FILE as described in README.md. Pull the repository, then choose ONE:
```bash
bash run_all.sh --dataset gsm8k --run-dir /persistent/llada_entropy_gsm8k
# OR
bash run_all.sh --dataset existing_discovery --run-dir /persistent/llada_entropy_existing
```

CPU-only recovery of final statistics:
```bash
bash run_all.sh --analyze-only --run-dir /persistent/llada_entropy_gsm8k
```

No GPU runtime estimate is asserted before measurement. PROGRESS.json and each STATUS_WORKER file give current completion counts and measured ETA. This is a full run, not an engineering sample.

## Outputs
Each result file is<10,000,000 bytes; normal writer bound9,000,000.
Raw arrays remain on persistent server storage.
SEND_TO_CODEX contains all compact result ZIPs plus INDEX.json.
All ZIPs are independently readable, not split archive fragments.
Full receipts preserve pins, source/config hashes, exact environment, hardware and seeds.

## Upload
No external upload configured or authorized by this package.
User sends every file in SEND_TO_CODEX directly after completion.
Keep raw realizations for later audit; do not delete after export.

## Acceptance
COMPLETE.json exists and verifies; correct UID population; exactly16 completions each; no missing/conflicting checkpoints;
native-equivalence GPU gates pass; process seal precedes gold join; every final result hash verifies; every return file<10MB.
Negative or nonidentifiable scientific results still count as successful execution.
