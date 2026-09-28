# Result file map

All paths below are relative to --run-dir.

- COMPLETE.json: final success marker with final artifact hashes; absence means no completed scientific analysis.
- RUN_BINDING.json: pre-run dataset seal, exact config, code and model-receipt hashes.
- MODEL_RECEIPT.json: all pinned checkpoint/tokenizer/source files verified by SHA256 and upstream blob identity.
- ENVIRONMENT.json: exact Python and pip freeze, immutable across resume.
- hardware/*.json: visible GPUs and CUDA/Torch metadata at launch.
- gpu_preflight/*.json: real model native-vs-observed token/RNG/forward-count checks.
- data/public_*.jsonl: prompts and UID metadata; data/protected_gold_*.jsonl: separate exact answers.
- data/DATASET_SEAL.json: selected population and hashes.
- realizations/<prefix>/<uid>/rXX/trace.npz: entropy float32[T,G], proposals int32[T,G], masks bool[T,G], canvas int32[T,G]; every array is BEFORE that denoising step's token transfer.
- realizations/.../process.json: six-category counts/rates by step, equal-step and token-step means, region support, identity and native metadata.
- realizations/.../response.json: complete generated IDs, first EOS, text before EOS and full decoded text.
- realizations/.../COMPLETE.json: seed, UID, replicate, binding and three file hashes.
- rXX.interrupted.* directories: prior interrupted unsealed attempt, preserved for diagnosis, never counted as another realization.
- process_frozen/process_features_*.jsonl: complete process-only run features.
- process_frozen/raw_index_*.jsonl: complete raw artifact locator/hash index.
- PROCESS_FEATURES_SEAL.json: durable process-first barrier before scoring.
- results/SUMMARY.json: machine-readable estimates, intervals, support, unavailable reasons and parser diagnostics.
- results/REPORT.md: concise scientific report including interpretation limits.
- results/question_results_*.csv: all questions, accuracy, parseability and mean features.
- results/effect_results_*.csv: primary/secondary fixed-effects estimates; p-values; Holm and bootstrap intervals.
- results/mask_region_results_*.csv: descriptive observed support by mask region and correctness.
- results/scored_realizations_*.jsonl: all16 scores, responses and features/question.
- results/per_step_category_means_*.csv: mean process categories across depth.
- results/category_rates_by_depth.png and within_question_t2.png: descriptive plots.
- STATUS_WORKER_*.json / PROGRESS.json: live progress and measured ETA, not scientific results.
- logs/*.log / events/*.json: bounded diagnostics.
- SEND_TO_CODEX/INDEX.json and RESULTS_*.zip: the files to send back; every part independently opens and is<9MB.

All numeric JSON writes disallow NaN/Infinity. Scientific unavailable estimates are null with a reason, never silently zero.

- results/raw_examples/: deterministic first 2 UID hashes per family, r0/r1; includes prompts and raw arrays independent of outcomes.
