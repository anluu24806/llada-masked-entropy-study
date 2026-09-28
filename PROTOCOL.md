# Pre-run measurement and analysis specification

## Population and native generation

One population is selected explicitly before launching: the existing S1 Discovery task set or pinned GSM8K main/test. Each question q receives R=16 realizations r=0,...,15. Seeds are SHA256-derived from study identifier, master seed, UID and r; GPU rank and resume order never enter the seed. The final RUN_BINDING records population and all code/config hashes before inference.

Model: GSAI-ML/LLaDA-8B-Instruct at revision 08b83a6feb34df1a6011b80c3c00c7563e963b07.
Native source: ML-GSAI/LLaDA commit 9182493720ed723ef8031210d85959364e51cbe0, generate.py SHA256 d4d2d3c015511e63fe9237cd3197346148fb0c795b6c89e0b5ef4846be25252e.
128 denoising steps, 256 generated positions, one block of 256, native temperature=1, low_confidence unmasking, cfg_scale=0. Batch=1. No EOS suppression, boost or new controller. Stochastic native proposal sampling allows distinct trajectories; 16 reruns with deterministic decoding would not serve this design.

No new model training. The pinned native function is executed with a passive forward hook; its selection/remasking/RNG code is not reimplemented. The hook returns no replacement output. A native-without-hook vs native-with-hook synthetic GPU test must match full output IDs, post-run RNG states and forward count exactly.

## Exact duplicate questions

The original optional S1 Discovery input has 2,700 UID rows but 2,644 distinct prompt texts (53 duplicate groups; 56 excess rows; maximum multiplicity 4).
For this NEW repeated-generation study only, before inference, group exact (family,prompt), retain the lexicographically smallest UID and record every alias.
Selection uses public fields only. Inconsistent gold within an identical-question group is a fatal data-integrity error.
The source JSONL is preserved unchanged. Thus existing_discovery runs 2,644 x 16 = 42,304 realizations.
GSM8K/custom duplicate exact prompts fail preflight for explicit correction; no semantic deduplication or outcome-dependent exclusion is used.

## Per-position definition

Let x_(q,r,t) be the token sequence supplied to native forward call t, before that call's unmask operation, t=0,...,127. Generation positions have fixed absolute offsets after the prompt. Let M_(q,r,t) be positions holding the native MASK token in x.

For each generated position j, the raw model logits z_(t,j,v) define p_(t,j,v)=softmax_v(z_(t,j,v)) in FP32, before sampling noise or temperature manipulation. Define

h_(t,j) = -sum_v p_(t,j,v) log p_(t,j,v) / log |V|,
a_(t,j) = argmax_v z_(t,j,v).

PyTorch's first-maximum tie rule defines argmax. Entropy is saved as float32. Zero probabilities contribute zero to entropy. No gold-conditioned span selection, final-answer span filtering, or top-k masked-position ranking is used.

For t>=1 compare only S_t=M_(t-1) intersect M_t at the SAME token positions. In ordinary monotone unmasking S_t=M_t. Define delta_(t,j)=h_(t,j)-h_(t-1,j) and b_(t,j)=1[a_(t,j)!=a_(t-1,j)]. The four requested event counts divide by |S_t|:

| Field | Entropy comparison | Argmax |
|---|---|---|
| t1 | delta>0 | changes |
| t2 | delta<0 | changes |
| t3 | delta>0 | unchanged |
| t4 | delta<0 | unchanged |
| tie_flip | delta=0 | changes |
| tie_same | delta=0 | unchanged |

Primary comparison tolerance is exactly zero on the saved FP32 entropy. Ties remain separate; t1+t2+t3+t4 can be below one, while all six categories sum to one. Missing/nonfinite entropy at any compared position is an implementation failure.

t=0 has no previous prediction and is excluded explicitly. An interval with |S_t|=0 is unavailable, not zero. For each eligible interval E, primary feature X_(q,r,k)=mean_(t in E) t_k(t), giving each step equal weight. Sensitivity feature =sum_t count_k(t)/sum_t |S_t|. Both are stored.

The fixed full generated canvas is the primary scope, even if a later completion contains EOS. Final scoring separately uses the generated prefix before the first tokenizer EOS. This is a canvas-level association study, not an answer-span mechanism experiment.

## Mask count

Predeclared current-mask-fraction regions |M_t|/256:
- many: >2/3;
- middle: >1/3 and <=2/3;
- few: <=1/3.

Store the actual count at every step. Report the six category fractions in each region, correctness-stratified summaries and secondary within-question region effects. Empty regions are unavailable and show support explicitly. The schedule largely fixes mask count as a function of depth; this design cannot identify its causal effect separately from depth.

## Correctness and parsing

The worker can read public prompts but imports no gold loader/scorer/analysis code. Dataset gold is held in separate protected files. This is programmatic pipeline separation, not an adversarial OS sandbox.

Before joining outcomes: verify exact 16R identities, payload hashes, seeds, source and completion hashes; materialize process-only rows and availability metadata; write PROCESS_FEATURES_SEAL.json. Only then load gold and parse final responses.

A single final nonempty line of the form Answer: <payload> is required; multiple answer lines or trailing prose are unresolved. Family payloads: exact signed integer; one alphabetic room name normalized lowercase; comma-separated signed integer list; GSM8K exact signed decimal with properly grouped commas allowed. No semantic interpretation, gold-conditioned matching, or LLM judge.

Y_(q,r)=1 for a parsed exact match; otherwise Y=0. Parser status stays separate and unresolved fractions are reported. A native residual MASK is preserved, flagged and processed by this same scorer, not used to reject or replace a realization.

## Statistical plan

Primary: univariate question-fixed-effects linear probability model
Y_(q,r)=alpha_q+beta_2 X_(q,r,2)+epsilon_(q,r).
Two-sided beta_2 test with UID-clustered CR1 standard error and G-1 t degrees of freedom. Report effect in percentage points per +0.10 feature proportion: 10 beta_2. Constant/no-within-question support is explicitly NONIDENTIFIABLE.

Primary interval: 10,000 whole-question bootstrap draws, stratified by task family, seed20260929. Keep all16 realizations for each selected question, including multiplicities. UID is the independent uncertainty unit; denoising steps, token positions and repeated answers are not additional independent questions.

Secondary:
1. Four separate within-question t1–t4 effects with Holm-adjusted p-values. Do not fit all four compositional fractions plus an intercept as independent free predictors.
2. Token-step weighting sensitivity.
3. T2 conditional on total_flip=t1+t2+tie_flip and entropy_down=t2+t4. Rank deficiency is reported, never repaired by feature selection.
4. Between-question Spearman corr(mean_r X_(q,r,k), mean_r Y_(q,r)), with tied ranks. Holm across the four categories per weighting. Bootstrap uses the first2,000 of the same UID draws for bounded CPU work; this fixed secondary cap is disclosed.
5. The three mask regions, descriptive/secondary, without a causal interpretation or outcome-selected region.

At least90% of bootstrap draws must identify a parameter to issue its interval; otherwise retain point estimate and flag interval unavailable. Singleton-only family strata cannot provide bootstrap uncertainty. Do not drop constant/awkward UIDs to manufacture support.

The same16 runs contribute to question-mean feature and accuracy, so between-question association mixes question effects and finite-run within-question covariance. It is not an isolated estimate of intrinsic question difficulty.

No threshold search, alternate parser selection, adaptive sample stopping, outcome-based exclusions, LLM judge, causal claim or new model. All16 runs are required regardless of accuracy.
