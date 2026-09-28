"""Offline, question-clustered analysis of the masked-entropy study.

The six cells are t1, t2, t3, t4, tie_flip, tie_same, in that order.
All coefficients use within-question variation; no model inference occurs here.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy import stats

CELLS = ("t1", "t2", "t3", "t4", "tie_flip", "tie_same")
WEIGHTINGS = ("step", "token_step")
REGIONS = ("many", "middle", "few")
RUNS_PER_UID = 16
SECONDARY_BOOTSTRAP_CAP = 2000
_VAR_TOL = 1e-24


def _native_float(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _validate_feature(value: Any, label: str, *, nullable: bool = False):
    if value is None and nullable:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain six finite proportions") from exc
    if array.shape != (6,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain six finite proportions")
    if np.any(array < -1e-12) or np.any(array > 1 + 1e-12):
        raise ValueError(f"{label} must contain proportions in [0, 1]")
    if not np.isclose(array.sum(), 1.0, rtol=0, atol=1e-8):
        raise ValueError(f"{label} must sum to one, including both tie cells")
    return array


def _prepare(rows: Iterable[dict]):
    rows = list(rows)
    if not rows:
        raise ValueError("Analysis requires at least one complete 16-run UID")
    by_uid: dict[str, dict[int, dict]] = {}
    for row in rows:
        uid = row.get("uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError("Each row needs a nonempty string uid")
        replicate = row.get("replicate")
        if isinstance(replicate, (bool, np.bool_)) or not isinstance(replicate, (int, np.integer)):
            raise ValueError(f"{uid}: replicate must be an integer from 0 to 15")
        if replicate not in range(RUNS_PER_UID):
            raise ValueError(f"{uid}: replicate must be an integer from 0 to 15")
        if row.get("correct") not in (0, 1) or not isinstance(row.get("correct"), (bool, int, np.integer)):
            raise ValueError(f"{uid}: correct must be 0 or 1")
        if not isinstance(row.get("parseable"), (bool, np.bool_)):
            raise ValueError(f"{uid}: parseable must be boolean")
        if not row["parseable"] and row["correct"]:
            raise ValueError(f"{uid}: an unparseable answer cannot be correct")
        if not isinstance(row.get("family"), str) or not row["family"]:
            raise ValueError(f"{uid}: family must be a nonempty string")
        if replicate in by_uid.setdefault(uid, {}):
            raise ValueError(f"{uid}: duplicate replicate {replicate}")
        by_uid[uid][int(replicate)] = row
    uids = sorted(by_uid)
    families, ordered = [], []
    for uid in uids:
        block = by_uid[uid]
        if set(block) != set(range(RUNS_PER_UID)):
            raise ValueError(f"{uid}: all 16 unique replicates 0..15 are required")
        family_set = {r["family"] for r in block.values()}
        if len(family_set) != 1:
            raise ValueError(f"{uid}: family must be constant across replicates")
        families.append(next(iter(family_set)))
        ordered.extend(block[r] for r in range(RUNS_PER_UID))
    shape = (len(uids), RUNS_PER_UID)
    y = np.array([r["correct"] for r in ordered], dtype=float).reshape(shape)
    parseable = np.array([r["parseable"] for r in ordered], dtype=bool).reshape(shape)
    arrays = {w: np.empty(shape + (6,), dtype=float) for w in WEIGHTINGS}
    regions = {region: {w: np.full(shape + (6,), np.nan) for w in WEIGHTINGS} for region in REGIONS}
    for idx, row in enumerate(ordered):
        i, j = divmod(idx, RUNS_PER_UID)
        feature = row.get("features", {})
        for weighting in WEIGHTINGS:
            arrays[weighting][i, j] = _validate_feature(feature.get(weighting), f"{row['uid']}/{j}/{weighting}")
        regional = feature.get("regions", {})
        for region in REGIONS:
            values = regional.get(region, {}) or {}
            for weighting in WEIGHTINGS:
                value = _validate_feature(values.get(weighting), f"{row['uid']}/{j}/{region}/{weighting}", nullable=True)
                if value is not None:
                    regions[region][weighting][i, j] = value
            available = [np.isfinite(regions[region][w][i, j]).all() for w in WEIGHTINGS]
            if available[0] != available[1]:
                raise ValueError(f"{row['uid']}/{j}/{region}: both weightings must have the same observation support")
    return uids, families, y, parseable, arrays, regions


@dataclass
class _Fit:
    row: dict
    xx: np.ndarray
    xy: np.ndarray
    yy: np.ndarray
    target: int
    boot: list[np.ndarray]


def _fit_within(y: np.ndarray, x: np.ndarray, metadata: dict) -> _Fit:
    """Fit after UID demeaning, keeping O(G p^2) sufficient statistics."""
    if x.ndim == 2:
        x = x[..., None]
    valid = np.isfinite(x).all(axis=2)
    counts = valid.sum(axis=1)
    safe_counts = np.maximum(counts, 1)
    x0 = np.where(valid[..., None], x, 0)
    y0 = np.where(valid, y, 0)
    xd = np.where(valid[..., None], x0 - (x0.sum(axis=1) / safe_counts[:, None])[:, None, :], 0)
    yd = np.where(valid, y0 - (y0.sum(axis=1) / safe_counts)[:, None], 0)
    xx = np.einsum("gri,grj->gij", xd, xd)
    xy = np.einsum("gri,gr->gi", xd, yd)
    yy = np.einsum("gr,gr->g", yd, yd)
    p, target = x.shape[2], x.shape[2] - 1
    a, b = xx.sum(axis=0), xy.sum(axis=0)
    n, g = int(counts.sum()), int(np.count_nonzero(counts))
    row = {
        **metadata,
        "status": "NONIDENTIFIABLE", "reason": None,
        "coefficient": None, "effect_pp_per_0_10": None, "sign": None,
        "cluster_se": None, "cluster_se_pp_per_0_10": None,
        "p_value": None, "p_holm": None, "ci95": None, "ci95_pp_per_0_10": None,
        "bootstrap_valid_replicates": 0, "bootstrap_requested_replicates": 0,
        "bootstrap_status": "UNAVAILABLE", "inference_status": "UNAVAILABLE",
        "n_runs": n,
        "n_missing_region_runs": int(y.size - n) if metadata.get("region") else 0,
        "n_uids": g,
        "n_uids_with_repeated_observations": int(np.count_nonzero(counts >= 2)),
        "mixed_outcome_uid_count": int(np.count_nonzero(yy > _VAR_TOL)),
        "n_uids_with_within_feature_variation": int(np.count_nonzero(xx[:, target, target] > _VAR_TOL)),
        "cluster_df": g - 1 if g > 1 else None, "n_predictors": p,
        "target_residual_sum_squares": None, "residual_degrees_of_freedom": n - g - p,
        "cluster_correction": "CR1: G/(G-1) * (N-1)/(N-G-p); UID fixed effects included in parameter count",
    }
    fit = _Fit(row=row, xx=xx, xy=xy, yy=yy, target=target, boot=[])
    if n == 0:
        row["reason"] = "region_unobserved"
        return fit
    if a[target, target] <= n * _VAR_TOL:
        row["reason"] = "no_within_question_feature_variation"
        return fit
    eigenvalues = np.linalg.eigvalsh(a)
    if eigenvalues[0] <= eigenvalues[-1] * 1e-12:
        row["reason"] = "rank_deficient_within_question_design"
        return fit
    inv = np.linalg.inv(a)
    row["target_residual_sum_squares"] = float(1 / inv[target, target])
    if yy.sum() <= n * _VAR_TOL:
        row["reason"] = "no_within_question_outcome_variation"
        return fit
    beta = np.linalg.solve(a, b)
    estimate = float(beta[target])
    row.update(status="IDENTIFIABLE", coefficient=estimate, effect_pp_per_0_10=10 * estimate,
               sign="positive" if estimate > 1e-12 else "negative" if estimate < -1e-12 else "zero")
    residual_ss = max(0.0, float(yy.sum() - 2 * np.dot(beta, b) + beta @ a @ beta))
    row["residual_sum_squares"] = residual_ss
    if g < 2 or n - g - p <= 0:
        row["inference_status"] = "UNAVAILABLE_INSUFFICIENT_CLUSTERS_OR_RESIDUAL_DF"
        return fit
    if residual_ss <= 1e-12 * max(float(yy.sum()), _VAR_TOL):
        row["inference_status"] = "UNAVAILABLE_ZERO_RESIDUAL_VARIANCE"
        return fit
    scores = xy - np.einsum("gij,j->gi", xx, beta)
    correction = (g / (g - 1)) * ((n - 1) / (n - g - p))
    covariance = correction * inv @ (scores.T @ scores) @ inv
    variance = float(covariance[target, target])
    if variance <= _VAR_TOL:
        row["inference_status"] = "UNAVAILABLE_DEGENERATE_CLUSTER_SCORE"
        return fit
    se = float(np.sqrt(variance))
    row.update(cluster_se=se, cluster_se_pp_per_0_10=10 * se,
               p_value=float(2 * stats.t.sf(abs(estimate / se), df=g - 1)),
               inference_status="AVAILABLE_CLUSTER_T")
    return fit


def _holm(rows: list[dict], family: str):
    """Holm with the declared family size, including unavailable tests."""
    m = len(rows)
    for row in rows:
        row["holm_family"] = family
        row["holm_family_size"] = m
    ordered = sorted((r for r in rows if r["p_value"] is not None), key=lambda r: r["p_value"])
    previous = 0.0
    for i, row in enumerate(ordered):
        previous = min(1.0, max(previous, (m - i) * row["p_value"]))
        row["p_holm"] = previous


def _rank_layout(values: np.ndarray):
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.r_[0, 1 + np.flatnonzero(sorted_values[1:] != sorted_values[:-1])]
    inverse = np.empty(len(values), dtype=int)
    inverse[order] = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(values)]))
    return order, starts, inverse


def _weighted_ranks(weights: np.ndarray, layout):
    order, starts, inverse = layout
    tied_counts = np.add.reduceat(weights[:, order], starts, axis=1)
    tied_ranks = np.cumsum(tied_counts, axis=1) - (tied_counts - 1) / 2
    return tied_ranks[:, inverse]


def _bootstrap(fits: list[_Fit], between: list[dict], problem_features: dict,
               accuracy: np.ndarray, families: list[str], replicates: int, seed: int):
    """Use one family-stratified UID resampling plan for every estimand.

    A repeated sampled UID contributes its full sufficient-statistic block each
    time. Weighted tie ranks exactly match expanding those repeated UIDs.
    """
    secondary_replicates = min(replicates, SECONDARY_BOOTSTRAP_CAP)
    rng = np.random.default_rng(seed)
    strata = [np.array([i for i, f in enumerate(families) if f == name]) for name in sorted(set(families))]
    g = len(families)
    for fit in fits:
        fit.row["bootstrap_requested_replicates"] = replicates
    for row in between:
        row["bootstrap_requested_replicates"] = secondary_replicates
        row["bootstrap_valid_replicates"] = 0
    if replicates == 0:
        for row in [f.row for f in fits] + between:
            row["bootstrap_status"] = "UNAVAILABLE_NOT_REQUESTED"
        return
    if not any(len(indices) > 1 for indices in strata):
        for row in [f.row for f in fits] + between:
            row["bootstrap_status"] = "UNAVAILABLE_NO_WITHIN_FAMILY_RESAMPLING_VARIATION"
        return
    for fit in fits:
        if fit.row["n_uids"] < 2 and fit.row["status"] == "IDENTIFIABLE":
            fit.row["bootstrap_status"] = "UNAVAILABLE_INSUFFICIENT_OBSERVED_UIDS"
    active = [f for f in fits if f.row["status"] == "IDENTIFIABLE" and f.row["n_uids"] >= 2]
    simple = [f for f in active if f.xx.shape[1] == 1]
    adjusted = [f for f in active if f.xx.shape[1] != 1]
    if not active and not any(r["status"] == "AVAILABLE" for r in between):
        return
    if simple:
        simple_xx = np.column_stack([f.xx[:, 0, 0] for f in simple])
        simple_xy = np.column_stack([f.xy[:, 0] for f in simple])
        simple_yy = np.column_stack([f.yy for f in simple])
    layouts = {key: _rank_layout(value) for key, value in problem_features.items()}
    y_layout = _rank_layout(accuracy)
    between_boot = {r["id"]: [] for r in between}
    batch_size = min(128, max(1, 1_000_000 // max(g, 1)))
    for start in range(0, replicates, batch_size):
        batch = min(batch_size, replicates - start)
        weights = np.zeros((batch, g), dtype=float)
        for indices in strata:
            weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=batch)
        if simple:
            a, b, c = weights @ simple_xx, weights @ simple_xy, weights @ simple_yy
            with np.errstate(divide="ignore", invalid="ignore"):
                betas = np.where((a > _VAR_TOL) & (c > _VAR_TOL), b / a, np.nan)
            for j, fit in enumerate(simple):
                fit.boot.append(betas[:, j])
        for fit in adjusted:
            p = fit.xx.shape[1]
            a = (weights @ fit.xx.reshape(g, -1)).reshape(batch, p, p)
            b, c = weights @ fit.xy, weights @ fit.yy
            eigenvalues = np.linalg.eigvalsh(a)
            valid = (eigenvalues[:, 0] > eigenvalues[:, -1] * 1e-12) & (c > _VAR_TOL)
            result = np.full(batch, np.nan)
            if np.any(valid):
                result[valid] = np.linalg.solve(a[valid], b[valid, :, None])[:, fit.target, 0]
            fit.boot.append(result)
        if start >= secondary_replicates:
            continue
        w = weights[:min(batch, secondary_replicates - start)]
        center = (g + 1) / 2
        yrank = _weighted_ranks(w, y_layout) - center
        yss = np.sum(w * yrank ** 2, axis=1)
        for row in between:
            if row["status"] != "AVAILABLE":
                continue
            key = (row["weighting"], row["feature"])
            xrank = _weighted_ranks(w, layouts[key]) - center
            numerator = np.sum(w * xrank * yrank, axis=1)
            denominator = np.sqrt(np.sum(w * xrank ** 2, axis=1) * yss)
            with np.errstate(divide="ignore", invalid="ignore"):
                correlation = np.where(denominator > 0, numerator / denominator, np.nan)
            between_boot[row["id"]].append(correlation)
    for fit in fits:
        row = fit.row
        row["bootstrap_requested_replicates"] = replicates
        if not fit.boot:
            continue
        samples = np.concatenate(fit.boot)
        samples = samples[np.isfinite(samples)]
        fit.boot.clear()
        row["bootstrap_valid_replicates"] = int(len(samples))
        _set_ci(row, samples, replicates, coefficient=True)
    for row in between:
        row["bootstrap_requested_replicates"] = secondary_replicates
        chunks = between_boot[row["id"]]
        samples = np.concatenate(chunks) if chunks else np.array([])
        samples = samples[np.isfinite(samples)]
        row["bootstrap_valid_replicates"] = int(len(samples))
        _set_ci(row, samples, secondary_replicates, coefficient=False)


def _set_ci(row: dict, samples: np.ndarray, requested: int, *, coefficient: bool):
    if len(samples) < 2:
        row["bootstrap_status"] = "UNAVAILABLE_TOO_FEW_VALID_RESAMPLES"
        return
    if len(samples) < 0.9 * requested:
        row["bootstrap_status"] = "UNAVAILABLE_OVER_10_PERCENT_DEGENERATE_RESAMPLES"
        return
    bounds = [float(v) for v in np.quantile(samples, [0.025, 0.975])]
    row["ci95"] = bounds
    if coefficient:
        row["ci95_pp_per_0_10"] = [v * 10 for v in bounds]
    row["bootstrap_status"] = "AVAILABLE" if len(samples) == requested else "AVAILABLE_CONDITIONAL_ON_IDENTIFIABLE_RESAMPLES"


def _between_rows(problem_features: dict, accuracy: np.ndarray) -> list[dict]:
    result = []
    for (weighting, feature), values in problem_features.items():
        row = {"id": f"between_{weighting}_{feature}", "analysis": "between_question_spearman",
               "weighting": weighting, "feature": feature, "status": "UNAVAILABLE",
               "reason": None, "spearman_rho": None, "p_value": None, "p_holm": None,
               "ci95": None, "bootstrap_status": "UNAVAILABLE", "n_uids": int(len(values)),
               "interpretation": "Secondary observational between-question association; not a within-question or causal effect"}
        if len(values) < 3:
            row["reason"] = "fewer_than_three_questions"
        elif len(np.unique(values)) < 2:
            row["reason"] = "constant_problem_mean_feature"
        elif len(np.unique(accuracy)) < 2:
            row["reason"] = "constant_problem_accuracy"
        else:
            rho, p = stats.spearmanr(values, accuracy)
            row.update(status="AVAILABLE", spearman_rho=_native_float(rho), p_value=_native_float(p))
        result.append(row)
    for weighting in WEIGHTINGS:
        _holm([r for r in result if r["weighting"] == weighting], f"between_question_{weighting}_four_cells")
    return result


def _depth_rows(y: np.ndarray, regions: dict, uids: list[str], families: list[str]):
    result = []
    for region in REGIONS:
        for weighting in WEIGHTINGS:
            array = regions[region][weighting]
            observed = np.isfinite(array).all(axis=2)
            for correct in (0, 1):
                selected = observed & (y == correct)
                support = selected.sum(axis=1)
                for j, feature in enumerate(CELLS):
                    values = array[..., j][selected]
                    per_question = np.divide(np.where(selected, array[..., j], 0).sum(axis=1),
                                             support, out=np.full(len(uids), np.nan), where=support > 0)
                    per_question = per_question[np.isfinite(per_question)]
                    result.append({"region": region, "weighting": weighting, "feature": feature,
                                   "correct": correct, "scope": "secondary_descriptive_observed_regions_only",
                                   "n_runs": int(len(values)), "n_uids": int(np.count_nonzero(support)),
                                   "total_correctness_group_runs": int(np.count_nonzero(y == correct)),
                                   "missing_region_correctness_group_runs": int(np.count_nonzero((~observed) & (y == correct))),
                                   "family_uid_support": dict(sorted(Counter(f for f, c in zip(families, support) if c).items())),
                                   "mean": float(values.mean()) if len(values) else None,
                                   "median": float(np.median(values)) if len(values) else None,
                                   "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                                   "question_equal_weight_mean": float(per_question.mean()) if len(per_question) else None,
                                   "status": "OBSERVED" if len(values) else "UNOBSERVED"})
    return result


def analyze(rows: Iterable[dict], bootstrap_replicates: int = 10000, seed: int = 20260929) -> dict:
    """Return JSON-native summary and tables for complete 16-run UID blocks.

    Primary: step-weighted t2, univariate within-question LPM. The reported
    effect is percentage points of accuracy per +0.10 feature proportion.
    All fixed-effect bootstraps resample whole UIDs within family. Spearman
    intervals use the first min(B, 2000) draws from the identical sample plan.
    Unparseable outputs remain incorrect runs. Missing region arrays are
    excluded only from that region's descriptive analyses, never zero-filled.
    """
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, (int, np.integer)) or bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be a nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    bootstrap_replicates, seed = int(bootstrap_replicates), int(seed)
    uids, families, y, parseable, arrays, regions = _prepare(rows)
    accuracy = y.mean(axis=1)
    problem_features = {(w, cell): arrays[w].mean(axis=1)[:, j] for w in WEIGHTINGS for j, cell in enumerate(CELLS[:4])}
    problem_rows = []
    for i, uid in enumerate(uids):
        row = {"uid": uid, "family": families[i], "n_runs": RUNS_PER_UID,
               "n_correct": int(y[i].sum()), "accuracy": float(accuracy[i]),
               "n_parseable": int(parseable[i].sum()), "mixed_outcome": bool(0 < y[i].sum() < RUNS_PER_UID)}
        for weighting in WEIGHTINGS:
            row[f"mean_{weighting}"] = {cell: float(arrays[weighting][i, :, j].mean()) for j, cell in enumerate(CELLS)}
        for region in REGIONS:
            row[f"observed_runs_{region}"] = int(np.isfinite(regions[region]["step"][i]).all(axis=1).sum())
        problem_rows.append(row)
    fits = []
    for weighting in WEIGHTINGS:
        these = []
        for j, cell in enumerate(CELLS[:4]):
            fit = _fit_within(y, arrays[weighting][..., j], {
                "id": f"within_{weighting}_{cell}", "analysis": "univariate_within_question_lpm",
                "weighting": weighting, "feature": cell, "region": None,
                "role": "primary" if weighting == "step" and cell == "t2" else "secondary" if weighting == "step" else "sensitivity",
                "estimand": "Accuracy percentage points per +0.10 feature proportion within the same question",
            })
            fits.append(fit)
            these.append(fit.row)
        _holm(these, f"within_{weighting}_four_cells")
        a = arrays[weighting]
        # Ties with a token flip remain in total_flip. No renormalization.
        design = np.stack((a[..., 0] + a[..., 1] + a[..., 4], a[..., 1] + a[..., 3], a[..., 1]), axis=2)
        fits.append(_fit_within(y, design, {
            "id": f"specificity_{weighting}_t2", "analysis": "parent_plus_conjunction_within_question_lpm",
            "weighting": weighting, "feature": "t2", "region": None, "role": "secondary_specificity",
            "predictors": ["total_flip=t1+t2+tie_flip", "entropy_down=t2+t4", "t2"],
            "estimand": "Adjusted t2 coefficient conditional on total_flip and entropy_down; separate from primary",
        }))
    for region in REGIONS:
        for weighting in WEIGHTINGS:
            for j, cell in enumerate(CELLS[:4]):
                fits.append(_fit_within(y, regions[region][weighting][..., j], {
                    "id": f"region_{region}_{weighting}_{cell}", "analysis": "region_within_question_lpm",
                    "weighting": weighting, "feature": cell, "region": region, "role": "secondary_descriptive",
                    "estimand": "Within-question association among observed runs in this mask region; not a primary test",
                }))
    between = _between_rows(problem_features, accuracy)
    _bootstrap(fits, between, problem_features, accuracy, families, bootstrap_replicates, seed)
    effect_rows = [fit.row for fit in fits]
    primary = next(r for r in effect_rows if r["role"] == "primary")
    summary = {
        "schema_version": "masked_entropy_analysis_v1", "n_uids": len(uids), "n_runs": int(y.size),
        "replicates_per_uid": RUNS_PER_UID, "accuracy": float(y.mean()),
        "n_correct": int(y.sum()), "n_parseable": int(parseable.sum()), "n_unparseable": int((~parseable).sum()),
        "mixed_outcome_uid_count": int(np.count_nonzero((accuracy > 0) & (accuracy < 1))),
        "family_uid_counts": dict(sorted(Counter(families).items())),
        "primary": primary,
        "secondary_univariate": [r for r in effect_rows if r["analysis"] == "univariate_within_question_lpm" and r["weighting"] == "step"],
        "token_step_sensitivity": [r for r in effect_rows if r["analysis"] == "univariate_within_question_lpm" and r["weighting"] == "token_step"],
        "specificity": [r for r in effect_rows if r["analysis"] == "parent_plus_conjunction_within_question_lpm"],
        "between_question_spearman": between,
        "bootstrap": {"seed": seed, "replicates": bootstrap_replicates,
                      "spearman_replicates": min(bootstrap_replicates, SECONDARY_BOOTSTRAP_CAP),
                      "unit": "UID", "stratification": "family", "interval": "percentile_95",
                      "same_uid_draws_for_all_analyses": True,
                      "singleton_family_count": int(sum(n == 1 for n in Counter(families).values())),
                      "minimum_valid_fraction": 0.9},
        "notes": [
            "All 16 outcomes are retained; unparseable answers are incorrect. Runs are not independent observations for uncertainty.",
            "Primary p-value is two-sided UID-clustered CR1 t inference with G-1 degrees of freedom. Confidence interval is a whole-UID family-stratified percentile bootstrap.",
            "Four univariate cells are fitted separately. Holm adjustment uses all four declared tests per weighting, including unavailable tests.",
            "Without within-question feature or outcome variation the result is NONIDENTIFIABLE; perfect fits retain an estimate but have unavailable cluster-t inference.",
            "The adjusted conjunction check includes total_flip=t1+t2+tie_flip and entropy_down=t2+t4. Rank-deficient designs are explicitly unavailable.",
            "Between-question Spearman results are secondary observational summaries over UID means and UID accuracy. Tied ranks are retained; constant inputs are unavailable.",
            "Spearman bootstrap uses at most 2000 of the shared UID draws for bounded computation; fixed-effect intervals use all requested draws.",
            "A bootstrap interval is unavailable if every family has only one question, fewer than two questions support an effect, or more than 10 percent of resamples are nonidentifiable.",
            "Region summaries and effects are secondary descriptive results, report observed support, and never fill missing regions with zeros.",
            "Clustered uncertainty accounts for repeated runs within question; these associations do not establish causation.",
            "The masked-entropy measure is distinct from the frozen T17 measure; this analysis neither reads nor redefines T17.",
        ],
    }
    return {"summary": summary, "problem_rows": problem_rows, "effect_rows": effect_rows,
            "depth_rows": _depth_rows(y, regions, uids, families)}

