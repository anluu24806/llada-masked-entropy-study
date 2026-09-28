"""Numerical and edge-case checks for the offline study analysis."""
from copy import deepcopy
import json
import random

import numpy as np
import pytest
from scipy import stats

from llada_study.analysis import analyze, _rank_layout, _weighted_ranks


def make_rows(n_uids=10, seed=25):
    rng = np.random.default_rng(seed)
    result = []
    for q in range(n_uids):
        features = rng.dirichlet([3, 4, 2, 3, 2, 2], size=16)
        token = 0.75 * features + 0.25 * rng.dirichlet([2] * 6, size=16)
        probability = np.clip(0.15 + 0.045 * q + 0.6 * features[:, 1], 0.05, 0.95)
        outcomes = rng.binomial(1, probability)
        # Every UID has support for within-question outcome variation.
        outcomes[:2] = [0, 1]
        for r in range(16):
            regions = {name: {"step": features[r].tolist(), "token_step": token[r].tolist()}
                       for name in ("many", "middle", "few")}
            result.append({"uid": f"q{q:03}", "replicate": r, "correct": int(outcomes[r]),
                           "parseable": bool(outcomes[r] or r % 3), "family": "A" if q < n_uids // 2 else "B",
                           "features": {"step": features[r].tolist(), "token_step": token[r].tolist(),
                                        "regions": regions}})
    return result


def primary(result):
    return result["summary"]["primary"]


def explicit_fe(rows, design):
    uids = sorted({row["uid"] for row in rows})
    uid_index = {uid: i for i, uid in enumerate(uids)}
    y = np.array([row["correct"] for row in rows], dtype=float)
    ids = np.array([uid_index[row["uid"]] for row in rows])
    x = np.asarray(design, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    full = np.column_stack([np.eye(len(uids))[ids], x])
    beta = np.linalg.lstsq(full, y, rcond=None)[0]
    residual = y - full @ beta
    scores = np.stack([full[ids == q].T @ residual[ids == q] for q in range(len(uids))])
    inverse = np.linalg.inv(full.T @ full)
    n, g, p = len(rows), len(uids), x.shape[1]
    correction = g / (g - 1) * (n - 1) / (n - g - p)
    covariance = correction * inverse @ (scores.T @ scores) @ inverse
    return beta[-1], np.sqrt(covariance[-1, -1])


def test_primary_matches_explicit_fe_and_uid_cluster_covariance():
    rows = make_rows()
    result = analyze(rows, bootstrap_replicates=40, seed=7)
    effect = primary(result)
    expected, se = explicit_fe(rows, [r["features"]["step"][1] for r in rows])
    assert effect["status"] == "IDENTIFIABLE"
    assert effect["coefficient"] == pytest.approx(expected, rel=1e-10, abs=1e-10)
    assert effect["cluster_se"] == pytest.approx(se, rel=1e-10)
    assert effect["effect_pp_per_0_10"] == pytest.approx(10 * expected)
    assert effect["cluster_df"] == 9
    assert effect["n_uids"] == 10
    assert effect["n_runs"] == 160
    assert effect["mixed_outcome_uid_count"] == 10
    assert effect["p_value"] == pytest.approx(2 * stats.t.sf(abs(expected / se), 9))
    assert result["summary"]["n_unparseable"] > 0
    json.dumps(result, allow_nan=False)


def test_within_and_between_effects_can_have_opposite_signs():
    rows = make_rows(6)
    for row in rows:
        q = int(row["uid"][1:])
        count = 2 * (q + 1)
        y = int(row["replicate"] < count)
        x = 0.12 + 0.045 * q + 0.03 * (count / 16 - y)
        cells = [0.1, x, 0.6 - x, 0.1, 0.05, 0.15]
        row.update(correct=y, parseable=True)
        row["features"]["step"] = cells
        row["features"]["token_step"] = cells
    result = analyze(rows, bootstrap_replicates=50)
    assert primary(result)["coefficient"] == pytest.approx(-1 / 0.03)
    assert primary(result)["sign"] == "negative"
    assert primary(result)["inference_status"] == "UNAVAILABLE_ZERO_RESIDUAL_VARIANCE"
    between = next(r for r in result["summary"]["between_question_spearman"]
                   if r["weighting"] == "step" and r["feature"] == "t2")
    assert between["spearman_rho"] == pytest.approx(1.0)
    assert between["n_uids"] == 6


def test_bootstrap_repeated_uid_counts_and_family_strata_match_manual():
    rows = make_rows(8)
    b, seed = 61, 809
    result = analyze(rows, bootstrap_replicates=b, seed=seed)
    x = np.array([r["features"]["step"][1] for r in rows]).reshape(8, 16)
    y = np.array([r["correct"] for r in rows], dtype=float).reshape(8, 16)
    xd, yd = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
    xx, xy = (xd ** 2).sum(axis=1), (xd * yd).sum(axis=1)
    rng = np.random.default_rng(seed)
    weights = np.zeros((b, 8))
    weights[:, :4] = rng.multinomial(4, [0.25] * 4, size=b)
    weights[:, 4:] = rng.multinomial(4, [0.25] * 4, size=b)
    assert np.any(weights > 1)  # Repeated UIDs must retain their multiplicity.
    assert np.all(weights[:, :4].sum(axis=1) == 4)
    manual = weights @ xy / (weights @ xx)
    assert primary(result)["ci95"] == pytest.approx(np.quantile(manual, [0.025, 0.975]))
    assert primary(result)["bootstrap_valid_replicates"] == b
    assert result["summary"]["bootstrap"]["unit"] == "UID"
    assert result["summary"]["bootstrap"]["stratification"] == "family"
    # Identical arrays use identical UID resamples, including regional effects.
    region = next(r for r in result["effect_rows"] if r["id"] == "region_many_step_t2")
    assert region["ci95"] == primary(result)["ci95"]


def test_weighted_tie_ranks_match_expanded_sample_spearman():
    x = np.array([1.0, 1.0, 3.0, 4.0, 6.0])
    y = np.array([2.0, 1.0, 2.0, 5.0, 5.0])
    weights = np.array([[2, 0, 1, 1, 1], [0, 2, 0, 1, 2]], dtype=float)
    xr = _weighted_ranks(weights, _rank_layout(x))
    yr = _weighted_ranks(weights, _rank_layout(y))
    for i, w in enumerate(weights):
        expanded = np.repeat(np.arange(len(x)), w.astype(int))
        center = (len(expanded) + 1) / 2
        numerator = np.sum(w * (xr[i] - center) * (yr[i] - center))
        denominator = np.sqrt(np.sum(w * (xr[i] - center) ** 2) * np.sum(w * (yr[i] - center) ** 2))
        assert numerator / denominator == pytest.approx(stats.spearmanr(x[expanded], y[expanded]).statistic)


def test_specificity_includes_tie_flips_and_matches_explicit_adjustment():
    rows = make_rows(12, 2)
    result = analyze(rows, bootstrap_replicates=30)
    cells = np.array([r["features"]["step"] for r in rows])
    design = np.column_stack([cells[:, 0] + cells[:, 1] + cells[:, 4],
                              cells[:, 1] + cells[:, 3], cells[:, 1]])
    expected, se = explicit_fe(rows, design)
    effect = next(r for r in result["summary"]["specificity"] if r["weighting"] == "step")
    assert effect["status"] == "IDENTIFIABLE"
    assert effect["coefficient"] == pytest.approx(expected, rel=1e-9)
    assert effect["cluster_se"] == pytest.approx(se, rel=1e-9)
    assert effect["target_residual_sum_squares"] > 0
    wrong = design.copy()
    wrong[:, 0] -= cells[:, 4]
    wrong_estimate, _ = explicit_fe(rows, wrong)
    assert abs(expected - wrong_estimate) > 1e-3


def test_compositional_rank_deficiency_is_explicit():
    rows = make_rows(6)
    for row in rows:
        x = 0.05 + row["replicate"] / 100
        cells = [0.1, x, 0.6 - x, 0.1, 0.05, 0.15]
        for weighting in ("step", "token_step"):
            row["features"][weighting] = cells
    result = analyze(rows, bootstrap_replicates=20)
    assert primary(result)["status"] == "IDENTIFIABLE"
    for effect in result["summary"]["specificity"]:
        assert effect["status"] == "NONIDENTIFIABLE"
        assert effect["reason"] == "rank_deficient_within_question_design"
        assert effect["coefficient"] is None


def test_constant_outcomes_and_features_have_explicit_unavailability():
    rows = make_rows(6)
    for row in rows:
        row["correct"] = 1
        row["parseable"] = True
    result = analyze(rows, bootstrap_replicates=20)
    assert primary(result)["reason"] == "no_within_question_outcome_variation"
    assert primary(result)["mixed_outcome_uid_count"] == 0
    assert all(r["reason"] == "constant_problem_accuracy"
               for r in result["summary"]["between_question_spearman"])
    rows = make_rows(6)
    for row in rows:
        row["features"]["step"] = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
    result = analyze(rows, bootstrap_replicates=20)
    assert primary(result)["reason"] == "no_within_question_feature_variation"
    between = [r for r in result["summary"]["between_question_spearman"] if r["weighting"] == "step"]
    assert all(r["reason"] == "constant_problem_mean_feature" for r in between)
    json.dumps(result, allow_nan=False)


def test_mixed_outcome_count_does_not_drop_constant_outcome_uids():
    rows = make_rows(7)
    for row in rows:
        if row["uid"] == "q000":
            row.update(correct=0, parseable=False)
        if row["uid"] == "q001":
            row.update(correct=1, parseable=True)
    result = analyze(rows, bootstrap_replicates=20)
    assert primary(result)["n_runs"] == 7 * 16
    assert primary(result)["n_uids"] == 7
    assert primary(result)["mixed_outcome_uid_count"] == 5
    assert result["summary"]["mixed_outcome_uid_count"] == 5
    assert len(result["problem_rows"]) == 7


def test_missing_regions_are_excluded_with_support_and_never_zero_filled():
    rows = make_rows(6)
    for row in rows:
        row["features"]["regions"]["few"] = {"step": None, "token_step": None}
        if row["replicate"] % 2:
            row["features"]["regions"]["middle"] = {"step": None, "token_step": None}
    result = analyze(rows, bootstrap_replicates=20)
    assert primary(result)["n_runs"] == 96
    few_effects = [r for r in result["effect_rows"] if r["region"] == "few"]
    assert all(r["reason"] == "region_unobserved" and r["n_runs"] == 0 for r in few_effects)
    few_summaries = [r for r in result["depth_rows"] if r["region"] == "few"]
    assert all(r["mean"] is None and r["status"] == "UNOBSERVED" for r in few_summaries)
    row = next(r for r in result["depth_rows"] if r["region"] == "middle"
               and r["weighting"] == "step" and r["feature"] == "t2" and r["correct"] == 1)
    observed = [r["features"]["step"][1] for r in rows if r["correct"] == 1 and r["replicate"] % 2 == 0]
    assert row["mean"] == pytest.approx(np.mean(observed))
    assert row["n_runs"] == len(observed)
    assert row["missing_region_correctness_group_runs"] > 0
    assert all(r["observed_runs_middle"] == 8 and r["observed_runs_few"] == 0 for r in result["problem_rows"])


def test_shuffle_invariance_and_json_serialization():
    rows = make_rows(8)
    result = analyze(rows, bootstrap_replicates=35, seed=42)
    shuffled = deepcopy(rows)
    random.Random(7).shuffle(shuffled)
    assert analyze(shuffled, bootstrap_replicates=35, seed=42) == result
    json.dumps(result, allow_nan=False)


def test_holm_adjusts_four_tests_and_primary_raw_p_remains_available():
    result = analyze(make_rows(14), bootstrap_replicates=0)
    for weighting in ("step", "token_step"):
        family = [r for r in result["effect_rows"]
                  if r["analysis"] == "univariate_within_question_lpm" and r["weighting"] == weighting]
        ordered = sorted(family, key=lambda r: r["p_value"])
        previous = 0
        for i, row in enumerate(ordered):
            previous = min(1, max(previous, (4 - i) * row["p_value"]))
            assert row["p_holm"] == pytest.approx(previous)
            assert row["holm_family_size"] == 4
    assert primary(result)["p_value"] <= primary(result)["p_holm"]


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows.pop(), "all 16 unique"),
    (lambda rows: rows.append(deepcopy(rows[0])), "duplicate replicate"),
    (lambda rows: rows[0].update(family="other"), "family must be constant"),
    (lambda rows: rows[0].update(correct=1, parseable=False), "unparseable"),
    (lambda rows: rows[0]["features"].update(step=[0.2] * 6), "sum to one"),
    (lambda rows: rows[0].update(replicate=0.5), "replicate must"),
])
def test_invalid_input_fails_before_analysis(mutation, match):
    rows = make_rows(4)
    mutation(rows)
    with pytest.raises(ValueError, match=match):
        analyze(rows, bootstrap_replicates=0)


def test_few_clusters_and_empty_bootstrap_are_reported_without_nan():
    rows = make_rows(1)
    result = analyze(rows, bootstrap_replicates=0)
    assert primary(result)["status"] == "IDENTIFIABLE"
    assert primary(result)["cluster_se"] is None
    assert primary(result)["cluster_df"] is None
    assert primary(result)["ci95"] is None
    json.dumps(result, allow_nan=False)


def test_singleton_only_family_strata_do_not_create_false_precision():
    rows = make_rows(4)
    for row in rows:
        row["family"] = row["uid"]
    result = analyze(rows, bootstrap_replicates=100)
    assert primary(result)["status"] == "IDENTIFIABLE"
    assert primary(result)["ci95"] is None
    assert primary(result)["bootstrap_status"] == "UNAVAILABLE_NO_WITHIN_FAMILY_RESAMPLING_VARIATION"
    assert result["summary"]["bootstrap"]["singleton_family_count"] == 4
    assert primary(result)["bootstrap_requested_replicates"] == 100
    assert primary(result)["bootstrap_valid_replicates"] == 0

