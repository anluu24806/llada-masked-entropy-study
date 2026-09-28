"""Pure NumPy summaries of adjacent pre-denoising masked-token observations.

Entropy is H(softmax(raw_logits.float())) / log(vocabulary_size). The six
categories are ordered t1, t2, t3, t4, tie_flip, tie_same; ties use epsilon=0.
A comparison uses the survivor intersection masks[t-1] & masks[t], never gold
answers, accepted tokens, or the union of positions masked at either step.
"""
from __future__ import annotations

import numpy as np

GROUPS = ("t1", "t2", "t3", "t4", "tie_flip", "tie_same")
REGIONS = ("many", "middle", "few")


def _region(masked: int, positions: int) -> str:
    # Integer comparisons make the 1/3 and 2/3 boundaries unambiguous.
    if 3 * masked > 2 * positions:
        return "many"
    if 3 * masked > positions:
        return "middle"
    return "few"


def _aggregate(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["eligible"]]
    common_count = sum(row["common_count"] for row in valid)
    if not valid:
        return {"step": None, "token_step": None, "n_intervals": 0,
                "common_count": 0, "counts": [0] * 6,
                "mean_entropy_delta": None}
    counts = np.asarray([row["counts"] for row in valid], dtype=np.int64).sum(axis=0)
    rates = np.asarray([row["rates"] for row in valid], dtype=np.float64)
    return {
        "step": rates.mean(axis=0).tolist(),
        "token_step": (counts / common_count).tolist(),
        "n_intervals": len(valid),
        "common_count": common_count,
        "counts": counts.tolist(),
        "mean_entropy_delta": float(sum(
            row["mean_entropy_delta"] * row["common_count"] for row in valid
        ) / common_count),
    }


def summarize_trace(entropy, proposals, masks) -> dict:
    """Summarize [T,G] arrays without silently dropping missing observations.

    ``steps`` has one row per observation, with zero-based step indices. Row 0
    has no preceding observation, is explicitly ineligible, and has zero count
    and rate placeholders. Empty survivor intersections are likewise explicit
    ineligible rows and are excluded from both averages. A feature with no
    eligible intervals is ``None`` (JSON null), never a manufactured zero or NaN.

    The region of interval t-1 -> t is the current pre-denoising masked fraction
    masks[t].sum()/G. The primary ``step`` feature weights eligible intervals
    equally. ``token_step`` weights every compared token-step equally.
    """
    arrays = tuple(np.asarray(value) for value in (entropy, proposals, masks))
    ent, prop, mask = arrays
    if any(arr.ndim != 2 for arr in arrays):
        raise ValueError("entropy, proposals, and masks must all have shape [T,G]")
    if any(arr.shape != ent.shape for arr in arrays) or min(ent.shape) == 0:
        raise ValueError("trace arrays must have the same nonempty shape")
    if ent.dtype.kind not in "fiu" or not np.isfinite(ent).all():
        raise ValueError("every entropy observation must be numeric and finite")
    if np.any(ent < -2e-6) or np.any(ent > 1.0 + 2e-6):
        raise ValueError("entropy must be normalized by log(vocabulary_size)")
    if prop.dtype.kind not in "iu" or np.any(prop < 0):
        raise ValueError("proposals must be nonnegative integer raw argmax IDs")
    if mask.dtype.kind != "b":
        raise ValueError("masks must be boolean pre-denoising indicators")

    n_steps, n_positions = ent.shape
    rows = []
    for step in range(n_steps):
        masked_count = int(mask[step].sum())
        row = {
            "step": step,
            "masked_count": masked_count,
            "common_count": 0,
            "mask_fraction": masked_count / n_positions,
            "region": _region(masked_count, n_positions),
            "counts": [0] * 6,
            "rates": [0.0] * 6,
            "eligible": False,
            "skipped_reason": None,
            "mean_entropy_delta": None,
        }
        if step == 0:
            row["skipped_reason"] = "first_step_no_previous"
        else:
            common = mask[step - 1] & mask[step]
            n_common = int(common.sum())
            row["common_count"] = n_common
            if n_common == 0:
                row["skipped_reason"] = (
                    "terminal_mask_zero" if masked_count == 0
                    else "empty_survivor_intersection"
                )
            else:
                previous = ent[step - 1, common]
                current = ent[step, common]
                up, down = current > previous, current < previous
                tie = current == previous
                flip = prop[step, common] != prop[step - 1, common]
                same = ~flip
                counts = [int(group.sum()) for group in (
                    up & flip, down & flip, up & same, down & same,
                    tie & flip, tie & same,
                )]
                if sum(counts) != n_common:
                    raise AssertionError("six categories must partition the survivor set")
                row.update(
                    counts=counts,
                    rates=[count / n_common for count in counts],
                    eligible=True,
                    mean_entropy_delta=float(np.mean(
                        current.astype(np.float64) - previous.astype(np.float64)
                    )),
                )
        rows.append(row)

    overall = _aggregate(rows)
    regions = {name: _aggregate([row for row in rows if row["region"] == name])
               for name in REGIONS}
    return {
        "steps": rows,
        "features": {**overall, "regions": regions},
        "diagnostics": {
            "group_order": list(GROUPS),
            "tie_epsilon": 0.0,
            "n_steps": n_steps,
            "n_positions": n_positions,
            "n_eligible_intervals": overall["n_intervals"],
            "n_skipped_first_steps": 1,
            "n_skipped_empty_intervals": sum(
                not row["eligible"] for row in rows[1:]
            ),
            "n_terminal_mask_zero_intervals": sum(
                row["skipped_reason"] == "terminal_mask_zero" for row in rows
            ),
            "total_common_count": overall["common_count"],
            "partition_counts_valid": all(
                sum(row["counts"]) == row["common_count"] for row in rows
            ),
            "last_observed_masked_count": rows[-1]["masked_count"],
            "normalization": "raw FP32 softmax entropy / log(vocabulary_size)",
            "region_basis": "current pre-denoising masked_count / generation_length",
            "comparison_set": "previous_mask AND current_mask",
            "first_step_is_interval": False,
        },
    }
