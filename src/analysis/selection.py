"""Validation-selected bit budgets per instance and target.

Vocabulary: a *setting* is one instance of a task
family at one error target; a *selection* is the configuration with the
smallest bit budget whose validation worst-case error bound meets the target
for one seed; a selection is *certified* when its test bound also meets the
target (column `heldout_pass`).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from theory.reference import shattered_set_bound

CONCEPT_COLUMNS = ("concept_family", "concept_parameters", "dual_vc_dim", "num_inputs", "num_tasks")


def select_thresholds(
    records: pd.DataFrame, expected: pd.DataFrame, epsilon_grid: list[float]
) -> pd.DataFrame:
    """One outcome for every declared instance, seed, and error target.

    Incomplete sweeps are missing_runs; complete sweeps without a passing
    configuration are no_validation_pass. Selected runs report heldout_pass
    (certified) or heldout_miss; test errors never choose a configuration.
    """
    indexed = records.set_index("run_id")
    rows = []
    for (condition, seed), group in expected.groupby(["condition", "seed"], sort=True):
        available = indexed.reindex(group["run_id"])
        present = available["config_hash"].notna()
        missing = int((~present).sum())
        finished = available[present & (available.get("status") == "completed")]
        meta = group.iloc[0]
        for eps in sorted(epsilon_grid):
            row = {
                **{key: meta.get(key) for key in CONCEPT_COLUMNS},
                "condition": condition,
                "seed": int(seed),
                "epsilon": float(eps),
                "n_runs_expected": len(group),
                "n_runs_present": int(present.sum()),
                "n_runs_failed": int(present.sum()) - len(finished),
                "selected": False,
                "heldout_pass": False,
                "run_id": None,
                "selected_bits": np.nan,
                "val_max_task_error": np.nan,
                "val_max_task_error_ucb": np.nan,
                "heldout_max_task_error": np.nan,
                "heldout_max_task_error_ucb": np.nan,
                "outcome": "missing_runs" if missing else "no_validation_pass",
            }
            if not missing and not finished.empty:
                passing = finished[
                    (finished["val_max_task_error_ucb"] <= eps + 1e-9)
                    & finished["declared_capacity_bits"].notna()
                ]
                if not passing.empty:
                    best = (
                        passing.reset_index()
                        .sort_values(
                            ["declared_capacity_bits", "val_max_task_error_ucb", "run_id"],
                            kind="stable",
                        )
                        .iloc[0]
                    )
                    heldout = bool(best["max_task_error_ucb"] <= eps + 1e-9)
                    row.update(
                        selected=True,
                        heldout_pass=heldout,
                        run_id=best["run_id"],
                        selected_bits=float(best["declared_capacity_bits"]),
                        val_max_task_error=best["val_max_task_error"],
                        val_max_task_error_ucb=best["val_max_task_error_ucb"],
                        heldout_max_task_error=best["max_task_error"],
                        heldout_max_task_error_ucb=best["max_task_error_ucb"],
                        outcome="heldout_pass" if heldout else "heldout_miss",
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_selected(outcomes: pd.DataFrame) -> pd.DataFrame:
    """One row per setting over every expected seed, including missing and unsuccessful cases."""
    rows = []
    for (condition, eps), group in outcomes.groupby(["condition", "epsilon"], sort=True):
        selected = group[group["selected"]]
        expected = len(group)
        count = len(selected)
        heldout = int(group["heldout_pass"].sum())
        rows.append(
            {
                **{key: group.iloc[0].get(key) for key in CONCEPT_COLUMNS},
                "condition": condition,
                "epsilon": eps,
                "median_bits": selected["selected_bits"].median() if count else np.nan,
                "min_bits": selected["selected_bits"].min() if count else np.nan,
                "max_bits": selected["selected_bits"].max() if count else np.nan,
                "n_seeds_expected": expected,
                "n_seeds_selected": count,
                "n_heldout_pass": heldout,
                "n_missing_seeds": int((group["outcome"] == "missing_runs").sum()),
                "n_no_validation_pass": int((group["outcome"] == "no_validation_pass").sum()),
                "n_heldout_misses": int((group["outcome"] == "heldout_miss").sum()),
                "all_seeds_selected": count == expected,
                "all_heldout_pass": heldout == expected,
            }
        )
    return pd.DataFrame(rows)


def attach_references(aggregated: pd.DataFrame) -> pd.DataFrame:
    """Add two noiseless references and the median's gap to each.

    `lower_bits` is the shattered-set bound B_sh(eps) and `zero_error_bits`
    the lossless reference B_0 of the instance. A target outside the bound's
    range raises instead of leaving a silent blank.
    """
    out = aggregated.copy()
    lower, zero = [], []
    for row in out.itertuples():
        params = row.concept_parameters
        params = json.loads(params) if isinstance(params, str) else params
        lower.append(
            shattered_set_bound(int(row.dual_vc_dim), params["dual_shatter_mass"], row.epsilon)
        )
        zero.append(float(params["zero_error_bits"]))
    out["lower_bits"] = lower
    out["zero_error_bits"] = zero
    out["shattered_gap_bits"] = out["median_bits"] - out["lower_bits"]
    out["zero_error_gap_bits"] = out["median_bits"] - out["zero_error_bits"]
    return out


RISE_FIELDS = (
    "strictest_epsilon",
    "loosest_epsilon",
    "bits_at_strictest",
    "bits_at_loosest",
    "rise_bits",
)


def target_rise(conditions: pd.DataFrame) -> pd.DataFrame:
    """Per instance: the median selected budget at its loosest and strictest selected targets.

    `rise_bits` is the growth between the two; it is undefined for instances
    with selections at fewer than two targets.
    """
    rows = []
    for condition, group in conditions.groupby("condition", sort=True):
        selected = group[group["n_seeds_selected"] > 0].sort_values("epsilon")
        row = {
            "concept_family": group["concept_family"].iloc[0],
            "condition": condition,
            "dual_vc_dim": int(group["dual_vc_dim"].iloc[0]),
            "zero_error_bits": float(group["zero_error_bits"].iloc[0]),
            "n_targets_selected": int(len(selected)),
            **dict.fromkeys(RISE_FIELDS, np.nan),
        }
        if len(selected) >= 2:
            strict, loose = selected.iloc[0], selected.iloc[-1]
            row.update(
                strictest_epsilon=float(strict["epsilon"]),
                loosest_epsilon=float(loose["epsilon"]),
                bits_at_strictest=float(strict["median_bits"]),
                bits_at_loosest=float(loose["median_bits"]),
                rise_bits=float(strict["median_bits"] - loose["median_bits"]),
            )
        rows.append(row)
    return pd.DataFrame(rows)
