"""Selection uses validation, retains every seed, and attaches the paper's references."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from analysis.selection import (
    aggregate_selected,
    attach_references,
    select_thresholds,
    target_rise,
)
from theory.reference import binary_entropy

PARAMS = json.dumps(
    {
        "C": 4,
        "family": "powerset",
        "dual_shatter_mass": 1.0,
        "zero_error_bits": 2,
        "zero_error_codewords": 4,
    }
)


def record(run_id="r0", **updates):
    row = {
        "run_id": run_id,
        "config_hash": f"hash-{run_id}",
        "condition": "powerset-4",
        "concept_family": "powerset",
        "concept_parameters": PARAMS,
        "dual_vc_dim": 2,
        "num_inputs": 4,
        "num_tasks": 16,
        "seed": 0,
        "status": "completed",
        "declared_capacity_bits": 6.0,
        "val_max_task_error": 0.02,
        "val_max_task_error_ucb": 0.03,
        "max_task_error": 0.02,
        "max_task_error_ucb": 0.04,
    }
    row.update(updates)
    return row


def conditions_for(records: pd.DataFrame, grid: list[float]) -> pd.DataFrame:
    return attach_references(aggregate_selected(select_thresholds(records, records, grid)))


def test_validation_selects_capacity_and_test_is_reported_once():
    records = pd.DataFrame(
        [
            record(
                "small",
                declared_capacity_bits=4,
                val_max_task_error_ucb=0.2,
                max_task_error_ucb=0.001,
            ),
            record("large", declared_capacity_bits=6, max_task_error_ucb=0.09),
        ]
    )
    out = select_thresholds(records, records, [0.05])
    assert len(out) == 1
    assert out.iloc[0]["run_id"] == "large"
    assert out.iloc[0]["selected_bits"] == 6
    assert out.iloc[0]["outcome"] == "heldout_miss"


def test_missing_no_pass_and_test_miss_are_distinct_and_keep_denominators():
    declared = pd.DataFrame(
        [
            record("s0", seed=0),
            record("s1", seed=1, val_max_task_error_ucb=0.2),
            record("s2", seed=2, max_task_error_ucb=0.2),
            record("s3", seed=3),
            record("s4", seed=4, status="diverged"),
        ]
    )
    available = declared[declared["seed"] != 3]
    out = select_thresholds(available, declared, [0.05])
    assert list(out["outcome"]) == [
        "heldout_pass",
        "no_validation_pass",
        "heldout_miss",
        "missing_runs",
        "no_validation_pass",
    ]
    row = aggregate_selected(out).iloc[0]
    assert row["n_seeds_expected"] == 5
    assert row["n_seeds_selected"] == 2
    assert row["n_heldout_pass"] == 1
    assert row["n_missing_seeds"] == 1
    assert row["n_no_validation_pass"] == 2
    assert row["n_heldout_misses"] == 1
    assert not row["all_seeds_selected"]
    assert not row["all_heldout_pass"]


def test_no_pass_keeps_condition_with_undefined_capacity():
    records = pd.DataFrame([record(val_max_task_error_ucb=0.4)])
    row = aggregate_selected(select_thresholds(records, records, [0.01])).iloc[0]
    assert row["n_seeds_selected"] == 0
    assert pd.isna(row["median_bits"])


def test_incomplete_variant_grid_does_not_claim_minimum_capacity():
    declared = pd.DataFrame([record("a"), record("b", declared_capacity_bits=3)])
    out = select_thresholds(declared.iloc[:1], declared, [0.1]).iloc[0]
    assert out["outcome"] == "missing_runs"
    assert not out["selected"]


def test_ties_use_validation_then_run_id():
    records = pd.DataFrame([record("b"), record("a"), record("c", val_max_task_error_ucb=0.04)])
    out = select_thresholds(records, records, [0.05]).iloc[0]
    assert out["run_id"] == "a"


def test_references_and_gaps():
    row = conditions_for(pd.DataFrame([record()]), [0.1]).iloc[0]
    # dual VC 2 with shattered mass 1: B_sh = (1 - h2(0.1)) * 2, B_0 = 2 bits
    assert row["lower_bits"] == pytest.approx((1 - binary_entropy(0.1)) * 2)
    assert row["zero_error_bits"] == 2
    assert row["zero_error_gap_bits"] == 4
    assert row["shattered_gap_bits"] == pytest.approx(6 - (1 - binary_entropy(0.1)) * 2)


def test_target_outside_the_bound_range_fails_loudly():
    params = json.loads(PARAMS)
    params["dual_shatter_mass"] = 2 / 3
    records = pd.DataFrame([record(concept_parameters=json.dumps(params))])
    assert conditions_for(records, [0.05]).iloc[0]["lower_bits"] > 0
    with pytest.raises(ValueError, match="outside the bound's range"):
        conditions_for(records, [0.4])


def test_target_rise_uses_only_selected_targets():
    records = pd.DataFrame(
        [
            record("small", declared_capacity_bits=3, val_max_task_error_ucb=0.08),
            record("large", declared_capacity_bits=5, val_max_task_error_ucb=0.02),
        ]
    )
    rise = target_rise(conditions_for(records, [0.01, 0.05, 0.1])).iloc[0]
    assert rise["n_targets_selected"] == 2
    assert rise["loosest_epsilon"] == 0.1 and rise["strictest_epsilon"] == 0.05
    assert rise["bits_at_loosest"] == 3 and rise["bits_at_strictest"] == 5
    assert rise["rise_bits"] == 2
    single = target_rise(conditions_for(records.iloc[:1], [0.01, 0.1])).iloc[0]
    assert single["n_targets_selected"] == 1 and np.isnan(single["rise_bits"])
