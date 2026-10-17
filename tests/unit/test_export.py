"""Paper tables: greedy grids, per-setting medians, per-seed rows, and file layout."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from analysis.export import (
    capacity_tables,
    closed_form_bits,
    greedy_tables,
    seed_tables,
    write_tables,
)
from analysis.selection import aggregate_selected, attach_references, select_thresholds

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


def greedy_run(d: int) -> dict:
    return {
        "d": d,
        "N": 8,
        "rows": [{"eps": 0.25, "t": 2, "bits": 2.0}, {"eps": 0.125, "t": 6, "bits": 4.0}],
    }


def test_greedy_tables_hold_one_grid_each_with_the_scaling_predictor():
    tables = greedy_tables({"runs": [greedy_run(1), greedy_run(2)]})
    assert list(tables) == ["greedy_threshold_d1", "greedy_threshold_d2"]
    table = tables["greedy_threshold_d2"]
    assert list(table["predictor"]) == pytest.approx([4.0, 6.0])
    assert list(table["bits"]) == [2.0, 4.0]
    assert list(table["closed_form_bits"]) == [2, 4]
    assert set(table["N"]) == {8} and set(table["dimension"]) == {2}


def test_closed_form_is_d_times_ceil_log2_of_one_over_two_eps():
    assert closed_form_bits(1, 0.25) == 1
    assert closed_form_bits(3, 0.15) == 6
    assert closed_form_bits(4, 0.004) == 4 * 7


def test_greedy_tables_reject_two_grids_with_one_dimension():
    with pytest.raises(ValueError, match="share dimension"):
        greedy_tables({"runs": [greedy_run(2), greedy_run(2)]})


def test_write_tables_saves_one_csv_per_table(tmp_path):
    write_tables({"a": pd.DataFrame({"x": [1]}), "b": pd.DataFrame({"y": [2]})}, tmp_path / "data")
    assert sorted(path.name for path in (tmp_path / "data").iterdir()) == ["a.csv", "b.csv"]
    assert pd.read_csv(tmp_path / "data" / "a.csv")["x"].tolist() == [1]


def test_capacity_tables_keep_unselected_settings():
    records = pd.DataFrame([record("a", val_max_task_error_ucb=0.03)])
    conditions = attach_references(
        aggregate_selected(select_thresholds(records, records, [0.01, 0.05]))
    )
    tables = capacity_tables(conditions)
    assert list(tables) == ["capacity_powerset"]
    table = tables["capacity_powerset"]
    assert list(table["epsilon"]) == [0.05, 0.01]  # strictest target last
    assert list(table["any_selected"]) == [1, 0]
    assert table.iloc[0]["median_bits"] == 6 and np.isnan(table.iloc[1]["median_bits"])
    assert table.iloc[0]["log2_inv_epsilon"] == pytest.approx(np.log2(20))
    assert list(table["zero_error_bits"]) == [2, 2]


def test_seed_tables_hold_one_row_per_selected_seed():
    outcomes = pd.DataFrame(
        [
            {
                "concept_family": "parity",
                "condition": "parity-2",
                "epsilon": 0.1,
                "seed": s,
                "selected": sel,
                "selected_bits": b,
                "heldout_pass": hp,
            }
            for s, sel, b, hp in [
                (0, True, 4.0, True),
                (1, True, 5.0, False),
                (2, False, float("nan"), False),
            ]
        ]
    )
    conditions = pd.DataFrame([{"condition": "parity-2", "epsilon": 0.1, "lower_bits": 1.06}])
    tables = seed_tables(outcomes, conditions)
    assert list(tables) == ["seeds_parity"]
    table = tables["seeds_parity"]
    assert list(table["seed"]) == [0, 1]
    assert list(table["heldout_pass"]) == [1, 0]
    assert list(table["lower_bits"]) == [1.06, 1.06]
    assert table["log2_inv_epsilon"].iloc[0] == pytest.approx(np.log2(10))
