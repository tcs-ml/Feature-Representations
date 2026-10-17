"""The analysis of the saved outputs is complete, repeatable, and matches the paper."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.data import (
    _validate_records,
    attach_binomial_bounds,
    expected_runs,
    load_analysis_inputs,
    read_records,
    validate_analysis_config,
)
from analysis.paper import run_analysis
from analysis.selection import aggregate_selected, select_thresholds
from evaluation.confidence import binomial_upper_bound
from runner import _execution_plan

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def analysis_config():
    return _execution_plan(ROOT / "configs/analysis.json")[0]


@pytest.fixture(scope="module")
def analysis(analysis_config, tmp_path_factory):
    out = tmp_path_factory.mktemp("analysis")
    return out, run_analysis(analysis_config, out)


def test_expected_grid_matches_the_saved_runs(analysis_config):
    # 19 instances x 36 configurations x 5 seeds: the paper's 3,420 training runs
    expected = expected_runs(analysis_config)
    assert len(expected) == 3420
    assert expected["condition"].nunique() == 19
    assert (expected["n_validation"] == 250 * expected["num_inputs"]).all()
    assert (expected["n_test"] == 500 * expected["num_inputs"]).all()
    records, manifest, _ = load_analysis_inputs(analysis_config)
    assert len(records) == len(expected)
    assert manifest["present"].all()
    assert (records["status"] == "completed").all()


def test_loaded_bounds_are_binomial_and_agree_with_the_records(analysis_config):
    records, manifest, _ = load_analysis_inputs(analysis_config)
    raw = read_records(Path(analysis_config["sources"]["records"])).set_index("run_id")
    sizes = manifest.set_index("run_id")
    sample = records.sort_values("run_id").head(40)
    for row in sample.itertuples():
        n = int(sizes.at[row.run_id, "n_test"])
        errors = round(float(row.max_task_error) * n)
        assert row.max_task_error_ucb == pytest.approx(binomial_upper_bound(errors, n, 0.95))
    # the runs recorded the same bound, so the recomputation changes nothing
    pd.testing.assert_series_equal(
        raw.loc[sample["run_id"], "max_task_error_ucb"],
        sample.set_index("run_id")["max_task_error_ucb"],
        check_names=False,
    )


def test_analysis_is_repeatable_and_writes_only_the_paper_tables(
    analysis_config, analysis, tmp_path
):
    out, first = analysis
    second = run_analysis(analysis_config, tmp_path / "second")
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name], check_exact=True)
    assert sorted(p.name for p in out.iterdir()) == ["aggregate", "paper_data"]
    assert sorted(p.stem for p in (out / "aggregate").glob("*.csv")) == sorted(first)
    assert sorted(p.stem for p in (out / "paper_data").glob("*.csv")) == sorted(
        [f"greedy_threshold_d{d}" for d in (1, 2, 3, 4)]
        + [f"capacity_{family}" for family in ("parity", "powerset", "transposed_powerset")]
        + [f"seeds_{family}" for family in ("parity", "powerset", "transposed_powerset")]
    )


def test_analysis_reproduces_the_paper_counts(analysis):
    _, tables = analysis
    conditions = tables["conditions"]
    assert len(conditions) == 114  # 19 instances x 6 targets
    assert (conditions["n_seeds_selected"] > 0).sum() == 99
    assert conditions["n_missing_seeds"].sum() == 0
    assert conditions["n_no_validation_pass"].sum() == 91
    assert conditions["n_heldout_misses"].sum() == 5
    outcomes = tables["seed_outcomes"]
    selected = outcomes[outcomes["selected"].eq(True)]
    assert len(outcomes) == 570 and len(selected) == 479
    assert int(selected["heldout_pass"].sum()) == 474
    # "all selections meet their targets in observed test error"
    assert not (selected["heldout_max_task_error"] > selected["epsilon"] + 1e-9).any()
    # "the median budget is typically one to three bits above B_0 and exceeds B_sh everywhere"
    plotted = conditions[conditions["n_seeds_selected"] > 0]
    assert plotted["zero_error_gap_bits"].between(1, 3).sum() == 94
    assert plotted["zero_error_gap_bits"].median() == 2
    assert plotted["shattered_gap_bits"].min() == pytest.approx(1.5727939142319125)
    assert (plotted["median_bits"] > plotted["lower_bits"]).all()
    assert (conditions["zero_error_bits"] <= conditions["dual_vc_dim"] + 1).all()
    # "in 15 of the 18 instances with results at multiple targets the median rises by <= 2 bits"
    rise = tables["target_rise"]
    assert rise["rise_bits"].notna().sum() == 18
    assert (rise["rise_bits"] <= 2).sum() == 15
    # "a nearest-prototype encoder meets the target with B_0 bits in every selected setting"
    prototype = tables["prototype_decoder"]
    assert len(prototype) == 19 * 5 * 2  # instances x seeds x splits
    assert (prototype["worst_task_error_ucb"] > prototype["worst_task_error"]).all()
    certification = tables["prototype_certification"].merge(
        conditions[["condition", "epsilon", "n_seeds_selected"]], on=["condition", "epsilon"]
    )
    assert len(certification) == 114
    with_selection = certification[certification["n_seeds_selected"] > 0]
    assert (with_selection["seeds_test_certified"] == with_selection["n_seeds"]).all()


def test_shipped_paper_data_matches_a_fresh_analysis(analysis):
    out, _ = analysis
    for path in sorted((out / "paper_data").glob("*.csv")):
        shipped = ROOT / "outputs/analysis/paper_data" / path.name
        assert shipped.read_text() == path.read_text(), path.name


def test_missing_source_fails_without_fallback(analysis_config, tmp_path):
    cfg = copy.deepcopy(analysis_config)
    cfg["sources"]["records"] = str(tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError, match="analysis input not found"):
        load_analysis_inputs(cfg)


@pytest.mark.parametrize(
    "key,value",
    [
        ("sources", None),
        ("sources", {"records": "a"}),
        ("sources", {"records": "a", "arrays": "b", "greedy": "c"}),
        ("sources", {"records": {"path": "a"}, "greedy": "c"}),
        ("sweep", ""),
        ("sae_sweeps", {"configs": ["x.json"]}),
        ("expected_seeds", [5]),
        ("epsilon_grid", [0.5]),
        ("outputs", ""),
    ],
)
def test_invalid_analysis_shape_has_clear_error(analysis_config, key, value):
    cfg = copy.deepcopy(analysis_config)
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_analysis_config(cfg)


def test_sweep_must_be_an_sae_experiment(analysis_config):
    cfg = copy.deepcopy(analysis_config)
    cfg["sweep"] = str(ROOT / "configs/experiment/greedy.json")
    with pytest.raises(ValueError, match="SAE experiment"):
        load_analysis_inputs(cfg)


def test_absent_runs_are_reported_as_missing(analysis_config):
    expected = expected_runs(analysis_config)
    nothing = pd.DataFrame(columns=["run_id", "config_hash", "status"])
    outcomes = select_thresholds(nothing, expected, analysis_config["epsilon_grid"])
    assert set(outcomes["outcome"]) == {"missing_runs"}
    assert not aggregate_selected(outcomes)["all_heldout_pass"].any()


def test_failed_records_need_no_metrics():
    _validate_records(pd.DataFrame([{"run_id": "failed", "status": "diverged"}]))


def test_completed_records_need_finite_metrics():
    records = pd.DataFrame(
        [
            {
                "run_id": "invalid",
                "status": "completed",
                "max_task_error": float("nan"),
                "val_max_task_error": 0.1,
                "max_task_error_ucb": 0.1,
                "val_max_task_error_ucb": 0.1,
                "declared_capacity_bits": 4,
            }
        ]
    )
    with pytest.raises(ValueError, match="invalid completed record"):
        _validate_records(records)


def test_attach_binomial_bounds_replaces_recorded_values():
    records = pd.DataFrame(
        [
            {
                "run_id": "r",
                "status": "completed",
                "max_task_error": 3 / 200,
                "val_max_task_error": 0.0,
                "max_task_error_ucb": 0.9,
                "val_max_task_error_ucb": 0.9,
            },
            {
                "run_id": "f",
                "status": "diverged",
                "max_task_error": np.nan,
                "val_max_task_error": np.nan,
                "max_task_error_ucb": np.nan,
                "val_max_task_error_ucb": np.nan,
            },
        ]
    )
    expected = pd.DataFrame(
        [
            {"run_id": "r", "n_validation": 100, "n_test": 200, "confidence_level": 0.95},
            {"run_id": "f", "n_validation": 100, "n_test": 200, "confidence_level": 0.95},
        ]
    )
    out = attach_binomial_bounds(records, expected).set_index("run_id")
    assert out.at["r", "max_task_error_ucb"] == pytest.approx(binomial_upper_bound(3, 200, 0.95))
    assert out.at["r", "val_max_task_error_ucb"] == pytest.approx(
        binomial_upper_bound(0, 100, 0.95)
    )
    assert np.isnan(out.at["f", "max_task_error_ucb"])
    records.loc[0, "max_task_error"] = 0.0149  # not a multiple of 1 / n
    with pytest.raises(ValueError, match="multiples"):
        attach_binomial_bounds(records, expected)


def test_analysis_cannot_overwrite_or_enclose_its_sources(analysis_config):
    for outputs in (ROOT / "outputs", ROOT / "outputs/sae", ROOT / "outputs/greedy"):
        with pytest.raises(ValueError, match="separate from source"):
            run_analysis(analysis_config, outputs)
