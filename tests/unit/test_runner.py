"""JSON-only orchestration, preflight validation and file-relative paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import ConfigError
from runner import _execution_plan, run_config


def write_config(path: Path, config: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config))
    return path


def greedy(output="results") -> dict:
    return {
        "kind": "greedy",
        "outputs": output,
        "experiment": {"name": "tiny", "grid_sizes": [[1, 8]], "epsilon_grid": [0.15, 0.25]},
    }


def test_suite_order_and_relative_paths(tmp_path, monkeypatch):
    suite = write_config(
        tmp_path / "suite.json", {"kind": "suite", "configs": ["a/one.json", "b/two.json"]}
    )
    write_config(tmp_path / "a/one.json", greedy("../result-a"))
    write_config(tmp_path / "b/two.json", greedy("../result-b"))
    called = []
    monkeypatch.setattr("runner._run", lambda cfg: called.append(cfg["outputs"]))
    other = tmp_path / "unrelated"
    other.mkdir()
    monkeypatch.chdir(other)
    assert run_config(suite) is None
    assert called == [str(tmp_path / "result-a"), str(tmp_path / "result-b")]
    assert Path.cwd() == other
    assert not (other / "results").exists()
    assert len(list((tmp_path / "result-a/configs").glob("*.json"))) == 1


def test_invalid_later_member_does_not_start_any_work(tmp_path, monkeypatch):
    suite = write_config(
        tmp_path / "suite.json", {"kind": "suite", "configs": ["ok.json", "bad.json"]}
    )
    write_config(tmp_path / "ok.json", greedy())
    write_config(tmp_path / "bad.json", {"kind": "removed"})
    monkeypatch.setattr("runner._run", lambda cfg: pytest.fail("started invalid suite"))
    with pytest.raises(ConfigError):
        run_config(suite)
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("configs", [["suite.json"], ["ok.json", "./ok.json"]])
def test_nested_or_duplicate_suite_rejected(tmp_path, configs):
    suite = write_config(tmp_path / "suite.json", {"kind": "suite", "configs": configs})
    write_config(tmp_path / "ok.json", greedy())
    with pytest.raises(ConfigError):
        run_config(suite)
    assert not (tmp_path / "results").exists()


def test_execution_errors_stop_suite(tmp_path, monkeypatch):
    suite = write_config(
        tmp_path / "suite.json", {"kind": "suite", "configs": ["a.json", "b.json"]}
    )
    write_config(tmp_path / "a.json", greedy("a-results"))
    write_config(tmp_path / "b.json", greedy("b-results"))

    def fail(cfg):
        raise RuntimeError("failed first stage")

    monkeypatch.setattr("runner._run", fail)
    with pytest.raises(RuntimeError, match="first stage"):
        run_config(suite)
    assert not (tmp_path / "b-results").exists()


def test_run_config_has_no_override_arguments(tmp_path):
    path = write_config(tmp_path / "greedy.json", greedy())
    with pytest.raises(TypeError):
        run_config(path, outputs="elsewhere")


def test_greedy_runs_from_json_and_is_repeatable(tmp_path):
    path = write_config(tmp_path / "greedy.json", greedy())
    run_config(path)
    result = tmp_path / "results/scaling_results.json"
    first = result.read_bytes()
    run_config(path)
    assert result.read_bytes() == first
    assert not (tmp_path / "results/runs").exists()


def test_shipped_configs_preflight_and_the_rerun_chains_its_stages():
    configs = Path(__file__).resolve().parents[2] / "configs"
    for path in configs.rglob("*.json"):
        _execution_plan(path)
    rerun = _execution_plan(configs / "rerun.json")
    assert [cfg["kind"] for cfg in rerun] == ["greedy", "sae", "threshold_pilot", "analysis"]
    greedy, sae, pilot, analysis = rerun
    assert analysis["sources"]["records"] == str(Path(sae["outputs"]) / "records.csv")
    assert analysis["sources"]["greedy"] == str(Path(greedy["outputs"]) / "scaling_results.json")
    assert len({Path(cfg["outputs"]) for cfg in (greedy, sae, pilot, analysis)}) == 4
    assert all(Path(path).is_absolute() for path in analysis["sources"].values())
    assert analysis == _execution_plan(configs / "analysis.json")[0]


def test_analysis_dispatch_never_invokes_training(tmp_path, monkeypatch):
    configs = Path(__file__).resolve().parents[2] / "configs"
    analysis = _execution_plan(configs / "analysis.json")[0]
    analysis["outputs"] = str(tmp_path / "analysis")
    path = write_config(tmp_path / "analysis.json", analysis)
    calls = []
    monkeypatch.setattr("analysis.paper.run_analysis", lambda cfg, out: calls.append(out))
    monkeypatch.setattr("experiments.sae.run_sae_sweep", lambda *_: pytest.fail("training"))
    run_config(path)
    assert calls == [str(tmp_path / "analysis")]


def test_analysis_sweep_preflight_runs_before_earlier_stages(tmp_path, monkeypatch):
    configs = Path(__file__).resolve().parents[2] / "configs"
    analysis = _execution_plan(configs / "analysis.json")[0]
    analysis["outputs"] = str(tmp_path / "analysis")
    analysis["sweep"] = str(tmp_path / "missing.json")
    write_config(tmp_path / "analysis.json", analysis)
    write_config(tmp_path / "greedy.json", greedy())
    suite = write_config(
        tmp_path / "suite.json", {"kind": "suite", "configs": ["greedy.json", "analysis.json"]}
    )
    monkeypatch.setattr("runner._run", lambda *_: pytest.fail("started before preflight"))
    with pytest.raises(ConfigError, match="missing.json"):
        run_config(suite)
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "analysis").exists()


def test_analysis_cannot_write_over_its_sources(tmp_path):
    configs = Path(__file__).resolve().parents[2] / "configs"
    cfg = _execution_plan(configs / "analysis.json")[0]
    cfg["sources"]["records"] = str(tmp_path / "inputs/records.csv")
    cfg["outputs"] = str(tmp_path / "inputs")
    path = write_config(tmp_path / "bad.json", cfg)
    with pytest.raises(ConfigError, match="separate"):
        run_config(path)
    assert not (tmp_path / "inputs").exists()
