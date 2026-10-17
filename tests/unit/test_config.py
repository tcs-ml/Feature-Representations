"""JSON validation and immutable scientific config payloads."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from config import ConfigError, load_config, validate_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def learned_cfg() -> dict:
    return load_config(CONFIGS / "experiment/sae.json")


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("experiment", "seeds", [0, 0]),
        ("experiment", "seeds", [True]),
        ("experiment", "epsilon_grid", [0.05]),
        ("experiment", "delta", 0.05),
        ("model", "value_bits_grid", []),
        ("model", "value_bits_grid", [0]),
        ("model", "value_bits_grid", [True]),
        ("model", "active_ks", [100]),
        ("model", "type", "autoencoder"),
        ("model", "tied", False),
        ("training", "epochs", 0),
        ("training", "lr", -1),
        ("training", "lr", float("inf")),
        ("training", "weight_decay", 0.0),
        ("training", "task_objective", "logsumexp"),
        ("analysis", "primary_task_metric", "max"),
        ("analysis", "confidence_level", 1),
        ("analysis", "n_bootstrap", 0),
        ("probe", "types", ["lookup", "mlp"]),
        ("probe", "types", ["linear"]),
        ("probe", "types", [{}]),
        ("probe", "linear_epochs", 10),
        ("observations", "samples_per_atom", {"train": 10, "test": 10}),
        ("observations", "observed_dim", 1),
        ("observations", "nuisance_dim_grid", [0]),
        ("observations", "nuisance_scale", 1.0),
        ("concept", "distribution", "uniform"),
    ],
)
def test_invalid_or_retired_scientific_config(section, key, value):
    cfg = learned_cfg()
    cfg.setdefault(section, {})[key] = value
    with pytest.raises(ConfigError):
        validate_config(cfg)


@pytest.mark.parametrize("key", ["save_models", "ae_baseline"])
def test_retired_top_level_keys_rejected(key):
    cfg = learned_cfg()
    cfg[key] = False
    with pytest.raises(ConfigError, match="unsupported"):
        validate_config(cfg)


@pytest.mark.parametrize("kind", ["bottleneck", "exact"])
def test_retired_experiment_kinds_rejected(kind):
    cfg = learned_cfg()
    cfg["kind"] = kind
    with pytest.raises(ConfigError, match="unsupported config kind"):
        validate_config(cfg)


@pytest.mark.parametrize("kind", ["real", "synthetic", "extract-activations", "concept", [], None])
def test_removed_or_unknown_kind_rejected(kind):
    cfg = learned_cfg()
    cfg["kind"] = kind
    with pytest.raises(ConfigError):
        validate_config(cfg)


@pytest.mark.parametrize("text", ["[]", "{", '{"kind":"sae","kind":"exact"}', '{"kind":NaN}'])
def test_bad_json_rejected_with_filename(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ConfigError, match="bad.json"):
        load_config(path)


def test_yaml_and_missing_files_rejected(tmp_path):
    with pytest.raises(ConfigError, match=".json"):
        load_config(tmp_path / "old.yaml")
    with pytest.raises(ConfigError, match="missing.json"):
        load_config(tmp_path / "missing.json")


def test_no_overrides_or_default_injection(tmp_path):
    cfg = learned_cfg()
    before = copy.deepcopy(cfg)
    validate_config(cfg)
    assert cfg == before
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(cfg))
    assert load_config(path) == before
    with pytest.raises(TypeError):
        load_config(path, overrides={"training": {"epochs": 1}})


def test_greedy_grids_are_explicit_and_validated():
    cfg = load_config(CONFIGS / "experiment/greedy.json")
    assert cfg["experiment"]["grid_sizes"] == [[1, 256], [2, 128], [3, 64], [4, 32]]
    cfg["experiment"]["grid_sizes"] = [[1, 2]]
    with pytest.raises(ConfigError, match="epsilon"):
        validate_config(cfg)
    cfg["experiment"]["grid_sizes"] = [[1, 8]]
    cfg["experiment"]["max_iters"] = 0
    with pytest.raises(ConfigError, match="unsupported"):
        validate_config(cfg)


def test_unknown_keys_rejected():
    cfg = learned_cfg()
    cfg["overrides"] = {"outputs": "elsewhere"}
    with pytest.raises(ConfigError, match="unsupported"):
        validate_config(cfg)


def test_single_greedy_epsilon_rejected_before_computation():
    cfg = load_config(CONFIGS / "experiment/greedy.json")
    cfg["experiment"]["epsilon_grid"] = [0.25]
    with pytest.raises(ConfigError, match="two epsilons"):
        validate_config(cfg)


@pytest.mark.parametrize(
    "concept",
    [
        {"family": "parity", "d_values": [13]},
        {"family": "powerset", "c_values": [21]},
        {"family": "transposed_powerset", "c_values": [15]},
        {"family": "parity", "d_values": [2], "c_values": [2]},
        {"family": "powerset", "families": [{"family": "parity", "d_values": [2]}]},
        {"families": [{"family": "powerset", "c_values": [2], "distribution": "uniform"}]},
    ],
)
def test_unsupported_concept_grid_rejected_during_preflight(concept):
    cfg = learned_cfg()
    cfg["concept"] = concept
    with pytest.raises(ConfigError):
        validate_config(cfg)
