"""The shipped threshold-product experiment is the paper's and its validation rejects drift."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from config import ConfigError, load_config, validate_config
from experiments.threshold_pilot import model_shape, pilot_run_config
from utils.io import config_hash

CONFIG = Path(__file__).resolve().parents[2] / "configs/experiment/threshold_pilot.json"


@pytest.fixture(scope="module")
def pilot() -> dict:
    return load_config(CONFIG)


def test_paper_sizes(pilot):
    # F_{r,1024} for r in {1, 2, 4, 8, 10}; W = 2r, k = r; q in 1..5; three seeds
    assert pilot["family"] == {"grid_size": 1024, "dimensions": [1, 2, 4, 8, 10]}
    assert pilot["samples"]["train"] == 65536 and pilot["samples"]["test"] == 65536
    models = len(pilot["family"]["dimensions"]) * len(pilot["experiment"]["initialization_seeds"])
    assert models == 15
    assert models * len(pilot["model"]["value_bits_grid"]) == 75
    assert pilot["model"]["value_bits_grid"] == [1, 2, 3, 4, 5]
    assert model_shape(pilot, 1) == (2, 1) and model_shape(pilot, 10) == (20, 10)


def test_run_identity_covers_data_model_and_training(pilot):
    base = config_hash(pilot_run_config(pilot, 2, 0))
    assert base != config_hash(pilot_run_config(pilot, 2, 1))
    assert base != config_hash(pilot_run_config(pilot, 4, 0))
    for section, key, value in [
        ("samples", "train", 1024),
        ("model", "value_bits_grid", [1]),
        ("training", "epochs", 1),
        ("experiment", "data_seed", 1),
    ]:
        cfg = copy.deepcopy(pilot)
        cfg[section][key] = value
        assert config_hash(pilot_run_config(cfg, 2, 0)) != base, (section, key)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("family", "grid_size", 1),
        ("family", "fit_dimensions", [1, 2]),
        ("family", "held_out_dimension", 4),
        ("model", "active_per_dimension", 3),
        ("model", "type", "dense"),
        ("model", "value_bits_grid", [0]),
        ("samples", "test", 0),
        ("samples", "task_validation", 100),
        ("experiment", "data_seed", -1),
        ("training", "task_objective", "ce"),
    ],
)
def test_invalid_pilot_configs_rejected(pilot, section, key, value):
    cfg = copy.deepcopy(pilot)
    cfg[section][key] = value
    with pytest.raises(ConfigError):
        validate_config(cfg)


@pytest.mark.parametrize("key", ["control", "analysis"])
def test_retired_pilot_sections_rejected(pilot, key):
    cfg = copy.deepcopy(pilot)
    cfg[key] = {}
    with pytest.raises(ConfigError, match="unsupported"):
        validate_config(cfg)
