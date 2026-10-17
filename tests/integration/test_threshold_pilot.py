"""A tiny end-to-end threshold-product experiment: outputs, resume, and the figure table."""

from __future__ import annotations

import json

import pandas as pd

from experiments import threshold_pilot
from runner import run_config

TINY = {
    "kind": "threshold_pilot",
    "outputs": "pilot",
    "experiment": {"name": "tiny_pilot", "initialization_seeds": [0], "data_seed": 11},
    "family": {"grid_size": 16, "dimensions": [1, 2]},
    "samples": {"train": 512, "reconstruction_validation": 64, "test": 256},
    "model": {
        "type": "topk_sae",
        "width_per_dimension": 2,
        "active_per_dimension": 1,
        "value_bits_grid": [1, 3],
    },
    "training": {"epochs": 3, "batch_size": 64, "lr": 0.001, "eval_every": 1},
}


def test_tiny_pilot_writes_every_table_and_resumes(tmp_path, monkeypatch):
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(TINY))
    trained = []
    trainer = threshold_pilot.train_reconstruction

    def record_training(*args, **kwargs):
        trained.append(args[-1].seed)
        return trainer(*args, **kwargs)

    monkeypatch.setattr(threshold_pilot, "train_reconstruction", record_training)
    run_config(path)
    root = tmp_path / "pilot"
    assert len(trained) == 2  # one model per dimension and seed
    assert len(list((root / "runs").glob("*/model.pt"))) == 2
    assert sorted(p.name for p in (root / "aggregate").iterdir()) == [
        "equal_width_quantization.csv",
        "models.csv",
        "points.csv",
    ]
    models = pd.read_csv(root / "aggregate/models.csv")
    assert list(models["dimension"]) == [1, 2]
    assert list(models["dictionary_width"]) == [2, 4] and list(models["active_k"]) == [1, 2]
    points = pd.read_csv(root / "aggregate/points.csv")
    assert len(points) == 2 * 2  # models x precisions
    assert set(points["value_bits"]) == {1, 3}
    assert (points["worst_error"] == points["worst_errors"] / 256).all()
    assert (points["bits_per_dimension"] == points["bit_budget"] / points["dimension"]).all()
    control = pd.read_csv(root / "aggregate/equal_width_quantization.csv")
    assert list(control["bits_per_coordinate"]) == [1, 2, 3]
    assert (control["worst_error"] == control["closed_form_error"]).all()
    summary = json.loads((root / "summary.json").read_text())
    assert summary["trained_models"] == 2 and summary["quantized_representations"] == 4
    assert {"commit_sha", "environment", "device"} <= set(summary)
    exported = pd.read_csv(root / "paper_data/threshold_pilot_points.csv")
    assert [p.name for p in (root / "paper_data").iterdir()] == ["threshold_pilot_points.csv"]
    assert list(exported["series"]) == [
        "reconstruction-r1-test",
        "reconstruction-r1-test",
        "reconstruction-r2-test",
        "reconstruction-r2-test",
    ]
    assert (exported["worst_error"].to_numpy() == points["worst_error"].to_numpy()).all()

    def unexpected_training(*args, **kwargs):
        raise AssertionError("a finished model must resume without training")

    monkeypatch.setattr(threshold_pilot, "train_reconstruction", unexpected_training)
    run_config(path)
    pd.testing.assert_frame_equal(points, pd.read_csv(root / "aggregate/points.csv"))
