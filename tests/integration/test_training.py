"""Fast end-to-end training smoke tests (CPU, seconds)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.data import read_records
from config import validate_config
from experiments import sae
from models.sae import TopKSAE
from training.reconstruction import ReconstructionTrainConfig, train_reconstruction


def test_reconstruction_trainer_on_sae():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 10)).astype(np.float32)
    model = TopKSAE(in_dim=10, dictionary_width=24, active_k=4)
    cfg = ReconstructionTrainConfig(epochs=25, batch_size=128, eval_every=5, patience=5, seed=0)
    result = train_reconstruction(model, x[:300], x[300:], cfg)
    assert result.status == "completed"
    assert np.isfinite(result.best_val_mse)
    assert result.history
    assert any("dead_features" in rec for rec in result.history)


def test_sae_sweep_saves_each_variant_and_resumes_without_retraining(tmp_path, monkeypatch):
    cfg = {
        "kind": "sae",
        "outputs": "unused",
        "experiment": {"name": "identity-smoke", "seeds": [0]},
        "concept": {"family": "powerset", "c_values": [2]},
        "observations": {
            "signal_dim": 4,
            "observed_dim": 8,
            "samples_per_atom": {"train": 20, "validation": 10, "test": 10},
            "signal_noise_std": 0.05,
            "mixing_condition_number": 3.0,
        },
        "model": {
            "type": "topk_sae",
            "dictionary_widths": [4],
            "active_ks": [1],
            "value_bits_grid": [1, 2],
        },
        "training": {"epochs": 2, "batch_size": 16, "lr": 1e-3, "eval_every": 1},
        "probe": {"types": ["lookup"]},
        "analysis": {"confidence_level": 0.95},
    }
    validate_config(cfg)
    trainer = sae.train_reconstruction
    training_seeds = []

    def record_training(*args, **kwargs):
        training_seeds.append(args[-1].seed)
        return trainer(*args, **kwargs)

    monkeypatch.setattr(sae, "train_reconstruction", record_training)
    records = sae.run_sae_sweep(cfg, str(tmp_path))
    assert len(records) == 2
    assert len(set(training_seeds)) == 2  # q variants must also train separately
    assert len({r["run_id"] for r in records}) == 2
    assert all(r["val_max_task_error_ucb"] is not None for r in records)
    assert all(r["declared_capacity_bits"] >= 1 for r in records)
    assert not list((tmp_path / "runs").glob("*/*.pt"))
    table = read_records(tmp_path / "records.csv")
    assert list(table["run_id"]) == sorted(r["run_id"] for r in records)
    assert list(table["declared_capacity_bits"]) == [
        r["declared_capacity_bits"] for r in sorted(records, key=lambda r: r["run_id"])
    ]
    assert all(isinstance(json.loads(value), dict) for value in table["environment"])
    saved = (tmp_path / "records.csv").read_bytes()

    def unexpected_training(*args, **kwargs):
        raise AssertionError("a finished configuration must resume without training")

    monkeypatch.setattr(sae, "train_reconstruction", unexpected_training)
    assert sae.run_sae_sweep(cfg, str(tmp_path)) == records
    assert (tmp_path / "records.csv").read_bytes() == saved
    pd.testing.assert_frame_equal(read_records(tmp_path / "records.csv"), table)
