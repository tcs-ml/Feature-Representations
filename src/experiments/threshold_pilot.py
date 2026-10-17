"""Accuracy against bit budget for reconstruction-trained SAEs on the threshold product F_{r,N}.

For every dimension r and initialization seed, one top-k SAE with W = 2r
features and k = r active is trained for reconstruction on noiseless
normalized coordinates, with the checkpoint chosen by reconstruction
validation MSE. Each trained model is then quantized at every value precision
q, which sets its representation bit budget B_bits(W, k, q). The quantized
code is decoded back to coordinates, rounded to the grid, and the worst error
over all r (N - 1) threshold tasks is measured on the test split. Task labels
enter nothing but this final evaluation.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from data.thresholds import ThresholdDataset, ThresholdProductConfig, generate_threshold_dataset
from evaluation.thresholds import (
    equal_width_bin_worst_error,
    round_to_grid,
    worst_threshold_errors,
)
from models.sae import TopKSAE, bit_budget, ceil_log2, quantized_activations
from training.reconstruction import ReconstructionTrainConfig, train_reconstruction
from utils import io
from utils.device import detect_device, device_metadata
from utils.seeds import child_seed


def pilot_run_config(cfg: dict, dimension: int, seed: int) -> dict:
    """The persisted identity of one trained model; every key is part of the hash."""
    return {
        "experiment": cfg["experiment"]["name"],
        "family": {"grid_size": cfg["family"]["grid_size"], "dimension": dimension},
        "samples": cfg["samples"],
        "data_seed": cfg["experiment"]["data_seed"],
        "model": cfg["model"],
        "training": cfg["training"],
        "seed": seed,
    }


def model_shape(cfg: dict, dimension: int) -> tuple[int, int]:
    """(W, k) of the autoencoder trained on `dimension` coordinates."""
    model = cfg["model"]
    return model["width_per_dimension"] * dimension, model["active_per_dimension"] * dimension


def build_pilot_dataset(cfg: dict, dimension: int) -> ThresholdDataset:
    return generate_threshold_dataset(
        ThresholdProductConfig(
            grid_size=cfg["family"]["grid_size"],
            dimension=dimension,
            samples=cfg["samples"],
            seed=cfg["experiment"]["data_seed"],
        )
    )


def train_pilot_model(
    cfg: dict, dataset: ThresholdDataset, dimension: int, seed: int, outputs_root: str
) -> tuple[TopKSAE, dict]:
    """Train (or resume from disk) the model of one dimension and seed; returns it on the CPU."""
    run_cfg = pilot_run_config(cfg, dimension, seed)
    run_id = io.make_run_id(
        f"{cfg['experiment']['name']}-r{dimension}", io.config_hash(run_cfg), seed
    )
    width, active = model_shape(cfg, dimension)
    model = TopKSAE(in_dim=dimension, dictionary_width=width, active_k=active)
    rdir = io.claim_run(outputs_root, run_id, run_cfg)
    if rdir is None:
        rdir = io.run_dir(outputs_root, run_id)
        model.load_state_dict(torch.load(rdir / "model.pt", map_location="cpu"))
        model.eval()
        return model, io.load_json(rdir / "summary.json")
    start = time.perf_counter()
    model_seed = child_seed(seed, "model", "threshold_pilot", dimension)
    torch.manual_seed(model_seed)
    model = TopKSAE(in_dim=dimension, dictionary_width=width, active_k=active)
    device = detect_device()
    training = cfg["training"]
    result = train_reconstruction(
        model,
        dataset.split("train").features,
        dataset.split("reconstruction_validation").features,
        ReconstructionTrainConfig(
            epochs=training["epochs"],
            batch_size=training["batch_size"],
            lr=training["lr"],
            eval_every=training.get("eval_every", 10),
            patience=training.get("patience", 10),
            resample_dead=training.get("resample_dead", True),
            seed=model_seed,
            device=str(device),
        ),
    )
    model.cpu().eval()
    torch.save(model.state_dict(), rdir / "model.pt")
    io.save_json(rdir / "training_history.json", result.history)
    record = {
        **io.base_record(run_cfg, run_id, seed),
        "dimension": dimension,
        "dictionary_width": width,
        "active_k": active,
        "status": result.status,
        "best_reconstruction_validation_mse": result.best_val_mse,
        "epochs_run": result.epochs_run,
        "dead_feature_count": result.dead_feature_count,
        "wall_clock_seconds": time.perf_counter() - start,
        **device_metadata(device),
    }
    io.write_summary(rdir, record)
    logger.info(
        f"[pilot] r={dimension} seed={seed}: val_mse={result.best_val_mse:.3e} ({result.status})"
    )
    return model, record


@torch.no_grad()
def _decode(model: TopKSAE, z: np.ndarray) -> np.ndarray:
    zt = torch.from_numpy(np.asarray(z, dtype=np.float32))
    return (zt @ model.w_dec + model.b_dec).numpy()


def representation_points(
    cfg: dict, model: TopKSAE, record: dict, dataset: ThresholdDataset, dimension: int, seed: int
) -> list[dict]:
    """Worst test threshold error of one trained model at every value precision.

    The quantization range is the 99.9th percentile of the positive training
    activations, as in the finite-family sweep; the quantized code is decoded
    by the model's own decoder and rounded to the grid.
    """
    grid_size = cfg["family"]["grid_size"]
    width, active = model_shape(cfg, dimension)
    train_codes = model.sparse_codes(dataset.split("train").features)
    positive = train_codes[train_codes > 0]
    vmax = float(np.quantile(positive, 0.999)) if positive.size else 1.0
    test = dataset.split("test")
    test_codes = model.sparse_codes(test.features)
    rows = []
    for value_bits in cfg["model"]["value_bits_grid"]:
        decoded = round_to_grid(
            _decode(model, quantized_activations(test_codes, value_bits, vmax)), grid_size
        )
        errors = worst_threshold_errors(test.x, decoded, grid_size)
        budget = bit_budget(width, active, value_bits)
        rows.append(
            {
                "run_id": record["run_id"],
                "dimension": dimension,
                "seed": seed,
                "value_bits": value_bits,
                "dictionary_width": width,
                "active_k": active,
                "bit_budget": budget,
                "bits_per_dimension": budget / dimension,
                "n": int(len(test.x)),
                "worst_errors": errors,
                "worst_error": errors / len(test.x),
            }
        )
    return rows


def equal_width_table(grid_size: int) -> pd.DataFrame:
    """Exact worst threshold error of equal-width quantization at every bit depth below log2(N).

    These are the crosses of the paper's figure; on a power-of-two grid the
    exact error equals the closed form 2^(-bits - 1) as long as every bin
    holds at least two grid points.
    """
    return pd.DataFrame(
        [
            {
                "bits_per_coordinate": bits,
                "worst_error": equal_width_bin_worst_error(grid_size, bits),
                "closed_form_error": 2.0 ** (-bits - 1),
            }
            for bits in range(1, ceil_log2(grid_size))
        ]
    )


def run_threshold_pilot(cfg: dict, outputs_root: str) -> dict[str, pd.DataFrame]:
    root = Path(outputs_root)
    device = detect_device()
    models: list[dict] = []
    points: list[dict] = []
    for dimension in cfg["family"]["dimensions"]:
        dataset = build_pilot_dataset(cfg, dimension)
        for seed in cfg["experiment"]["initialization_seeds"]:
            model, record = train_pilot_model(cfg, dataset, dimension, seed, outputs_root)
            models.append(record)
            points.extend(representation_points(cfg, model, record, dataset, dimension, seed))
    tables = {
        "models": pd.DataFrame(models),
        "points": pd.DataFrame(points),
        "equal_width_quantization": equal_width_table(cfg["family"]["grid_size"]),
    }
    aggregate = root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(aggregate / f"{name}.csv", index=False)
    io.save_json(
        root / "summary.json",
        {
            "trained_models": len(models),
            "quantized_representations": len(points),
            "commit_sha": io.git_commit_sha(),
            "environment": io.environment_metadata(),
            "device": device_metadata(device),
        },
    )
    write_paper_data(root / "paper_data", tables["points"])
    return tables


def write_paper_data(paper: Path, points: pd.DataFrame) -> None:
    """Write the table the paper's accuracy-bit figure reads: one row per model and precision.

    `series` is `reconstruction-r{dimension}-test`, the key of the figure's row filter.
    """
    paper.mkdir(parents=True, exist_ok=True)
    series = "reconstruction-r" + points["dimension"].astype(str) + "-test"
    points.assign(series=series).to_csv(paper / "threshold_pilot_points.csv", index=False)
