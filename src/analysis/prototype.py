"""Known-prototype atom decoder: a B0-bit pipeline that sees the generating prototypes.

For every SAE instance and seed, the observation splits are regenerated from
the recorded seed, each observation is decoded to the nearest clean prototype
image in observation space, and every task is predicted from the decoded atom.
Transmitting the decoded atom index costs B0 = ceil(log2 n_atoms) bits. The
decoder uses the prototypes, which the autoencoders never see, so it is a
diagnostic: when its worst-task error meets a target, observation noise alone
does not force more than B0 bits in that setting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.data import load_sweep
from data.atoms import AtomDataset
from evaluation.confidence import worst_task_bound
from experiments.common import build_dataset, concept_fields, enumerate_concepts

SPLITS = ("validation", "test")


def decode_nearest_prototype(x: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """Index of the nearest clean prototype image per observation; ties go to the lowest index."""
    x = np.asarray(x, dtype=np.float64)
    clean = np.asarray(clean, dtype=np.float64)
    distances = (x**2).sum(axis=1, keepdims=True) - 2.0 * x @ clean.T + (clean**2).sum(axis=1)
    return distances.argmin(axis=1)


def prototype_rows(dataset: AtomDataset, label_matrix: np.ndarray, confidence: float) -> list[dict]:
    """Worst-task error of the nearest-prototype decoder on the validation and test splits."""
    y = np.asarray(label_matrix, dtype=bool)
    clean = dataset.prototypes @ dataset.mixing.T
    rows = []
    for split_name in SPLITS:
        split = dataset.split(split_name)
        decoded = decode_nearest_prototype(split.x, clean)
        per_task = np.empty(y.shape[0])
        for start in range(0, y.shape[0], 4096):  # chunk tasks for large families
            block = y[start : start + 4096]
            per_task[start : start + 4096] = (block[:, decoded] != block[:, split.atom_ids]).mean(
                axis=1
            )
        bound = worst_task_bound(per_task, len(decoded), confidence)
        rows.append(
            {
                "split": split_name,
                "n": int(len(decoded)),
                "atom_error": float((decoded != split.atom_ids).mean()),
                "worst_task_error": bound.point,
                "worst_task_errors": bound.errors,
                "worst_task_error_ucb": bound.ucb,
            }
        )
    return rows


def prototype_table(cfg: dict) -> pd.DataFrame:
    """One row per instance, seed, and split of the declared SAE experiment."""
    exp_cfg = load_sweep(cfg)
    confidence = float(exp_cfg["analysis"]["confidence_level"])
    rows = []
    for concept in enumerate_concepts(exp_cfg["concept"]):
        fields = concept_fields(concept)
        params = concept.metadata().extra
        condition = f"{params['family']}-{params.get('C', params.get('d'))}"
        for seed in exp_cfg["experiment"]["seeds"]:
            dataset = build_dataset(exp_cfg["observations"], concept, seed)
            for row in prototype_rows(dataset, concept.label_matrix(), confidence):
                rows.append(
                    {
                        "concept_family": fields["concept_family"],
                        "condition": condition,
                        "dual_vc_dim": fields["dual_vc_dim"],
                        "num_inputs": fields["num_inputs"],
                        "zero_error_bits": int(params["zero_error_bits"]),
                        "seed": int(seed),
                        **row,
                    }
                )
    return pd.DataFrame(rows)


def prototype_certification(table: pd.DataFrame, epsilon_grid: list[float]) -> pd.DataFrame:
    """Per setting: how many seeds the B0-bit prototype decoder certifies on each split."""
    rows = []
    for (family, condition), group in table.groupby(["concept_family", "condition"], sort=True):
        seeds = sorted(group["seed"].unique())
        for eps in sorted(epsilon_grid):
            row = {
                "concept_family": family,
                "condition": condition,
                "epsilon": float(eps),
                "zero_error_bits": int(group["zero_error_bits"].iloc[0]),
                "n_seeds": len(seeds),
            }
            for split_name in SPLITS:
                part = group[group["split"] == split_name]
                row[f"seeds_{split_name}_certified"] = int(
                    (part["worst_task_error_ucb"] <= eps + 1e-9).sum()
                )
                row[f"median_{split_name}_worst_task_error"] = float(
                    part["worst_task_error"].median()
                )
            rows.append(row)
    return pd.DataFrame(rows)
