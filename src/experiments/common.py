"""Shared experiment plumbing: concept enumeration, records, frozen-code evaluation.

Every SAE run produces one immutable summary record with exactly the
RECORD_SCHEMA fields. `max_task_error` is the paper's worst-case error (the
largest error over the tasks of the family) on the test split,
`val_max_task_error` the same on the validation split, and the `_ucb` fields
their one-sided binomial bounds.
"""

from __future__ import annotations

import json

import numpy as np

from concepts import build_concept
from concepts.matrix import LabelMatrixClass
from data.atoms import AtomDataset, AtomObservationConfig, generate_atom_dataset
from evaluation.confidence import worst_task_bound
from models.probes import lookup_task_errors
from utils.seeds import child_seed

# The retired nuisance axis is a frozen component of every run identity
# (run_cfg, run_id, and the data/model child seeds). It is always 0.
NUISANCE_DIM = 0

RECORD_SCHEMA = [
    "run_id",
    "commit_sha",
    "config_hash",
    "seed",
    "concept_family",
    "concept_parameters",
    "dual_vc_dim",
    "num_inputs",
    "num_tasks",
    "dictionary_width",
    "active_k",
    "quantization_bits",
    "declared_capacity_bits",  # the paper's representation bit budget B_bits(W, k, q)
    "max_task_error",
    "max_task_error_ucb",
    "val_max_task_error",
    "val_max_task_error_ucb",
    "reconstruction_error",
    "dead_feature_count",
    "epochs_run",
    "status",
    "wall_clock_seconds",
    "device",
    "device_name",
    "cuda_version",
    "gpu_total_memory_gb",
    "environment",
]


def finalize_record(record: dict) -> dict:
    """Order the record by RECORD_SCHEMA and reject missing or unknown fields."""
    if set(record) != set(RECORD_SCHEMA):
        missing = sorted(set(RECORD_SCHEMA) - set(record))
        unknown = sorted(set(record) - set(RECORD_SCHEMA))
        raise ValueError(
            f"record fields differ from the schema: missing={missing} unknown={unknown}"
        )
    return {key: record[key] for key in RECORD_SCHEMA}


def concept_fields(concept: LabelMatrixClass) -> dict:
    meta = concept.metadata()
    return {
        "concept_family": meta.extra["family"],
        "concept_parameters": json.dumps(meta.extra, sort_keys=True, default=str),
        "dual_vc_dim": meta.dual_vc_dim,
        "num_inputs": meta.num_inputs,
        "num_tasks": meta.num_tasks,
    }


def enumerate_concepts(concept_cfg: dict) -> list[LabelMatrixClass]:
    """Expand one family block or a `families` list into concrete class instances."""
    out: list[LabelMatrixClass] = []
    for block in concept_cfg.get("families", [concept_cfg]):
        family = block["family"]
        sizes = block["d_values"] if family == "parity" else block["c_values"]
        out.extend(build_concept(family, size) for size in sizes)
    return out


def build_dataset(obs_cfg: dict, concept: LabelMatrixClass, seed: int) -> AtomDataset:
    cfg = AtomObservationConfig(
        num_atoms=concept.metadata().num_inputs,
        signal_dim=obs_cfg["signal_dim"],
        observed_dim=obs_cfg["observed_dim"],
        samples_per_atom=obs_cfg["samples_per_atom"],
        signal_noise_std=obs_cfg["signal_noise_std"],
        mixing_condition_number=obs_cfg["mixing_condition_number"],
        seed=child_seed(seed, "data", concept.name, NUISANCE_DIM),
    )
    return generate_atom_dataset(cfg, atom_probs=concept.input_probabilities())


def evaluate_code_representation(
    keys: dict[str, np.ndarray],
    dataset: AtomDataset,
    concept: LabelMatrixClass,
    confidence: float,
) -> dict:
    """Score a frozen discrete code (keys per split) with the train-fitted lookup decoder.

    The validation fields drive configuration selection; the test fields are
    the one-shot held-out report. `max_task_error` is the paper's worst-case
    error, the largest error over the tasks of the family, and each `_ucb`
    field is the one-sided binomial bound on that largest task error count
    (evaluation.confidence).
    """
    if set(keys) != {"train", "validation", "test"}:
        raise ValueError("lookup evaluation requires a discrete code on every split")
    y = concept.label_matrix()
    train = dataset.split("train")
    fields: dict = {}
    for name, prefix in (("test", ""), ("validation", "val_")):
        split = dataset.split(name)
        res = lookup_task_errors(keys["train"], train.atom_ids, keys[name], split.atom_ids, y)
        bound = worst_task_bound(res.per_task_errors, len(split.atom_ids), confidence)
        fields[f"{prefix}max_task_error"] = float(res.per_task_errors.max())
        fields[f"{prefix}max_task_error_ucb"] = bound.ucb
    return fields
