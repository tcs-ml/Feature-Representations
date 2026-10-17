"""Saved experiment outputs and experiment-grid validation for the analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config
from evaluation.confidence import worst_task_bound
from experiments.common import NUISANCE_DIM, concept_fields, enumerate_concepts
from experiments.sae import enumerate_variants, sae_run_config
from utils.io import config_hash, make_run_id

SOURCES = ("records", "greedy")
TEXT_COLUMNS = (
    "run_id",
    "commit_sha",
    "config_hash",
    "concept_family",
    "concept_parameters",
    "status",
    "device",
    "device_name",
    "cuda_version",
    "environment",
)
METRIC_COLUMNS = (
    "max_task_error",
    "val_max_task_error",
    "max_task_error_ucb",
    "val_max_task_error_ucb",
    "declared_capacity_bits",
)


def validate_analysis_config(cfg: dict) -> None:
    """Validate configuration structure without requiring the input files to exist."""
    allowed = {"kind", "outputs", "sweep", "sources", "epsilon_grid"}
    if cfg.keys() - allowed:
        raise ValueError(f"unsupported analysis keys: {sorted(cfg.keys() - allowed)}")
    for key in ("outputs", "sweep"):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError(f"analysis {key} must be a nonempty path string")
    sources = cfg.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(SOURCES):
        raise ValueError(f"analysis sources must be exactly {list(SOURCES)}")
    for name, path in sources.items():
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"sources.{name} must be a path string")
    eps = cfg.get("epsilon_grid")
    if (
        not isinstance(eps, list)
        or not eps
        or any(type(e) not in (int, float) or not 0 < e < 0.5 for e in eps)
    ):
        raise ValueError("epsilon_grid must contain error targets strictly between zero and 0.5")
    if len(eps) != len(set(eps)):
        raise ValueError("epsilon_grid must not contain duplicates")


def load_sweep(cfg: dict) -> dict:
    """The SAE experiment config whose run grid the analysis expects."""
    sweep = load_config(cfg["sweep"])
    if sweep["kind"] != "sae":
        raise ValueError(f"analysis sweep must be an SAE experiment: {cfg['sweep']}")
    return sweep


def expected_runs(cfg: dict) -> pd.DataFrame:
    """Rebuild the run identities of the declared SAE experiment."""
    exp_cfg = load_sweep(cfg)
    exp = exp_cfg["experiment"]
    samples = exp_cfg["observations"]["samples_per_atom"]
    confidence = float(exp_cfg["analysis"]["confidence_level"])
    rows = []
    for concept in enumerate_concepts(exp_cfg["concept"]):
        params = concept.metadata().extra
        fields = concept_fields(concept)
        identity = {
            **fields,
            "condition": f"{params['family']}-{params.get('C', params.get('d'))}",
            "n_validation": int(samples["validation"]) * fields["num_inputs"],
            "n_test": int(samples["test"]) * fields["num_inputs"],
            "confidence_level": confidence,
        }
        for seed in exp["seeds"]:
            for variant in enumerate_variants(exp_cfg):
                run_cfg = sae_run_config(exp["name"], concept.name, variant, seed, exp_cfg)
                digest = config_hash(run_cfg)
                prefix = f"{exp['name']}-{concept.name}-nu{NUISANCE_DIM}-{variant['label']}"
                rows.append(
                    {
                        **identity,
                        "seed": seed,
                        "run_id": make_run_id(prefix, digest, seed),
                        "config_hash": digest,
                        "config_json": json.dumps(run_cfg, sort_keys=True),
                    }
                )
    expected = pd.DataFrame(rows)
    if expected["run_id"].duplicated().any():
        raise ValueError("the SAE experiment config declares duplicate run identities")
    return expected.sort_values("run_id").reset_index(drop=True)


def _source(cfg: dict, name: str) -> Path:
    path = Path(cfg["sources"][name])
    if not path.is_file():
        raise FileNotFoundError(f"analysis input not found: {path}")
    return path


def read_records(path: Path) -> pd.DataFrame:
    """The sweep's records table, with identity and text columns kept as text."""
    return pd.read_csv(path, dtype=dict.fromkeys(TEXT_COLUMNS, str))


def load_analysis_inputs(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read the saved outputs, check them against the declared grid, and retain missing runs."""
    validate_analysis_config(cfg)
    expected = expected_runs(cfg)
    records = read_records(_source(cfg, "records"))
    if records.empty:
        records = pd.DataFrame(columns=["run_id", "config_hash"])
    records = records[records["run_id"].isin(expected["run_id"])].copy()
    if records["run_id"].duplicated().any():
        raise ValueError("analysis sources contain duplicate run IDs")
    hashes = records.set_index("run_id")["config_hash"]
    present = expected["run_id"].isin(hashes.index)
    if not np.array_equal(
        expected.loc[present, "config_hash"], expected.loc[present, "run_id"].map(hashes)
    ):
        raise ValueError("record config_hash disagrees with its declared experiment config")
    expected["present"] = present
    actual = records.set_index("run_id")
    for field in ("seed", "concept_family", "num_inputs", "num_tasks"):
        if field in actual:
            wanted = expected.loc[present, field]
            observed = expected.loc[present, "run_id"].map(actual[field])
            if not wanted.eq(observed).all():
                raise ValueError(f"record {field} disagrees with its declared experiment identity")
    _validate_records(records)
    records = attach_binomial_bounds(records, expected)
    greedy = json.loads(_source(cfg, "greedy").read_text())
    return records, expected, greedy


def attach_binomial_bounds(records: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """Recompute both upper bounds of every completed record from its worst-case error.

    The bound is the one-sided binomial bound on the largest task error count
    (evaluation.confidence) at the split size and confidence level declared by
    the run's experiment config. Recorded values are replaced, so every
    analysis uses one definition regardless of when a run was produced.
    """
    if records.empty or "status" not in records:
        return records
    sizes = expected.drop_duplicates("run_id").set_index("run_id")
    records = records.copy()
    completed = records["status"] == "completed"
    for field, size in (("max_task_error", "n_test"), ("val_max_task_error", "n_validation")):
        records.loc[completed, f"{field}_ucb"] = [
            worst_task_bound(
                np.array([rate]),
                int(sizes.at[run_id, size]),
                float(sizes.at[run_id, "confidence_level"]),
            ).ucb
            for run_id, rate in zip(
                records.loc[completed, "run_id"], records.loc[completed, field], strict=True
            )
        ]
    return records


def _validate_records(records: pd.DataFrame) -> None:
    """Completed records need finite metrics: error rates in [0, 1] and a nonnegative budget."""
    if records.empty or "status" not in records:
        return
    completed = records[records["status"] == "completed"]
    if completed.empty:
        return
    for field in METRIC_COLUMNS:
        if field not in completed:
            raise ValueError(f"completed records are missing {field}")
        values = pd.to_numeric(completed[field], errors="coerce")
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (field != "declared_capacity_bits" and (values > 1 + 1e-12).any())
        ):
            raise ValueError(f"invalid completed record field: {field}")
