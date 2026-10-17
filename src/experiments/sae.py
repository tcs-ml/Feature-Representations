"""Reconstruction-trained top-k SAE sweeps with frozen-code task evaluation.

Task labels never enter training here. After training, the encoder is frozen,
the sparse code is post-quantized for exact capacity accounting, and the
lookup decoder evaluates every task on the discrete code. The sweep's saved
result is `records.csv`, one row per run with the RECORD_SCHEMA fields.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from concepts.matrix import LabelMatrixClass
from data.atoms import AtomDataset
from experiments.common import (
    NUISANCE_DIM,
    RECORD_SCHEMA,
    build_dataset,
    concept_fields,
    enumerate_concepts,
    evaluate_code_representation,
    finalize_record,
)
from models.sae import TopKSAE, bit_budget, quantize_sparse_codes
from training.reconstruction import ReconstructionTrainConfig, train_reconstruction
from utils import io
from utils.device import detect_device, device_metadata
from utils.seeds import child_seed

SPLITS = ("train", "validation", "test")


def _recon_config(training_cfg: dict, seed: int, device: str) -> ReconstructionTrainConfig:
    return ReconstructionTrainConfig(
        epochs=training_cfg["epochs"],
        batch_size=training_cfg["batch_size"],
        lr=training_cfg["lr"],
        eval_every=training_cfg.get("eval_every", 10),
        patience=training_cfg.get("patience", 10),
        resample_dead=training_cfg.get("resample_dead", True),
        seed=seed,
        device=device,
    )


@torch.no_grad()
def _test_mse(model: TopKSAE, x: np.ndarray) -> float:
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(model.w_enc.device)
    _, recon = model(xt)
    return float(((recon - xt) ** 2).mean())


def sae_run_config(name: str, concept_name: str, variant: dict, seed: int, cfg: dict) -> dict:
    """The persisted run identity; every key is part of the config hash."""
    return {
        "experiment": name,
        "concept": concept_name,
        "nuisance_dim": NUISANCE_DIM,
        "variant": variant,
        "seed": seed,
        "observations": cfg["observations"],
        "training": cfg["training"],
        "probe": cfg["probe"],
    }


def run_one_sae_variant(
    name: str,
    concept: LabelMatrixClass,
    dataset: AtomDataset,
    variant: dict,
    seed: int,
    cfg: dict,
    outputs_root: str,
) -> dict:
    run_cfg = sae_run_config(name, concept.name, variant, seed, cfg)
    vname = variant["label"]
    run_id = io.make_run_id(
        f"{name}-{concept.name}-nu{NUISANCE_DIM}-{vname}", io.config_hash(run_cfg), seed
    )
    rdir = io.claim_run(outputs_root, run_id, run_cfg)
    if rdir is None:
        return io.load_json(io.run_dir(outputs_root, run_id) / "summary.json")

    start = time.perf_counter()
    model_seed = child_seed(seed, "model", concept.name, NUISANCE_DIM, vname)
    torch.manual_seed(model_seed)  # weight init must not depend on process history
    model = TopKSAE(
        in_dim=dataset.input_dim,
        dictionary_width=variant["dictionary_width"],
        active_k=variant["active_k"],
    )
    device = detect_device()
    train_result = train_reconstruction(
        model,
        dataset.split("train").x,
        dataset.split("validation").x,
        _recon_config(cfg["training"], model_seed, str(device)),
    )
    test_mse = _test_mse(model, dataset.split("test").x)

    # The quantization rule is fixed from train data only; the key layout is
    # fixed by the encoding rule (k slots) so all splits share it.
    value_bits = variant["value_bits"]
    k_slots = variant["active_k"]
    codes = {s: model.sparse_codes(dataset.split(s).x) for s in SPLITS}
    positive = codes["train"][codes["train"] > 0]
    vmax = float(np.quantile(positive, 0.999)) if positive.size else 1.0
    keys = {s: quantize_sparse_codes(z, value_bits, vmax, k_slots) for s, z in codes.items()}
    fields = evaluate_code_representation(
        keys, dataset, concept, confidence=cfg["analysis"]["confidence_level"]
    )
    # declared_capacity_bits is the paper's representation bit budget
    # B_bits(W, k, q): at most k of W features, q-bit nonzero values.
    declared_bits = float(bit_budget(variant["dictionary_width"], k_slots, value_bits))

    record = io.base_record(run_cfg, run_id, seed)
    record.update(concept_fields(concept))
    record.update(fields)
    record.update(
        declared_capacity_bits=declared_bits,
        dictionary_width=variant["dictionary_width"],
        active_k=variant["active_k"],
        quantization_bits=value_bits,
        reconstruction_error=test_mse,
        dead_feature_count=train_result.dead_feature_count,
        status=train_result.status,
        epochs_run=train_result.epochs_run,
        wall_clock_seconds=time.perf_counter() - start,
        **device_metadata(device),
    )
    io.save_json(rdir / "training_history.json", train_result.history)
    record = finalize_record(record)
    io.write_summary(rdir, record)
    logger.info(
        f"[sae] {concept.name} {vname} seed={seed}: "
        f"mse={test_mse:.5f} max_err={record['max_task_error']} ({record['status']})"
    )
    return record


def enumerate_variants(cfg: dict) -> list[dict]:
    """One variant per (value bits, width, active k); the dict is part of the run identity."""
    model_cfg = cfg["model"]
    return [
        {
            "type": "topk_sae",
            "dictionary_width": width,
            "active_k": k,
            "value_bits": q,
            "tied": False,  # frozen identity component of the historical grid
            "label": f"sae-K{width}-k{k}-q{q}",
        }
        for q in model_cfg["value_bits_grid"]
        for width in model_cfg["dictionary_widths"]
        for k in model_cfg["active_ks"]
        if k <= width
    ]


def write_records(records: list[dict], outputs_root: str) -> pd.DataFrame:
    """The sweep's saved result: one row per run in run-id order, environment as JSON text."""
    table = pd.DataFrame(records)[RECORD_SCHEMA].sort_values("run_id").reset_index(drop=True)
    table["environment"] = table["environment"].map(lambda v: json.dumps(v, sort_keys=True))
    table.to_csv(Path(outputs_root) / "records.csv", index=False)
    return table


def run_sae_sweep(cfg: dict, outputs_root: str) -> list[dict]:
    exp = cfg["experiment"]
    logger.info(f"[sae] device: {device_metadata(detect_device())}")
    variants = enumerate_variants(cfg)
    records: list[dict] = []
    for concept in enumerate_concepts(cfg["concept"]):
        for seed in exp["seeds"]:
            dataset = build_dataset(cfg["observations"], concept, seed)
            for variant in variants:
                records.append(
                    run_one_sae_variant(
                        exp["name"], concept, dataset, variant, seed, cfg, outputs_root
                    )
                )
    write_records(records, outputs_root)
    return records
