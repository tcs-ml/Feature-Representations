"""Run-record store: resumability, idempotence, overwrite protection."""

from __future__ import annotations

import pytest

from utils import io
from utils.seeds import child_seed


def test_config_hash_stable_and_order_independent():
    a = io.config_hash({"x": 1, "y": [1, 2]})
    b = io.config_hash({"y": [1, 2], "x": 1})
    assert a == b
    assert a != io.config_hash({"x": 2, "y": [1, 2]})


def test_claim_run_skip_and_mismatch(tmp_path):
    cfg = {"model": "m", "bits": 2}
    rdir = io.claim_run(tmp_path, "run-a", cfg)
    assert rdir is not None
    record = io.base_record(cfg, "run-a", seed=0)
    record["status"] = "completed"
    io.write_summary(rdir, record)
    # same config: skipped
    assert io.claim_run(tmp_path, "run-a", cfg) is None
    # different config for a completed run: refuse to overwrite
    with pytest.raises(io.RunConfigMismatch):
        io.claim_run(tmp_path, "run-a", {"model": "m", "bits": 3})


def test_claim_reclaims_incomplete_run(tmp_path):
    cfg = {"a": 1}
    first = io.claim_run(tmp_path, "run-b", cfg)
    assert first is not None  # no summary written -> crashed mid-run
    second = io.claim_run(tmp_path, "run-b", cfg)
    assert second is not None


def test_summary_requires_provenance(tmp_path):
    rdir = io.claim_run(tmp_path, "run-c", {"a": 1})
    with pytest.raises(ValueError):
        io.write_summary(rdir, {"status": "completed"})


def test_child_seeds_distinct_and_stable():
    assert child_seed(0, "data", "x") == child_seed(0, "data", "x")
    assert child_seed(0, "data", "x") != child_seed(0, "model", "x")
    assert child_seed(0, "data", "x") != child_seed(1, "data", "x")
