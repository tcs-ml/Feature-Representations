"""Run records, config hashing, and immutable result storage.

Layout under an outputs root:

    outputs/
      runs/<run_id>/summary.json        one immutable summary record per run
      runs/<run_id>/config.json         resolved configuration for the run
      runs/<run_id>/...                 further artifacts of the experiment, such as a checkpoint

Runs are resumable and idempotent; a completed run is never overwritten by a
different configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class RunConfigMismatch(RuntimeError):
    """A run directory exists with a different resolved configuration."""


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(obj: Any) -> str:
    """12-hex stable hash of a JSON-serializable configuration."""
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()[:12]


def git_commit_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parent,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except OSError:
        pass
    return "unknown"


def environment_metadata() -> dict[str, str]:
    import numpy

    meta = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
    }
    try:
        import torch

        meta["torch"] = torch.__version__
    except ImportError:
        pass
    return meta


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def save_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str))


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def run_dir(outputs_root: str | Path, run_id: str) -> Path:
    return Path(outputs_root) / "runs" / run_id


def make_run_id(name: str, cfg_hash: str, seed: int | None = None) -> str:
    suffix = "" if seed is None else f"-s{seed}"
    return f"{name}-{cfg_hash}{suffix}"


def claim_run(outputs_root: str | Path, run_id: str, resolved_config: dict) -> Path | None:
    """Prepare a run directory; return None when the run is already complete.

    A directory holding a `summary.json` for the same config hash is terminal
    and skipped, whatever its status. A directory with a *different* config
    hash raises RunConfigMismatch instead of overwriting.
    """
    rdir = run_dir(outputs_root, run_id)
    cfg_h = config_hash(resolved_config)
    summary_path = rdir / "summary.json"
    if summary_path.exists():
        existing = load_json(summary_path)
        if existing.get("config_hash") != cfg_h:
            raise RunConfigMismatch(
                f"run {run_id} is complete with config_hash={existing.get('config_hash')}, "
                f"refusing to overwrite with config_hash={cfg_h}"
            )
        return None
    ensure_dir(rdir)
    save_json(rdir / "config.json", resolved_config)
    return rdir


def base_record(resolved_config: dict, run_id: str, seed: int | None) -> dict:
    """Shared provenance fields required on every summary record."""
    return {
        "run_id": run_id,
        "commit_sha": git_commit_sha(),
        "config_hash": config_hash(resolved_config),
        "seed": seed,
        "environment": environment_metadata(),
    }


def write_summary(rdir: Path, record: dict) -> None:
    if "config_hash" not in record or "run_id" not in record:
        raise ValueError("summary records must carry run_id and config_hash")
    save_json(rdir / "summary.json", record)
