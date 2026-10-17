"""The JSON-only experiment entry point; no command-line interface or overrides."""

from __future__ import annotations

import copy
from pathlib import Path

from config import ConfigError, load_config
from utils import io


def _absolute(path: str, directory: Path) -> str:
    return str((directory / path).resolve())


def _resolve_paths(cfg: dict, directory: Path) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["outputs"] = _absolute(cfg["outputs"], directory)
    if cfg["kind"] == "analysis":
        output = Path(cfg["outputs"])
        for name, path in cfg["sources"].items():
            cfg["sources"][name] = _absolute(path, directory)
            source = Path(cfg["sources"][name])
            if output == source or output in source.parents:
                raise ConfigError("analysis outputs must be separate from source inputs")
        cfg["sweep"] = _absolute(cfg["sweep"], directory)
        if load_config(cfg["sweep"])["kind"] != "sae":
            raise ConfigError(f"analysis sweep must be an SAE experiment: {cfg['sweep']}")
    return cfg


def _execution_plan(path: str | Path) -> list[dict]:
    """Read every config before execution, so an invalid suite does no partial work."""
    path = Path(path).resolve()
    cfg = load_config(path)
    if cfg["kind"] != "suite":
        return [_resolve_paths(cfg, path.parent)]
    plan = []
    seen: set[Path] = set()
    for relative in cfg["configs"]:
        child_path = (path.parent / relative).resolve()
        if child_path in seen:
            raise ConfigError(f"duplicate suite config: {child_path}")
        seen.add(child_path)
        child = load_config(child_path)
        if child["kind"] == "suite":
            raise ConfigError(f"nested suites are not supported: {child_path}")
        plan.append(_resolve_paths(child, child_path.parent))
    return plan


def _run(cfg: dict) -> None:
    kind, outputs = cfg["kind"], cfg["outputs"]
    if kind == "greedy":
        from experiments.greedy import run_greedy

        run_greedy(cfg, outputs)
    elif kind == "sae":
        from experiments.sae import run_sae_sweep

        run_sae_sweep(cfg, outputs)
    elif kind == "threshold_pilot":
        from experiments.threshold_pilot import run_threshold_pilot

        run_threshold_pilot(cfg, outputs)
    else:
        from analysis.paper import run_analysis

        run_analysis(cfg, outputs)


def run_config(path: str | Path) -> None:
    """Run a standalone JSON experiment or an ordered suite of JSON experiments.

    Relative paths are anchored at the containing config, not the working
    directory. Scientific settings cannot be overridden by Python arguments,
    flags, or environment variables. Existing completed run IDs are resumed.
    """
    plan = _execution_plan(path)
    for cfg in plan:
        # This audit snapshot is separate from the established per-run hash payload.
        snapshot = Path(cfg["outputs"]) / "configs" / f"{cfg['kind']}-{io.config_hash(cfg)}.json"
        io.save_json(snapshot, cfg)
        _run(cfg)
