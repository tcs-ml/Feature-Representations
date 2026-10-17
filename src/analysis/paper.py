"""Reproduce the data behind the paper's figures from explicitly saved inputs."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from analysis.data import load_analysis_inputs
from analysis.export import capacity_tables, greedy_tables, seed_tables, write_tables
from analysis.prototype import prototype_certification, prototype_table
from analysis.selection import aggregate_selected, attach_references, select_thresholds, target_rise


def run_analysis(cfg: dict, outputs_root: str | Path) -> dict[str, pd.DataFrame]:
    """Load, audit, select, and aggregate the saved runs; write the paper's data files.

    `aggregate/` holds the intermediate tables (one row per seed outcome, per
    setting, per instance, and the prototype decoder's errors); `paper_data/`
    holds the tables the figures read. Both folders hold only what this run
    writes.
    """
    root = Path(outputs_root).resolve()
    for source in cfg["sources"].values():
        path = Path(source).resolve()
        if root == path or root in path.parents:
            raise ValueError("analysis outputs must be separate from source inputs")
    records, expected, greedy = load_analysis_inputs(cfg)
    outcomes = select_thresholds(records, expected, cfg["epsilon_grid"])
    conditions = attach_references(aggregate_selected(outcomes))
    prototype = prototype_table(cfg)
    tables = {
        "seed_outcomes": outcomes,
        "conditions": conditions,
        "target_rise": target_rise(conditions),
        "prototype_decoder": prototype,
        "prototype_certification": prototype_certification(prototype, cfg["epsilon_grid"]),
    }
    for stale in ("aggregate", "paper_data"):
        shutil.rmtree(root / stale, ignore_errors=True)
    write_tables(tables, root / "aggregate")
    write_tables(
        {
            **greedy_tables(greedy),
            **capacity_tables(conditions),
            **seed_tables(outcomes, conditions),
        },
        root / "paper_data",
    )
    return tables
