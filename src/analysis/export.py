"""CSV tables behind the paper's figures."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def closed_form_bits(d: int, eps: float) -> int:
    """Equation (8) of the paper: d * ceil(log2(1 / (2 eps))) bits on a power-of-two grid."""
    return d * math.ceil(math.log2(1.0 / (2.0 * eps)))


def greedy_tables(payload: dict) -> dict[str, pd.DataFrame]:
    """One table per grid: representation length against d * log2(1 / eps).

    `predictor` and `bits` are the figure's axes; `closed_form_bits` is the
    length the paper's closed form predicts for the same target.
    """
    tables = {}
    for run in payload["runs"]:
        name = f"greedy_threshold_d{run['d']}"
        if name in tables:
            raise ValueError(f"two greedy grids share dimension {run['d']}")
        tables[name] = pd.DataFrame(
            [
                {
                    "predictor": run["d"] * float(np.log2(1 / row["eps"])),
                    "bits": row["bits"],
                    "closed_form_bits": closed_form_bits(run["d"], row["eps"]),
                    "eps": row["eps"],
                    "t": row["t"],
                    "N": run["N"],
                    "dimension": run["d"],
                }
                for row in run["rows"]
            ]
        )
    return tables


def capacity_tables(conditions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One table per family: median selected budget against log2(1 / eps) for every setting.

    Settings without a selected seed keep their row with an undefined median
    and `any_selected = 0`, so the figure can mark them explicitly; the
    dotted lossless reference is `zero_error_bits`.
    """
    table = pd.DataFrame(
        {
            "concept_family": conditions["concept_family"],
            "condition": conditions["condition"],
            "epsilon": conditions["epsilon"],
            "log2_inv_epsilon": np.log2(1.0 / conditions["epsilon"]),
            "dual_vc_dim": conditions["dual_vc_dim"],
            "zero_error_bits": conditions["zero_error_bits"],
            "median_bits": conditions["median_bits"],
            "n_seeds_selected": conditions["n_seeds_selected"],
            "n_seeds_expected": conditions["n_seeds_expected"],
            "any_selected": (conditions["n_seeds_selected"] > 0).astype(int),
        }
    ).sort_values(["concept_family", "condition", "epsilon"], ascending=[True, True, False])
    return {
        f"capacity_{family}": group.reset_index(drop=True)
        for family, group in table.groupby("concept_family", sort=True)
    }


def seed_tables(outcomes: pd.DataFrame, conditions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One table per family with every selected seed as its own row.

    Each row carries the selected budget, whether the selection was certified
    on test (`heldout_pass`), and the shattered-set bound of its setting
    (`lower_bits`), so the figure draws one dot per seed.
    """
    reference = conditions[["condition", "epsilon", "lower_bits"]]
    rows = outcomes[outcomes["selected"].eq(True)].merge(reference, on=["condition", "epsilon"])
    tables = {}
    for family, group in rows.groupby("concept_family", sort=True):
        tables[f"seeds_{family}"] = (
            pd.DataFrame(
                {
                    "condition": group["condition"],
                    "epsilon": group["epsilon"],
                    "log2_inv_epsilon": np.log2(1.0 / group["epsilon"]),
                    "seed": group["seed"].astype(int),
                    "selected_bits": group["selected_bits"],
                    "heldout_pass": group["heldout_pass"].astype(int),
                    "lower_bits": group["lower_bits"],
                }
            )
            .sort_values(["condition", "epsilon", "seed"], ascending=[True, False, True])
            .reset_index(drop=True)
        )
    return tables


def write_tables(tables: dict[str, pd.DataFrame], directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(directory / f"{name}.csv", index=False, na_rep="nan")
