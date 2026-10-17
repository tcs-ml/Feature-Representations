"""The paper's greedy experiment: representation length against d* log2(1 / eps) on [N]^d."""

from __future__ import annotations

from pathlib import Path

from solvers.greedy import greedy_product_exact, stopping_bits
from utils.io import save_json


def run_greedy(cfg: dict, outputs_root: str) -> dict:
    """Run every configured grid once and save its complete deterministic trace.

    Each grid is refined down to the strictest target it supports (eps >= 1 / N),
    and the first state meeting every target is read off the trace.
    """
    exp = cfg["experiment"]
    epsilon_grid = exp["epsilon_grid"]
    runs = []
    for d, N in exp["grid_sizes"]:
        targets = [eps for eps in epsilon_grid if eps >= 1.0 / N]
        if len(targets) < 2:
            raise ValueError(f"grid ({d}, {N}) needs at least two epsilons at or above 1/N")
        result = greedy_product_exact(d, N, min(targets))
        rows = []
        for eps in targets:
            t, bits = stopping_bits(result, eps)
            rows.append({"eps": eps, "t": t, "bits": bits})
        runs.append(
            {
                "d": d,
                "N": N,
                "rows": rows,
                "selected": result.selected,
                "history": result.history,
            }
        )
    payload = {"epsilon_grid": epsilon_grid, "runs": runs}
    save_json(Path(outputs_root) / "scaling_results.json", payload)
    return payload
