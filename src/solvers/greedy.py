"""Maximum-error greedy refinement by coordinate thresholds on the uniform grid [N]^d.

This is the variant of the paper's Algorithm 1 used in its first experiment:
at every step the threshold with the largest majority error is added. On the
uniform grid the error of a threshold depends only on the interval of its
coordinate that contains it, so the partition is tracked through the cut
boundaries of every coordinate and the number of cells is the product of the
interval counts. The representation length is log2(number of cells).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _validate_grid(d: int, N: int, eps: float) -> None:
    if d < 1 or N < 2:
        raise ValueError("need d >= 1 and N >= 2")
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")


@dataclass
class GreedyResult:
    d: int
    N: int
    eps: float
    history: list[dict] = field(default_factory=list)  # one entry per state, before each cut
    selected: list[tuple[int, int]] = field(default_factory=list)  # (coordinate, cut) in order

    @property
    def n_cells(self) -> int:
        return self.history[-1]["n_cells"]


def _record(result: GreedyResult, n_cells: int, max_err: float) -> None:
    result.history.append(
        {
            "t": len(result.selected),
            "n_cells": n_cells,
            "bits": float(np.log2(n_cells)),
            "max_err": max_err,
        }
    )


def greedy_product_exact(d: int, N: int, eps: float) -> GreedyResult:
    """Refine uniform [N]^d until every threshold has majority error at most eps.

    Inside the interval [l, r) of a coordinate, the cut at p = k + 1 has
    error min(p - l, r - p) / N. Ties use coordinate, then cut order.
    """
    _validate_grid(d, N, eps)
    boundaries = [np.array([0, N], dtype=np.int64) for _ in range(d)]
    result = GreedyResult(d=d, N=N, eps=eps)
    p = np.arange(1, N)
    # d * (N - 1) distinct cuts refine the grid fully, so the loop terminates.
    for _ in range(d * (N - 1) + 1):
        best_err, best_j, best_k = -1.0, -1, -1
        for j, b in enumerate(boundaries):
            idx = np.searchsorted(b, p, side="right")
            errs = np.minimum(p - b[idx - 1], b[idx] - p) / N
            k = int(np.argmax(errs))
            if errs[k] > best_err:
                best_err, best_j, best_k = float(errs[k]), j, k
        n_cells = math.prod(len(b) - 1 for b in boundaries)
        _record(result, n_cells, best_err)
        if best_err <= eps:
            break
        result.selected.append((best_j, best_k))
        boundaries[best_j] = np.unique(np.append(boundaries[best_j], best_k + 1))
    return result


def stopping_bits(result: GreedyResult, eps: float) -> tuple[int, float]:
    """Selected-threshold count and representation length at the first state meeting eps."""
    for rec in result.history:
        if rec["max_err"] <= eps:
            return rec["t"], rec["bits"]
    raise ValueError(f"run never reached eps={eps}")
