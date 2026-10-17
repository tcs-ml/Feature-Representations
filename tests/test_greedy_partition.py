"""The paper's greedy construction on [N]^d, checked against an explicit point-based solver."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from analysis.export import closed_form_bits
from experiments.greedy import run_greedy
from solvers.greedy import greedy_product_exact, stopping_bits

ROOT = Path(__file__).resolve().parents[1]


def make_grid(d: int, N: int) -> np.ndarray:
    """Enumerate {0, ..., N-1}^d as an array of shape (N^d, d)."""
    return np.indices((N,) * d).reshape(d, -1).T.astype(np.int64)


def feature_errors(points, weights, cell_ids, n_cells, N) -> np.ndarray:
    """Majority error of every threshold 1{x_j <= k} on the current cells, shape (d, N - 1)."""
    errs = np.empty((points.shape[1], N - 1))
    cell_w = np.bincount(cell_ids, weights=weights, minlength=n_cells)
    for j in range(points.shape[1]):
        hist = np.bincount(
            cell_ids * N + points[:, j], weights=weights, minlength=n_cells * N
        ).reshape(n_cells, N)
        cum = np.cumsum(hist, axis=1)[:, :-1]
        errs[j] = np.minimum(cum, cell_w[:, None] - cum).sum(axis=0)
    return errs


def greedy_partition(points: np.ndarray, N: int, eps: float) -> tuple[list, list]:
    """Reference maximum-error greedy on explicit uniform points: (selected cuts, history)."""
    n, d = points.shape
    weights = np.full(n, 1.0 / n)
    cell_ids = np.zeros(n, dtype=np.int64)
    n_cells, selected, history = 1, [], []
    for _ in range(d * (N - 1) + 1):
        errs = feature_errors(points, weights, cell_ids, n_cells, N)
        max_err = float(errs.max())
        history.append({"n_cells": n_cells, "max_err": max_err})
        if max_err <= eps:
            break
        j, k = np.unravel_index(int(np.argmax(errs)), errs.shape)
        selected.append((int(j), int(k)))
        _, cell_ids = np.unique(cell_ids * 2 + (points[:, j] <= k), return_inverse=True)
        n_cells = int(cell_ids.max()) + 1
    return selected, history


def brute_force_errors(points, weights, cell_ids, n_cells, N):
    n, d = points.shape
    errs = np.zeros((d, N - 1))
    for j in range(d):
        for k in range(N - 1):
            f = points[:, j] <= k
            for c in range(n_cells):
                in_c = cell_ids == c
                errs[j, k] += min(weights[in_c & ~f].sum(), weights[in_c & f].sum())
    return errs


def test_reference_feature_errors_match_brute_force():
    rng = np.random.default_rng(3)
    n, d, N, n_cells = 200, 3, 5, 7
    points = rng.integers(0, N, size=(n, d))
    weights = rng.random(n)
    weights /= weights.sum()
    cell_ids = rng.integers(0, n_cells, size=n)
    fast = feature_errors(points, weights, cell_ids, n_cells, N)
    assert np.allclose(fast, brute_force_errors(points, weights, cell_ids, n_cells, N))


def test_first_cut_is_median_and_error_halves():
    res = greedy_product_exact(1, 32, eps=1e-12)
    assert res.selected[0] == (0, 15)
    assert res.history[0]["max_err"] == 0.5
    assert np.isclose(res.history[1]["max_err"], 0.25)


def test_max_err_monotone_and_terminates():
    res = greedy_product_exact(2, 16, eps=0.03)
    errs = [r["max_err"] for r in res.history]
    assert all(a >= b - 1e-12 for a, b in zip(errs, errs[1:], strict=False))
    assert errs[-1] <= 0.03


def test_cell_count_is_product_of_intervals():
    res = greedy_product_exact(2, 16, eps=0.05)
    cuts = [len({k for (j, k) in res.selected if j == jj}) for jj in range(2)]
    assert res.n_cells == (cuts[0] + 1) * (cuts[1] + 1)
    assert np.isclose(res.history[-1]["bits"], np.log2(res.n_cells))


def test_full_refinement_reaches_zero_error():
    res = greedy_product_exact(1, 8, eps=0.0)
    assert res.history[-1]["max_err"] == 0.0
    assert res.n_cells == 8
    assert len(res.selected) == 7
    assert len(res.history) == 8
    with pytest.raises(ValueError, match="never reached"):
        stopping_bits(res, -1.0)


def test_invalid_grids_and_epsilons_rejected():
    with pytest.raises(ValueError):
        greedy_product_exact(0, 8, 0.1)
    with pytest.raises(ValueError):
        greedy_product_exact(1, 8, float("nan"))
    with pytest.raises(ValueError):
        greedy_product_exact(1, 8, -0.1)


def test_stopping_bits_prefix_property():
    tight = greedy_product_exact(2, 32, eps=0.02)
    loose = greedy_product_exact(2, 32, eps=0.1)
    t_star, bits = stopping_bits(tight, 0.1)
    assert t_star == len(loose.selected)
    assert np.isclose(bits, loose.history[-1]["bits"])
    assert tight.selected[:t_star] == loose.selected


@pytest.mark.parametrize("d,N,eps", [(2, 16, 1e-12), (3, 8, 1e-12), (2, 33, 0.005)])
def test_product_solver_matches_explicit_points(d, N, eps):
    # dyadic sizes make every weight sum exact, so the cut sequences must agree;
    # on N = 33 floating-point ties may reorder cuts, so only the error trace
    # and the final partition are compared
    selected, history = greedy_partition(make_grid(d, N), N, eps)
    product = greedy_product_exact(d, N, eps)
    assert len(history) == len(product.history)
    for rg, rp in zip(history, product.history, strict=True):
        assert abs(rg["max_err"] - rp["max_err"]) <= 1e-9
    assert history[-1]["n_cells"] == product.n_cells
    if N & (N - 1) == 0:
        assert selected == product.selected, (d, N)
        assert [h["n_cells"] for h in history] == [h["n_cells"] for h in product.history]


def test_paper_scaling_preserves_saved_traces(tmp_path):
    saved = json.loads((ROOT / "outputs/greedy/scaling_results.json").read_text())
    cfg = {
        "experiment": {
            "grid_sizes": [[run["d"], run["N"]] for run in saved["runs"]],
            "epsilon_grid": saved["epsilon_grid"],
        }
    }
    run_greedy(cfg, str(tmp_path))
    generated = json.loads((tmp_path / "scaling_results.json").read_text())
    assert generated == saved


def test_saved_traces_follow_the_paper_closed_form():
    # equation (8): log|P| = d ceil(log2(1 / (2 eps))) on every power-of-two grid and target
    saved = json.loads((ROOT / "outputs/greedy/scaling_results.json").read_text())
    assert [(run["d"], run["N"]) for run in saved["runs"]] == [(1, 256), (2, 128), (3, 64), (4, 32)]
    assert len(saved["epsilon_grid"]) == 11
    for run in saved["runs"]:
        assert [row["eps"] for row in run["rows"]] == [
            eps for eps in saved["epsilon_grid"] if eps >= 1 / run["N"]
        ]
        for row in run["rows"]:
            assert row["bits"] == closed_form_bits(run["d"], row["eps"])
