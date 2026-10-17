"""Exact worst-threshold error counting on [N]^r without a task matrix.

The threshold task f_{j,a}(x) = 1{x_j <= a} for coordinate j and a in
1..N-1. A decoded coordinate x_hat_j disagrees with the truth x_j exactly on
the thresholds a in [min(x_j, x_hat_j), max(x_j, x_hat_j) - 1], so every
task's error count follows from one difference array per coordinate. A
reconstruction is turned into coordinates by rounding to the grid. The
equal-width quantizer of each coordinate is the analytic comparison in the
paper's figure.
"""

from __future__ import annotations

import numpy as np


def _check_grid(x: np.ndarray, grid_size: int, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError(f"{name} must be [n, r], got shape {x.shape}")
    if x.size and (x.min() < 1 or x.max() > grid_size):
        raise ValueError(f"{name} must hold grid coordinates in 1..{grid_size}")
    return x


def threshold_error_counts(x: np.ndarray, x_hat: np.ndarray, grid_size: int) -> np.ndarray:
    """Error counts [r, N - 1] of predictions 1{x_hat_j <= a} against 1{x_j <= a}.

    Column a - 1 holds threshold a. A sample contributes one error to every
    threshold between its true and decoded coordinate.
    """
    x = _check_grid(x, grid_size, "x")
    x_hat = _check_grid(x_hat, grid_size, "x_hat")
    if x.shape != x_hat.shape:
        raise ValueError("x and x_hat must have the same shape")
    r = x.shape[1]
    low, high = np.minimum(x, x_hat), np.maximum(x, x_hat)
    difference = np.zeros((r, grid_size + 1), dtype=np.int64)
    for j in range(r):
        np.add.at(difference[j], low[:, j], 1)
        np.add.at(difference[j], high[:, j], -1)
    return np.cumsum(difference, axis=1)[:, 1:grid_size]


def worst_threshold_errors(x: np.ndarray, x_hat: np.ndarray, grid_size: int) -> int:
    """The largest error count over all r (N - 1) threshold tasks."""
    return int(threshold_error_counts(x, x_hat, grid_size).max())


def round_to_grid(y_hat: np.ndarray, grid_size: int) -> np.ndarray:
    """Nearest grid coordinate of a reconstructed normalized coordinate."""
    y_hat = np.asarray(y_hat, dtype=np.float64)
    return np.clip(np.floor(1.0 + (grid_size - 1) * y_hat + 0.5), 1, grid_size).astype(np.int64)


def lower_median(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """The ceil(m / 2)-th smallest value along an axis."""
    values = np.asarray(values)
    m = values.shape[axis]
    if m < 1:
        raise ValueError("lower median of an empty set")
    return np.take(np.sort(values, axis=axis), (m - 1) // 2, axis=axis)


def equal_width_bin_codes(x: np.ndarray, grid_size: int, bits: int) -> np.ndarray:
    """Per-coordinate index of the equal-width bin among 2^bits bins of the grid."""
    x = _check_grid(x, grid_size, "x")
    return ((x - 1) * 2**bits) // grid_size


def equal_width_bin_readout(grid_size: int, bits: int) -> np.ndarray:
    """Lower-median grid point of every equal-width bin under the uniform distribution."""
    grid = np.arange(1, grid_size + 1)
    bins = equal_width_bin_codes(grid[:, None], grid_size, bits)[:, 0]
    return np.array([lower_median(grid[bins == b]) for b in range(2**bits)], dtype=np.int64)


def equal_width_bin_worst_error(grid_size: int, bits: int) -> float:
    """Exact worst threshold error of equal-width quantization with `bits` bits per coordinate.

    Computed on the whole uniform grid, so it is a population quantity, not a
    sample estimate. On a power-of-two grid it equals 2^(-bits - 1).
    """
    grid = np.arange(1, grid_size + 1)[:, None]
    readout = equal_width_bin_readout(grid_size, bits)
    x_hat = readout[equal_width_bin_codes(grid, grid_size, bits)]
    return worst_threshold_errors(grid, x_hat, grid_size) / grid_size
