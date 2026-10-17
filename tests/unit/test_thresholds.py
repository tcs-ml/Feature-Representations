"""Threshold-product data, exact error counting, grid rounding, and equal-width quantization."""

from __future__ import annotations

import numpy as np
import pytest

from data.thresholds import ThresholdProductConfig, generate_threshold_dataset
from evaluation.thresholds import (
    equal_width_bin_codes,
    equal_width_bin_readout,
    equal_width_bin_worst_error,
    lower_median,
    round_to_grid,
    threshold_error_counts,
    worst_threshold_errors,
)
from models.sae import quantize_sparse_codes, quantized_activations

SAMPLES = {"train": 40, "reconstruction_validation": 8, "test": 24}


def brute_force_counts(x: np.ndarray, x_hat: np.ndarray, grid_size: int) -> np.ndarray:
    r = x.shape[1]
    counts = np.zeros((r, grid_size - 1), dtype=np.int64)
    for j in range(r):
        for a in range(1, grid_size):
            counts[j, a - 1] = int(((x[:, j] <= a) != (x_hat[:, j] <= a)).sum())
    return counts


def test_error_counts_match_brute_force():
    rng = np.random.default_rng(0)
    for grid_size, r, n in [(2, 1, 5), (7, 3, 50), (16, 2, 200)]:
        x = rng.integers(1, grid_size + 1, size=(n, r))
        x_hat = rng.integers(1, grid_size + 1, size=(n, r))
        expected = brute_force_counts(x, x_hat, grid_size)
        np.testing.assert_array_equal(threshold_error_counts(x, x_hat, grid_size), expected)
        assert worst_threshold_errors(x, x_hat, grid_size) == expected.max()
        assert worst_threshold_errors(x, x, grid_size) == 0
    with pytest.raises(ValueError):
        threshold_error_counts(np.array([[0]]), np.array([[1]]), 4)
    with pytest.raises(ValueError):
        threshold_error_counts(np.array([[1, 2]]), np.array([[1]]), 4)


def test_round_to_grid_endpoints_and_midpoints():
    grid_size = 5  # normalized spacing 0.25
    y = np.array([-0.3, 0.0, 0.124, 0.126, 0.5, 1.0, 1.7])
    assert round_to_grid(y, grid_size).tolist() == [1, 1, 1, 2, 3, 5, 5]


def test_lower_median_is_the_ceil_half_order_statistic():
    assert lower_median(np.array([5, 1, 3, 9])) == 3  # second smallest of four
    assert lower_median(np.array([5, 1, 3])) == 3
    assert lower_median(np.array([[4, 1], [2, 8], [6, 5], [0, 7]]), axis=0).tolist() == [2, 5]
    with pytest.raises(ValueError):
        lower_median(np.array([]))


def test_equal_width_quantization_error_is_the_paper_closed_form_on_dyadic_grids():
    # the figure's crosses: 2^(-bits - 1) with `bits` bits per coordinate
    for grid_size, bits in [(8, 1), (8, 2), (64, 3), (1024, 1), (1024, 5), (1024, 9)]:
        assert equal_width_bin_worst_error(grid_size, bits) == pytest.approx(2.0 ** (-bits - 1))
        readout = equal_width_bin_readout(grid_size, bits)
        assert len(readout) == 2**bits
        codes = equal_width_bin_codes(np.arange(1, grid_size + 1)[:, None], grid_size, bits)[:, 0]
        assert codes.min() == 0 and codes.max() == 2**bits - 1
        assert np.all(np.diff(codes) >= 0)
    assert equal_width_bin_worst_error(10, 1) == pytest.approx(0.2)  # width-5 bins: 2 of 10 points
    assert equal_width_bin_worst_error(1024, 10) == 0.0  # one grid point per bin


def test_dataset_streams_are_deterministic_and_separate():
    cfg = ThresholdProductConfig(grid_size=16, dimension=2, samples=SAMPLES, seed=7)
    first, second = generate_threshold_dataset(cfg), generate_threshold_dataset(cfg)
    for split in SAMPLES:
        np.testing.assert_array_equal(first.split(split).x, second.split(split).x)
        assert first.split(split).x.shape == (SAMPLES[split], 2)
        assert first.split(split).x.min() >= 1 and first.split(split).x.max() <= 16
        np.testing.assert_allclose(
            first.split(split).features, (first.split(split).x - 1) / 15, atol=1e-6
        )
    assert not np.array_equal(first.split("train").x, first.split("test").x)
    other = generate_threshold_dataset(
        ThresholdProductConfig(grid_size=16, dimension=1, samples=SAMPLES, seed=7)
    )
    assert not np.array_equal(first.split("train").x[:, :1], other.split("train").x)
    with pytest.raises(ValueError):
        ThresholdProductConfig(grid_size=1, dimension=1, samples=SAMPLES, seed=0)
    with pytest.raises(ValueError):
        ThresholdProductConfig(grid_size=4, dimension=1, samples={"train": 1}, seed=0)


def test_quantized_activations_agree_with_keys():
    rng = np.random.default_rng(2)
    z = np.abs(rng.standard_normal((50, 6)))
    z[rng.random((50, 6)) < 0.5] = 0.0
    keys = quantize_sparse_codes(z, value_bits=3, vmax=1.5, k_slots=6)
    zq = quantized_activations(z, value_bits=3, vmax=1.5)
    assert zq.dtype == np.float32
    assert set(np.unique(np.rint(zq / 1.5 * 7))) <= set(range(8))
    for row, key in zip(zq, keys, strict=True):
        support = [i for i in key[:6] if i >= 0]
        assert sorted(np.flatnonzero(row).tolist()) == sorted(support)
    with pytest.raises(ValueError):
        quantized_activations(z, 2, 0.0)
