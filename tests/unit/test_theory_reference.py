"""The shattered-set bound B_sh(eps) and the lossless reference B_0 of every family."""

from __future__ import annotations

import pytest

from concepts import make_parity, make_powerset, make_transposed_powerset
from theory.reference import binary_entropy, shattered_set_bound


def test_binary_entropy_known_values():
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert binary_entropy(0.5) == pytest.approx(1.0)
    assert binary_entropy(0.25) == pytest.approx(0.8112781244591328)
    with pytest.raises(ValueError):
        binary_entropy(1.5)


def test_bound_formula_with_full_shatter_mass():
    meta = make_transposed_powerset(4).metadata()  # d* = 4, p_sh = 1
    assert meta.extra["dual_shatter_mass"] == 1.0
    assert shattered_set_bound(4, 1.0, 0.01) == pytest.approx((1.0 - binary_entropy(0.01)) * 4)


def test_shatter_mass_dilution_for_non_power_of_two_powerset():
    meta = make_powerset(6).metadata()  # d* = 2, p_sh = 4/6
    assert meta.extra["dual_shatter_mass"] == pytest.approx(4 / 6)
    expected = (1.0 - binary_entropy(0.05 / (4 / 6))) * 2
    assert shattered_set_bound(2, 4 / 6, 0.05) == pytest.approx(expected)


def test_bound_is_positive_below_one_half_and_vanishes_there():
    assert shattered_set_bound(1, 1.0, 0.1) == pytest.approx(1.0 - binary_entropy(0.1))
    assert shattered_set_bound(1, 1.0, 0.5) == 0.0


def test_bound_never_exceeds_the_lossless_reference():
    for meta in (
        make_parity(5).metadata(),
        make_powerset(10).metadata(),
        make_transposed_powerset(6).metadata(),
    ):
        for eps in (0.005, 0.05, 0.2):
            bound = shattered_set_bound(meta.dual_vc_dim, meta.extra["dual_shatter_mass"], eps)
            assert 0.0 <= bound <= meta.extra["zero_error_bits"]


@pytest.mark.parametrize(
    "args", [(3, 1.0, 0.51), (3, 1.0, 0.0), (3, 1.0, -0.01), (0, 1.0, 0.1), (3, 0.0, 0.1)]
)
def test_out_of_range_arguments_raise(args):
    with pytest.raises(ValueError):
        shattered_set_bound(*args)


def test_shatter_masses_of_the_paper_families():
    for d in (1, 2, 3):
        assert make_parity(d).metadata().extra["dual_shatter_mass"] == 1.0
    assert make_powerset(4).metadata().extra["dual_shatter_mass"] == 1.0
    assert make_powerset(12).metadata().extra["dual_shatter_mass"] == pytest.approx(8 / 12)
    assert make_transposed_powerset(3).metadata().extra["dual_shatter_mass"] == 1.0
