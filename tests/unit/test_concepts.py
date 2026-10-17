"""Known-value tests for concept families."""

from __future__ import annotations

import numpy as np
import pytest

from concepts import (
    LabelMatrixClass,
    build_concept,
    make_parity,
    make_powerset,
    make_transposed_powerset,
)


def test_powerset_contains_every_binary_pattern():
    for c in (1, 2, 3, 4):
        y = make_powerset(c).label_matrix()
        assert y.shape == (2**c, c)
        assert y.dtype == bool
        assert np.unique(y, axis=0).shape[0] == 2**c


def test_powerset_metadata():
    meta = make_powerset(4).metadata()
    assert meta.dual_vc_dim == 2
    assert meta.num_tasks == 16
    assert meta.num_inputs == 4
    assert meta.extra["zero_error_codewords"] == 4
    assert meta.extra["zero_error_bits"] == 2


def test_transposed_powerset_swaps_axes_and_metadata():
    c = 3
    base = make_powerset(c)
    t = make_transposed_powerset(c)
    assert t.label_matrix().shape == (c, 2**c)
    np.testing.assert_array_equal(t.label_matrix(), base.label_matrix().T)
    meta = t.metadata()
    assert meta.dual_vc_dim == c
    assert base.metadata().dual_vc_dim == 1
    assert meta.extra["zero_error_bits"] == c
    assert meta.extra["zero_error_codewords"] == 2**c


def test_parity_matches_direct_finite_field_computation():
    d = 3
    y = make_parity(d).label_matrix()
    assert y.shape == (2**d, 2**d)
    for a in range(2**d):
        for x in range(2**d):
            expected = bin(a & x).count("1") % 2
            assert y[a, x] == bool(expected)


def test_parity_metadata():
    meta = make_parity(3).metadata()
    assert meta.dual_vc_dim == 3
    assert meta.extra["zero_error_bits"] == 3


def test_atoms_are_uniform():
    for cls in (make_powerset(3), make_transposed_powerset(3), make_parity(2)):
        p = cls.input_probabilities()
        np.testing.assert_allclose(p, 1.0 / cls.metadata().num_inputs)


def test_label_matrix_validation():
    with pytest.raises(ValueError):
        LabelMatrixClass("bad", np.array([[0, 2]]), dual_vc_dim=1)
    with pytest.raises(ValueError):
        LabelMatrixClass("bad", np.zeros((0, 3), dtype=bool), dual_vc_dim=0)


def test_zero_error_metadata_counts_distinct_columns():
    y = np.array([[1, 1, 0], [0, 0, 1]], dtype=bool)
    meta = LabelMatrixClass("two", y, dual_vc_dim=1).metadata()
    assert meta.extra["zero_error_codewords"] == 2
    assert meta.extra["zero_error_bits"] == 1


def test_registry():
    assert build_concept("powerset", 3).label_matrix().shape == (8, 3)
    assert build_concept("transposed_powerset", 3).label_matrix().shape == (3, 8)
    assert build_concept("parity", 2).label_matrix().shape == (4, 4)
    with pytest.raises(ValueError):
        build_concept("nope", 3)
