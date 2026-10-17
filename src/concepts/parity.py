"""Parity concept family over F_2^d.

h_a(x) = <a, x> mod 2 for every a in F_2^d (the constant-zero task a=0 is
included). tasks 2^d, atoms 2^d, dual VC = d, zero-error code: d bits.
"""

from __future__ import annotations

import numpy as np

from concepts.matrix import LabelMatrixClass


def parity_matrix(dim: int) -> np.ndarray:
    if dim < 1 or dim > 12:
        raise ValueError(f"parity needs 1 <= d <= 12, got {dim}")
    n = 2**dim
    a = np.arange(n, dtype=np.int64)[:, None]
    x = np.arange(n, dtype=np.int64)[None, :]
    return (np.bitwise_count(a & x) % 2).astype(bool)


def make_parity(dim: int) -> LabelMatrixClass:
    return LabelMatrixClass(
        name=f"parity-d{dim}",
        label_matrix=parity_matrix(dim),
        dual_vc_dim=dim,
        # the d singleton parities are dually shattered by all 2^d atoms
        extra={"family": "parity", "d": dim, "dual_shatter_mass": 1.0},
    )
