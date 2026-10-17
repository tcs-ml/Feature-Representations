"""Powerset (all labelings) and transposed-powerset (coordinate) concept families.

Powerset on C atoms: every binary labeling of the atoms.
  tasks 2^C, atoms C, dual VC = floor(log2 C),
  zero-error deterministic code: C codewords = ceil(log2 C) bits.

Transposed powerset: the same matrix transposed.
  tasks C, atoms 2^C, dual VC = C,
  zero-error deterministic code: 2^C codewords = C bits (the coordinate tasks
  separate every input pattern).
"""

from __future__ import annotations

import math

import numpy as np

from concepts.matrix import LabelMatrixClass


def powerset_matrix(num_atoms: int) -> np.ndarray:
    """All 2^C binary labelings of C atoms; row i is the bit pattern of i."""
    if num_atoms < 1 or num_atoms > 20:
        raise ValueError(f"powerset needs 1 <= C <= 20, got {num_atoms}")
    rows = np.arange(2**num_atoms, dtype=np.int64)[:, None]
    return ((rows >> np.arange(num_atoms, dtype=np.int64)) & 1).astype(bool)


def make_powerset(num_atoms: int) -> LabelMatrixClass:
    dual_vc = int(math.floor(math.log2(num_atoms)))
    return LabelMatrixClass(
        name=f"powerset-C{num_atoms}",
        label_matrix=powerset_matrix(num_atoms),
        dual_vc_dim=dual_vc,
        extra={
            "family": "powerset",
            "C": num_atoms,
            # mass of a dually shattered atom set under the uniform atom
            # distribution: the 2^floor(log2 C) atoms whose indices realize
            # every pattern of the bit-indicator tasks
            "dual_shatter_mass": 2**dual_vc / num_atoms,
        },
    )


def make_transposed_powerset(num_atoms_of_base: int) -> LabelMatrixClass:
    c = num_atoms_of_base
    if c > 14:
        raise ValueError(f"transposed powerset with C={c} would have 2^{c} atoms; use C <= 14")
    return LabelMatrixClass(
        name=f"transposed-powerset-C{c}",
        label_matrix=powerset_matrix(c).T,  # [C tasks, 2^C atoms]
        dual_vc_dim=c,
        # the C coordinate tasks are dually shattered by all 2^C atoms
        extra={"family": "transposed_powerset", "C": c, "dual_shatter_mass": 1.0},
    )
