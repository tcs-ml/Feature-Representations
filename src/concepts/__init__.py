"""Concept-class registry."""

from __future__ import annotations

from concepts.matrix import LabelMatrixClass
from concepts.parity import make_parity
from concepts.powerset import make_powerset, make_transposed_powerset

__all__ = [
    "LabelMatrixClass",
    "build_concept",
    "make_parity",
    "make_powerset",
    "make_transposed_powerset",
]

FAMILIES = {
    "powerset": make_powerset,
    "transposed_powerset": make_transposed_powerset,
    "parity": make_parity,
}


def build_concept(family: str, size: int) -> LabelMatrixClass:
    """Build one concept-class instance; `size` is C for powerset families and d for parity."""
    if family not in FAMILIES:
        raise ValueError(f"unknown concept family {family!r}")
    return FAMILIES[family](size)
