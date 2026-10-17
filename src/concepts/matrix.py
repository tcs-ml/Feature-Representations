"""Concrete finite concept class backed by an explicit label matrix."""

from __future__ import annotations

import math

import numpy as np

from theory.metadata import ConceptMetadata


class LabelMatrixClass:
    """A finite concept class Y in {0,1}^[num_tasks x num_atoms] with uniform atoms.

    `extra` carries family metadata; the zero-error codeword count (distinct
    label columns) and its bit length are always derived from the matrix.
    """

    def __init__(
        self, name: str, label_matrix: np.ndarray, dual_vc_dim: int, extra: dict | None = None
    ) -> None:
        y = np.asarray(label_matrix)
        if y.ndim != 2 or y.size == 0:
            raise ValueError(f"label matrix must be nonempty and 2-D, got shape {y.shape}")
        if y.dtype != bool:
            if not np.isin(y, (0, 1)).all():
                raise ValueError("label matrix entries must be binary")
            y = y.astype(bool)
        self._name = name
        self._y = y.copy()
        self._y.setflags(write=False)
        self._p = np.full(y.shape[1], 1.0 / y.shape[1])
        self._p.setflags(write=False)
        self._dual_vc = dual_vc_dim
        codewords = int(np.unique(y, axis=1).shape[1])
        self._extra = {
            **dict(extra or {}),
            "zero_error_codewords": codewords,
            "zero_error_bits": math.ceil(math.log2(codewords)),
        }

    @property
    def name(self) -> str:
        return self._name

    def label_matrix(self) -> np.ndarray:
        return self._y

    def input_probabilities(self) -> np.ndarray:
        return self._p

    def metadata(self) -> ConceptMetadata:
        return ConceptMetadata(
            name=self._name,
            dual_vc_dim=self._dual_vc,
            num_inputs=int(self._y.shape[1]),
            num_tasks=int(self._y.shape[0]),
            extra=dict(self._extra),
        )
