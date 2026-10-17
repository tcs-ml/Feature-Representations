"""Concept-class metadata carried alongside every label matrix."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConceptMetadata:
    name: str
    dual_vc_dim: int
    num_inputs: int
    num_tasks: int
    extra: dict[str, float | int | str] = field(default_factory=dict)
