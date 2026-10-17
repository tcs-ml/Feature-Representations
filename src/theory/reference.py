"""The paper's noiseless references for a finite task family.

The lossless reference B_0 = ceil(log2(number of atoms)) is the length of a
code that names the atom, so every task is answered without error; the
concept classes record it as `zero_error_bits`. The shattered-set bound
B_sh(eps) = (1 - h2(eps / p_sh)) d* (Proposition 3.3) is a lower bound on
the bits any representation with error eps needs, where p_sh is the mass of
a dually shattered atom set; the concept classes record it as
`dual_shatter_mass`.
"""

from __future__ import annotations

import math


def binary_entropy(p: float) -> float:
    """Binary entropy h2(p) in bits, including its endpoint values."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"h2 needs p in [0, 1], got {p}")
    if p in (0.0, 1.0):
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def shattered_set_bound(dual_vc_dim: int, shatter_mass: float, epsilon: float) -> float:
    """B_sh(eps) = (1 - h2(eps / p_sh)) d* for 0 < eps <= p_sh / 2.

    Conditioned on the shattered atoms, the d* shattered tasks are independent
    fair bits, each decoded with error at most eps / p_sh, so Fano's inequality
    bounds the representation entropy from below by this quantity.
    """
    if dual_vc_dim < 1:
        raise ValueError(f"dual VC dimension must be >= 1, got {dual_vc_dim}")
    if not 0.0 < shatter_mass <= 1.0:
        raise ValueError(f"dual_shatter_mass must lie in (0, 1], got {shatter_mass}")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be finite and > 0, got {epsilon}")
    conditional = epsilon / shatter_mass
    if conditional > 0.5:
        raise ValueError(f"eps / p_sh = {conditional:g} lies outside the bound's range (0, 1/2]")
    return (1.0 - binary_entropy(conditional)) * dual_vc_dim
