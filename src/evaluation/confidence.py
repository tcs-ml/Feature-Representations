"""One-sided binomial upper confidence bounds for the worst-task error.

A frozen encoder and decoder are evaluated on n independent, identically
distributed held-out observations, and every task's error count is recorded.
Let k be the largest count. The exact one-sided binomial (Clopper-Pearson)
bound U solves P(Bin(n, U) <= k) = 1 - confidence, with U = 1 when k = n.
The task with the largest true error has at most k errors, and the bound is
monotone in the count, so U covers the worst-task error with the stated
confidence without any correction for the number of tasks. A sample with zero
errors still has the positive bound 1 - alpha^(1/n).

The observations must be independent draws from one fixed distribution.
Stratified, dependent, or weighted samples need a different construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, log1p

import numpy as np


@dataclass(frozen=True)
class WorstTaskBound:
    errors: int  # largest error count over all tasks
    n: int  # held-out observations
    point: float  # errors / n
    ucb: float  # one-sided upper confidence bound on the worst-task error


def binomial_cdf(errors: int, n: int, p: float) -> float:
    """P(Bin(n, p) <= errors), accumulated in log space."""
    if errors < 0:
        return 0.0
    if errors >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    j = np.arange(1, errors + 1, dtype=float)
    log_comb = np.concatenate([[0.0], np.cumsum(np.log((n - j + 1.0) / j))])
    i = np.arange(errors + 1, dtype=float)
    log_terms = log_comb + i * log(p) + (n - i) * log1p(-p)
    peak = float(log_terms.max())
    return float(min(1.0, exp(peak) * float(np.exp(log_terms - peak).sum())))


def binomial_upper_bound(errors: int, n: int, confidence: float = 0.95) -> float:
    """Exact one-sided upper confidence bound on a binomial proportion."""
    if n < 1 or not 0 <= errors <= n:
        raise ValueError(f"need 0 <= errors <= n with n >= 1, got errors={errors}, n={n}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly between 0 and 1, got {confidence}")
    alpha = 1.0 - confidence
    if errors == n:
        return 1.0
    if errors == 0:
        return 1.0 - alpha ** (1.0 / n)
    low, high = errors / n, 1.0
    while high - low > 1e-13:
        mid = 0.5 * (low + high)
        if binomial_cdf(errors, n, mid) > alpha:
            low = mid
        else:
            high = mid
    return high


def worst_task_bound(per_task_errors: np.ndarray, n: int, confidence: float) -> WorstTaskBound:
    """Bound the worst-task error from per-task error rates measured on n observations."""
    rates = np.asarray(per_task_errors, dtype=float)
    if rates.size == 0:
        raise ValueError("worst-task bound needs at least one task")
    counts = np.rint(rates * n)
    if not np.allclose(counts, rates * n, rtol=0.0, atol=1e-6):
        raise ValueError("per-task error rates are not multiples of 1 / n")
    errors = int(counts.max())
    return WorstTaskBound(
        errors=errors,
        n=int(n),
        point=errors / n,
        ucb=binomial_upper_bound(errors, int(n), confidence),
    )
