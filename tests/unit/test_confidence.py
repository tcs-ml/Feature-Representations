"""One-sided binomial bounds on the worst-task error."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from evaluation.confidence import binomial_cdf, binomial_upper_bound, worst_task_bound


def test_cdf_matches_direct_summation():
    from math import comb

    n, p = 20, 0.3
    for k in range(n + 1):
        direct = sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))
        assert binomial_cdf(k, n, p) == pytest.approx(direct, rel=1e-12)
    assert binomial_cdf(-1, n, p) == 0.0
    assert binomial_cdf(n, n, p) == 1.0
    assert binomial_cdf(3, n, 0.0) == 1.0 and binomial_cdf(3, n, 1.0) == 0.0


def test_bound_solves_the_defining_equation():
    for errors, n in [(1, 50), (7, 200), (40, 1000), (999, 1000)]:
        upper = binomial_upper_bound(errors, n, 0.95)
        assert errors / n < upper < 1.0
        assert binomial_cdf(errors, n, upper) == pytest.approx(0.05, abs=1e-9)


def test_endpoints_and_monotonicity():
    n = 598
    assert binomial_upper_bound(0, n, 0.95) == pytest.approx(1 - 0.05 ** (1 / n))
    assert binomial_upper_bound(0, n, 0.95) <= 0.005  # zero errors on 598 draws certify 0.5%
    assert binomial_upper_bound(0, 500, 0.95) > 0.005  # but not on 500 draws
    assert binomial_upper_bound(n, n, 0.95) == 1.0
    bounds = [binomial_upper_bound(k, n, 0.95) for k in range(0, n + 1, 7)]
    assert all(a < b for a, b in pairwise(bounds))
    assert binomial_upper_bound(3, 100, 0.99) > binomial_upper_bound(3, 100, 0.95)
    with pytest.raises(ValueError):
        binomial_upper_bound(5, 4, 0.95)
    with pytest.raises(ValueError):
        binomial_upper_bound(1, 4, 1.0)


def test_worst_task_bound_uses_the_largest_count():
    rates = np.array([0.0, 2 / 400, 5 / 400])
    bound = worst_task_bound(rates, 400, 0.95)
    assert bound.errors == 5 and bound.n == 400
    assert bound.point == pytest.approx(5 / 400)
    assert bound.ucb == pytest.approx(binomial_upper_bound(5, 400, 0.95))
    zero = worst_task_bound(np.zeros(3), 400, 0.95)
    assert zero.errors == 0 and zero.ucb > 0.0
    with pytest.raises(ValueError, match="multiples"):
        worst_task_bound(np.array([0.3]), 7, 0.95)
    with pytest.raises(ValueError, match="at least one task"):
        worst_task_bound(np.array([]), 7, 0.95)
