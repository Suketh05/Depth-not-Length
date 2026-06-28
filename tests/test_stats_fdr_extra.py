"""Tests for Storey q-values and Benjamini-Yekutieli FDR control.

Golden values are derived by hand (see the worked arithmetic in each test) and
never by running the code under test, so the assertions are independent of the
implementation.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from membench.stats.fdr_extra import (
    benjamini_yekutieli_adjust,
    harmonic_factor,
    storey_pi0,
    storey_qvalues,
)

# Fixed p-value vector used across the golden checks (already sorted ascending).
P = [0.001, 0.008, 0.039, 0.041, 0.9]
M = len(P)  # = 5


def test_harmonic_factor_golden() -> None:
    # c(5) = 1 + 1/2 + 1/3 + 1/4 + 1/5 = 137/60 = 2.28333...
    assert harmonic_factor(5) == pytest.approx(137.0 / 60.0)
    assert harmonic_factor(1) == pytest.approx(1.0)


def test_harmonic_factor_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="m must be"):
        harmonic_factor(0)


def test_benjamini_yekutieli_golden() -> None:
    # Step-up: raw_k = c(5) * (m/k) * p_(k), then running-min from the right,
    # then clip to 1. With c(5) = 137/60:
    #   k=1: (137/60)*5*0.001   = 0.0114166667
    #   k=2: (137/60)*(5/2)*0.008 = 0.0456666667
    #   k=3: (137/60)*(5/3)*0.039 = 0.1484166667
    #   k=4: (137/60)*(5/4)*0.041 = 0.1170208333
    #   k=5: (137/60)*1*0.9       = 2.055  -> clip -> 1.0
    # running-min from k=5 down pulls k=3 down to k=4's 0.1170208333:
    #   -> [0.0114166667, 0.0456666667, 0.1170208333, 0.1170208333, 1.0]
    c5 = 137.0 / 60.0
    expected = [
        c5 * 5.0 * 0.001,
        c5 * 2.5 * 0.008,
        c5 * 1.25 * 0.041,  # k=3 inherits the smaller k=4 value
        c5 * 1.25 * 0.041,
        1.0,
    ]
    got = benjamini_yekutieli_adjust(P)
    assert got == pytest.approx(expected)


def test_by_is_monotone_and_bounded() -> None:
    got = benjamini_yekutieli_adjust(P)
    # P is sorted ascending, so the adjusted values must be non-decreasing
    # and bounded by 1.
    assert all(a <= b + 1e-12 for a, b in pairwise(got))
    assert all(0.0 <= v <= 1.0 for v in got)


def test_by_is_bh_times_cm() -> None:
    # BY = c(m) * BH (before the cap at 1). With pi0 = 1 the Storey q-values
    # equal the BH adjusted p-values, so BY = c(m) * q_{pi0=1}, capped at 1.
    c5 = 137.0 / 60.0
    bh = list(storey_qvalues(P, pi0=1.0).qvalues)
    expected = [min(1.0, c5 * v) for v in bh]
    assert benjamini_yekutieli_adjust(P) == pytest.approx(expected)


def test_storey_qvalues_pi0_one_equals_bh_golden() -> None:
    # With pi0 = 1, q(p_(i)) = min_{k>=i} m/k * p_(k) (Benjamini-Hochberg):
    #   k=1: 5*0.001    = 0.005
    #   k=2: 2.5*0.008  = 0.020
    #   k=3: 1.6667*0.039 = 0.065
    #   k=4: 1.25*0.041 = 0.05125
    #   k=5: 1*0.9      = 0.9
    # running-min from the right pulls k=3 down to 0.05125:
    #   -> [0.005, 0.020, 0.05125, 0.05125, 0.9]
    expected = [0.005, 0.020, 0.05125, 0.05125, 0.9]
    result = storey_qvalues(P, pi0=1.0)
    assert result.pi0 == 1.0
    assert list(result.qvalues) == pytest.approx(expected)


def test_storey_qvalues_preserve_input_order() -> None:
    # Unsorted input must produce q-values aligned to the input positions.
    shuffled = [0.9, 0.001, 0.041, 0.008, 0.039]
    expected_by_value = dict(zip(P, [0.005, 0.020, 0.05125, 0.05125, 0.9], strict=True))
    got = storey_qvalues(shuffled, pi0=1.0).qvalues
    expected = [expected_by_value[v] for v in shuffled]
    assert list(got) == pytest.approx(expected)


def test_storey_pi0_single_lambda_golden() -> None:
    # Fixed-lambda Storey (2002): pi0(lambda) = #{p > lambda} / (m * (1 - lambda)).
    # With lambda = 0.5, only 0.9 exceeds 0.5 -> #{p>0.5} = 1, m = 5:
    #   pi0 = 1 / (5 * 0.5) = 0.4
    assert storey_pi0(P, lambdas=(0.5,)) == pytest.approx(0.4)


def test_storey_pi0_smoother_in_unit_interval() -> None:
    # Deterministic fixture: 45 "null" p-values spread across (0,1) plus 5 tiny
    # "signal" p-values -> the true null fraction is 45/50 = 0.9.
    nulls = list(np.linspace(0.02, 0.99, 45))
    signal = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4]
    pvals = nulls + signal
    pi0 = storey_pi0(pvals)
    assert 0.0 < pi0 <= 1.0
    # Should be in the right neighbourhood of the planted 0.9.
    assert 0.7 < pi0 < 1.0


def test_storey_qvalues_default_invariants() -> None:
    nulls = list(np.linspace(0.02, 0.99, 45))
    signal = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4]
    pvals = nulls + signal
    result = storey_qvalues(pvals)
    assert 0.0 < result.pi0 <= 1.0
    # q-values are bounded by 1 and monotone in the p-value ranking.
    assert all(0.0 <= q <= 1.0 for q in result.qvalues)
    order = np.argsort(pvals)
    sorted_q = [result.qvalues[i] for i in order]
    assert all(a <= b + 1e-12 for a, b in pairwise(sorted_q))
    # pi0 <= 1 means Storey q-values never exceed BH adjusted p-values.
    bh = storey_qvalues(pvals, pi0=1.0).qvalues
    assert all(q <= b + 1e-12 for q, b in zip(result.qvalues, bh, strict=True))


def test_rejects_empty_and_out_of_range() -> None:
    with pytest.raises(ValueError, match="at least one p-value"):
        benjamini_yekutieli_adjust([])
    with pytest.raises(ValueError, match="p-values must lie"):
        storey_qvalues([0.1, 1.5])
    with pytest.raises(ValueError, match="pi0 must lie"):
        storey_qvalues(P, pi0=1.5)
    with pytest.raises(ValueError, match="lambda values must lie"):
        storey_pi0(P, lambdas=(1.0,))
