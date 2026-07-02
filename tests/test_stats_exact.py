r"""Golden tests for exact small-sample inference (``membench.stats.exact``).

Every asserted number is either (a) hand-derived from a closed form and quoted in the
comment above the assertion, (b) transcribed from the paper's worked examples with the
label cited, or (c) checked against the defining Beta-quantile / binomial-tail identity
directly -- never a value produced by running the module under test.

Paper anchors: App. ``app:worked`` worked examples (j) chain-recovery ``40/40 ->
[0.912, 1.000]`` and (n) supersession; Fig. ``fig:forest`` (F024) exact intervals /
McNemar at the ``p_hat = 1`` boundary cells.
"""

from __future__ import annotations

import pytest

from membench.stats.exact import ClopperPearsonInterval, clopper_pearson


class TestClopperPearsonClosedForms:
    def test_zero_successes_lower_is_zero_upper_is_closed_form(self) -> None:
        # x = 0/n has an exact upper limit 1 - (alpha/2)^(1/n) (Beta(1, n) CDF is
        # 1 - (1-x)^n, so ppf(1 - alpha/2) solves (1-x)^n = alpha/2). n = 20, alpha 0.05:
        # upper = 1 - 0.025^(1/20).
        ci = clopper_pearson(0, 20, 0.05)
        assert isinstance(ci, ClopperPearsonInterval)
        assert ci.low == 0.0
        assert ci.high == pytest.approx(1.0 - 0.025 ** (1.0 / 20), abs=1e-12)
        assert ci.point == 0.0

    def test_all_successes_upper_is_one_lower_is_closed_form(self) -> None:
        # x = n/n has an exact lower limit (alpha/2)^(1/n) (Beta(n, 1) CDF is x^n, so
        # ppf(alpha/2) solves x^n = alpha/2) and upper limit exactly 1.
        ci = clopper_pearson(20, 20, 0.05)
        assert ci.high == 1.0
        assert ci.low == pytest.approx(0.025 ** (1.0 / 20), abs=1e-12)
        assert ci.point == 1.0

    def test_closed_form_tracks_alpha(self) -> None:
        # Same closed form at a 99% interval (alpha = 0.01): lower = 0.005^(1/10).
        ci = clopper_pearson(10, 10, 0.01)
        assert ci.low == pytest.approx(0.005 ** (1.0 / 10), abs=1e-12)
        assert ci.high == 1.0


class TestClopperPearsonPaperWorkedExample:
    def test_chain_recovery_40_of_40_boundary_cell(self) -> None:
        # Paper App. app:worked worked example (j): a typed store recovers the chain at
        # every one of n = 40 paired queries (40/40), reported as the exact interval
        # [0.912, 1.000]. This is the n/n saturation cell: upper = 1 exactly, and
        # lower = (alpha/2)^(1/n) = 0.025^(1/40), which rounds to 0.912 at 3 dp.
        ci = clopper_pearson(40, 40, 0.05)
        assert ci.high == 1.0
        # Paper-published 3-dp value (worked example (j)).
        assert ci.low == pytest.approx(0.912, abs=1e-3)
        # Exact closed form the paper value is a rounding of.
        assert ci.low == pytest.approx(0.025 ** (1.0 / 40), abs=1e-12)
        assert round(ci.low, 3) == 0.912
