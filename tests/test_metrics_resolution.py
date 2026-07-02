"""Tests for SWE-ContextBench resolution scoring (sec:pubbench, tab:stdbench).

The SWE-bench resolution predicate: resolved iff every FAIL_TO_PASS test passes and
every PASS_TO_PASS test stays passing. Goldens are hand-computed rates; none asserts
the paper's 47.3 (that is the full-run outcome, not a unit identity).
"""

from __future__ import annotations

import pytest

from membench.metrics.resolution import (
    InstanceResolution,
    TestStatus,
    resolution_rate,
    resolve_instance,
)


def _resolution(instance_id: str, *, resolved: bool) -> InstanceResolution:
    return InstanceResolution(
        instance_id=instance_id,
        resolved=resolved,
        fail_to_pass_ok=resolved,
        pass_to_pass_ok=resolved,
        unmet_tests=() if resolved else ("t_missing",),
    )


class TestResolveInstance:
    def test_all_required_pass_is_resolved(self) -> None:
        result = resolve_instance(
            "django-1",
            fail_to_pass=["test_new_behaviour"],
            pass_to_pass=["test_regression_guard"],
            test_results={
                "test_new_behaviour": TestStatus.PASSED,
                "test_regression_guard": TestStatus.PASSED,
            },
        )
        assert result.resolved is True
        assert result.fail_to_pass_ok is True
        assert result.pass_to_pass_ok is True
        assert result.unmet_tests == ()

    def test_fail_to_pass_still_failing_is_unresolved(self) -> None:
        result = resolve_instance(
            "django-2",
            fail_to_pass=["test_new_behaviour"],
            pass_to_pass=[],
            test_results={"test_new_behaviour": TestStatus.FAILED},
        )
        assert result.resolved is False
        assert result.fail_to_pass_ok is False
        assert result.unmet_tests == ("test_new_behaviour",)

    def test_pass_to_pass_regression_is_unresolved(self) -> None:
        # The fix works but breaks a previously-passing test: not resolved.
        result = resolve_instance(
            "sphinx-3",
            fail_to_pass=["test_fix"],
            pass_to_pass=["test_existing"],
            test_results={"test_fix": TestStatus.PASSED, "test_existing": TestStatus.ERROR},
        )
        assert result.resolved is False
        assert result.fail_to_pass_ok is True
        assert result.pass_to_pass_ok is False
        assert result.unmet_tests == ("test_existing",)

    def test_missing_required_test_counts_as_unmet(self) -> None:
        # A required test absent from the results map did not pass.
        result = resolve_instance(
            "pytest-4",
            fail_to_pass=["test_a", "test_b"],
            pass_to_pass=[],
            test_results={"test_a": TestStatus.PASSED},
        )
        assert result.resolved is False
        assert result.unmet_tests == ("test_b",)

    def test_skipped_required_test_is_not_a_pass(self) -> None:
        result = resolve_instance(
            "django-5",
            fail_to_pass=["test_a"],
            pass_to_pass=[],
            test_results={"test_a": TestStatus.SKIPPED},
        )
        assert result.resolved is False

    def test_empty_fail_to_pass_raises(self) -> None:
        with pytest.raises(ValueError, match="no fail_to_pass"):
            resolve_instance("bad-6", fail_to_pass=[], pass_to_pass=["x"], test_results={})

    def test_unmet_lists_fail_to_pass_before_pass_to_pass(self) -> None:
        result = resolve_instance(
            "django-7",
            fail_to_pass=["f2p_bad"],
            pass_to_pass=["p2p_bad"],
            test_results={"f2p_bad": TestStatus.FAILED, "p2p_bad": TestStatus.FAILED},
        )
        assert result.unmet_tests == ("f2p_bad", "p2p_bad")


class TestResolutionRate:
    def test_three_of_eight_resolved_golden(self) -> None:
        # 3 resolved of 8 => rate 3/8 = 0.375 => 37.5%.
        resolutions = [_resolution(f"i{i}", resolved=i < 3) for i in range(8)]
        summary = resolution_rate(resolutions)
        assert summary.n_resolved == 3
        assert summary.n_total == 8
        assert summary.rate == pytest.approx(0.375)
        assert summary.percent == pytest.approx(37.5)

    def test_all_resolved_saturates(self) -> None:
        summary = resolution_rate([_resolution("i0", resolved=True)])
        assert summary.rate == pytest.approx(1.0)
        assert summary.percent == pytest.approx(100.0)

    def test_none_resolved_floors(self) -> None:
        summary = resolution_rate([_resolution(f"i{i}", resolved=False) for i in range(4)])
        assert summary.rate == 0.0
        assert summary.percent == 0.0

    def test_percent_is_rate_times_100_identity(self) -> None:
        # Unit identity only: percent == 100 * rate for any clean fraction.
        # 47 resolved of 100 => 47.0%; demonstrates the tab:stdbench percent unit
        # (this is arithmetic, NOT a reproduction of the paper's 47.3 headline).
        resolutions = [_resolution(f"i{i}", resolved=i < 47) for i in range(100)]
        summary = resolution_rate(resolutions)
        assert summary.percent == pytest.approx(100.0 * summary.rate)
        assert summary.percent == pytest.approx(47.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            resolution_rate([])

    def test_end_to_end_predicate_feeds_rate(self) -> None:
        # Two instances scored through resolve_instance, one resolved -> 50%.
        good = resolve_instance(
            "ok",
            fail_to_pass=["t"],
            pass_to_pass=[],
            test_results={"t": TestStatus.PASSED},
        )
        bad = resolve_instance(
            "no",
            fail_to_pass=["t"],
            pass_to_pass=[],
            test_results={"t": TestStatus.FAILED},
        )
        summary = resolution_rate([good, bad])
        assert summary.n_resolved == 1
        assert summary.percent == pytest.approx(50.0)
