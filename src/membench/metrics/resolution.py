r"""SWE-ContextBench *resolution* scoring (Section ``sec:pubbench``, ``tab:stdbench``).

The third headline row of the public-benchmark sweep (Table~\ref{tab:stdbench},
Figure~\ref{fig:stdbench}; also the leaderboard ``tab:t070``) is
**SWE-ContextBench resolution (%)** -- for ``Brief``, ``47.3`` -- the fraction of code
instances an arm *resolves*. Unlike the LoCoMo/DMR answer scorers of
:mod:`membench.metrics.answer_scoring`, resolution is a test-execution outcome, not a
comparison of an answer string to gold, so it lives in its own module.

Resolution predicate
--------------------
We use the standard SWE-bench resolution criterion [Jimenez2024]_: an instance is
*resolved* iff, after the candidate patch is applied, **every** ``FAIL_TO_PASS`` test
goes from failing to passing **and** every ``PASS_TO_PASS`` test stays passing (no
regression). A required test that did not run (absent from the results) counts as not
passing. A test "passes" iff its recorded status is :attr:`TestStatus.PASSED`.

Formally, with ``F2P`` the fail-to-pass set, ``P2P`` the pass-to-pass set, and
``status`` the post-patch status map,

.. math::

    \text{resolved} \iff
    \bigl(\forall t \in \text{F2P}:\ \text{status}(t) = \text{PASSED}\bigr)
    \ \wedge\
    \bigl(\forall t \in \text{P2P}:\ \text{status}(t) = \text{PASSED}\bigr).

The resolution *rate* is then ``|resolved| / |instances|``, reported as a percentage
(``sec:pubbench``). The paper's ``swe3`` compliance slice is small (``n = 12``), so a
single flipped instance moves the rate by ``\approx 8`` points -- the reason the paper
reads that slice as inside the noise band.

References
----------
.. [Jimenez2024] Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., &
   Narasimhan, K. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub
   Issues?* ICLR 2024. (Defines the FAIL_TO_PASS / PASS_TO_PASS resolution predicate.)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "InstanceResolution",
    "TestStatus",
    "resolve_instance",
]


class TestStatus(str, Enum):
    """Post-patch status of a single test case (SWE-bench test outcomes)."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True, slots=True)
class InstanceResolution:
    """Whether one SWE-ContextBench instance was resolved, with the failing tests."""

    instance_id: str
    resolved: bool
    fail_to_pass_ok: bool
    pass_to_pass_ok: bool
    unmet_tests: tuple[str, ...]


def _all_passed(
    required: Sequence[str], results: Mapping[str, TestStatus]
) -> tuple[bool, list[str]]:
    """Return whether every required test passed, plus the list that did not."""
    unmet = [name for name in required if results.get(name) is not TestStatus.PASSED]
    return not unmet, unmet


def resolve_instance(
    instance_id: str,
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str],
    test_results: Mapping[str, TestStatus],
) -> InstanceResolution:
    r"""Score one SWE-ContextBench instance as resolved or not.

    Applies the SWE-bench resolution predicate [Jimenez2024]_ (see module docstring):
    resolved iff every ``fail_to_pass`` test and every ``pass_to_pass`` test has
    status :attr:`TestStatus.PASSED` in ``test_results`` (a required test missing
    from the map counts as unmet).

    Parameters
    ----------
    instance_id
        Identifier of the instance (carried through to the result).
    fail_to_pass
        Tests that must transition from failing to passing (must be non-empty: an
        instance with no fail-to-pass test cannot demonstrate resolution).
    pass_to_pass
        Tests that must remain passing (guards against regressions); may be empty.
    test_results
        Post-patch status per test name; tests absent from the map are treated as not
        passing.

    Returns
    -------
    InstanceResolution
        The resolved flag, the two per-set checks, and the sorted list of required
        tests that did not pass (fail-to-pass first, then pass-to-pass).

    Raises
    ------
    ValueError
        If ``fail_to_pass`` is empty.
    """
    if not fail_to_pass:
        raise ValueError(f"instance {instance_id!r} has no fail_to_pass tests to resolve")

    f2p_ok, f2p_unmet = _all_passed(fail_to_pass, test_results)
    p2p_ok, p2p_unmet = _all_passed(pass_to_pass, test_results)
    return InstanceResolution(
        instance_id=instance_id,
        resolved=f2p_ok and p2p_ok,
        fail_to_pass_ok=f2p_ok,
        pass_to_pass_ok=p2p_ok,
        unmet_tests=(*f2p_unmet, *p2p_unmet),
    )
