"""Tests for the personalized PageRank retrieval arm.

The golden test pins the stationary vector of a hand-solvable 3-node walk against
its closed-form solution (computed below, independently of the implementation), and
the structural tests assert the PageRank invariants: the mass is a probability
distribution and the power iteration converges.
"""

from __future__ import annotations

import numpy as np
import pytest

from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.retrieval.ppr import (
    PersonalizedPageRankMemory,
    personalized_pagerank,
)
from membench.types import MemoryItem


class TestRegistration:
    def test_ppr_registered_as_link_following_graph_arm(self) -> None:
        spec = get_spec("ppr")
        assert spec.capabilities.family is ArmFamily.GRAPH
        assert spec.capabilities.follows_links is True
        assert spec.capabilities.uses_embeddings is True

    def test_buildable_with_no_args(self) -> None:
        # The harness calls build_arm(name) with no kwargs.
        arm = build_arm("ppr")
        assert isinstance(arm, PersonalizedPageRankMemory)


class TestGoldenStationaryVector:
    r"""Closed-form golden for ``r = (1 - alpha) P^T r + alpha s``.

    Take the row-stochastic transition matrix (no dangling rows)::

        P = [[0.0, 0.5, 0.5],
             [0.0, 0.0, 1.0],
             [1.0, 0.0, 0.0]]

    teleport ``alpha = 0.2`` (so ``1 - alpha = 0.8``) and the personalization
    ``s = [1, 0, 0]``. With ``P^T r`` written out, the fixed point satisfies::

        r0 = 0.8 * r2          + 0.2
        r1 = 0.8 * (0.5 r0)        = 0.4 r0
        r2 = 0.8 * (0.5 r0 + r1)   = 0.4 r0 + 0.8 r1

    Substituting r1 = 0.4 r0 gives r2 = 0.72 r0, then
    r0 = 0.8 * 0.72 r0 + 0.2  =>  0.424 r0 = 0.2  =>  r0 = 0.2 / 0.424 = 25/53.
    Hence the exact stationary vector is

        r = [25/53, 10/53, 18/53] = [0.4716981..., 0.1886792..., 0.3396226...]

    which sums to (25 + 10 + 18)/53 = 1. These rationals are derived purely by
    hand; the implementation is never consulted to produce them.
    """

    TRANSITION = np.array([[0.0, 0.5, 0.5], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    RESTART = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    GOLDEN = np.array([25.0 / 53.0, 10.0 / 53.0, 18.0 / 53.0], dtype=np.float64)

    def test_matches_closed_form_solution(self) -> None:
        result = personalized_pagerank(
            self.TRANSITION, self.RESTART, alpha=0.2, max_iter=1000, tol=1e-15
        )
        assert result.converged is True
        np.testing.assert_allclose(result.mass, self.GOLDEN, atol=1e-9)

    def test_mass_is_a_probability_distribution(self) -> None:
        result = personalized_pagerank(self.TRANSITION, self.RESTART, alpha=0.2)
        assert pytest.approx(float(result.mass.sum()), abs=1e-9) == 1.0
        assert np.all(result.mass >= 0.0)

    def test_satisfies_the_fixed_point_equation(self) -> None:
        # Independent check: the returned vector is a fixed point of the iteration.
        alpha = 0.2
        result = personalized_pagerank(
            self.TRANSITION, self.RESTART, alpha=alpha, max_iter=1000, tol=1e-15
        )
        recomputed = (1.0 - alpha) * (self.TRANSITION.T @ result.mass) + alpha * self.RESTART
        np.testing.assert_allclose(recomputed, result.mass, atol=1e-9)


class TestInvariants:
    def test_restart_is_l1_normalised(self) -> None:
        # An unnormalised, non-negative restart vector is normalised internally.
        transition = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        result = personalized_pagerank(transition, np.array([3.0, 1.0]), alpha=0.3)
        assert pytest.approx(float(result.mass.sum()), abs=1e-9) == 1.0

    def test_dangling_node_mass_is_conserved(self) -> None:
        # Node 1 is dangling (all-zero row); its mass must be reinjected, so the
        # stationary vector still sums to 1 rather than leaking away.
        transition = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
        result = personalized_pagerank(
            transition, np.array([0.5, 0.5]), alpha=0.2, max_iter=1000, tol=1e-15
        )
        assert pytest.approx(float(result.mass.sum()), abs=1e-9) == 1.0
        assert np.all(result.mass >= 0.0)

    def test_no_edges_collapses_to_restart(self) -> None:
        # Every node dangling -> r = s exactly (graceful degradation to similarity).
        transition = np.zeros((3, 3), dtype=np.float64)
        restart = np.array([0.6, 0.3, 0.1], dtype=np.float64)
        result = personalized_pagerank(transition, restart, alpha=0.15)
        np.testing.assert_allclose(result.mass, restart, atol=1e-9)

    def test_converges_within_tolerance(self) -> None:
        transition = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]], dtype=np.float64)
        restart = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        result = personalized_pagerank(transition, restart, alpha=0.15, tol=1e-12)
        assert result.converged is True
        assert result.iterations < 200

    @pytest.mark.parametrize("bad_alpha", [-0.1, 1.5])
    def test_rejects_out_of_range_alpha(self, bad_alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            personalized_pagerank(
                np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([1.0, 0.0]), alpha=bad_alpha
            )

    def test_rejects_non_row_stochastic_transition(self) -> None:
        # A row summing to 0.5 is neither stochastic nor dangling -> rejected.
        with pytest.raises(ValueError, match="transition rows"):
            personalized_pagerank(
                np.array([[0.5, 0.0], [1.0, 0.0]]), np.array([1.0, 0.0]), alpha=0.2
            )


# A governing decision (D-1) whose text shares no vocabulary with the query, reached
# only through a CONSTRAINS edge from the query-similar justification (D-1-rationale).
_DEPTH_CORPUS = [
    MemoryItem(
        "D-1",
        "primary buttons must use the shared Button component",
        metadata={"node_type": "constraint"},
    ),
    MemoryItem(
        "D-1-rationale",
        "the export dialog call to action should be obvious and accessible to users",
        metadata={"node_type": "justification", "constrains": "D-1"},
    ),
    MemoryItem("D-2", "tooltips use the Popover primitive"),
    MemoryItem("D-3", "dates use the DateRangePicker widget"),
]
_QUERY = "make the export dialog call to action accessible"


class TestArmEndToEnd:
    def test_recovers_governing_decision_through_link(self) -> None:
        arm = build_arm("ppr")
        arm.write(_DEPTH_CORPUS)  # type: ignore[attr-defined]
        ids = arm.retrieve(_QUERY, budget_tokens=10_000).ids  # type: ignore[attr-defined]
        # PageRank mass flows from the query-similar justification along its
        # CONSTRAINS edge to the governing decision, so D-1 is surfaced near the top.
        assert "D-1-rationale" in ids
        assert "D-1" in ids
        assert ids.index("D-1") <= 1

    def test_empty_corpus_returns_empty(self) -> None:
        arm = build_arm("ppr")
        ctx = arm.retrieve(_QUERY, budget_tokens=10_000)  # type: ignore[attr-defined]
        assert ctx.ids == ()
        assert ctx.text == ""

    def test_flat_corpus_ranks_most_similar_first(self) -> None:
        # No edges -> restart (similarity) ranking stands; the relevant item leads.
        arm = build_arm("ppr")
        flat = [MemoryItem("A", "audit log on export"), MemoryItem("B", "button colours")]
        arm.write(flat)  # type: ignore[attr-defined]
        ids = arm.retrieve("audit log export", 10_000).ids  # type: ignore[attr-defined]
        assert ids[0] == "A"
