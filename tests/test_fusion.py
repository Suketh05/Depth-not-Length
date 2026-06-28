"""Tests for the CombSUM / CombMNZ / CombANZ score-fusion arm.

The golden values are hand-computed from two rankers with given raw scores, so the
assertions are independent of the implementation under test (no circularity).

Ranker A raw scores: {a: 10, b: 4, d: 7, c: 0}  -> min 0, max 10, range 10
    min-max normalised:  a = 10/10 = 1.0
                         b =  4/10 = 0.4
                         d =  7/10 = 0.7
                         c =  0/10 = 0.0

Ranker B raw scores: {b: 4, d: 11, c: 1}        -> min 1, max 11, range 10
    min-max normalised:  b = (4-1)/10 = 0.3
                         d = (11-1)/10 = 1.0
                         c = (1-1)/10 = 0.0
    (document ``a`` is absent from ranker B -> contributes 0, z-count not incremented)

z(d) = number of rankers with a strictly positive normalised score:
    a -> 1 (A only)   b -> 2 (A and B)   c -> 0   d -> 2 (A and B)

CombSUM = sum of normalised:
    a = 1.0 + 0.0 = 1.0
    b = 0.4 + 0.3 = 0.7
    c = 0.0 + 0.0 = 0.0
    d = 0.7 + 1.0 = 1.7

CombMNZ = CombSUM * z:
    a = 1.0 * 1 = 1.0
    b = 0.7 * 2 = 1.4
    c = 0.0 * 0 = 0.0
    d = 1.7 * 2 = 3.4

CombANZ = CombSUM / z  (0 when z = 0):
    a = 1.0 / 1 = 1.0
    b = 0.7 / 2 = 0.35
    c = 0.0     = 0.0
    d = 1.7 / 2 = 0.85

Resulting orders (ties broken by first-seen order a, b, d, c):
    CombSUM:  d (1.7) > a (1.0) > b (0.7) > c (0.0)  -> [d, a, b, c]
    CombMNZ:  d (3.4) > b (1.4) > a (1.0) > c (0.0)  -> [d, b, a, c]   (b and a swap)
    CombANZ:  a (1.0) > d (0.85) > b (0.35) > c (0.0) -> [a, d, b, c]
"""

from __future__ import annotations

import pytest

from membench.retrieval import fusion  # noqa: F401  (registration side effect)
from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.retrieval.fusion import (
    FusionMethod,
    fuse_scores,
    min_max_normalize,
)
from membench.types import MemoryItem

# Two rankers' raw scores (see module docstring for the hand-computed golden).
_RANKER_A = {"a": 10.0, "b": 4.0, "d": 7.0, "c": 0.0}
_RANKER_B = {"b": 4.0, "d": 11.0, "c": 1.0}


def _order(fused: dict[str, float]) -> list[str]:
    """Reproduce the arm's stable rank: sort by -score, ties by first-seen order."""
    return sorted(fused, key=lambda item_id: -fused[item_id])


class TestMinMaxNormalize:
    def test_normalises_to_unit_range(self) -> None:
        norm = min_max_normalize(_RANKER_A)
        assert norm["a"] == pytest.approx(1.0)
        assert norm["b"] == pytest.approx(0.4)
        assert norm["d"] == pytest.approx(0.7)
        assert norm["c"] == pytest.approx(0.0)

    def test_constant_ranker_maps_to_zero(self) -> None:
        # All-equal scores carry no signal -> every document normalises to 0.0.
        assert min_max_normalize({"x": 5.0, "y": 5.0}) == {"x": 0.0, "y": 0.0}

    def test_empty(self) -> None:
        assert min_max_normalize({}) == {}


class TestFuseScoresGolden:
    def test_combsum_values(self) -> None:
        fused = fuse_scores([_RANKER_A, _RANKER_B], FusionMethod.COMBSUM)
        assert fused["a"] == pytest.approx(1.0)
        assert fused["b"] == pytest.approx(0.7)
        assert fused["c"] == pytest.approx(0.0)
        assert fused["d"] == pytest.approx(1.7)

    def test_combmnz_values(self) -> None:
        fused = fuse_scores([_RANKER_A, _RANKER_B], FusionMethod.COMBMNZ)
        assert fused["a"] == pytest.approx(1.0)
        assert fused["b"] == pytest.approx(1.4)
        assert fused["c"] == pytest.approx(0.0)
        assert fused["d"] == pytest.approx(3.4)

    def test_combanz_values(self) -> None:
        fused = fuse_scores([_RANKER_A, _RANKER_B], FusionMethod.COMBANZ)
        assert fused["a"] == pytest.approx(1.0)
        assert fused["b"] == pytest.approx(0.35)
        assert fused["c"] == pytest.approx(0.0)
        assert fused["d"] == pytest.approx(0.85)

    def test_default_method_is_combsum(self) -> None:
        assert fuse_scores([_RANKER_A, _RANKER_B]) == fuse_scores(
            [_RANKER_A, _RANKER_B], FusionMethod.COMBSUM
        )

    def test_combsum_order(self) -> None:
        assert _order(fuse_scores([_RANKER_A, _RANKER_B], FusionMethod.COMBSUM)) == [
            "d",
            "a",
            "b",
            "c",
        ]

    def test_combmnz_reorders_multi_ranker_doc_above_single(self) -> None:
        # The research point: b (found by BOTH rankers) overtakes a (one strong
        # ranker) under CombMNZ even though a outranks b under CombSUM.
        assert _order(fuse_scores([_RANKER_A, _RANKER_B], FusionMethod.COMBMNZ)) == [
            "d",
            "b",
            "a",
            "c",
        ]

    def test_combanz_order(self) -> None:
        assert _order(fuse_scores([_RANKER_A, _RANKER_B], FusionMethod.COMBANZ)) == [
            "a",
            "d",
            "b",
            "c",
        ]


_CORPUS = [
    MemoryItem("D-1", "all data exports must call withAuditLog for SOC-2 compliance"),
    MemoryItem("D-2", "use DateRangePicker for date range selection in the UI"),
    MemoryItem("D-3", "every export endpoint requires an audit log entry before streaming"),
    MemoryItem("D-4", "primary actions use the Button component variant"),
]


def _arm(name: str = "comb_sum", **kw: object) -> object:
    arm = build_arm(name, **kw)
    arm.write(_CORPUS)  # type: ignore[attr-defined]
    return arm


class TestScoreFusionArm:
    @pytest.mark.parametrize(
        ("name", "method"),
        [
            ("comb_sum", FusionMethod.COMBSUM),
            ("comb_mnz", FusionMethod.COMBMNZ),
            ("comb_anz", FusionMethod.COMBANZ),
        ],
    )
    def test_registered_as_hybrid(self, name: str, method: FusionMethod) -> None:
        spec = get_spec(name)
        assert spec.capabilities.family is ArmFamily.HYBRID
        assert spec.capabilities.uses_embeddings
        assert not spec.capabilities.follows_links

    def test_builds_with_no_args(self) -> None:
        # The harness calls build_arm(name) with no kwargs.
        for name in ("comb_sum", "comb_mnz", "comb_anz"):
            assert build_arm(name) is not None

    def test_fuses_to_relevant_items(self) -> None:
        ctx = _arm().retrieve("audit log on data export", 10_000)  # type: ignore[attr-defined]
        assert set(ctx.ids[:2]) == {"D-1", "D-3"}

    def test_ranks_all_items(self) -> None:
        ctx = _arm().retrieve("audit", 10_000)  # type: ignore[attr-defined]
        assert set(ctx.ids) == {"D-1", "D-2", "D-3", "D-4"}

    def test_deterministic(self) -> None:
        assert _arm().retrieve("audit export", 10_000).ids == (  # type: ignore[attr-defined]
            _arm().retrieve("audit export", 10_000).ids  # type: ignore[attr-defined]
        )

    def test_empty_corpus(self) -> None:
        assert build_arm("comb_sum").retrieve("x", 1000).ids == ()

    def test_respects_budget(self) -> None:
        assert _arm().retrieve("audit", 0).ids == ()  # type: ignore[attr-defined]
