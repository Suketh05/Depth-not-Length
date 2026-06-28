"""Tests for the deterministic SPLADE-style sparse-expansion arm.

The golden values below are derived BY HAND from the algorithm in
``splade_lite.py`` (saturated log-tf weights + co-occurrence expansion, scored by
sparse dot product), independently of running the implementation.

Toy corpus (term sets after tokenisation)::

    D-1 "audit log export"  -> {audit, log, export}
    D-2 "audit log review"  -> {audit, log, review}
    D-3 "export report"     -> {export, report}

Document frequencies (df = #docs containing the term)::

    audit=2, log=2, export=2, review=1, report=1

Document-level co-occurrence (#docs containing both)::

    (audit,log)=2  (audit,export)=1  (audit,review)=1
    (log,export)=1 (log,review)=1    (export,report)=1
    all other pairs = 0

Let L = log(1 + 1) = ln 2 (every term appears once per document/query).
EXPANSION_FACTOR beta = 0.5.

Query "audit" -> base {audit: L}. Expansion of an absent term e:
    expand_e = beta * L * cooc(audit, e) / df(audit),   df(audit) = 2
    log:    0.5 * L * 2/2 = 0.5 L
    export: 0.5 * L * 1/2 = 0.25 L
    review: 0.5 * L * 1/2 = 0.25 L
=> q = {audit: L, log: 0.5 L, export: 0.25 L, review: 0.25 L}

Document D-1 {audit, log, export} (all base L). Expansion of absent terms:
    review = beta * (L*cooc(audit,review)/2 + L*cooc(log,review)/2 + L*cooc(export,review)/2)
           = 0.5 * (L*1/2 + L*1/2 + 0) = 0.5 L
    report = beta * (0 + 0 + L*cooc(export,report)/2) = 0.5 * (L*1/2) = 0.25 L
=> d1 = {audit: L, log: L, export: L, review: 0.5 L, report: 0.25 L}

Dot product q . d1 over shared terms {audit, log, export, review}::

    L*L + 0.5L*L + 0.25L*L + 0.25L*0.5L
      = (1 + 0.5 + 0.25 + 0.125) * L^2
      = 1.875 * (ln 2)^2
      ~= 0.9008494010966277
"""

from __future__ import annotations

import math

from membench.retrieval import splade_lite  # noqa: F401  (registration side effect)
from membench.retrieval.base import ArmFamily, build_arm, get_spec
from membench.retrieval.splade_lite import SpladeLiteMemory
from membench.types import MemoryItem

_CORPUS = [
    MemoryItem("D-1", "audit log export"),
    MemoryItem("D-2", "audit log review"),
    MemoryItem("D-3", "export report"),
]

_L = math.log(2.0)  # log(1 + 1) for a single occurrence
# Hand-computed golden: q . d1 = 1.875 * (ln 2)^2 (see module docstring).
_GOLDEN_DOT = 1.875 * _L * _L


def _arm() -> SpladeLiteMemory:
    arm = SpladeLiteMemory()
    arm.write(_CORPUS)
    return arm


class TestSpladeLite:
    def test_registered_as_lexical(self) -> None:
        assert get_spec("splade_lite").capabilities.family is ArmFamily.LEXICAL

    def test_constructible_with_no_args(self) -> None:
        # The harness calls build_arm("splade_lite") with no kwargs.
        assert isinstance(build_arm("splade_lite"), SpladeLiteMemory)

    def test_query_vector_matches_hand_values(self) -> None:
        q = _arm().sparse_vector("audit")
        assert q.keys() == {"audit", "log", "export", "review"}
        assert math.isclose(q["audit"], _L)
        assert math.isclose(q["log"], 0.5 * _L)
        assert math.isclose(q["export"], 0.25 * _L)
        assert math.isclose(q["review"], 0.25 * _L)

    def test_document_vector_matches_hand_values(self) -> None:
        d1 = _arm().sparse_vector("audit log export")
        assert d1.keys() == {"audit", "log", "export", "review", "report"}
        assert math.isclose(d1["audit"], _L)
        assert math.isclose(d1["log"], _L)
        assert math.isclose(d1["export"], _L)
        assert math.isclose(d1["review"], 0.5 * _L)
        assert math.isclose(d1["report"], 0.25 * _L)

    def test_query_doc_dot_product_is_golden(self) -> None:
        arm = _arm()
        from membench.retrieval.splade_lite import _sparse_dot

        dot = _sparse_dot(arm.sparse_vector("audit"), arm.sparse_vector("audit log export"))
        assert math.isclose(dot, _GOLDEN_DOT)
        assert math.isclose(dot, 0.9008494010966277, rel_tol=1e-12)

    def test_expansion_adds_expected_cooccurring_term(self) -> None:
        # "audit" never contains "log", but the two co-occur in D-1 and D-2,
        # so the expansion MUST add "log" to the query vector.
        q = _arm().sparse_vector("audit")
        assert "log" not in {"audit"}  # "log" is not a literal query term
        assert "log" in q
        assert q["log"] > 0.0

    def test_saturation_is_sublinear(self) -> None:
        # log(1 + tf): two occurrences weigh less than twice one occurrence.
        arm = _arm()
        once = arm.sparse_vector("zzz")["zzz"]  # out-of-corpus term, no expansion
        twice = arm.sparse_vector("zzz zzz")["zzz"]
        assert math.isclose(once, math.log(2.0))
        assert math.isclose(twice, math.log(3.0))
        assert twice < 2.0 * once

    def test_ranks_by_sparse_dot_product(self) -> None:
        # q . d1 = q . d2 = 1.875 L^2 (tie, broken by corpus order); q . d3 = 0.625 L^2.
        ctx = _arm().retrieve("audit", 10_000)
        assert ctx.ids == ("D-1", "D-2", "D-3")

    def test_deterministic(self) -> None:
        assert _arm().retrieve("audit log", 10_000).ids == _arm().retrieve("audit log", 10_000).ids

    def test_empty_corpus(self) -> None:
        assert SpladeLiteMemory().retrieve("audit", 1000).ids == ()

    def test_empty_query_is_safe(self) -> None:
        # No query terms -> all-zero scores -> corpus order, no crash.
        ctx = _arm().retrieve("", 10_000)
        assert ctx.ids == ("D-1", "D-2", "D-3")

    def test_respects_budget(self) -> None:
        assert _arm().retrieve("audit", 0).ids == ()
