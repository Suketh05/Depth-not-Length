"""Tests for the capture experiment (paper sec:capture / tab:capexp).

Covers the three storage-condition transforms (determinism, surgical link
strip, shred text conservation / count reversibility), the load-bearing
*mechanism* on a hand-built 2-hop corpus (an intact typed store recovers the
governing rule through its links; the edge-stripped store loses multi-hop
reachability; a lexical arm is unaffected by the strip), and the driver's
paired, well-formed rows plus the tab:capexp-shaped aggregation.

No published tab:capexp cell value is asserted against this code's output:
the mechanism tests assert reachability events on corpora built by hand, and
the only literal goldens are hand-computed (or quoted from the canonical
golden pack with provenance).
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from membench.datasets.capture_conditions import (
    DEFAULT_SIGMA,
    TYPED_LINK_KEYS,
    CaptureCondition,
    apply_condition,
    shred_scattered,
    strip_edges,
)
from membench.datasets.synthetic import load_synthetic_tasks
from membench.metrics.retrieval import score_retrieval
from membench.retrieval.graph.arm import BriefGraphMemory
from membench.types import MemoryItem, Task


def _synthetic_task(depth: int = 2, seed: int = 0) -> Task:
    return load_synthetic_tasks(depths=(depth,), per_depth=1, seed=seed)[0]


def _shredded_fragments(task: Task) -> dict[str, list[MemoryItem]]:
    """Group a Raw-scattered corpus's fragments by the item they were shredded from."""
    groups: dict[str, list[MemoryItem]] = defaultdict(list)
    for item in task.memory_corpus:
        if item.metadata.get("node_type") == "fragment":
            groups[str(item.metadata["shredded_from"])].append(item)
    for fragments in groups.values():
        fragments.sort(key=lambda i: int(i.metadata["fragment_index"]))
    return groups


class TestStripEdges:
    def test_removes_exactly_the_typed_link_keys(self) -> None:
        task = _synthetic_task(depth=3)
        # the CAPTURED corpus really carries links (else the strip is vacuous)
        assert any("edges" in item.metadata for item in task.memory_corpus)
        stripped = strip_edges(task)
        for item in stripped.memory_corpus:
            for key in TYPED_LINK_KEYS:
                assert key not in item.metadata

    def test_preserves_ids_texts_order_and_other_metadata(self) -> None:
        task = _synthetic_task(depth=3)
        stripped = strip_edges(task)
        assert len(stripped.memory_corpus) == len(task.memory_corpus)
        for before, after in zip(task.memory_corpus, stripped.memory_corpus, strict=True):
            assert after.item_id == before.item_id
            assert after.text == before.text  # no text lost
            expected = {k: v for k, v in before.metadata.items() if k not in TYPED_LINK_KEYS}
            assert dict(after.metadata) == expected  # node_type etc. survive

    def test_preserves_every_other_task_field(self) -> None:
        task = _synthetic_task(depth=2)
        stripped = strip_edges(task)
        assert stripped.task_id == task.task_id
        assert stripped.query == task.query
        assert stripped.depth == task.depth
        assert stripped.governing_decisions == task.governing_decisions
        assert stripped.dataset == task.dataset

    def test_deterministic_and_idempotent(self) -> None:
        task = _synthetic_task(depth=3)
        once = strip_edges(task)
        assert once == strip_edges(task)  # pure function
        assert strip_edges(once) == once  # fixed point

    def test_captured_condition_is_identity(self) -> None:
        task = _synthetic_task(depth=2)
        assert apply_condition(task, CaptureCondition.CAPTURED) is task
        # string values dispatch too (the enum is string-valued for row serialisation)
        assert apply_condition(task, "captured") is task
        assert apply_condition(task, "discrete_no_links") == strip_edges(task)

    def test_unknown_condition_rejected(self) -> None:
        task = _synthetic_task(depth=1)
        with pytest.raises(ValueError):
            apply_condition(task, "shredded")  # not one of the three conditions


class TestShredScattered:
    def test_deterministic_with_seed(self) -> None:
        task = _synthetic_task(depth=3)
        a = shred_scattered(task, 3, seed=7)
        b = shred_scattered(task, 3, seed=7)
        assert a == b  # ids, texts, metadata, gold: all identical

    def test_no_text_lost(self) -> None:
        # Reversible-in-count: for every shredded unit, the fragments' payloads
        # joined in fragment_index order reproduce the original whitespace-token
        # sequence exactly. Nothing is dropped, only de-unitised.
        task = _synthetic_task(depth=3)
        originals = {i.item_id: i for i in task.memory_corpus if i.node_type is not None}
        for sigma in (1, 2, 3, 5):
            shredded = shred_scattered(task, sigma, seed=0)
            groups = _shredded_fragments(shredded)
            assert set(groups) == set(originals)  # every unit shredded, none invented
            for item_id, fragments in groups.items():
                joined = " ".join(str(f.metadata["payload"]) for f in fragments)
                assert joined.split() == originals[item_id].text.split()

    def test_reversible_in_count(self) -> None:
        # Each captured unit maps to exactly min(sigma, token_count) fragments,
        # and the corpus size is fillers + sum of fragment counts (count identity).
        task = _synthetic_task(depth=2)
        sigma = 3
        shredded = shred_scattered(task, sigma, seed=0)
        originals = [i for i in task.memory_corpus if i.node_type is not None]
        fillers = [i for i in task.memory_corpus if i.node_type is None]
        groups = _shredded_fragments(shredded)
        expected_total = len(fillers)
        for item in originals:
            expected = min(sigma, len(item.text.split()))
            fragments = groups[item.item_id]
            assert len(fragments) == expected
            assert all(int(f.metadata["fragment_count"]) == expected for f in fragments)
            expected_total += expected
        assert len(shredded.memory_corpus) == expected_total

    def test_fragments_carry_no_typed_links(self) -> None:
        # Raw-scattered = strip AND shred; a fragment must never be traversable.
        shredded = shred_scattered(_synthetic_task(depth=3), 3, seed=0)
        for item in shredded.memory_corpus:
            for key in TYPED_LINK_KEYS:
                assert key not in item.metadata

    def test_gold_remapped_to_all_fragments(self) -> None:
        task = _synthetic_task(depth=2)
        shredded = shred_scattered(task, 3, seed=0)
        (gold_id,) = task.governing_decisions
        expected = tuple(f.item_id for f in _shredded_fragments(shredded)[gold_id])
        assert shredded.governing_decisions == expected
        assert len(expected) == 3  # the rule text is long enough for sigma fragments

    def test_fragments_of_one_decision_land_in_distinct_areas(self) -> None:
        # "split each decision across sigma distractor areas": with sigma at most
        # the 10-area pool size, the sigma fragments occupy sigma distinct areas.
        task = _synthetic_task(depth=3)
        for sigma in (2, 3, 4):
            shredded = shred_scattered(task, sigma, seed=0)
            for fragments in _shredded_fragments(shredded).values():
                areas = [str(f.metadata["scatter_area"]) for f in fragments]
                assert len(set(areas)) == len(areas)

    def test_sigma_one_keeps_whole_payload_in_one_fragment(self) -> None:
        task = _synthetic_task(depth=1)
        shredded = shred_scattered(task, 1, seed=0)
        for item_id, fragments in _shredded_fragments(shredded).items():
            assert len(fragments) == 1
            payload = str(fragments[0].metadata["payload"])
            assert payload.split() == dict(task.corpus_by_id)[item_id].text.split()

    def test_saturation_when_text_shorter_than_sigma(self) -> None:
        # A 3-token decision cannot shed into 5 fragments; it saturates at one
        # fragment per token (and still conserves the text).
        tiny = Task(
            task_id="cap-tiny",
            dataset="synthetic",
            query="anything",
            repo_ref="synthetic://capture",
            memory_corpus=(
                MemoryItem("d1", "use audit logging", metadata={"node_type": "governing"}),
            ),
            governing_decisions=("d1",),
            depth=1,
            spec_variant="stripped",
            scorer="compliance",
        )
        shredded = shred_scattered(tiny, 5, seed=0)
        fragments = _shredded_fragments(shredded)["d1"]
        assert len(fragments) == 3
        assert [str(f.metadata["payload"]) for f in fragments] == ["use", "audit", "logging"]
        assert shredded.governing_decisions == tuple(f.item_id for f in fragments)

    def test_filler_items_pass_through_link_stripped_only(self) -> None:
        task = _synthetic_task(depth=2)
        shredded = shred_scattered(task, 3, seed=0)
        by_id = dict(shredded.corpus_by_id)
        for item in task.memory_corpus:
            if item.node_type is None:
                assert by_id[item.item_id].text == item.text

    def test_rejects_sigma_below_one(self) -> None:
        task = _synthetic_task(depth=1)
        with pytest.raises(ValueError, match="sigma"):
            shred_scattered(task, 0)
        with pytest.raises(ValueError, match="sigma"):
            shred_scattered(task, -2)

    def test_default_sigma_used_by_dispatcher(self) -> None:
        task = _synthetic_task(depth=2)
        via_dispatch = apply_condition(task, CaptureCondition.RAW_SCATTERED, seed=3)
        assert via_dispatch == shred_scattered(task, DEFAULT_SIGMA, seed=3)


def _two_hop_task() -> Task:
    """Hand-built 2-hop chain: query-similar entry -> policy -> dissimilar rule.

    The governing rule shares no content vocabulary with the query; the only
    route to it is entry --derived_from--> policy --constrains--> rule, i.e.
    exactly the "similarity finds the neighbourhood, links find the rule"
    construction of the synthetic dataset (paper sec:capture, thm:struct).
    """
    entry = MemoryItem(
        "entry",
        "The csv export button dashboard analytics surface is where users begin.",
        metadata={"node_type": "entry", "edges": [("policy", "derived_from")]},
    )
    policy = MemoryItem(
        "policy",
        "Retention policy layer, refining an upstream constraint it derives from.",
        metadata={"node_type": "policy", "edges": [("rule", "constrains")]},
    )
    rule = MemoryItem(
        "rule",
        "RULE: every operation in this path must call withAuditLog() before proceeding.",
        metadata={"node_type": "governing"},
    )
    fillers = tuple(
        MemoryItem(f"filler-{i}", text)
        for i, text in enumerate(
            (
                "telemetry pipeline ingestion buffer",
                "localization string catalogue plural",
                "infrastructure terraform module registry",
            )
        )
    )
    return Task(
        task_id="cap-2hop",
        dataset="synthetic",
        query="Add a csv export button to the analytics dashboard so users can download data.",
        repo_ref="synthetic://capture",
        memory_corpus=(entry, policy, rule, *fillers),
        governing_decisions=("rule",),
        depth=2,
        spec_variant="stripped",
        scorer="compliance",
    )


class TestTwoHopMechanism:
    """The mechanism claim of tab:capexp, on a hand-built case (no headline numbers).

    Conclusion (i) of sec:capconc: the typed store's advantage is the
    traversable links, not item discreteness. Verified here as a reachability
    event: the identical arm, identical query, identical budget recovers the
    governing rule if and only if the typed edges are present.
    """

    BUDGET = 150  # generous: both items fit; the contrast is reachability, not packing

    def _retrieve_ids(self, task: Task) -> tuple[str, ...]:
        # seeds=1 makes the test sharp: the query-similar entry node is the only
        # similarity seed, so anything else retrieved arrived through a link.
        arm = BriefGraphMemory(max_hops=3, seeds=1)
        arm.write(task.memory_corpus)
        return arm.retrieve(task.query, self.BUDGET).ids

    def test_intact_store_recovers_the_rule_through_links(self) -> None:
        ids = self._retrieve_ids(_two_hop_task())
        assert "rule" in ids  # 2-hop reachability via derived_from -> constrains
        score = score_retrieval(ids, ("rule",))
        assert score.recall == 1.0
        assert score.chain_recovered is True

    def test_stripped_store_loses_multi_hop_reachability(self) -> None:
        # Same arm, same query, same budget; only the edges are gone. The rule
        # shares no vocabulary with the query, so similarity cannot reach it.
        ids = self._retrieve_ids(strip_edges(_two_hop_task()))
        assert "rule" not in ids
        score = score_retrieval(ids, ("rule",))
        assert score.recall == 0.0
        assert score.chain_recovered is False

    def test_entry_node_is_still_found_after_the_strip(self) -> None:
        # The strip removes reachability, not the similarity seed: the arm still
        # finds the query-similar entry, it just cannot walk onward from it.
        ids = self._retrieve_ids(strip_edges(_two_hop_task()))
        assert "entry" in ids

    def test_lexical_arm_is_unaffected_by_the_strip(self) -> None:
        # The paper's surgical-manipulation check: "the lexical blocks bm25,
        # tfidf, and dense are identical across CAPTURED and Discrete-no-links
        # ... removing the edges does nothing to an arm that never read them"
        # (sec:capture). Asserted as exact per-task retrieved-id equality.
        from membench.retrieval.bm25 import BM25Memory
        from membench.retrieval.tfidf import TfidfMemory

        for factory in (BM25Memory, TfidfMemory):
            for task in load_synthetic_tasks(depths=(1, 2, 3), per_depth=2, seed=0):
                captured_arm = factory()
                captured_arm.write(task.memory_corpus)
                stripped_arm = factory()
                stripped_arm.write(strip_edges(task).memory_corpus)
                assert (
                    captured_arm.retrieve(task.query, self.BUDGET).ids
                    == stripped_arm.retrieve(task.query, self.BUDGET).ids
                )


class TestAssemblySemantics:
    """Raw-scattered recall is an assembly fraction (eq:scatter / thm:scatter)."""

    def test_recall_is_fraction_of_fragments_assembled(self) -> None:
        # Hand-computed: gold = 4 fragments, 2 retrieved => recall 2/4 = 0.5,
        # and full-chain recovery requires all four.
        gold = ("d::frag0", "d::frag1", "d::frag2", "d::frag3")
        score = score_retrieval(("d::frag0", "x", "d::frag2"), gold)
        assert score.recall == 0.5
        assert score.chain_recovered is False
        assert score_retrieval(gold, gold).chain_recovered is True

    def test_fragment_count_is_the_sigma_of_the_scatter_law(self) -> None:
        # The shred's fragment_count is the sigma that theory/scatter.py raises
        # p to. GOLDEN (quoted verbatim from the canonical golden pack, A.5,
        # provenance tests/test_theory_scatter.py / paper eq:scatter):
        # scatter_factor(0.5, 3) = 0.5**3 = 0.125 (exact 1/8).
        from membench.theory.scatter import scatter_factor

        task = _synthetic_task(depth=2)
        shredded = shred_scattered(task, 3, seed=0)
        (first_gold, *_rest) = shredded.governing_decisions
        sigma = int(dict(shredded.corpus_by_id)[first_gold].metadata["fragment_count"])
        assert sigma == 3
        assert scatter_factor(0.5, sigma) == 0.125  # exact 1/8
