"""Tests for the LoCoMo multi-session conversational-memory loader.

The fixture is a tiny three-session conversation built under ``tmp_path`` -- no
external download. Golden depth values are hand-computed from the depth rule

    d = (max evidence session) - (min evidence session) + 1

(see the module docstring); they are NOT read back from the loader's own output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench.datasets.locomo import gold_answers, load_locomo_tasks
from membench.retrieval import builtin  # noqa: F401  # populate the arm REGISTRY
from membench.retrieval.base import build_arm
from membench.types import Scorer, SpecVariant

# --- Tiny in-test fixture --------------------------------------------------
# Three sessions; QA evidence chosen so the session SPAN (hence depth) is known
# by hand:
#   q0 multi-hop, evidence {D1:1, D3:2} -> sessions {1, 3} -> d = 3 - 1 + 1 = 3
#   q1 single-hop, evidence {D3:1}      -> sessions {3}    -> d = 3 - 3 + 1 = 1
#   q2 multi-hop, evidence {D1:1, D2:2} -> sessions {1, 2} -> d = 2 - 1 + 1 = 2
#   q3 adversarial (category 5), no evidence -> dropped
_SAMPLE = {
    "sample_id": "tiny",
    "conversation": {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "session_1_date_time": "1:00 pm on 5 May, 2023",
        "session_1": [
            {
                "speaker": "Alice",
                "text": "I just adopted a beagle puppy named Cooper.",
                "dia_id": "D1:1",
            },
            {"speaker": "Bob", "text": "Congratulations on the new puppy!", "dia_id": "D1:2"},
        ],
        "session_2_date_time": "9:00 am on 2 June, 2023",
        "session_2": [
            {"speaker": "Bob", "text": "How is the new job going?", "dia_id": "D2:1"},
            {"speaker": "Alice", "text": "Busy, lots of meetings every day.", "dia_id": "D2:2"},
        ],
        "session_3_date_time": "6:00 pm on 14 August, 2023",
        "session_3": [
            {
                "speaker": "Alice",
                "text": "Cooper just learned to fetch the newspaper.",
                "dia_id": "D3:1",
            },
            {"speaker": "Bob", "text": "That beagle is a smart dog!", "dia_id": "D3:2"},
        ],
    },
    "qa": [
        {
            "question": "What breed is Alice's dog Cooper?",
            "answer": "beagle",
            "evidence": ["D1:1", "D3:2"],
            "category": 2,
        },
        {
            "question": "What did Alice say her dog Cooper learned to do?",
            "answer": "fetch the newspaper",
            "evidence": ["D3:1"],
            "category": 1,
        },
        {
            "question": "What was Alice doing around when she got her dog?",
            "answer": "starting a busy new job",
            "evidence": ["D1:1", "D2:2"],
            "category": 2,
        },
        {
            "question": "What is the name of Alice's cat?",
            "answer": "Not mentioned.",
            "evidence": [],
            "category": 5,
        },
    ],
}


def _write_fixture(tmp_path: Path, sample: object = _SAMPLE) -> Path:
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps([sample]))
    return path


class TestDepthGolden:
    def test_multihop_depth_equals_session_span(self, tmp_path: Path) -> None:
        tasks = load_locomo_tasks(_write_fixture(tmp_path))
        by_id = {t.task_id: t for t in tasks}
        # Hand-computed golden: evidence {D1:1, D3:2} -> sessions {1,3} -> d = 3.
        assert by_id["locomo-tiny-q0"].depth == 3
        # Single-session evidence is depth 1; intermediate gap is depth 2.
        assert by_id["locomo-tiny-q1"].depth == 1
        assert by_id["locomo-tiny-q2"].depth == 2

    def test_depth_increases_with_session_gap(self, tmp_path: Path) -> None:
        by_id = {t.task_id: t for t in load_locomo_tasks(_write_fixture(tmp_path))}
        # Earliest evidence fixed at session 1; latest moves 1 -> 2 -> 3, so the
        # session gap (and depth) is strictly monotonic.
        assert by_id["locomo-tiny-q1"].depth < by_id["locomo-tiny-q2"].depth
        assert by_id["locomo-tiny-q2"].depth < by_id["locomo-tiny-q0"].depth


class TestStructuralInvariants:
    def test_qa_match_scorer_and_no_repo(self, tmp_path: Path) -> None:
        for t in load_locomo_tasks(_write_fixture(tmp_path)):
            assert t.dataset == "locomo"
            assert t.scorer is Scorer.QA_MATCH
            assert t.repo_ref is None
            assert not t.is_coding
            assert t.spec_variant is SpecVariant.FULL

    def test_governing_decisions_are_evidence_present_in_corpus(self, tmp_path: Path) -> None:
        by_id = {t.task_id: t for t in load_locomo_tasks(_write_fixture(tmp_path))}
        # Task references exactly the gold evidence turns, all in its corpus.
        assert by_id["locomo-tiny-q0"].governing_decisions == ("D1:1", "D3:2")
        for t in by_id.values():
            assert t.governing_decisions
            for dia_id in t.governing_decisions:
                assert dia_id in t.corpus_by_id

    def test_corpus_is_every_turn_keyed_by_dia_id(self, tmp_path: Path) -> None:
        tasks = load_locomo_tasks(_write_fixture(tmp_path))
        corpus = tasks[0].memory_corpus
        # 3 sessions x 2 turns = 6 dialogue turns, one MemoryItem each.
        assert len(corpus) == 6
        assert {item.item_id for item in corpus} == {"D1:1", "D1:2", "D2:1", "D2:2", "D3:1", "D3:2"}
        assert all(item.node_type == "turn" for item in corpus)

    def test_adversarial_questions_dropped(self, tmp_path: Path) -> None:
        ids = {t.task_id for t in load_locomo_tasks(_write_fixture(tmp_path))}
        # q3 is category-5 adversarial with empty evidence -> no task emitted.
        assert ids == {"locomo-tiny-q0", "locomo-tiny-q1", "locomo-tiny-q2"}


class TestFilteringAndValidation:
    def test_category_filter(self, tmp_path: Path) -> None:
        single_hop = load_locomo_tasks(_write_fixture(tmp_path), categories=[1])
        assert {t.task_id for t in single_hop} == {"locomo-tiny-q1"}

    def test_missing_evidence_raises(self, tmp_path: Path) -> None:
        broken = json.loads(json.dumps(_SAMPLE))
        broken["qa"][1]["evidence"] = ["D9:9"]  # turn absent from the conversation
        with pytest.raises(ValueError, match="evidence not in conversation"):
            load_locomo_tasks(_write_fixture(tmp_path, broken))


class TestRetrievalReachesEvidence:
    def test_lexical_arm_ranks_evidence_turn_first(self, tmp_path: Path) -> None:
        # An arm must be able to pull the gold evidence turn out of the transcript.
        task = next(
            t for t in load_locomo_tasks(_write_fixture(tmp_path)) if t.task_id == "locomo-tiny-q1"
        )
        arm = build_arm("bm25")
        arm.write(task.memory_corpus)
        ctx = arm.retrieve(task.query, budget_tokens=150)
        # Gold evidence is D3:1 ("Cooper just learned to fetch the newspaper"),
        # which the query overlaps most strongly -> retrieved and ranked top.
        assert "D3:1" in ctx.ids
        assert ctx.ids[0] == "D3:1"


class TestGoldAnswers:
    def test_gold_answers_align_with_tasks(self, tmp_path: Path) -> None:
        path = _write_fixture(tmp_path)
        gold = gold_answers(path)
        task_ids = {t.task_id for t in load_locomo_tasks(path)}
        assert set(gold) == task_ids
        assert gold["locomo-tiny-q0"] == "beagle"
        assert all(isinstance(v, str) and v for v in gold.values())
