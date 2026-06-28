"""Configurable supersession-chain dataset: the recency / knowledge-update test.

A *supersession chain* models a single decision slot whose value is overwritten
repeatedly over time: decision ``v0`` is later superseded by ``v1``, which is
superseded by ``v2``, ... up to the head ``v_{k-1}``. Only the head is currently
in force; every earlier link is stale. The agent must answer with the *latest*
decision, not whichever revision a similarity search happens to surface first --
old and new revisions look almost identical, so lexical/dense retrieval has no
way to tell which one is current, whereas a link-following memory walks the
typed ``supersedes`` edges straight to the head.

Formally, the corpus induces a strict linear order under the typed ``supersedes``
relation, ``v_i ──supersedes──▶ v_{i-1}``. The currently valid answer is the
unique maximal element of that order -- the head of the chain, i.e. the single
decision node with in-degree ``0`` under ``supersedes`` (no later decision
supersedes it)::

    valid(slot) = argmax_i { v_i : v_i is a decision on slot }
                = head of the supersession chain
                = v_{k-1}            (k = chain_len)

This isolates the "knowledge update" competency: among temporally ordered
assertions about the same fact, the most recent valid one wins. It is the
benchmark analogue of (a) valid-time / "as-of-now" semantics in temporal
databases, where the currently valid row is the most recent; (b) AGM belief
revision, where an incoming belief overrides the old one; and (c) the
``Superseded by`` status link of Architecture Decision Records.

The generator mirrors :mod:`membench.datasets.synthetic`: it is fully offline,
deterministic under ``seed`` (a single :class:`random.Random` drives slot
rotation and corpus shuffling), needs no external data download, and emits plain
:class:`~membench.types.Task` objects so the flat harness loop can run any arm
over it (see :func:`membench.harness.run_benchmark` / the lower-level
``build_system`` + ``run_task`` usage).

References
----------
.. [1] Alchourrón, C. E., Gärdenfors, P., & Makinson, D. (1985). "On the Logic
       of Theory Change: Partial Meet Contraction and Revision Functions."
       *The Journal of Symbolic Logic*, 50(2), 510-530. (Belief revision: an
       incoming belief supersedes the prior one.)
.. [2] Snodgrass, R. T. (1999). *Developing Time-Oriented Database Applications
       in SQL.* Morgan Kaufmann. (Valid-time recency: the currently valid fact
       is the most recent assertion about the slot.)
.. [3] Wu, D., et al. (2024). "LongMemEval: Benchmarking Chat Assistants on
       Long-Term Interactive Memory." arXiv:2410.10813. (The "knowledge update"
       ability this generator isolates.)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from membench.types import MemoryItem, Scorer, SpecVariant, Task

__all__ = ["SLOTS", "Slot", "generate_supersession_tasks"]

# The typed relation a newer decision carries back to the one it replaces. This
# string is the value of ``membench.retrieval.graph.store.RelationshipType.SUPERSEDES``,
# so the structured-graph arm parses these edges directly.
SUPERSEDES = "supersedes"


@dataclass(frozen=True, slots=True)
class Slot:
    """A decision slot whose value is revised over a supersession chain.

    Parameters
    ----------
    slug
        Short identifier used to build stable node ids.
    subject
        Natural-language noun phrase naming what the decision governs (used in
        the query and the node text).
    values
        Successive code-identifier values the slot takes, oldest first. Revision
        ``k`` of a chain uses ``values[k % len(values)]`` so chains longer than
        the pool simply cycle, while the head value stays deterministic.
    """

    slug: str
    subject: str
    values: tuple[str, ...]


SLOTS: tuple[Slot, ...] = (
    Slot(
        "region",
        "primary deployment region",
        ("US_EAST_1", "EU_WEST_1", "AP_SOUTH_1", "SA_EAST_1"),
    ),
    Slot(
        "database",
        "primary database engine",
        ("POSTGRES_14", "POSTGRES_15", "AURORA_PG", "COCKROACH_DB"),
    ),
    Slot(
        "auth",
        "default authentication provider",
        ("CLERK", "AUTH0", "COGNITO", "FIREBASE_AUTH"),
    ),
    Slot(
        "queue",
        "background job queue backend",
        ("SQS", "RABBITMQ", "KAFKA", "REDIS_STREAMS"),
    ),
    Slot(
        "cache",
        "shared cache tier",
        ("MEMCACHED", "REDIS_7", "DRAGONFLY", "ELASTICACHE"),
    ),
    Slot(
        "payments",
        "payments processor",
        ("STRIPE", "ADYEN", "BRAINTREE", "PADDLE"),
    ),
)


def _supersession_chain(slot: Slot, chain_len: int, id_prefix: str) -> tuple[list[MemoryItem], str]:
    """Build a ``chain_len``-revision supersession chain for one slot.

    Returns the list of decision nodes (oldest ``v0`` first) and the id of the
    head -- the latest, currently-valid revision. Revision ``k`` (for ``k >= 1``)
    carries a typed ``supersedes`` edge to revision ``k - 1``; the oldest
    revision has no outgoing edge and the head has no incoming one.
    """
    ids = [f"{id_prefix}-v{k}" for k in range(chain_len)]
    items: list[MemoryItem] = []
    for k, node_id in enumerate(ids):
        value = slot.values[k % len(slot.values)]
        text = (
            f"DECISION (rev {k}): the canonical {slot.subject} is set to {value}. "
            f"This revision is authoritative for all new work."
        )
        edges: list[tuple[str, str]] = []
        if k >= 1:
            edges = [(ids[k - 1], SUPERSEDES)]  # rev k supersedes rev k-1
        items.append(
            MemoryItem(
                node_id,
                text,
                metadata={
                    "node_type": "decision",
                    "slot": slot.slug,
                    "rev": k,
                    "edges": edges,
                },
            )
        )
    return items, ids[-1]


def generate_supersession_tasks(
    chain_len: int,
    n: int,
    seed: int = 0,
    *,
    distractor_chains: int = 2,
    shuffle_corpus: bool = True,
) -> list[Task]:
    """Generate supersession-chain recovery tasks.

    Each task targets one slot whose decision has been revised ``chain_len``
    times; the gold answer is the head (latest) revision, reached by following
    the typed ``supersedes`` edges back from the newest node. Optional distractor
    chains from *other* slots form query-irrelevant noise. The corpus is shuffled
    (when ``shuffle_corpus``) so position alone cannot reveal the head.

    Parameters
    ----------
    chain_len
        Number of revisions in the target chain (``>= 1``). Equals the task
        ``depth``: depth ``d`` means the head sits ``d - 1`` supersession hops
        from the oldest revision.
    n
        Number of tasks to generate (``>= 0``). Each rotates the target slot.
    seed
        Seed for the single :class:`random.Random` that drives slot rotation and
        corpus shuffling, making the output reproducible.
    distractor_chains
        Number of *other* slots that each contribute a full ``chain_len``
        supersession chain as noise (``0`` for a clean single-chain corpus).
        Clamped to the number of available non-target slots.
    shuffle_corpus
        Whether to shuffle the corpus order. The gold id is independent of order.

    Returns
    -------
    list[membench.types.Task]
        One :class:`~membench.types.Task` per requested task, with
        ``governing_decisions`` holding the single head id and ``depth`` equal to
        ``chain_len``.

    Raises
    ------
    ValueError
        If ``chain_len < 1`` or ``n < 0``.
    """
    if chain_len < 1:
        raise ValueError(f"chain_len must be >= 1, got {chain_len}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")

    rng = random.Random(seed)
    max_distractors = max(0, min(distractor_chains, len(SLOTS) - 1))
    tasks: list[Task] = []
    for i in range(n):
        target = SLOTS[(i + chain_len) % len(SLOTS)]
        task_id = f"supersede-c{chain_len}-{i:02d}"

        target_items, head_id = _supersession_chain(target, chain_len, task_id)
        corpus: list[MemoryItem] = list(target_items)

        others = [s for s in SLOTS if s.slug != target.slug]
        rng.shuffle(others)
        for slot in others[:max_distractors]:
            distractor_items, _ = _supersession_chain(slot, chain_len, f"{task_id}-{slot.slug}")
            corpus.extend(distractor_items)

        if shuffle_corpus:
            rng.shuffle(corpus)

        query = (
            f"What is the current {target.subject}? Earlier decisions have been "
            f"superseded; answer with the latest one that is still in force."
        )
        tasks.append(
            Task(
                task_id=task_id,
                dataset="supersession_synth",
                query=query,
                repo_ref="synthetic://supersession-bench",
                memory_corpus=tuple(corpus),
                governing_decisions=(head_id,),
                depth=chain_len,
                spec_variant=SpecVariant.STRIPPED,
                scorer=Scorer.COMPLIANCE,
            )
        )
    return tasks
