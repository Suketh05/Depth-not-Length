r"""Parametric N-hop reasoning-chain generator: a controllable depth ladder.

This loader produces clean, *parametric* multi-hop reasoning chains in which the
governing decision sits exactly ``N`` typed-link hops away from the query. Where
``synthetic.py`` draws its chains from a fixed pool of ten realistic engineering
areas (so the maximum usable depth is bounded by that hand-written vocabulary),
this generator templates entities, relations and the governing rule from word
pools, so an arbitrarily deep, fully controlled depth ladder can be emitted for a
clean stress test of link-following retrieval -- while staying deterministic and
offline.

Algorithm
---------
For a requested depth ``d`` the target chain is a **path graph** ``P_{d+1}``::

    h0  --r_1-->  h1  --r_2-->  ...  --r_{d-1}-->  h_{d-1}  --CONSTRAINS-->  h_d
   (entry,                      (typed intermediate                         (governing
    query-similar)              dependency links)                            decision)

A path graph on ``d + 1`` vertices has exactly ``d`` edges, and the graph distance
between its two endpoints equals the number of edges on the unique path between
them, i.e. ``dist(h0, h_d) = d`` (Bondy & Murty, *Graph Theory*, 2008, ch. 1 --
the length of a path is its number of edges). So the gold governing node is, by
construction, *exactly* ``d`` typed-link hops from the query node -- the invariant
the whole depth-bench rests on.

Each task's corpus is the target chain plus several distractor chains (other
templated topics whose vocabulary does not overlap the query) and plain filler
nodes, all shuffled, so an arm cannot infer relevance from position or corpus
size. Query-keyword overlap decays geometrically along the target chain
(``s_i = s_0 rho^i``), so the entry node resembles the query and the governing
node does not -- the regime in which a flat similarity retriever loses the deep
node under a fixed budget while a link-following arm walks straight to it.

This construction is the standard recipe for synthetic, depth-controlled
multi-hop reasoning benchmarks:

* Sinha, Sodhani, Dong, Pineau & Hamilton (2019), "CLUTRR: A Diagnostic
  Benchmark for Inductive Reasoning from Text", EMNLP 2019 -- composing ``k``
  typed (kinship) relations along a chain to infer the ``k``-hop endpoint
  relation, with ``k`` as an explicit difficulty/length dial.
* Saparov & He (2022), "Language Models Are Greedy Reasoners: A Systematic
  Formal Analysis of Chain-of-Thought" (PrOntoQA) -- proofs of controllable hop
  count generated from a synthetic ontology.

The governing node carries a code identifier (``guard()``) so compliance scoring
has a concrete target, and the final hop is a ``constrains`` edge so the
structured-graph arm recovers it via a ``CONSTRAINS`` traversal -- matching the
metadata contract that ``synthetic.py`` and ``dcbench.py`` already emit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from membench.types import MemoryItem, Scorer, SpecVariant, Task

__all__ = ["DEFAULT_RELATIONS", "TOPICS", "Topic", "generate_depth_chain_tasks"]


@dataclass(frozen=True, slots=True)
class Topic:
    """A templated reasoning topic: a distinctive subject and its governing guard.

    Parameters
    ----------
    slug
        Stable short identifier, unique within :data:`TOPICS`.
    subject
        The distinctive vocabulary word that ties the query to this topic's entry
        node (and to nothing else), so similarity seeding lands on the right chain.
    feature
        The artefact the query asks to build (carries the subject's keywords).
    guard
        The code identifier the governing rule mandates -- the compliance target.
    """

    slug: str
    subject: str
    feature: str
    guard: str


def _topic(word: str) -> Topic:
    """Template a :class:`Topic` from one distinctive subject word (parametric)."""
    return Topic(
        slug=word,
        subject=word,
        feature=f"{word} workflow surface",
        guard=f"enforce{word.capitalize()}Policy",
    )


# A large parametric pool of distinctive subject words. Each yields a chain whose
# vocabulary overlaps only its own query -- this is what keeps similarity seeding
# unambiguous and lets the depth ladder go arbitrarily deep without hand-writing
# new domains.
_SUBJECT_WORDS: tuple[str, ...] = (
    "aurora",
    "beacon",
    "cobalt",
    "delta",
    "ember",
    "fjord",
    "granite",
    "harbor",
    "indigo",
    "jasper",
    "kelvin",
    "lumen",
    "meridian",
    "nimbus",
    "onyx",
    "polar",
    "quartz",
    "ripple",
    "summit",
    "tundra",
    "umbra",
    "vertex",
    "willow",
    "zephyr",
)

TOPICS: tuple[Topic, ...] = tuple(_topic(word) for word in _SUBJECT_WORDS)

# Typed intermediate dependency relations cycled along the chain (every value is a
# valid edge type the graph arm can traverse). The *final* hop is always
# ``constrains`` so the governing node is recovered as a ``CONSTRAINS`` target.
DEFAULT_RELATIONS: tuple[str, ...] = (
    "derived_from",
    "informs",
    "motivates",
    "enables",
)

# Query-irrelevant filler vocabulary (the haystack), reused from the same spirit
# as the synthetic dataset's distractor nodes.
_FILLER: tuple[str, ...] = (
    "telemetry pipeline ingestion buffer",
    "design system token palette spacing",
    "infrastructure terraform module registry",
    "changelog release notes versioning",
)


def _node_text(topic: Topic, hop: int, depth: int, *, governing: bool) -> str:
    """Chain-node text whose query-keyword share decays geometrically with the hop.

    The entry node (hop 0) carries the full subject vocabulary; the share falls as
    ``(depth - hop + 1) / (depth + 1)`` toward the governing node, instantiating
    the similarity-decay model ``s_i = s_0 rho^i`` the theory assumes. The
    governing node carries the code identifier so compliance has a concrete target.
    """
    fraction = (depth - hop + 1) / (depth + 1)
    repeats = max(0 if governing else 1, round(depth * fraction))
    subject = " ".join([topic.subject] * repeats)
    if governing:
        head = f"{subject}. " if subject else ""
        return f"{head}RULE: every change in this path must call {topic.guard}() before proceeding."
    if hop == 0:
        return f"The {topic.subject} {topic.feature} is where this work begins."
    return f"{subject} intermediate policy layer, refining the upstream constraint it derives from."


def _chain(
    topic: Topic, depth: int, key: str, relations: tuple[str, ...]
) -> tuple[list[MemoryItem], str]:
    """Build a depth-``d`` path-graph chain ``entry -> ... -> governing`` for a topic.

    Returns the ``depth + 1`` nodes (with typed ``edges`` metadata) and the
    ``item_id`` of the governing node (the terminal, ``depth`` hops from the entry).
    """
    ids = [f"{key}-h{hop}" for hop in range(depth + 1)]
    texts = [_node_text(topic, hop, depth, governing=(hop == depth)) for hop in range(depth + 1)]
    items: list[MemoryItem] = []
    for idx, (node_id, text) in enumerate(zip(ids, texts, strict=True)):
        edges: list[tuple[str, str]] = []
        if idx < len(ids) - 1:
            # Final hop is CONSTRAINS (the link a flat retriever cannot make);
            # earlier hops cycle through the typed intermediate relations.
            rel = "constrains" if idx == len(ids) - 2 else relations[idx % len(relations)]
            edges = [(ids[idx + 1], rel)]
        node_type = "entry" if idx == 0 else ("governing" if idx == len(ids) - 1 else "policy")
        items.append(MemoryItem(node_id, text, metadata={"node_type": node_type, "edges": edges}))
    return items, ids[-1]


def generate_depth_chain_tasks(
    depths: tuple[int, ...] = (1, 2, 3, 4),
    per_depth: int = 10,
    seed: int = 0,
    *,
    distractor_chains: int = 5,
    relations: tuple[str, ...] = DEFAULT_RELATIONS,
) -> list[Task]:
    """Generate parametric N-hop reasoning-chain tasks (a controllable depth ladder).

    For each requested depth ``d`` and task index, one topic's path-graph chain
    carries the governing decision exactly ``d`` typed-link hops from the
    query-similar entry node; several other topics' chains and plain filler nodes
    form the distractor haystack. The target topic rotates per task so no single
    topic drives the result, and the corpus is shuffled deterministically.

    Parameters
    ----------
    depths
        The chain depths to emit (each ``>= 1``). The governing node is ``depth``
        typed-link hops from the query for every task at that depth.
    per_depth
        Number of tasks generated per depth.
    seed
        Seed for the deterministic shuffles (topic selection order and corpus
        order). Identical seeds yield byte-identical task lists.
    distractor_chains
        Number of non-target topic chains mixed into each corpus.
    relations
        Typed intermediate edge relations cycled along each chain (the final hop is
        always ``constrains``). Every value must be a valid graph edge type.

    Returns
    -------
    list[Task]
        ``len(depths) * per_depth`` tasks, ordered by depth then index.

    Raises
    ------
    ValueError
        If any depth is ``< 1``, ``per_depth < 1``, ``distractor_chains < 0``, or
        ``relations`` is empty.

    Notes
    -----
    The depth invariant follows the path-graph length formula: a chain of
    ``d + 1`` nodes has ``d`` edges and endpoint distance ``d`` (Bondy & Murty,
    *Graph Theory*, 2008). See the module docstring for the CLUTRR / PrOntoQA
    references this construction follows.
    """
    if per_depth < 1:
        raise ValueError(f"per_depth must be >= 1, got {per_depth}")
    if distractor_chains < 0:
        raise ValueError(f"distractor_chains must be >= 0, got {distractor_chains}")
    if not relations:
        raise ValueError("relations must be non-empty")

    rng = random.Random(seed)
    tasks: list[Task] = []
    for depth in depths:
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        for i in range(per_depth):
            target = TOPICS[(i + depth) % len(TOPICS)]
            corpus: list[MemoryItem] = []
            target_items, governing_id = _chain(
                target, depth, key=f"d{depth}-{i:02d}-tgt", relations=relations
            )
            corpus.extend(target_items)

            others = [t for t in TOPICS if t.slug != target.slug]
            rng.shuffle(others)
            for j, topic in enumerate(others[:distractor_chains]):
                corpus.extend(
                    _chain(topic, depth, key=f"d{depth}-{i:02d}-dis{j}", relations=relations)[0]
                )

            for k, filler in enumerate(_FILLER):
                corpus.append(MemoryItem(f"d{depth}-{i:02d}-fill{k}", filler))

            rng.shuffle(corpus)
            query = (
                f"Build the {target.subject} {target.feature} so the team can "
                f"ship the {target.subject} capability."
            )
            tasks.append(
                Task(
                    task_id=f"depth_chain-d{depth}-{i:02d}",
                    dataset="depth_chain",
                    query=query,
                    repo_ref="depth_chain://parametric-bench",
                    memory_corpus=tuple(corpus),
                    governing_decisions=(governing_id,),
                    depth=depth,
                    spec_variant=SpecVariant.STRIPPED,
                    scorer=Scorer.COMPLIANCE,
                )
            )
    return tasks
