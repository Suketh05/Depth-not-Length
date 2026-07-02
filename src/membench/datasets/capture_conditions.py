"""Corpus transforms for the three storage conditions of the capture experiment.

Implements the manipulations of paper Section ``sec:capture`` ("The Capture
Experiment: Links, Not Discreteness"), Subsection ``sec:capconds`` ("Three
storage conditions"); the measured outcome they feed is Table ``tab:capexp``.
Every experiment before that section hands all arms the *same* already-extracted
decision corpus (the fairness lock of Section ``sec:offline``); this module is
the code that varies what "captured" means, by direct construction:

* **CAPTURED** -- the full typed store: each decision is a discrete
  :class:`~membench.types.MemoryItem` *and* the governance edges (``constrains``,
  ``supersedes``, ``implements``) are present and traversable. The identity
  transform: :func:`apply_condition` returns the task unchanged.
* **Discrete-no-links** (:func:`strip_edges`) -- keeps each decision as a clean,
  discrete item but *strips the typed edges*, so a retriever must fall back on
  similarity between discrete items. Text, ids, item count, corpus order, and
  every other metadata key are preserved verbatim; only the link-bearing
  metadata keys (:data:`TYPED_LINK_KEYS`) are removed. This isolates the value
  of the *links* (paper conclusion (i), Subsection ``sec:capconc``): with the
  edges gone the structured store reverts to the similarity ceiling of Theorem
  ``thm:sim``, the empirical confirmation of Theorem ``thm:struct``.
* **Raw-scattered** (:func:`shred_scattered`) -- strips the edges *and* shreds
  each captured decision back into raw fragments dispersed across ``sigma``
  distractor areas, "the closest in-harness proxy for 'no capture at all'"
  (Subsection ``sec:capconds``). Recovering a decision now requires *assembling*
  all ``sigma`` of its fragments, which is exactly the scatter-assembly penalty
  ``P_asm(sigma) = p^sigma`` of Equation ``eq:scatter`` / Theorem
  ``thm:scatter``.

Paper numbers these transforms feed (provenance: Table ``tab:capexp``, recall by
depth d1/d2/d3 over n=40 tasks per depth, paired across conditions; quoted
verbatim for cross-checking, never asserted against this module's output):
Brief 0.99/0.99/0.96 (CAPTURED) -> 0.98/0.95/0.76 (Discrete-no-links) ->
0.41/0.41/0.47 (Raw-scattered); bm25 0.98/0.97/0.95 -> 0.98/0.97/0.70 ->
0.33/0.33/0.33; tfidf 0.98/0.97/0.95 -> 0.98/0.97/0.70 -> 0.37/0.37/0.37;
dense 0.97/0.97/0.94 -> 0.97/0.94/0.68 -> 0.37/0.37/0.28. The section's prose
quotes a coarser ladder ("Brief holds 1.00 ... falls to 0.82/0.70/0.42 ...
floor band 0.27--0.37, mean 0.35", Section ``sec:capture``) whose cells differ
from the table's; both are reproduced here exactly as published so a
cross-checker sees the discrepancy rather than a silently harmonised number.

The transforms are pure and deterministic: the same task and parameters yield an
identical output (shredding is seeded). They are also conservative in text
("reversible in count"): stripping loses no text at all, and shredding
partitions each decision's whitespace-token sequence into contiguous fragments
whose payloads concatenate back to the original token sequence -- nothing is
dropped, only de-unitised.
"""

from __future__ import annotations

import random
from dataclasses import replace
from enum import Enum

from membench.datasets.synthetic import AREAS
from membench.types import MemoryItem, Task

__all__ = [
    "DEFAULT_SIGMA",
    "TYPED_LINK_KEYS",
    "CaptureCondition",
    "apply_condition",
    "shred_scattered",
    "strip_edges",
]

TYPED_LINK_KEYS: tuple[str, ...] = ("edges", "constrains")
"""Metadata keys that carry typed governance links.

``edges`` holds explicit ``(target_id, relationship_type)`` pairs (the synthetic
generator's ``constrains`` / ``derived_from`` chain links) and ``constrains``
holds the single back-link a stripped justification node carries; both are read
by the graph arm's ``write`` (``membench.retrieval.graph.arm``) and by nothing
else. Removing exactly these keys is therefore the surgical link strip of
Subsection ``sec:capconds``: "removing the edges does nothing to an arm that
never read them" (Section ``sec:capture``).
"""

DEFAULT_SIGMA = 3
"""Default fragment count for :func:`shred_scattered`.

The paper does not publish the ``sigma`` used for its Raw-scattered condition;
it states only that "the floor depends on the sigma we chose" (conclusion (ii),
Subsection ``sec:capconc``), i.e. the Raw-scattered magnitude is
harness-defined. The default is therefore an explicit, documented choice here,
not a paper-derived constant.
"""


class CaptureCondition(str, Enum):
    """The three storage conditions of Subsection ``sec:capconds``.

    Declaration order is the paper's degradation order (Table ``tab:capexp``
    rows within each retriever block): full typed store, then links stripped,
    then links stripped and decisions shredded. The string values serialise
    transparently into result rows, mirroring the enums in
    :mod:`membench.types`.
    """

    CAPTURED = "captured"
    DISCRETE_NO_LINKS = "discrete_no_links"
    RAW_SCATTERED = "raw_scattered"


def _strip_item(item: MemoryItem) -> MemoryItem:
    """Return ``item`` with the typed-link metadata keys removed.

    Everything else -- id, text, and every non-link metadata key (``node_type``,
    dataset annotations, ...) -- is preserved verbatim, so the item stays a
    clean discrete unit and only its traversability changes.
    """
    metadata = {k: v for k, v in item.metadata.items() if k not in TYPED_LINK_KEYS}
    return MemoryItem(item.item_id, item.text, metadata=metadata)


def strip_edges(task: Task) -> Task:
    """Transform a task into the **Discrete-no-links** condition.

    Removes the typed governance links (:data:`TYPED_LINK_KEYS`) from every
    corpus item while keeping each decision as a clean, discrete item: ids,
    texts, item count, corpus order, gold ``governing_decisions``, depth, and
    all other task fields are unchanged. "Discrete-no-links keeps each decision
    as a clean, discrete item but strips the edges, so the retriever must fall
    back on similarity between discrete items" (Subsection ``sec:capconds``).

    The manipulation is surgical by construction: only the graph arm reads the
    stripped keys, so an arm that never read them retrieves identically under
    CAPTURED and Discrete-no-links -- the paper's own falsification check
    ("the lexical blocks bm25, tfidf, and dense are identical across CAPTURED
    and Discrete-no-links", Section ``sec:capture``).

    Parameters
    ----------
    task
        The CAPTURED-condition task to transform. Never mutated (tasks and
        items are frozen); a new :class:`~membench.types.Task` is returned.

    Returns
    -------
    Task
        The same task with a link-free corpus. Pure, deterministic, and
        idempotent: ``strip_edges(strip_edges(t)) == strip_edges(t)``.
    """
    return replace(
        task,
        memory_corpus=tuple(_strip_item(item) for item in task.memory_corpus),
    )


def _partition(words: list[str], n_fragments: int) -> list[list[str]]:
    """Split ``words`` into ``n_fragments`` contiguous chunks, as evenly as possible.

    The standard even partition (``numpy.array_split`` semantics): the first
    ``len(words) % n_fragments`` chunks receive one extra word. Contiguity is
    what makes the shred "reversible in count": concatenating the chunks in
    order reproduces the original token sequence exactly.
    """
    total = len(words)
    base, extra = divmod(total, n_fragments)
    chunks: list[list[str]] = []
    start = 0
    for k in range(n_fragments):
        size = base + (1 if k < extra else 0)
        chunks.append(words[start : start + size])
        start += size
    return chunks


def shred_scattered(task: Task, sigma: int = DEFAULT_SIGMA, *, seed: int = 0) -> Task:
    """Transform a task into the **Raw-scattered** condition.

    Strips the typed edges *and* shreds every captured decision unit back into
    raw fragments dispersed across ``sigma`` distractor areas -- "the closest
    in-harness proxy for 'no capture at all'" (Subsection ``sec:capconds``).
    Concretely, for every corpus item that was captured as a typed unit (it
    carries a ``node_type``: the synthetic generator's entry / policy /
    governing nodes), the item's whitespace-token sequence is partitioned into
    ``min(sigma, token_count)`` contiguous fragments; each fragment becomes its
    own :class:`~membench.types.MemoryItem` whose text is the fragment payload
    followed by a parenthetical aside written in a *different* engineering
    area's vocabulary, as if the decision had only ever been mentioned in
    passing across unrelated raw notes. Items that were never captured as units
    (no ``node_type``; the corpus filler) pass through with only the link strip.

    Fragment placement rotates through a seed-shuffled pool of areas
    (:data:`~membench.datasets.synthetic.AREAS`), so the fragments of one
    decision land in ``sigma`` *distinct* areas whenever ``sigma`` does not
    exceed the pool size -- the "split each decision across sigma distractor
    areas" manipulation. Recovering a decision therefore requires retrieving
    all ``sigma`` of its fragments, the assembly event whose success the theory
    bounds by ``P_asm(sigma) = p^sigma`` (Equation ``eq:scatter``, Theorem
    ``thm:scatter``); under this condition every arm in the paper collapses
    into the floor band 0.27--0.37, mean 0.35 (Table ``tab:capexp``, Section
    ``sec:capture``).

    Gold semantics: every gold id in ``governing_decisions`` that was shredded
    is replaced by all of its fragment ids (order preserved), so per-task
    recall becomes the *fraction of the governing decision's fragments
    assembled* and full-chain recovery requires all of them.

    Conservation guarantees (asserted by the test suite):

    * no text lost -- for every shredded item, joining the fragments'
      ``payload`` metadata in ``fragment_index`` order reproduces the original
      whitespace-token sequence exactly;
    * count-reversible -- each shredded item maps to exactly
      ``fragment_count`` fragments and every original id is recoverable from
      ``shredded_from``.

    Parameters
    ----------
    task
        The CAPTURED-condition task to transform (never mutated).
    sigma
        Number of fragments (and target distractor areas) per decision,
        ``>= 1``. Items with fewer tokens than ``sigma`` produce one fragment
        per token (saturation). The paper does not publish its Raw-scattered
        ``sigma`` (see :data:`DEFAULT_SIGMA`).
    seed
        Seed for the deterministic area-pool shuffle; the same task, ``sigma``,
        and ``seed`` always produce an identical output.

    Returns
    -------
    Task
        A new task whose corpus contains link-free fragments in place of the
        captured units, with ``governing_decisions`` remapped to fragment ids.

    Raises
    ------
    ValueError
        If ``sigma < 1``.
    """
    if sigma < 1:
        raise ValueError(f"sigma must be >= 1, got {sigma}")
    rng = random.Random(seed)
    pool = list(AREAS)
    rng.shuffle(pool)

    corpus: list[MemoryItem] = []
    fragment_ids: dict[str, tuple[str, ...]] = {}
    cursor = 0  # advances through the area pool so consecutive items scatter differently
    for item in task.memory_corpus:
        base = _strip_item(item)
        if base.node_type is None:
            corpus.append(base)
            continue
        words = base.text.split()
        n_fragments = max(1, min(sigma, len(words)))
        ids: list[str] = []
        for k, chunk in enumerate(_partition(words, n_fragments)):
            area = pool[(cursor + k) % len(pool)]
            payload = " ".join(chunk)
            frag_id = f"{base.item_id}::frag{k}"
            text = f"{payload} (noted in passing during {area.name} work on the {area.feature})"
            corpus.append(
                MemoryItem(
                    frag_id,
                    text,
                    metadata={
                        "node_type": "fragment",
                        "shredded_from": base.item_id,
                        "fragment_index": k,
                        "fragment_count": n_fragments,
                        "payload": payload,
                        "scatter_area": area.slug,
                    },
                )
            )
            ids.append(frag_id)
        fragment_ids[base.item_id] = tuple(ids)
        cursor += n_fragments

    gold: list[str] = []
    for gold_id in task.governing_decisions:
        gold.extend(fragment_ids.get(gold_id, (gold_id,)))
    return replace(task, memory_corpus=tuple(corpus), governing_decisions=tuple(gold))


def apply_condition(
    task: Task,
    condition: CaptureCondition | str,
    *,
    sigma: int = DEFAULT_SIGMA,
    seed: int = 0,
) -> Task:
    """Apply one of the three storage conditions of Subsection ``sec:capconds``.

    The single dispatch point the experiment driver uses, so every condition is
    produced by exactly one code path:

    * ``CAPTURED`` -- the identity transform. The task is returned as-is (it is
      frozen, so sharing it is safe); this is the same fairness-lock corpus
      every other experiment in the paper consumes (Section ``sec:offline``).
    * ``DISCRETE_NO_LINKS`` -- :func:`strip_edges`.
    * ``RAW_SCATTERED`` -- :func:`shred_scattered` (which also strips links).

    Parameters
    ----------
    task
        The CAPTURED-condition task to transform.
    condition
        A :class:`CaptureCondition` member or its string value.
    sigma
        Fragment count for the Raw-scattered condition (ignored otherwise).
    seed
        Shred seed for the Raw-scattered condition (ignored otherwise).

    Returns
    -------
    Task
        The task under the requested storage condition, with the same
        ``task_id`` -- conditions are *paired* per task, as in the paper's
        "n=40 tasks per depth, paired across conditions" (Table ``tab:capexp``).

    Raises
    ------
    ValueError
        If ``condition`` is not one of the three conditions, or ``sigma < 1``
        for the Raw-scattered condition.
    """
    member = CaptureCondition(condition)
    if member is CaptureCondition.CAPTURED:
        return task
    if member is CaptureCondition.DISCRETE_NO_LINKS:
        return strip_edges(task)
    return shred_scattered(task, sigma, seed=seed)
