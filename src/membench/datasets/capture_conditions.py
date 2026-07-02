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

from dataclasses import replace
from enum import Enum

from membench.types import MemoryItem, Task

__all__ = [
    "DEFAULT_SIGMA",
    "TYPED_LINK_KEYS",
    "CaptureCondition",
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
