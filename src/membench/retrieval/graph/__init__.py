"""The structured typed-graph memory: Brief's mechanism, reproduced locally.

This subpackage is the heart of the benchmark's positive claim. Where every
similarity arm ranks by resemblance to the query, the structured arm *follows
explicit typed links* between records — the operation similarity retrieval cannot
perform, and the one the depth thesis says escapes the depth ceiling.

The store (:mod:`membench.retrieval.graph.store`) mirrors Brief's data model: a
node registry (``entity_objects``) and a typed, confidence-weighted, usage-tracked
edge table (``entity_link``) with FK-enforced endpoints. Traversal
(:mod:`membench.retrieval.graph.traversal`) reproduces the semantics of Brief's
``traverseEntityGraph`` (bounded multi-hop BFS). Usage-aware confidence decay
(:mod:`membench.retrieval.graph.decay`) mirrors Brief's per-type edge decay that
rewards traversed links and prunes stale ones.
"""
