"""Memory arms: the swappable systems under comparison.

Every arm implements the :class:`membench.retrieval.base.MemorySystem` contract
(``write`` then ``retrieve(query, budget_tokens)``) and is registered in the arm
registry. Arms span four families: trivial controls (none / full-context /
random), lexical similarity (BM25, TF-IDF), dense and hybrid neural retrieval
(embeddings, RRF, cross-encoder rerank, HyDE, RAPTOR), the structured typed-graph
walk that models Brief's context graph, and adapters onto third-party memory
systems. Memory architecture is the only variable that changes across arms; model
and token budget are held fixed by the harness.
"""
