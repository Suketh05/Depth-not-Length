"""Adapters onto third-party memory systems, benchmarked under identical conditions.

Each adapter wraps a real external memory system (Mem0, Zep/Graphiti, GraphRAG,
Letta, Cognee, LangChain) behind the same :class:`~membench.retrieval.base.MemorySystem`
contract, so it competes on the exact same tasks, budget, and model as every other
arm. These are *measured* (Tier-1) competitors when their package is installed;
their authors' *published* numbers are kept separately in the cited vendor registry
(Tier-2), never mixed.

Every external package pins mutually-incompatible dependencies, so none is in the
project lockfile: each adapter lazy-imports its backend and raises an actionable
error pointing at an isolated environment file under ``competitors/envs/``. All
adapters accept an injected client, which is how their logic is unit-tested without
the heavyweight (and often networked) real backend.
"""
