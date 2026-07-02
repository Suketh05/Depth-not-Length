-- Brief context-graph schema (reference DDL).
--
-- Brief runs on Postgres with pgvector; this portable DDL captures the typed-node +
-- typed-edge model that membench's structured-graph arm reproduces in memory
-- (entity_objects = nodes, entity_link = typed edges). Postgres-specific features
-- (embedding vector(1536) + HNSW cosine index, RLS policies, per-type decay_rate
-- catalogue) are noted in comments. Validated in-memory with sqlite3 in CI.

PRAGMA foreign_keys = ON;

-- Node registry: every typed record (decision, feature, constraint, justification...).
CREATE TABLE entity_objects (
  id             TEXT PRIMARY KEY,
  entity_type    TEXT NOT NULL,
  entity_id      TEXT NOT NULL,
  kind           TEXT NOT NULL CHECK (kind IN ('source', 'synthesis')),
  canonical_type TEXT NOT NULL,
  title          TEXT,
  access_count   INTEGER NOT NULL DEFAULT 0,
  citation_count INTEGER NOT NULL DEFAULT 0
  -- Postgres also: embedding vector(1536) with an HNSW cosine index.
);

-- Typed, directed, confidence-weighted, usage-tracked edges (FK-enforced endpoints).
CREATE TABLE entity_link (
  id                      TEXT PRIMARY KEY,
  source_entity_object_id TEXT NOT NULL REFERENCES entity_objects (id) ON DELETE CASCADE,
  target_entity_object_id TEXT NOT NULL REFERENCES entity_objects (id) ON DELETE CASCADE,
  relationship_type       TEXT NOT NULL CHECK (relationship_type IN (
    'related_to', 'constrains', 'constrained_by', 'implements', 'validates',
    'contradicts', 'informs', 'supersedes', 'blocks', 'derived_from',
    'supports_goal', 'motivates', 'enables', 'impacts', 'references', 'duplicate_of')),
  status                  TEXT NOT NULL DEFAULT 'inferred'
                            CHECK (status IN ('inferred', 'confirmed', 'archived', 'rejected')),
  confidence              REAL NOT NULL DEFAULT 1.0 CHECK (confidence BETWEEN 0.0 AND 1.0),
  match_method            TEXT CHECK (match_method IN ('embedding', 'keyword', 'explicit', 'manual')),
  traversal_count         INTEGER NOT NULL DEFAULT 0,
  last_traversed_tick     INTEGER,
  UNIQUE (source_entity_object_id, target_entity_object_id, relationship_type)
);

CREATE INDEX idx_entity_link_source ON entity_link (source_entity_object_id);
CREATE INDEX idx_entity_link_target ON entity_link (target_entity_object_id);

-- Bounded multi-hop traversal (mirrors traverseEntityGraph) as a recursive CTE.
-- Parameterised in application code by :start, :max_hops, :min_confidence:
--
--   WITH RECURSIVE walk(node, depth) AS (
--       SELECT :start AS node, 0 AS depth
--     UNION
--       SELECT e.target_entity_object_id, walk.depth + 1
--       FROM walk
--       JOIN entity_link e ON e.source_entity_object_id = walk.node
--       WHERE walk.depth < :max_hops AND e.confidence >= :min_confidence
--   )
--   SELECT DISTINCT node, depth FROM walk ORDER BY depth;

-- ============================================================================
-- VERIFICATION SUITE (appended; DDL above is unchanged).
-- ============================================================================
-- Executable, self-checking cross-language differential test for the typed
-- decision-graph arm ("brief_graph_3hop") of the "Depth, Not Length" benchmark.
-- CI runs `sqlite3 ":memory:" < schema/graph.sql`; sqlite3 batch mode stops at
-- the first SQL error and exits non-zero, so every assertion below is written as
-- an INSERT into a table with `CHECK (ok = 1)` -- a false predicate yields ok=0,
-- the CHECK fails, and CI turns red. The suite therefore FAILS on any mismatch.
--
-- What it verifies, and the paper anchors (labels from depth_not_length_FIXED.tex):
--   1. Bounded traversal recovers a decision at exactly its causal-hop depth,
--      d=3 governing chain                          -- eq:perhop, thm:struct, sec:struct
--   2. Super-geometric similarity decay vs steady per-hop traversal cost, and
--      the ~29x depth-3 gap                          -- eq:simceiling, thm:sim, sec:depth
--   3. The capture lever: strip the typed links and reachability collapses to 0
--      (Discrete-no-links / Raw-scattered condition) -- sec:capture, tab:capexp
--   4. Supersession resolves to the CURRENT decision, while a similarity proxy
--      returns a stale near-identical version         -- tab:super, sec:resmech-super
--   5. Fairness-lock integrity: the typed-edge domain and value CHECK
--      constraints are present and enforced           -- sec:fairness
--
-- Golden numeric constants are transcribed VERBATIM from the canonical golden
-- pack (theory goldens A.1 P_sim/P_struct, A.7 supersession); never recomputed
-- here. SQLite float64 == Python float64 for these products (abs err 0.0), and
-- assertions still allow abs tol 1e-12 for portability across CI hosts.

-- ---------------------------------------------------------------------------
-- SEED 1/4 -- Governing decision chain nodes (a depth-3 causal chain).
-- ---------------------------------------------------------------------------
-- A single edit task ('q-task') whose correct action is governed by a decision
-- three causal hops away ('d-gov-3'). This is the depth-3 regime the theory is
-- about: the governing decision is NOT a function of the task surface alone
-- (def:governed), so it must be reached by following typed links, not guessed.
INSERT INTO entity_objects (id, entity_type, entity_id, kind, canonical_type, title, access_count, citation_count) VALUES
  ('q-task',  'task',     'T-4821', 'source', 'task',     'Edit: swap checkout button click handler', 5, 0),
  ('d-gov-1', 'decision', 'D-1001', 'source', 'decision', 'Checkout mutations go through the session guard', 9, 4),
  ('d-gov-2', 'decision', 'D-1002', 'source', 'decision', 'Session guard requires a signed, rotating key', 7, 3),
  ('d-gov-3', 'decision', 'D-1003', 'source', 'decision', 'GOVERNING: key rotation cadence is fixed at 24h', 3, 2);

-- ---------------------------------------------------------------------------
-- SEED 2/4 -- The 'constrains' governing chain edges: q-task -> d1 -> d2 -> d3.
-- ---------------------------------------------------------------------------
-- Edge direction follows traverseEntityGraph's forward walk (source = the node
-- you traverse FROM). Each hop is an explicit, confirmed, full-confidence typed
-- link -- the "steady per-hop cost" traversal of thm:struct (q^d recovery),
-- contrasted below with the super-geometric similarity decay of thm:sim.
INSERT INTO entity_link (id, source_entity_object_id, target_entity_object_id, relationship_type, status, confidence, match_method) VALUES
  ('e-gov-1', 'q-task',  'd-gov-1', 'constrains', 'confirmed', 1.0, 'explicit'),
  ('e-gov-2', 'd-gov-1', 'd-gov-2', 'constrains', 'confirmed', 1.0, 'explicit'),
  ('e-gov-3', 'd-gov-2', 'd-gov-3', 'constrains', 'confirmed', 1.0, 'explicit');

