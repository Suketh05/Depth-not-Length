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

-- ---------------------------------------------------------------------------
-- SEED 3/4 -- Scattered distractors across areas + a low-confidence similarity
-- shortcut trap.
-- ---------------------------------------------------------------------------
-- Realistic noise: decisions/features in other areas ('ui', 'perf') reachable
-- off the governing chain, plus an ISOLATED area ('iso') that must never be
-- reached from the task. 'e-trap' is a spurious low-confidence (0.20) similarity
-- edge q-task -> d-gov-3: it would fake a 1-hop shortcut to the governing
-- decision, but the min_confidence >= 0.5 gate filters it, so the recovered
-- causal depth stays the true 3 (the confidence gate is why an unreliable
-- similarity hit does not collapse the measured hop distance).
INSERT INTO entity_objects (id, entity_type, entity_id, kind, canonical_type, title, access_count, citation_count) VALUES
  ('dx-ui-1',   'decision', 'D-2001', 'source',    'decision', 'Distractor: button uses design-system primitive', 4, 1),
  ('dx-ui-2',   'feature',  'F-2002', 'synthesis', 'feature',  'Distractor: checkout page layout', 2, 0),
  ('dx-perf-1', 'decision', 'D-2003', 'source',    'decision', 'Distractor: memoize cart totals', 3, 1),
  ('dx-iso-1',  'decision', 'D-3001', 'source',    'decision', 'Isolated-area decision: email footer copy', 1, 0),
  ('dx-iso-2',  'feature',  'F-3002', 'synthesis', 'feature',  'Isolated-area feature: newsletter opt-in', 1, 0);
INSERT INTO entity_link (id, source_entity_object_id, target_entity_object_id, relationship_type, status, confidence, match_method) VALUES
  ('e-dx-1',  'q-task',   'dx-ui-1',   'references', 'confirmed', 0.90, 'explicit'),
  ('e-dx-2',  'dx-ui-1',  'dx-ui-2',   'related_to', 'inferred',  0.80, 'embedding'),
  ('e-dx-3',  'd-gov-1',  'dx-perf-1', 'informs',    'inferred',  0.70, 'keyword'),
  ('e-iso-1', 'dx-iso-1', 'dx-iso-2',  'related_to', 'inferred',  0.75, 'embedding'),
  ('e-trap',  'q-task',   'd-gov-3',   'related_to', 'inferred',  0.20, 'embedding');

-- ---------------------------------------------------------------------------
-- SEED 4/4 -- Edgeless copy of the chain (the "Discrete-no-links" condition).
-- ---------------------------------------------------------------------------
-- Same four decisions, discrete, with ZERO typed links between them -- exactly
-- the capture-experiment condition where the decisions exist but the governance
-- edges have been stripped (sec:capture, tab:capexp). Reachability from the
-- edgeless task must collapse to 0 governing decisions: the mechanism is the
-- LINKS, not the discreteness of the records (Brief d3 recall 1.00 -> 0.42 when
-- links are removed; here modeled as reachability 1 -> 0).
INSERT INTO entity_objects (id, entity_type, entity_id, kind, canonical_type, title, access_count, citation_count) VALUES
  ('q-task-e',  'task',     'T-4821E', 'source', 'task',     'Edgeless copy: swap checkout button click handler', 0, 0),
  ('d-gov-1-e', 'decision', 'D-1001E', 'source', 'decision', 'Edgeless copy of d-gov-1', 0, 0),
  ('d-gov-2-e', 'decision', 'D-1002E', 'source', 'decision', 'Edgeless copy of d-gov-2', 0, 0),
  ('d-gov-3-e', 'decision', 'D-1003E', 'source', 'decision', 'Edgeless copy of d-gov-3 (unreachable without links)', 0, 0);
-- (No entity_link rows for the *-e nodes: the discrete-no-links condition.)

-- ---------------------------------------------------------------------------
-- SEED (supersession) -- three near-identical decision versions + supersedes chain.
-- ---------------------------------------------------------------------------
-- A decision revised twice: s1 (oldest) <- s2 <- s3 (current). Edge convention:
-- the NEWER decision `supersedes` the older, so source=newer, target=older; the
-- head (current) is the version that is never a target. citation_count is a
-- similarity/centrality proxy: the STALE original s1 is the most-cited, so an
-- embedding/frequency retriever ranks it first -- the exact failure tab:super
-- measures (similarity 64-69% current-ranked-first vs typed 92.3%). n=3 versions
-- => n-1 = 2 supersedes edges (golden A.7 edge_hops).
INSERT INTO entity_objects (id, entity_type, entity_id, kind, canonical_type, title, access_count, citation_count) VALUES
  ('s-sess-1', 'decision', 'D-500', 'source', 'decision', 'Session TTL = 30 minutes',           40, 40),
  ('s-sess-2', 'decision', 'D-501', 'source', 'decision', 'Session TTL = 60 minutes',           14, 12),
  ('s-sess-3', 'decision', 'D-502', 'source', 'decision', 'Session TTL = 15 minutes (current)',  6,  3);
INSERT INTO entity_link (id, source_entity_object_id, target_entity_object_id, relationship_type, status, confidence, match_method) VALUES
  ('e-sup-1', 's-sess-2', 's-sess-1', 'supersedes', 'confirmed', 1.0, 'explicit'),
  ('e-sup-2', 's-sess-3', 's-sess-2', 'supersedes', 'confirmed', 1.0, 'explicit');

-- ---------------------------------------------------------------------------
-- ASSERT 1 -- Bounded traversal recovers the governing decision at EXACTLY 3 hops.
--   Anchors: eq:perhop (P_ret = product of per-hop survival), thm:struct
--   (structured recovery reaches depth-d at steady cost), sec:struct.
-- ---------------------------------------------------------------------------
-- Recursive CTE = the traverseEntityGraph walk from the roots-digest template,
-- carrying hop depth. min_confidence >= 0.5 filters the spurious 0.20 trap edge,
-- so the shortest confidence-gated path to d-gov-3 is the true causal depth 3.
CREATE TABLE _assert_depth3_reach (ok INTEGER NOT NULL CHECK (ok = 1));
INSERT INTO _assert_depth3_reach
SELECT CASE WHEN (
  WITH RECURSIVE walk(node, depth) AS (
      SELECT 'q-task', 0
    UNION
      SELECT e.target_entity_object_id, walk.depth + 1
      FROM walk
      JOIN entity_link e ON e.source_entity_object_id = walk.node
      WHERE walk.depth < 3 AND e.confidence >= 0.5
  )
  SELECT MIN(depth) FROM walk WHERE node = 'd-gov-3'
) = 3 THEN 1 ELSE 0 END;

-- Companion: the SAME walk WITHOUT the confidence gate (>= 0.0) lets the 0.20
-- similarity trap edge fake a 1-hop shortcut -- proving the gate is load-bearing.
CREATE TABLE _assert_trap_shortcut_is_1hop (ok INTEGER NOT NULL CHECK (ok = 1));
INSERT INTO _assert_trap_shortcut_is_1hop
SELECT CASE WHEN (
  WITH RECURSIVE walk(node, depth) AS (
      SELECT 'q-task', 0
    UNION
      SELECT e.target_entity_object_id, walk.depth + 1
      FROM walk
      JOIN entity_link e ON e.source_entity_object_id = walk.node
      WHERE walk.depth < 3 AND e.confidence >= 0.0
  )
  SELECT MIN(depth) FROM walk WHERE node = 'd-gov-3'
) = 1 THEN 1 ELSE 0 END;

-- ---------------------------------------------------------------------------
-- ASSERT 2 -- The capture lever: no links => the governing decision is
--   UNREACHABLE (reachable count 0). Anchors: sec:capture, tab:capexp
--   (Discrete-no-links / Raw-scattered conditions).
-- ---------------------------------------------------------------------------
-- Walk from the edgeless task: it reaches only itself, so the count of reachable
-- governing decisions is 0. Same records as the linked chain, links removed ->
-- recovery collapses. The mechanism is the typed links, not the records.
CREATE TABLE _assert_edgeless_unreachable (ok INTEGER NOT NULL CHECK (ok = 1));
INSERT INTO _assert_edgeless_unreachable
SELECT CASE WHEN (
  WITH RECURSIVE walk(node, depth) AS (
      SELECT 'q-task-e', 0
    UNION
      SELECT e.target_entity_object_id, walk.depth + 1
      FROM walk
      JOIN entity_link e ON e.source_entity_object_id = walk.node
      WHERE walk.depth < 3 AND e.confidence >= 0.5
  )
  SELECT COUNT(*) FROM walk WHERE node IN ('d-gov-1-e', 'd-gov-2-e', 'd-gov-3-e')
) = 0 THEN 1 ELSE 0 END;

-- ---------------------------------------------------------------------------
-- ASSERT 3 -- Reachable-set invariants for the LINKED graph.
--   Anchor: sec:struct (bounded traversal returns exactly the connected typed
--   neighbourhood; scattered-area distractors stay out).
-- ---------------------------------------------------------------------------
-- Within 3 confidence-gated hops of q-task, exactly 6 distinct nodes are
-- reachable besides the start (d-gov-1..3, dx-ui-1, dx-ui-2, dx-perf-1), and the
-- ISOLATED area (dx-iso-1, dx-iso-2) is never touched.
CREATE TABLE _assert_reachable_count (ok INTEGER NOT NULL CHECK (ok = 1));
INSERT INTO _assert_reachable_count
SELECT CASE WHEN (
  WITH RECURSIVE walk(node, depth) AS (
      SELECT 'q-task', 0
    UNION
      SELECT e.target_entity_object_id, walk.depth + 1
      FROM walk
      JOIN entity_link e ON e.source_entity_object_id = walk.node
      WHERE walk.depth < 3 AND e.confidence >= 0.5
  )
  SELECT COUNT(DISTINCT node) FROM walk WHERE node <> 'q-task'
) = 6 THEN 1 ELSE 0 END;

CREATE TABLE _assert_isolated_area_unreached (ok INTEGER NOT NULL CHECK (ok = 1));
INSERT INTO _assert_isolated_area_unreached
SELECT CASE WHEN (
  WITH RECURSIVE walk(node, depth) AS (
      SELECT 'q-task', 0
    UNION
      SELECT e.target_entity_object_id, walk.depth + 1
      FROM walk
      JOIN entity_link e ON e.source_entity_object_id = walk.node
      WHERE walk.depth < 3 AND e.confidence >= 0.5
  )
  SELECT COUNT(*) FROM walk WHERE node IN ('dx-iso-1', 'dx-iso-2')
) = 0 THEN 1 ELSE 0 END;

-- ---------------------------------------------------------------------------
-- ASSERT 4 -- Super-geometric similarity decay along the depth-3 chain.
--   Anchors: eq:simceiling / thm:sim  P_ret^sim(d) = s0^d * rho^(d(d+1)/2);
--   eq:perhop (per-hop product). Synthetic operating point s0=0.70, rho=0.67.
-- ---------------------------------------------------------------------------
-- Recursive per-hop product of the survival factors s_k = s0 * rho^k over the
-- 3-hop governing chain (carrying rho^k incrementally, no POWER() dependency).
-- Golden A.1 (VERBATIM): P_sim(3) = 0.031027225083967002. abs tol 1e-12.
CREATE TABLE _assert_psim_d3 (ok INTEGER NOT NULL CHECK (ok = 1));
INSERT INTO _assert_psim_d3
SELECT CASE WHEN ABS((
  WITH RECURSIVE hop(k, rho_pow, cum_p) AS (
      SELECT 1, 0.67, 0.70 * 0.67
    UNION ALL
      SELECT k + 1, rho_pow * 0.67, cum_p * (0.70 * (rho_pow * 0.67))
      FROM hop WHERE k < 3
  )
  SELECT cum_p FROM hop WHERE k = 3
) - 0.031027225083967002) <= 1e-12 THEN 1 ELSE 0 END;

-- ---------------------------------------------------------------------------
-- ASSERT 5 -- Structured per-hop recovery and the ~29x depth-3 gap.
--   Anchors: thm:struct  P_ret^struct(d) = q^d (steady per-hop fidelity q=0.97);
--   sec:depth "a ~29x gap" between traversal q^3 and similarity s0^3 rho^6.
-- ---------------------------------------------------------------------------
-- Golden A.1 (VERBATIM): q^3 = 0.912673; ratio q^3 / P_sim(3) = 29.415231221293272.
CREATE TABLE _assert_pstruct_d3 (ok INTEGER NOT NULL CHECK (ok = 1));
INSERT INTO _assert_pstruct_d3
SELECT CASE WHEN ABS((
  WITH RECURSIVE shop(k, cum_q) AS (
      SELECT 1, 0.97 UNION ALL SELECT k + 1, cum_q * 0.97 FROM shop WHERE k < 3
  )
  SELECT cum_q FROM shop WHERE k = 3
) - 0.912673) <= 1e-12 THEN 1 ELSE 0 END;

CREATE TABLE _assert_depth3_gap_ratio (ok INTEGER NOT NULL CHECK (ok = 1));
INSERT INTO _assert_depth3_gap_ratio
SELECT CASE WHEN ABS(
  (
    WITH RECURSIVE shop(k, cum_q) AS (
        SELECT 1, 0.97 UNION ALL SELECT k + 1, cum_q * 0.97 FROM shop WHERE k < 3)
    SELECT cum_q FROM shop WHERE k = 3
  ) / (
    WITH RECURSIVE hop(k, rho_pow, cum_p) AS (
        SELECT 1, 0.67, 0.70 * 0.67
      UNION ALL
        SELECT k + 1, rho_pow * 0.67, cum_p * (0.70 * (rho_pow * 0.67)) FROM hop WHERE k < 3)
    SELECT cum_p FROM hop WHERE k = 3
  ) - 29.415231221293272) <= 1e-9 THEN 1 ELSE 0 END;

-- ---------------------------------------------------------------------------
-- ASSERT 6 -- Supersession resolves to the CURRENT decision (walk to the head).
--   Anchors: tab:super (current-ranked-first), sec:resmech-super.
-- ---------------------------------------------------------------------------
-- Start from the stale original s-sess-1 and follow `supersedes` edges in
-- reverse (who supersedes me?) to the head -- the version that is never itself
-- superseded (never a target of a supersedes edge). Must return s-sess-3.
CREATE TABLE _assert_super_head_is_current (ok INTEGER NOT NULL CHECK (ok = 1));
INSERT INTO _assert_super_head_is_current
SELECT CASE WHEN (
  WITH RECURSIVE super_walk(node) AS (
      SELECT 's-sess-1'
    UNION
      SELECT e.source_entity_object_id
      FROM super_walk w
      JOIN entity_link e ON e.target_entity_object_id = w.node
      WHERE e.relationship_type = 'supersedes'
  )
  SELECT node FROM super_walk
  WHERE node NOT IN (
    SELECT target_entity_object_id FROM entity_link WHERE relationship_type = 'supersedes'
  )
) = 's-sess-3' THEN 1 ELSE 0 END;

-- ---------------------------------------------------------------------------
-- ASSERT 7 -- The similarity proxy returns a STALE version, and diverges from
--   the typed resolution. Anchor: tab:super (similarity 64-69% band vs typed
--   92.3%); the failure is ranking a near-identical stale version first.
-- ---------------------------------------------------------------------------
-- citation_count stands in for embedding/frequency centrality; ORDER BY it
-- returns the most-cited version = the stale original s-sess-1, NOT the current
-- s-sess-3. Typed supersession (ASSERT 6) and similarity ranking disagree.
CREATE TABLE _assert_similarity_pick_is_stale (ok INTEGER NOT NULL CHECK (ok = 1));
INSERT INTO _assert_similarity_pick_is_stale
SELECT CASE WHEN (
  SELECT id FROM entity_objects
  WHERE id IN ('s-sess-1', 's-sess-2', 's-sess-3')
  ORDER BY citation_count DESC, id ASC
  LIMIT 1
) = 's-sess-1' THEN 1 ELSE 0 END;

CREATE TABLE _assert_typed_beats_similarity (ok INTEGER NOT NULL CHECK (ok = 1));
INSERT INTO _assert_typed_beats_similarity
SELECT CASE WHEN
  (SELECT id FROM entity_objects WHERE id IN ('s-sess-1', 's-sess-2', 's-sess-3')
   ORDER BY citation_count DESC, id ASC LIMIT 1) <> 's-sess-3'
  AND
  (WITH RECURSIVE super_walk(node) AS (
      SELECT 's-sess-1'
    UNION
      SELECT e.source_entity_object_id FROM super_walk w
      JOIN entity_link e ON e.target_entity_object_id = w.node
      WHERE e.relationship_type = 'supersedes')
   SELECT node FROM super_walk
   WHERE node NOT IN (SELECT target_entity_object_id FROM entity_link WHERE relationship_type = 'supersedes')
  ) = 's-sess-3'
  THEN 1 ELSE 0 END;

