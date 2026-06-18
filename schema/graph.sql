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
