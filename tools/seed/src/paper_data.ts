/**
 * paper_data.ts — typed loaders for the committed paper CSVs, plus the
 * paper-table golden constants consumed by verify_paper_consistency.ts.
 *
 * This is the data layer of a one-command audit that the published tables of
 * "Depth, Not Length: A Benchmark and Theory of Memory, Retrieval, and Decision
 * Compliance in LLM Coding Agents" are INTERNALLY consistent. The CSVs under
 * results/data/ are the published-number-of-record; this module reads them and
 * exposes the paper anchors each one backs:
 *   - results/data/paper_realcode_pooled.csv     -> tab:realret  (use factor kappa = compliance / recall, eq:factor)
 *   - results/data/paper_token_economics.csv     -> tab:tok_context_winrate (session-token win rate 2880/3600)
 *   - results/data/paper_scorecard.csv           -> tab:scorecard (22 W + 14 m + 5 l = 41 axes)
 *   - results/data/paper_synthetic_compliance.csv-> tab:synthall  (synthetic compliance ordering)
 *   - results/data/paper_depth_slope.csv         -> tab:slope     (depth slope + Brief-minus-best-similarity gap)
 *
 * A dependency-free RFC-4180 reader (quoted fields, doubled-quote escape, CRLF)
 * keeps tools/seed free of a CSV library, per the ESM/NodeNext conventions of
 * this package (no new deps; relative imports carry an explicit .js extension).
 */
import { readFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

/**
 * Parse CSV text into a matrix of string cells (RFC 4180 subset).
 *
 * Handles: comma-separated fields; fields quoted with `"` that may themselves
 * contain commas, newlines, or a doubled-quote `""` escape; `\n`, `\r`, and
 * `\r\n` line terminators. A blank trailing line yields no extra record beyond
 * the empty-row filtering done by {@link readCsvTable}. Unquoted whitespace is
 * preserved verbatim (callers trim where a numeric parse is expected).
 */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;
  const n = text.length;
  let i = 0;
  while (i < n) {
    const ch = text[i];
    if (ch === undefined) {
      break;
    }
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 2;
          continue;
        }
        inQuotes = false;
        i += 1;
        continue;
      }
      field += ch;
      i += 1;
      continue;
    }
    if (ch === '"') {
      inQuotes = true;
      i += 1;
      continue;
    }
    if (ch === ",") {
      row.push(field);
      field = "";
      i += 1;
      continue;
    }
    if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") {
        i += 1;
      }
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
      i += 1;
      continue;
    }
    field += ch;
    i += 1;
  }
  if (field !== "" || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

/** A parsed CSV table: the header row and the remaining data rows. */
export interface CsvTable {
  readonly header: readonly string[];
  readonly rows: readonly (readonly string[])[];
}

/**
 * Read and parse a CSV file into a header + data rows. Fully-blank rows (a lone
 * empty trailing line) are dropped so a trailing newline never yields a phantom
 * record. Throws if the file has no header row.
 */
export function readCsvTable(path: string): CsvTable {
  const text = readFileSync(path, "utf-8");
  const raw = parseCsv(text).filter((r) => !(r.length === 1 && r[0] === ""));
  const header = raw[0];
  if (header === undefined) {
    throw new Error(`empty CSV (no header row): ${path}`);
  }
  return { header, rows: raw.slice(1) };
}

/** Index of a named column, throwing with context if it is absent. */
export function columnIndex(header: readonly string[], name: string): number {
  const idx = header.indexOf(name);
  if (idx < 0) {
    throw new Error(`missing expected column "${name}" (have: ${header.join(", ")})`);
  }
  return idx;
}

/** Parse a required numeric cell, throwing on empty/non-numeric with context. */
export function parseNumberCell(cell: string | undefined, ctx: string): number {
  if (cell === undefined || cell.trim() === "") {
    throw new Error(`expected a number at ${ctx}, got empty cell`);
  }
  const value = Number(cell.trim());
  if (Number.isNaN(value)) {
    throw new Error(`expected a number at ${ctx}, got ${JSON.stringify(cell)}`);
  }
  return value;
}

/** Parse an optional numeric cell; an empty cell yields null (not an error). */
export function parseOptionalNumberCell(cell: string | undefined, ctx: string): number | null {
  if (cell === undefined || cell.trim() === "") {
    return null;
  }
  return parseNumberCell(cell, ctx);
}

/** Read a required string cell, throwing on a missing column value. */
export function stringCell(row: readonly string[], idx: number, ctx: string): string {
  const cell = row[idx];
  if (cell === undefined) {
    throw new Error(`missing cell at ${ctx} (column index ${idx})`);
  }
  return cell;
}

/**
 * Locate the repo's results/data directory by walking up from `startDir`.
 * Deterministic and CWD-tolerant: the runtime verifier compiles out-of-tree and
 * is launched from tools/seed, so we search upward for a directory that holds
 * paper_realcode_pooled.csv. `MEMBENCH_DATA_DIR` overrides the search entirely.
 */
export function resolveDataDir(startDir: string = process.cwd()): string {
  const override = process.env.MEMBENCH_DATA_DIR;
  if (override !== undefined && override !== "") {
    return override;
  }
  let dir = resolve(startDir);
  for (let hops = 0; hops < 12; hops += 1) {
    const candidate = join(dir, "results", "data");
    if (existsSync(join(candidate, "paper_realcode_pooled.csv"))) {
      return candidate;
    }
    const parent = dirname(dir);
    if (parent === dir) {
      break;
    }
    dir = parent;
  }
  throw new Error("could not locate results/data (set MEMBENCH_DATA_DIR to override)");
}

// ---------------------------------------------------------------------------
// Typed loaders — one per committed paper CSV, each carrying its paper anchor.
// ---------------------------------------------------------------------------

/**
 * One arm of tab:realret — real-code retrieval and use, pooled dcbench+swebench
 * (Claude), sorted by recall. `kappa` is the published use factor; it is `null`
 * for the `none` arm, whose kappa the paper deliberately leaves blank (no
 * retrieval mechanism to factorize — golden-pack B.1).
 */
export interface RealcodePooledRow {
  readonly arm: string;
  readonly recall: number;
  readonly compliance: number;
  readonly precision: number;
  readonly kappa: number | null;
}

/** Load results/data/paper_realcode_pooled.csv (tab:realret). */
export function loadRealcodePooled(dataDir: string = resolveDataDir()): RealcodePooledRow[] {
  const path = join(dataDir, "paper_realcode_pooled.csv");
  const table = readCsvTable(path);
  const armIdx = columnIndex(table.header, "arm");
  const recallIdx = columnIndex(table.header, "recall");
  const complianceIdx = columnIndex(table.header, "compliance");
  const precisionIdx = columnIndex(table.header, "precision");
  const kappaIdx = columnIndex(table.header, "kappa");
  return table.rows.map((row) => {
    const arm = stringCell(row, armIdx, `${path} arm`);
    return {
      arm,
      recall: parseNumberCell(row[recallIdx], `${path} ${arm}.recall`),
      compliance: parseNumberCell(row[complianceIdx], `${path} ${arm}.compliance`),
      precision: parseNumberCell(row[precisionIdx], `${path} ${arm}.precision`),
      kappa: parseOptionalNumberCell(row[kappaIdx], `${path} ${arm}.kappa`),
    };
  });
}

/**
 * One row of paper_token_economics.csv — a flat metric/value/note ledger backing
 * tab:tok_context_winrate and tab:tok_context_by_competitor. The note field
 * carries the paper label in brackets.
 */
export interface TokenEconomicsRow {
  readonly metric: string;
  readonly value: number;
  readonly note: string;
}

/** Load results/data/paper_token_economics.csv, keyed lookups via {@link tokenMetric}. */
export function loadTokenEconomics(dataDir: string = resolveDataDir()): TokenEconomicsRow[] {
  const path = join(dataDir, "paper_token_economics.csv");
  const table = readCsvTable(path);
  const metricIdx = columnIndex(table.header, "metric");
  const valueIdx = columnIndex(table.header, "value");
  const noteIdx = columnIndex(table.header, "note");
  return table.rows.map((row) => {
    const metric = stringCell(row, metricIdx, `${path} metric`);
    return {
      metric,
      value: parseNumberCell(row[valueIdx], `${path} ${metric}.value`),
      note: row[noteIdx] ?? "",
    };
  });
}

/** Look up a single token-economics metric by name, throwing if absent. */
export function tokenMetric(rows: readonly TokenEconomicsRow[], metric: string): number {
  const found = rows.find((r) => r.metric === metric);
  if (found === undefined) {
    throw new Error(`token-economics metric "${metric}" not found`);
  }
  return found.value;
}

/** One row of paper_scorecard.csv (tab:scorecard): an outcome bucket and its count. */
export interface ScorecardRow {
  readonly outcome: string;
  readonly count: number;
}

/** Load results/data/paper_scorecard.csv (tab:scorecard). */
export function loadScorecard(dataDir: string = resolveDataDir()): ScorecardRow[] {
  const path = join(dataDir, "paper_scorecard.csv");
  const table = readCsvTable(path);
  const outcomeIdx = columnIndex(table.header, "outcome");
  const countIdx = columnIndex(table.header, "count");
  return table.rows.map((row) => {
    const outcome = stringCell(row, outcomeIdx, `${path} outcome`);
    return { outcome, count: parseNumberCell(row[countIdx], `${path} ${outcome}.count`) };
  });
}

/** Look up a single scorecard bucket count by outcome, throwing if absent. */
export function scorecardCount(rows: readonly ScorecardRow[], outcome: string): number {
  const found = rows.find((r) => r.outcome === outcome);
  if (found === undefined) {
    throw new Error(`scorecard outcome "${outcome}" not found`);
  }
  return found.count;
}

/**
 * One arm of paper_synthetic_compliance.csv — synthetic-suite mean compliance
 * over 120 tasks (Claude), the tab:synthall compliance column, listed best-first.
 */
export interface SyntheticComplianceRow {
  readonly arm: string;
  readonly compliance: number;
}

/** Load results/data/paper_synthetic_compliance.csv (tab:synthall compliance column). */
export function loadSyntheticCompliance(dataDir: string = resolveDataDir()): SyntheticComplianceRow[] {
  const path = join(dataDir, "paper_synthetic_compliance.csv");
  const table = readCsvTable(path);
  const armIdx = columnIndex(table.header, "arm");
  const complianceIdx = columnIndex(table.header, "compliance");
  return table.rows.map((row) => {
    const arm = stringCell(row, armIdx, `${path} arm`);
    return { arm, compliance: parseNumberCell(row[complianceIdx], `${path} ${arm}.compliance`) };
  });
}

/**
 * One row of paper_depth_slope.csv (tab:slope). Two shapes share the file:
 *   - per-arm compliance slope rows  (arm_or_group in {brief,bm25,...}, slope = comply(d3)-comply(d1))
 *   - Brief-minus-best-similarity gap rows (arm_or_group = "brief_minus_best_similarity@d{1,2,3}")
 * `slopeOrGap` is signed; the CSV writes an explicit `+`/`-` sign which Number() parses.
 */
export interface DepthSlopeRow {
  readonly armOrGroup: string;
  readonly slopeOrGap: number;
  readonly note: string;
}

/** Load results/data/paper_depth_slope.csv (tab:slope). */
export function loadDepthSlope(dataDir: string = resolveDataDir()): DepthSlopeRow[] {
  const path = join(dataDir, "paper_depth_slope.csv");
  const table = readCsvTable(path);
  const keyIdx = columnIndex(table.header, "arm_or_group");
  const valIdx = columnIndex(table.header, "slope_or_gap");
  const noteIdx = columnIndex(table.header, "note");
  return table.rows.map((row) => {
    const armOrGroup = stringCell(row, keyIdx, `${path} arm_or_group`);
    return {
      armOrGroup,
      slopeOrGap: parseNumberCell(row[valIdx], `${path} ${armOrGroup}.slope_or_gap`),
      note: row[noteIdx] ?? "",
    };
  });
}

/** Look up a single depth-slope value by key, throwing if absent. */
export function depthSlopeValue(rows: readonly DepthSlopeRow[], key: string): number {
  const found = rows.find((r) => r.armOrGroup === key);
  if (found === undefined) {
    throw new Error(`depth-slope key "${key}" not found`);
  }
  return found.slopeOrGap;
}

// ---------------------------------------------------------------------------
// Paper-table GOLDEN constants — transcribed VERBATIM from the golden pack /
// the paper LaTeX; NEVER produced by running this code. These pin the checker
// to published values so a silently edited CSV cannot self-certify.
// ---------------------------------------------------------------------------

/**
 * PAPER_TAB_SLOPE — the per-depth synthetic compliance grid (Claude), paper
 * tab:slope, transcribed verbatim (golden-pack B.4). The committed
 * paper_depth_slope.csv stores only the derived slope/gap columns; this grid is
 * the source those columns must reproduce, which is exactly the depth-slope
 * arithmetic identity the checker enforces:
 *   slope(arm) = comply(d3) - comply(d1)                          [per-arm rows]
 *   gap(dk)    = comply(brief, dk) - max_{sim} comply(sim, dk)    [gap rows]
 * `isSimilarity` marks the arms eligible for the best-similarity maximum (Brief
 * is the typed store, excluded from its own comparison baseline).
 */
export interface SlopeDepthRow {
  readonly arm: string;
  readonly isSimilarity: boolean;
  readonly d1: number;
  readonly d2: number;
  readonly d3: number;
}

export const PAPER_TAB_SLOPE: readonly SlopeDepthRow[] = [
  { arm: "brief", isSimilarity: false, d1: 0.95, d2: 0.975, d3: 0.875 },
  { arm: "bm25", isSimilarity: true, d1: 0.925, d2: 0.925, d3: 0.775 },
  { arm: "tfidf", isSimilarity: true, d1: 0.925, d2: 0.925, d3: 0.8 },
  { arm: "dense", isSimilarity: true, d1: 0.925, d2: 0.9, d3: 0.75 },
  { arm: "hybrid_rrf", isSimilarity: true, d1: 0.925, d2: 0.95, d3: 0.775 },
  { arm: "rerank_ce", isSimilarity: true, d1: 0.925, d2: 0.9, d3: 0.725 },
];

/**
 * PAPER_KAPPA_GOLDEN — full-precision float64 kappa = compliance / recall for
 * each published tab:realret arm, transcribed VERBATIM from golden-pack B.1.
 * (The `none` arm is intentionally omitted — its kappa is unpublished.) Used to
 * prove the CSV's recall/compliance columns reproduce these exact ratios, on
 * top of the CSV-internal round-trip against its own published kappa column.
 */
export const PAPER_KAPPA_GOLDEN: Readonly<Record<string, number>> = {
  brief: 0.7031484257871063,
  dense: 0.6393700787401575,
  hybrid_rrf: 0.6439024390243903,
  tfidf: 0.6374172185430463,
  bm25: 0.5695364238410596,
  rerank_ce: 0.6072041166380789,
  raptor: 0.5636998254799302,
};
