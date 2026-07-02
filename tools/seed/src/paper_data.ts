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
