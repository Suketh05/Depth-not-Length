/**
 * verify_paper_consistency.ts — a one-command audit that the published tables of
 * "Depth, Not Length: A Benchmark and Theory of Memory, Retrieval, and Decision
 * Compliance in LLM Coding Agents" are INTERNALLY consistent.
 *
 * It loads the committed results/data CSVs and verifies the paper-internal
 * identities of golden-pack section B — quantities a reader can recompute from
 * the published numbers alone, no model or dataset required:
 *
 *   B.1  tab:realret  — use factor kappa = compliance / recall for all 7
 *        published arms; |compliance/recall - kappa| <= 0.0005 (eq:factor,
 *        P_comply = P_ret * kappa), and the ratio reproduces golden-pack B.1
 *        full precision. The none arm's kappa is intentionally unpublished.
 *   B.2  tab:tok_context_winrate — session-token win rate 2880 / 3600 = 0.800,
 *        with won + lost + tied = total (0 ties).
 *   B.3  tab:scorecard — 22 W + 14 m + 5 l = 41 axes.
 *   B.4  tab:synthall  — synthetic compliance ordering (Brief top, none bottom).
 *   B.4  tab:slope     — depth-slope arithmetic: slope = comply(d3) - comply(d1)
 *        reproduces paper_depth_slope.csv, as does the Brief-minus-best-similarity
 *        gap at each depth.
 *
 * Usage (out-of-tree runtime, per tools/seed conventions):
 *   npx tsc -p tsconfig.json --noEmit false --outDir /tmp/tsbuild && \
 *     node /tmp/tsbuild/verify_paper_consistency.js
 *   MEMBENCH_DATA_DIR=/path/to/results/data node .../verify_paper_consistency.js
 *
 * Exit code is 0 iff every identity holds; any violation exits non-zero so the
 * check is usable as a CI/pre-publication gate.
 */
import { fileURLToPath } from "node:url";
import {
  parseCsv,
  loadRealcodePooled,
  loadTokenEconomics,
  loadScorecard,
  loadSyntheticCompliance,
  loadDepthSlope,
  tokenMetric,
  scorecardCount,
  depthSlopeValue,
  resolveDataDir,
  PAPER_TAB_SLOPE,
  PAPER_KAPPA_GOLDEN,
} from "./paper_data.js";

/** Outcome of one verified identity, rendered as a table row. */
interface CheckResult {
  readonly identity: string;
  readonly sourceFile: string;
  readonly paperLabel: string;
  readonly expected: string;
  readonly actual: string;
  readonly ok: boolean;
}

/** Absolute-tolerance float comparison. */
function approxEqual(a: number, b: number, tol: number): boolean {
  return Math.abs(a - b) <= tol;
}

/** Round-half-away shortest decimal for display (keeps the table readable). */
function fmt(n: number, places = 6): string {
  return Number.isInteger(n) ? String(n) : n.toFixed(places);
}

// ---------------------------------------------------------------------------
// CSV-parser self-tests — deterministic edge cases run before any identity so a
// broken parser cannot silently pass the audit. Covers: empty input, quoted
// field with an embedded comma, doubled-quote escape, trailing empty field,
// and CRLF line endings.
// ---------------------------------------------------------------------------

function assertParse(actual: string[][], expected: string[][], label: string): void {
  const got = JSON.stringify(actual);
  const want = JSON.stringify(expected);
  if (got !== want) {
    throw new Error(`CSV parser self-test failed [${label}]: got ${got}, want ${want}`);
  }
}

function runParserSelfTests(): void {
  assertParse(parseCsv(""), [], "empty input");
  assertParse(parseCsv("a,b,c"), [["a", "b", "c"]], "single unterminated row");
  assertParse(parseCsv("a,b\n"), [["a", "b"]], "trailing newline drops phantom row");
  assertParse(parseCsv("a,"), [["a", ""]], "trailing empty field, no newline");
  assertParse(
    parseCsv('h1,h2,h3\n"x,y","he said ""hi""",\n'),
    [
      ["h1", "h2", "h3"],
      ["x,y", 'he said "hi"', ""],
    ],
    "quoted comma + doubled-quote escape + trailing empty",
  );
  assertParse(parseCsv("p,q\r\n1,2\r\n"), [["p", "q"], ["1", "2"]], "CRLF line endings");
  // Mirrors the real none-arm row: a trailing blank numeric cell -> "".
  assertParse(parseCsv("none,0.219,0.073,0.052,\n"), [["none", "0.219", "0.073", "0.052", ""]], "blank trailing kappa cell");
}

/** Render the collected identities as an aligned status table. */
function renderTable(results: readonly CheckResult[]): string {
  const headers = ["STATUS", "IDENTITY", "SOURCE FILE", "PAPER LABEL", "EXPECTED", "ACTUAL"];
  const rows = results.map((r) => [
    r.ok ? "PASS" : "FAIL",
    r.identity,
    r.sourceFile,
    r.paperLabel,
    r.expected,
    r.actual,
  ]);
  const widths = headers.map((h, col) =>
    Math.max(h.length, ...rows.map((row) => (row[col] ?? "").length)),
  );
  const line = (cells: readonly string[]): string =>
    cells.map((c, col) => c.padEnd(widths[col] ?? 0)).join("  ");
  const sep = widths.map((w) => "-".repeat(w)).join("  ");
  return [line(headers), sep, ...rows.map(line)].join("\n");
}
