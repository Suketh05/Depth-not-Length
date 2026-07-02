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
import { realpathSync } from "node:fs";
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

// ---------------------------------------------------------------------------
// Identity B.1 — use factor kappa = compliance / recall (tab:realret, eq:factor).
// For each of the 7 published arms: (1) the CSV-internal round trip
// |compliance/recall - kappa_published| <= 0.0005; (2) round(ratio,3) equals the
// published 3-dp kappa; (3) the ratio reproduces golden-pack B.1 full precision.
// The none arm carries no published kappa (blank) and is asserted null.
// ---------------------------------------------------------------------------

const KAPPA_TOL = 0.0005; // task tolerance on |compliance/recall - kappa|
const GOLDEN_TOL = 1e-12; // float64 identity vs the golden-pack B.1 constants

function checkKappaRoundTrip(): CheckResult[] {
  const rows = loadRealcodePooled();
  const results: CheckResult[] = [];
  for (const row of rows) {
    if (row.kappa === null) {
      results.push({
        identity: `${row.arm}: kappa unpublished (no retrieval mechanism to factorize)`,
        sourceFile: "paper_realcode_pooled.csv",
        paperLabel: "tab:realret / B.1",
        expected: "null",
        actual: "null",
        ok: true,
      });
      continue;
    }
    const ratio = row.compliance / row.recall;
    const golden = PAPER_KAPPA_GOLDEN[row.arm];
    const withinTol = approxEqual(ratio, row.kappa, KAPPA_TOL);
    const roundsToPublished = Math.round(ratio * 1000) / 1000 === row.kappa;
    const matchesGolden = golden !== undefined && approxEqual(ratio, golden, GOLDEN_TOL);
    results.push({
      identity: `${row.arm}: compliance/recall = ${row.compliance}/${row.recall} ~= kappa`,
      sourceFile: "paper_realcode_pooled.csv",
      paperLabel: "tab:realret / eq:factor",
      expected: `${row.kappa} (|d|<=${KAPPA_TOL})`,
      actual: `${fmt(ratio, 10)} (|d|=${fmt(Math.abs(ratio - row.kappa), 5)})`,
      ok: withinTol && roundsToPublished && matchesGolden,
    });
  }
  return results;
}

const IDENTITY_TOL = 1e-9; // integer/decimal identities carry only float dust

// ---------------------------------------------------------------------------
// Identity B.2 — session-token win rate (tab:tok_context_winrate).
// 2880 / 3600 = 0.800; won + lost + tied = total with 0 ties.
// ---------------------------------------------------------------------------

function checkTokenWinRate(): CheckResult[] {
  const rows = loadTokenEconomics();
  const won = tokenMetric(rows, "matchups_won");
  const lost = tokenMetric(rows, "matchups_lost");
  const tied = tokenMetric(rows, "matchups_tied");
  const total = tokenMetric(rows, "matchups_total");
  const pct = tokenMetric(rows, "session_token_win_rate_pct");
  const rate = won / total;
  const rateOk =
    approxEqual(rate, 0.8, IDENTITY_TOL) &&
    approxEqual(rate, pct / 100, IDENTITY_TOL) &&
    Math.round(rate * 1000) / 1000 === 0.8;
  const conservationOk = approxEqual(won + lost + tied, total, IDENTITY_TOL) && tied === 0;
  return [
    {
      identity: `win rate = matchups_won/matchups_total = ${won}/${total}`,
      sourceFile: "paper_token_economics.csv",
      paperLabel: "tab:tok_context_winrate",
      expected: `0.800 (= ${pct}%)`,
      actual: fmt(rate, 6),
      ok: rateOk,
    },
    {
      identity: `won + lost + tied = ${won} + ${lost} + ${tied}`,
      sourceFile: "paper_token_economics.csv",
      paperLabel: "tab:tok_context_winrate",
      expected: `${total} (ties = 0)`,
      actual: `${won + lost + tied} (ties = ${tied})`,
      ok: conservationOk,
    },
  ];
}

// ---------------------------------------------------------------------------
// Identity B.3 — unbiased scorecard tally (tab:scorecard): 22 + 14 + 5 = 41.
// ---------------------------------------------------------------------------

function checkScorecard(): CheckResult[] {
  const rows = loadScorecard();
  const win = scorecardCount(rows, "win");
  const marginal = scorecardCount(rows, "marginal");
  const loss = scorecardCount(rows, "loss");
  const total = win + marginal + loss;
  return [
    {
      identity: `win + marginal + loss = ${win} + ${marginal} + ${loss}`,
      sourceFile: "paper_scorecard.csv",
      paperLabel: "tab:scorecard",
      expected: "41",
      actual: String(total),
      ok: total === 41 && win === 22 && marginal === 14 && loss === 5,
    },
  ];
}

// ---------------------------------------------------------------------------
// Identity B.4a — synthetic compliance ordering (tab:synthall): the typed store
// (brief) tops the arms and the context-free arm (none) is the floor, and the
// CSV is listed in non-increasing compliance order.
// ---------------------------------------------------------------------------

function checkSyntheticOrdering(): CheckResult[] {
  const rows = loadSyntheticCompliance();
  if (rows.length === 0) {
    throw new Error("paper_synthetic_compliance.csv has no rows");
  }
  let topArm = rows[0]!.arm;
  let topVal = rows[0]!.compliance;
  let botArm = rows[0]!.arm;
  let botVal = rows[0]!.compliance;
  let descending = true;
  for (let i = 0; i < rows.length; i += 1) {
    const cur = rows[i]!;
    if (cur.compliance > topVal) {
      topVal = cur.compliance;
      topArm = cur.arm;
    }
    if (cur.compliance < botVal) {
      botVal = cur.compliance;
      botArm = cur.arm;
    }
    if (i > 0 && cur.compliance > rows[i - 1]!.compliance + IDENTITY_TOL) {
      descending = false;
    }
  }
  return [
    {
      identity: "argmax compliance = brief (typed store leads)",
      sourceFile: "paper_synthetic_compliance.csv",
      paperLabel: "tab:synthall",
      expected: "brief",
      actual: `${topArm} (${fmt(topVal)})`,
      ok: topArm === "brief",
    },
    {
      identity: "argmin compliance = none (context-free floor)",
      sourceFile: "paper_synthetic_compliance.csv",
      paperLabel: "tab:synthall",
      expected: "none",
      actual: `${botArm} (${fmt(botVal)})`,
      ok: botArm === "none",
    },
    {
      identity: "rows listed in non-increasing compliance order",
      sourceFile: "paper_synthetic_compliance.csv",
      paperLabel: "tab:synthall",
      expected: "true",
      actual: String(descending),
      ok: descending,
    },
  ];
}

// ---------------------------------------------------------------------------
// Identity B.4b — depth-slope arithmetic (tab:slope). The per-depth compliance
// grid PAPER_TAB_SLOPE (verbatim from the paper) must reproduce the derived
// columns committed to paper_depth_slope.csv:
//   per-arm slope = comply(d3) - comply(d1)
//   gap(dk)       = comply(brief, dk) - max over similarity arms of comply(., dk)
// ---------------------------------------------------------------------------

function checkDepthSlope(): CheckResult[] {
  const csv = loadDepthSlope();
  const results: CheckResult[] = [];
  const brief = PAPER_TAB_SLOPE.find((r) => r.arm === "brief");
  if (brief === undefined) {
    throw new Error("PAPER_TAB_SLOPE is missing the brief row");
  }
  for (const arm of PAPER_TAB_SLOPE) {
    const slope = arm.d3 - arm.d1;
    const published = depthSlopeValue(csv, arm.arm);
    results.push({
      identity: `${arm.arm}: slope = comply(d3) - comply(d1) = ${arm.d3} - ${arm.d1}`,
      sourceFile: "paper_depth_slope.csv",
      paperLabel: "tab:slope",
      expected: fmt(published),
      actual: fmt(slope),
      ok: approxEqual(slope, published, IDENTITY_TOL),
    });
  }
  const depths: readonly [number, (r: (typeof PAPER_TAB_SLOPE)[number]) => number][] = [
    [1, (r) => r.d1],
    [2, (r) => r.d2],
    [3, (r) => r.d3],
  ];
  const sims = PAPER_TAB_SLOPE.filter((r) => r.isSimilarity);
  for (const [d, pick] of depths) {
    const bestSim = Math.max(...sims.map(pick));
    const gap = pick(brief) - bestSim;
    const published = depthSlopeValue(csv, `brief_minus_best_similarity@d${d}`);
    results.push({
      identity: `d${d}: brief - best_similarity = ${fmt(pick(brief))} - ${fmt(bestSim)}`,
      sourceFile: "paper_depth_slope.csv",
      paperLabel: "tab:slope",
      expected: fmt(published),
      actual: fmt(gap),
      ok: approxEqual(gap, published, IDENTITY_TOL),
    });
  }
  return results;
}

// ---------------------------------------------------------------------------
// Entry point — run the parser self-tests, collect every identity, print the
// report, and exit non-zero on any violation.
// ---------------------------------------------------------------------------

function runAllChecks(): CheckResult[] {
  return [
    ...checkKappaRoundTrip(),
    ...checkTokenWinRate(),
    ...checkScorecard(),
    ...checkSyntheticOrdering(),
    ...checkDepthSlope(),
  ];
}

function main(): void {
  runParserSelfTests();
  const dataDir = resolveDataDir();
  console.log(`Depth, Not Length — paper-CSV internal-consistency audit`);
  console.log(`data dir: ${dataDir}\n`);
  const results = runAllChecks();
  console.log(renderTable(results));
  const failures = results.filter((r) => !r.ok);
  const passed = results.length - failures.length;
  console.log(`\n${passed}/${results.length} identities hold.`);
  if (failures.length > 0) {
    console.error(`\nFAILED (${failures.length}):`);
    for (const f of failures) {
      console.error(`  - ${f.identity} [${f.paperLabel}]: expected ${f.expected}, got ${f.actual}`);
    }
    process.exit(1);
  }
  console.log("All paper-internal identities are consistent.");
}

/**
 * Detect direct invocation robustly across symlinked temp dirs (macOS /tmp ->
 * /private/tmp): compare the real paths of this module and argv[1], not the raw
 * strings, so the out-of-tree runtime gate reports instead of silently exiting.
 */
function isInvokedDirectly(): boolean {
  const entry = process.argv[1];
  if (entry === undefined) {
    return false;
  }
  try {
    return realpathSync(fileURLToPath(import.meta.url)) === realpathSync(entry);
  } catch {
    return false;
  }
}

if (isInvokedDirectly()) {
  try {
    main();
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    process.exit(1);
  }
}
