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
