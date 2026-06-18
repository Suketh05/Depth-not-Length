/**
 * Seed a Brief workspace with the dcbench decisions ahead of a benchmark run.
 *
 * The `brief_live` arm queries a real, pre-seeded Brief workspace; Brief's ingestion
 * is interactive and approval-gated, so the workspace must be seeded out of band
 * (the upstream dcbench harness does the same via a separate `seed.ts` step). This
 * tool reads the vendored decision records and posts them to Brief's API, mapping
 * each decision's stable id so retrieval scoring can join back to the gold ids.
 *
 * Usage:
 *   BRIEF_OAUTH_TOKEN=... node dist/seed.js [path/to/decisions.json]
 *   node dist/seed.js --dry-run            # print the payload without seeding
 */
import { readFileSync } from "node:fs";

interface DcDecision {
  id: string;
  topic: string;
  decision: string;
  rationale: string;
  category: string;
  severity: string;
}

interface DecisionsFile {
  decisions: DcDecision[];
}

interface SeedPayload {
  external_id: string;
  decision: string;
  rationale: string;
  topic: string;
  tags: string[];
}

const DEFAULT_PATH = "src/membench/datasets/data/dcbench/decisions.json";
const ENDPOINT = process.env.BRIEF_API ?? "https://app.briefhq.ai/api/v1/decisions";

function toPayload(d: DcDecision): SeedPayload {
  return {
    external_id: d.id,
    decision: d.decision,
    rationale: d.rationale,
    topic: d.topic,
    tags: [d.category, d.severity],
  };
}

async function seed(path: string, dryRun: boolean): Promise<void> {
  const file = JSON.parse(readFileSync(path, "utf-8")) as DecisionsFile;
  const token = process.env.BRIEF_OAUTH_TOKEN;
  if (!dryRun && (token === undefined || token === "")) {
    throw new Error("BRIEF_OAUTH_TOKEN is required (or pass --dry-run); never hardcode it");
  }
  let seeded = 0;
  for (const decision of file.decisions) {
    const payload = toPayload(decision);
    if (dryRun) {
      console.log(`[dry-run] ${payload.external_id}: ${payload.decision.slice(0, 60)}…`);
      continue;
    }
    const res = await fetch(ENDPOINT, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      throw new Error(`seeding ${payload.external_id} failed: ${res.status} ${res.statusText}`);
    }
    seeded += 1;
    console.log(`seeded ${payload.external_id}`);
  }
  console.log(dryRun ? `[dry-run] ${file.decisions.length} decisions` : `seeded ${seeded} decisions`);
}

const args = process.argv.slice(2);
const dryRun = args.includes("--dry-run");
const pathArg = args.find((a) => !a.startsWith("--")) ?? DEFAULT_PATH;

seed(pathArg, dryRun).catch((err: unknown) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});
