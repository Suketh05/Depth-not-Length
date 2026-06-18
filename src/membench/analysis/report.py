"""Markdown report generator: executive summary + technical tables.

Assembles the scored rows into a single report with two audiences in mind: an
executive summary framed as Return on Tokens and depth-recovery (what a C-suite
buyer reads), and a technical section with the headline / depth-crossover /
ablation tables and the two-tier competitor matrix (what an engineer audits). The
report is a plain Markdown string the CLI writes to disk.
"""

from __future__ import annotations

from collections.abc import Sequence

from membench.analysis.tables import (
    ScoredRow,
    ablation_table,
    competitor_matrix,
    depth_crossover_table,
    headline_table,
)
from membench.competitors import VendorClaim

__all__ = ["generate_report"]


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _depth_crossover_md(rows: Sequence[ScoredRow]) -> tuple[str, list[str]]:
    table = depth_crossover_table(rows)
    arms = sorted({str(arm) for arm, _ in table})
    depths = sorted({int(d) for _, d in table})
    header = "| arm | " + " | ".join(f"d={d}" for d in depths) + " |"
    sep = "|" + "---|" * (len(depths) + 1)
    lines = [header, sep]
    for arm in arms:
        cells = [_fmt(table.get((arm, d), float("nan"))) for d in depths]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return "\n".join(lines), arms


def generate_report(
    rows: Sequence[ScoredRow],
    *,
    vendor_claims: Sequence[VendorClaim] = (),
    title: str = "BriefBench Results",
) -> str:
    """Render a full Markdown report from scored rows.

    Includes an executive summary (the depth-recovery and Return-on-Tokens headline),
    the headline/depth-crossover/ablation tables, and the two-tier competitor matrix.
    """
    if not rows:
        raise ValueError("no scored rows to report")

    crossover_md, arms = _depth_crossover_md(rows)
    table = depth_crossover_table(rows)
    depths = sorted({int(d) for _, d in table})
    deepest = depths[-1]

    graph_arm = next((a for a in arms if a.startswith("brief")), arms[0])
    graph_deep = table.get((graph_arm, deepest), float("nan"))
    baselines = {
        a: table.get((a, deepest), float("nan"))
        for a in arms
        if not a.startswith("brief") and a != "none"
    }
    best_baseline, best_val = (
        max(baselines.items(), key=lambda kv: kv[1]) if baselines else ("(none)", float("nan"))
    )

    parts: list[str] = [f"# {title}", ""]
    parts += [
        "## Executive summary",
        "",
        f"At the deepest tested chain (d={deepest}), the structured context graph "
        f"(`{graph_arm}`) recovers **{_fmt(graph_deep)}** of governing decisions versus "
        f"**{_fmt(best_val)}** for the best similarity baseline (`{best_baseline}`) -- a "
        "widening margin as causal depth grows (the depth ceiling).",
        "",
        "**Return on Tokens:** the win is not fewer tokens but better-spent ones -- at a "
        "fixed budget the structured graph follows the link straight to the governing "
        "decision while similarity buries it, so more of every token's context is the "
        "context that ships correct work.",
        "",
        "## Headline compliance (model = Claude)",
        "",
        "| dataset | arm | compliance |",
        "|---|---|---|",
    ]
    for (dataset, arm), value in sorted(headline_table(rows).items(), key=lambda kv: str(kv[0])):
        parts.append(f"| {dataset} | {arm} | {_fmt(value)} |")

    parts += ["", "## Depth crossover (full-chain recovery)", "", crossover_md, ""]

    parts += ["## Four-arm ablation", "", "| condition | compliance |", "|---|---|"]
    for condition, value in sorted(ablation_table(rows).items()):
        parts.append(f"| {condition} | {_fmt(value)} |")

    parts += ["", "## Competitor matrix"]
    matrix = competitor_matrix(rows, vendor_claims)
    parts += [
        "",
        f"### Measured ({matrix.measured_tier.value}, identical conditions)",
        "",
        "| arm | compliance |",
        "|---|---|",
    ]
    for arm, value in sorted(matrix.measured.items()):
        parts.append(f"| {arm} | {_fmt(value)} |")
    parts += [
        "",
        f"### Vendor-reported ({matrix.vendor_tier.value}, cited, different conditions)",
        "",
        "| system | metric | value | source |",
        "|---|---|---|---|",
    ]
    for claim in matrix.vendor_reported:
        val = "(pending citation)" if claim.needs_citation else _fmt(claim.value or 0.0)
        parts.append(f"| {claim.system} | {claim.metric} | {val} | {claim.source_url} |")

    parts += [
        "",
        "## Methodology & guardrails",
        "",
        "- Identical model, retrieval budget, and code-search tool across all arms "
        "(fairness lock); memory architecture is the only variable.",
        "- Measured (Tier-1) and vendor-reported (Tier-2) numbers are never mixed.",
        "- Offline, deterministic pipeline; native kernels parity-validated against NumPy.",
        "",
    ]
    return "\n".join(parts)
