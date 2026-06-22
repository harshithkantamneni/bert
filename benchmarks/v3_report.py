"""v3 report — SINGLE MODEL (Claude). Two QA tracks over the same code corpora:
  CODE-FACT track   (exact-value gold, programmatic grading)
  SEMANTIC track    (conceptual gold, no greppable keyword, judge grading)
Plus Track A (BEIR retrieval quality, model-free) if present, and tokenomics.

Every arm runs on Claude, so the ONLY variable is the retrieval method — the
honest, model-controlled comparison. Emits benchmarks/V3_REPORT.md.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from benchmarks import v2_stats as ST

OUT = Path(__file__).resolve().parents[1] / "benchmarks/results/v2"
REPORT = Path(__file__).resolve().parents[1] / "benchmarks/V3_REPORT.md"

NAME = {
    "A0": "closed-book (no retrieval)", "A1": "naive-truncation", "A2": "smart-truncation",
    "A3": "bert hybrid-RAG", "A4": "vector-only", "A5": "BM25-only",
    "A6": "graph / Aider RepoMap", "A7grep": "agentic grep", "A_mcp": "bert via live MCP",
}
ARMS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7grep", "A_mcp"]


def _rows(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _by_arm(rows):
    d = collections.defaultdict(dict)  # arm -> {id: correct}
    for r in rows:
        d[r["arm"]][r["id"]] = int(r["correct"])
    return d


def _track_section(title, rows, tok_summary=None):
    if not rows:
        return f"## {title}\n\n_(no rows yet)_\n"
    ba = _by_arm(rows)
    # tokenomics straight from the rows (each carries real tokens_in/out)
    tok = collections.defaultdict(lambda: [0, 0, 0])  # arm -> [tot_tokens, n, correct]
    for r in rows:
        d = tok[r["arm"]]
        d[0] += int(r.get("tokens_in", 0)) + int(r.get("tokens_out", 0)); d[1] += 1; d[2] += int(r["correct"])
    present = [a for a in ARMS if a in ba and ba[a]]
    stats = {a: ST.arm_stat(a, list(ba[a].values())) for a in present}
    order = sorted(present, key=lambda a: -stats[a].accuracy)

    L = [f"## {title}", "",
         f"_{len({r['id'] for r in rows})} questions · Claude · accuracy with 95% bootstrap CI_", "",
         "| rank | arm | method | accuracy | 95% CI | n | tokens/q | tokens/correct |",
         "|---|---|---|---|---|---|---|---|"]
    for i, a in enumerate(order, 1):
        s = stats[a]
        tt, tn, tc = tok[a]
        tpq = round(tt / tn) if tn else "—"
        tpc = round(tt / tc) if tc else "—"
        L.append(f"| {i} | `{a}` | {NAME[a]} | **{s.accuracy:.3f}** | "
                 f"[{s.ci_low:.2f}, {s.ci_high:.2f}] | {s.n} | {tpq} | {tpc} |")
    L.append("")

    # pairwise McNemar + Holm across all present arms (paired on common ids)
    tests = []
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a, b = present[i], present[j]
            common = sorted(set(ba[a]) & set(ba[b]))
            if len(common) < 8:
                continue
            tests.append(ST.pair_test(a, [ba[a][k] for k in common],
                                      b, [ba[b][k] for k in common]))
    tests = ST.holm_bonferroni(tests)

    # bert-centric: A3 vs the rest
    L += [f"**Is `bert hybrid-RAG (A3)` significantly different from each arm?** "
          f"(paired exact McNemar, Holm-corrected)", "",
          "| vs | Δacc (A3 − other) | 95% CI | Holm p | significant |",
          "|---|---|---|---|---|"]
    for t in tests:
        if "A3" in (t.arm_a, t.arm_b):
            other = t.arm_b if t.arm_a == "A3" else t.arm_a
            diff = t.diff if t.arm_a == "A3" else -t.diff
            lo, hi = (t.diff_ci if t.arm_a == "A3" else (-t.diff_ci[1], -t.diff_ci[0]))
            sig = "**yes**" if t.significant else "no"
            L.append(f"| `{other}` {NAME[other]} | {diff:+.3f} | [{lo:+.2f}, {hi:+.2f}] | "
                     f"{(t.p_holm if t.p_holm is not None else t.p_value):.3f} | {sig} |")
    L.append("")
    return "\n".join(L)


def _beir_section():
    p = OUT / "beir_multi.json"
    if not p.exists():
        return ""
    try:
        d = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return ""
    rows = d.get("results") or d
    L = ["## Track A — retrieval quality on public IR benchmarks (model-free)", "",
         "_bert's hybrid retriever on independent BEIR qrels; nDCG@10 unless noted._", "",
         "| dataset | metric | bert hybrid |", "|---|---|---|"]
    if isinstance(rows, dict):
        for ds, v in rows.items():
            val = v.get("ndcg@10") or v.get("ndcg") or v
            L.append(f"| {ds} | nDCG@10 | {val} |")
    L.append("")
    return "\n".join(L)


def main() -> int:
    code = _rows(OUT / "claude_code_rows.jsonl")
    sem = _rows(OUT / "claude_semantic_rows.jsonl")

    parts = [
        "# bert retrieval benchmark — v3 (single-model, Claude)", "",
        "**Design.** Every arm answers with the *same* model (Claude), so the only "
        "variable is the retrieval method. Two question types over the same indexed "
        "code corpora (httpx+starlette, pydantic, gin/Go):", "",
        "- **Code-fact track** — exact values (defaults, constants, signatures); "
        "AST-extracted, programmatically graded. This is *grep's* natural task.",
        "- **Semantic track** — conceptual/behavioral questions deliberately phrased "
        "with no keyword overlap with the code, so naive grep can't keyword-match; "
        "judge-graded. This is *semantic retrieval's* natural task.", "",
        "Gold is frozen and the retrieval cache is aligned to it (the earlier "
        "gold/cache-drift bug that starved the RAG arms is fixed). Accuracy carries "
        "95% bootstrap CIs; pairwise differences use exact paired McNemar with "
        "Holm-Bonferroni correction.", "",
        _beir_section(),
        _track_section("Track B1 — code-fact lookup (grep's home turf)", code),
        _track_section("Track B2 — semantic / conceptual recall (bert's home turf)", sem),
        "## What the results actually show", "",
        "- **Agentic methods dominate both tracks.** `A7grep` (agentic grep) and "
        "`A_mcp` (bert live over MCP) are the top two on code-fact *and* semantic — "
        "multi-turn search-and-read beats any one-shot retrieval.",
        "- **Between the two agentic methods, grep-tools edge bert-MCP** — significantly "
        "on code-fact (0.97 vs 0.86), numerically on semantic (0.97 vs 0.90; n=30 is "
        "underpowered, p≈0.26).",
        "- **The semantic track did NOT favor bert, contrary to the original hypothesis.** "
        "The reason: `A7grep` is *agentic*, not naive keyword grep — an agent that can "
        "grep, read, and reason over code answers conceptual questions fine. \"Naive grep "
        "fails\" ≠ \"agentic grep fails.\" On code corpora, bert's one-shot hybrid-RAG "
        "(`A3`) is statistically indistinguishable from closed-book, vector, and BM25 on "
        "semantic questions.",
        "- **Where bert's one-shot retriever does win:** on code-fact it significantly "
        "beats vector-only (+0.15) — the hybrid+rerank earns its keep — and it is more "
        "**token-efficient per correct answer** (~49k) than agentic grep (~73k).",
        "- `A_mcp` is Claude calling bert's `memory_search` tool **live over MCP** (the "
        "real deployment path). It is a strong #2 on both tracks at lower token cost than "
        "grep — the most defensible single claim for bert here.", "",
        "## Limitations", "",
        "- **This benchmark tests retrieval over source CODE, not over prose/accumulated "
        "project memory** — which is bert's actual design target. The semantic edge of "
        "dense retrieval is expected to be larger over decisions/findings/notes where "
        "there is genuinely no symbol to grep; that is untested here.",
        "- Corpora are **famous open-source libraries**, so the closed-book baseline "
        "(`A0`, 0.61/0.67) is inflated by parametric knowledge; on novel/proprietary code "
        "the retrieval arms' margin over `A0` would widen.",
        "- Semantic track is **n=30, judge-graded** (3 non-Claude llama judges, majority "
        "vote) — underpowered for the close agentic-vs-agentic comparisons (several "
        "differences are non-significant).",
        "- Single reader model (Claude); the method ranking can differ on weaker models.",
        "",
    ]
    REPORT.write_text("\n".join(p for p in parts if p is not None))
    print(f"-> {REPORT}  (code={len(code)} rows, semantic={len(sem)} rows)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
