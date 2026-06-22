"""Compile the v2 SOTA benchmark REPORT.md from all artifacts:
  - Track A retrieval: benchmarks/results/b2_beir_multi_*.json (independent qrels)
  - QA factorial stats: benchmarks/results/v2/stats.json
  - corpora manifest, gold set
Robust to missing pieces (writes whatever sections have data).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V2 = REPO / "benchmarks/results/v2"
ARM_NAMES = {
    "A0": "closed-book (no context)", "A1": "naive truncation", "A2": "smart truncation",
    "A3": "hybrid RAG (vec+bm25+RRF+rerank)", "A4": "vector RAG", "A5": "bm25 RAG",
    "A6": "graph RAG (real Aider RepoMap)", "A7w": "agentic-grep (weak/free agent)",
    "A7f": "agentic-grep (frontier Claude agent)",
}


def _load_trackA() -> dict:
    """latest b2 result per (dataset, method)."""
    out: dict = {}
    for f in sorted(glob.glob(str(REPO / "benchmarks/results/b2_beir_multi_*.json"))):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        for r in d.get("results", []):
            out[(r["dataset"], r["method"])] = r  # later files overwrite -> newest wins
    return out


def _trackA_section() -> str:
    ta = _load_trackA()
    if not ta:
        return "_(Track A results not available)_\n"
    datasets = sorted({d for d, _ in ta})
    methods = ["bm25_only", "vector_only", "hybrid_no_rerank", "hybrid_with_rerank"]
    lines = ["Retrieval quality on established datasets with INDEPENDENT qrels "
             "(driven by bert's production embedder bge-base-en-v1.5; RRF k=60; "
             "cross-encoder = bge-reranker-v2-m3). Metric: nDCG@10 with bootstrap 95% CI.\n"]
    for ds in datasets:
        lines.append(f"\n**{ds}**\n")
        lines.append("| method | nDCG@10 [95% CI] | R@10 | MRR@10 | p95 lat |")
        lines.append("|---|---|---|---|---|")
        for m in methods:
            r = ta.get((ds, m))
            if not r:
                continue
            ci = r.get("ndcg_at_10_ci", [0, 0])
            lines.append(f"| {m} | {r['ndcg_at_10']:.3f} [{ci[0]:.3f},{ci[1]:.3f}] | "
                         f"{r.get('recall_at_10',0):.3f} | {r.get('mrr_at_10',0):.3f} | "
                         f"{r.get('p95_latency_ms',0):.0f}ms |")
    return "\n".join(lines) + "\n"


def _qa_section() -> str:
    sp = V2 / "stats.json"
    if not sp.exists():
        return "_(QA factorial stats not available yet)_\n"
    s = json.loads(sp.read_text())
    arms = s["arms"]
    nba = s.get("n_by_arm", {})
    nrange = f"{min(nba.values())}-{max(nba.values())}" if nba else "?"
    L = [f"End-to-end answer accuracy ({nrange} questions per arm; A7f frontier runs on a "
         "subset, so pairwise tests align per-pair on shared questions) "
         "(deterministic tier graded programmatically; multi-hop tier by a non-Claude judge). "
         "Reader pinned to llama-3.3-70b; k=3 repeats collapsed per item. "
         "Accuracy with bootstrap 95% CI.\n",
         "\n| arm | description | accuracy [95% CI] | p50 lat | mean $ |",
         "|---|---|---|---|---|"]
    cl = s.get("cost_latency", {})
    for a in sorted(arms, key=lambda x: -s["arm_stats"][x]["accuracy"]):
        st = s["arm_stats"][a]
        c = cl.get(a, {})
        L.append(f"| {a} | {ARM_NAMES.get(a, a)} | {st['accuracy']:.3f} "
                 f"[{st['ci_low']:.3f},{st['ci_high']:.3f}] | "
                 f"{c.get('p50_latency_ms',0):.0f}ms | ${c.get('mean_cost_usd',0):.4f} |")
    # llama + bert-via-tool (live MCP-style) — separate subset run
    btp = V2 / "bert_tool_arm.json"
    if btp.exists():
        b = json.loads(btp.read_text())
        L.append(f"| A_mcp_llama | llama + bert via tool (live) | {b['accuracy']} "
                 f"[subset n={b['n']}, tool-rate {b.get('tool_call_rate','?')}] | — | $0 |")
    # significance
    L.append("\n**Significant pairwise differences (McNemar, Holm-corrected p<.05):**\n")
    sig = [p for p in s.get("pairwise", []) if p.get("sig")]
    if sig:
        for p in sorted(sig, key=lambda x: -abs(x["diff"])):
            L.append(f"- **{p['a']} vs {p['b']}**: Δacc={p['diff']:+.3f} "
                     f"CI=[{p['diff_ci'][0]:+.3f},{p['diff_ci'][1]:+.3f}], Holm p={p['p_holm']:.4f}")
    else:
        L.append("- _(none reached significance)_")
    # non-significant headline pairs (honesty)
    L.append("\n**Notable NON-significant pairs (cannot distinguish at this n):**\n")
    for p in s.get("pairwise", []):
        if not p.get("sig") and {p["a"], p["b"]} & {"A3", "A4"} and {p["a"], p["b"]} <= {"A3", "A4", "A5"}:
            L.append(f"- {p['a']} vs {p['b']}: Δacc={p['diff']:+.3f} "
                     f"CI=[{p['diff_ci'][0]:+.3f},{p['diff_ci'][1]:+.3f}], Holm p={p['p_holm']:.3f}")
    # by tier / corpus
    L.append("\n**Accuracy by tier:**\n")
    bt = s.get("acc_by_tier", {})
    tiers = sorted({t for a in bt for t in bt[a]})
    L.append("| arm | " + " | ".join(tiers) + " |")
    L.append("|---|" + "---|" * len(tiers))
    for a in arms:
        L.append(f"| {a} | " + " | ".join(f"{bt.get(a,{}).get(t,'-')}" for t in tiers) + " |")
    L.append("\n**Accuracy by corpus:**\n")
    bc = s.get("acc_by_corpus", {})
    corp = sorted({c for a in bc for c in bc[a]})
    L.append("| arm | " + " | ".join(corp) + " |")
    L.append("|---|" + "---|" * len(corp))
    for a in arms:
        L.append(f"| {a} | " + " | ".join(f"{bc.get(a,{}).get(c,'-')}" for c in corp) + " |")
    # budget sweep
    sw = s.get("budget_sweep", {})
    if sw:
        L.append("\n**Truncation budget sweep (accuracy vs token window):**\n")
        budgets = sorted({b for a in sw for b in sw[a]}, key=int)
        L.append("| arm | " + " | ".join(f"{int(b)//1000}K" for b in budgets) + " |")
        L.append("|---|" + "---|" * len(budgets))
        for a in sw:
            L.append(f"| {a} | " + " | ".join(f"{sw[a].get(b,'-')}" for b in budgets) + " |")
    return "\n".join(L) + "\n"


def _claude_tier_section() -> str:
    """All-Claude tier: model held constant at Claude (bert's realistic MCP
    consumer); only the retrieval method varies. The honest, deployment-realistic
    comparison."""
    fr, mc, st = V2 / "frontier_reader.json", V2 / "mcp_arm.json", V2 / "stats.json"
    if not fr.exists() and not mc.exists():
        return "_(all-Claude tier not yet run)_\n"
    L = ["The SAME comparison with the model held at **Claude** (bert's real MCP "
         "consumer). Only the retrieval method varies — so any gap is the method, "
         "not the model. This is the deployment-realistic question: when Claude works "
         "in a codebase, is calling bert's MCP retrieval better than just grepping?\n",
         "| method (all on Claude) | accuracy |", "|---|---|"]
    if fr.exists():
        s = json.loads(fr.read_text()).get("summary", {})
        nm = {"A0": "closed-book", "A1": "naive-trunc", "A2": "smart-trunc",
              "A3": "bert hybrid-RAG (chunks fed)", "A4": "vector", "A5": "bm25",
              "A6": "graph/Aider"}
        for a in ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]:
            if a in s:
                L.append(f"| {nm[a]} | {s[a]['accuracy']} (n={s[a]['n']}) |")
    if st.exists():
        ss = json.loads(st.read_text()).get("arm_stats", {})
        if "A7f" in ss:
            L.append(f"| agentic-grep (A7w on Claude) | {round(ss['A7f']['accuracy'],3)} |")
    if mc.exists():
        m = json.loads(mc.read_text())
        L.append(f"| **bert via MCP (Claude calls memory_search LIVE)** | "
                 f"**{m['accuracy']}** (n={m['n']}, tool-call rate {m['tool_call_rate']}) |")
    return "\n".join(L) + "\n"


def _tokenomics_section() -> str:
    tp = V2 / "tokenomics.json"
    if not tp.exists():
        return "_(tokenomics pass not yet run)_\n"
    s = json.loads(tp.read_text()).get("summary", {})
    if not s:
        return "_(tokenomics pass produced no summary)_\n"
    L = ["Real tokens burned per method (actual prompt+completion usage summed "
         "across every call a method makes — agentic/MCP arms make several), for "
         "EVERY arm in BOTH model tiers. Efficiency = **tokens per correct answer** "
         "(lower is better). Measured on a small balanced subset (tokens are stable).\n"]
    for tier in ("llama", "claude"):
        keys = sorted([k for k in s if k.startswith(tier + "/")],
                      key=lambda k: (s[k]["tokens_per_correct"] is None, s[k].get("tokens_per_correct") or 1e18))
        if not keys:
            continue
        L.append(f"\n**Tier: {tier}**\n")
        L.append("| arm | tok_in/q | tok_out/q | total/q | acc | **tokens/correct** |")
        L.append("|---|---|---|---|---|---|")
        for k in keys:
            d = s[k]; arm = k.split("/", 1)[1]; tpc = d.get("tokens_per_correct")
            L.append(f"| {arm} | {d['tokens_in_per_q']} | {d['tokens_out_per_q']} | "
                     f"{d['total_per_q']} | {d['accuracy']} | {tpc if tpc is not None else '—'} |")
    return "\n".join(L) + "\n"


def _corpora_section() -> str:
    mp = REPO / "benchmarks/results/v2_corpora_manifest.json"
    if not mp.exists():
        return ""
    man = json.loads(mp.read_text())
    L = ["| corpus | lang | files | ~tokens |", "|---|---|---|---|"]
    for c in man:
        L.append(f"| {c['name']} | {c.get('lang','?')} | {c.get('n_files','?')} | "
                 f"{c.get('est_tokens',0):,} |")
    return "\n".join(L) + "\n"


def main() -> int:
    gold = json.loads((V2 / "gold.json").read_text()) if (V2 / "gold.json").exists() else []
    n_prog = sum(1 for g in gold if g.get("grade_mode") == "programmatic")
    n_judge = sum(1 for g in gold if g.get("grade_mode") == "judge")
    md = f"""# bert retrieval benchmark — v2 (research-grade)

A from-scratch redo of the v1 pilot with the controls v1 lacked: a closed-book
baseline, deterministic + programmatically-graded gold, paired statistics with
confidence intervals, multiple corpora (incl. a large-scale one), an established
independent-qrels retrieval track, and the real alternatives (graph/Aider,
agentic-grep with both a weak AND a frontier agent).

## Methodology (what makes this defensible)
- **Closed-book control (A0):** the reader answers with NO context, so accuracy
  attributable to retrieval is separated from parametric knowledge (popular libs
  leak into pretraining).
- **Deterministic, method-blind gold:** {n_prog} questions extracted from code by
  AST (default values, constants, regexes, enum members) and graded by exact /
  numeric / regex match — no LLM judge, no authoring bias. Plus {n_judge}
  method-blind multi-hop questions (judge-graded, non-Claude judges).
- **Paired statistics:** per-arm bootstrap 95% CIs; arm-vs-arm via exact
  McNemar on discordant pairs; Holm-Bonferroni family-wise correction. Fine
  differences are reported as significant ONLY when they clear that bar.
- **Multiple corpora** of varied language/size + a large-scale corpus that
  exceeds any context window. **Pinned reader provider** (recorded per call),
  k=3 repeats for reader variance. Cost + latency tracked per arm.

## Corpora
{_corpora_section()}
## Track A — retrieval quality on established datasets (independent qrels)
{_trackA_section()}
## Track B — end-to-end QA accuracy — TIER 1: free-tier reader (llama-3.3-70b, bert's runtime)
{_qa_section()}
## Track B — TIER 2: all-Claude (model held constant; bert-as-MCP, deployment-realistic)
{_claude_tier_section()}
## Track C — tokenomics (efficiency: how much each method burns)
{_tokenomics_section()}
## Limitations
- Multi-hop gold (judge tier) is LLM-generated (method-blind) + LLM-judged;
  softer than the deterministic tier. The deterministic tier is the rigorous core.
- BEIR datasets use a gold-preserving doc-pool subsample (max-docs) for tractable
  encoding on an 18 GB M3 Pro; codesearchnet (2M docs) was excluded for
  tractability — cqadupstack/programmers is the programming-domain proxy.
- Reader is a single free model (llama-3.3-70b); absolute accuracies would shift
  with a stronger reader, but the BETWEEN-ARM comparison holds the reader fixed.
- Agentic-grep frontier arm (A7f) runs on a question subset (Max-plan cost).
"""
    (V2 / "REPORT.md").write_text(md)
    # also copy to repo benchmarks/ for visibility
    (REPO / "benchmarks" / "V2_REPORT.md").write_text(md)
    print(f"-> {V2/'REPORT.md'} and benchmarks/V2_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
