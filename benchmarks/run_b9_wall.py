"""B9 long-context WALL run (Max-only). Demonstrates: full-context (A0) works
below the 1M window and is INFEASIBLE past it, while RAG (A3) stays flat and
correct — using the SAME httpx/starlette gold questions, with the corpus padded
to ~3M tokens (needle buried in a bigger haystack).

Quota-safe: A0 runs on the Max Opus bridge for only a couple of needle questions
at the feasible tier (≈2 normal messages); A0 past the wall is pre-flight gated
to INFEASIBLE (no call). RAG/truncation run on free llama.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks import b9_rag as rag  # noqa: E402
from benchmarks import b9_rag_runner as RR  # noqa: E402
from benchmarks import b9_rag_stats as st  # noqa: E402

FREE_READER = [("nvidia", "meta/llama-3.3-70b-instruct"), ("groq", "llama-3.3-70b-versatile")]
GRADER = [("mistral", "mistral-large-latest"), ("openrouter", "deepseek/deepseek-v4-pro")]
TRUNC_WINDOW = 15000
A0_SUBSAMPLE = 2          # needle questions to actually stuff through Max Opus (quota-safe)


def _needles():
    gold = json.loads((REPO / "benchmarks" / "b9_gold" / "gold_qa.json").read_text())
    return [q for q in gold["questions"] if q["tier"] == "needle"]


def main() -> int:
    needles = _needles()
    max_reader = RR.make_max_reader(model="opus")
    free_reader = RR.make_reader(FREE_READER)
    grader = RR.make_grader(GRADER)
    rows = []

    # ── Tier 1: 131K (httpx+starlette) — A0 FEASIBLE ──────────────────
    t1_lab = Path("/tmp/b9_pilot_lab")            # already indexed httpx+starlette
    t1_corpus = RR.load_corpus_files(Path("/tmp/b9_corpus"))
    t1_tok = st.est_tokens(rag._concat(t1_corpus))
    print(f"[T1] corpus ~{t1_tok} tok; A0 feasible={rag.a0_feasible(t1_tok)[0]}", flush=True)
    # A0 on a couple of needles (Max Opus, full corpus stuffed)
    for q in needles[:A0_SUBSAMPLE]:
        retr = RR.retrieve_for(q["question"], t1_lab, method="hybrid", top_n=10)
        r = RR.run_one_query(q["question"], q["gold_answer"], [], arm="A0",
                             corpus_files=t1_corpus, retrieved=retr,
                             budget_tokens=TRUNC_WINDOW, reader=max_reader,
                             grader=grader, gold_spans=q["gold_spans"])
        r["tier"] = "T1_131K"; r["input_tokens"] = t1_tok; rows.append(r)
        print(f"  [T1 A0] {q['id']}: correct={r['correct']} in_tok={t1_tok}", flush=True)
    # A3 RAG at T1 (free) for the same needles
    for q in needles[:A0_SUBSAMPLE]:
        retr = RR.retrieve_for(q["question"], t1_lab, method="hybrid", top_n=10)
        r = RR.run_one_query(q["question"], q["gold_answer"], [], arm="A3",
                             corpus_files=t1_corpus, retrieved=retr,
                             budget_tokens=TRUNC_WINDOW, reader=free_reader,
                             grader=grader, gold_spans=q["gold_spans"])
        r["tier"] = "T1_131K"; rows.append(r)

    # ── Tier big: ~3M (httpx+starlette+numpy+sympy) — A0 INFEASIBLE ────
    big_lab = Path("/tmp/b9_big_lab")
    big_corpus_dir = Path("/tmp/b9_corpus_big")
    big_files = RR.load_corpus_files(big_corpus_dir)
    big_tok = st.est_tokens(rag._concat(big_files))
    feasible, _ = rag.a0_feasible(big_tok)
    print(f"\n[Tbig] corpus ~{big_tok} tok; A0 feasible={feasible} (the WALL)", flush=True)
    RR.ingest_corpus_into_lab(big_corpus_dir, big_lab)
    # A0 INFEASIBLE — pre-flight gated, recorded as a binary result (no call)
    for q in needles:
        rows.append({"arm": "A0", "tier": "Tbig_3M", "question": q["question"],
                     "correct": 0, "input_tokens": big_tok, "infeasible": True,
                     "recall_at_10": None, "ndcg_at_10": None})
    # A3 RAG + A1 truncation at 3M (free llama) — the contrast
    for arm in ("A3", "A1"):
        for q in needles:
            retr = (RR.retrieve_for(q["question"], big_lab, method="hybrid", top_n=10)
                    if arm in rag.RAG_ARMS else [])
            r = RR.run_one_query(q["question"], q["gold_answer"], [], arm=arm,
                                 corpus_files=big_files, retrieved=retr,
                                 budget_tokens=TRUNC_WINDOW, reader=free_reader,
                                 grader=grader, gold_spans=q["gold_spans"])
            r["tier"] = "Tbig_3M"; rows.append(r)
            print(f"  [Tbig {arm}] {q['id']}: correct={r['correct']} "
                  f"in_tok={r['input_tokens']}", flush=True)

    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    out = {"tiers": {"T1_131K": t1_tok, "Tbig_3M": big_tok}, "window": 1_000_000,
           "a0_subsample": A0_SUBSAMPLE, "rows": rows}
    p = REPO / "benchmarks" / "results" / f"b9_wall_{ts}.json"
    p.write_text(json.dumps(out, indent=2))

    # summarize
    print("\n=== WALL RESULT ===")
    for tier in ("T1_131K", "Tbig_3M"):
        print(f"\n{tier}:")
        for arm in ("A0", "A3", "A1"):
            rs = [r for r in rows if r["tier"] == tier and r["arm"] == arm]
            if not rs:
                continue
            if rs[0].get("infeasible"):
                print(f"  {arm}: INFEASIBLE (corpus {rs[0]['input_tokens']} tok > 1M window)")
            else:
                acc = sum(r["correct"] for r in rs) / len(rs)
                tok = sum(r["input_tokens"] for r in rs) / len(rs)
                print(f"  {arm}: acc={acc:.2f}  in_tok={tok:.0f}  (n={len(rs)})")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
