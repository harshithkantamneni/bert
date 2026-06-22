"""B9 Phase-2 head-to-head: agentic-grep (A7) vs bert hybrid-RAG (A3) vs naive
truncation (A1), end-to-end ANSWER ACCURACY. Same free reader (llama-3.3-70b),
same non-Claude grader, same questions — the only variable is how each arm
gathers context.

  A1  naive truncation to a 15K-token window (head of corpus)
  A3  bert hybrid retrieval (RRF + cross-encoder rerank), top-10 chunks
  A7  agentic grep: model iteratively greps/reads the real files (Claude-Code style)

Usage:
    .venv/bin/python benchmarks/run_b9_grep_pilot.py [n_questions]
n_questions defaults to 4 (a cheap pilot); pass 20 for the full set.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks import b9_agentic_grep as AG  # noqa: E402
from benchmarks import b9_rag_runner as RR  # noqa: E402
from benchmarks import b9_rag_stats as st  # noqa: E402

READER = [("nvidia", "meta/llama-3.3-70b-instruct"),
          ("groq", "llama-3.3-70b-versatile")]
GRADER = [("mistral", "mistral-large-latest"),
          ("openrouter", "deepseek/deepseek-v4-pro")]
CORPUS = Path("/tmp/b9_corpus")
LAB = Path("/tmp/b9_grep_pilot_lab")
TRUNC_WINDOW = 15000
TOP_N = 10


def pick_questions(all_q: list[dict], n: int) -> list[dict]:
    """Balanced subset across tiers for the pilot; full set if n>=len."""
    if n >= len(all_q):
        return all_q
    by_tier: dict[str, list] = {}
    for q in all_q:
        by_tier.setdefault(q.get("tier", "?"), []).append(q)
    out, tiers = [], sorted(by_tier)
    i = 0
    while len(out) < n:
        t = tiers[i % len(tiers)]
        if by_tier[t]:
            out.append(by_tier[t].pop(0))
        i += 1
        if i > 1000:
            break
    return out


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    gold = json.loads((REPO / "benchmarks/b9_gold/gold_qa.json").read_text())
    questions = pick_questions(gold["questions"], n)
    reader = RR.make_reader(READER)
    grader = RR.make_grader(GRADER)

    print(f"[grep-pilot] {len(questions)} questions | arms A1(trunc) A3(hybrid-RAG) "
          f"A7(agentic-grep) | reader llama-3.3-70b free | grader mistral+deepseek", flush=True)

    # A1/A2/A3/A4 via the existing sweep (shares ingest/retrieve/build_context/reader/grade)
    t0 = time.monotonic()
    ts0 = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    ckpt = REPO / "benchmarks/results" / f"b9_grep_pilot_{ts0}_partial.json"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    sweep = RR.run_sweep(CORPUS, questions, arms=["A1", "A2", "A3", "A4"], reader=reader,
                         grader=grader, lab=LAB, trunc_window_tokens=TRUNC_WINDOW,
                         top_n=TOP_N, checkpoint_path=ckpt)
    rows = list(sweep["rows"])

    # A7 agentic grep (separate loop; same reader cascade + grader)
    print("[grep-pilot] running A7 agentic-grep ...", flush=True)
    for i, q in enumerate(questions, 1):
        res = AG.agentic_grep_answer(q["question"], CORPUS, READER, max_steps=8)
        correct = grader(q["question"], q.get("gold_answer", ""), res["answer"])
        rows.append({"arm": "A7", "question": q["question"], "correct": correct,
                     "tier": q.get("tier"), "input_tokens": None,
                     "recall_at_10": None, "ndcg_at_10": None,
                     "steps": res["steps"], "answer": res["answer"][:300]})
        print(f"[A7 {i}/{len(questions)}] {q.get('tier'):10} steps={res['steps']} "
              f"{'✓' if correct else '✗'}  {q['question'][:42]}", flush=True)

    elapsed = round(time.monotonic() - t0, 1)
    by_arm = st.aggregate_by_arm(rows)
    by_tier_arm: dict[tuple, list] = {}
    for r in rows:
        by_tier_arm.setdefault((r.get("tier"), r["arm"]), []).append(r)

    print(f"\n[grep-pilot] done in {elapsed}s\n")
    print(f"{'arm':5} {'accuracy':>9} {'n':>4}")
    for arm in ["A1", "A2", "A3", "A4", "A7"]:
        a = by_arm.get(arm)
        if a:
            print(f"{arm:5} {a['accuracy']:>9.3f} {a['n']:>4}")
    print("\nby tier (accuracy):")
    tiers = sorted({t for t, _ in by_tier_arm})
    print(f"  {'tier':12} " + "  ".join(f"{a:>8}" for a in ["A1", "A2", "A3", "A4", "A7"]))
    for t in tiers:
        cells = []
        for arm in ["A1", "A2", "A3", "A4", "A7"]:
            v = by_tier_arm.get((t, arm), [])
            cells.append(f"{sum(x['correct'] for x in v)/len(v):.3f}" if v else "  na  ")
        print(f"  {t:12} " + "  ".join(f"{c:>8}" for c in cells))

    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    out = {"n_questions": len(questions), "arms": ["A1", "A2", "A3", "A4", "A7"],
           "trunc_window": TRUNC_WINDOW, "reader": "llama-3.3-70b (free)",
           "grader": "mistral-large + deepseek (non-Claude)", "elapsed_secs": elapsed,
           "by_arm": by_arm, "rows": rows}
    p = REPO / "benchmarks/results" / f"b9_grep_pilot_{ts}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
