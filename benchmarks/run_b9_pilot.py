"""B9 RAG pilot run: free reader (llama, bert's real runtime) + strong non-Claude
grader, over the httpx+starlette corpus with a constrained truncation window so
the corpus exceeds it (the regime where RAG matters). Writes results JSON.

Arms: A1 naive-trunc / A2 smart-trunc / A3 bert-RAG / A4 vector-only. A0 full-
context ceiling is deferred to the Opus phase (131K > free-llama window).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks import b9_rag_runner as RR  # noqa: E402
from benchmarks import b9_rag_stats as st  # noqa: E402

READER = [("nvidia", "meta/llama-3.3-70b-instruct"),
          ("groq", "llama-3.3-70b-versatile")]
GRADER = [("mistral", "mistral-large-latest"),
          ("openrouter", "deepseek/deepseek-v4-pro")]
ARMS = ["A1", "A2", "A3", "A4"]
TRUNC_WINDOW = 15000


def main() -> int:
    gold = json.loads((REPO / "benchmarks" / "b9_gold" / "gold_qa.json").read_text())
    questions = gold["questions"]
    corpus = Path("/tmp/b9_corpus")
    lab = Path("/tmp/b9_pilot_lab")
    print(f"[b9] {len(questions)} questions x {len(ARMS)} arms over "
          f"{gold['corpus']} (~131K tok), trunc window {TRUNC_WINDOW}")
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    ckpt = REPO / "benchmarks" / "results" / f"b9_rag_pilot_{ts}_partial.json"
    t0 = time.monotonic()
    res = RR.run_sweep(corpus, questions, arms=ARMS,
                       reader=RR.make_reader(READER), grader=RR.make_grader(GRADER),
                       lab=lab, trunc_window_tokens=TRUNC_WINDOW, top_n=10,
                       checkpoint_path=ckpt)
    elapsed = round(time.monotonic() - t0, 1)

    # by-tier accuracy too (the headline: needle tier should separate most)
    by_tier_arm = {}
    for r in res["rows"]:
        by_tier_arm.setdefault((r.get("tier"), r["arm"]), []).append(r)
    tier_acc = {f"{t}/{a}": round(sum(x["correct"] for x in rows) / len(rows), 3)
                for (t, a), rows in sorted(by_tier_arm.items()) if rows}

    out = {"corpus": gold["corpus"], "n_questions": len(questions), "arms": ARMS,
           "trunc_window": TRUNC_WINDOW, "reader": "llama-3.3-70b (free)",
           "grader": "mistral-large + deepseek (non-Claude)", "elapsed_secs": elapsed,
           "by_arm": res["by_arm"], "by_tier_arm": tier_acc, "rows": res["rows"]}
    p = REPO / "benchmarks" / "results" / f"b9_rag_pilot_{ts}.json"
    p.write_text(json.dumps(out, indent=2))

    print(f"\n[b9] done in {elapsed}s -> {p}\n")
    print(f"{'arm':5} {'accuracy':>9} {'mean_in_tok':>12} {'recall@10':>10}")
    for arm in ARMS:
        a = res["by_arm"].get(arm, {})
        rc = a.get("mean_recall_at_10")
        print(f"{arm:5} {a.get('accuracy', 0):>9.3f} "
              f"{a.get('mean_input_tokens', 0):>12.0f} "
              f"{(f'{rc:.3f}' if rc is not None else 'n/a'):>10}")
    print("\nby tier/arm accuracy:")
    for k, v in tier_acc.items():
        print(f"  {k:20} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
