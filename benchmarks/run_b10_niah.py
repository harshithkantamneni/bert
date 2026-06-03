"""B10 NIAH run (Max-only, quota-safe). Sweeps haystack length across the 1M
window: full-context (Max Opus 1M bridge) works below the window and is
INFEASIBLE above it; bert-RAG retrieves the needle at every length.

Quota: full-context Opus runs only at the feasible tiers (50K, 200K) — a couple
of modest calls; the >1M tier is pre-flight gated to INFEASIBLE (no call).
bert-RAG uses the free llama reader at all tiers.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks import b9_rag as rag  # noqa: E402
from benchmarks import b9_rag_runner as RR  # noqa: E402
from benchmarks import b9_rag_stats as st  # noqa: E402
from benchmarks import b10_niah as niah  # noqa: E402

FILLER_PKGS = ["sympy", "numpy", "scipy"]   # ~12M tokens of local text, no network
LENGTH_TIERS = [50_000, 200_000, 2_000_000]  # straddle the 1M window
DEPTH = 0.5
FREE_READER = [("nvidia", "meta/llama-3.3-70b-instruct"), ("groq", "llama-3.3-70b-versatile")]


def _build_filler() -> str:
    venv = REPO.parent / "bert-lab" / ".venv" / "lib" / "python3.13" / "site-packages"
    if not venv.exists():
        venv = REPO / ".venv" / "lib" / "python3.13" / "site-packages"
    parts = []
    total = 0
    for pkg in FILLER_PKGS:
        for f in sorted((venv / pkg).rglob("*.py")):
            try:
                t = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            parts.append(t)
            total += len(t)
            if total > 9_000_000 * 4:    # ~9M tokens of filler is plenty for 2M tiers
                break
        if total > 9_000_000 * 4:
            break
    return "\n".join(parts)


def main() -> int:
    print("[B10] building local filler haystack…", flush=True)
    filler = _build_filler()
    print(f"  filler: ~{st.est_tokens(filler)} tokens available", flush=True)
    max_reader = RR.make_max_reader(model="opus")
    free_reader = RR.make_reader(FREE_READER)
    rows = []

    for L in LENGTH_TIERS:
        hay = niah.build_haystack(filler, L, DEPTH)
        hay_tok = st.est_tokens(hay)
        feasible, _ = rag.a0_feasible(hay_tok)
        print(f"\n[L={L} tok, depth={DEPTH}] haystack ~{hay_tok} tok; "
              f"full-context feasible={feasible}", flush=True)

        # ── full-context arm (the standard NIAH arm) ──
        if feasible:
            ans = max_reader(rag.reader_prompt(niah.QUESTION, hay))
            fc = niah.score_recall(ans)
            print(f"  full-context: recall={fc}  in_tok={hay_tok}", flush=True)
            rows.append({"length": L, "arm": "full-context", "recall": fc,
                         "input_tokens": hay_tok, "infeasible": False})
        else:
            print(f"  full-context: INFEASIBLE ({hay_tok} > 1M window) — the WALL", flush=True)
            rows.append({"length": L, "arm": "full-context", "recall": 0,
                         "input_tokens": hay_tok, "infeasible": True})

        # ── bert-RAG arm (ingest haystack, retrieve the needle) ──
        lab = Path(f"/tmp/b10_lab_{L}")
        if lab.exists():
            shutil.rmtree(lab)
        src = Path(f"/tmp/b10_hay_{L}")
        if src.exists():
            shutil.rmtree(src)
        src.mkdir(parents=True)
        (src / "haystack.md").write_text(hay)
        RR.ingest_corpus_into_lab(src, lab)
        retr = RR.retrieve_for(niah.QUESTION, lab, method="hybrid", top_n=10)
        ctx = rag.build_context("A3", corpus_files=[], retrieved_chunks=[c for _, c in retr],
                                budget_tokens=None)
        ans = free_reader(rag.reader_prompt(niah.QUESTION, ctx))
        rr = niah.score_recall(ans)
        needle_retrieved = st.hit_spans([c for _, c in retr], niah.GOLD_SPANS, 10)
        print(f"  bert-RAG: recall={rr}  in_tok={st.est_tokens(ctx)}  "
              f"needle_in_top10={needle_retrieved}", flush=True)
        rows.append({"length": L, "arm": "bert-RAG", "recall": rr,
                     "input_tokens": st.est_tokens(ctx),
                     "needle_retrieved": needle_retrieved, "infeasible": False})

    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    out = {"benchmark": "NIAH (needle-in-a-haystack)", "needle": niah.NEEDLE,
           "depth": DEPTH, "length_tiers": LENGTH_TIERS, "window": 1_000_000,
           "filler": "+".join(FILLER_PKGS), "rows": rows}
    (REPO / "benchmarks" / "results" / f"b10_niah_{ts}.json").write_text(json.dumps(out, indent=2))
    print("\n=== NIAH RESULT (recall by length) ===")
    print(f"{'length':>10} {'full-context':>16} {'bert-RAG':>12}")
    for L in LENGTH_TIERS:
        fc = next((r for r in rows if r["length"] == L and r["arm"] == "full-context"), {})
        rg = next((r for r in rows if r["length"] == L and r["arm"] == "bert-RAG"), {})
        fcs = "INFEASIBLE" if fc.get("infeasible") else f"recall={fc.get('recall')}"
        print(f"{L:>10} {fcs:>16} {('recall='+str(rg.get('recall'))):>12}")
    print(f"\nwrote benchmarks/results/b10_niah_{ts}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
