"""v2 frontier-reader condition — removes the model confound.

The main factorial reads with a free model (llama-3.3-70b), which is bert's real
runtime. But A7f (agentic grep) uses Claude, so comparing it to the llama-read
RAG arms mixes "method" with "model". This pass re-runs the KEY context arms with
CLAUDE as the reader (no tools — it only sees the provided context), so:

  closed-book / truncation / hybrid-RAG / vector  — ALL on Claude
  vs A7f (Claude agentic grep, already measured)

gives a clean frontier comparison: with a strong model, does grep still beat RAG?

Run AFTER the free-tier factorial (uses Max-plan `claude -p`). Subset + k=1 to
bound cost.

  .venv/bin/python benchmarks/v2_frontier_reader.py [n_per_corpus] [model]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "benchmarks/results/v2"

from benchmarks import b9_rag as RAG  # noqa: E402
from benchmarks import b9_rag_runner as RR  # noqa: E402
from benchmarks import v2_arms as A  # noqa: E402
from benchmarks import v2_grader as GR  # noqa: E402

_MTH = {"A3": "hybrid", "A4": "vector", "A5": "bm25"}
# All context arms on Claude (A6 = graph/Aider chunks). A7w-on-Claude IS A7f
# (agentic grep with the frontier agent), so the agentic arm is covered there.
ARMS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]


def claude_reader(model="sonnet"):
    """reader(prompt) -> answer text via `claude -p` (no tools — context only)."""
    def _r(prompt):
        try:
            p = subprocess.run(["claude", "-p", "--model", model, "--output-format", "json"],
                               input=prompt, capture_output=True, text=True, timeout=240)
            if p.returncode != 0:
                return f"[claude rc={p.returncode}]"
            return json.loads(p.stdout).get("result", "") or ""
        except Exception as e:  # noqa: BLE001
            return f"[claude err: {e}]"
    return _r


def _ctx(arm, g, corpus_files, rcache, acache):
    if arm == "A0":
        return A._closed_book_prompt(g["question"])
    if arm in ("A1", "A2"):
        c = RAG.build_context(arm, corpus_files=corpus_files, retrieved_chunks=[], budget_tokens=15000)
        return RAG.reader_prompt(g["question"], c)
    if arm in ("A3", "A4", "A5"):
        chunks = (rcache.get(g["id"], {}) or {}).get(_MTH[arm], [])
        return RAG.reader_prompt(g["question"], A._rag_context(chunks))
    if arm == "A6":
        chunks = (acache.get(g["id"], {}) or {}).get("graph", [])
        return RAG.reader_prompt(g["question"], A._rag_context(chunks))
    raise ValueError(arm)


def main() -> int:
    npc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    model = sys.argv[2] if len(sys.argv) > 2 else "sonnet"
    gold = json.loads((OUT / "gold.json").read_text())
    prog = [g for g in gold if g.get("grade_mode") == "programmatic"]
    import collections
    by = collections.defaultdict(list)
    for g in prog:
        by[g["corpus"]].append(g)
    subset = [g for gs in by.values() for g in gs[:npc]]
    man = {c["name"]: c for c in json.loads((REPO / "benchmarks/results/v2_corpora_manifest.json").read_text())}
    rcache = json.loads((OUT / "retrieval_cache.json").read_text()) if (OUT / "retrieval_cache.json").exists() else {}
    acache = json.loads((OUT / "aider_cache.json").read_text()) if (OUT / "aider_cache.json").exists() else {}
    cfiles = {c: RR.load_corpus_files(Path(man[c]["root"])) for c in by if c in man}
    reader = claude_reader(model)

    ckpt = OUT / "frontier_reader_rows.jsonl"
    done = set()
    if ckpt.exists():
        for ln in ckpt.read_text().splitlines():
            try:
                r = json.loads(ln); done.add((r["id"], r["arm"]))
            except Exception:  # noqa: BLE001
                pass
    print(f"[frontier-reader] model={model} {len(subset)} Q x {len(ARMS)} arms "
          f"(resume {len(done)} done)", flush=True)
    # Parallel: these are pure claude -p calls reading cached chunks — no MPS, no
    # bert server — so they parallelize cleanly. (Bounded so Max isn't hammered.)
    import os as _os
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cells = [(g, arm) for g in subset for arm in ARMS if (g["id"], arm) not in done]
    workers = int(_os.environ.get("BERT_CLAUDE_WORKERS", "4"))

    def _work(cell):
        g, arm = cell
        ans = reader(_ctx(arm, g, cfiles.get(g["corpus"], []), rcache, acache))
        corr = GR.grade_programmatic(ans, g.get("gold_answer", ""), g.get("answer_regex"))
        return {"id": g["id"], "corpus": g["corpus"], "arm": arm, "correct": corr, "answer": ans[:200]}

    lock = threading.Lock()
    f = ckpt.open("a")
    t0 = time.monotonic(); n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_work, c) for c in cells]):
            row = fut.result()
            with lock:
                f.write(json.dumps(row) + "\n"); f.flush()
                n += 1
                if n % 10 == 0:
                    print(f"  {n}/{len(cells)} cells ({round(time.monotonic()-t0)}s)", flush=True)
    f.close()

    rows = [json.loads(ln) for ln in ckpt.read_text().splitlines() if ln.strip()]
    agg = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        agg[r["arm"]][0] += r["correct"]; agg[r["arm"]][1] += 1
    summary = {a: {"accuracy": round(c / n, 3), "n": n} for a, (c, n) in agg.items()}
    (OUT / "frontier_reader.json").write_text(json.dumps(
        {"model": model, "n_questions": len(subset), "summary": summary}, indent=2))
    print("\nfrontier-reader (Claude) accuracy:")
    for a in ARMS:
        if a in summary:
            print(f"  {a} {summary[a]['accuracy']:.3f} (n={summary[a]['n']})")
    print(f"-> {OUT/'frontier_reader.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
