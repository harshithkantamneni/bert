"""v2 pre-flight: exercise the WHOLE pipeline on a tiny slice before the long
run. Times real indexing throughput (decides MPS vs CPU + corpus sizing) and
validates precompute -> arms -> grade -> stats wiring end-to-end. Writes a JSON
result so output capture is robust.

Run (with the overnight env to test the fast path):
  BERT_EMBED_DEVICE=mps BERT_EMBED_BATCH=128 HF_HUB_OFFLINE=1 \
    .venv/bin/python benchmarks/v2_preflight.py
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = "/tmp/v2_preflight.json"
res: dict = {"ok": False}

try:
    from benchmarks import b9_rag_runner as RR
    from benchmarks import v2_arms as A
    from benchmarks import v2_gold_ast as G
    from benchmarks import v2_grader as GR
    from benchmarks import v2_stats as S
    from core import lab_context, memory

    corpus = Path("/tmp/b9_corpus")
    lab = Path("/tmp/v2_preflight_lab")
    shutil.rmtree(lab, ignore_errors=True)
    lab.mkdir(parents=True, exist_ok=True)

    # 1. timed ingest+index (real throughput on the real path)
    t = time.monotonic()
    tok = lab_context.set_active_lab_path(lab)
    try:
        nfiles = memory.ingest_corpus(corpus, eager_index=True)
    finally:
        lab_context.reset_active_lab_path(tok)
    dt = time.monotonic() - t
    con = sqlite3.connect(str(lab / "memory.db"))
    nch = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    con.close()
    res["ingest"] = {"files": nfiles, "chunks": nch, "secs": round(dt, 1),
                     "chunks_per_s": round(nch / dt, 1) if dt else 0}

    # 2. tiny gold + full pipeline
    gold = G.extract_gold(str(corpus))[:3]
    reader = A.make_pinned_reader()
    cf = RR.load_corpus_files(corpus)
    method_of = {"A3": "hybrid", "A4": "vector", "A5": "bm25"}
    rows = []
    arm_results = []
    for g in gold:
        pre = {m: [c for _id, c in RR.retrieve_for(g["question"], lab, method=m, top_n=10)]
               for m in ("hybrid", "vector", "bm25")}
        for arm in ("A0", "A1", "A4", "A7w"):
            chunks = pre.get(method_of.get(arm)) if arm in method_of else None
            r = A.run_arm(arm, g["question"], corpus_root=corpus, lab=lab, corpus_files=cf,
                          budget_tokens=15000, reader=reader, top_n=10, precomputed_chunks=chunks)
            correct = GR.grade_programmatic(r["answer"], g.get("gold_answer", ""), g.get("answer_regex"))
            rows.append({"arm": arm, "id": g["id"], "correct": correct})
            arm_results.append({"q": g["question"][:50], "arm": arm, "correct": correct,
                                "prov": r["provider"], "ans": r["answer"][:60]})
    # 3. stats sanity
    from collections import defaultdict
    byarm = defaultdict(list)
    for r in rows:
        byarm[r["arm"]].append(r["correct"])
    res["arm_acc"] = {a: round(S.arm_stat(a, v).accuracy, 2) for a, v in byarm.items()}
    res["samples"] = arm_results
    res["n_arm_runs"] = len(rows)
    res["ok"] = True
except Exception as e:
    import traceback
    res["error"] = f"{type(e).__name__}: {e}"
    res["trace"] = traceback.format_exc()[-1500:]

Path(OUT).write_text(json.dumps(res, indent=2))
print("PREFLIGHT", "OK" if res["ok"] else "FAILED", "->", OUT)
