"""Fix the mislabeled A5 (bm25) arm: refresh the retrieval cache's bm25 entries
with the REAL bm25 path (retrieve_for now has a genuine bm25 branch), then delete
the stale A5 rows so the next `v2_run.py --run` re-runs only A5. CPU-only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "benchmarks/results/v2"

from benchmarks import b9_rag_runner as RR  # noqa: E402

gold = json.loads((OUT / "gold.json").read_text())
man = {c["name"]: c for c in json.loads((REPO / "benchmarks/results/v2_corpora_manifest.json").read_text())}
cache = json.loads((OUT / "retrieval_cache.json").read_text())
labs = {c: Path("/tmp/v2_labs") / c for c in man}

n = 0
for g in gold:
    lab = labs.get(g["corpus"])
    if not lab or not lab.exists():
        continue
    try:
        bm = [c for _id, c in RR.retrieve_for(g["question"], lab, method="bm25", top_n=10)]
        cache.setdefault(g["id"], {})["bm25"] = bm
        n += 1
    except Exception as e:  # noqa: BLE001
        print(f"  warn {g['id']}: {e}")
(OUT / "retrieval_cache.json").write_text(json.dumps(cache))
print(f"refreshed REAL bm25 cache for {n} questions")

rows = [ln for ln in (OUT / "factorial_rows.jsonl").read_text().splitlines() if ln.strip()]
keep = [ln for ln in rows if json.loads(ln).get("arm") != "A5"]
(OUT / "factorial_rows.jsonl").write_text("\n".join(keep) + ("\n" if keep else ""))
print(f"deleted {len(rows)-len(keep)} stale A5 rows; {len(keep)} remain (A5 will re-run)")
