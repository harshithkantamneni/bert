"""Recovery: drop factorial rows corrupted by the gold/cache drift. Keep only the
cache-INDEPENDENT arms (A0/A1/A2/A7w/A7f — they don't read the retrieval cache, so
their rows are valid as long as the question is in the frozen gold). Drop all RAG
arms (A3/A4/A5/A6 — re-run with the aligned cache) and any stale-id rows. Dedup."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "benchmarks/results/v2"
gold_ids = {g["id"] for g in json.loads((OUT / "gold.json").read_text())}
rows = [json.loads(l) for l in (OUT / "factorial_rows.jsonl").read_text().splitlines() if l.strip()]
KEEP = {"A0", "A1", "A2", "A7w", "A7f"}  # cache-independent → valid as-is

seen, kept = set(), []
for r in rows:
    if r["id"] not in gold_ids or r["arm"] not in KEEP:
        continue
    k = (r["id"], r["arm"], r["budget"], r["rep"])
    if k in seen:
        continue
    seen.add(k)
    kept.append(r)

(OUT / "factorial_rows.jsonl").write_text("\n".join(json.dumps(r) for r in kept) + ("\n" if kept else ""))
import collections

print(f"kept {len(kept)} cache-independent rows (dropped {len(rows)-len(kept)}); "
      f"by arm: {dict(collections.Counter(r['arm'] for r in kept))}")
