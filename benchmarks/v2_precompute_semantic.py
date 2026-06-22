"""Precompute hybrid/vector/bm25 retrievals for the SEMANTIC gold and MERGE them
into the shared retrieval_cache.json (keyed by question id). The semantic
questions retrieve over the SAME indexed code labs as the code track."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks import b9_rag_runner as RR  # noqa: E402
from benchmarks import v2_run as VR  # noqa: E402

OUT = VR.OUT


def main() -> int:
    gold = json.loads((OUT / "semantic_gold.json").read_text())
    corpora = VR.load_corpora()
    by = {c["name"]: c for c in corpora}
    labs = {cn: VR.ensure_lab(by[cn]) for cn in {g["corpus"] for g in gold} if cn in by}
    cache = json.loads((OUT / "retrieval_cache.json").read_text()) if (OUT / "retrieval_cache.json").exists() else {}
    t0 = time.monotonic()
    for i, g in enumerate(gold, 1):
        lab = labs.get(g["corpus"])
        if lab is None:
            continue
        entry = {}
        for m in ("hybrid", "vector", "bm25"):
            try:
                entry[m] = [c for _id, c in RR.retrieve_for(g["question"], lab, method=m, top_n=VR.TOP_N)]
            except Exception as e:  # noqa: BLE001
                entry[m] = []
                print(f"  [warn] {m} {g['id']}: {e}", flush=True)
        cache[g["id"]] = entry
        if i % 10 == 0:
            print(f"  semantic precomputed {i}/{len(gold)} ({round(time.monotonic()-t0)}s)", flush=True)
    (OUT / "retrieval_cache.json").write_text(json.dumps(cache))
    print(f"-> merged {len(gold)} semantic entries into retrieval_cache.json ({len(cache)} total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
