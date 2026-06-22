"""Precompute the A6 (graph / real-Aider RepoMap) retrievals for the v2 gold set.

Runs in the AIDER venv (/tmp/aider_venv) because it needs the `aider` package;
it only touches b9_aider_retrieve (stdlib + networkx + aider) and the lab's
sqlite chunks DB — no bert core deps. The main v2_run factorial reads the
emitted aider_cache.json for arm A6.

Usage:
    /tmp/aider_venv/bin/python benchmarks/v2_precompute_aider.py
(Requires v2_run.py --precompute to have ingested the corpora into /tmp/v2_labs.)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks import b9_aider_retrieve as AID  # noqa: E402

OUT = REPO / "benchmarks" / "results" / "v2"
TOP_N = 10


def main() -> int:
    # optional argv[1] = gold file (default gold.json); merges into aider_cache.json
    gold_file = OUT / (sys.argv[1] if len(sys.argv) > 1 else "gold.json")
    gold = json.loads(gold_file.read_text())
    manifest = {c["name"]: c for c in
                json.loads((REPO / "benchmarks/results/v2_corpora_manifest.json").read_text())}
    cache: dict[str, dict] = json.loads((OUT / "aider_cache.json").read_text()) \
        if (OUT / "aider_cache.json").exists() else {}
    by_corpus: dict[str, list] = {}
    for g in gold:
        by_corpus.setdefault(g["corpus"], []).append(g)

    for cname, qs in by_corpus.items():
        c = manifest.get(cname)
        lab_db = Path("/tmp/v2_labs") / cname / "memory.db"
        if not c or not lab_db.exists():
            print(f"  [{cname}] SKIP (no corpus/lab db at {lab_db})", flush=True)
            continue
        root = c["root"]
        t0 = time.monotonic()
        rm, files, rroot = AID.build_repomap(root)
        fc = AID.load_file_chunks(lab_db)
        print(f"  [{cname}] RepoMap {len(files)} files, {len(fc)} chunked files "
              f"({round(time.monotonic()-t0)}s); {len(qs)} questions", flush=True)
        for g in qs:
            try:
                hits = AID.aider_filerank_retrieve(
                    g["question"], rm=rm, all_files=files, root=rroot,
                    corpus_dir=root, file_chunks=fc, top_n=TOP_N)
                cache[g["id"]] = {"graph": [c for _id, c in hits]}
            except Exception as e:  # noqa: BLE001
                cache[g["id"]] = {"graph": []}
                print(f"    [warn] {g['id']}: {e}", flush=True)

    (OUT / "aider_cache.json").write_text(json.dumps(cache))
    print(f"-> aider_cache.json ({len(cache)} questions)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
