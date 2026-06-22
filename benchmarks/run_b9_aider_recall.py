"""Phase-1 recall@10 for the REAL Aider RepoMap arms vs the from-scratch graph
arm, over the B9 gold set. Identical span-based recall metric as
run_b9_graph_recall.py, so the columns merge directly with the bert arms
(hybrid / vector / bm25) computed there.

Run with the aider venv:
    /tmp/aider_venv/bin/python benchmarks/run_b9_aider_recall.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks import b9_aider_retrieve as A  # noqa: E402
from benchmarks import b9_graph_retrieve as gr  # noqa: E402

CORPUS = Path("/tmp/b9_corpus")
DB = Path("/tmp/b9_graph_lab/memory.db")
K = 10


def recall_spans(retrieved_texts, gold_spans, k):
    """Identical to b9_rag_stats.recall_spans."""
    if not gold_spans:
        return 0.0
    blob = "\n".join(retrieved_texts[:k])
    return sum(1 for s in gold_spans if s in blob) / len(gold_spans)


def main() -> int:
    gold = json.loads((REPO / "benchmarks/b9_gold/gold_qa.json").read_text())["questions"]
    rm, files, root = A.build_repomap(CORPUS)
    fc = A.load_file_chunks(DB)
    mygraph, sym2file = gr.build_graph(CORPUS)
    print(f"aider RepoMap: {len(files)} files | from-scratch graph: "
          f"{mygraph.number_of_nodes()} nodes / {mygraph.number_of_edges()} edges | "
          f"{len(gold)} questions\n")

    methods = ["aider_tags", "aider_filerank", "graph_mine"]
    agg = {m: [] for m in methods}
    by_tier: dict[tuple, list] = {}
    for q in gold:
        spans = q.get("gold_spans") or []
        tier = q.get("tier")
        if not spans:
            continue
        row = {}
        tags = A.aider_tags_retrieve(q["question"], rm=rm, all_files=files, root=root,
                                     corpus_dir=CORPUS, file_chunks=fc, top_n=K)
        frank = A.aider_filerank_retrieve(q["question"], rm=rm, all_files=files, root=root,
                                          corpus_dir=CORPUS, file_chunks=fc, top_n=K)
        mine = gr.graph_retrieve(q["question"], db_path=DB, graph=mygraph,
                                 sym2file=sym2file, corpus_dir=CORPUS, top_n=K)
        results = {
            "aider_tags": [c for _id, c in tags],
            "aider_filerank": [c for _id, c in frank],
            "graph_mine": [c for _id, c in mine],
        }
        for m in methods:
            r = recall_spans(results[m], spans, K)
            row[m] = r
            agg[m].append(r)
            by_tier.setdefault((tier, m), []).append(r)
        print(f"[{tier:10}] " + "  ".join(f"{m}={row[m]:.2f}" for m in methods)
              + f"   | {q['question'][:40]}")

    print("\n=== mean recall@10 (span-based) ===")
    for m in methods:
        v = agg[m]
        print(f"  {m:16} {sum(v)/len(v):.3f}  (n={len(v)})")
    print("\n  for reference (from run_b9_graph_recall.py, same gold + metric):")
    print("  hybrid           0.783")
    print("  bm25             0.783")
    print("  vector           0.725")

    print("\n=== by tier ===")
    tiers = sorted({t for t, _ in by_tier})
    print(f"  {'tier':12} " + "  ".join(f"{m:>16}" for m in methods))
    for t in tiers:
        cells = []
        for m in methods:
            vals = by_tier.get((t, m), [])
            cells.append(f"{sum(vals)/len(vals):.3f}" if vals else "  na  ")
        print(f"  {t:12} " + "  ".join(f"{c:>16}" for c in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
