"""Phase-1 (local, no API): retrieval recall@10 for hybrid / vector / bm25 /
graph-PageRank over the B9 gold set. Answer-accuracy (LLM reader) is Phase 2.

This isolates the retrieval question a senior engineer actually asks: does a
graph/PageRank baseline (Aider-style) find the gold spans as well as bert's
hybrid index? Pure local: span-based recall@10, no reader, no quota."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks import b9_graph_retrieve as gr  # noqa: E402
from benchmarks import b9_rag_runner as RR  # noqa: E402
from benchmarks import b9_rag_stats as st  # noqa: E402

CORPUS = Path("/tmp/b9_corpus")
LAB = Path("/tmp/b9_graph_lab")
DB = LAB / "memory.db"
K = 10


def main() -> int:
    gold = json.loads((REPO / "benchmarks/b9_gold/gold_qa.json").read_text())["questions"]
    graph, sym2file = gr.build_graph(CORPUS)
    print(f"graph: {graph.number_of_nodes()} files / {graph.number_of_edges()} edges; "
          f"{len(gold)} questions\n")

    from core import lab_context
    tok = lab_context.set_active_lab_path(LAB)
    methods = ["hybrid", "vector", "bm25", "graph"]
    agg = {m: [] for m in methods}
    by_tier = {}
    try:
        for q in gold:
            spans = q.get("gold_spans") or []
            tier = q.get("tier")
            row = {}
            for m in methods:
                if m == "graph":
                    hits = gr.graph_retrieve(q["question"], db_path=DB, graph=graph,
                                             sym2file=sym2file, corpus_dir=CORPUS, top_n=K)
                    texts = [c for _id, c in hits]
                else:
                    texts = [c for _id, c in RR.retrieve_for(q["question"], LAB, method=m, top_n=K)]
                r = st.recall_spans(texts, spans, K) if spans else None
                row[m] = r
                if r is not None:
                    agg[m].append(r)
                    by_tier.setdefault((tier, m), []).append(r)
            print(f"[{tier:10}] " + "  ".join(f"{m}={row[m]:.2f}" if row[m] is not None else f"{m}=na" for m in methods)
                  + f"   | {q['question'][:46]}")
    finally:
        lab_context.reset_active_lab_path(tok)

    print("\n=== mean recall@10 (span-based) ===")
    for m in methods:
        vals = agg[m]
        print(f"  {m:8} {sum(vals)/len(vals):.3f}  (n={len(vals)})")
    print("\n=== by tier ===")
    tiers = sorted({t for t, _ in by_tier})
    print(f"  {'tier':12} " + "  ".join(f"{m:>8}" for m in methods))
    for t in tiers:
        cells = []
        for m in methods:
            v = by_tier.get((t, m), [])
            cells.append(f"{sum(v)/len(v):.3f}" if v else "  na  ")
        print(f"  {t:12} " + "  ".join(f"{c:>8}" for c in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
