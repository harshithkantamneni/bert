"""B2-real — Retrieval quality on BEIR scifact (300 queries, 5183 docs).

The synthetic B2 was procedural; this is the credibility move — real
gold-judged dataset from BEIR (Thakur et al., NeurIPS 2021).

Methodology:
  - Load scifact test split via ir_datasets (5183 corpus, 300 queries,
    339 qrels — published BEIR numbers)
  - For each method: build index ONCE, then run all 300 queries
  - Metrics: Recall@10, MRR@10, nDCG@10 with bootstrap 95% CI
  - 4 methods: vector-only / BM25-only / hybrid_no_rerank /
    hybrid_with_rerank (bge-reranker-v2-m3)
  - Latencies measured per-query (warm cache, post-index-build)

Published BEIR baselines for scifact (Thakur et al. 2021, Table 4):
  BM25:           0.665 nDCG@10
  DPR:            0.318 nDCG@10  (zero-shot dense retrieval)
  ANCE:           0.507 nDCG@10
  ColBERT:        0.665 nDCG@10
  TAS-B:          0.643 nDCG@10
  GenQ:           0.644 nDCG@10
  BM25 + CE:      0.688 nDCG@10  (BM25 + cross-encoder rerank)
  ColBERTv2 (2022): ~0.694 nDCG@10
  E5-mistral (2023): ~0.76 nDCG@10  (large dense embedder)
  bge-large-en-v1.5 (2024): ~0.72 nDCG@10
  bge-m3 (2024): ~0.74-0.78 nDCG@10

What we test (and where bert's MiniLM + bge-reranker stack should land
relative to these published numbers):
  vector_only (all-MiniLM-L6-v2):     expect 0.55-0.65 (2020 embedder)
  bm25_only (rank_bm25):              expect ~0.66 (published BM25)
  hybrid_no_rerank:                   expect 0.62-0.68
  hybrid_with_rerank (bge-rerank-v2-m3): expect 0.70-0.78

Output: benchmarks/results/b2_beir_scifact_<ts>.json + summary.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
os.environ.setdefault("BERT_DISABLE_RERANKER", "1")

import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

RESULTS_DIR = LAB_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset load ─────────────────────────────────────────────────


def load_beir_scifact(max_queries: int | None = None,
                       max_docs: int | None = None) -> tuple[dict, dict, dict]:
    """Return (corpus, queries, qrels). Caps allow fast smoke runs."""
    import ir_datasets
    ds = ir_datasets.load("beir/scifact/test")
    print(f"  loading {ds.docs_count()} docs…", flush=True)
    corpus: dict[str, str] = {}
    for d in ds.docs_iter():
        text = (d.title or "") + " " + (d.text or "")
        corpus[d.doc_id] = text.strip()
        if max_docs is not None and len(corpus) >= max_docs:
            break
    print(f"  loading {ds.queries_count()} queries…", flush=True)
    queries: dict[str, str] = {}
    for q in ds.queries_iter():
        queries[q.query_id] = q.text
        if max_queries is not None and len(queries) >= max_queries:
            break
    print(f"  loading qrels…", flush=True)
    qrels: dict[str, dict[str, int]] = {}
    for r in ds.qrels_iter():
        if r.query_id not in queries or r.doc_id not in corpus:
            continue
        qrels.setdefault(r.query_id, {})[r.doc_id] = r.relevance
    # Drop queries with no qrels
    queries = {qid: q for qid, q in queries.items() if qid in qrels}
    return corpus, queries, qrels


# ── Metrics ────────────────────────────────────────────────────


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & gold) / len(gold)


def mrr_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    for rank, did in enumerate(retrieved[:k], 1):
        if did in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], qrel: dict[str, int], k: int) -> float:
    def dcg(rels):
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    actual = [qrel.get(d, 0) for d in retrieved[:k]]
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(actual) / idcg if idcg > 0 else 0.0


def bootstrap_ci95(xs: list[float], n=1000, seed=42) -> tuple[float, float]:
    if not xs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(
        statistics.mean([xs[rng.randrange(len(xs))] for _ in xs])
        for _ in range(n)
    )
    return means[int(n * 0.025)], means[int(n * 0.975)]


# ── Methods ─────────────────────────────────────────────────────


def index_vector(corpus: dict[str, str]):
    """One-time index build for vector-only baseline."""
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_ids = list(corpus.keys())
    print(f"    encoding {len(doc_ids)} docs (vector)…", flush=True)
    t0 = time.monotonic()
    embs = model.encode(
        [corpus[d] for d in doc_ids],
        normalize_embeddings=True, show_progress_bar=False, batch_size=64,
    )
    print(f"    {time.monotonic()-t0:.1f}s", flush=True)
    return model, doc_ids, np.array(embs)


def index_bm25(corpus: dict[str, str]):
    """Use bert's actual tokenizer from core.bm25 so this benchmark
    measures the SYSTEM, not a naive inline impl."""
    from rank_bm25 import BM25Okapi
    from core.bm25 import tokenize as bert_tokenize
    doc_ids = list(corpus.keys())
    print(f"    tokenizing {len(doc_ids)} docs (BM25 via core.bm25.tokenize)…", flush=True)
    t0 = time.monotonic()
    tokenized = [bert_tokenize(corpus[d]) for d in doc_ids]
    bm = BM25Okapi(tokenized)
    print(f"    {time.monotonic()-t0:.1f}s", flush=True)
    return bm, doc_ids


def _query_tokens(query: str) -> list[str]:
    from core.bm25 import tokenize as bert_tokenize
    return bert_tokenize(query)


def retrieve_vector(query, model, doc_ids, embs, k):
    import numpy as np
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    sims = embs @ q_emb
    ranked = sorted(zip(doc_ids, sims, strict=False), key=lambda x: -x[1])
    return [d for d, _ in ranked[:k]]


def retrieve_bm25(query, bm, doc_ids, k):
    """Use bert's actual tokenizer for the query side too."""
    scores = bm.get_scores(_query_tokens(query))
    ranked = sorted(zip(doc_ids, scores, strict=False), key=lambda x: -x[1])
    return [d for d, _ in ranked[:k]]


def retrieve_hybrid_no_rerank(query, model, doc_ids, embs, bm, k):
    v_ranking = retrieve_vector(query, model, doc_ids, embs, k=len(doc_ids))
    b_ranking = retrieve_bm25(query, bm, doc_ids, k=len(doc_ids))
    rrf: dict[str, float] = {}
    for r, d in enumerate(v_ranking):
        rrf[d] = rrf.get(d, 0) + 1 / (60 + r)
    for r, d in enumerate(b_ranking):
        rrf[d] = rrf.get(d, 0) + 1 / (60 + r)
    ranked = sorted(rrf.items(), key=lambda x: -x[1])
    return [d for d, _ in ranked[:k]]


def retrieve_hybrid_with_rerank(query, corpus, model, doc_ids, embs, bm,
                                  reranker_fn, k, rerank_pool=30):
    candidates = retrieve_hybrid_no_rerank(query, model, doc_ids, embs, bm,
                                             k=rerank_pool)
    if reranker_fn is None:
        return candidates[:k]
    passages = [corpus[d] for d in candidates]
    scores = reranker_fn(query, passages)
    if not scores or len(scores) != len(passages):
        return candidates[:k]
    reranked = sorted(zip(candidates, scores, strict=False), key=lambda x: -x[1])
    return [d for d, _ in reranked[:k]]


@dataclass
class MethodResult:
    method: str
    n_queries: int
    recall_at_10: float
    recall_at_10_ci: tuple[float, float]
    mrr_at_10: float
    mrr_at_10_ci: tuple[float, float]
    ndcg_at_10: float
    ndcg_at_10_ci: tuple[float, float]
    mean_latency_ms: float
    p95_latency_ms: float


def evaluate(method: str, queries: dict, qrels: dict,
             corpus: dict, model, doc_ids, embs, bm, reranker_fn=None,
             k: int = 10) -> MethodResult:
    r10s, m10s, n10s, lats = [], [], [], []
    for qid, q in queries.items():
        gold = set(qrels.get(qid, {}).keys())
        if not gold:
            continue
        t0 = time.perf_counter()
        if method == "vector_only":
            retrieved = retrieve_vector(q, model, doc_ids, embs, k)
        elif method == "bm25_only":
            retrieved = retrieve_bm25(q, bm, doc_ids, k)
        elif method == "hybrid_no_rerank":
            retrieved = retrieve_hybrid_no_rerank(q, model, doc_ids, embs, bm, k)
        elif method == "hybrid_with_rerank":
            retrieved = retrieve_hybrid_with_rerank(
                q, corpus, model, doc_ids, embs, bm, reranker_fn, k,
            )
        else:
            raise ValueError(f"unknown method {method!r}")
        lats.append((time.perf_counter() - t0) * 1000)
        r10s.append(recall_at_k(retrieved, gold, 10))
        m10s.append(mrr_at_k(retrieved, gold, 10))
        n10s.append(ndcg_at_k(retrieved, qrels[qid], 10))
    return MethodResult(
        method=method, n_queries=len(r10s),
        recall_at_10=statistics.mean(r10s),
        recall_at_10_ci=bootstrap_ci95(r10s),
        mrr_at_10=statistics.mean(m10s),
        mrr_at_10_ci=bootstrap_ci95(m10s),
        ndcg_at_10=statistics.mean(n10s),
        ndcg_at_10_ci=bootstrap_ci95(n10s),
        mean_latency_ms=statistics.mean(lats),
        p95_latency_ms=sorted(lats)[int(len(lats) * 0.95)],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--skip-rerank", action="store_true",
                    help="skip the slow cross-encoder rerank pass")
    args = ap.parse_args()

    print("════════════════════════════════════════════════════════════════")
    print("  B2 (real) — BEIR scifact retrieval quality")
    print("════════════════════════════════════════════════════════════════")
    print()

    print("Loading dataset…")
    corpus, queries, qrels = load_beir_scifact(
        max_queries=args.max_queries, max_docs=args.max_docs,
    )
    print(f"  → {len(corpus)} docs, {len(queries)} queries (with qrels)")
    print()

    print("Building indexes…")
    model, doc_ids, embs = index_vector(corpus)
    bm, _ = index_bm25(corpus)
    print()

    methods = ["vector_only", "bm25_only", "hybrid_no_rerank"]
    if not args.skip_rerank:
        methods.append("hybrid_with_rerank")

    reranker_fn = None
    if "hybrid_with_rerank" in methods:
        print("Loading cross-encoder reranker (one-time cost)…")
        os.environ.pop("BERT_DISABLE_RERANKER", None)
        from core import reranker as _rr
        t0 = time.monotonic()
        _ = _rr.is_available()  # force load
        print(f"  loaded in {time.monotonic()-t0:.1f}s")
        reranker_fn = _rr.rerank
        print()

    results: list[MethodResult] = []
    for m in methods:
        print(f"Evaluating {m}…", flush=True)
        t0 = time.monotonic()
        r = evaluate(m, queries, qrels, corpus, model, doc_ids, embs, bm,
                     reranker_fn=reranker_fn)
        elapsed = time.monotonic() - t0
        print(f"  nDCG@10={r.ndcg_at_10:.4f} [{r.ndcg_at_10_ci[0]:.3f},"
              f"{r.ndcg_at_10_ci[1]:.3f}]  R@10={r.recall_at_10:.4f}  "
              f"MRR@10={r.mrr_at_10:.4f}  ({elapsed:.1f}s total, "
              f"{r.mean_latency_ms:.1f}ms/query)", flush=True)
        results.append(r)

    ts = time.strftime("%Y%m%dT%H%M%S")
    json_path = RESULTS_DIR / f"b2_beir_scifact_{ts}.json"
    json_path.write_text(json.dumps({
        "dataset": "BEIR scifact (test)",
        "corpus_size": len(corpus),
        "n_queries": len(queries),
        "results": [asdict(r) for r in results],
        "timestamp": ts,
    }, indent=2))
    summary = RESULTS_DIR / f"b2_beir_scifact_summary_{ts}.md"

    # Published baselines for headline framing
    published = {
        "BM25 (BEIR paper)": 0.665,
        "DPR zero-shot": 0.318,
        "ColBERT v1": 0.665,
        "BM25 + cross-encoder": 0.688,
        "ColBERTv2 (2022)": 0.694,
        "bge-large-en-v1.5 (2024)": 0.72,
        "bge-m3 (2024)": 0.76,
        "E5-mistral (2023)": 0.76,
    }

    lines = [
        f"# B2 (real) — BEIR scifact retrieval quality",
        f"",
        f"_Generated: {ts}_",
        f"_Dataset: BEIR scifact test split — {len(corpus)} docs, "
        f"{len(queries)} queries (with qrels)_",
        f"",
        "## Our results",
        "",
        "| Method | nDCG@10 [95% CI] | R@10 [95% CI] | MRR@10 | mean lat |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        n_ci = r.ndcg_at_10_ci
        r_ci = r.recall_at_10_ci
        lines.append(
            f"| `{r.method}` | "
            f"**{r.ndcg_at_10:.4f}** [{n_ci[0]:.3f}–{n_ci[1]:.3f}] | "
            f"{r.recall_at_10:.4f} [{r_ci[0]:.3f}–{r_ci[1]:.3f}] | "
            f"{r.mrr_at_10:.4f} | {r.mean_latency_ms:.1f}ms |"
        )
    lines += [
        "",
        "## Published SOTA / competitor baselines (BEIR scifact, nDCG@10)",
        "",
        "| System | nDCG@10 | Source |",
        "|---|---:|---|",
    ]
    for sys_name, ndcg in published.items():
        lines.append(f"| {sys_name} | {ndcg:.3f} | published |")
    # Best-by-method
    best = max(results, key=lambda r: r.ndcg_at_10)
    closest_pub = min(published.items(), key=lambda kv: abs(kv[1] - best.ndcg_at_10))
    lines += [
        "",
        f"### Headline",
        f"",
        f"- Our **best is `{best.method}`** at nDCG@10 = **{best.ndcg_at_10:.4f}**",
        f"- Closest published baseline: **{closest_pub[0]}** at {closest_pub[1]:.3f}",
        f"- Delta vs closest: **{best.ndcg_at_10 - closest_pub[1]:+.4f}**",
        f"",
    ]
    summary.write_text("\n".join(lines))
    print()
    print(f"Wrote: {json_path}")
    print(f"Wrote: {summary}")
    print(f"All {len(results)} methods evaluated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
