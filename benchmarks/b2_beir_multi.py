"""B2-multi — Retrieval quality across multiple BEIR datasets.

Extends the scifact-only B2 to a multi-dataset gauntlet (scifact + nfcorpus +
fiqa) and — the important part — drives the SAME embedder the running system
uses, by importing EMBED_MODEL_NAME and the asymmetric query/passage affixes
from core.memory. There is no hardcoded model string here: whatever bert
embeds with, this benchmark measures. That closes the production-vs-benchmark
drift gap (a prior B9 bug hid a broken hybrid retriever precisely because the
BEIR harness embedded with its own separate copy).

The cross-encoder rerank pass uses core.reranker, now batch- and length-bounded
so bge-reranker-v2-m3 actually runs on 18 GB MPS instead of OOM-ing and
silently degrading hybrid_with_rerank to hybrid_no_rerank.

Methods per dataset:
  vector_only / bm25_only / hybrid_no_rerank / hybrid_with_rerank
Metrics: Recall@10, MRR@10, nDCG@10 with bootstrap 95% CI.

Published BEIR baselines (Thakur et al., NeurIPS 2021, Table 4) and the
bge-base-en-v1.5 reference numbers are shown alongside for context.

Usage (from repo root, in the uv env):
  HF_HUB_OFFLINE=1 uv run python benchmarks/b2_beir_multi.py
  uv run python benchmarks/b2_beir_multi.py --datasets scifact --max-queries 50
  uv run python benchmarks/b2_beir_multi.py --skip-rerank

Output: benchmarks/results/b2_beir_multi_<ts>.json + _summary_<ts>.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
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

# Single source of truth for the embedder: whatever bert uses in production.
from core.memory import (  # noqa: E402
    EMBED_MODEL_NAME,
    EMBED_PASSAGE_PREFIX,
    EMBED_QUERY_PREFIX,
)

# ── Dataset registry ─────────────────────────────────────────────────
# irds: the ir_datasets identifier. published: well-known external nDCG@10
# numbers for context only (BEIR paper for BM25 / dense baselines; BGE paper /
# MTEB for bge-base-en-v1.5). These are references, not our measurements.
DATASETS: dict[str, dict] = {
    "scifact": {
        "irds": "beir/scifact/test",
        "published": {
            "BM25 (BEIR paper)": 0.665,
            "ColBERTv2 (2022)": 0.694,
            "BM25 + cross-encoder": 0.688,
            "bge-base-en-v1.5 (ref)": 0.741,
            "E5-mistral (2023)": 0.760,
        },
    },
    "nfcorpus": {
        "irds": "beir/nfcorpus/test",
        "published": {
            "BM25 (BEIR paper)": 0.325,
            "bge-base-en-v1.5 (ref)": 0.373,
        },
    },
    "fiqa": {
        "irds": "beir/fiqa/test",
        "published": {
            "BM25 (BEIR paper)": 0.236,
            "ColBERTv2 (2022)": 0.356,
            "bge-base-en-v1.5 (ref)": 0.407,
        },
    },
    # Programming-domain retrieval with independent qrels (StackExchange
    # Programmers duplicate-question retrieval) — the closest tractable
    # established proxy for code/programming retrieval (codesearchnet is 2M docs
    # with no cheap subsample, excluded for tractability; see REPORT limitations).
    "cqadupstack-programmers": {
        "irds": "beir/cqadupstack/programmers",
        "published": {
            "BM25 (BEIR paper)": 0.281,
            "bge-base-en-v1.5 (ref)": 0.402,
        },
    },
    "scidocs": {
        "irds": "beir/scidocs",
        "published": {
            "BM25 (BEIR paper)": 0.158,
            "bge-base-en-v1.5 (ref)": 0.217,
        },
    },
    "arguana": {
        "irds": "beir/arguana",
        "published": {
            "BM25 (BEIR paper)": 0.315,
            "bge-base-en-v1.5 (ref)": 0.636,
        },
    },
}


def load_beir(irds_id: str, max_queries: int | None = None,
              max_docs: int | None = None, seed: int = 13) -> tuple[dict, dict, dict]:
    """Return (corpus, queries, qrels) for a BEIR dataset.

    GOLD-PRESERVING subsample: when max_docs caps the corpus (for tractable
    encoding on the M3 Pro), the pool ALWAYS keeps every gold doc for the kept
    queries, then fills with random non-gold docs up to max_docs. This keeps the
    task valid (recall is computable — gold is always in the pool) instead of the
    naive head-truncation that silently drops queries whose gold sorts late."""
    import random as _rnd

    import ir_datasets
    ds = ir_datasets.load(irds_id)
    # queries (capped)
    queries: dict[str, str] = {}
    for q in ds.queries_iter():
        queries[q.query_id] = q.text
        if max_queries is not None and len(queries) >= max_queries:
            break
    # qrels for the kept queries; collect gold doc ids
    qrels: dict[str, dict[str, int]] = {}
    gold_docs: set[str] = set()
    for r in ds.qrels_iter():
        if r.query_id not in queries or r.relevance <= 0:
            continue
        qrels.setdefault(r.query_id, {})[r.doc_id] = r.relevance
        gold_docs.add(r.doc_id)
    queries = {qid: q for qid, q in queries.items() if qid in qrels}
    gold_docs = {d for qid in queries for d in qrels[qid]}
    # docs: load all, then gold-preserving subsample if capped
    print(f"  loading {ds.docs_count()} docs (cap={max_docs}, gold={len(gold_docs)})…", flush=True)
    corpus: dict[str, str] = {}
    for d in ds.docs_iter():
        title = getattr(d, "title", "") or ""
        body = getattr(d, "text", "") or ""
        corpus[d.doc_id] = (title + " " + body).strip()
    if max_docs is not None and len(corpus) > max_docs:
        rng = _rnd.Random(seed)
        fill = [k for k in corpus if k not in gold_docs]
        rng.shuffle(fill)
        keep = set(gold_docs) | set(fill[: max(0, max_docs - len(gold_docs))])
        corpus = {k: v for k, v in corpus.items() if k in keep}
    # drop any gold not present (defensive) and queries left without qrels
    for qid in list(qrels):
        qrels[qid] = {d: r for d, r in qrels[qid].items() if d in corpus}
        if not qrels[qid]:
            del qrels[qid]
    queries = {qid: q for qid, q in queries.items() if qid in qrels}
    return corpus, queries, qrels


# ── Metrics ──────────────────────────────────────────────────────────


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


def bootstrap_ci95(xs: list[float], n: int = 1000, seed: int = 42) -> tuple[float, float]:
    if not xs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(
        statistics.mean([xs[rng.randrange(len(xs))] for _ in xs])
        for _ in range(n)
    )
    return means[int(n * 0.025)], means[int(n * 0.975)]


# ── Indexing + retrieval (embedder single-sourced from core.memory) ──


def index_vector(corpus: dict[str, str], model):
    """Encode the corpus with the production embedder + passage affix."""
    import numpy as np
    doc_ids = list(corpus.keys())
    texts = [EMBED_PASSAGE_PREFIX + corpus[d] for d in doc_ids] \
        if EMBED_PASSAGE_PREFIX else [corpus[d] for d in doc_ids]
    print(f"    encoding {len(doc_ids)} docs with {EMBED_MODEL_NAME}…", flush=True)
    t0 = time.monotonic()
    embs = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=False,
        batch_size=int(os.environ.get("BERT_EMBED_BATCH", "64")),
    )
    print(f"    {time.monotonic()-t0:.1f}s", flush=True)
    return doc_ids, np.array(embs)


def index_bm25(corpus: dict[str, str]):
    """Use bert's actual tokenizer (core.bm25.tokenize) so this measures the
    SYSTEM, not a naive inline impl."""
    from rank_bm25 import BM25Okapi

    from core.bm25 import tokenize as bert_tokenize
    doc_ids = list(corpus.keys())
    print(f"    tokenizing {len(doc_ids)} docs (core.bm25.tokenize)…", flush=True)
    t0 = time.monotonic()
    tokenized = [bert_tokenize(corpus[d]) for d in doc_ids]
    bm = BM25Okapi(tokenized)
    print(f"    {time.monotonic()-t0:.1f}s", flush=True)
    return bm, doc_ids


def _query_tokens(query: str) -> list[str]:
    from core.bm25 import tokenize as bert_tokenize
    return bert_tokenize(query)


def retrieve_vector(query, model, doc_ids, embs, k):
    qtext = (EMBED_QUERY_PREFIX + query) if EMBED_QUERY_PREFIX else query
    q_emb = model.encode([qtext], normalize_embeddings=True)[0]
    sims = embs @ q_emb
    ranked = sorted(zip(doc_ids, sims, strict=False), key=lambda x: -x[1])
    return [d for d, _ in ranked[:k]]


def retrieve_bm25(query, bm, doc_ids, k):
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
    dataset: str
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


def evaluate(dataset, method, queries, qrels, corpus, model, doc_ids, embs, bm,
             reranker_fn=None, k: int = 10) -> MethodResult:
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
        dataset=dataset, method=method, n_queries=len(r10s),
        recall_at_10=statistics.mean(r10s) if r10s else 0.0,
        recall_at_10_ci=bootstrap_ci95(r10s),
        mrr_at_10=statistics.mean(m10s) if m10s else 0.0,
        mrr_at_10_ci=bootstrap_ci95(m10s),
        ndcg_at_10=statistics.mean(n10s) if n10s else 0.0,
        ndcg_at_10_ci=bootstrap_ci95(n10s),
        mean_latency_ms=statistics.mean(lats) if lats else 0.0,
        p95_latency_ms=sorted(lats)[int(len(lats) * 0.95)] if lats else 0.0,
    )


def _write_results(all_results, ts, args):
    """Write JSON + markdown for whatever results exist so far (called after
    each dataset so a long run produces usable output incrementally)."""
    json_path = RESULTS_DIR / f"b2_beir_multi_{ts}.json"
    json_path.write_text(json.dumps({
        "embedder": EMBED_MODEL_NAME,
        "query_prefix": EMBED_QUERY_PREFIX,
        "passage_prefix": EMBED_PASSAGE_PREFIX,
        "reranker": (None if args.skip_rerank else "BAAI/bge-reranker-v2-m3"),
        "datasets": args.datasets,
        "results": [asdict(r) for r in all_results],
        "timestamp": ts,
    }, indent=2))
    lines = [
        "# B2-multi — BEIR retrieval quality (real bert stack)",
        "",
        f"_Generated: {ts}_",
        f"_Embedder: `{EMBED_MODEL_NAME}` "
        f"(query affix `{EMBED_QUERY_PREFIX.strip() or 'none'}`)_  ",
        f"_Reranker: {'(skipped)' if args.skip_rerank else '`BAAI/bge-reranker-v2-m3`'}_  ",
        "_BM25 via `core.bm25.tokenize`; fusion = RRF(k=60); metric = nDCG@10 "
        "with bootstrap 95% CI._",
        "",
    ]
    done = [d for d in args.datasets if any(r.dataset == d for r in all_results)]
    for ds_name in done:
        ds_results = [r for r in all_results if r.dataset == ds_name]
        n_q = ds_results[0].n_queries
        lines += [
            f"## {ds_name}  ({n_q} queries)",
            "",
            "| Method | nDCG@10 [95% CI] | R@10 | MRR@10 | mean lat |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in ds_results:
            ci = r.ndcg_at_10_ci
            lines.append(
                f"| `{r.method}` | **{r.ndcg_at_10:.4f}** "
                f"[{ci[0]:.3f}–{ci[1]:.3f}] | {r.recall_at_10:.4f} | "
                f"{r.mrr_at_10:.4f} | {r.mean_latency_ms:.1f}ms |"
            )
        lines += ["", "_Published reference (nDCG@10):_ " + ", ".join(
            f"{k} {v:.3f}" for k, v in DATASETS[ds_name]["published"].items()
        ), ""]
        best = max(ds_results, key=lambda r: r.ndcg_at_10)
        bm25_pub = DATASETS[ds_name]["published"].get("BM25 (BEIR paper)")
        if bm25_pub is not None:
            lines.append(
                f"- best method `{best.method}` = **{best.ndcg_at_10:.4f}** "
                f"vs published BM25 {bm25_pub:.3f} "
                f"(**{best.ndcg_at_10 - bm25_pub:+.4f}**)"
            )
        lines.append("")
    summary = RESULTS_DIR / f"b2_beir_multi_summary_{ts}.md"
    summary.write_text("\n".join(lines))
    return json_path, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=list(DATASETS),
                    help="which BEIR datasets to run")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--skip-rerank", action="store_true",
                    help="skip the cross-encoder rerank pass")
    args = ap.parse_args()

    print("════════════════════════════════════════════════════════════════")
    print("  B2-multi — BEIR retrieval quality (real bert stack)")
    print(f"  embedder: {EMBED_MODEL_NAME}")
    print(f"  query affix: {EMBED_QUERY_PREFIX!r}")
    print(f"  datasets: {', '.join(args.datasets)}")
    print("════════════════════════════════════════════════════════════════")
    print()

    # Load the embedder ONCE; reuse across datasets.
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedder {EMBED_MODEL_NAME} (one-time)…", flush=True)
    t0 = time.monotonic()
    model = SentenceTransformer(EMBED_MODEL_NAME, device=os.environ.get("BERT_EMBED_DEVICE") or None)
    print(f"  embedder device: {model.device}", flush=True)
    print(f"  loaded in {time.monotonic()-t0:.1f}s "
          f"(dim={model.get_sentence_embedding_dimension()})")
    print()

    methods = ["vector_only", "bm25_only", "hybrid_no_rerank"]
    reranker_fn = None
    if not args.skip_rerank:
        methods.append("hybrid_with_rerank")
        print("Loading cross-encoder reranker (one-time cost)…")
        os.environ.pop("BERT_DISABLE_RERANKER", None)
        from core import reranker as _rr
        t0 = time.monotonic()
        avail = _rr.is_available()  # force load
        print(f"  reranker available={avail} model={_rr.status().model} "
              f"({time.monotonic()-t0:.1f}s)")
        if avail:
            reranker_fn = _rr.rerank
        else:
            print("  WARNING: reranker unavailable — hybrid_with_rerank will "
                  "fall back to RRF order")
        print()

    ts = time.strftime("%Y%m%dT%H%M%S")
    all_results: list[MethodResult] = []
    for ds_name in args.datasets:
        irds = DATASETS[ds_name]["irds"]
        print(f"━━━ {ds_name} ({irds}) ━━━")
        corpus, queries, qrels = load_beir(
            irds, max_queries=args.max_queries, max_docs=args.max_docs,
        )
        print(f"  → {len(corpus)} docs, {len(queries)} queries (with qrels)")
        print("  Building indexes…")
        doc_ids, embs = index_vector(corpus, model)
        bm, _ = index_bm25(corpus)
        for m in methods:
            t0 = time.monotonic()
            r = evaluate(ds_name, m, queries, qrels, corpus, model, doc_ids,
                         embs, bm, reranker_fn=reranker_fn)
            print(f"  {m:22s} nDCG@10={r.ndcg_at_10:.4f} "
                  f"[{r.ndcg_at_10_ci[0]:.3f},{r.ndcg_at_10_ci[1]:.3f}] "
                  f"R@10={r.recall_at_10:.4f} MRR@10={r.mrr_at_10:.4f} "
                  f"({time.monotonic()-t0:.1f}s, {r.mean_latency_ms:.1f}ms/q)",
                  flush=True)
            all_results.append(r)
        _, sp = _write_results(all_results, ts, args)  # incremental flush
        print(f"  → flushed results through {ds_name}: {sp.name}")
        print()

    print(f"All {len(all_results)} (dataset × method) cells evaluated.")
    print(f"Results: benchmarks/results/b2_beir_multi_{ts}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
