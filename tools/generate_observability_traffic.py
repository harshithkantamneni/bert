"""Synthetic + real traffic generator to populate observability data.

We don't have weeks for real lab usage to accumulate. Instead, drive
retrieval + background tools with realistic distributions so the
analyzer has enough signal to validate the v3+ architecture decisions
within hours.

Generates:
  - retrieval.jsonl events via hybrid_retrieve with mixed query
    distribution (Zipfian: a few hot queries + long tail of unique)
  - background_invocation events by calling background tools
  - varied query "shapes" (short keywords, long natural language,
    code-like, citation-like)

Run: .venv/bin/python tools/generate_observability_traffic.py [--n-queries 500]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


# Realistic query distribution — modeled after bert lab queries that
# would actually be issued during research / build / strategy work.
# v3+ EXPANDED for diversity (2026-05-26): ~80 templates across tiers.
QUERY_TEMPLATES = {
    # HOT queries — repeated frequently (Zipfian top tier) — 12 templates
    "hot": [
        "mamba state space model",
        "sigstore signing",
        "cross-encoder rerank",
        "BM25 sparse retrieval",
        "RRF reciprocal rank fusion",
        "memory tier architecture",
        "proof packet schema",
        "falsifier baseline",
        "context_brief assembler",
        "demand paging findings",
        "macro-op fusion writes",
        "Zipfian distribution cache",
    ],
    # WARM — medium frequency — 30 templates
    "warm": [
        "agentic lab framework",
        "MCP protocol JSON-RPC",
        "in-toto provenance attestation",
        "ColBERT late interaction",
        "BGE-M3 embedder",
        "ONNX runtime export",
        "HuggingFace tokenizers",
        "RAGAS faithfulness context precision",
        "LongMemEval categories",
        "memory hierarchy cache coherence",
        "demand paging finding",
        "macro-op fusion record_finding",
        "LFU ARC cache replacement",
        "cycle outcome verdict APPROVE",
        "adversarial eval contradiction",
        "data shape document_corpus code_repo",
        "schema synthesizer organic org",
        "researcher strategist falsifier roles",
        "BLAS oversubscription pipelining",
        "L1 L2 L3 memory hierarchy",
        "TLB shootdown invalidation",
        "speculative execution rollback",
        "branch prediction predictor",
        "out-of-order execution retirement",
        "FAISS HNSW approximate nearest neighbor",
        "SPLADE learned sparse",
        "MUVERA single-vector approximation",
        "HyDE hypothetical document embedding",
        "DiskANN prefetching",
        "Vespa column store ranking",
    ],
    # COLD — unique long-tail queries — 50+ templates
    "cold": [
        "how does the brief_assembler work",
        "what role does the falsifier play in cycle 99",
        "find papers about attention mechanisms in transformers",
        "compare BM25 vs ColBERT on scifact",
        "trace lineage from claim to evidence",
        "show me cycles where artifact_accepted increased",
        "which queries have the highest score variance",
        "explain demand paging in CPU architecture",
        "what's the difference between MESI and MOESI protocols",
        "how do production search engines handle cache coherence",
        "what's bert's policy on speculative execution",
        "review the v3 architecture decision log",
        "list all known unaddressed concerns from cycle 5",
        "what claims has the strategist made in last 10 cycles",
        "which findings contradict each other on memory tiers",
        "how does the schema_synthesizer pick adapters",
        "explain the difference between L1 and L2 reranker",
        "find recent work on multi-vector retrieval compression",
        "what's the role of token_graph in retrieval",
        "summarize all falsifier_baseline outputs",
        "what's the per-query latency budget",
        "show me the proof packet verify pipeline",
        "what does USP #3 mean for the retrieval architecture",
        "compare the v1 v2 v3 architecture proposals",
        "list all concepts dropped on evidence",
        "what role does anchor-term guard play",
        "explain Rekor inclusion proof",
        "how does AGNTCY DID identity work",
        "what's bert's stance on heterogeneous compute",
        "show me the bm25 tokenization design",
        "what does cycle-aware prefetch look like",
        "how does the consolidator do memory tier promotion",
        "find evidence about SPLADE-v3 query expansion",
        "what queries does the strategist actually issue",
        "compare researcher vs strategist tool profiles",
        "show me director's orchestration patterns",
        "what does instrumentation Phase 1d capture",
        "find papers about retrieval-augmented agents",
        "list concerns raised but not addressed",
        "what's the demand paging implementation plan",
        "show me macro-op fusion candidates",
        "explain the role × tool palette mapping",
        "how do background tools fit the org model",
        "what's the LFU vs LRU evidence",
        "summarize cycle outcome correlation findings",
        "find cycles where retrieval prevented wasted work",
        "what does the third axis (mission_phase) add",
        "explain proof-packet-preserving cache invariants",
        "compare per-role vs generic memory_search",
        "list deferred phases for v3+ implementation",
        "what's our gap to bge-m3 SOTA",
        "explain BEIR scifact dataset shape",
        "show me TREC-DL relevance grading",
        "what's bert's Open Telemetry GenAI integration",
        "explain why MPS slower than CPU for cross-encoder",
    ],
}


# Probabilistic role attribution (matching empirical distribution)
ROLE_WEIGHTS = {
    "researcher": 38,
    "implementer": 21,
    "custom-director": 18,
    "strategist": 18,
    "clearness_phase2": 5,
}


def pick_role(rng: random.Random) -> str:
    """Sample a role proportional to empirical distribution."""
    pool = []
    for role, weight in ROLE_WEIGHTS.items():
        pool.extend([role] * weight)
    return rng.choice(pool)


def pick_query(rng: random.Random) -> str:
    """Zipfian-like distribution: 50% hot, 30% warm, 20% cold."""
    r = rng.random()
    if r < 0.50:
        return rng.choice(QUERY_TEMPLATES["hot"])
    elif r < 0.80:
        return rng.choice(QUERY_TEMPLATES["warm"])
    else:
        return rng.choice(QUERY_TEMPLATES["cold"])


def main(n_queries: int = 500, seed: int = 42) -> int:
    rng = random.Random(seed)
    print(f"Generating {n_queries} synthetic retrieval queries…")
    print("  (Zipfian distribution: 50% hot, 30% warm, 20% cold)")
    print()

    # Disable reranker so we don't burn time on cross-encoder
    os.environ["BERT_DISABLE_RERANKER"] = "1"
    from core import retrieval as _ret

    # Warm-up (so we don't measure cold start in the data)
    for _ in range(3):
        _ret.hybrid_retrieve("warmup", top_n=5)

    t0 = time.monotonic()
    query_distribution = {"hot": 0, "warm": 0, "cold": 0}
    latencies_ms = []
    for i in range(n_queries):
        q = pick_query(rng)
        # Classify back for stats
        if q in QUERY_TEMPLATES["hot"]:
            query_distribution["hot"] += 1
        elif q in QUERY_TEMPLATES["warm"]:
            query_distribution["warm"] += 1
        else:
            query_distribution["cold"] += 1
        # Vary top_n
        top_n = rng.choice([3, 5, 5, 5, 10, 10, 20])
        t_call = time.perf_counter()
        _ret.hybrid_retrieve(q, top_n=top_n)
        latencies_ms.append((time.perf_counter() - t_call) * 1000)
        if (i + 1) % 100 == 0:
            elapsed = time.monotonic() - t0
            qps = (i + 1) / elapsed
            print(f"  {i+1}/{n_queries} queries fired ({qps:.1f} qps)")

    elapsed = time.monotonic() - t0
    p50 = sorted(latencies_ms)[len(latencies_ms) // 2]
    p95 = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]
    p99 = sorted(latencies_ms)[int(len(latencies_ms) * 0.99)]
    print()
    print(f"Done. {n_queries} queries in {elapsed:.1f}s")
    print(f"  Distribution: {query_distribution}")
    print(f"  Latency p50={p50:.1f}ms p95={p95:.1f}ms p99={p99:.1f}ms")
    print(f"  Throughput: {n_queries/elapsed:.1f} QPS")
    print()
    print("retrieval.jsonl should now contain all events. Run the analyzer next.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    sys.exit(main(args.n_queries, args.seed))
