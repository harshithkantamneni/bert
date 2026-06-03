"""Pure retrieval-quality + aggregation metrics for the B9 long-context RAG
benchmark. No network, no deps — fully unit-testable. Scores the RETRIEVER
(recall@k / nDCG@k vs gold chunk ids) separately from the reader so a wrong
answer can be attributed to retrieval miss vs reading failure.
"""

from __future__ import annotations

import math
from collections import defaultdict


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """Fraction of gold chunk ids present in the top-k retrieved. 0.0 if no gold."""
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    return len(top & set(gold)) / len(set(gold))


def hit_rate_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """1.0 if at least one gold chunk is in the top-k, else 0.0."""
    if not gold:
        return 0.0
    return 1.0 if set(retrieved[:k]) & set(gold) else 0.0


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """Binary-relevance nDCG@k: DCG over the top-k retrieved (rel=1 if in gold)
    normalized by the ideal DCG (all gold ranked first). 0.0 if no gold."""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 2)
              for rank, cid in enumerate(retrieved[:k]) if cid in gold_set)
    ideal_hits = min(k, len(gold_set))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def recall_spans(retrieved_texts: list[str], gold_spans: list[str], k: int) -> float:
    """Content-match recall: fraction of gold verbatim substrings present in the
    top-k retrieved texts. Robust to internal chunk-id churn (gold is defined by
    distinctive substrings, not chunk ids). 0.0 if no gold spans."""
    if not gold_spans:
        return 0.0
    blob = "\n".join(retrieved_texts[:k])
    return sum(1 for s in gold_spans if s in blob) / len(gold_spans)


def hit_spans(retrieved_texts: list[str], gold_spans: list[str], k: int) -> float:
    """1.0 if ANY gold span appears in the top-k retrieved texts, else 0.0."""
    if not gold_spans:
        return 0.0
    blob = "\n".join(retrieved_texts[:k])
    return 1.0 if any(s in blob for s in gold_spans) else 0.0


def est_tokens(text: str) -> int:
    """~4 chars/token heuristic — for the input-tokens-vs-corpus-size curve."""
    return len(text or "") // 4


def _mean_or_none(vals: list) -> float | None:
    nums = [v for v in vals if v is not None]
    return sum(nums) / len(nums) if nums else None


def aggregate_by_arm(rows: list[dict]) -> dict:
    """Aggregate per-query rows into per-arm accuracy + cost + retrieval quality.
    `correct` is 0/1 (judge verdict); retrieval metrics are None for non-RAG arms
    and excluded from their means (don't dilute accuracy with N/A retrieval)."""
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    out: dict[str, dict] = {}
    for arm, rs in by_arm.items():
        n = len(rs)
        out[arm] = {
            "n": n,
            "accuracy": sum(r.get("correct", 0) for r in rs) / n if n else 0.0,
            "mean_input_tokens": _mean_or_none([r.get("input_tokens") for r in rs]),
            "mean_recall_at_10": _mean_or_none([r.get("recall_at_10") for r in rs]),
            "mean_ndcg_at_10": _mean_or_none([r.get("ndcg_at_10") for r in rs]),
            "mean_cost_usd": _mean_or_none([r.get("cost_usd") for r in rs]),
            "feasible": sum(1 for r in rs if not r.get("infeasible")),
        }
    return out
