"""Organic-vs-benchmark retrieval traffic analysis (memory-v3+ measurement tool).

The deferred memory-architecture decisions (Tier-1 result cache, demand-paging
defer-page-in, schema-aware tools) all hinge on ORGANIC retrieval traffic — but
the log is dominated by benchmark/test traffic (all tagged lab="lab", the
supervisor lab where benchmarks and tests run). This tool separates organic
(named user labs) from benchmark and reports the cache + demand-paging metrics
per origin, so those decisions auto-re-measure as real multi-lab data accrues.

It packages the one-off analysis from
benchmarks/results/tier1_cache_decision_2026-05-30.md into a reusable gate:
re-run it whenever organic data has accumulated; if the organic LFU hit rate at
K<=50 clears 30%, the Tier-1 cache re-opens.

Usage:
  .venv/bin/python tools/retrieval_traffic_analysis.py
  .venv/bin/python tools/retrieval_traffic_analysis.py <path/to/retrieval.jsonl>
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
DEFAULT_LOG = _REPO / "state" / "observability" / "retrieval.jsonl"
# Labs treated as benchmark/test traffic, not organic mission work.
SUPERVISOR_LABS = frozenset({"lab", None, ""})
TIER1_GATE = 0.30  # organic LFU hit rate (K<=50) needed to re-open the cache
# Minimum organic queries before the hit-rate recommendation is trustworthy.
# A tiny sample (e.g. a hand-seeded pipeline check) can clear the gate by chance;
# the Tier-1 decision needs a real volume of autonomous-agent traffic.
MIN_ORGANIC_N = 200


def load_events(path: Path | None = None) -> list[dict]:
    p = Path(path) if path is not None else DEFAULT_LOG
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def split_by_origin(events: list[dict], *,
                    supervisor_labs: frozenset = SUPERVISOR_LABS) -> dict:
    """Partition events into organic (named user labs) vs benchmark (supervisor)."""
    organic, benchmark = [], []
    for e in events:
        (benchmark if e.get("lab") in supervisor_labs else organic).append(e)
    return {"organic": organic, "benchmark": benchmark}


def lfu_hit_rate(queries: list[str], k: int) -> float:
    """Hit rate replaying queries through an LFU cache of size k."""
    cache: dict[str, int] = {}
    freq: collections.Counter = collections.Counter()
    hits = 0
    for q in queries:
        if q in cache:
            hits += 1
        elif len(cache) >= k:
            victim = min(cache, key=lambda x: freq[x])
            del cache[victim]
        cache[q] = 1
        freq[q] += 1
    return hits / len(queries) if queries else 0.0


def windowed_hit_rate(queries: list[str], w: int) -> float:
    """Hit rate of a TTL-analog cache: a query hits if it recurred within the
    last `w` calls."""
    from collections import deque
    recent: deque = deque(maxlen=w)
    seen: set[str] = set()
    hits = 0
    for q in queries:
        if q in seen:
            hits += 1
        recent.append(q)
        seen = set(recent)
    return hits / len(queries) if queries else 0.0


def repeat_rate(queries: list[str]) -> float:
    """Fraction of queries that are repeats (theoretical max hit rate)."""
    if not queries:
        return 0.0
    return 1.0 - len(set(queries)) / len(queries)


def _origin_metrics(events: list[dict], *, k_values=(10, 20, 50),
                    w_values=(10, 50, 100)) -> dict:
    queries = [e.get("query", "") for e in events if e.get("query")]
    return {
        "n": len(queries),
        "unique": len(set(queries)),
        "repeat_rate": round(repeat_rate(queries), 3),
        "lfu": {k: round(lfu_hit_rate(queries, k), 3) for k in k_values},
        "windowed": {w: round(windowed_hit_rate(queries, w), 3) for w in w_values},
    }


def analyze(path: Path | None = None) -> dict:
    events = load_events(path)
    split = split_by_origin(events)
    by_lab = collections.Counter(e.get("lab") for e in events)
    organic = _origin_metrics(split["organic"])
    benchmark = _origin_metrics(split["benchmark"])
    if organic["n"] == 0:
        rec = "defer (no organic data)"
    elif organic["n"] < MIN_ORGANIC_N:
        rec = f"insufficient organic data (n={organic['n']}, need >={MIN_ORGANIC_N})"
    else:
        best = max(organic["lfu"].get(k, 0.0) for k in (10, 20, 50))
        rec = "re-open (organic hit rate clears gate)" if best >= TIER1_GATE else "defer (organic hit rate below gate)"
    return {
        "total": len(events),
        "by_lab": dict(by_lab),
        "organic": organic,
        "benchmark": benchmark,
        "tier1_gate": TIER1_GATE,
        "tier1_cache_recommendation": rec,
    }


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    path = Path(argv[0]) if argv else DEFAULT_LOG
    rep = analyze(path)
    print(f"retrieval traffic: {rep['total']} events across {len(rep['by_lab'])} labs")
    print("  by lab:", ", ".join(f"{k}={v}" for k, v in
                                  sorted(rep["by_lab"].items(), key=lambda kv: -kv[1])))
    for origin in ("organic", "benchmark"):
        m = rep[origin]
        print(f"  {origin}: {m['n']} queries · {m['unique']} unique · "
              f"repeat={m['repeat_rate']:.0%} · LFU "
              + " ".join(f"K{k}={v:.0%}" for k, v in m["lfu"].items()))
    print(f"  Tier-1 cache: {rep['tier1_cache_recommendation']} "
          f"(gate: organic LFU K<=50 >= {rep['tier1_gate']:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
