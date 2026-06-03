"""Slice retrieval events by timestamp window to analyze each Zipfian
alpha run separately. We don't tag events at write time, so we recover
the split from known gen start/end times.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
OBS_DIR = LAB_ROOT / "state" / "observability"


def all_retrieval_events() -> list[dict]:
    out: list[dict] = []
    for p in [OBS_DIR / "retrieval.jsonl"] + sorted((OBS_DIR / "archive").rglob("retrieval_*.jsonl")):
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def slice_window(events: list[dict], start: str, end: str) -> list[dict]:
    """Filter events with ts ∈ [start, end] (ISO strings, UTC)."""
    s = dt.datetime.fromisoformat(start)
    e = dt.datetime.fromisoformat(end)
    out = []
    for ev in events:
        ts = ev.get("ts")
        if not ts:
            continue
        try:
            t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            continue
        if s <= t <= e:
            out.append(ev)
    return out


def zipfian_stats(events: list[dict]) -> dict:
    counts: Counter[str] = Counter()
    for ev in events:
        q = (ev.get("query") or "").strip()
        if q:
            counts[q] += 1
    if not counts:
        return {"n_total": 0, "n_unique": 0}
    n_total = sum(counts.values())
    n_unique = len(counts)
    freqs = sorted(counts.values(), reverse=True)
    top1 = freqs[0]
    top5 = sum(freqs[:5])
    top10 = sum(freqs[:10])
    top20 = sum(freqs[:20])
    top50 = sum(freqs[:50])
    top100 = sum(freqs[:100])
    # Cache hit rate (LFU): top-K queries cached, hits = total_K_hits - K
    cache = {}
    for K in (5, 10, 20, 50, 100, 200):
        if n_unique < K:
            continue
        hits = max(0, sum(freqs[:K]) - K)
        cache[K] = hits / n_total
    return {
        "n_total": n_total,
        "n_unique": n_unique,
        "top1_pct": top1 / n_total,
        "top5_pct": top5 / n_total,
        "top10_pct": top10 / n_total,
        "top20_pct": top20 / n_total,
        "top50_pct": top50 / n_total,
        "top100_pct": top100 / n_total,
        "top1_top5_ratio": top1 / top5 if top5 else 0,
        "cache_hit_rate": cache,
    }


def main():
    # Known runs (UTC). Padded a few seconds for safety.
    runs = [
        ("α=1.0",  "2026-05-26T19:10:25+00:00", "2026-05-26T19:10:58+00:00"),
        ("α=0.8",  "2026-05-26T19:12:55+00:00", "2026-05-26T19:13:15+00:00"),
        ("α=1.5",  "2026-05-26T19:13:15+00:00", "2026-05-26T19:13:30+00:00"),
        ("α=0.6",  "2026-05-26T19:13:30+00:00", "2026-05-26T19:13:45+00:00"),
    ]
    events = all_retrieval_events()
    print(f"Total events available: {len(events)}")
    print()
    print(f"{'Run':<8} {'n_total':>8} {'n_uniq':>8} "
          f"{'top1%':>7} {'top5%':>7} {'top10%':>8} "
          f"{'t1/t5':>6} "
          f"{'K=5':>6} {'K=10':>6} {'K=20':>6} {'K=50':>6}")
    print("-" * 90)
    for label, start, end in runs:
        sliced = slice_window(events, start, end)
        s = zipfian_stats(sliced)
        if not s["n_total"]:
            print(f"{label:<8} (no events in window)")
            continue
        c = s["cache_hit_rate"]
        print(f"{label:<8} {s['n_total']:>8} {s['n_unique']:>8} "
              f"{s['top1_pct']:>6.1%} {s['top5_pct']:>6.1%} {s['top10_pct']:>7.1%} "
              f"{s['top1_top5_ratio']:>6.2f} "
              f"{c.get(5,0):>5.1%} {c.get(10,0):>5.1%} {c.get(20,0):>5.1%} {c.get(50,0):>5.1%}")
    print()


if __name__ == "__main__":
    main()
