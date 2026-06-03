"""Lightweight BM25-only traffic generator.

When the embedder cold-start is being killed by system memory pressure
(macOS mediaanalysisd / iCloud sync), we still want diverse retrieval
events to validate Zipfian distribution + cache hit rates. This gen
monkey-patches `core.memory.search` to return [] before calling
hybrid_retrieve, so each call exercises BM25 + graph + RRF + (no
vector) and emits a retrieval event with bm25-only signal.

This is honest synthetic data:
  - Real BM25 lookups against the real corpus
  - Real RRF mixing of available signals
  - Empty vector signal (clearly marked in the event payload)
  - No torch / no embedder required → fast cold start

Use to complement the full diverse-traffic gen, not replace.

Run:
  .venv/bin/python tools/generate_bm25_traffic.py --target 3000
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

# Reuse the 320-template diverse set from the full generator
from tools.generate_diverse_traffic import QUERY_CLUSTERS, flatten, zipf_sample


def main(target: int, alpha: float, max_hours: float, seed: int) -> int:
    rng = random.Random(seed)
    templates = flatten()
    rng.shuffle(templates)

    print(f"[{time.strftime('%H:%M:%S')}] starting BM25-only traffic gen")
    print(f"  templates: {len(templates)} across {len(QUERY_CLUSTERS)} clusters")
    print(f"  target:    {target}  alpha: {alpha}  max_hours: {max_hours}")
    sys.stdout.flush()

    # Disable reranker AND patch vector source to []
    os.environ["BERT_DISABLE_RERANKER"] = "1"
    import core.memory as _mem
    _original_search = _mem.search
    _mem.search = lambda q, k=20: []
    print(f"[{time.strftime('%H:%M:%S')}] patched core.memory.search → [] (BM25-only mode)")
    sys.stdout.flush()

    from core import retrieval as _ret
    print(f"[{time.strftime('%H:%M:%S')}] retrieval module imported")
    sys.stdout.flush()

    # No warmup needed — BM25 is hot from previous events
    t = time.perf_counter()
    _ret.hybrid_retrieve("warmup", top_n=3)
    print(f"[{time.strftime('%H:%M:%S')}] first call: {(time.perf_counter()-t)*1000:.0f}ms")
    sys.stdout.flush()

    deadline = time.monotonic() + max_hours * 3600
    t0 = time.monotonic()
    latencies = []
    cluster_counts = dict.fromkeys(QUERY_CLUSTERS, 0)
    template_to_cluster = {}
    for cname, tlist in QUERY_CLUSTERS.items():
        for t in tlist:
            template_to_cluster[t] = cname

    completed = 0
    errors = 0
    while completed < target:
        if time.monotonic() > deadline:
            print(f"[{time.strftime('%H:%M:%S')}] deadline reached")
            break
        q = zipf_sample(rng, templates, alpha)
        cluster_counts[template_to_cluster[q]] += 1
        top_n = rng.choice([3, 5, 5, 5, 10, 10, 20])
        try:
            t = time.perf_counter()
            _ret.hybrid_retrieve(q, top_n=top_n)
            latencies.append((time.perf_counter() - t) * 1000)
            completed += 1
        except Exception:
            errors += 1
            time.sleep(0.5)
            continue

        if completed % 100 == 0:
            elapsed = time.monotonic() - t0
            qps = completed / elapsed if elapsed > 0 else 0.0
            srt = sorted(latencies)
            p50 = srt[len(srt) // 2]
            p95 = srt[int(len(srt) * 0.95)]
            print(f"[{time.strftime('%H:%M:%S')}] {completed}/{target}  "
                  f"qps={qps:.1f}  p50={p50:.0f}ms p95={p95:.0f}ms")
            sys.stdout.flush()

    elapsed = time.monotonic() - t0
    print()
    print(f"[{time.strftime('%H:%M:%S')}] Done. {completed} BM25-only queries in {elapsed:.1f}s")
    if latencies:
        srt = sorted(latencies)
        print(f"  p50={srt[len(srt)//2]:.1f}ms p95={srt[int(len(srt)*0.95)]:.1f}ms")
        print(f"  throughput {completed/elapsed:.2f} QPS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=3000)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--max-hours", type=float, default=4.5)
    ap.add_argument("--seed", type=int, default=239)
    args = ap.parse_args()
    sys.exit(main(args.target, args.alpha, args.max_hours, args.seed))
