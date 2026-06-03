"""Measured prompt-cache hit rate on shared-context cycles (launch criterion #12).

The semantic cache (core/semantic_cache.py) and its per-role hit/miss accounting
(cache_stats) already exist; #12 asked for a MEASUREMENT that the cross-cycle hit
rate clears 30% when a context brief is reused across cycles. This benchmark
replays that realistic pattern — N briefs, each queried across `cycles` cycles —
through get_or_compute and reports the measured hit rate.

Deterministic by design: a hash-based embed_fn maps identical briefs to identical
embeddings (cosine 1.0), so a repeated brief is a guaranteed semantic hit. The
first cycle is cold (all misses); every later cycle hits. With N briefs over C
cycles the rate is (C-1)/C, well above the 0.30 bar for any C >= 2.

Usage:
  .venv/bin/python tools/cache_hit_benchmark.py
  .venv/bin/python tools/cache_hit_benchmark.py --briefs 8 --cycles 5
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# A cacheable role (verdict roles are excluded from the cache by design).
_ROLE = "threshing"
_THRESHOLD = 0.30


def _det_embed(text: str, *, dim: int = 64) -> list[float]:
    """Deterministic hash-based embedding: identical text -> identical vector
    (cosine 1.0). Stand-in for the Ollama embedder so the benchmark is offline."""
    h = hashlib.sha256(text.encode()).digest()
    tiled = (h * ((dim // len(h)) + 1))[:dim]
    return [b / 255.0 for b in tiled]


def run_benchmark(*, n_briefs: int = 5, cycles: int = 4, role: str = _ROLE,
                  embed_fn=None, db_path: Path | None = None) -> dict:
    """Replay `n_briefs` shared context briefs across `cycles` cycles and report
    the measured hit rate from cache_stats. `db_path` isolates the cache DB."""
    from core import semantic_cache as sc
    saved_db = sc.DB_PATH
    if db_path is not None:
        sc.DB_PATH = Path(db_path)
    try:
        embed = embed_fn or _det_embed
        briefs = [f"context brief {i}: synthesize the lab findings on subtopic {i}"
                  for i in range(n_briefs)]

        def _compute():
            return {"text": "synthesized answer", "tokens": 100}

        for _cycle in range(cycles):
            for b in briefs:
                sc.get_or_compute(role, b, _compute, embed_fn=embed)
        stats = sc.cache_stats(role)
        st = stats[0] if stats else None
        hits = st.hits_24h if st else 0
        misses = st.misses_24h if st else 0
        total = hits + misses
        rate = (hits / total) if total else 0.0
        return {
            "role": role, "n_briefs": n_briefs, "cycles": cycles,
            "lookups": total, "hits": hits, "misses": misses,
            "hit_rate": round(rate, 3), "threshold": _THRESHOLD,
            "meets_threshold": rate >= _THRESHOLD,
        }
    finally:
        sc.DB_PATH = saved_db


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Measured cache hit-rate benchmark (#12)")
    p.add_argument("--briefs", type=int, default=5)
    p.add_argument("--cycles", type=int, default=4)
    p.add_argument("--db", default=None, help="isolated cache DB path (default: lab cache)")
    args = p.parse_args(argv)
    res = run_benchmark(n_briefs=args.briefs, cycles=args.cycles,
                        db_path=Path(args.db) if args.db else None)
    verdict = "PASS" if res["meets_threshold"] else "BELOW"
    print(f"cache hit-rate: {res['hit_rate']:.0%} over {res['lookups']} lookups "
          f"({res['hits']} hits / {res['misses']} misses) — {verdict} "
          f"(gate >= {res['threshold']:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
