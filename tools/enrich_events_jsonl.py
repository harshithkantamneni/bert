"""LLM enrichment pass for the canonical event stream.

Step 1.2 of canvas v2 phase 1 (per
findings/canvas_v2_phase1_plan_2026-05-07.md §6 step 1):

For each event in lab/sor/events.jsonl, run a Cerebras llama3.1-8b
free-tier extraction call against the event's source_path content
to populate two semantic fields the Trail renderer needs:

  tags    — 3-7 hashtag-style free-form tokens reflecting bert's
            natural vocabulary (e.g., #cross-family, #stuck-quaker,
            #falsifier-fail, #provider-deprecation). The Trail's
            color-encoding maps tag → low-saturation ocean palette;
            Mind's region-clustering uses tag co-occurrence.

  lineage — list of paths or D-N IDs that this event derived from.
            Trail's confluences (multiple-stream merges) are
            mathematically defined by lineage cardinality ≥ 2;
            without real lineage the metaphor is a lie.

Quality-first decisions (per the plan):
  - LLM extraction, not regex. Cerebras llama3.1-8b takes ~3s per
    artifact + is free; ~70 artifacts × 3s = ~4 minutes total.
  - Idempotent: skip events with non-empty tags (already enriched).
  - Schema-constrained via core.decode.call_with_schema so the LLM
    emits exactly the {tags, lineage} shape we expect.
  - Conservative truncation: long source files truncated to 8K
    chars before sending (the L1+ canvas reads full content via
    source_path; the LLM only needs enough to recognize themes).

Usage:
  python tools/enrich_events_jsonl.py
  python tools/enrich_events_jsonl.py --limit 10 --dry-run
  python tools/enrich_events_jsonl.py --re-enrich  # ignore tags-already-set check
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import log  # noqa: E402
from core.enrichment import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
)
from core.enrichment import (
    enrich_one as _enrich_one,
)

LOG = log.get_logger("bert.enrich")
EVENTS_PATH = LAB_ROOT / "lab" / "sor" / "events.jsonl"


def enrich_all(
    *,
    events_path: Path = EVENTS_PATH,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    limit: int | None = None,
    re_enrich: bool = False,
    dry_run: bool = False,
    workers: int = 1,
) -> dict:
    """Walk events.jsonl, enrich each event in-place, persist."""
    if not events_path.exists():
        return {"error": f"events.jsonl not found at {events_path}"}

    rows: list[dict] = []
    for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    targets = [r for r in rows if re_enrich or not r.get("tags")]
    if limit is not None:
        targets = targets[:limit]

    LOG.info(
        "enrich: %d total events, %d need enrichment, model=%s/%s, dry_run=%s",
        len(rows), len(targets), provider, model, dry_run,
    )

    stats = {
        "total_events": len(rows),
        "targeted": len(targets),
        "enriched_llm": 0,
        "enriched_heuristic": 0,
        "skipped": 0,
        "elapsed_secs": 0.0,
    }
    if dry_run:
        return stats

    t0 = time.monotonic()
    stats_lock = threading.Lock()

    def _process(ev: dict) -> None:
        result = _enrich_one(ev, provider=provider, model=model)
        if result is None:
            with stats_lock:
                stats["skipped"] += 1
            return
        # Each worker writes to its OWN event dict — no collision; only
        # the shared stats counter needs a lock.
        ev["tags"] = result["tags"]
        ev["lineage"] = result["lineage"]
        ev["enrichment_provenance"] = result["provenance"]
        with stats_lock:
            if result["provenance"] == "llm":
                stats["enriched_llm"] += 1
            else:
                stats["enriched_heuristic"] += 1

    if workers <= 1:
        for i, ev in enumerate(targets):
            _process(ev)
            if (i + 1) % 25 == 0:
                LOG.info("enrich: %d/%d done", i + 1, len(targets))
    else:
        LOG.info("enrich: parallel mode, %d workers", workers)
        with ThreadPoolExecutor(max_workers=workers) as exe:
            futures = [exe.submit(_process, ev) for ev in targets]
            done = 0
            for f in as_completed(futures):
                f.result()  # surface any unhandled exception
                done += 1
                if done % 25 == 0:
                    LOG.info("enrich: %d/%d done", done, len(targets))
    stats["elapsed_secs"] = round(time.monotonic() - t0, 1)

    # Persist all rows back (preserving non-targeted events untouched)
    with events_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")

    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, default=EVENTS_PATH)
    ap.add_argument("--provider", default=DEFAULT_PROVIDER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--re-enrich", action="store_true",
                    help="re-enrich events that already have tags")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel LLM calls (default 1; 5-8 saturates rate limits)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stats = enrich_all(
        events_path=args.events,
        provider=args.provider,
        model=args.model,
        limit=args.limit,
        re_enrich=args.re_enrich,
        workers=args.workers,
        dry_run=args.dry_run,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
