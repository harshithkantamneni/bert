"""Collate the canonical event stream for the Bert canvas.

Walks the lab's archive across multiple corpora and produces a single
chronologically-ordered `lab/sor/events.jsonl` — the source of truth
the Trail / Now / Mind / Strata / Cathedral / Lighthouse / Atlas
renderers read from. One row per logical event.

Sources walked:

  findings/*.md                — 1 event per finding file
                                 (kind="finding", role inferred from
                                 filename prefix: researcher_, architect_,
                                 evaluator_, strategist_, threshing_, etc.)
  state/results/*.json          — 1 event per result packet
                                 (kind="dispatch_result", role+cycle+verdict
                                 from packet)
  memories/log.md               — 1 event per `## D-N` decision entry
                                 (kind="decision", cycle from header)
  lab/sod/seasoning.jsonl       — 1 event per seasoning entry
  state/observability/*.jsonl   — events as-is (already structured;
                                 carries tool_call, model_call,
                                 subagent_*, verdict, etc.)

Schema (per build_roadmap_v2_amendment §1):

  {
    "id": "evt_<sha8>",
    "ts": ISO 8601 UTC,
    "agent": role identifier or null,
    "content": ≤500-char summary (full content via source_path),
    "tags": [],            # populated by enrich_events_jsonl.py (next step)
    "lineage": [],         # populated by enrich_events_jsonl.py
    "cycle": int or null,
    "significance": float or null,
    "phase": OODA marker or null,
    "system": VSM tag or null,
    "event_class": "finding" | "dispatch_result" | "decision" | …,
    "severity_grade": null | "whisper" | "voice" | "weight",
    "memory_tier": null | "core" | "recall" | "archival",
    "judge_provider": null | "provider/model",
    "position_swap_delta": null | float,
    "revival_conditions": null | [str],
    "source_path": str (path to underlying artifact for L3/L4 zoom)
  }

Output: `lab/sor/events.jsonl`, chronologically sorted, deduped by
event id. Re-running is idempotent — existing events.jsonl gets
loaded, new events merged, full file rewritten in time order.

Usage:
  python tools/collate_events_jsonl.py
  python tools/collate_events_jsonl.py --output lab/sor/events.jsonl --verbose
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import log  # noqa: E402

LOG = log.get_logger("bert.collate")
DEFAULT_OUTPUT = LAB_ROOT / "lab" / "sor" / "events.jsonl"


@dataclass
class Event:
    """One row in the canonical event stream. Mirrors the v2.1
    amendment schema. tags + lineage are placeholder-empty here
    (later filled by enrich_events_jsonl.py via LLM extraction)."""
    id: str
    ts: str
    event_class: str
    source_path: str
    agent: str | None = None
    content: str = ""
    tags: list[str] = field(default_factory=list)
    lineage: list[str] = field(default_factory=list)
    cycle: int | None = None
    significance: float | None = None
    phase: str | None = None
    system: str | None = None
    severity_grade: str | None = None
    memory_tier: str | None = None
    judge_provider: str | None = None
    position_swap_delta: float | None = None
    revival_conditions: list[str] | None = None
    confidence_1to10: int | None = None
    verdict: str | None = None


def _hash_id(prefix: str, *parts: Any) -> str:
    # SHA1 used as a deterministic, short id — not for security
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8"),
                     usedforsecurity=False).hexdigest()[:12]
    return f"{prefix}_{h}"


def _iso_from_mtime(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime, tz=UTC).isoformat()


def _rel_to_lab(p: Path) -> str:
    """Best-effort relative path from LAB_ROOT. Falls back to the
    absolute path string if the file is outside the lab tree (e.g.
    when walking a temp directory in tests)."""
    try:
        return str(p.relative_to(LAB_ROOT))
    except ValueError:
        return str(p)


# ── Corpus walkers ──────────────────────────────────────────────────


_ROLE_FROM_FILENAME = re.compile(
    r"^(researcher|architect|evaluator|strategist|threshing|clearness|implementer|director|falsifier|build_roadmap|FINAL)"
)


def _infer_role(filename: str) -> str | None:
    m = _ROLE_FROM_FILENAME.match(filename)
    return m.group(1) if m else None


def _take_summary(text: str, max_chars: int = 500) -> str:
    """First non-empty paragraph or the file's content trimmed."""
    for chunk in text.split("\n\n"):
        s = chunk.strip()
        # Skip pure-header lines
        if not s or all(line.startswith("#") for line in s.splitlines() if line.strip()):
            continue
        return s[:max_chars]
    return text.strip()[:max_chars]


def walk_findings(findings_dir: Path) -> Iterable[Event]:
    """One event per .md file under findings/."""
    if not findings_dir.exists():
        return
    for p in sorted(findings_dir.rglob("*.md")):
        # skip the auto-archive subtree
        if "archive" in p.parts:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = _rel_to_lab(p)
        role = _infer_role(p.name)
        # Try to extract cycle from filename: researcher_C12_*.md or _cycle12_
        cycle = None
        m = re.search(r"_C(\d+)_|_cycle(\d+)|_C(\d+)\.|cycle_(\d+)", p.name)
        if m:
            for grp in m.groups():
                if grp is not None:
                    cycle = int(grp)
                    break
        yield Event(
            id=_hash_id("find", rel),
            ts=_iso_from_mtime(p),
            event_class="finding",
            source_path=rel,
            agent=role,
            content=_take_summary(text),
            cycle=cycle,
        )


def walk_result_packets(results_dir: Path) -> Iterable[Event]:
    """One event per result packet JSON. Skip the archive subtree."""
    if not results_dir.exists():
        return
    for p in sorted(results_dir.glob("*.json")):
        try:
            packet = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rel = _rel_to_lab(p)
        cycle_val = packet.get("cycle")
        cycle = int(cycle_val) if isinstance(cycle_val, (int, float, str)) and str(cycle_val).lstrip("-").isdigit() else None
        ts = _iso_from_mtime(p)
        yield Event(
            id=_hash_id("disp", rel),
            ts=ts,
            event_class="dispatch_result",
            source_path=rel,
            agent=str(packet.get("role") or ""),
            content=(packet.get("calibration_reasoning") or "")[:500],
            cycle=cycle,
            confidence_1to10=packet.get("confidence_1to10"),
            verdict=packet.get("verdict"),
            severity_grade=packet.get("severity_grade"),
            judge_provider=(packet.get("telemetry") or {}).get("model_used"),
        )


def walk_log_decisions(log_path: Path) -> Iterable[Event]:
    """One event per `## D-N` block in memories/log.md."""
    if not log_path.exists():
        return
    text = log_path.read_text(encoding="utf-8", errors="replace")
    # Match `## D-N (...) — title` or just `## D-N`
    blocks = re.split(r"(?m)^##\s+D-(\d+)\b", text)
    base_ts = _iso_from_mtime(log_path)
    rel = _rel_to_lab(log_path)
    # blocks[0] is preamble; pairs of (number, body) follow
    for i in range(1, len(blocks), 2):
        try:
            n = int(blocks[i])
        except ValueError:
            continue
        body_full = blocks[i + 1] if (i + 1) < len(blocks) else ""
        body = re.split(r"(?m)^##\s+", body_full, maxsplit=1)[0].strip()
        # Try to extract cycle from body — pattern "cycle 12" or "C12"
        cycle = None
        m = re.search(r"[cC]ycle\s+(\d+)|\bC(\d+)\b", body[:400])
        if m:
            cycle = int(m.group(1) or m.group(2))
        yield Event(
            id=_hash_id("dec", "D-" + str(n)),
            ts=base_ts,
            event_class="decision",
            source_path=f"{rel}#D-{n}",
            agent="director",
            content=body[:500],
            cycle=cycle,
        )


def walk_seasoning(seasoning_path: Path) -> Iterable[Event]:
    """One event per row in lab/sod/seasoning.jsonl."""
    if not seasoning_path.exists():
        return
    for line in seasoning_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        eid = entry.get("id") or _hash_id("seas", line[:60])
        yield Event(
            id=eid if eid.startswith("seas") else f"seas_{eid}",
            ts=entry.get("ts") or _iso_from_mtime(seasoning_path),
            event_class="seasoning_entry",
            source_path=_rel_to_lab(seasoning_path),
            agent="seasoning",
            content=(entry.get("summary") or "")[:500],
            cycle=entry.get("cycle"),
            severity_grade=entry.get("severity"),
            revival_conditions=entry.get("revival_conditions"),
        )


def walk_observability(obs_dir: Path) -> Iterable[Event]:
    """Events from state/observability/*.jsonl. Skip the archive
    subtree (rotated files); the canvas's L3/L4 retrospective layer
    can read those separately via observability.read_archived()."""
    if not obs_dir.exists():
        return
    for p in sorted(obs_dir.iterdir()):
        if not p.is_file() or p.suffix != ".jsonl":
            continue
        event_class = p.stem  # filename = event_class
        for idx, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = _hash_id(event_class[:4], p.name, idx)
            content_parts = []
            if rec.get("role"):
                content_parts.append(f"role={rec['role']}")
            if rec.get("model"):
                content_parts.append(f"model={rec['model']}")
            if rec.get("provider"):
                content_parts.append(f"provider={rec['provider']}")
            if rec.get("verdict"):
                content_parts.append(f"verdict={rec['verdict']}")
            if rec.get("tool"):
                content_parts.append(f"tool={rec['tool']}")
            content = " · ".join(content_parts)[:500]
            yield Event(
                id=eid,
                ts=rec.get("ts") or _iso_from_mtime(p),
                event_class=event_class,
                source_path=f"{_rel_to_lab(p)}#L{idx + 1}",
                agent=rec.get("role"),
                content=content,
                cycle=rec.get("cycle"),
                verdict=rec.get("verdict"),
                severity_grade=rec.get("severity_grade"),
                judge_provider=rec.get("judge_provider"),
                position_swap_delta=rec.get("position_swap_delta"),
            )


# ── Driver ──────────────────────────────────────────────────────────


def collate(*, output_path: Path = DEFAULT_OUTPUT, verbose: bool = False) -> dict:
    """Walk every corpus, dedupe by id, sort chronologically, write
    output_path. Returns a stats dict."""
    seen: dict[str, Event] = {}
    # Load existing events for idempotency (preserves any tags/lineage
    # populated by a later enrichment pass)
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = rec.get("id")
            if eid:
                seen[eid] = Event(
                    id=rec["id"], ts=rec.get("ts", ""),
                    event_class=rec.get("event_class", "unknown"),
                    source_path=rec.get("source_path", ""),
                    agent=rec.get("agent"),
                    content=rec.get("content", ""),
                    tags=rec.get("tags") or [],
                    lineage=rec.get("lineage") or [],
                    cycle=rec.get("cycle"),
                    significance=rec.get("significance"),
                    phase=rec.get("phase"),
                    system=rec.get("system"),
                    severity_grade=rec.get("severity_grade"),
                    memory_tier=rec.get("memory_tier"),
                    judge_provider=rec.get("judge_provider"),
                    position_swap_delta=rec.get("position_swap_delta"),
                    revival_conditions=rec.get("revival_conditions"),
                    confidence_1to10=rec.get("confidence_1to10"),
                    verdict=rec.get("verdict"),
                )

    counts: dict[str, int] = {}
    walkers = [
        ("findings", walk_findings(LAB_ROOT / "findings")),
        ("result_packets", walk_result_packets(LAB_ROOT / "state" / "results")),
        ("log_decisions", walk_log_decisions(LAB_ROOT / "memories" / "log.md")),
        ("seasoning", walk_seasoning(LAB_ROOT / "lab" / "sod" / "seasoning.jsonl")),
        ("observability", walk_observability(LAB_ROOT / "state" / "observability")),
    ]

    new_count = 0
    refreshed_count = 0
    for source_name, iterator in walkers:
        n = 0
        for ev in iterator:
            n += 1
            existing = seen.get(ev.id)
            if existing is None:
                seen[ev.id] = ev
                new_count += 1
            else:
                # Preserve tags + lineage (LLM-enriched), refresh other fields
                ev.tags = existing.tags or ev.tags
                ev.lineage = existing.lineage or ev.lineage
                seen[ev.id] = ev
                refreshed_count += 1
        counts[source_name] = n
        if verbose:
            LOG.info("collate: %s → %d events", source_name, n)

    # Sort by ts
    all_events = sorted(seen.values(), key=lambda e: e.ts)

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ev in all_events:
            f.write(json.dumps(asdict(ev), default=str, separators=(",", ":")) + "\n")

    stats = {
        "output_path": str(output_path.relative_to(LAB_ROOT)),
        "total_events": len(all_events),
        "new_events": new_count,
        "refreshed_events": refreshed_count,
        "per_source": counts,
        "earliest_ts": all_events[0].ts if all_events else None,
        "latest_ts": all_events[-1].ts if all_events else None,
    }
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    stats = collate(output_path=args.output, verbose=args.verbose)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
