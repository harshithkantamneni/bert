"""Export bert state for the static web companion (F.10).

Per C4 §5.9 days 13-18. Produces:
  web/data/atlas.json     — single-lab Atlas summary
  web/data/cathedral.json — PI-curated page exports

Respects lab/PRIVATE.md: nothing in that file's listing gets exported.
Designed to run as part of the same nightly cron that does backups.

Usage:
  python tools/export_for_web.py
  python tools/export_for_web.py --max-cathedral-pages 8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent.parent
EVENTS_PATH = LAB_ROOT / "lab" / "sor" / "events.jsonl"
SEASONING_PATH = LAB_ROOT / "lab" / "sod" / "seasoning.jsonl"
PRIVATE_MD = LAB_ROOT / "lab" / "PRIVATE.md"
WEB_DATA = LAB_ROOT / "web" / "data"
FINDINGS_DIR = LAB_ROOT / "findings"
CATHEDRAL_DIR = LAB_ROOT / "lab" / "soi" / "cathedral"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _private_paths() -> set[str]:
    """Read lab/PRIVATE.md for paths to exclude from the export."""
    paths: set[str] = set()
    if not PRIVATE_MD.exists():
        return paths
    for line in PRIVATE_MD.read_text().splitlines():
        m = re.match(r"^- `?([^`\s]+)`?", line.strip())
        if m:
            paths.add(m.group(1).rstrip("/"))
    return paths


def build_atlas() -> dict[str, Any]:
    """Compose the atlas.json shape consumed by web/index.html."""
    events = _read_jsonl(EVENTS_PATH)
    class_counts: Counter = Counter()
    agents: Counter = Counter()
    Counter()
    Counter()
    Counter()
    last_cycle = None
    for ev in events:
        ec = ev.get("event_class")
        class_counts[ec] = class_counts.get(ec, 0) + 1
        agent = ev.get("agent")
        if agent:
            agents[agent] += 1
        if isinstance(ev.get("cycle"), int):
            last_cycle = ev["cycle"]
    seasoning_entries = _read_jsonl(SEASONING_PATH)

    # Pull live quota stats for cache hit %
    try:
        sys.path.insert(0, str(LAB_ROOT))
        from core import quota as _quota
        prov_stats = _quota.stats()
    except Exception:
        prov_stats = {}

    return {
        "ts": _now_iso(),
        "cycle": last_cycle,
        "totalEvents": len(events),
        "verdictCount": class_counts.get("verdict", 0) + class_counts.get("stand_aside_verdict", 0),
        "findingCount": class_counts.get("finding", 0),
        "seasoningCount": len(seasoning_entries),
        "agents": [
            {"agent": a, "count": n}
            for a, n in agents.most_common(20)
        ],
        "providers": [
            {
                "name": name,
                "calls_24h": s.get("calls_24h", 0),
                "cache_hit_pct_24h": s.get("cache_hit_pct_24h", 0),
            }
            for name, s in sorted(prov_stats.items(), key=lambda kv: kv[1].get("calls_24h", 0), reverse=True)
        ],
    }


def build_cathedral(*, max_pages: int = 8) -> dict[str, Any]:
    """Curate PI-blessed findings as Cathedral pages.

    Source: findings/*.md files. We skip private paths and pick the
    most recent N by mtime. Each page gets:
      {id, title, role, cycle, ts, paragraphs: [...]}
    """
    private = _private_paths()
    pages: list[dict] = []
    sources: list[Path] = []
    if CATHEDRAL_DIR.exists():
        sources.extend(sorted(CATHEDRAL_DIR.glob("*.md"),
                              key=lambda p: p.stat().st_mtime, reverse=True))
    if FINDINGS_DIR.exists():
        sources.extend(sorted(FINDINGS_DIR.glob("strategist_*.md"),
                              key=lambda p: p.stat().st_mtime, reverse=True))
        sources.extend(sorted(FINDINGS_DIR.glob("architect_*.md"),
                              key=lambda p: p.stat().st_mtime, reverse=True))

    for p in sources:
        rel = str(p.relative_to(LAB_ROOT))
        if any(priv in rel for priv in private):
            continue
        try:
            text = p.read_text()
        except OSError:
            continue
        title_match = re.search(r"^#\s+(.+)$", text, re.M)
        title = title_match.group(1).strip() if title_match else p.stem
        # Detect role + cycle from filename pattern
        role = "anon"
        cycle = None
        m_role = re.match(r"(\w+?)_", p.stem)
        if m_role:
            role = m_role.group(1)
        m_cycle = re.search(r"_C(\d+)", p.stem)
        if m_cycle:
            cycle = int(m_cycle.group(1))
        # First few non-empty paragraphs after the H1
        paragraphs: list[str] = []
        body = text.split("\n", 1)[1] if "\n" in text else text
        for chunk in body.split("\n\n"):
            chunk = chunk.strip()
            if not chunk or chunk.startswith(("#", "```", "-", "|")):
                continue
            paragraphs.append(chunk[:1200])
            if len(paragraphs) >= 5:
                break
        pages.append({
            "id": p.stem,
            "title": title,
            "role": role,
            "cycle": cycle,
            "ts": datetime.fromtimestamp(p.stat().st_mtime, tz=UTC).isoformat(),
            "paragraphs": paragraphs,
        })
        if len(pages) >= max_pages:
            break

    return {
        "ts": _now_iso(),
        "pages": pages,
    }


def export(*, max_cathedral_pages: int = 8, output_dir: Path | None = None) -> dict:
    out_dir = output_dir or WEB_DATA
    out_dir.mkdir(parents=True, exist_ok=True)
    atlas = build_atlas()
    cathedral = build_cathedral(max_pages=max_cathedral_pages)
    (out_dir / "atlas.json").write_text(json.dumps(atlas, indent=2))
    (out_dir / "cathedral.json").write_text(json.dumps(cathedral, indent=2))
    return {
        "atlas_path": str((out_dir / "atlas.json").relative_to(LAB_ROOT)) if out_dir == WEB_DATA else str(out_dir / "atlas.json"),
        "cathedral_path": str((out_dir / "cathedral.json").relative_to(LAB_ROOT)) if out_dir == WEB_DATA else str(out_dir / "cathedral.json"),
        "atlas_events": atlas.get("totalEvents"),
        "atlas_agents": len(atlas.get("agents", [])),
        "atlas_providers": len(atlas.get("providers", [])),
        "cathedral_pages": len(cathedral.get("pages", [])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export bert state for the web companion")
    parser.add_argument("--max-cathedral-pages", type=int, default=8)
    args = parser.parse_args()
    summary = export(max_cathedral_pages=args.max_cathedral_pages)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
