"""SkillOS-style zero-invocation skill pruning.

Scans skills/active/, computes
each skill's invocation count from lab/sor/events.jsonl over the last
N cycles (default 30), and demotes (moves to skills/archived/) any
skill with zero invocations.

Run nightly via the same launchd plist installed by F.6's
setup_backup_cron.sh, or manually.

Usage:
  python tools/prune_zero_invocation_skills.py --dry-run
  python tools/prune_zero_invocation_skills.py --cycles 30
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
EVENTS_PATH = LAB_ROOT / "lab" / "sor" / "events.jsonl"
ACTIVE_DIR = LAB_ROOT / "skills" / "active"
ARCHIVED_DIR = LAB_ROOT / "skills" / "archived"


def _max_cycle() -> int:
    if not EVENTS_PATH.exists():
        return 0
    last = 0
    for line in EVENTS_PATH.read_text().splitlines()[-2000:]:
        try:
            ev = json.loads(line)
            c = ev.get("cycle")
            if isinstance(c, int) and c > last:
                last = c
        except json.JSONDecodeError:
            continue
    return last


def _invocation_counts(min_cycle: int) -> Counter:
    """Count tool_call events per tool_name within cycle range."""
    counter: Counter = Counter()
    if not EVENTS_PATH.exists():
        return counter
    for line in EVENTS_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event_class") != "tool_call":
            continue
        cycle = ev.get("cycle")
        if isinstance(cycle, int) and cycle < min_cycle:
            continue
        # Pull tool name from common shapes
        name = (ev.get("tool_name") or ev.get("tool")
                or ev.get("name") or "")
        if name:
            counter[name] += 1
    return counter


def _list_active_skills() -> list[Path]:
    if not ACTIVE_DIR.exists():
        return []
    return sorted(p for p in ACTIVE_DIR.iterdir() if p.is_dir())


def _skill_name(skill_dir: Path) -> str:
    """Derive the registered tool name from the SKILL.md frontmatter."""
    md = skill_dir / "SKILL.md"
    if not md.exists():
        return skill_dir.name
    for line in md.read_text().splitlines()[:20]:
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip()
    return skill_dir.name


def prune(*, cycles: int = 30, dry_run: bool = False) -> dict:
    max_cycle = _max_cycle()
    min_cycle = max(0, max_cycle - cycles)
    counts = _invocation_counts(min_cycle)
    skills = _list_active_skills()
    summary: dict = {
        "max_cycle": max_cycle,
        "window_min_cycle": min_cycle,
        "active_skills_total": len(skills),
        "pruned": [],
        "kept": [],
        "dry_run": dry_run,
    }
    if not dry_run:
        ARCHIVED_DIR.mkdir(parents=True, exist_ok=True)
    for skill_dir in skills:
        name = _skill_name(skill_dir)
        n = counts.get(name, 0)
        entry = {"skill_dir": skill_dir.name, "name": name, "invocations": n}
        if n == 0 and (max_cycle - min_cycle) >= cycles:
            if not dry_run:
                dest = ARCHIVED_DIR / skill_dir.name
                if dest.exists():
                    import shutil
                    shutil.rmtree(dest)
                skill_dir.rename(dest)
                try:
                    entry["moved_to"] = str(dest.relative_to(LAB_ROOT))
                except ValueError:
                    entry["moved_to"] = str(dest)
            summary["pruned"].append(entry)
        else:
            summary["kept"].append(entry)
    summary["pruned_count"] = len(summary["pruned"])
    summary["kept_count"] = len(summary["kept"])
    summary["ts"] = datetime.now(UTC).isoformat()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune zero-invocation skills")
    parser.add_argument("--cycles", type=int, default=30,
                        help="Cycle window to count invocations over")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report plan without moving files")
    args = parser.parse_args()
    print(json.dumps(prune(cycles=args.cycles, dry_run=args.dry_run),
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
