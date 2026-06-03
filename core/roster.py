"""Roster manager — per-lab agent roster + organic spawn from templates.

Phase C4 of the v3 plan. The director freely spawns inline specializations
from base templates (per the AGI lab pattern: instantiate template +
inline name; after 3+ reuses, promote to permanent role via C5).

State layout per lab:

    ~/.bert/labs/<lab>/
    ├── lab.yaml                      ← roster_core + roster_active (post-A4)
    ├── agents/
    │   ├── <permanent_role>/
    │   │   ├── procedural.md         ← from library or promoted from inline
    │   │   ├── episodic/
    │   │   └── semantic.md
    │   └── _spawn_tracker.json       ← inline-use counts (this module owns)

Operations:
  - register_template_spawn(lab, template, inline_name, cycle) — tracker write
  - get_role_procedural(lab, role) — returns content (permanent OR template
    + inline rendering); falls back to library base template
  - list_permanent_roster(lab) — what agents/<role>/procedural.md files exist
  - list_specializations(lab) — inline spawns with use counts
  - candidates_for_promotion(lab, threshold=3) — what consolidator should
    promote to permanent
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

LOG = logging.getLogger("bert.roster")

LAB_ROOT = Path(__file__).resolve().parent.parent
LIBRARY_DIR = LAB_ROOT / "core" / "library"


# ── Data shapes ──────────────────────────────────────────────────


@dataclass
class TemplateSpawn:
    """One record of a template instantiation."""
    template: str               # e.g., 'researcher'
    inline_name: str            # e.g., 'patent_researcher'
    first_seen_cycle: int
    last_seen_cycle: int
    use_count: int
    spawned_at_ts: float = field(default_factory=time.time)
    promoted: bool = False      # True after consolidator promotes to permanent


@dataclass
class SpawnTracker:
    """Per-lab tracker — what inline specs have been used + how often."""
    specializations: dict[str, TemplateSpawn] = field(default_factory=dict)

    def key(self, template: str, inline_name: str) -> str:
        return f"{template}::{inline_name}"

    def record_use(self, template: str, inline_name: str, cycle: int) -> None:
        k = self.key(template, inline_name)
        existing = self.specializations.get(k)
        if existing is None:
            self.specializations[k] = TemplateSpawn(
                template=template, inline_name=inline_name,
                first_seen_cycle=cycle, last_seen_cycle=cycle,
                use_count=1,
            )
        else:
            existing.last_seen_cycle = cycle
            existing.use_count += 1

    def to_dict(self) -> dict:
        return {
            "specializations": {
                k: asdict(v) for k, v in self.specializations.items()
            }
        }

    @classmethod
    def from_dict(cls, d: dict) -> SpawnTracker:
        out = cls()
        for k, v in (d.get("specializations") or {}).items():
            out.specializations[k] = TemplateSpawn(**v)
        return out


# ── Persistence ──────────────────────────────────────────────────


def _tracker_path(lab_path: Path) -> Path:
    p = lab_path / "agents" / "_spawn_tracker.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_tracker(lab_path: Path) -> SpawnTracker:
    p = _tracker_path(lab_path)
    if not p.exists():
        return SpawnTracker()
    try:
        return SpawnTracker.from_dict(json.loads(p.read_text()))
    except (json.JSONDecodeError, OSError) as e:
        LOG.warning("tracker load failed: %s; starting fresh", e)
        return SpawnTracker()


def save_tracker(lab_path: Path, tracker: SpawnTracker) -> None:
    p = _tracker_path(lab_path)
    p.write_text(json.dumps(tracker.to_dict(), indent=2, sort_keys=True))


# ── Template lookup ──────────────────────────────────────────────


def _find_template_file(template: str) -> Path | None:
    """Find a template file in core/library/agents/.

    Lookup order:
      1. core/library/agents/_base/<template>.md (base)
      2. core/library/agents/<domain>/<template>.md (specialized)
    """
    base = LIBRARY_DIR / "agents" / "_base" / f"{template}.md"
    if base.exists():
        return base
    # Search all subdirs
    for sub in (LIBRARY_DIR / "agents").glob("*/"):
        if not sub.is_dir():
            continue
        candidate = sub / f"{template}.md"
        if candidate.exists():
            return candidate
    return None


def _permanent_role_dir(lab_path: Path, role: str) -> Path:
    """Where a permanent role lives."""
    return lab_path / "agents" / role


def list_permanent_roster(lab_path: Path) -> list[str]:
    """All permanent agent roles in the lab (have agents/<role>/procedural.md)."""
    out = []
    agents_dir = lab_path / "agents"
    if not agents_dir.exists():
        return []
    for sub in sorted(agents_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        if (sub / "procedural.md").exists():
            out.append(sub.name)
    return out


# ── Public spawn API ─────────────────────────────────────────────


def spawn_inline(
    *,
    lab_path: Path,
    template: str,
    inline_name: str | None = None,
    cycle: int = 0,
) -> dict:
    """Director calls this to spawn an agent from a template + optional
    inline specialization.

    Returns a dict with:
      role:        the resolved role name (template if no inline, else
                   "<template>__<inline_name>")
      procedural:  the procedural.md content (template body + inline note)
      already_permanent: True if this role is already a permanent agent
      template:    base template used
      inline_name: specialization name (if any)
    """
    template = (template or "").strip()
    if not template:
        return {"ok": False, "error": "template required"}

    # If no inline_name, this is a request for the base template itself
    if not inline_name or inline_name == template:
        role = template
        if (_permanent_role_dir(lab_path, role) / "procedural.md").exists():
            return {
                "ok": True, "role": role,
                "template": template, "inline_name": None,
                "already_permanent": True,
                "procedural": (_permanent_role_dir(lab_path, role) /
                                "procedural.md").read_text(),
            }
        tpl_file = _find_template_file(template)
        if tpl_file is None:
            return {"ok": False,
                    "error": f"no template found for {template!r}"}
        return {
            "ok": True, "role": role,
            "template": template, "inline_name": None,
            "already_permanent": False,
            "procedural": tpl_file.read_text(),
        }

    # Inline specialization
    role = f"{template}__{inline_name}"
    tpl_file = _find_template_file(template)
    if tpl_file is None:
        return {"ok": False,
                "error": f"no template found for {template!r}"}

    # Record the spawn for C5 promotion tracking
    tracker = load_tracker(lab_path)
    tracker.record_use(template, inline_name, cycle)
    save_tracker(lab_path, tracker)

    tpl_body = tpl_file.read_text()
    inline_header = (
        f"# Inline specialization — {inline_name}\n\n"
        f"*This agent is a specialization of `{template}` instantiated\n"
        f"by the director for this lab. After 3+ reuses, the consolidator\n"
        f"will propose promoting it to a permanent role.*\n\n"
        f"---\n\n"
    )
    return {
        "ok": True, "role": role,
        "template": template, "inline_name": inline_name,
        "already_permanent": False,
        "procedural": inline_header + tpl_body,
        "use_count": tracker.specializations[
            tracker.key(template, inline_name)
        ].use_count,
    }


def list_specializations(lab_path: Path) -> list[dict]:
    """List all inline specializations + their use counts (for the
    director's brief + the consolidator's promotion scan)."""
    tracker = load_tracker(lab_path)
    return [
        {**asdict(spawn), "key": k}
        for k, spawn in tracker.specializations.items()
    ]


def candidates_for_promotion(
    lab_path: Path, *, threshold: int = 3,
) -> list[TemplateSpawn]:
    """Inline specs used ≥ threshold times AND not yet promoted.

    The consolidator (C5) reads this + writes the permanent
    agents/<role>/procedural.md."""
    tracker = load_tracker(lab_path)
    return [
        s for s in tracker.specializations.values()
        if s.use_count >= threshold and not s.promoted
    ]


def mark_promoted(lab_path: Path, template: str, inline_name: str) -> bool:
    """Mark a specialization as promoted (called by consolidator after
    it writes the permanent role file)."""
    tracker = load_tracker(lab_path)
    k = tracker.key(template, inline_name)
    s = tracker.specializations.get(k)
    if s is None:
        return False
    s.promoted = True
    save_tracker(lab_path, tracker)
    return True


# ── CLI ──────────────────────────────────────────────────────────


def _cli(argv: list[str]) -> int:
    """python -m core.roster spawn <lab> <template> [<inline_name>] [cycle]
    python -m core.roster list <lab>
    python -m core.roster candidates <lab> [threshold]
    """
    import sys
    if len(argv) < 2:
        print("usage: roster spawn|list|candidates ...", file=sys.stderr)
        return 2
    cmd = argv[1]
    if cmd == "spawn":
        if len(argv) < 4:
            print("usage: roster spawn <lab> <template> [inline] [cycle]",
                  file=sys.stderr)
            return 2
        lab = Path(argv[2]).expanduser()
        result = spawn_inline(
            lab_path=lab,
            template=argv[3],
            inline_name=argv[4] if len(argv) >= 5 else None,
            cycle=int(argv[5]) if len(argv) >= 6 else 0,
        )
        print(json.dumps({k: v for k, v in result.items()
                          if k != "procedural"}, indent=2))
        return 0 if result.get("ok") else 1
    if cmd == "list":
        if len(argv) < 3:
            print("usage: roster list <lab>", file=sys.stderr)
            return 2
        lab = Path(argv[2]).expanduser()
        permanent = list_permanent_roster(lab)
        specs = list_specializations(lab)
        print(json.dumps({
            "permanent": permanent,
            "specializations": specs,
        }, indent=2))
        return 0
    if cmd == "candidates":
        if len(argv) < 3:
            print("usage: roster candidates <lab> [threshold]",
                  file=sys.stderr)
            return 2
        lab = Path(argv[2]).expanduser()
        threshold = int(argv[3]) if len(argv) >= 4 else 3
        cands = candidates_for_promotion(lab, threshold=threshold)
        print(json.dumps([asdict(c) for c in cands], indent=2))
        return 0
    print(f"unknown cmd: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(_cli(sys.argv))
