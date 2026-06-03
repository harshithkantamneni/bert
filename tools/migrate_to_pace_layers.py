"""Migration script for the pace-layered file structure.

Gartner SoR / SoD / SoI three-tier pace-layered systems framework
applied to bert's file layout:

  lab/sor/  System of Record — append-only, Merkle-hashed, audit-trail
            perfect. The CANONICAL truth bert can never destroy.
            Examples: events.jsonl (canvas event stream), historical
            ResultPackets, decision log entries (after ratification),
            findings.

  lab/sod/  System of Differentiation — ratification-edit-only. Edits
            require a D-N entry; otherwise immutable. The lab's
            DELIBERATELY MAINTAINED commitments.
            Examples: procedures.md, mission.md, seasoning.jsonl,
            calibration data, governance/* files.

  lab/soi/  System of Innovation — freely mutable. The experimental,
            in-progress, draft surface bert and PI both touch fluidly.
            Examples: cycle_queue.md, current.md, drafts/*, working
            findings before ratification.

  lab/stream/  Live event stream surface for canvas. events.jsonl is
               the canonical canvas data source.

This script provides DRY-RUN ONLY by default. Set --execute to actually
move files. The default dry-run prints the move plan + identifies any
hardcoded references that would break. We don't auto-migrate during the
build phase because:
  - bert's harness is mid-construction; broken references compound
  - the migration is reversible only via git; rather not blow up state
  - safer pattern: create the new structure (Day 4), let new code
    write to the canonical lab/* paths, migrate legacy paths
    incrementally as touched

Run dry-run:  `.venv/bin/python tools/migrate_to_pace_layers.py`
Run execute:  `.venv/bin/python tools/migrate_to_pace_layers.py --execute`
              (NOT recommended during build phase; defer to post-Phase-H4)
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

LAB_ROOT = Path(__file__).resolve().parent.parent


class MovePlan(NamedTuple):
    src: Path                    # current path (relative to LAB_ROOT)
    dst: Path                    # new path (relative to LAB_ROOT)
    tier: str                    # "sor" | "sod" | "soi" | "stream"
    rationale: str               # why this file lives in this tier


# Pace-layer assignments for known bert files. Anything not in this list
# stays where it is until added explicitly (conservative migration).
PLAN: list[MovePlan] = [
    # ── SoR (append-only, Merkle-hashed) ────────────────────────────
    MovePlan(
        src=Path("memories/log.md"),
        dst=Path("lab/sor/log.md"),
        tier="sor",
        rationale="Decision log is append-only by P-004; canonical record",
    ),
    MovePlan(
        src=Path("findings"),
        dst=Path("lab/sor/findings"),
        tier="sor",
        rationale="Research/architect findings are append-only artifacts",
    ),
    MovePlan(
        src=Path("state/results"),
        dst=Path("lab/sor/results"),
        tier="sor",
        rationale="Sub-agent ResultPackets are append-only per P-VS-04",
    ),

    # ── SoD (ratification-edit-only) ────────────────────────────────
    MovePlan(
        src=Path("memories/procedures.md"),
        dst=Path("lab/sod/procedures.md"),
        tier="sod",
        rationale="Pattern catalogue edited only via D-N ratifications",
    ),
    MovePlan(
        src=Path("memories/procedures.index.yaml"),
        dst=Path("lab/sod/procedures.index.yaml"),
        tier="sod",
        rationale="Mirrors procedures.md edit-discipline",
    ),
    MovePlan(
        src=Path("memories/mission.md"),
        dst=Path("lab/sod/mission.md"),
        tier="sod",
        rationale="Mission edits require PI ratification (D-4 in current case)",
    ),
    MovePlan(
        src=Path("memories/governance"),
        dst=Path("lab/sod/governance"),
        tier="sod",
        rationale="Constitutional preamble + values + pi_notes",
    ),
    # seasoning.jsonl will be created directly at lab/sod/seasoning.jsonl;
    # no migration needed because it doesn't exist yet.

    # ── SoI (freely mutable) ────────────────────────────────────────
    MovePlan(
        src=Path("state/cycle_queue.md"),
        dst=Path("lab/soi/cycle_queue.md"),
        tier="soi",
        rationale="Working queue, freely modified per cycle",
    ),
    MovePlan(
        src=Path("state/lab_org_ideas_queue.md"),
        dst=Path("lab/soi/lab_org_ideas_queue.md"),
        tier="soi",
        rationale="Idea queue, freely modified by PI",
    ),
    MovePlan(
        src=Path("memories/current.md"),
        dst=Path("lab/soi/current.md"),
        tier="soi",
        rationale="Hot tier, overwritten by KM each cycle",
    ),
    MovePlan(
        src=Path("memories/heuristics.md"),
        dst=Path("lab/soi/heuristics.md"),
        tier="soi",
        rationale="Working heuristics; promoted to SoD only on ratification",
    ),
    MovePlan(
        src=Path("memories/killed.md"),
        dst=Path("lab/soi/killed.md"),
        tier="soi",
        rationale="Failed-idea log; mutable as lessons evolve",
    ),
    MovePlan(
        src=Path("drafts"),
        dst=Path("lab/soi/drafts"),
        tier="soi",
        rationale="Draft material; explicitly mutable",
    ),
]


SKIP_DIR_NAMES = {
    "node_modules", "dist", "build", "release", ".venv", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "__pycache__", ".git",
}


def _walk_skipping(root: Path, exts: tuple[str, ...]) -> list[Path]:
    """Like Path.rglob(*) but skips known-large vendor/build dirs."""
    results: list[Path] = []
    if not root.exists():
        return results
    stack = [root]
    while stack:
        d = stack.pop()
        if d.name in SKIP_DIR_NAMES:
            continue
        try:
            for child in d.iterdir():
                if child.is_dir():
                    stack.append(child)
                elif any(child.name.endswith(e[1:]) for e in exts):  # "*.py" → ".py"
                    results.append(child)
        except (PermissionError, OSError):
            continue
    return results


def find_hardcoded_refs(plan: list[MovePlan]) -> dict[str, list[str]]:
    """Scan the codebase for hardcoded references to to-be-moved paths.
    Skips node_modules / dist / .venv etc. to keep the scan fast."""
    refs: dict[str, list[str]] = {}
    scan_dirs = [LAB_ROOT / "core", LAB_ROOT / "prompts",
                 LAB_ROOT / "tests", LAB_ROOT / "tools",
                 LAB_ROOT / "phase1", LAB_ROOT / "bot"]
    scan_files: list[Path] = []
    for d in scan_dirs:
        scan_files.extend(_walk_skipping(d, ("*.py", "*.md", "*.ts", "*.tsx", "*.json")))
    for top in (LAB_ROOT / "lab.py", LAB_ROOT / "run.sh", LAB_ROOT / "README.md"):
        if top.exists():
            scan_files.append(top)

    for move in plan:
        src_str = str(move.src)
        hits: list[str] = []
        for f in scan_files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except (PermissionError, IsADirectoryError):
                continue
            if src_str in text:
                rel = str(f.relative_to(LAB_ROOT))
                hits.append(rel)
        if hits:
            refs[src_str] = hits
    return refs


def print_plan(plan: list[MovePlan], refs: dict[str, list[str]]) -> None:
    print("=" * 70)
    print("PACE-LAYER MIGRATION PLAN (DRY-RUN)")
    print("=" * 70)
    print()
    by_tier: dict[str, list[MovePlan]] = {"sor": [], "sod": [], "soi": [], "stream": []}
    for m in plan:
        by_tier.setdefault(m.tier, []).append(m)

    for tier in ("sor", "sod", "soi", "stream"):
        moves = by_tier.get(tier, [])
        if not moves:
            continue
        print(f"── lab/{tier}/ ─────────────────────────────────────────────")
        for m in moves:
            print(f"  {m.src} → {m.dst}")
            print(f"    rationale: {m.rationale}")
            ref_count = len(refs.get(str(m.src), []))
            if ref_count:
                print(f"    ⚠  {ref_count} hardcoded reference(s) in codebase — "
                      f"will need updating before --execute")
        print()

    print("=" * 70)
    print(f"Total moves: {len(plan)}")
    print(f"Files with hardcoded references requiring update: "
          f"{sum(1 for src, hits in refs.items() if hits)}")
    print()

    if refs:
        print("HARDCODED REFERENCES (these must be updated for --execute "
              "to be safe):")
        for src, hits in refs.items():
            print(f"  {src}:")
            for h in hits[:5]:
                print(f"    - {h}")
            if len(hits) > 5:
                print(f"    ... and {len(hits) - 5} more")
        print()
    print("Re-run with --execute to actually move files (NOT recommended "
          "during build phase).")


def execute_plan(plan: list[MovePlan]) -> int:
    failures: list[tuple[Path, str]] = []
    for m in plan:
        src = LAB_ROOT / m.src
        dst = LAB_ROOT / m.dst
        if not src.exists():
            print(f"SKIP (src missing): {m.src}")
            continue
        if dst.exists():
            print(f"SKIP (dst exists): {m.dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(src), str(dst))
            print(f"MOVED: {m.src} → {m.dst}")
        except (OSError, shutil.Error) as e:
            failures.append((m.src, str(e)))
            print(f"FAIL: {m.src}: {e}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate bert files to the pace-layered structure."
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually move files. Without this flag, runs in dry-run mode "
             "(default; recommended during build phase).",
    )
    args = parser.parse_args()

    refs = find_hardcoded_refs(PLAN)

    if not args.execute:
        print_plan(PLAN, refs)
        return 0

    if any(refs.values()):
        print("REFUSING --execute: hardcoded references exist in codebase. "
              "Resolve them first, then re-run.")
        print_plan(PLAN, refs)
        return 1

    return execute_plan(PLAN)


if __name__ == "__main__":
    sys.exit(main())
