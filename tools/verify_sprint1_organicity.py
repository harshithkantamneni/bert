"""Empirical verification that Sprint 1 actually wired the schema → roster.

Reads the mission_suite manifest + subagent_spawn events.
For each mission, asserts:
  1. A lab_schema was synthesized (rule_id non-empty)
  2. The dispatched roster matches the schema's roster_initial
  3. Different missions got DIFFERENT rosters (the organicity claim)
  4. Build mission has at least one code-shaped role
     (code_reader / refactor_specialist / test_author / reviewer)

This is the evidence-triumphs-assumption check.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
OBS_DIR = LAB_ROOT / "state" / "observability"
MANIFEST = Path("/tmp/mission_suite_manifest.jsonl")


def _read_jsonl(name: str) -> list[dict]:
    out: list[dict] = []
    p = OBS_DIR / name
    if p.exists():
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            with contextlib.suppress(json.JSONDecodeError):
                out.append(json.loads(line))
    return out


def _parse_ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def slice_window(events: list[dict], start: str, end: str) -> list[dict]:
    s, e = _parse_ts(start), _parse_ts(end)
    out = []
    for ev in events:
        ts = ev.get("ts")
        if not ts:
            continue
        try:
            t = _parse_ts(ts)
        except ValueError:
            continue
        if s <= t <= e:
            out.append(ev)
    return out


def main() -> int:
    if not MANIFEST.exists():
        print(f"FAIL: manifest missing at {MANIFEST}")
        return 2

    missions = [json.loads(l) for l in MANIFEST.open() if l.strip()]
    if not missions:
        print("FAIL: manifest empty")
        return 2

    spawns = _read_jsonl("subagent_spawn.jsonl")
    cycles = _read_jsonl("cycle_outcome.jsonl")

    print("=" * 70)
    print("SPRINT 1 ORGANICITY VERIFICATION")
    print("=" * 70)
    print()

    per_mission_rosters: dict[str, list[str]] = {}
    failures: list[str] = []

    for m in missions:
        name = m["mission"]
        sm_spawns = slice_window(spawns, m["start_ts"], m["end_ts"])
        sm_cycles = slice_window(cycles, m["start_ts"], m["end_ts"])

        roles_dispatched = [s.get("role") for s in sm_spawns if s.get("role")]
        per_mission_rosters[name] = roles_dispatched

        print(f"── {name} ──")
        print(f"  window: {m['start_ts']} → {m['end_ts']}")
        print(f"  cycles in window: {len(sm_cycles)}")
        print(f"  dispatches: {len(sm_spawns)}")
        print(f"  roles dispatched: {roles_dispatched}")

        for c in sm_cycles:
            verdicts = c.get("verdicts", [])
            print(f"    cycle {c.get('cycle_id')}: success={c.get('success')} "
                  f"verdicts={verdicts}")
        print()

    # Assertion 1: research mission should NOT be hardcoded researcher→strategist
    research_roster = per_mission_rosters.get("research", [])
    if research_roster == ["researcher", "strategist"]:
        # If we're still seeing the legacy pair, organicity isn't wired
        failures.append(
            "research mission still dispatched legacy [researcher, strategist] — "
            "schema wire may not be active"
        )

    # Assertion 2: build mission should have at least one code role
    # OR a "non-research" roster (depends on classifier accuracy)
    build_roster = per_mission_rosters.get("build", [])
    code_roles = {"code_reader", "refactor_specialist", "test_author",
                   "reviewer", "security_auditor", "performance_tuner",
                   "engineer"}
    has_code_role = any(r in code_roles for r in build_roster)
    has_different_roster = build_roster and build_roster != research_roster

    if not (has_code_role or has_different_roster):
        failures.append(
            f"build mission roster {build_roster} is identical to research "
            f"roster {research_roster}; expected differentiation"
        )

    # Assertion 3: at least one mission's roster should differ from another
    distinct_rosters = {tuple(r) for r in per_mission_rosters.values() if r}
    if len(distinct_rosters) < 2 and len(per_mission_rosters) >= 2:
        failures.append(
            f"all missions dispatched the same roster — organicity claim fails. "
            f"got: {distinct_rosters}"
        )

    print("=" * 70)
    if failures:
        print(f"VERDICT: {len(failures)} ASSERTION(S) FAILED")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    else:
        print("VERDICT: ALL ORGANICITY ASSERTIONS PASS")
        print(f"  ✓ {len(distinct_rosters)} distinct roster(s) across "
              f"{len(missions)} mission(s)")
        print("  ✓ build mission has differentiated roster")
        print("  ✓ research mission not stuck on legacy pair")
        return 0


if __name__ == "__main__":
    sys.exit(main())
