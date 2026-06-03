"""Compare cycle outcome metrics across the 3 mission types.

Uses the manifest from `tools/run_mission_suite.sh` to slice
cycle_outcome.jsonl + retrieval.jsonl by mission window.

Output: a side-by-side table of metrics for {research, build, analysis}.
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
OBS_DIR = LAB_ROOT / "state" / "observability"
MANIFEST = Path("/tmp/mission_suite_manifest.jsonl")


def _read_jsonl(name: str) -> list[dict]:
    out: list[dict] = []
    paths: list[Path] = []
    live = OBS_DIR / name
    if live.exists():
        paths.append(live)
    arch = OBS_DIR / "archive"
    if arch.exists():
        stem = name.replace(".jsonl", "")
        for day in sorted(arch.iterdir()):
            if day.is_dir():
                paths.extend(sorted(day.glob(f"{stem}_*.jsonl")))
    for p in paths:
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


def _parse_ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def slice_by_window(events: list[dict], start_iso: str, end_iso: str) -> list[dict]:
    s = _parse_ts(start_iso)
    e = _parse_ts(end_iso)
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


def main():
    if not MANIFEST.exists():
        print(f"Manifest missing: {MANIFEST}")
        print("Run tools/run_mission_suite.sh first.")
        return 1

    missions = []
    with MANIFEST.open() as f:
        for line in f:
            line = line.strip()
            if line:
                missions.append(json.loads(line))

    cycles = _read_jsonl("cycle_outcome.jsonl")
    retrievals = _read_jsonl("retrieval.jsonl")
    tool_calls = _read_jsonl("tool_call.jsonl")

    print("=== Mission outcome comparison ===\n")

    headers = ["metric"] + [m["mission"] for m in missions]
    rows = []

    per_mission_cycles = {}
    per_mission_rets = {}
    per_mission_tcs = {}

    for m in missions:
        c = slice_by_window(cycles, m["start_ts"], m["end_ts"])
        r = slice_by_window(retrievals, m["start_ts"], m["end_ts"])
        t = slice_by_window(tool_calls, m["start_ts"], m["end_ts"])
        per_mission_cycles[m["mission"]] = c
        per_mission_rets[m["mission"]] = r
        per_mission_tcs[m["mission"]] = t

    def row(name: str, fn):
        rows.append([name] + [fn(m["mission"]) for m in missions])

    row("cycles_total",
        lambda m: str(len(per_mission_cycles[m])))
    row("cycles_real_timing",
        lambda m: str(sum(1 for c in per_mission_cycles[m] if c.get("elapsed_secs"))))
    row("success_rate_HONEST",
        lambda m: (
            f"{sum(1 for c in per_mission_cycles[m] if c.get('success')) / max(len(per_mission_cycles[m]), 1):.0%}"
            if per_mission_cycles[m] else "n/a"
        ))
    row("artifacts_per_cycle",
        lambda m: (
            f"{statistics.mean(c.get('artifacts_accepted', 0) for c in per_mission_cycles[m]):.2f}"
            if per_mission_cycles[m] else "n/a"
        ))
    row("findings_per_cycle",
        lambda m: (
            f"{statistics.mean(c.get('findings_produced', 0) for c in per_mission_cycles[m]):.2f}"
            if per_mission_cycles[m] else "n/a"
        ))
    row("elapsed_p50_secs",
        lambda m: (
            f"{sorted(c['elapsed_secs'] for c in per_mission_cycles[m] if c.get('elapsed_secs'))[len([c for c in per_mission_cycles[m] if c.get('elapsed_secs')])//2]:.0f}"
            if any(c.get("elapsed_secs") for c in per_mission_cycles[m]) else "n/a"
        ))
    row("retrievals_total",
        lambda m: str(len(per_mission_rets[m])))
    row("retrievals_per_cycle",
        lambda m: (
            f"{len(per_mission_rets[m]) / max(sum(1 for c in per_mission_cycles[m] if c.get('elapsed_secs')), 1):.2f}"
        ))
    row("tool_calls_total",
        lambda m: str(len(per_mission_tcs[m])))
    row("tool_calls_per_cycle",
        lambda m: (
            f"{len(per_mission_tcs[m]) / max(sum(1 for c in per_mission_cycles[m] if c.get('elapsed_secs')), 1):.1f}"
        ))

    # Verdict breakdown
    print("Cycle outcome metrics:")
    print()
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=False)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(r, widths, strict=False)))
    print()

    print("Verdict distribution per mission:")
    print()
    for m in missions:
        name = m["mission"]
        verdicts = Counter()
        for c in per_mission_cycles[name]:
            for v in c.get("verdicts", []):
                verdicts[v] += 1
        print(f"  {name}:")
        for v, n in verdicts.most_common():
            print(f"    {v}: {n}")
        if not verdicts:
            print("    (no verdicts)")
        print()

    print("Most-used tools per mission:")
    print()
    for m in missions:
        name = m["mission"]
        tools = Counter(t.get("tool", "?") for t in per_mission_tcs[name])
        print(f"  {name}:")
        for t, n in tools.most_common(8):
            print(f"    {t}: {n}")
        print()


if __name__ == "__main__":
    sys.exit(main() or 0)
