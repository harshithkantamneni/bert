"""Join cycle_outcome events with retrieval activity by time window.

For each cycle outcome, find all retrieval events that fell within
its [cycle_start, cycle_end] window. cycle_start = event ts -
elapsed_secs, cycle_end = event ts. Report:

  - retrievals per cycle (mean, p50, p99)
  - success rate WITH retrieval (n≥1) vs WITHOUT
  - per-verdict retrieval activity
  - artifacts_accepted by retrieval activity
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
from collections import Counter
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
OBS_DIR = LAB_ROOT / "state" / "observability"


def read_jsonl(name: str) -> list[dict]:
    """Read live + archived JSONL for an event class."""
    out: list[dict] = []
    for p in [OBS_DIR / name] + sorted((OBS_DIR / "archive").rglob(f"{name.replace('.jsonl', '')}_*.jsonl")):
        if not p.exists():
            continue
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


def parse_ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def main():
    cycles = read_jsonl("cycle_outcome.jsonl")
    retrievals = read_jsonl("retrieval.jsonl")
    print(f"  cycle_outcomes: {len(cycles)}")
    print(f"  retrieval events: {len(retrievals)}")
    print()

    # Build retrieval timestamp index for fast lookup
    ret_by_ts = sorted(
        [(parse_ts(r["ts"]), r) for r in retrievals if "ts" in r],
        key=lambda x: x[0],
    )

    # Only consider cycles with real elapsed_secs (skip backfilled n/a)
    real_cycles = [c for c in cycles if c.get("elapsed_secs")]
    print(f"  cycles with real timing data: {len(real_cycles)}")
    print()

    cycle_records = []
    for c in real_cycles:
        end = parse_ts(c["ts"])
        start = end - dt.timedelta(seconds=c["elapsed_secs"])
        # Count retrievals in window
        rets_in_window = [r for ts, r in ret_by_ts if start <= ts <= end]
        cycle_records.append({
            "cycle_id": c["cycle_id"],
            "success": c["success"],
            "elapsed_secs": c["elapsed_secs"],
            "verdicts": c.get("verdicts", []),
            "artifacts": c.get("artifacts_accepted", 0),
            "findings": c.get("findings_produced", 0),
            "n_retrievals": len(rets_in_window),
            "retrievals": rets_in_window,
        })

    if not cycle_records:
        print("(no cycles with timing data)")
        return

    # Sort by retrievals so we can split with vs without
    with_ret = [c for c in cycle_records if c["n_retrievals"] > 0]
    without_ret = [c for c in cycle_records if c["n_retrievals"] == 0]

    print("=== Retrieval activity per cycle ===")
    print(f"  total cycles with timing data: {len(cycle_records)}")
    print(f"  cycles WITH retrieval (n≥1):    {len(with_ret)}  "
          f"({100*len(with_ret)/len(cycle_records):.0f}%)")
    print(f"  cycles WITHOUT retrieval:        {len(without_ret)}")
    print()

    if with_ret:
        ns = [c["n_retrievals"] for c in with_ret]
        ns.sort()
        print(f"  retrievals/cycle (with):  mean={statistics.mean(ns):.1f}  "
              f"p50={ns[len(ns)//2]}  max={max(ns)}")
        elapsed_with = [c["elapsed_secs"] for c in with_ret]
        print(f"  cycle duration (with):    mean={statistics.mean(elapsed_with):.0f}s")
    if without_ret:
        elapsed_without = [c["elapsed_secs"] for c in without_ret]
        print(f"  cycle duration (without): mean={statistics.mean(elapsed_without):.0f}s")
    print()

    print("=== Success rate × retrieval ===")
    if with_ret:
        sr = sum(1 for c in with_ret if c["success"]) / len(with_ret)
        print(f"  success WITH retrieval:    {sr:.0%} (n={len(with_ret)})")
    if without_ret:
        sr = sum(1 for c in without_ret if c["success"]) / len(without_ret)
        print(f"  success WITHOUT retrieval: {sr:.0%} (n={len(without_ret)})")
    print()

    print("=== Artifacts accepted × retrieval ===")
    if with_ret:
        avg_art = statistics.mean(c["artifacts"] for c in with_ret)
        print(f"  artifacts WITH retrieval:    {avg_art:.2f} (n={len(with_ret)})")
    if without_ret:
        avg_art = statistics.mean(c["artifacts"] for c in without_ret)
        print(f"  artifacts WITHOUT retrieval: {avg_art:.2f} (n={len(without_ret)})")
    print()

    print("=== Verdict mix in cycles WITH retrieval ===")
    verdicts = Counter()
    for c in with_ret:
        for v in c["verdicts"]:
            verdicts[v] += 1
    for v, n in verdicts.most_common():
        print(f"  {v}: {n}")
    print()

    print("=== Verdict mix in cycles WITHOUT retrieval ===")
    verdicts = Counter()
    for c in without_ret:
        for v in c["verdicts"]:
            verdicts[v] += 1
    for v, n in verdicts.most_common():
        print(f"  {v}: {n}")


if __name__ == "__main__":
    main()
