"""Analyze observability data to inform v3+ architectural decisions.

Reads:
  state/observability/retrieval.jsonl           ← per-call retrieval data
  state/observability/cycle_outcome.jsonl       ← per-cycle rollups
  state/observability/background_invocation.jsonl ← background tools
  state/observability/tool_call.jsonl           ← agent tool calls
  state/observability/verdict.jsonl             ← verdicts per dispatch

Reports:
  1. Retrieval latency distribution + per-stage breakdown
  2. Query frequency distribution (test the Zipfian hypothesis empirically)
  3. Per-signal contribution to final top-K
  4. Cache potential (hit rate if Tier 1 result cache had been live)
  5. Cycle outcome correlations (success rate × retrieval rate)
  6. Cross-encoder utility check (rerank impact on ordering)
  7. Background tool footprint

Run: .venv/bin/python tools/analyze_observability.py
     .venv/bin/python tools/analyze_observability.py --output report.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
OBS_DIR = LAB_ROOT / "state" / "observability"


def _read_jsonl(name: str) -> list[dict]:
    """Read live JSONL plus any rotated archives for the same event class."""
    out: list[dict] = []
    paths: list[Path] = []
    live = OBS_DIR / name
    if live.exists():
        paths.append(live)
    archive_root = OBS_DIR / "archive"
    if archive_root.exists():
        stem = name.replace(".jsonl", "")
        for day_dir in sorted(archive_root.iterdir()):
            if not day_dir.is_dir():
                continue
            for p in sorted(day_dir.glob(f"{stem}_*.jsonl")):
                paths.append(p)
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


def _pct(part: int | float, whole: int | float) -> str:
    if not whole:
        return "0%"
    return f"{100*part/whole:.1f}%"


# ── Analysis sections ─────────────────────────────────────────────


def section_retrieval_latency(events: list[dict]) -> list[str]:
    out = ["## Retrieval latency distribution", ""]
    if not events:
        out.append("(no retrieval events)")
        return out
    totals = [e["timings_ms"]["total_ms"] for e in events if "timings_ms" in e]
    if not totals:
        out.append("(no timing data)")
        return out
    totals.sort()
    n = len(totals)
    p50 = totals[n // 2]
    p95 = totals[int(n * 0.95)]
    p99 = totals[min(int(n * 0.99), n - 1)]
    out += [
        f"- Total calls: **{n}**",
        f"- p50: **{p50:.1f}ms**, p95: {p95:.1f}ms, p99: {p99:.1f}ms",
        f"- min: {totals[0]:.1f}ms, max: {totals[-1]:.1f}ms, mean: {statistics.mean(totals):.1f}ms",
        "",
        "### Per-stage breakdown (mean ms)",
        "",
    ]
    stages = ["vector_ms", "bm25_ms", "graph_ms", "rrf_ms", "rerank_ms"]
    means = {s: [] for s in stages}
    for e in events:
        for s in stages:
            v = e.get("timings_ms", {}).get(s, 0)
            if v is not None:
                means[s].append(v)
    for s in stages:
        if means[s]:
            avg = statistics.mean(means[s])
            out.append(f"- {s}: {avg:.2f}ms (mean of {len(means[s])} calls)")
    out.append("")
    return out


def section_query_distribution(events: list[dict]) -> list[str]:
    out = ["## Query frequency distribution (Zipfian check)", ""]
    if not events:
        out.append("(no events)")
        return out
    # Hash queries to count uniques + frequencies
    counts: Counter[str] = Counter()
    for e in events:
        q = (e.get("query") or "").strip()
        if q:
            counts[q] += 1
    n_unique = len(counts)
    n_total = sum(counts.values())
    if not n_unique:
        out.append("(no queries)")
        return out
    sorted_counts = sorted(counts.values(), reverse=True)
    top1 = sorted_counts[0] if sorted_counts else 0
    top5 = sum(sorted_counts[:5])
    top10 = sum(sorted_counts[:10])
    out += [
        f"- Total queries: {n_total}",
        f"- Unique queries: {n_unique}",
        f"- Repeat rate: {_pct(n_total - n_unique, n_total)}",
        f"- top-1 query: {top1} times ({_pct(top1, n_total)} of all calls)",
        f"- top-5 queries hold {_pct(top5, n_total)} of all calls",
        f"- top-10 queries hold {_pct(top10, n_total)} of all calls",
    ]
    # Zipfian check: top-1 / top-5 ratio
    if top5:
        ratio = top1 / top5
        out.append(f"- top-1 / top-5 ratio: {ratio:.2f} "
                    f"(> 0.4 = strong Zipfian → LFU/ARC preferred over LRU)")
    # Top 5 hottest queries
    out += ["", "### Hottest 5 queries", ""]
    for q, c in counts.most_common(5):
        out.append(f"- `{q[:60]}` — {c} hits ({_pct(c, n_total)})")
    out.append("")
    return out


def section_signal_contribution(events: list[dict]) -> list[str]:
    out = ["## Signal contribution to final top-K", ""]
    if not events:
        out.append("(no events)")
        return out
    # How often does each signal contribute to a final result?
    source_in_topk: Counter[str] = Counter()
    n_with_results = 0
    multi_source_topk = 0
    for e in events:
        topk = e.get("final_top_k") or []
        if not topk:
            continue
        n_with_results += 1
        seen_sources = set()
        for r in topk:
            for s in r.get("sources") or []:
                source_in_topk[s] += 1
                seen_sources.add(s)
        if len(seen_sources) > 1:
            multi_source_topk += 1
    out += [
        f"- Queries with results: {n_with_results}",
        f"- Queries where final top-K had multiple signals: {multi_source_topk} ({_pct(multi_source_topk, n_with_results)})",
        "",
        "### How often each signal appears in final top-K",
        "",
    ]
    for s, c in source_in_topk.most_common():
        out.append(f"- {s}: {c} occurrences across all queries ({_pct(c, sum(source_in_topk.values()))})")
    out.append("")
    return out


def section_cache_potential(events: list[dict]) -> list[str]:
    out = ["## Tier 1 result cache potential", ""]
    if not events:
        out.append("(no events)")
        return out
    # If we'd had a cache with various sizes, what hit rate would we see?
    # Assume LFU eviction. Simulate.
    counts: Counter[str] = Counter()
    n_total = 0
    for e in events:
        q = (e.get("query") or "").strip()
        if q:
            counts[q] += 1
            n_total += 1
    if not counts:
        out.append("(no queries)")
        return out
    # Sort by frequency, calculate cumulative hits for different K
    freqs = sorted(counts.values(), reverse=True)
    out += [
        f"- Total queries: {n_total}",
        f"- Unique queries: {len(counts)}",
        "",
        "### Hit rate by cache size (LFU eviction)",
        "",
    ]
    for K in (5, 10, 20, 50, 100, 200):
        if len(freqs) < K:
            continue
        cached_hits = sum(freqs[:K]) - K  # -K because first hit per cached entry is a miss
        cached_hits = max(0, cached_hits)
        cached_hits / n_total if n_total else 0
        out.append(f"- K={K:3d}: hit rate **{_pct(cached_hits, n_total)}** "
                    f"({cached_hits} hits of {n_total} calls)")
    out.append("")
    return out


def section_cycle_outcome_correlation(retrieval_events: list[dict],
                                        cycle_events: list[dict],
                                        tool_call_events: list[dict]) -> list[str]:
    out = ["## Cycle outcome correlation", ""]
    if not cycle_events:
        out.append("(no cycle_outcome events — backfill or run cycles)")
        return out
    # Per-cycle outcome
    cycle_index = {e["cycle_id"]: e for e in cycle_events if isinstance(e.get("cycle_id"), int)}
    n_total = len(cycle_index)
    n_success = sum(1 for e in cycle_index.values() if e.get("success"))
    out += [
        f"- Total cycles graded: {n_total}",
        f"- Successful: {n_success} ({_pct(n_success, n_total)})",
        f"- With ≥1 artifact accepted: {sum(1 for e in cycle_index.values() if e.get('artifacts_accepted', 0) > 0)}",
    ]
    # If we have tool_call events with cycle, count retrieval-per-cycle
    retrieval_per_cycle: dict[int, int] = defaultdict(int)
    for e in tool_call_events:
        if e.get("tool") == "memory_search":
            cid = e.get("cycle")
            if isinstance(cid, int):
                retrieval_per_cycle[cid] += 1
    out += [
        "",
        "### Retrieval activity by cycle outcome",
        "",
    ]
    cycles_with_retrieval = sum(1 for cid in cycle_index if retrieval_per_cycle.get(cid, 0) > 0)
    out.append(f"- Cycles with ≥1 memory_search: {cycles_with_retrieval}/{n_total} ({_pct(cycles_with_retrieval, n_total)})")
    # Cross-tab: success rate WITH retrieval vs WITHOUT
    with_r = [cid for cid in cycle_index if retrieval_per_cycle.get(cid, 0) > 0]
    without_r = [cid for cid in cycle_index if retrieval_per_cycle.get(cid, 0) == 0]
    if with_r:
        sr_with = sum(1 for cid in with_r if cycle_index[cid].get("success")) / len(with_r)
        out.append(f"- Success rate (cycles WITH retrieval): {sr_with:.0%} (n={len(with_r)})")
    if without_r:
        sr_without = sum(1 for cid in without_r if cycle_index[cid].get("success")) / len(without_r)
        out.append(f"- Success rate (cycles WITHOUT retrieval): {sr_without:.0%} (n={len(without_r)})")
    out.append("")
    return out


def section_background_tools(events: list[dict]) -> list[str]:
    out = ["## Background tool footprint", ""]
    if not events:
        out.append("(no background_invocation events)")
        return out
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        by_tool[e.get("tool", "?")].append(e)
    for tool, runs in by_tool.items():
        durations = [r.get("duration_ms") for r in runs if r.get("duration_ms")]
        n_findings = sum(len(r.get("findings_produced") or []) for r in runs)
        success_rate = sum(1 for r in runs if r.get("success")) / len(runs)
        out.append(f"- `{tool}`: {len(runs)} runs, avg duration "
                    f"{statistics.mean(durations)/1000:.1f}s, "
                    f"{n_findings} findings produced, "
                    f"success rate {success_rate:.0%}")
    out.append("")
    return out


def section_per_query_class_latency(events: list[dict]) -> list[str]:
    """Bucket queries by length (proxy for complexity) and compare latencies."""
    out = ["## Latency by query characteristics", ""]
    if not events:
        return out
    buckets: dict[str, list[float]] = defaultdict(list)
    for e in events:
        qlen = e.get("query_len") or 0
        total = e.get("timings_ms", {}).get("total_ms")
        if total is None:
            continue
        if qlen <= 20:
            bucket = "short (≤20 chars)"
        elif qlen <= 60:
            bucket = "medium (21-60)"
        else:
            bucket = "long (60+)"
        buckets[bucket].append(total)
    for bucket, lats in buckets.items():
        if len(lats) < 2:
            continue
        p50 = sorted(lats)[len(lats) // 2]
        out.append(f"- {bucket}: n={len(lats)}, p50={p50:.1f}ms, mean={statistics.mean(lats):.1f}ms")
    out.append("")
    return out


# ── Main ─────────────────────────────────────────────────────────


def main(output_path: str | None = None) -> int:
    retrieval_events = _read_jsonl("retrieval.jsonl")
    cycle_events = _read_jsonl("cycle_outcome.jsonl")
    bg_events = _read_jsonl("background_invocation.jsonl")
    tool_call_events = _read_jsonl("tool_call.jsonl")
    verdict_events = _read_jsonl("verdict.jsonl")

    lines = [
        "# bert observability analysis",
        "",
        f"_Generated from data in `{OBS_DIR}` — {sum([len(retrieval_events), len(cycle_events), len(bg_events), len(tool_call_events), len(verdict_events)])} total events analyzed_",
        "",
        "## Event stream sizes",
        "",
        f"- retrieval.jsonl: {len(retrieval_events)}",
        f"- cycle_outcome.jsonl: {len(cycle_events)}",
        f"- background_invocation.jsonl: {len(bg_events)}",
        f"- tool_call.jsonl: {len(tool_call_events)}",
        f"- verdict.jsonl: {len(verdict_events)}",
        "",
    ]
    lines += section_retrieval_latency(retrieval_events)
    lines += section_query_distribution(retrieval_events)
    lines += section_signal_contribution(retrieval_events)
    lines += section_cache_potential(retrieval_events)
    lines += section_per_query_class_latency(retrieval_events)
    lines += section_cycle_outcome_correlation(retrieval_events, cycle_events,
                                                  tool_call_events)
    lines += section_background_tools(bg_events)

    report = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(report)
        print(f"Wrote {output_path}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", help="Write report to this path (else stdout)")
    args = ap.parse_args()
    sys.exit(main(args.output))
