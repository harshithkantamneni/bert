"""Weekly self-measurement report (H.6) — the A/B-tier separator.

Per both May-2026 SOTA research agents' conclusion:
"The A/B separator is whether the system measures itself."

This single CLI produces the markdown digest covering every dimension
the production rubric grades on:

  1. Cross-family verdict agreement (from Inspect AI runs + verdict history)
  2. Skill curator gain (falsifier baseline drift week-over-week)
  3. Cache hit-similarity drift (semantic_cache stats)
  4. Token redundancy (LLMLingua compressed vs actual; see H.8 for the
     dispatch-level measurement; here we aggregate the snapshot)
  5. Memory tier budget compliance (core ≤2K)
  6. Falsifier baseline drift (delta vs last-week run)
  7. Idle compute pass count + avg duration
  8. MCP replay protection coverage

Output: findings/weekly_quality_report_<YYYY-MM-DD>.md

Usage:
  python tools/weekly_quality_report.py
  python tools/weekly_quality_report.py --window-days 7
  python tools/weekly_quality_report.py --json   # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

FINDINGS = LAB_ROOT / "findings"
WINDOW_SECS_DEFAULT = 7 * 86400


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe(fn, default: Any = None) -> Any:
    """Run fn; return default on any exception. Single-process,
    best-effort report — never let one section break the others."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "default": default}


# ── Section gatherers ────────────────────────────────────────────────


def section_cross_family_agreement(window_secs: int) -> dict:
    """Compute % of recent verdict events where the evaluator's family
    differed from the producer's family (P-VS-02 compliance)."""
    events_path = LAB_ROOT / "lab" / "sor" / "events.jsonl"
    if not events_path.exists():
        return {"compliance_pct": None, "n": 0, "note": "no events.jsonl"}
    import time
    cutoff = time.time() - window_secs
    total = 0
    compliant = 0
    family_pairs: list[tuple[str, str]] = []
    for line in events_path.read_text().splitlines()[-2000:]:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event_class") not in ("verdict", "dispatch_result"):
            continue
        ts_str = ev.get("ts", "")
        try:
            ev_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
        except (ValueError, AttributeError):
            continue
        if ev_ts < cutoff:
            continue
        judge = ev.get("judge_provider")
        producer_role = ev.get("agent")
        if not judge or not producer_role:
            continue
        # judge_provider is usually "provider/model" or just a model
        # name; the family signal is in the model substring.
        j_str = judge.lower()
        if "qwen" in j_str:
            j_family = "qwen"
        elif "mistral" in j_str:
            j_family = "mistral"
        elif "deepseek" in j_str:
            j_family = "deepseek"
        elif "gemini" in j_str:
            j_family = "gemini"
        elif "llama" in j_str or "meta" in j_str:
            j_family = "llama"
        elif "gpt" in j_str:
            j_family = "gpt"
        else:
            j_family = "unknown"
        total += 1
        # Producer family for bert's dominant cascade is llama. Cross-
        # family compliance = judge family ≠ llama.
        if j_family not in ("llama", "unknown"):
            compliant += 1
        family_pairs.append((producer_role or "unknown", j_family))
    pct = (100.0 * compliant / total) if total else None
    return {
        "compliance_pct": round(pct, 1) if pct is not None else None,
        "n": total,
        "compliant_n": compliant,
        "family_distribution": {f: sum(1 for _, x in family_pairs if x == f)
                                     for f in {x for _, x in family_pairs}},
    }


def section_skill_curator() -> dict:
    """Recent skills mined + drafted + promoted from creator.py."""
    drafts_dir = LAB_ROOT / "skills" / "draft"
    active_dir = LAB_ROOT / "skills" / "active"
    archived_dir = LAB_ROOT / "skills" / "archived"
    return {
        "drafts": len(list(drafts_dir.iterdir())) if drafts_dir.exists() else 0,
        "active": len(list(active_dir.iterdir())) if active_dir.exists() else 0,
        "archived": len(list(archived_dir.iterdir())) if archived_dir.exists() else 0,
    }


def section_cache_drift(window_secs: int) -> dict:
    """Mean similarity of cache hits over the window. Drift signal:
    if mean similarity declines week-over-week, embedding model is
    diverging from query distribution → re-index recommended."""
    from core import semantic_cache
    stats_list = semantic_cache.cache_stats(window_secs=window_secs)
    return {
        "per_role": [
            {
                "role": s.role,
                "rows": s.rows,
                "hits": s.hits_24h * (window_secs // 86400),  # approximate scale
                "misses": s.misses_24h * (window_secs // 86400),
                "hit_rate": s.hit_rate,
                "avg_similarity_on_hit": s.avg_similarity_on_hit,
            }
            for s in stats_list
        ],
        "note": "drift signal = avg_similarity declining week-over-week",
    }


def section_memory_tier_budget() -> dict:
    """Core tier compliance (≤2K tokens). Production rubric."""
    from core import memory_tiers
    return memory_tiers.core_budget_status()


def section_falsifier_baseline() -> dict:
    """Current state of the 14 pre-registered falsifier targets."""
    from tools import falsifier_baseline as fb
    results = fb.run_all(window=30)
    pass_n = sum(1 for r in results if r.status == fb.Status.PASS)
    fail_n = sum(1 for r in results if r.status == fb.Status.FAIL)
    insuff_n = sum(1 for r in results if r.status == fb.Status.INSUFFICIENT)
    return {
        "total": len(results),
        "pass": pass_n,
        "fail": fail_n,
        "insufficient": insuff_n,
        "per_target": [
            {
                "id": r.target_id, "name": r.name, "status": r.status.value,
                "value": r.current_value, "n": r.sample_size,
            }
            for r in results
        ],
    }


def section_idle_compute(window_secs: int) -> dict:
    from core import idle_compute
    return idle_compute.idle_stats(window_secs=window_secs)


def section_mcp_replay() -> dict:
    from core import mcp_replay
    return mcp_replay.stats()


def section_delegation() -> dict:
    """Director delegation discipline (F.9)."""
    from core import delegation
    return delegation.stats()


# ── Composer ─────────────────────────────────────────────────────────


def section_accepted_artifacts(window_secs: int) -> dict:
    """I.1 — §9 north-star metric: accepted artifacts per lab-week.

    Reads artifact_accepted.jsonl + cross-references shippable verdicts
    in the same window. Returns rate, count, and breakdown by kind / type.
    """
    from core import artifact_acceptance
    g = artifact_acceptance.grade(window_secs=window_secs)
    return {
        "accepted_n": g["accepted_n"],
        "shippable_verdicts_n": g["shippable_verdicts_n"],
        "acceptance_rate": g["acceptance_rate"],
        "by_kind": g["by_kind"],
        "by_type": g["by_type"],
        "by_role": g["by_role"],
        "letter": g["letter"],
        "reason": g["reason"],
    }


def gather_all(window_secs: int = WINDOW_SECS_DEFAULT) -> dict:
    return {
        "ts": _now_iso(),
        "window_secs": window_secs,
        "accepted_artifacts": _safe(
            lambda: section_accepted_artifacts(window_secs)),
        "cross_family_agreement": _safe(
            lambda: section_cross_family_agreement(window_secs)),
        "skill_curator": _safe(section_skill_curator),
        "cache_drift": _safe(lambda: section_cache_drift(window_secs)),
        "memory_tier_budget": _safe(section_memory_tier_budget),
        "falsifier_baseline": _safe(section_falsifier_baseline),
        "idle_compute": _safe(lambda: section_idle_compute(window_secs)),
        "mcp_replay": _safe(section_mcp_replay),
        "delegation": _safe(section_delegation),
    }


def grade(report: dict) -> dict:
    """Translate the report into A/B/C grades per the May-2026 rubric.

    Each dimension grades against the production threshold from the
    research agents' rubric tables.
    """
    grades: dict = {}

    # Cross-family compliance: A=≥90%, B=80-90, C=<80
    cf = report.get("cross_family_agreement", {})
    cf_pct = cf.get("compliance_pct") if isinstance(cf, dict) else None
    if cf_pct is None:
        grades["cross_family_agreement"] = "INSUFFICIENT_DATA"
    elif cf_pct >= 90:
        grades["cross_family_agreement"] = "A"
    elif cf_pct >= 80:
        grades["cross_family_agreement"] = "B"
    else:
        grades["cross_family_agreement"] = "C"

    # Memory tier: A=no overflow, B=<5 overflow, C=≥5 overflow
    mt = report.get("memory_tier_budget", {})
    overflow = mt.get("overflow_items", 0) if isinstance(mt, dict) else 0
    if overflow == 0:
        grades["memory_tier_budget"] = "A"
    elif overflow < 5:
        grades["memory_tier_budget"] = "B"
    else:
        grades["memory_tier_budget"] = "C"

    # Falsifier: A=all PASS or INSUFFICIENT, B=≤2 FAIL, C=≥3 FAIL
    fb = report.get("falsifier_baseline", {})
    fail_n = fb.get("fail", 0) if isinstance(fb, dict) else 0
    if fail_n == 0:
        grades["falsifier_baseline"] = "A"
    elif fail_n <= 2:
        grades["falsifier_baseline"] = "B"
    else:
        grades["falsifier_baseline"] = "C"

    # Idle compute: A=≥10 passes/24h, B=≥3, C=<3
    ic = report.get("idle_compute", {})
    passes = ic.get("passes_24h", 0) if isinstance(ic, dict) else 0
    if passes >= 10:
        grades["idle_compute"] = "A"
    elif passes >= 3:
        grades["idle_compute"] = "B"
    else:
        grades["idle_compute"] = "C"

    # Accepted artifacts (I.1, §9 north-star). artifact_acceptance.grade()
    # already computed the letter; pass it through so the grade rubric
    # lives in one place.
    aa = report.get("accepted_artifacts", {})
    grades["accepted_artifacts"] = (
        aa.get("letter") if isinstance(aa, dict) else "INSUFFICIENT_DATA"
    )

    return grades


def render_markdown(report: dict, grades: dict) -> str:
    lines: list[str] = []
    lines.append("# bert · weekly quality report")
    lines.append("")
    lines.append(f"**Generated:** {report['ts']}")
    lines.append(f"**Window:** last {report['window_secs'] // 86400} days")
    lines.append("")
    lines.append("## Scorecard")
    lines.append("")
    lines.append("| Dimension | Grade |")
    lines.append("|---|---|")
    for k, v in grades.items():
        lines.append(f"| {k.replace('_', ' ')} | **{v}** |")
    lines.append("")

    lines.append("## Cross-family verdict agreement")
    cf = report.get("cross_family_agreement", {})
    if isinstance(cf, dict):
        if cf.get("compliance_pct") is not None:
            lines.append(f"- compliance: **{cf['compliance_pct']}%** "
                          f"({cf.get('compliant_n', 0)}/{cf.get('n', 0)})")
            for f, n in (cf.get("family_distribution") or {}).items():
                lines.append(f"- judge family `{f}`: {n} dispatches")
        else:
            lines.append(f"- INSUFFICIENT_DATA: {cf.get('note', '')}")
    lines.append("")

    lines.append("## Skill curator")
    sk = report.get("skill_curator", {})
    if isinstance(sk, dict):
        lines.append(f"- drafts: {sk.get('drafts', 0)}")
        lines.append(f"- active: {sk.get('active', 0)}")
        lines.append(f"- archived: {sk.get('archived', 0)}")
    lines.append("")

    lines.append("## Cache hit-similarity drift")
    cd = report.get("cache_drift", {})
    per_role = cd.get("per_role") if isinstance(cd, dict) else []
    if per_role:
        lines.append("| role | rows | hit rate | avg sim |")
        lines.append("|---|---|---|---|")
        for r in per_role:
            lines.append(f"| {r['role']} | {r['rows']} | {r['hit_rate']} | "
                          f"{r['avg_similarity_on_hit']} |")
    else:
        lines.append("(no cache activity in window)")
    lines.append("")

    lines.append("## Memory tier budget")
    mt = report.get("memory_tier_budget", {})
    if isinstance(mt, dict):
        for k in ("token_budget", "token_total_unenforced", "items_total",
                   "overflow_items", "headroom_tokens", "headroom_pct"):
            lines.append(f"- {k}: {mt.get(k)}")
    lines.append("")

    lines.append("## Falsifier baseline (14 targets)")
    fb = report.get("falsifier_baseline", {})
    if isinstance(fb, dict):
        lines.append(f"- PASS: {fb.get('pass', 0)}")
        lines.append(f"- FAIL: {fb.get('fail', 0)}")
        lines.append(f"- INSUFFICIENT: {fb.get('insufficient', 0)}")
        for t in (fb.get("per_target") or []):
            status_emoji = {"PASS": "✓", "FAIL": "✗",
                             "INSUFFICIENT_DATA": "·"}.get(t["status"], "?")
            lines.append(f"  - {status_emoji} T{t['id']:02d} {t['name']}: "
                          f"{t['value']} (n={t['n']})")
    lines.append("")

    lines.append("## Idle compute")
    ic = report.get("idle_compute", {})
    if isinstance(ic, dict):
        for k in ("passes_24h", "avg_duration_ms", "max_duration_ms",
                   "ops_total_24h", "passes_with_errors"):
            lines.append(f"- {k}: {ic.get(k)}")
    lines.append("")

    lines.append("## MCP replay protection")
    mr = report.get("mcp_replay", {})
    if isinstance(mr, dict):
        for k in ("window_secs", "active_nonces"):
            lines.append(f"- {k}: {mr.get(k)}")
        if mr.get("by_tool"):
            lines.append("- by tool:")
            for tool, n in mr["by_tool"].items():
                lines.append(f"  - {tool}: {n}")
    lines.append("")

    lines.append("## Director delegation")
    dg = report.get("delegation", {})
    if isinstance(dg, dict) and dg:
        for role, stats in dg.items():
            if isinstance(stats, dict):
                lines.append(f"- {role}: ratio={stats.get('delegation_ratio')} "
                              f"(out={stats.get('delegations_out')}, "
                              f"self={stats.get('self_handled')})")
    else:
        lines.append("(no dispatch activity)")
    lines.append("")

    lines.append("---")
    lines.append("*Generated by `tools/weekly_quality_report.py`. "
                  "Run weekly via cron or manually.*")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="bert weekly quality report")
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--json", action="store_true",
                        help="emit JSON only; don't write the markdown file")
    parser.add_argument("--output-dir", default=str(FINDINGS),
                        help="markdown output directory")
    args = parser.parse_args()

    window_secs = args.window_days * 86400
    report = gather_all(window_secs=window_secs)
    grades = grade(report)
    report["grades"] = grades

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0

    md = render_markdown(report, grades)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    out_path = out_dir / f"weekly_quality_report_{today}.md"
    out_path.write_text(md)
    # Persist the JSON twin so programmatic consumers (bert
    # `/api/quality-report`, future automation) can read structured
    # data instead of re-parsing markdown. Same basename, .json suffix.
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {out_path}")
    print(f"wrote {json_path}")
    print()
    print("Grades:")
    for k, v in grades.items():
        print(f"  {k:30s} {v}")
    return 0


def _instrumented_main() -> int:
    """v3+ Phase 1d — emit a background_invocation event for this tool."""
    import time as _t
    try:
        from core import observability as _obs
    except Exception:  # noqa: BLE001
        _obs = None
    t0 = _t.monotonic()
    success = True
    try:
        rc = main()
        success = (rc == 0)
        return rc
    except SystemExit as e:
        success = (e.code == 0 if e.code is not None else True)
        raise
    except Exception:
        success = False
        raise
    finally:
        if _obs is not None:
            try:
                _obs.emit_background_invocation(
                    "weekly_quality_report",
                    args={"argv": sys.argv[1:]},
                    duration_ms=(_t.monotonic() - t0) * 1000,
                    success=success,
                )
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    sys.exit(_instrumented_main())
