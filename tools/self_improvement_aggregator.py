"""Passive cross-lab self-improvement signal aggregator.

Self-improvement is a PROPERTY of the engine, not a mission. This tool
walks every lab's observability events, computes rolling stats and
drift signals, and emits a single `self_improvement_signal.jsonl` for
the PI to review.

It spends NO tokens. No LLM calls. ~ms per event. The PI sees the
aggregated picture and decides which engine-level changes ship.

Inputs (walked by mtime):
  - ~/.bert/labs/<lab>/state/observability/cycle_outcome.jsonl  (multi-lab future)
  - state/observability/cycle_outcome.jsonl                     (single-lab today)
  - state/observability/verdict.jsonl
  - state/observability/retrieval.jsonl  (+ archived)
  - state/observability/artifact_accepted.jsonl
  - state/observability/concern_raised.jsonl + addressed
  - state/observability/circuit_breaker_event.jsonl

Outputs:
  state/observability/self_improvement_signal.jsonl
    Each line is one signal: {ts, lab, signal_type, severity, payload}

Signal types this round:
  - cycle_success_drop      success rate dropped >15pp vs 7-cycle rolling avg
  - latency_rise            p99 elapsed_secs > 2× p50
  - artifact_zero_streak    N consecutive cycles with artifacts_accepted=0
  - verdict_concentration   one verdict type >70% of last N
  - retrieval_failure_spike circuit_breaker on retrieval > threshold
  - concern_open_growth     more concerns raised than addressed in window
  - cross_mission_drift     (future, when multi-lab) one mission type degrading

Run:
  .venv/bin/python tools/self_improvement_aggregator.py
  .venv/bin/python tools/self_improvement_aggregator.py --window 10 --since 24h
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
OBS_DIR = LAB_ROOT / "state" / "observability"
OUTPUT = OBS_DIR / "self_improvement_signal.jsonl"


def _read_jsonl(name: str) -> list[dict]:
    """Live JSONL plus any rotated archives."""
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


def _emit_signal(signal_type: str, severity: str, payload: dict, lab: str = "lab") -> dict:
    return {
        "ts": dt.datetime.now(dt.UTC).isoformat(),
        "lab": lab,
        "signal_type": signal_type,
        "severity": severity,
        "payload": payload,
    }


def signal_cycle_success_drop(cycles: list[dict], window: int) -> list[dict]:
    """Compare success rate of last `window` to prior `window`."""
    real = [c for c in cycles if c.get("elapsed_secs")]
    if len(real) < 2 * window:
        return []
    recent = real[-window:]
    prior = real[-2*window:-window]
    sr_recent = sum(1 for c in recent if c.get("success")) / len(recent)
    sr_prior = sum(1 for c in prior if c.get("success")) / len(prior)
    drop = sr_prior - sr_recent
    if drop > 0.15:
        return [_emit_signal(
            "cycle_success_drop",
            "high" if drop > 0.30 else "medium",
            {"sr_recent": round(sr_recent, 3), "sr_prior": round(sr_prior, 3),
             "drop_pp": round(drop * 100, 1), "window": window},
        )]
    return []


def signal_latency_rise(cycles: list[dict], window: int) -> list[dict]:
    """Flag when p99 > 2x p50 in recent window."""
    real = [c for c in cycles if c.get("elapsed_secs")][-window:]
    if len(real) < 10:
        return []
    el = sorted(c["elapsed_secs"] for c in real)
    p50 = el[len(el) // 2]
    p99 = el[min(int(len(el) * 0.99), len(el) - 1)]
    if p99 > 2 * p50 and p50 > 0:
        return [_emit_signal(
            "latency_rise",
            "medium" if p99 < 5 * p50 else "high",
            {"p50_secs": round(p50, 1), "p99_secs": round(p99, 1),
             "ratio": round(p99 / p50, 2), "window": window},
        )]
    return []


def signal_artifact_zero_streak(cycles: list[dict]) -> list[dict]:
    """Cycles that succeed but produce no artifacts — the bug we hit today."""
    real = [c for c in cycles if c.get("elapsed_secs")]
    if not real:
        return []
    streak = 0
    for c in reversed(real):
        if c.get("artifacts_accepted", 0) == 0:
            streak += 1
        else:
            break
    if streak >= 5:
        return [_emit_signal(
            "artifact_zero_streak",
            "high" if streak >= 10 else "medium",
            {"streak": streak,
             "implication": "lab is not landing artifacts; "
                            "verification/decode/prompt may be broken"},
        )]
    return []


def signal_verdict_concentration(cycles: list[dict], window: int) -> list[dict]:
    """One verdict type dominating the recent window."""
    real = [c for c in cycles if c.get("elapsed_secs")][-window:]
    if len(real) < 5:
        return []
    verdicts = Counter()
    total = 0
    for c in real:
        for v in c.get("verdicts", []):
            verdicts[v] += 1
            total += 1
    if not total:
        return []
    top_verdict, top_n = verdicts.most_common(1)[0]
    ratio = top_n / total
    if ratio > 0.70 and top_verdict in {"BUILD_FAIL", "OTHER", "REJECT"}:
        return [_emit_signal(
            "verdict_concentration",
            "high" if ratio > 0.90 else "medium",
            {"dominant_verdict": top_verdict, "ratio": round(ratio, 2),
             "window": window, "implication":
                "consistent failure mode — investigate prompt/verification gate"},
        )]
    return []


def signal_retrieval_failure_spike(rets: list[dict], cb_events: list[dict],
                                    window_hours: float) -> list[dict]:
    """Circuit-breaker hits on the retrieval path in the last N hours."""
    cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(hours=window_hours)
    recent_cb = []
    for e in cb_events:
        try:
            ts = _parse_ts(e["ts"])
        except (KeyError, ValueError):
            continue
        if ts >= cutoff and "retriev" in str(e.get("source", "")).lower():
            recent_cb.append(e)
    if len(recent_cb) >= 3:
        return [_emit_signal(
            "retrieval_failure_spike",
            "high" if len(recent_cb) >= 10 else "medium",
            {"events_in_window": len(recent_cb), "window_hours": window_hours},
        )]
    return []


def signal_concern_open_growth(raised: list[dict], addressed: list[dict],
                                window_hours: float) -> list[dict]:
    cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(hours=window_hours)
    def _within(events):
        out = 0
        for e in events:
            try:
                if _parse_ts(e["ts"]) >= cutoff:
                    out += 1
            except (KeyError, ValueError):
                continue
        return out
    n_raised = _within(raised)
    n_addressed = _within(addressed)
    if n_raised - n_addressed >= 5:
        return [_emit_signal(
            "concern_open_growth",
            "medium",
            {"raised": n_raised, "addressed": n_addressed,
             "open_delta": n_raised - n_addressed,
             "window_hours": window_hours},
        )]
    return []


# ── Sprint 1 commit 6: artifact_accepted correlator ─────────────────


def signal_acceptance_rate_drop(accepted_events: list[dict],
                                  window: int) -> list[dict]:
    """Detect window-over-window drop in user-acceptance rate.

    Reads `artifact_accepted.jsonl` (user's accept/reject signal — the
    ground truth for whether bert is producing useful work). Drops
    ≥10pp surface as MEDIUM, ≥25pp as HIGH.
    """
    real = [a for a in accepted_events if a.get("ts")]
    if len(real) < 2 * window:
        return []
    recent = real[-window:]
    prior = real[-2*window:-window]
    sr_recent = sum(1 for a in recent if a.get("acceptance_kind") == "accept") / max(len(recent), 1)
    sr_prior = sum(1 for a in prior if a.get("acceptance_kind") == "accept") / max(len(prior), 1)
    drop = sr_prior - sr_recent
    if drop > 0.10:
        return [_emit_signal(
            "acceptance_rate_drop",
            "high" if drop > 0.25 else "medium",
            {"sr_recent": round(sr_recent, 3),
             "sr_prior": round(sr_prior, 3),
             "drop_pp": round(drop * 100, 1),
             "window": window,
             "implication": "user-visible quality has degraded; check "
                            "recent model/prompt/skill changes"},
        )]
    return []


def signal_per_role_acceptance(accepted_events: list[dict]) -> list[dict]:
    """Per-role acceptance rate. If any role's acceptance < 40% over
    ≥10 attempts, flag for prompt/skill review.

    Closes launch criterion 35 (per-role acceptance surfaced).
    """
    by_role: dict[str, list[dict]] = {}
    for a in accepted_events:
        role = a.get("role")
        if role:
            by_role.setdefault(role, []).append(a)
    out: list[dict] = []
    for role, evs in by_role.items():
        if len(evs) < 10:
            continue
        accept_count = sum(1 for e in evs if e.get("acceptance_kind") == "accept")
        rate = accept_count / len(evs)
        if rate < 0.40:
            out.append(_emit_signal(
                "per_role_acceptance_low",
                "high" if rate < 0.20 else "medium",
                {"role": role,
                 "acceptance_rate": round(rate, 3),
                 "n": len(evs),
                 "implication": f"{role} outputs are getting rejected; "
                                f"prompt or skill plan needs revision"},
            ))
    return out


def signal_per_model_acceptance(accepted_events: list[dict],
                                 cycles: list[dict]) -> list[dict]:
    """Per-(provider, model) acceptance rate. Joins artifact_accepted
    with cycle_outcome (which records model_used in dispatches).

    Closes launch criterion 35 (per-model acceptance surfaced).
    """
    # Map cycle_id → model_used from cycle_outcome dispatches
    cycle_to_models: dict[int, list[str]] = {}
    for c in cycles:
        if c.get("cycle_id") is None:
            continue
        # Cycle outcomes nest model in dispatches; older format may not have it
        models = []
        for d in c.get("dispatches", []) if isinstance(c.get("dispatches"), list) else []:
            telem = (d.get("telemetry") or {}) if isinstance(d, dict) else {}
            mdl = telem.get("model_used") or d.get("model_used") if isinstance(d, dict) else None
            if mdl:
                models.append(mdl)
        if models:
            cycle_to_models[c["cycle_id"]] = models
    # Aggregate acceptance per model
    by_model: dict[str, list[dict]] = {}
    for a in accepted_events:
        cid = a.get("source_cycle") or a.get("cycle_id")
        if not isinstance(cid, int):
            continue
        for mdl in cycle_to_models.get(cid, []):
            by_model.setdefault(mdl, []).append(a)
    out: list[dict] = []
    for model, evs in by_model.items():
        if len(evs) < 10:
            continue
        accept_count = sum(1 for e in evs if e.get("acceptance_kind") == "accept")
        rate = accept_count / len(evs)
        if rate < 0.40:
            out.append(_emit_signal(
                "per_model_acceptance_low",
                "high" if rate < 0.20 else "medium",
                {"model": model,
                 "acceptance_rate": round(rate, 3),
                 "n": len(evs),
                 "implication": f"artifacts produced by {model} get rejected "
                                f"{round((1-rate)*100)}% of the time; "
                                f"consider tier-up or model swap"},
            ))
    return out


def main(window: int, window_hours: float, dry_run: bool) -> int:
    cycles = _read_jsonl("cycle_outcome.jsonl")
    rets = _read_jsonl("retrieval.jsonl")
    cb = _read_jsonl("circuit_breaker_event.jsonl")
    raised = _read_jsonl("concern_raised.jsonl")
    addressed = _read_jsonl("concern_addressed.jsonl")
    accepted = _read_jsonl("artifact_accepted.jsonl")

    print("=== Self-improvement aggregator ===")
    print(f"  cycles read: {len(cycles)}")
    print(f"  retrievals: {len(rets)}")
    print(f"  circuit_breaker events: {len(cb)}")
    print(f"  concerns raised/addressed: {len(raised)}/{len(addressed)}")
    print(f"  artifact_accepted events: {len(accepted)}")
    print(f"  window: {window} cycles  |  window_hours: {window_hours}")
    print()

    signals: list[dict] = []
    signals += signal_cycle_success_drop(cycles, window)
    signals += signal_latency_rise(cycles, window)
    signals += signal_artifact_zero_streak(cycles)
    signals += signal_verdict_concentration(cycles, window)
    signals += signal_retrieval_failure_spike(rets, cb, window_hours)
    signals += signal_concern_open_growth(raised, addressed, window_hours)
    # Sprint 1 commit 6: artifact_accepted correlator signals
    signals += signal_acceptance_rate_drop(accepted, window)
    signals += signal_per_role_acceptance(accepted)
    signals += signal_per_model_acceptance(accepted, cycles)

    print(f"=== Signals detected: {len(signals)} ===")
    for s in signals:
        print(f"  [{s['severity']:>6}] {s['signal_type']}: {s['payload']}")
    print()

    if signals and not dry_run:
        OBS_DIR.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("a", encoding="utf-8") as f:
            for s in signals:
                f.write(json.dumps(s, separators=(",", ":")) + "\n")
        print(f"  wrote {len(signals)} → {OUTPUT.relative_to(LAB_ROOT)}")

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=10,
                    help="rolling cycle window for trend signals")
    ap.add_argument("--window-hours", type=float, default=24.0,
                    help="hours window for event-spike signals")
    ap.add_argument("--dry-run", action="store_true",
                    help="print signals but don't append to JSONL")
    args = ap.parse_args()
    sys.exit(main(args.window, args.window_hours, args.dry_run))
