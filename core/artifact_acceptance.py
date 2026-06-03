"""artifact_accepted event class — anchors the §9 north-star metric.

Per the platform strategy report §9, the product's north-star metric is
"accepted artifacts per lab-week." This module makes that metric
load-bearing by emitting an explicit `artifact_accepted` event each
time bert produces an artifact that meets the acceptance bar — either
via PI blessing or via a non-caveat APPROVE verdict on a shippable
role.

Two acceptance paths:

1. PI blessing (`acceptance_kind="pi_blessing"`):
   The /api/bless/{decision_id} endpoint emits this when the human PI
   explicitly accepts a finding/decision. Strongest signal — equivalent
   to "user accepts, uses, publishes, ships, or decides from" the §9
   doc's definition.

2. Verdict APPROVE on a shippable role (`acceptance_kind="verdict_approve"`):
   When a sub-agent dispatch in a SHIPPABLE_ROLE returns verdict=APPROVE
   (NOT APPROVE_WITH_CAVEATS — those still have open concerns), the
   produced artifact is auto-accepted. The lab is allowed to ship its
   own work when the verdict is clean.

SHIPPABLE_ROLES are roles whose outputs are terminal artifacts:
researcher / strategist / implementer / evaluator / reflector /
consolidator / clearness_phase2. NON-shippable roles like
threshing_pass and clearness_phase1 always emit SCOPE_STOP, not a
shippable verdict — so they're excluded from auto-acceptance.

Aggregation + grade:

  count_accepted_in_window(window_secs)
  acceptance_rate_in_window(window_secs)
  grade(window_secs) → "A" | "A-" | "B" | "C" | "INSUFFICIENT_DATA"

Grade rubric (§9 north-star metric):
  - A:                 ≥80% acceptance rate AND ≥5 accepted artifacts in window
  - A-:                60-79% rate AND ≥3 accepted
  - B:                 40-59% rate AND ≥1 accepted
  - C:                 <40% rate OR 0 accepted (with shippable activity)
  - INSUFFICIENT_DATA: no shippable verdicts in window
"""

from __future__ import annotations

import time
from pathlib import Path

from core import observability

LAB_ROOT = Path(__file__).resolve().parent.parent
OBS_DIR = LAB_ROOT / "state" / "observability"

# Roles whose APPROVE verdict counts as a shippable artifact.
SHIPPABLE_ROLES = frozenset({
    "researcher", "strategist", "implementer", "evaluator",
    "reflector", "consolidator", "clearness_phase2",
})

# Acceptance kinds emitted to artifact_accepted.jsonl.
KIND_PI_BLESSING = "pi_blessing"
KIND_VERDICT_APPROVE = "verdict_approve"
KIND_VERDICT_AWC = "verdict_approve_with_caveats"  # auto-accepted w/ reservation

# Artifact types — taxonomy of what bert produces.
TYPE_FINDING = "finding"
TYPE_CODE = "code"
TYPE_DECISION = "decision"
TYPE_REPORT = "report"
TYPE_PROOF_PACKET = "proof_packet"
TYPE_OTHER = "other"

# Role → artifact_type heuristic for verdict-path emissions.
_ROLE_TYPE = {
    "researcher": TYPE_REPORT,
    "strategist": TYPE_FINDING,
    "implementer": TYPE_CODE,
    "evaluator": TYPE_DECISION,
    "reflector": TYPE_FINDING,
    "consolidator": TYPE_FINDING,
    "clearness_phase2": TYPE_DECISION,
}


def emit_artifact_accepted(
    *,
    artifact_id: str,
    source_dispatch_id: str | None,
    cycle: int,
    acceptance_kind: str,
    artifact_type: str = TYPE_OTHER,
    rationale: str | None = None,
    role: str | None = None,
) -> None:
    """Append an artifact_accepted event to the observability log.

    Idempotent at the event-stream level — the same artifact may emit
    multiple events (e.g., verdict_approve then later pi_blessing); the
    aggregator dedupes by artifact_id when computing acceptance_rate.
    """
    if acceptance_kind not in (KIND_PI_BLESSING, KIND_VERDICT_APPROVE, KIND_VERDICT_AWC):
        raise ValueError(f"unknown acceptance_kind: {acceptance_kind!r}")
    observability.emit("artifact_accepted", {
        "artifact_id": artifact_id,
        "source_dispatch_id": source_dispatch_id,
        "cycle": cycle,
        "acceptance_kind": acceptance_kind,
        "artifact_type": artifact_type,
        "role": role,
        "rationale": (rationale or "")[:240],
    })


def _read_events_in_window(event_class: str, window_secs: int) -> list[dict]:
    """Read recent events of `event_class`, newest first by mtime/log
    order (the log is append-only so file order IS chronological)."""
    path = OBS_DIR / f"{event_class}.jsonl"
    if not path.exists():
        return []
    import json
    cutoff_ts = time.time() - window_secs
    out: list[dict] = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Events have an ISO timestamp `ts` — parse to unix.
            ts_str = ev.get("ts")
            if not ts_str:
                continue
            try:
                from datetime import datetime
                ev_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
            except (ValueError, AttributeError):
                continue
            if ev_ts >= cutoff_ts:
                out.append(ev)
    except OSError:
        return []
    return out


def count_accepted_in_window(window_secs: int = 7 * 86400) -> dict:
    """Count distinct accepted artifacts in the window.

    De-duplicates by artifact_id (so a verdict_approve followed by a
    later pi_blessing on the same artifact counts once). Returns a
    dict with totals + breakdowns by kind / artifact_type / role.
    """
    events = _read_events_in_window("artifact_accepted", window_secs)
    seen: dict[str, dict] = {}
    for ev in events:
        aid = ev.get("artifact_id")
        if not aid or aid in seen:
            continue
        seen[aid] = ev
    by_kind: dict[str, int] = {}
    by_type: dict[str, int] = {}
    by_role: dict[str, int] = {}
    for ev in seen.values():
        by_kind[ev.get("acceptance_kind", "?")] = by_kind.get(ev.get("acceptance_kind", "?"), 0) + 1
        by_type[ev.get("artifact_type", "?")] = by_type.get(ev.get("artifact_type", "?"), 0) + 1
        role = ev.get("role") or "?"
        by_role[role] = by_role.get(role, 0) + 1
    return {
        "window_secs": window_secs,
        "total": len(seen),
        "by_kind": by_kind,
        "by_type": by_type,
        "by_role": by_role,
    }


def acceptance_rate_in_window(window_secs: int = 7 * 86400) -> dict:
    """Compute acceptance rate = accepted / shippable-verdicts-emitted.

    Denominator: verdict events on SHIPPABLE_ROLES in the window
    (excludes infrastructure roles like threshing_pass that always
    SCOPE_STOP). A high acceptance rate means bert's shippable work
    survives review; low means lots of REJECTs or unblessed work.
    """
    accepted = count_accepted_in_window(window_secs)
    accepted_n = accepted["total"]
    # Count shippable verdicts as denominator.
    verdicts = _read_events_in_window("verdict", window_secs)
    shippable_verdicts = [
        v for v in verdicts if v.get("role") in SHIPPABLE_ROLES
    ]
    denom = len(shippable_verdicts)
    rate = (accepted_n / denom) if denom > 0 else 0.0
    return {
        "window_secs": window_secs,
        "accepted_n": accepted_n,
        "shippable_verdicts_n": denom,
        "acceptance_rate": round(rate, 3),
        "by_kind": accepted["by_kind"],
        "by_type": accepted["by_type"],
        "by_role": accepted["by_role"],
    }


def grade(window_secs: int = 7 * 86400) -> dict:
    """Grade the lab's acceptance signal per the §9 rubric.

    Returns dict with `letter`, `reason`, plus the underlying rate data.
    """
    r = acceptance_rate_in_window(window_secs)
    rate = r["acceptance_rate"]
    accepted_n = r["accepted_n"]
    shippable_n = r["shippable_verdicts_n"]
    if shippable_n == 0:
        return {"letter": "INSUFFICIENT_DATA",
                "reason": f"no shippable verdicts in last {window_secs // 86400}d",
                **r}
    if rate >= 0.80 and accepted_n >= 5:
        letter, reason = "A", f"{int(rate*100)}% acceptance, {accepted_n} accepted"
    elif rate >= 0.60 and accepted_n >= 3:
        letter, reason = "A-", f"{int(rate*100)}% acceptance, {accepted_n} accepted"
    elif rate >= 0.40 and accepted_n >= 1:
        letter, reason = "B", f"{int(rate*100)}% acceptance, {accepted_n} accepted"
    else:
        letter, reason = "C", f"{int(rate*100)}% acceptance, {accepted_n} accepted (rubric floor: 40% + ≥1)"
    return {"letter": letter, "reason": reason, **r}
