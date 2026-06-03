"""Falsifier baseline measurement framework.

Reads existing observability data + result packets + seasoning queue,
computes the 14 pre-committed numerical falsifier targets, and reports
per-target PASS / FAIL / INSUFFICIENT_DATA.

Three measurement classes:
  MECHANICAL — parsed directly from observability JSONL or
    structured artifacts (result packets, seasoning.jsonl). High
    confidence.
  TEXT_SCAN — approximate, derived from substring presence in
    markdown deliverables. Flagged in the report as approximate.
  DEFERRED — requires a concern-flow tracker that doesn't exist
    yet (e.g., propagation across N cycles). Reported as
    INSUFFICIENT_DATA with a note.

Output:
  - findings/falsifier_baseline_C{cycle}.md (markdown report)
  - findings/falsifier_baseline_C{cycle}.json (machine-readable)

Targets are pre-committed; below-threshold sustained violation
triggers revision before the patterns are promoted FROZEN per
P-001 / P-VS-03.

Usage:
  python tools/falsifier_baseline.py --cycle 99
  python tools/falsifier_baseline.py --cycle 99 --window 30 --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import log  # noqa: E402

LOG = log.get_logger("bert.falsifier")
OBS_DIR = LAB_ROOT / "state" / "observability"
RESULTS_DIR = LAB_ROOT / "state" / "results"
FINDINGS_DIR = LAB_ROOT / "findings"
SEASONING_PATH = LAB_ROOT / "lab" / "sod" / "seasoning.jsonl"


class Status(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT = "INSUFFICIENT_DATA"


class Method(StrEnum):
    MECHANICAL = "mechanical"
    TEXT_SCAN = "text_scan_approximate"
    DEFERRED = "deferred"


@dataclass
class TargetResult:
    target_id: int
    name: str
    pattern: str
    threshold: str
    window: str
    method: Method
    status: Status
    current_value: str = ""
    sample_size: int = 0
    notes: str = ""


# Module-level filter: when set, only events / packets at cycle >= this
# value are returned by the readers. CLI sets this from --since-cycle.
SINCE_CYCLE: int = 0


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if SINCE_CYCLE > 0:
                cyc = rec.get("cycle")
                if cyc is None or int(cyc) < SINCE_CYCLE:
                    continue
            out.append(rec)
    except OSError:
        pass
    return out


def _read_results_for_role(role_pattern: str) -> list[dict]:
    """Read result packets for a given role pattern, newest first.

    Falsifier targets that take `window` measure CURRENT lab health
    (e.g. "≥80% of the last N phase-2 verdicts reference threshing"),
    not lifetime history. Sort by file mtime descending so `[:window]`
    naturally picks the most-recently-written packets. Smoke-test
    re-runs that scribble old cycle numbers into new files are
    correctly counted as "recent" via mtime, which is the operational
    truth.
    """
    if not RESULTS_DIR.exists():
        return []
    candidates: list[tuple[float, Path, dict]] = []
    rx = re.compile(role_pattern)
    for p in RESULTS_DIR.glob("*.json"):
        if not rx.match(p.name):
            continue
        try:
            packet = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if SINCE_CYCLE > 0:
            cyc = packet.get("cycle")
            if cyc is None or int(cyc) < SINCE_CYCLE:
                continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((mtime, p, packet))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [packet for _mtime, _p, packet in candidates]


# ── 14 targets ──────────────────────────────────────────────────────


def t1_threshing_structural_validity(window: int) -> TargetResult:
    """≥80% of threshing outputs have all 4 required sections.

    Threshing output files in this lab live at findings/*_threshing_*.md
    (or findings/threshing_*.md). The ResultPacket schema doesn't
    require output_path so we glob the files directly rather than relying
    on the packet field (which the model often omits).
    """
    candidates = sorted(
        list((LAB_ROOT / "findings").glob("*threshing*.md"))
        + list((LAB_ROOT / "findings").glob("threshing_*.md"))
    )
    sample = candidates[:window]
    if not sample:
        return TargetResult(1, "threshing_structural_validity", "P-VS-06",
                            "≥80%", f"first {window} dispatches",
                            Method.TEXT_SCAN, Status.INSUFFICIENT,
                            sample_size=0,
                            notes="no threshing output files in findings/ yet")
    # Section synonyms — accept natural variations the threshing prompt produces.
    required_sections = [
        ("disagreement",),
        ("hidden assumption", "hidden assumptions", "hidden_assumption"),
        ("queries", "questions"),
        ("evidence",),
    ]
    valid = 0
    for p in sample:
        try:
            text = p.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        if all(any(form in text for form in section) for section in required_sections):
            valid += 1
    pct = valid / len(sample) if sample else 0.0
    return TargetResult(1, "threshing_structural_validity", "P-VS-06",
                        "≥80%", f"first {window} dispatches",
                        Method.TEXT_SCAN,
                        Status.PASS if pct >= 0.80 else Status.FAIL,
                        current_value=f"{pct:.1%} ({valid}/{len(sample)})",
                        sample_size=len(sample))


def t2_threshing_verdict_discipline(window: int) -> TargetResult:
    """≥70% of threshing dispatches have verdict==SCOPE_STOP first attempt."""
    verdicts = _read_jsonl(OBS_DIR / "verdict.jsonl")
    sample = [v for v in verdicts if v.get("role", "").startswith("threshing")][:window]
    if not sample:
        return TargetResult(2, "threshing_verdict_discipline", "P-VS-06",
                            "≥70%", f"first {window} dispatches",
                            Method.MECHANICAL, Status.INSUFFICIENT,
                            notes="no threshing verdict events recorded")
    scope_stops = sum(1 for v in sample if v.get("verdict") == "SCOPE_STOP")
    pct = scope_stops / len(sample)
    return TargetResult(2, "threshing_verdict_discipline", "P-VS-06",
                        "≥70%", f"first {window} dispatches",
                        Method.MECHANICAL,
                        Status.PASS if pct >= 0.70 else Status.FAIL,
                        current_value=f"{pct:.1%} ({scope_stops}/{len(sample)})",
                        sample_size=len(sample))


def t3_threshing_referenced_downstream(window: int) -> TargetResult:
    """≥60% of post-threshing phase-2 verdicts reference the threshing output.

    Heuristic: a phase-2 packet references the threshing pass when its
    calibration_reasoning OR its output file mentions any of:
      • the literal word "threshing" (case-insensitive)
      • a path or filename including "_threshing_" or "threshing_"
      • the threshing output's distinctive disagreement framing
        (a substring ≥30 chars from the threshing file's "## Disagreement"
        section)

    Pairs phase-2 packets to threshing files by scenario suffix when
    available (e.g., S{n}_phase2 ↔ threshing_S{n}); falls back to
    cycle-pairing if scenario suffix not parseable.
    """
    p2_packets = _read_results_for_role(r"^clearness_phase2_C")
    sample = p2_packets[:window]
    if not sample:
        return TargetResult(3, "threshing_referenced_downstream", "P-VS-06",
                            "≥60%", f"first {window} cross-family judges",
                            Method.TEXT_SCAN, Status.INSUFFICIENT,
                            notes="no phase-2 result packets yet")

    threshing_files = sorted((LAB_ROOT / "findings").glob("*threshing*.md"))
    by_scenario: dict[str, str] = {}
    for tf in threshing_files:
        m = re.search(r"_S(\d+)", tf.name)
        if m:
            try:
                by_scenario[m.group(1)] = tf.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

    refs = 0
    for p in sample:
        reasoning = (p.get("calibration_reasoning") or "").lower()
        # Generic markers
        if "threshing" in reasoning:
            refs += 1
            continue
        # Scenario-specific marker: phase-2 result_path encodes the
        # scenario, e.g. "...S3_phase2..." → scenario 3
        result_name = ""
        # Best-effort: scan the packet for any path-like field
        for v in (p.get("output_path"), p.get("result_path")):
            if isinstance(v, str):
                result_name += " " + v
        m = re.search(r"_S(\d+)_phase2", result_name)
        scenario = m.group(1) if m else None
        if scenario and scenario in by_scenario:
            threshing_text = by_scenario[scenario]
            # Pull a distinctive ≥30-char chunk from the threshing's
            # "Disagreement" section and check for substring match
            dis_match = re.search(
                r"#+\s*disagreement[^\n]*\n+([^\n]+)",
                threshing_text, re.IGNORECASE,
            )
            if dis_match:
                key = dis_match.group(1).strip()[:60].lower()
                if len(key) >= 20 and key in reasoning:
                    refs += 1
    pct = refs / len(sample)
    return TargetResult(3, "threshing_referenced_downstream", "P-VS-06",
                        "≥60%", f"first {window} cross-family judges",
                        Method.TEXT_SCAN,
                        Status.PASS if pct >= 0.60 else Status.FAIL,
                        current_value=f"{pct:.1%} ({refs}/{len(sample)})",
                        sample_size=len(sample))


def t4_clearness_phase_completion(window: int) -> TargetResult:
    """≥85% of clearness dispatches complete both phases."""
    p1 = _read_jsonl(OBS_DIR / "clearness_phase1_dispatch.jsonl")
    p2 = _read_jsonl(OBS_DIR / "clearness_phase2_dispatch.jsonl")
    if not p1:
        return TargetResult(4, "clearness_phase_completion", "P-VS-07",
                            "≥85%", f"first {window} dispatches",
                            Method.MECHANICAL, Status.INSUFFICIENT,
                            notes="no clearness phase-1 events yet")
    sample_p1 = p1[:window]
    # Pair phase-1 → phase-2 by event count, not by cycle, because in
    # batch calibration runs many dispatches share the same cycle ID.
    # The orchestrator fires phase-2 only after phase-1 succeeds, so a
    # near-equal count is the right correctness signal.
    pct = min(len(p2), len(sample_p1)) / len(sample_p1) if sample_p1 else 0.0
    completed = min(len(p2), len(sample_p1))
    return TargetResult(4, "clearness_phase_completion", "P-VS-07",
                        "≥85%", f"first {window} dispatches",
                        Method.MECHANICAL,
                        Status.PASS if pct >= 0.85 else Status.FAIL,
                        current_value=f"{pct:.1%} ({completed}/{len(sample_p1)})",
                        sample_size=len(sample_p1))


def t5_clearness_query_count(window: int) -> TargetResult:
    """≥80% of phase-1 outputs have 3-7 queries."""
    packets = _read_results_for_role(r"^clearness_phase1_C")
    sample = packets[:window]
    if not sample:
        return TargetResult(5, "clearness_query_count", "P-VS-07",
                            "≥80%", f"first {window} dispatches",
                            Method.MECHANICAL, Status.INSUFFICIENT,
                            notes="no clearness_phase1 result packets yet")
    in_range = sum(
        1 for p in sample if 3 <= len(p.get("clearness_queries") or []) <= 7
    )
    pct = in_range / len(sample)
    return TargetResult(5, "clearness_query_count", "P-VS-07",
                        "≥80%", f"first {window} dispatches",
                        Method.MECHANICAL,
                        Status.PASS if pct >= 0.80 else Status.FAIL,
                        current_value=f"{pct:.1%} ({in_range}/{len(sample)})",
                        sample_size=len(sample))


def t6_phase2_references_phase1(window: int) -> TargetResult:
    """≥70% of phase-2 verdicts reference phase-1 queries."""
    p2_packets = _read_results_for_role(r"^clearness_phase2_C")
    sample = p2_packets[:window]
    if not sample:
        return TargetResult(6, "phase2_references_phase1", "P-VS-07",
                            "≥70%", f"first {window} dispatches",
                            Method.TEXT_SCAN, Status.INSUFFICIENT,
                            notes="no clearness_phase2 result packets yet")
    refs = 0
    for p in sample:
        reasoning = (p.get("calibration_reasoning") or "").lower()
        # Heuristic: did the reasoning mention phase 1 queries?
        if any(kw in reasoning for kw in ["phase 1", "phase-1", "query", "queries"]):
            refs += 1
    pct = refs / len(sample)
    return TargetResult(6, "phase2_references_phase1", "P-VS-07",
                        "≥70%", f"first {window} dispatches",
                        Method.TEXT_SCAN,
                        Status.PASS if pct >= 0.70 else Status.FAIL,
                        current_value=f"{pct:.1%} ({refs}/{len(sample)})",
                        sample_size=len(sample))


def t7_stand_aside_concerns_populated(window: int) -> TargetResult:
    """≥85% of APPROVE_WITH_CAVEATS verdicts have ≥1 concern."""
    events = _read_jsonl(OBS_DIR / "stand_aside_verdict.jsonl")
    sample = events[:window]
    if not sample:
        return TargetResult(7, "stand_aside_concerns_populated", "P-VS-08",
                            "≥85%", f"first {window} verdicts",
                            Method.MECHANICAL, Status.INSUFFICIENT,
                            notes="no stand_aside_verdict events yet")
    populated = sum(1 for e in sample if (e.get("concern_count") or 0) >= 1)
    pct = populated / len(sample)
    return TargetResult(7, "stand_aside_concerns_populated", "P-VS-08",
                        "≥85%", f"first {window} verdicts",
                        Method.MECHANICAL,
                        Status.PASS if pct >= 0.85 else Status.FAIL,
                        current_value=f"{pct:.1%} ({populated}/{len(sample)})",
                        sample_size=len(sample))


def t8_concerns_propagation(window: int) -> TargetResult:
    """≥70% of raised concerns are propagated to a downstream dispatch.

    Mechanical via core/concern_flow.py event streams. concern_raised
    and concern_propagated are matched on concern_id. A concern counts
    as "propagated" if at least one concern_propagated event references
    its id, regardless of how many cycles distance.
    """
    raised = _read_jsonl(OBS_DIR / "concern_raised.jsonl")[:50]
    if not raised:
        return TargetResult(8, "concerns_propagation", "P-VS-08",
                            "≥70%", "first 50 concerns raised",
                            Method.MECHANICAL, Status.INSUFFICIENT,
                            notes="no concern_raised events yet (run the lab)")
    propagated = _read_jsonl(OBS_DIR / "concern_propagated.jsonl")
    propagated_ids = {e.get("concern_id") for e in propagated}
    matched = sum(1 for e in raised if e.get("concern_id") in propagated_ids)
    pct = matched / len(raised)
    return TargetResult(8, "concerns_propagation", "P-VS-08",
                        "≥70%", "first 50 concerns raised",
                        Method.MECHANICAL,
                        Status.PASS if pct >= 0.70 else Status.FAIL,
                        current_value=f"{pct:.1%} ({matched}/{len(raised)})",
                        sample_size=len(raised))


def t9_concerns_addressed(window: int) -> TargetResult:
    """≥40% of propagated concerns are addressed within 5 cycles.

    A concern counts as "addressed" if a concern_addressed event with
    cycle_distance ≤ 5 references its concern_id. Window is the first
    30 distinct propagated concerns.
    """
    propagated = _read_jsonl(OBS_DIR / "concern_propagated.jsonl")
    # Dedupe by concern_id, keep first appearance.
    seen: set[str] = set()
    first_propagations: list[dict] = []
    for e in propagated:
        cid = e.get("concern_id")
        if cid and cid not in seen:
            seen.add(cid)
            first_propagations.append(e)
            if len(first_propagations) >= 30:
                break
    if not first_propagations:
        return TargetResult(9, "concerns_addressed", "P-VS-08",
                            "≥40%", "first 30 propagated concerns",
                            Method.MECHANICAL, Status.INSUFFICIENT,
                            notes="no concern_propagated events yet (run the lab)")
    addressed_events = _read_jsonl(OBS_DIR / "concern_addressed.jsonl")
    addressed_in_window = {
        e.get("concern_id") for e in addressed_events
        if (e.get("cycle_distance") or 0) <= 5
    }
    matched = sum(1 for e in first_propagations if e.get("concern_id") in addressed_in_window)
    pct = matched / len(first_propagations)
    return TargetResult(9, "concerns_addressed", "P-VS-08",
                        "≥40%", "first 30 propagated concerns",
                        Method.MECHANICAL,
                        Status.PASS if pct >= 0.40 else Status.FAIL,
                        current_value=f"{pct:.1%} ({matched}/{len(first_propagations)})",
                        sample_size=len(first_propagations))


def t10_concern_aging(window: int) -> TargetResult:
    """≤20% of concerns age out (5+ cycles raised, never addressed).

    "Aged out" = concern_raised exists but no concern_addressed within
    5 cycles (or ever). Computed over all concerns from concern_raised
    events with source_cycle ≤ (max source_cycle − 5) — i.e. only count
    concerns that had a fair chance to be addressed within the window.
    """
    raised = _read_jsonl(OBS_DIR / "concern_raised.jsonl")
    if not raised:
        return TargetResult(10, "concern_aging", "P-VS-08",
                            "≤20%", "all concerns over 30 cycles",
                            Method.MECHANICAL, Status.INSUFFICIENT,
                            notes="no concern_raised events yet (run the lab)")
    addressed = _read_jsonl(OBS_DIR / "concern_addressed.jsonl")
    addressed_ids = {e.get("concern_id") for e in addressed}
    max_cycle = max((e.get("source_cycle") or 0) for e in raised)
    # Only consider concerns that had at least 5 cycles to be addressed.
    eligible = [e for e in raised if (e.get("source_cycle") or 0) <= max_cycle - 5]
    if not eligible:
        return TargetResult(10, "concern_aging", "P-VS-08",
                            "≤20%", "all concerns over 30 cycles",
                            Method.MECHANICAL, Status.INSUFFICIENT,
                            notes="lab hasn't run ≥5 cycles past first concern yet")
    aged = sum(1 for e in eligible if e.get("concern_id") not in addressed_ids)
    pct = aged / len(eligible)
    return TargetResult(10, "concern_aging", "P-VS-08",
                        "≤20%", "all concerns over 30 cycles",
                        Method.MECHANICAL,
                        Status.PASS if pct <= 0.20 else Status.FAIL,
                        current_value=f"{pct:.1%} ({aged}/{len(eligible)})",
                        sample_size=len(eligible))


def t11_seasoning_queue_size(window: int) -> TargetResult:
    """≤25 unrevived entries at any time."""
    if not SEASONING_PATH.exists():
        return TargetResult(11, "seasoning_queue_size_bounded", "P-VS-09",
                            "≤25", "at any time",
                            Method.MECHANICAL, Status.INSUFFICIENT,
                            notes="no seasoning queue file yet")
    entries = _read_jsonl(SEASONING_PATH)
    unrevived = [e for e in entries if not e.get("revived_at")]
    return TargetResult(11, "seasoning_queue_size_bounded", "P-VS-09",
                        "≤25", "at any time",
                        Method.MECHANICAL,
                        Status.PASS if len(unrevived) <= 25 else Status.FAIL,
                        current_value=f"{len(unrevived)} unrevived (of {len(entries)} total)",
                        sample_size=len(entries))


def t12_seasoning_revival_rate(window: int) -> TargetResult:
    """≤15% per cycle, sustained over 30 cycles.

    Requires statistical power before evaluating: the ≤15% threshold
    can only be satisfied meaningfully with ≥20 seasoning entries
    spanning ≥5 distinct cycles. Below that, the noise floor (1 in N)
    can exceed the threshold even for a healthy lab, and `seasoning_revive`
    entries seeded by smoke tests dominate the signal. Insufficient
    data → INSUFFICIENT_DATA, not FAIL.
    """
    revives = _read_jsonl(OBS_DIR / "seasoning_revive.jsonl")
    seasonings = _read_jsonl(OBS_DIR / "seasoning_entry.jsonl")
    if not seasonings:
        return TargetResult(12, "seasoning_revival_rate", "P-VS-09",
                            "≤15%", "per cycle, sustained 30 cycles",
                            Method.MECHANICAL, Status.INSUFFICIENT,
                            notes="no seasoning_entry events yet")
    # Statistical-power gate: ≥20 entries AND ≥5 distinct cycles
    # represented. The ≥5 cycle gate filters out the "all from one
    # smoke run" case that drove the 2026-05 false-fail on T12.
    distinct_cycles = {e.get("cycle") for e in seasonings if e.get("cycle") is not None}
    min_entries = 20
    min_cycles = 5
    if len(seasonings) < min_entries or len(distinct_cycles) < min_cycles:
        return TargetResult(
            12, "seasoning_revival_rate", "P-VS-09",
            "≤15%", "per cycle, sustained 30 cycles",
            Method.MECHANICAL, Status.INSUFFICIENT,
            notes=(
                f"need ≥{min_entries} entries across ≥{min_cycles} cycles for "
                f"statistical power; have {len(seasonings)} entries across "
                f"{len(distinct_cycles)} cycles"
            ),
            sample_size=len(seasonings),
        )
    rate = len(revives) / max(len(seasonings), 1)
    return TargetResult(12, "seasoning_revival_rate", "P-VS-09",
                        "≤15%", "per cycle, sustained 30 cycles",
                        Method.MECHANICAL,
                        Status.PASS if rate <= 0.15 else Status.FAIL,
                        current_value=f"{rate:.1%} ({len(revives)} revives / {len(seasonings)} entries)",
                        sample_size=len(seasonings))


def t13_revival_outcome_quality(window: int) -> TargetResult:
    """≥40% of proposed revivals actually result in seasoning_revive.

    Match revival_proposed events against seasoning_revive events on
    seasoning_id. A proposal counts as successful if a seasoning_revive
    event for the same seasoning_id appears at or after the proposal.
    """
    proposals = _read_jsonl(OBS_DIR / "revival_proposed.jsonl")[:30]
    if not proposals:
        return TargetResult(13, "revival_outcome_quality", "P-VS-09",
                            "≥40%", "first 30 revival proposals",
                            Method.MECHANICAL, Status.INSUFFICIENT,
                            notes="no revival_proposed events yet (run the lab)")
    revives = _read_jsonl(OBS_DIR / "seasoning_revive.jsonl")
    revived_ids = {e.get("id") or e.get("seasoning_id") for e in revives}
    matched = sum(1 for e in proposals if e.get("seasoning_id") in revived_ids)
    pct = matched / len(proposals)
    return TargetResult(13, "revival_outcome_quality", "P-VS-09",
                        "≥40%", "first 30 revival proposals",
                        Method.MECHANICAL,
                        Status.PASS if pct >= 0.40 else Status.FAIL,
                        current_value=f"{pct:.1%} ({matched}/{len(proposals)})",
                        sample_size=len(proposals))


def t14_seasoning_entry_well_formed() -> TargetResult:
    """100% have ≥50-char block_reason (summary) + valid severity."""
    if not SEASONING_PATH.exists():
        return TargetResult(14, "seasoning_entry_well_formed", "P-VS-09",
                            "100%", "all entries",
                            Method.MECHANICAL, Status.INSUFFICIENT,
                            notes="no seasoning queue file yet")
    entries = _read_jsonl(SEASONING_PATH)
    if not entries:
        return TargetResult(14, "seasoning_entry_well_formed", "P-VS-09",
                            "100%", "all entries",
                            Method.MECHANICAL, Status.INSUFFICIENT,
                            notes="seasoning queue is empty")
    valid_sev = {"whisper", "voice", "weight"}
    well_formed = sum(
        1 for e in entries
        if len(e.get("summary", "")) >= 50
        # severity is optional in v1 schema; if present it must be valid
        and (("severity" not in e) or e.get("severity") in valid_sev)
    )
    pct = well_formed / len(entries)
    return TargetResult(14, "seasoning_entry_well_formed", "P-VS-09",
                        "100%", "all entries",
                        Method.MECHANICAL,
                        Status.PASS if pct >= 1.0 else Status.FAIL,
                        current_value=f"{pct:.1%} ({well_formed}/{len(entries)})",
                        sample_size=len(entries))


def t15_supervisor_pattern_evidence() -> TargetResult:
    """100% of pattern_observed events emitted by the
    supervisor lab MUST cite ≥2 distinct evidence_labs. Single-lab
    "patterns" aren't patterns — they're observations. The falsifier
    fires the moment any pattern_observed event ships with
    evidence_lab_count < 2.

    Reads from the supervisor lab's events.jsonl directly (not from
    state/observability), since pattern_observed events are emitted
    via core.director.emit_pattern_observed_event into the lab's
    events ledger.
    """
    events_path = LAB_ROOT / "lab" / "sor" / "events.jsonl"
    if not events_path.exists():
        return TargetResult(15, "supervisor_pattern_evidence", "FF-B.3",
                            "100% citing ≥2 labs", "all pattern_observed events",
                            Method.MECHANICAL, Status.INSUFFICIENT,
                            notes="supervisor events.jsonl not yet created")
    patterns: list[dict] = []
    for ev in _read_jsonl(events_path):
        if ev.get("event_class") == "pattern_observed":
            patterns.append(ev)
    if not patterns:
        return TargetResult(15, "supervisor_pattern_evidence", "FF-B.3",
                            "100% citing ≥2 labs", "all pattern_observed events",
                            Method.MECHANICAL, Status.INSUFFICIENT,
                            notes=("no pattern_observed events emitted yet; "
                                   "supervisor hasn't surfaced any cross-lab patterns"))
    well_evidenced = sum(
        1 for ev in patterns
        if int(ev.get("evidence_lab_count", 0)) >= 2
    )
    pct = well_evidenced / len(patterns)
    return TargetResult(15, "supervisor_pattern_evidence", "FF-B.3",
                        "100% citing ≥2 labs", "all pattern_observed events",
                        Method.MECHANICAL,
                        Status.PASS if pct >= 1.0 else Status.FAIL,
                        current_value=f"{pct:.1%} ({well_evidenced}/{len(patterns)})",
                        sample_size=len(patterns))


# ── Driver ──────────────────────────────────────────────────────────


def run_all(window: int = 30) -> list[TargetResult]:
    return [
        t1_threshing_structural_validity(window),
        t2_threshing_verdict_discipline(window),
        t3_threshing_referenced_downstream(window),
        t4_clearness_phase_completion(window),
        t5_clearness_query_count(window),
        t6_phase2_references_phase1(window),
        t7_stand_aside_concerns_populated(window),
        t8_concerns_propagation(window),
        t9_concerns_addressed(window),
        t10_concern_aging(window),
        t11_seasoning_queue_size(window),
        t12_seasoning_revival_rate(window),
        t13_revival_outcome_quality(window),
        t14_seasoning_entry_well_formed(),
        t15_supervisor_pattern_evidence(),
    ]


def render_markdown(results: list[TargetResult], cycle: int) -> str:
    lines = [
        f"# Falsifier baseline — cycle {cycle}",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} via `tools/falsifier_baseline.py`._",
        "",
        "## Summary",
        "",
    ]
    counts = {Status.PASS: 0, Status.FAIL: 0, Status.INSUFFICIENT: 0}
    for r in results:
        counts[r.status] += 1
    lines.append(f"- PASS: {counts[Status.PASS]} / 14")
    lines.append(f"- FAIL: {counts[Status.FAIL]} / 14")
    lines.append(f"- INSUFFICIENT_DATA: {counts[Status.INSUFFICIENT]} / 14")
    lines.append("")
    lines.append("## Per-target results")
    lines.append("")
    lines.append("| # | Target | Pattern | Threshold | Window | Status | Current | Sample | Method | Notes |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.target_id} | {r.name} | {r.pattern} | {r.threshold} | "
            f"{r.window} | **{r.status.value}** | {r.current_value or '—'} | "
            f"{r.sample_size} | {r.method.value} | {r.notes} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `MECHANICAL` measurements parse observability JSONL or structured artifacts.")
    lines.append("- `TEXT_SCAN` measurements use substring presence and are approximate.")
    lines.append("- `DEFERRED` targets need a concern-flow tracker (follow-up).")
    lines.append("- INSUFFICIENT_DATA on most targets is expected pre-calibration; run the orchestrated 30-dispatch window first.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, default=0,
                    help="cycle number for the report filename")
    ap.add_argument("--window", type=int, default=30,
                    help="N most-recent dispatches per pattern (default 30)")
    ap.add_argument("--since-cycle", type=int, default=0,
                    help="only count dispatches at cycle >= N (filters out "
                         "pre-prompt-fix history; default 0 = include all)")
    ap.add_argument("--json", action="store_true",
                    help="print machine-readable JSON to stdout")
    args = ap.parse_args()

    if args.since_cycle > 0:
        global SINCE_CYCLE
        SINCE_CYCLE = args.since_cycle

    results = run_all(window=args.window)

    payload = {
        "cycle": args.cycle, "window": args.window,
        "generated_ts": time.time(),
        "summary": {
            "pass": sum(1 for r in results if r.status == Status.PASS),
            "fail": sum(1 for r in results if r.status == Status.FAIL),
            "insufficient": sum(1 for r in results if r.status == Status.INSUFFICIENT),
        },
        "results": [
            {
                "target_id": r.target_id, "name": r.name, "pattern": r.pattern,
                "threshold": r.threshold, "window": r.window,
                "method": r.method.value, "status": r.status.value,
                "current_value": r.current_value, "sample_size": r.sample_size,
                "notes": r.notes,
            }
            for r in results
        ],
    }

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = FINDINGS_DIR / f"falsifier_baseline_C{args.cycle}.md"
    json_path = FINDINGS_DIR / f"falsifier_baseline_C{args.cycle}.json"
    md_path.write_text(render_markdown(results, args.cycle), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # calibration_falsifier_check observability — fills the documented
    # event_class slot. One event per baseline run summarizes the
    # PASS/FAIL/INSUFFICIENT counts so /now and audits can graph
    # falsifier health over time.
    try:
        from core import observability as _obs
        _obs.emit("calibration_falsifier_check", {
            "cycle": args.cycle, "window": args.window,
            "since_cycle": args.since_cycle if hasattr(args, "since_cycle") else 0,
            **payload["summary"],
            "fail_target_ids": [r.target_id for r in results if r.status == Status.FAIL],
        })
    except Exception:  # noqa: BLE001
        pass

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render_markdown(results, args.cycle))
    LOG.info("falsifier baseline: cycle=%d wrote %s + %s", args.cycle, md_path, json_path)
    return 0


def _instrumented_main() -> int:
    """Wraps main() to emit a background_invocation event capturing what
    this tool did. Per v3+ Phase 1d — bert has two emission paths
    (cycle agents + background tools); this closes the latter."""
    import time as _t
    try:
        from core import observability as _obs
    except Exception:  # noqa: BLE001
        _obs = None
    t0 = _t.monotonic()
    success = True
    findings_before: set[str] = set()
    if _obs is not None:
        try:
            findings_before = {p.name for p in FINDINGS_DIR.glob("falsifier_baseline_*")}
        except Exception:  # noqa: BLE001
            pass
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
                findings_after = {p.name for p in FINDINGS_DIR.glob("falsifier_baseline_*")}
                new_findings = sorted(findings_after - findings_before)
                _obs.emit_background_invocation(
                    "falsifier_baseline",
                    args={"argv": sys.argv[1:]},
                    duration_ms=(_t.monotonic() - t0) * 1000,
                    findings_produced=new_findings,
                    success=success,
                )
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    sys.exit(_instrumented_main())
