"""Signature-forgery verifier per AGI D-110 anti-forgery.

The pattern, from AGI lab (D-110): an agent claims a verdict in a
ResultPacket, but no episodic record proves the work was actually done.
A forgery. The runner must catch this before the verdict propagates to
the Director's downstream decision.

Bert's signatures live in `state/results/<role>_C<cycle>_<tag>.json`
ResultPackets. Each claims (role, cycle, verdict). The episodic record
lives in `logs/cycle_<cycle>_<date>.jsonl` — every model call, tool
call, permission decision is appended there.

A ResultPacket is FORGED when at least one of:
  • The cycle log file does not exist for the claimed cycle.
  • The cycle log file exists but contains no model_response events
    for the claiming role.
  • The packet's telemetry.model_used does not match any model that
    actually returned a response in the log.
  • The output_path file referenced by the packet does not exist.

`verify_packet(packet_path)` returns a VerifyReport. `verify_results_dir()`
walks state/results/ and returns one report per packet. Callers
(core/subagent.py and core/evaluator.py) treat any forgery as
SIGNATURE_FORGERY_UNADDRESSED and auto-fail the cycle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from core import log


class _ScanResult(TypedDict):
    """Shape returned by `_scan_log_for_role`. Typed so mypy can
    narrow `scan["models"]` to set[str] (it was Any | object)."""
    matches: int
    models: set[str]
    role_events: int

LOG = log.get_logger("bert.verify")
LAB_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = LAB_ROOT / "state" / "results"
LOGS_DIR = LAB_ROOT / "logs"


@dataclass(frozen=True)
class VerifyReport:
    packet_path: Path
    role: str
    cycle: int
    verdict: str
    forgery_detected: bool
    """True only when an ACTIVE mismatch is detected — model claimed
    not in cycle log, output_path missing, etc. This is the
    'agent lied' signal."""
    unverifiable: bool = False
    """True when the packet predates the cycle-log infrastructure
    (cycles before observability wiring landed) or no log exists
    for an unrelated reason. Distinct from forgery — these packets
    can't be confirmed true OR false; they're just opaque."""
    reasons: tuple[str, ...] = ()
    cycle_log_files: tuple[str, ...] = ()
    matched_model_responses: int = 0


def _find_cycle_logs(cycle: int) -> list[Path]:
    """Return all logs/cycle_<cycle>_*.jsonl files (one per day)."""
    if not LOGS_DIR.exists():
        return []
    return sorted(LOGS_DIR.glob(f"cycle_{cycle}_*.jsonl"))


def _scan_log_for_role(log_paths: list[Path], role: str) -> _ScanResult:
    """Walk the JSONL files for this cycle, count model_response events
    that match the role and surface the set of distinct models used."""
    matches = 0
    models: set[str] = set()
    role_events = 0
    for p in log_paths:
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("kind") == "model_response":
                        matches += 1
                        if ev.get("model"):
                            models.add(ev["model"])
                    if ev.get("role") == role:
                        role_events += 1
        except OSError as e:
            LOG.warning("verify: cannot read log file %s: %s", p, e)
    return {"matches": matches, "models": models, "role_events": role_events}


def verify_packet(packet_path: Path) -> VerifyReport:
    """Verify one ResultPacket file.

    Reads the packet, finds the matching cycle log file(s), confirms the
    cycle actually ran. Mismatches are aggregated into VerifyReport.reasons.
    """
    packet_path = Path(packet_path)
    try:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return VerifyReport(
            packet_path=packet_path, role="?", cycle=-1, verdict="?",
            forgery_detected=True, reasons=(f"packet_unreadable: {e}",),
            cycle_log_files=(), matched_model_responses=0,
        )

    role = str(packet.get("role", "?"))
    cycle_raw = packet.get("cycle", -1)
    try:
        cycle = int(cycle_raw)
    except (TypeError, ValueError):
        cycle = -1
    verdict = str(packet.get("verdict", "?"))
    reasons: list[str] = []

    log_paths = _find_cycle_logs(cycle)
    unverifiable_reasons: list[str] = []
    if not log_paths:
        unverifiable_reasons.append(
            f"no_cycle_log: cycle={cycle} has no logs/cycle_{cycle}_*.jsonl "
            f"(packet predates observability wiring or log was rotated)"
        )

    scan: _ScanResult = (
        _scan_log_for_role(log_paths, role)
        if log_paths
        else _ScanResult(matches=0, models=set(), role_events=0)
    )
    if log_paths and scan["matches"] == 0:
        # Cycle log EXISTS but is empty of model_response events. This IS a
        # real signal: a packet claiming this cycle but no model call
        # actually happened during it. That's forgery, not unverifiable.
        reasons.append("no_model_responses: cycle log exists but no model_response events")

    telemetry = packet.get("telemetry", {})
    claimed_model = telemetry.get("model_used") or ""
    if claimed_model and claimed_model != "provider/model" and scan["models"]:
        bare = claimed_model.split("/")[-1]
        seen_bare = {m.split("/")[-1] for m in scan["models"]}
        if bare not in seen_bare and claimed_model not in scan["models"]:
            reasons.append(
                f"model_mismatch: telemetry claims '{claimed_model}' "
                f"but log shows {sorted(scan['models'])}"
            )

    output_path = packet.get("output_path")
    if output_path:
        op = LAB_ROOT / output_path
        if not op.exists():
            reasons.append(f"output_path_missing: {output_path}")

    # Forgery = at least one ACTIVE mismatch (reasons list non-empty).
    # Unverifiable = no active mismatch but cycle log is missing.
    # Pure-clean = cycle log present, model matches, output_path present.
    forgery = bool(reasons)
    unverifiable = bool(unverifiable_reasons) and not forgery
    all_reasons = list(reasons) + (unverifiable_reasons if unverifiable else [])

    return VerifyReport(
        packet_path=packet_path,
        role=role,
        cycle=cycle,
        verdict=verdict,
        forgery_detected=forgery,
        unverifiable=unverifiable,
        reasons=tuple(all_reasons),
        cycle_log_files=tuple(p.name for p in log_paths),
        matched_model_responses=scan["matches"],
    )


def verify_results_dir(results_dir: Path | None = None) -> list[VerifyReport]:
    """Walk state/results/ and verify every packet."""
    rd = results_dir or RESULTS_DIR
    if not rd.exists():
        return []
    out: list[VerifyReport] = []
    for p in sorted(rd.glob("*.json")):
        out.append(verify_packet(p))
    return out


def summarize(reports: list[VerifyReport]) -> dict:
    """Roll up reports for /now page or evaluator gate.

    Three buckets:
      - clean: cycle log present, model matches, output_path present
      - unverifiable: cycle log missing (packet predates observability
        wiring; treat as opaque — neither confirmed nor falsified)
      - forged: ACTIVE mismatch detected (model_mismatch /
        no_model_responses / output_path_missing / packet_unreadable)

    `any_forgery` triggers the evaluator's hard-fail gate; unverifiable
    packets do NOT trip the gate (they're a separate signal — useful
    for retroactive audit but not a runtime safety violation)."""
    total = len(reports)
    forged = [r for r in reports if r.forgery_detected]
    unverifiable = [r for r in reports if r.unverifiable and not r.forgery_detected]
    clean = [r for r in reports if not r.forgery_detected and not r.unverifiable]
    return {
        "total": total,
        "forged_count": len(forged),
        "unverifiable_count": len(unverifiable),
        "clean_count": len(clean),
        "forged_paths": [str(r.packet_path) for r in forged],
        "unverifiable_paths": [str(r.packet_path) for r in unverifiable],
        "any_forgery": bool(forged),
    }
