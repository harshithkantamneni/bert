"""DispatchSpec + ResultPacket schema migration v1 → v2.

Per FINAL_implementation_plan_2026-05-07.md §5.2 H2 day 1.

v1 (pre-2026-05-07) schemas didn't include the Quaker pipeline fields
introduced by A6 (P-VS-06..09). v2 adds:

  DispatchSpec:
    + threshing_input_paths: array (P-VS-06)
    + clearness_phase1_output_path: string (P-VS-07 phase 2 input)

  ResultPacket:
    + caveats_embedded: array of ConcernEntry (P-VS-08)
    + clearness_queries: array of ClearnessQuery (P-VS-07 phase 1)
    + severity_grade: enum (P-VS-08)
    + judge_provider, position_swap_delta (P-VS-10)
    + cross-field invariants: threshing_pass role → SCOPE_STOP;
      clearness_phase1 → SCOPE_STOP + non-empty clearness_queries;
      APPROVE_WITH_CAVEATS → ≥1 caveats_embedded

Backward compatibility:
  - All v1 fields still present in v2; v1 packets validate cleanly under
    v2 schema (the new fields are optional)
  - The cross-field invariants only fire on v2-shape inputs (e.g.,
    role=threshing_pass); v1 packets without those role values pass
    through unchanged
  - schema_version: 2 added to v2 schemas as a top-level field; v1
    packets can be detected by absence of this field

This migration is forward-only; no v2→v1 downgrade. v1 archived
ResultPackets in state/results/ continue to load and validate under v2.

Usage:
  from schemas.migrations.v1_to_v2 import upgrade_dispatch_spec, upgrade_result_packet
  upgraded = upgrade_dispatch_spec(v1_dict)  # adds default empty fields
  validate(upgraded, v2_schema)              # passes
"""

from __future__ import annotations

from typing import Any


def upgrade_dispatch_spec(v1: dict[str, Any]) -> dict[str, Any]:
    """Convert a v1 DispatchSpec dict to v2 shape.

    Idempotent: calling on a v2 dict returns it unchanged. New optional
    fields are NOT auto-populated (they remain absent unless caller adds
    them); migration is mostly about validation passing.
    """
    out = dict(v1)
    # No required-field additions; v2 just adds optional fields.
    # threshing_input_paths and clearness_phase1_output_path are only
    # set when the dispatch is actually a Quaker pipeline role.
    return out


def upgrade_result_packet(v1: dict[str, Any]) -> dict[str, Any]:
    """Convert a v1 ResultPacket dict to v2 shape.

    The added optional fields stay absent for non-Quaker verdicts.
    For APPROVE_WITH_CAVEATS verdicts in v1 that had the legacy
    `caveats_blocking_downstream` array (string entries), we DO NOT
    auto-promote them to ConcernEntry shape — that's a semantic
    upgrade requiring caller knowledge of severity_grade.
    """
    out = dict(v1)
    # No required-field additions; v2 cross-field invariants only fire
    # on v2-shape inputs. v1 packets without role=threshing_pass etc.
    # pass through unchanged.
    return out


def detect_version(packet: dict[str, Any]) -> int:
    """Return 1 or 2 based on schema_version field or v2-distinctive fields."""
    if packet.get("schema_version") == 2:
        return 2
    # Heuristic: presence of v2-only fields
    v2_fields = ("threshing_input_paths", "clearness_phase1_output_path",
                 "clearness_queries", "severity_grade", "judge_provider",
                 "position_swap_delta")
    if any(f in packet for f in v2_fields):
        return 2
    return 1


def upgrade(packet: dict[str, Any]) -> dict[str, Any]:
    """Auto-detect packet type (DispatchSpec vs ResultPacket) and upgrade."""
    # ResultPacket has 'verdict'; DispatchSpec has 'success_criterion'
    if "verdict" in packet:
        return upgrade_result_packet(packet)
    if "success_criterion" in packet:
        return upgrade_dispatch_spec(packet)
    # Unknown — return as-is
    return packet
