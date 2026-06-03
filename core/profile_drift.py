"""Mission profile drift detection + within-shape reshape.

Phase C6 of the v3 plan (locked-in answer L-11: within-shape reshape
only in v1; cross-shape DEFERRED to v1.1).

drift_score(profile, recent_cycles) → float ∈ [0, 1]
  Compares the lab's INITIAL profile against the ACTUAL signature of
  recent cycles. High drift means the lab is behaving differently than
  it was initially classified — e.g., a profile.primary_work='monitor'
  lab whose recent cycles all produced `decide`-shaped outcomes.

propose_reshape(lab_path) → MissionProfile | None
  Returns a proposed updated profile based on what the lab has actually
  been doing. Director-side; surfaces via needs_user_input.

within_shape_reshape(lab_path, new_profile)
  Apply a same-data_shape reshape: re-run schema_synthesizer + scaffold
  missing knowledge files + update lab.yaml. NO data migration needed
  because data_shape stays the same.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger("bert.profile_drift")


@dataclass
class DriftReport:
    score: float                      # 0..1, higher = more drift
    cycles_inspected: int
    actual_work_signature: dict       # what the lab was actually doing
    declared_work_signature: dict     # what the profile said
    recommendation: str               # human-readable hint to director
    proposed_changes: dict            # field-level diff (suggested updates)


# ── Drift signature extraction ────────────────────────────────────


def _read_recent_events(lab_path: Path, *, n_cycles: int = 5) -> list[dict]:
    """Read recent events for the last N completed cycles."""
    ev_path = lab_path / "sor" / "events.jsonl"
    if not ev_path.exists():
        return []
    try:
        lines = ev_path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    events: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    # Find the latest cycle number; keep only events from last n_cycles
    cycle_ids = sorted({int(e.get("cycle", 0)) for e in events
                         if isinstance(e.get("cycle"), int)})
    if not cycle_ids:
        return events
    target_cycles = set(cycle_ids[-n_cycles:])
    return [e for e in events if int(e.get("cycle", 0)) in target_cycles]


def _actual_work_signature(events: list[dict]) -> dict:
    """Aggregate observed cycle_shapes + verdicts + roles into a signature."""
    cycle_shapes = Counter()
    verdicts = Counter()
    roles = Counter()
    for e in events:
        ec = e.get("event_class") or ""
        if ec == "director_decision":
            shape = e.get("cycle_shape", "")
            if shape:
                cycle_shapes[shape] += 1
        if ec == "verdict":
            v = e.get("verdict", "")
            if v:
                verdicts[v] += 1
        if ec == "subagent_spawn":
            role = e.get("role", "")
            if role:
                roles[role] += 1
    return {
        "cycle_shapes": dict(cycle_shapes),
        "verdicts": dict(verdicts),
        "roles": dict(roles),
        "total_events": len(events),
    }


def _profile_work_signature(profile: dict) -> dict:
    """What the profile's `primary_work` + `work_types` imply about
    expected cycle_shapes."""
    primary = profile.get("primary_work", "discover")
    work_types = profile.get("work_types", [])
    # Approximate mapping work_type → expected cycle_shapes
    expected_shapes_map = {
        "discover":   ["research-deeper", "synthesis"],
        "monitor":    ["research-deeper"],
        "synthesize": ["synthesis"],
        "decide":     ["strategy-refine", "synthesis"],
        "compare":    ["research-deeper", "strategy-refine"],
        "build":      ["strategy-refine", "verification-tighten"],
        "audit":      ["verification-tighten"],
        "refute":     ["verification-tighten"],
    }
    expected_shapes: list[str] = []
    for wt in [primary, *work_types]:
        expected_shapes.extend(expected_shapes_map.get(wt, []))
    return {
        "expected_shapes": expected_shapes,
        "primary_work": primary,
        "work_types": work_types,
    }


# ── Drift score ───────────────────────────────────────────────────


def drift_score(
    lab_path: Path, *, recent_n_cycles: int = 5,
) -> DriftReport:
    """Compute drift between profile and actual recent cycle activity.

    Returns DriftReport with:
      score      0 (perfect alignment) .. 1 (no alignment)
      cycles_inspected
      actual_work_signature   what the lab was doing
      declared_work_signature what the profile said
      recommendation          natural-language hint
      proposed_changes        field-level diff
    """
    # Load profile from lab.yaml
    profile = _load_profile(lab_path)
    if not profile:
        return DriftReport(
            score=0.0, cycles_inspected=0,
            actual_work_signature={}, declared_work_signature={},
            recommendation="No mission_profile in lab.yaml; nothing to compare.",
            proposed_changes={},
        )

    events = _read_recent_events(lab_path, n_cycles=recent_n_cycles)
    actual = _actual_work_signature(events)
    declared = _profile_work_signature(profile)
    cycles_inspected = len({int(e.get("cycle", 0)) for e in events
                              if isinstance(e.get("cycle"), int)})

    if cycles_inspected < 3:
        # Too few cycles to judge drift; return neutral
        return DriftReport(
            score=0.0, cycles_inspected=cycles_inspected,
            actual_work_signature=actual, declared_work_signature=declared,
            recommendation=f"Only {cycles_inspected} recent cycles; need ≥3 for drift score.",
            proposed_changes={},
        )

    # Score = fraction of observed cycle_shapes NOT in expected_shapes
    observed = actual.get("cycle_shapes", {})
    total_observed = sum(observed.values()) or 1
    expected = set(declared.get("expected_shapes", []))
    mismatched = sum(c for shape, c in observed.items() if shape not in expected)
    raw_score = mismatched / total_observed

    # Build natural-language recommendation
    recommendation, proposed = _build_recommendation(
        profile=profile, observed=observed, expected=expected,
        score=raw_score,
    )
    return DriftReport(
        score=round(raw_score, 3),
        cycles_inspected=cycles_inspected,
        actual_work_signature=actual,
        declared_work_signature=declared,
        recommendation=recommendation,
        proposed_changes=proposed,
    )


def _build_recommendation(
    *, profile: dict, observed: dict, expected: set,
    score: float,
) -> tuple[str, dict]:
    """Translate drift into a director-readable hint + proposed field changes."""
    if score < 0.3:
        return ("Lab is operating in line with declared profile.", {})

    # What's the dominant observed shape that's NOT in expected?
    surprising = {s: c for s, c in observed.items() if s not in expected}
    if not surprising:
        return ("Profile and recent activity are aligned.", {})

    dominant = max(surprising, key=surprising.get)
    # Map cycle_shape → suggested primary_work change
    shape_to_work = {
        "strategy-refine":       "decide",
        "verification-tighten":  "audit",
        "research-deeper":       "discover",
        "synthesis":             "synthesize",
        "idle":                  None,
    }
    suggested_work = shape_to_work.get(dominant)
    proposed: dict = {}
    if suggested_work and suggested_work != profile.get("primary_work"):
        proposed["primary_work"] = suggested_work
        # Keep data_shape; we're within-shape reshape only (v1)
        proposed["work_types"] = sorted({
            *(profile.get("work_types") or []),
            suggested_work,
        })

    if proposed:
        return (
            f"Lab profile says primary_work={profile.get('primary_work')!r} "
            f"but recent cycles dominantly produced cycle_shape={dominant!r}. "
            f"Propose updating primary_work → {suggested_work!r} "
            f"(same data_shape; within-shape reshape).",
            proposed,
        )
    return (
        f"Drift detected (score={score:.2f}) but no clean reshape "
        f"recommendation. Director should review.",
        {},
    )


# ── Reshape (within-shape only in v1) ─────────────────────────────


def within_shape_reshape(
    lab_path: Path,
    new_profile_dict: dict,
) -> dict:
    """Apply a within-shape reshape:
      1. Verify new_profile.data_shape == old_profile.data_shape (else
         this is a cross-shape change which v1 does not support)
      2. Re-run schema_synthesizer on new profile
      3. Scaffold any new knowledge files (existing ones preserved)
      4. Update lab.yaml with the new profile + schema

    Returns dict with:
      ok:                 bool
      from_profile_summary:
      to_profile_summary:
      knowledge_files_added: list of new files scaffolded
      error:              str (only if ok=False)
    """
    import yaml

    yaml_path = lab_path / "lab.yaml"
    if not yaml_path.exists():
        return {"ok": False, "error": "lab.yaml not found"}
    try:
        cfg = yaml.safe_load(yaml_path.read_text()) or {}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"lab.yaml parse failed: {e}"}

    old_profile = cfg.get("mission_profile") or {}
    old_shape = old_profile.get("data_shape", "")
    new_shape = new_profile_dict.get("data_shape", old_shape)
    if new_shape != old_shape:
        return {
            "ok": False,
            "error": (
                f"Cross-shape reshape (old={old_shape!r}, "
                f"new={new_shape!r}) is deferred to v1.1. To change "
                f"shapes, archive the lab + create a new one."
            ),
        }

    # Build a MissionProfile-like object from the merged dict
    merged = {**old_profile, **new_profile_dict}
    try:
        from core import mission_profile as mp
        from core import schema_synthesizer as ss
        # Use the same shape; just construct a profile dataclass instance
        # from the merged dict so synthesize() gets the right type
        profile = mp.MissionProfile(
            domain=str(merged.get("domain", "general")),
            domain_confidence=float(merged.get("domain_confidence", 0.5)),
            work_types=tuple(merged.get("work_types") or ("discover",)),
            primary_work=str(merged.get("primary_work", "discover")),
            horizon=str(merged.get("horizon", "short")),
            cadence=merged.get("cadence"),
            output_kind=str(merged.get("output_kind", "report")),
            rigor=str(merged.get("rigor", "cited")),
            data_shape=str(merged.get("data_shape", "document_corpus")),
            expected_volume=str(merged.get("expected_volume", "medium")),
            input_surfaces=tuple(merged.get("input_surfaces") or ()),
            audience=str(merged.get("audience", "self")),
            success_criteria=tuple(merged.get("success_criteria") or ()),
            classifier_confidence=float(merged.get("classifier_confidence", 0.5)),
            stage_used=str(merged.get("stage_used", "reshape")),
        )
        schema = ss.synthesize(profile)
        added_files = ss.scaffold_knowledge_files(lab_path, schema)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"reshape failed: {type(e).__name__}: {e}"}

    # Write updated lab.yaml — preserve existing top-level keys
    cfg["mission_profile"] = profile.to_dict()
    cfg["lab_schema"] = {
        "rule_id": schema.rule_id,
        "profile_id": schema.profile_id,
        "roster_core": list(schema.roster_core),
        "roster_initial": list(schema.roster_initial),
        "memory_adapters": list(schema.memory_adapters),
        "knowledge_files": list(schema.knowledge_files),
        "graph_schema": schema.graph_schema,
        "workflow": schema.workflow,
        "output_format": schema.output_format,
    }
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))

    return {
        "ok": True,
        "from_profile_summary": (
            f"{old_profile.get('domain', '?')} / "
            f"{old_profile.get('primary_work', '?')} / "
            f"{old_profile.get('horizon', '?')}"
        ),
        "to_profile_summary": (
            f"{profile.domain} / {profile.primary_work} / {profile.horizon}"
        ),
        "knowledge_files_added": [p.name for p in added_files],
        "new_workflow": schema.workflow,
        "data_shape_preserved": old_shape,
    }


# ── Helpers ────────────────────────────────────────────────────────


def _load_profile(lab_path: Path) -> dict:
    yaml_path = lab_path / "lab.yaml"
    if not yaml_path.exists():
        return {}
    try:
        import yaml
        return (yaml.safe_load(yaml_path.read_text()) or {}).get(
            "mission_profile") or {}
    except Exception:  # noqa: BLE001
        return {}


# ── CLI ────────────────────────────────────────────────────────────


def _cli(argv: list[str]) -> int:
    """python -m core.profile_drift score <lab>
    python -m core.profile_drift reshape <lab> <field>=<value> ...
    """
    import sys
    if len(argv) < 3:
        print("usage: profile_drift score|reshape ...", file=sys.stderr)
        return 2
    cmd = argv[1]
    lab = Path(argv[2]).expanduser()
    if cmd == "score":
        report = drift_score(lab)
        from dataclasses import asdict
        print(json.dumps(asdict(report), indent=2))
        return 0
    if cmd == "reshape":
        # Parse key=value args
        updates: dict = {}
        for kv in argv[3:]:
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            # Try to parse JSON for complex values
            try:
                updates[k] = json.loads(v)
            except json.JSONDecodeError:
                updates[k] = v
        result = within_shape_reshape(lab, updates)
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 1
    print(f"unknown cmd: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(_cli(sys.argv))
