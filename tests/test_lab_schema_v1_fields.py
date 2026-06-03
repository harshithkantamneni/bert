"""Sprint 3 c5: LabSchema extended with v1.0 fields (spec §1.1).

Adds skill_plan, quality_contract, fitness_command, output_path_pattern,
estimated_cost_usd, estimated_time_minutes, classifier_confidence. Must
be default-safe: old persisted schemas (without these fields) still
deserialize, and existing synthesis still works.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import lab_schema_io, mission_profile, quality, schema_synthesizer  # noqa: E402


def test_labschema_has_v1_fields_with_safe_defaults():
    # Construct the OLD way (only the original required fields) — the new
    # fields must default so this doesn't raise.
    s = schema_synthesizer.LabSchema(
        profile_id="x", rule_id="default",
        roster_core=("director",), roster_initial=("researcher",),
        memory_adapters=(), knowledge_files=(), graph_schema="g",
        workflow="research_iterate", output_format="report",
    )
    assert s.skill_plan == ()
    assert s.quality_contract is None
    assert s.fitness_command is None
    assert s.output_path_pattern == ""
    assert s.estimated_cost_usd is None
    assert s.estimated_time_minutes is None
    assert s.classifier_confidence == pytest.approx(0.0)


def test_synthesize_populates_classifier_confidence_from_profile():
    profile = mission_profile.default_profile("Audit findings for stale claims")
    schema = schema_synthesizer.synthesize(profile)
    # classifier_confidence is carried from the profile onto the schema
    assert 0.0 <= schema.classifier_confidence <= 1.0


def test_to_dict_includes_v1_fields():
    qc = quality.QualityContract(**dict.fromkeys(quality.DIMENSIONS, 3))
    s = schema_synthesizer.LabSchema(
        profile_id="x", rule_id="default",
        roster_core=("director",), roster_initial=("writer",),
        memory_adapters=(), knowledge_files=(), graph_schema="g",
        workflow="research_iterate", output_format="report",
        skill_plan=("gap_finder", "finalize_project"),
        quality_contract=qc, fitness_command="true",
        output_path_pattern="findings/x.md",
        estimated_cost_usd=0.05, estimated_time_minutes=8,
        classifier_confidence=0.82,
    )
    d = s.to_dict()
    assert d["skill_plan"] == ("gap_finder", "finalize_project")
    assert d["quality_contract"]["correctness"] == 3
    assert d["classifier_confidence"] == pytest.approx(0.82)


def test_from_dict_old_schema_backward_compatible():
    # An OLD persisted schema (no v1.0 fields) must still load.
    old = {
        "profile_id": "x", "rule_id": "default",
        "roster_core": ["director"], "roster_initial": ["researcher"],
        "memory_adapters": [], "knowledge_files": [], "graph_schema": "g",
        "workflow": "research_iterate", "output_format": "report",
        "routing_overrides": {},
    }
    s = lab_schema_io._from_dict(old)
    assert s.skill_plan == ()
    assert s.quality_contract is None
    assert s.classifier_confidence == pytest.approx(0.0)


def test_from_dict_new_schema_rehydrates_quality_contract():
    qc = quality.QualityContract(**dict.fromkeys(quality.DIMENSIONS, 4))
    s = schema_synthesizer.LabSchema(
        profile_id="x", rule_id="default",
        roster_core=("director",), roster_initial=("writer",),
        memory_adapters=(), knowledge_files=(), graph_schema="g",
        workflow="w", output_format="report",
        skill_plan=("finalize_project",), quality_contract=qc,
        classifier_confidence=0.9,
    )
    # roundtrip through dict
    rehydrated = lab_schema_io._from_dict(s.to_dict())
    assert rehydrated.skill_plan == ("finalize_project",)
    assert isinstance(rehydrated.quality_contract, quality.QualityContract)
    assert rehydrated.quality_contract.correctness == 4
    assert rehydrated.classifier_confidence == pytest.approx(0.9)
