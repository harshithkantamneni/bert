"""Sprint 3 c2: Feature DSL (core/feature_dsl.py, spec §1.5).

A Feature is a productized mission: typed parameters + a jinja2 mission
template + skill_plan + quality_contract + fitness + output shape +
mcp_tool_signature. Mirrors the skill DSL parse/validate pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import feature_dsl, quality  # noqa: E402


_MINIMAL = """---
name: literature_survey
display_name: "Literature Survey"
short_description: "Comparison table of papers on a topic."
long_description: |
  Surveys recent papers and produces a comparison table.
parameters:
  - name: topic
    type: string
    required: true
    placeholder: "Vector databases Q2 2026"
    validate:
      min_words: 3
      max_words: 20
      error_if_vague: true
  - name: num_papers
    type: int
    default: 10
    min: 3
    max: 30
    help: "Target paper count"
  - name: dimensions
    type: list[string]
    min_items: 3
    max_items: 12
    default: ["license", "latency", "recall@10"]
mission_template: |
  Survey papers on {{topic}}. Compare columns: {{dimensions | comma}}.
roster_override: null
skill_plan:
  - web_search_and_dedup
  - gap_finder
  - finalize_project
quality_contract:
  correctness: 5
  completeness: 4
  provenance: 5
  defensibility: 4
  usability: 3
  honesty: 5
  reproducibility: 3
  efficiency: 2
  pass_threshold: 0.70
fitness_command: |
  .venv/bin/python tools/eval/grade_artifact.py --artifact "{{output_path}}"
output_shape: "findings/{{topic_slug}}_survey.md"
estimated_cost_usd: 0.05
estimated_time_minutes: 8
typical_acceptance_rate: null
mcp_tool_signature:
  name: bert.literature_survey
  description: "Generate a comparison table of papers on a topic."
  returns_schema: literature_survey_result.v1.json
---
# Literature Survey

User-facing description here.
"""


def _write(tmp_path: Path, text: str, name: str = "feat.md") -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def test_parse_minimal_feature_core_fields(tmp_path):
    f = feature_dsl.parse_feature_file(_write(tmp_path, _MINIMAL))
    assert f.name == "literature_survey"
    assert f.display_name == "Literature Survey"
    assert "comparison table" in f.short_description.lower()
    assert "Surveys recent papers" in f.long_description
    assert f.estimated_cost_usd == pytest.approx(0.05)
    assert f.estimated_time_minutes == 8
    assert f.typical_acceptance_rate is None
    assert f.origin == "hand_authored"
    assert f.runs == 0
    assert "User-facing description" in f.body


def test_parse_parameters(tmp_path):
    f = feature_dsl.parse_feature_file(_write(tmp_path, _MINIMAL))
    params = {p.name: p for p in f.parameters}
    assert set(params) == {"topic", "num_papers", "dimensions"}
    assert params["topic"].type == "string"
    assert params["topic"].required is True
    assert params["topic"].constraint["min_words"] == 3
    assert params["num_papers"].type == "int"
    assert params["num_papers"].default == 10
    assert params["num_papers"].constraint["max"] == 30
    assert params["dimensions"].type == "list[string]"
    assert params["dimensions"].default == ["license", "latency", "recall@10"]


def test_parse_quality_contract_is_typed(tmp_path):
    f = feature_dsl.parse_feature_file(_write(tmp_path, _MINIMAL))
    assert isinstance(f.quality_contract, quality.QualityContract)
    assert f.quality_contract.provenance == 5
    assert f.quality_contract.pass_threshold == pytest.approx(0.70)


def test_parse_skill_plan_and_roster_override(tmp_path):
    f = feature_dsl.parse_feature_file(_write(tmp_path, _MINIMAL))
    assert f.skill_plan == ("web_search_and_dedup", "gap_finder", "finalize_project")
    assert f.roster_override is None  # null → None (let classifier decide)


def test_parse_mcp_tool_signature(tmp_path):
    f = feature_dsl.parse_feature_file(_write(tmp_path, _MINIMAL))
    assert f.mcp_tool_signature["name"] == "bert.literature_survey"
    assert f.mcp_tool_signature["returns_schema"] == "literature_survey_result.v1.json"


def test_missing_frontmatter_raises(tmp_path):
    with pytest.raises(feature_dsl.FeatureParseError):
        feature_dsl.parse_feature_file(_write(tmp_path, "# no frontmatter\n"))


def test_missing_required_field_raises(tmp_path):
    bad = _MINIMAL.replace("name: literature_survey\n", "")
    with pytest.raises(feature_dsl.FeatureParseError):
        feature_dsl.parse_feature_file(_write(tmp_path, bad))


def test_invalid_name_raises(tmp_path):
    bad = _MINIMAL.replace("name: literature_survey", "name: Literature Survey")
    with pytest.raises(feature_dsl.FeatureParseError):
        feature_dsl.parse_feature_file(_write(tmp_path, bad))


def test_validate_feature_flags_unresolved_skill(tmp_path):
    f = feature_dsl.parse_feature_file(_write(tmp_path, _MINIMAL))
    # available skills missing 'gap_finder' → should be flagged
    errs = feature_dsl.validate_feature(f, available_skills={"web_search_and_dedup", "finalize_project"})
    assert any("gap_finder" in e for e in errs)


def test_validate_feature_clean_when_all_skills_present(tmp_path):
    f = feature_dsl.parse_feature_file(_write(tmp_path, _MINIMAL))
    errs = feature_dsl.validate_feature(
        f, available_skills={"web_search_and_dedup", "gap_finder", "finalize_project"},
    )
    assert errs == []


def test_render_mission_template_substitutes_params(tmp_path):
    f = feature_dsl.parse_feature_file(_write(tmp_path, _MINIMAL))
    rendered = feature_dsl.render_mission(f, {
        "topic": "vector databases", "dimensions": ["license", "latency"],
    })
    assert "vector databases" in rendered
    assert "license, latency" in rendered  # comma filter joins the list
