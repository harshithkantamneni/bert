"""Sprint 3 c3: Feature registry (core/feature_registry.py).

Snapshot-based loader mirroring skill_registry: discover + parse every
feature under a directory, cache, expose get/all/all_names/snapshot, and
validate_all against the available skill set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import feature_registry  # noqa: E402


def _feature_text(name: str, skills: list[str]) -> str:
    plan = "\n".join(f"  - {s}" for s in skills)
    return f"""---
name: {name}
display_name: "{name.title()}"
short_description: "test feature {name}"
mission_template: |
  Do {name} on {{{{topic}}}}.
roster_override: null
skill_plan:
{plan}
quality_contract:
  correctness: 5
  completeness: 4
  provenance: 4
  defensibility: 4
  usability: 3
  honesty: 5
  reproducibility: 3
  efficiency: 2
  pass_threshold: 0.70
fitness_command: "true"
output_shape: "findings/{name}.md"
estimated_cost_usd: 0.04
estimated_time_minutes: 6
mcp_tool_signature:
  name: bert.{name}
  description: "{name}"
  returns_schema: {name}_result.v1.json
---
# {name}
"""


@pytest.fixture
def feat_dir(tmp_path):
    d = tmp_path / "features"
    d.mkdir()
    (d / "alpha.md").write_text(_feature_text("alpha", ["web_search_and_dedup", "gap_finder"]))
    (d / "beta.md").write_text(_feature_text("beta", ["finalize_project"]))
    return d


def test_load_all_returns_all_features(feat_dir):
    feats = feature_registry.load_all(features_dir=feat_dir, force_reload=True)
    assert {f.name for f in feats} == {"alpha", "beta"}


def test_get_returns_feature(feat_dir):
    feature_registry.load_all(features_dir=feat_dir, force_reload=True)
    assert feature_registry.get("alpha").name == "alpha"
    assert feature_registry.get("nonexistent") is None


def test_all_names(feat_dir):
    feature_registry.load_all(features_dir=feat_dir, force_reload=True)
    assert feature_registry.all_names() == {"alpha", "beta"}


def test_all_returns_list(feat_dir):
    feature_registry.load_all(features_dir=feat_dir, force_reload=True)
    names = sorted(f.name for f in feature_registry.all())
    assert names == ["alpha", "beta"]


def test_snapshot_is_independent_copy(feat_dir):
    feature_registry.load_all(features_dir=feat_dir, force_reload=True)
    snap = feature_registry.snapshot()
    snap.clear()
    # clearing the snapshot must not empty the live registry
    assert feature_registry.get("alpha") is not None


def test_validate_all_flags_unresolved_skills(feat_dir):
    feature_registry.load_all(features_dir=feat_dir, force_reload=True)
    # available skills missing 'gap_finder' (used by alpha)
    problems = feature_registry.validate_all(
        available_skills={"web_search_and_dedup", "finalize_project"},
    )
    assert "alpha" in problems
    assert any("gap_finder" in e for e in problems["alpha"])
    assert "beta" not in problems  # beta's only skill resolves


def test_validate_all_clean_when_skills_present(feat_dir):
    feature_registry.load_all(features_dir=feat_dir, force_reload=True)
    problems = feature_registry.validate_all(
        available_skills={"web_search_and_dedup", "gap_finder", "finalize_project"},
    )
    assert problems == {}


def test_malformed_file_skipped_not_fatal(feat_dir):
    (feat_dir / "broken.md").write_text("no frontmatter here\n")
    feats = feature_registry.load_all(features_dir=feat_dir, force_reload=True)
    # the 2 good ones still load; broken is skipped
    assert {f.name for f in feats} == {"alpha", "beta"}
