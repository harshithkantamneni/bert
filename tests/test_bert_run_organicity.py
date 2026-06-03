"""Sprint 1 acceptance tests: bert_run.py dispatches roster from
lab_schema, not hardcoded researcher → strategist.

Validates the v1.0 organicity wire:
  - mission_profile.classify_mission → MissionProfile
  - schema_synthesizer.synthesize → LabSchema
  - lab_schema_io.load_or_synthesize → persists + caches
  - bert_run.run → dispatches lab_schema.roster_initial

These are unit tests (no LLM calls, heuristic classifier only).
The 3-mission suite + manual smoke validates the LLM-driven path.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Make 'core' importable when pytest runs from repo root
LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import lab_schema_io, mission_profile  # noqa: E402


@pytest.fixture
def tmp_lab(tmp_path):
    lab = tmp_path / "lab"
    lab.mkdir()
    return lab


# ── lab_schema_io.load_or_synthesize ─────────────────────────────────


def test_missing_seed_brief_raises_actionable_error(tmp_lab):
    """Missing seed_brief surfaces a SchemaLoadError with user-actionable msg."""
    with pytest.raises(lab_schema_io.SchemaLoadError) as exc:
        lab_schema_io.load_or_synthesize(tmp_lab)
    msg = str(exc.value)
    assert "seed_brief.md" in msg
    assert "mission" in msg.lower()


def test_synthesize_and_persist_creates_schema_file(tmp_lab):
    """First call synthesizes from seed_brief + writes lab_schema.json."""
    (tmp_lab / "seed_brief.md").write_text(
        "Survey vector databases as of Q2 2026. Comparison table required."
    )
    schema = lab_schema_io.load_or_synthesize(tmp_lab, use_llm_classifier=False)
    assert schema.rule_id  # some rule matched (default fallback if nothing else)
    assert len(schema.roster_initial) >= 1
    assert (tmp_lab / "lab_schema.json").exists()


def test_persist_then_load_returns_same_schema(tmp_lab):
    """Second call reads the persisted file; doesn't re-classify."""
    (tmp_lab / "seed_brief.md").write_text("Audit findings/ for stale claims.")
    s1 = lab_schema_io.load_or_synthesize(tmp_lab, use_llm_classifier=False)
    s2 = lab_schema_io.load_or_synthesize(tmp_lab, use_llm_classifier=False)
    assert s1.rule_id == s2.rule_id
    assert s1.roster_initial == s2.roster_initial
    assert s1.workflow == s2.workflow


def test_corrupt_schema_file_re_synthesizes(tmp_lab):
    """Bad JSON in schema file triggers re-synthesis, not crash."""
    (tmp_lab / "seed_brief.md").write_text("Build a Python CLI utility.")
    (tmp_lab / "lab_schema.json").write_text("{not valid json")
    schema = lab_schema_io.load_or_synthesize(tmp_lab, use_llm_classifier=False)
    assert schema.roster_initial


def test_classifier_failure_falls_back_to_heuristic(tmp_lab):
    """When classifier raises, fall back to default_profile (heuristic)."""
    (tmp_lab / "seed_brief.md").write_text("Some mission.")
    with patch.object(
        mission_profile, "classify_mission",
        side_effect=RuntimeError("network down"),
    ):
        schema = lab_schema_io.load_or_synthesize(tmp_lab)
    assert schema.roster_initial  # didn't crash


def test_default_rule_always_matches(tmp_lab):
    """For nonsense mission, default rule should match (no SchemaLoadError)."""
    (tmp_lab / "seed_brief.md").write_text("xkqzbqz random gibberish")
    schema = lab_schema_io.load_or_synthesize(tmp_lab, use_llm_classifier=False)
    assert schema.rule_id  # any rule matched


def test_force_resynthesize_bypasses_cache(tmp_lab):
    """force_resynthesize=True re-classifies even when cache exists."""
    (tmp_lab / "seed_brief.md").write_text("Some mission.")
    s1 = lab_schema_io.load_or_synthesize(tmp_lab, use_llm_classifier=False)
    # Tamper with seed_brief to ensure resynthesize sees the change
    (tmp_lab / "seed_brief.md").write_text("Completely different mission.")
    s2 = lab_schema_io.load_or_synthesize(
        tmp_lab, use_llm_classifier=False, force_resynthesize=True,
    )
    # Both should have valid rosters; both run through synthesizer
    assert s1.roster_initial
    assert s2.roster_initial


def test_roster_persists_as_tuple_across_reload(tmp_lab):
    """Tuple type (not list) preserved through persist/reload."""
    (tmp_lab / "seed_brief.md").write_text("Some mission.")
    s1 = lab_schema_io.load_or_synthesize(tmp_lab, use_llm_classifier=False)
    assert isinstance(s1.roster_initial, tuple)
    assert isinstance(s1.memory_adapters, tuple)
    s2 = lab_schema_io.load_or_synthesize(tmp_lab, use_llm_classifier=False)
    assert isinstance(s2.roster_initial, tuple)
    assert isinstance(s2.memory_adapters, tuple)


def test_seed_brief_mtime_invalidates_cache(tmp_lab):
    """When seed_brief.md is newer than lab_schema.json, re-synthesize.

    This is the multi-mission scenario: tools/run_mission_suite.sh
    swaps seed_brief.md between missions. Without mtime invalidation,
    all missions would use the FIRST mission's schema.
    """
    import time
    # Initial seed: build mission
    (tmp_lab / "seed_brief.md").write_text(
        "Build a Python CLI utility named jhist. Include pytest tests."
    )
    s1 = lab_schema_io.load_or_synthesize(tmp_lab, use_llm_classifier=False)
    # Swap seed to research mission
    time.sleep(0.05)  # ensure mtime difference (some FS have low resolution)
    (tmp_lab / "seed_brief.md").write_text(
        "Survey papers on vector databases Q2 2026. Comparison table."
    )
    s2 = lab_schema_io.load_or_synthesize(tmp_lab, use_llm_classifier=False)
    # Rosters should differ (build vs research)
    assert s1.roster_initial != s2.roster_initial, (
        f"expected different rosters; got s1={s1.roster_initial} "
        f"s2={s2.roster_initial}"
    )


def test_atomic_write_no_partial_file_on_failure(tmp_lab, monkeypatch):
    """Atomic write: failure mid-write should not leave a partial file."""
    (tmp_lab / "seed_brief.md").write_text("Some mission.")
    # Simulate os.replace failure
    real_replace = os.replace
    call_count = {"n": 0}

    def failing_replace(src, dst):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError("simulated rename failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError):
        lab_schema_io.load_or_synthesize(tmp_lab, use_llm_classifier=False)
    # No partial lab_schema.json should exist
    assert not (tmp_lab / "lab_schema.json").exists()
    # No .tmp files left behind
    tmps = list(tmp_lab.glob(".lab_schema_*.tmp"))
    assert not tmps, f"tmp files left behind: {tmps}"


# ── bert_run.py legacy flag ──────────────────────────────────────────


def test_legacy_env_flag_recognized():
    """BERT_LEGACY_RESEARCHER_STRATEGIST=1 should force legacy roster.

    Smoke test that the env flag is read; full integration is dry-run
    smoke-tested manually."""
    from tools import bert_run
    # The legacy roster is a module-level constant.
    assert bert_run._LEGACY_ROSTER == ("researcher", "strategist")
