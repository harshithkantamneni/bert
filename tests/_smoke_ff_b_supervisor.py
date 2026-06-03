"""Smoke test for FF-B — cross-lab supervisor signal.

FF-B.1 — core/lab_aggregator.py reads every share-enabled lab in
~/.bert/labs/ and produces a CrossLabSignal with per-lab snapshots +
cross-lab rollups.
FF-B.2 — gather_observation for role:supervisor labs enriches the
Observation with the aggregator output. Standard labs receive an
empty cross_lab_signal (perfect isolation).
FF-B.3 — t15_supervisor_pattern_evidence falsifier asserts every
pattern_observed event in the supervisor's events.jsonl cites ≥2
distinct evidence_labs.

Covers:
  - lab_aggregator module shape + exports
  - gather_cross_lab_signal: empty root, single lab, multiple labs,
    share_with_supervisor=false exclusion, supervisor self-exclusion
  - Privacy: opt-out lab is invisible
  - Director gather_observation for role:standard gets empty signal
  - Director gather_observation for role:supervisor gets populated
    signal (when labs_root is wired through; via repo's lab dir we
    can verify the supervisor branch runs without crashing)
  - emit_pattern_observed_event writes correct shape
  - t15 falsifier: INSUFFICIENT when no events, PASS when ≥2-lab
    citation, FAIL when single-lab citation
  - Supervisor prompt section documents the cross-lab discipline
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import lab_aggregator as agg  # noqa: E402
from core import director as dir_mod  # noqa: E402


# ─── FF-B.1 module shape ────────────────────────────────────────────


def test_aggregator_module_exports() -> None:
    for name in ("LabSnapshot", "CrossLabSignal",
                 "gather_cross_lab_signal", "USER_LABS_DIR"):
        assert hasattr(agg, name), f"core.lab_aggregator missing {name!r}"


# ─── FF-B.1 gather_cross_lab_signal paths ───────────────────────────


def test_aggregator_empty_root_returns_empty_signal() -> None:
    tmp = Path(tempfile.mkdtemp())
    nonexistent = tmp / "nope"
    try:
        sig = agg.gather_cross_lab_signal(labs_root=nonexistent)
        assert sig.labs == []
        assert "no user labs directory" in sig.note
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_aggregator_single_lab_visible() -> None:
    tmp_root = Path(tempfile.mkdtemp())
    try:
        a = tmp_root / "lab-a"
        (a / "sor").mkdir(parents=True)
        (a / "lab.yaml").write_text(
            "lab_schema_version: 1\nname: lab-a\nrole: standard\n"
            "focus_areas: [alpha, beta, gamma]\nshare_with_supervisor: true\n"
        )
        with (a / "sor" / "events.jsonl").open("w") as f:
            f.write(json.dumps({
                "event_class": "director_decision_outcome",
                "label": "success", "decision_shape": "research-deeper",
                "decision_area": "alpha", "decision_confidence_1to10": 7,
            }) + "\n")
        sig = agg.gather_cross_lab_signal(labs_root=tmp_root)
        assert len(sig.labs) == 1
        assert sig.labs[0].name == "lab-a"
        assert "alpha" in sig.labs[0].focus_areas
        assert sig.rollups["lab_count"] == 1
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_aggregator_opt_out_lab_excluded() -> None:
    tmp_root = Path(tempfile.mkdtemp())
    try:
        a = tmp_root / "private-lab"
        (a / "sor").mkdir(parents=True)
        (a / "lab.yaml").write_text(
            "lab_schema_version: 1\nname: private-lab\nrole: standard\n"
            "focus_areas: [foo, bar, baz]\n"
            "share_with_supervisor: false\n"
        )
        with (a / "sor" / "events.jsonl").open("w") as f:
            f.write(json.dumps({"event_class": "provider_cooldown",
                                "provider": "mistral"}) + "\n")
        sig = agg.gather_cross_lab_signal(labs_root=tmp_root)
        assert len(sig.labs) == 0
        assert "private-lab" in sig.excluded_labs
        assert sig.exclusion_reasons["private-lab"] == "share_with_supervisor=false"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_aggregator_supervisor_self_excludes() -> None:
    """If a lab in the aggregator root claims role:supervisor, it's
    excluded (a supervisor never aggregates another supervisor's state,
    and never reads itself this way)."""
    tmp_root = Path(tempfile.mkdtemp())
    try:
        a = tmp_root / "rogue-super"
        (a / "sor").mkdir(parents=True)
        (a / "lab.yaml").write_text(
            "lab_schema_version: 1\nname: rogue\nrole: supervisor\n"
            "focus_areas: [routing, memory, discipline, ux]\n"
            "share_with_supervisor: true\n"
        )
        sig = agg.gather_cross_lab_signal(labs_root=tmp_root)
        assert len(sig.labs) == 0
        assert "rogue-super" in sig.excluded_labs
        assert "supervisor" in sig.exclusion_reasons["rogue-super"]
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_aggregator_multi_lab_rollups() -> None:
    tmp_root = Path(tempfile.mkdtemp())
    try:
        for name, provider in [("lab-1", "mistral"), ("lab-2", "mistral"),
                                ("lab-3", "groq")]:
            d = tmp_root / name
            (d / "sor").mkdir(parents=True)
            (d / "lab.yaml").write_text(
                f"lab_schema_version: 1\nname: {name}\nrole: standard\n"
                "focus_areas: [a, b, c]\n"
            )
            with (d / "sor" / "events.jsonl").open("w") as f:
                f.write(json.dumps({"event_class": "provider_cooldown",
                                    "provider": provider}) + "\n")
                f.write(json.dumps({
                    "event_class": "director_decision_outcome",
                    "label": "success",
                    "decision_shape": "research-deeper",
                    "decision_area": "a",
                    "decision_confidence_1to10": 6,
                }) + "\n")
        sig = agg.gather_cross_lab_signal(labs_root=tmp_root)
        assert len(sig.labs) == 3
        # Two labs hit mistral cooldown, one hit groq
        assert sig.rollups["provider_cooldowns_by_provider"]["mistral"] == 2
        assert sig.rollups["provider_cooldowns_by_provider"]["groq"] == 1
        # 3 outcomes total, all success
        assert sig.rollups["outcome_label_distribution"]["success"] == 3
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_aggregator_malformed_lab_skipped_not_crashes() -> None:
    tmp_root = Path(tempfile.mkdtemp())
    try:
        bad = tmp_root / "bad-lab"
        bad.mkdir()
        (bad / "lab.yaml").write_text("this is not: valid: yaml: at all")
        # No sor/ at all
        sig = agg.gather_cross_lab_signal(labs_root=tmp_root)
        # The malformed lab.yaml gets default standard role + defaults,
        # so it's actually included (graceful fallback). The aggregator
        # shouldn't crash.
        assert isinstance(sig, agg.CrossLabSignal)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# ─── FF-B.2 director gather_observation integration ─────────────────


def test_observation_has_cross_lab_signal_field() -> None:
    assert "cross_lab_signal" in dir_mod.Observation.__dataclass_fields__


def test_standard_lab_observation_has_empty_cross_lab_signal() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "seed_brief.md").write_text("# x")
        (tmp / "lab.yaml").write_text(
            "lab_schema_version: 1\nname: cust\nrole: standard\n"
            "focus_areas: [a, b, c]\n"
        )
        (tmp / "sor").mkdir()
        (tmp / "sor" / "events.jsonl").write_text("")
        (tmp / "state").mkdir()
        obs = dir_mod.gather_observation(tmp, iteration=1)
        # Standard labs get empty cross_lab_signal
        assert obs.cross_lab_signal == {}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_supervisor_lab_observation_runs_aggregator() -> None:
    """The repo's own lab/ is role:supervisor; gather_observation
    must populate cross_lab_signal (even if empty, the keys must exist)."""
    obs = dir_mod.gather_observation(LAB_ROOT / "lab", iteration=1)
    # Supervisor labs get a populated cross_lab_signal dict (lab_count
    # may be 0 if no user labs exist yet, but the structure is there)
    assert "lab_count" in obs.cross_lab_signal
    assert "labs" in obs.cross_lab_signal
    assert "rollups" in obs.cross_lab_signal


# ─── FF-B.2 pattern_observed emission ──────────────────────────────


def test_emit_pattern_observed_event_writes_correct_shape() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "sor").mkdir()
        ev = dir_mod.emit_pattern_observed_event(
            tmp,
            pattern_summary="Mistral 1RPM cooldown across multiple labs",
            evidence_labs=["lab-a", "lab-b"],
            iteration=5,
            related_event_classes=["provider_cooldown"],
        )
        assert ev["event_class"] == "pattern_observed"
        assert ev["evidence_lab_count"] == 2
        assert ev["evidence_labs"] == ["lab-a", "lab-b"]
        assert ev["iteration"] == 5
        # And it's appended to events.jsonl
        lines = (tmp / "sor" / "events.jsonl").read_text().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["event_class"] == "pattern_observed"
        assert parsed["evidence_lab_count"] == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_emit_pattern_observed_dedupes_evidence_count() -> None:
    """evidence_lab_count must be DISTINCT count, not raw list length."""
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "sor").mkdir()
        ev = dir_mod.emit_pattern_observed_event(
            tmp,
            pattern_summary="x",
            evidence_labs=["lab-a", "lab-a", "lab-b"],  # 3 entries, 2 distinct
            iteration=1,
        )
        assert ev["evidence_lab_count"] == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── FF-B.3 t15 falsifier ───────────────────────────────────────────


def test_t15_insufficient_when_no_pattern_observed_events() -> None:
    """Before any pattern_observed event lands, the falsifier reports
    INSUFFICIENT_DATA (not FAIL). Fresh supervisor labs shouldn't
    blow up on first cycle just because no patterns have been
    observed yet."""
    # Use a controlled state — we don't want the test to depend on
    # whether the repo's own events.jsonl has patterns yet. Patch
    # LAB_ROOT in the falsifier module to point at a tmp.
    import importlib
    from tools import falsifier_baseline as fb
    importlib.reload(fb)

    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "lab" / "sor").mkdir(parents=True)
        # No events
        original_lab_root = fb.LAB_ROOT
        try:
            fb.LAB_ROOT = tmp
            r = fb.t15_supervisor_pattern_evidence()
            assert r.target_id == 15
            assert r.name == "supervisor_pattern_evidence"
            assert r.status.value == "INSUFFICIENT_DATA"
        finally:
            fb.LAB_ROOT = original_lab_root
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t15_passes_when_all_patterns_cite_two_or_more_labs() -> None:
    import importlib
    from tools import falsifier_baseline as fb
    importlib.reload(fb)

    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "lab" / "sor").mkdir(parents=True)
        with (tmp / "lab" / "sor" / "events.jsonl").open("w") as f:
            f.write(json.dumps({"event_class": "pattern_observed",
                                "evidence_lab_count": 2}) + "\n")
            f.write(json.dumps({"event_class": "pattern_observed",
                                "evidence_lab_count": 3}) + "\n")
        original_lab_root = fb.LAB_ROOT
        try:
            fb.LAB_ROOT = tmp
            r = fb.t15_supervisor_pattern_evidence()
            assert r.status.value == "PASS"
            assert r.sample_size == 2
            assert "100.0%" in r.current_value or "2/2" in r.current_value
        finally:
            fb.LAB_ROOT = original_lab_root
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t15_fails_when_a_pattern_cites_only_one_lab() -> None:
    import importlib
    from tools import falsifier_baseline as fb
    importlib.reload(fb)

    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "lab" / "sor").mkdir(parents=True)
        with (tmp / "lab" / "sor" / "events.jsonl").open("w") as f:
            # 2 well-evidenced, 1 single-lab
            f.write(json.dumps({"event_class": "pattern_observed",
                                "evidence_lab_count": 2}) + "\n")
            f.write(json.dumps({"event_class": "pattern_observed",
                                "evidence_lab_count": 1}) + "\n")
            f.write(json.dumps({"event_class": "pattern_observed",
                                "evidence_lab_count": 3}) + "\n")
        original_lab_root = fb.LAB_ROOT
        try:
            fb.LAB_ROOT = tmp
            r = fb.t15_supervisor_pattern_evidence()
            assert r.status.value == "FAIL"
            assert r.sample_size == 3
        finally:
            fb.LAB_ROOT = original_lab_root
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_t15_wired_into_run_all() -> None:
    """run_all must include the new target."""
    from tools.falsifier_baseline import run_all
    results = run_all()
    target_ids = [r.target_id for r in results]
    assert 15 in target_ids
    assert len(results) == 15  # 14 prior + new t15


# ─── FF-B.2 prompt content ──────────────────────────────────────────


def test_director_prompt_has_supervisor_cross_lab_section() -> None:
    text = (LAB_ROOT / "prompts" / "director_decision.md").read_text()
    assert "Patterns across runtime labs" in text
    assert "supervisor labs only" in text.lower() or "supervisor labs" in text
    # ≥2-lab citation rule is mentioned
    assert "≥2-lab citation" in text or "≥2 distinct source labs" in text
    # supervisor_pattern_evidence falsifier is named
    assert "supervisor_pattern_evidence" in text


def main() -> int:
    tests = [
        test_aggregator_module_exports,
        test_aggregator_empty_root_returns_empty_signal,
        test_aggregator_single_lab_visible,
        test_aggregator_opt_out_lab_excluded,
        test_aggregator_supervisor_self_excludes,
        test_aggregator_multi_lab_rollups,
        test_aggregator_malformed_lab_skipped_not_crashes,
        test_observation_has_cross_lab_signal_field,
        test_standard_lab_observation_has_empty_cross_lab_signal,
        test_supervisor_lab_observation_runs_aggregator,
        test_emit_pattern_observed_event_writes_correct_shape,
        test_emit_pattern_observed_dedupes_evidence_count,
        test_t15_insufficient_when_no_pattern_observed_events,
        test_t15_passes_when_all_patterns_cite_two_or_more_labs,
        test_t15_fails_when_a_pattern_cites_only_one_lab,
        test_t15_wired_into_run_all,
        test_director_prompt_has_supervisor_cross_lab_section,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
