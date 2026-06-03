"""Smoke: core/profile_drift.py — profile↔activity drift detector (was 27%).

All file-based: seeds a temp lab with sor/events.jsonl + lab.yaml and
drives signature extraction, drift_score (no-profile / too-few-cycles /
aligned / drifted branches), _build_recommendation (aligned / clean-
reshape / no-clean-reshape), within_shape_reshape (same-shape success via
the rule-based schema synthesizer, cross-shape rejection, missing-yaml),
and the score/reshape CLI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import profile_drift as pd  # noqa: E402

_PROFILE = {
    "domain": "ml_research", "domain_confidence": 0.9,
    "data_shape": "document_corpus", "primary_work": "discover",
    "work_types": ["discover"], "horizon": "short", "cadence": None,
    "output_kind": "report", "rigor": "cited", "expected_volume": "medium",
    "input_surfaces": ["web_search"], "audience": "self",
    "success_criteria": ["cited findings"], "classifier_confidence": 0.85,
    "stage_used": "stage1",
}


def _seed_lab(tmp: Path, *, shapes: list[tuple[int, str]] | None = None,
              with_profile: bool = True) -> Path:
    (tmp / "sor").mkdir(parents=True, exist_ok=True)
    events = []
    for cyc, shape in (shapes or []):
        events.append({"event_class": "director_decision", "cycle": cyc, "cycle_shape": shape})
        events.append({"event_class": "verdict", "cycle": cyc, "verdict": "APPROVE"})
        events.append({"event_class": "subagent_spawn", "cycle": cyc, "role": "researcher"})
    (tmp / "sor" / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + ("\n" if events else ""))
    if with_profile:
        (tmp / "lab.yaml").write_text(yaml.safe_dump({"mission_profile": dict(_PROFILE)}))
    return tmp


def test_signature_helpers():
    events = [
        {"event_class": "director_decision", "cycle": 1, "cycle_shape": "research-deeper"},
        {"event_class": "verdict", "cycle": 1, "verdict": "APPROVE"},
        {"event_class": "subagent_spawn", "cycle": 1, "role": "researcher"},
    ]
    sig = pd._actual_work_signature(events)
    assert sig["cycle_shapes"]["research-deeper"] == 1
    assert sig["verdicts"]["APPROVE"] == 1 and sig["roles"]["researcher"] == 1
    decl = pd._profile_work_signature({"primary_work": "discover", "work_types": ["compare"]})
    assert "research-deeper" in decl["expected_shapes"]


def test_read_recent_events(tmp_path):
    lab = _seed_lab(tmp_path, shapes=[(c, "research-deeper") for c in range(1, 8)])
    recent = pd._read_recent_events(lab, n_cycles=5)
    cycles = {e["cycle"] for e in recent if e.get("event_class") == "director_decision"}
    assert cycles == {3, 4, 5, 6, 7}  # last 5 of 1..7
    # missing file → []
    assert pd._read_recent_events(tmp_path / "nope") == []
    # malformed line + a no-cycle event are tolerated
    (lab / "sor" / "events.jsonl").open("a").write("not json\n")
    noc = tmp_path / "noc"
    (noc / "sor").mkdir(parents=True)
    (noc / "sor" / "events.jsonl").write_text('{"event_class":"x"}\nbad line\n')
    assert isinstance(pd._read_recent_events(noc), list)  # no cycle ids → all events


def test_drift_score_no_profile(tmp_path):
    lab = _seed_lab(tmp_path, shapes=[(1, "research-deeper")], with_profile=False)
    r = pd.drift_score(lab)
    assert r.score == 0.0 and "No mission_profile" in r.recommendation


def test_drift_score_few_cycles(tmp_path):
    lab = _seed_lab(tmp_path, shapes=[(1, "research-deeper"), (2, "research-deeper")])
    r = pd.drift_score(lab)
    assert r.cycles_inspected < 3 and "need ≥3" in r.recommendation


def test_drift_score_aligned(tmp_path):
    lab = _seed_lab(tmp_path, shapes=[(c, "research-deeper") for c in (1, 2, 3)])
    r = pd.drift_score(lab)
    assert r.score < 0.3 and "line with declared profile" in r.recommendation


def test_drift_score_drifted(tmp_path):
    lab = _seed_lab(tmp_path, shapes=[(c, "verification-tighten") for c in (1, 2, 3)])
    r = pd.drift_score(lab)
    assert r.score > 0.3
    assert r.proposed_changes.get("primary_work") == "audit"


def test_build_recommendation_branches():
    # aligned
    msg, proposed = pd._build_recommendation(
        profile=_PROFILE, observed={"research-deeper": 5}, expected={"research-deeper"}, score=0.1)
    assert proposed == {} and "in line" in msg
    # clean reshape
    msg2, proposed2 = pd._build_recommendation(
        profile=_PROFILE, observed={"verification-tighten": 5}, expected={"research-deeper"}, score=0.8)
    assert proposed2.get("primary_work") == "audit"
    # drift but no clean reshape (shape maps to None)
    msg3, proposed3 = pd._build_recommendation(
        profile=_PROFILE, observed={"idle": 5}, expected={"research-deeper"}, score=0.8)
    assert proposed3 == {} and "no clean reshape" in msg3
    # high score but everything observed is in expected → surprising empty
    msg4, proposed4 = pd._build_recommendation(
        profile=_PROFILE, observed={"research-deeper": 5}, expected={"research-deeper"}, score=0.8)
    assert proposed4 == {} and "aligned" in msg4


def test_within_shape_reshape_same_shape(tmp_path):
    lab = _seed_lab(tmp_path, shapes=[(1, "research-deeper")])
    res = pd.within_shape_reshape(lab, {"primary_work": "compare"})
    assert res["ok"], res.get("error")
    assert res["data_shape_preserved"] == "document_corpus"
    # lab.yaml was rewritten with the new profile
    cfg = yaml.safe_load((lab / "lab.yaml").read_text())
    assert cfg["mission_profile"]["primary_work"] == "compare"


def test_within_shape_reshape_cross_shape_rejected(tmp_path):
    lab = _seed_lab(tmp_path, shapes=[(1, "research-deeper")])
    res = pd.within_shape_reshape(lab, {"data_shape": "code_repo"})
    assert res["ok"] is False and "Cross-shape" in res["error"]


def test_within_shape_reshape_missing_yaml(tmp_path):
    res = pd.within_shape_reshape(tmp_path / "empty_lab", {"primary_work": "compare"})
    assert res["ok"] is False and "lab.yaml not found" in res["error"]


def test_cli(tmp_path):
    lab = _seed_lab(tmp_path, shapes=[(c, "research-deeper") for c in (1, 2, 3)])
    assert pd._cli(["x"]) == 2                          # usage
    assert pd._cli(["x", "score", str(lab)]) == 0
    assert pd._cli(["x", "reshape", str(lab), "primary_work=compare"]) == 0
    # unknown command → 2; reshape arg without '=' is skipped (no-op reshape still ok)
    assert pd._cli(["x", "bogus_cmd", str(lab)]) == 2
    assert pd._cli(["x", "reshape", str(lab), "no_equals_here"]) == 0


def main() -> int:
    import shutil
    import tempfile
    tests = [
        test_signature_helpers,
        test_read_recent_events,
        test_drift_score_no_profile,
        test_drift_score_few_cycles,
        test_drift_score_aligned,
        test_drift_score_drifted,
        test_build_recommendation_branches,
        test_within_shape_reshape_same_shape,
        test_within_shape_reshape_cross_shape_rejected,
        test_within_shape_reshape_missing_yaml,
        test_cli,
    ]
    import inspect
    for t in tests:
        td = Path(tempfile.mkdtemp())
        try:
            kwargs = {"tmp_path": td} if "tmp_path" in inspect.signature(t).parameters else {}
            t(**kwargs)
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
        finally:
            shutil.rmtree(td, ignore_errors=True)
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
