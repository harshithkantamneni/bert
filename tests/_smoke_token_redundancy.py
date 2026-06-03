"""Smoke test for tools/measure_token_redundancy.py (H.8)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "measure_token_redundancy",
    LAB_ROOT / "tools" / "measure_token_redundancy.py",
)
mtr = importlib.util.module_from_spec(spec)
sys.modules["measure_token_redundancy"] = mtr
spec.loader.exec_module(mtr)


def test_approx_tokens_lower_bound() -> None:
    assert mtr._approx_tokens("") == 1
    assert mtr._approx_tokens("a" * 100) == 25


def test_role_extraction() -> None:
    assert mtr._role_of({"event_class": "threshing_emit"}) == "threshing"
    assert mtr._role_of({"event_class": "clearness_phase1_query"}) == "clearness_p1"
    assert mtr._role_of({"phase": "phase2_clearness"}) == "clearness_p2"
    assert mtr._role_of({"event_class": "seasoning_proposal"}) == "seasoning"
    assert mtr._role_of({"agent": "judge_qwen"}) == "judge"
    assert mtr._role_of({"event_class": "random"}) == "other"


def test_proxy_redundancy_signal() -> None:
    text_lean = "alpha beta gamma delta epsilon zeta eta theta iota"
    text_redundant = "the cat sat on the mat the cat sat on the mat the cat sat on the mat"
    r_lean = mtr._proxy_redundancy(text_lean)
    r_red = mtr._proxy_redundancy(text_redundant)
    assert r_lean < 0.3, f"lean prose flagged as redundant: {r_lean}"
    assert r_red > 0.5, f"redundant prose not detected: {r_red}"


def test_extract_prompt_proxy_handles_dict_content() -> None:
    ev = {"content": {"foo": "bar", "baz": 42}}
    text = mtr._extract_prompt_proxy(ev)
    assert "foo" in text and "bar" in text


def test_measure_via_proxy_returns_aggregate_shape() -> None:
    events = [
        {"event_class": "threshing_emit", "content": "x" * 200},
        {"event_class": "threshing_emit",
         "content": "the cat sat on the mat the cat sat on the mat " * 5},
        {"event_class": "judge", "agent": "judge_qwen",
         "content": "verdict: alpha beta gamma " * 10},
    ]
    report = mtr.measure_via_proxy(events)
    assert report["method"] == "trigram_proxy"
    assert "roles" in report
    assert "overall" in report


def test_grade_token_waste_thresholds() -> None:
    assert mtr.grade_token_waste({"overall": {"mean_redundancy_pct": 10, "samples": 5}}) == "A"
    assert mtr.grade_token_waste({"overall": {"mean_redundancy_pct": 40, "samples": 5}}) == "A-"
    assert mtr.grade_token_waste({"overall": {"mean_redundancy_pct": 60, "samples": 5}}) == "B"
    assert mtr.grade_token_waste({"overall": {"mean_redundancy_pct": 80, "samples": 5}}) == "C"
    assert mtr.grade_token_waste({"overall": {"samples": 0}}) == "N/A"


def test_render_markdown_has_required_sections() -> None:
    report = {
        "method": "trigram_proxy",
        "roles": {
            "threshing": {"samples": 3, "mean_redundancy_pct": 25.0,
                          "p50_redundancy_pct": 22, "p95_redundancy_pct": 35,
                          "mean_origin_tokens": 120},
        },
        "overall": {"samples": 3, "mean_redundancy_pct": 25.0,
                    "p50_redundancy_pct": 22, "p95_redundancy_pct": 35},
    }
    md = mtr.render_markdown(report, sample_size=10, grade="A")
    assert "Token Redundancy Report" in md
    assert "Overall grade" in md
    assert "threshing" in md
    assert "Per-Role Breakdown" in md


def test_main_writes_outputs(tmp_path=None) -> None:
    """Top-to-bottom: tiny events file → main() produces JSON + md."""
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="bert_h8_"))
    ev_path = tmp / "events.jsonl"
    with ev_path.open("w") as f:
        for i in range(15):
            f.write(json.dumps({
                "event_class": "threshing_emit" if i % 2 else "judge",
                "agent": "judge_qwen" if not i % 2 else "thresher",
                "content": "the cat sat on the mat the cat sat on the mat " * 4,
            }) + "\n")
    orig_events = mtr.EVENTS_PATH
    orig_findings = mtr.FINDINGS_DIR
    try:
        mtr.EVENTS_PATH = ev_path
        mtr.FINDINGS_DIR = tmp / "findings"
        old_argv = sys.argv
        sys.argv = ["measure_token_redundancy", "--proxy-only", "--sample-size", "10"]
        rc = mtr.main()
        sys.argv = old_argv
        assert rc == 0
        outputs = list((tmp / "findings").glob("token_redundancy_*"))
        assert any(p.suffix == ".json" for p in outputs)
        assert any(p.suffix == ".md" for p in outputs)
    finally:
        mtr.EVENTS_PATH = orig_events
        mtr.FINDINGS_DIR = orig_findings


def main() -> int:
    tests = [
        test_approx_tokens_lower_bound,
        test_role_extraction,
        test_proxy_redundancy_signal,
        test_extract_prompt_proxy_handles_dict_content,
        test_measure_via_proxy_returns_aggregate_shape,
        test_grade_token_waste_thresholds,
        test_render_markdown_has_required_sections,
        test_main_writes_outputs,
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
