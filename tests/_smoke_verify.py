"""Smoke test for core/verify.py — ResultPacket signature-forgery checks.

Per FINAL_implementation_plan_2026-05-07.md §5.4 H4 Track D
(AGI D-110 anti-forgery).

Tests:
  1. Genuine packet (matching log + output file) → no forgery
  2. Packet for cycle with no log → forgery (no_cycle_log)
  3. Cycle log exists but no model_response events → forgery
  4. telemetry model mismatches log → forgery (model_mismatch)
  5. output_path file missing → forgery (output_path_missing)
  6. Unreadable packet JSON → forgery (packet_unreadable)
  7. summarize correctly aggregates

Run: `.venv/bin/python tests/_smoke_verify.py`
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

# Use a temp tree so the smoke doesn't read real lab state
TMP = Path(tempfile.mkdtemp(prefix="bert_verify_smoke_"))
RESULTS_DIR = TMP / "state" / "results"
LOGS_DIR = TMP / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

from core import verify as verify_mod  # noqa: E402

verify_mod.LAB_ROOT = TMP
verify_mod.RESULTS_DIR = RESULTS_DIR
verify_mod.LOGS_DIR = LOGS_DIR


def _write_packet(name: str, **kw) -> Path:
    p = RESULTS_DIR / name
    base = {
        "role": "researcher",
        "cycle": 7,
        "verdict": "APPROVE",
        "findings_count": {"high": 0, "med": 0, "low": 0, "nit": 0},
        "confidence_1to10": 5,
        "calibration_reasoning": "x" * 100,
        "telemetry": {"model_used": "meta/llama-3.3-70b-instruct"},
        "output_path": "findings/x.md",
    }
    base.update(kw)
    p.write_text(json.dumps(base))
    return p


def _write_cycle_log(cycle: int, model_responses: list[dict]) -> Path:
    p = LOGS_DIR / f"cycle_{cycle}_20260507.jsonl"
    with p.open("w") as f:
        for ev in model_responses:
            f.write(json.dumps({"kind": "model_response", **ev}) + "\n")
    return p


def _write_output_file(rel: str) -> None:
    op = TMP / rel
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text("placeholder")


def test_genuine_packet_no_forgery() -> None:
    _write_cycle_log(7, [{"model": "meta/llama-3.3-70b-instruct", "tool_calls": 0}])
    _write_output_file("findings/x.md")
    p = _write_packet("good.json")
    rep = verify_mod.verify_packet(p)
    assert not rep.forgery_detected, f"expected clean; got reasons={rep.reasons}"


def test_no_cycle_log_is_unverifiable_not_forgery() -> None:
    """Quality-first refinement: missing cycle log = packet predates
    observability or log was rotated. That's UNVERIFIABLE, not
    forgery. Forgery requires an ACTIVE mismatch."""
    p = _write_packet("orphan.json", cycle=999)
    rep = verify_mod.verify_packet(p)
    assert rep.unverifiable, "missing cycle log should be unverifiable"
    assert not rep.forgery_detected, "missing cycle log alone is not forgery"
    assert any("no_cycle_log" in r for r in rep.reasons), rep.reasons


def test_empty_cycle_log_is_forgery() -> None:
    # Cycle log file exists but has no model_response events
    (LOGS_DIR / "cycle_8_20260507.jsonl").write_text("")
    p = _write_packet("empty.json", cycle=8)
    rep = verify_mod.verify_packet(p)
    assert rep.forgery_detected
    assert any("no_model_responses" in r for r in rep.reasons), rep.reasons


def test_model_mismatch_is_forgery() -> None:
    _write_cycle_log(9, [{"model": "qwen-3-235b", "tool_calls": 0}])
    _write_output_file("findings/x.md")
    p = _write_packet("mismatch.json", cycle=9,
                     telemetry={"model_used": "gpt-4-turbo"})
    rep = verify_mod.verify_packet(p)
    assert rep.forgery_detected
    assert any("model_mismatch" in r for r in rep.reasons), rep.reasons


def test_missing_output_path_is_forgery() -> None:
    _write_cycle_log(10, [{"model": "meta/llama-3.3-70b-instruct"}])
    p = _write_packet("missing_out.json", cycle=10,
                     output_path="findings/does_not_exist.md")
    rep = verify_mod.verify_packet(p)
    assert rep.forgery_detected
    assert any("output_path_missing" in r for r in rep.reasons), rep.reasons


def test_unreadable_packet_is_forgery() -> None:
    p = RESULTS_DIR / "broken.json"
    p.write_text("{not valid json")
    rep = verify_mod.verify_packet(p)
    assert rep.forgery_detected
    assert any("packet_unreadable" in r for r in rep.reasons), rep.reasons


def test_summarize_aggregates() -> None:
    """Genuine + forgery + unverifiable all distinct buckets."""
    _write_cycle_log(7, [{"model": "meta/llama-3.3-70b-instruct"}])
    _write_output_file("findings/x.md")
    _write_packet("good2.json")  # genuine clean
    # Forgery: cycle log exists, model mismatches
    _write_cycle_log(9, [{"model": "qwen-3-235b"}])
    _write_packet("forged.json", cycle=9, telemetry={"model_used": "gpt-4-turbo"})
    # Unverifiable: missing cycle log, no other mismatch
    _write_packet("unverif.json", cycle=8888)

    reports = verify_mod.verify_results_dir(RESULTS_DIR)
    summary = verify_mod.summarize(reports)
    assert summary["total"] >= 3
    assert summary["forged_count"] >= 1, f"expected ≥1 forgery; got {summary}"
    assert summary["unverifiable_count"] >= 1, f"expected ≥1 unverifiable; got {summary}"
    assert summary["any_forgery"] is True
    # Forgery + unverifiable + clean must be disjoint
    assert (summary["forged_count"] + summary["unverifiable_count"]
            + summary["clean_count"]) == summary["total"]


def main() -> int:
    tests = [
        test_genuine_packet_no_forgery,
        test_no_cycle_log_is_unverifiable_not_forgery,
        test_empty_cycle_log_is_forgery,
        test_model_mismatch_is_forgery,
        test_missing_output_path_is_forgery,
        test_unreadable_packet_is_forgery,
        test_summarize_aggregates,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__} (exception)")
            print(f"        {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
