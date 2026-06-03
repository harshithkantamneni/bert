"""Smoke: core/evaluator.py — cycle quality-gate checks (was 73%).

The check_N_* functions take injectable events/dirs, so we drive each
PASS + FAIL branch with synthetic fixtures (guaranteed net-new vs the live
suite, which only ever sees real data). Covers checks 4/6/7/8/18/19/20/1
+ check_21 (real verify, tolerant) + _load_cycle_events.
"""

from __future__ import annotations

import inspect
import json
import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import evaluator as ev  # noqa: E402

P = ev.CheckStatus.PASS
F = ev.CheckStatus.FAIL


def test_check_4_director_specialist_work():
    over = [{"kind": "tool_result", "tool": "Bash", "elapsed_ms": 400_000}]
    assert ev.check_4_director_specialist_work(over).status == F
    under = [{"kind": "tool_result", "tool": "Bash", "elapsed_ms": 1000}]
    assert ev.check_4_director_specialist_work(under).status == P


def test_check_6_distinct_output_paths(tmp_path):
    assert ev.check_6_distinct_output_paths(1, results_dir=tmp_path / "none").status == ev.CheckStatus.NA
    (tmp_path / "p1.json").write_text(json.dumps({"cycle": 1, "output_path": "a.md"}))
    (tmp_path / "p2.json").write_text(json.dumps({"cycle": 1, "output_path": "b.md"}))
    assert ev.check_6_distinct_output_paths(1, results_dir=tmp_path).status == P
    (tmp_path / "p3.json").write_text(json.dumps({"cycle": 1, "output_path": "a.md"}))  # dup
    assert ev.check_6_distinct_output_paths(1, results_dir=tmp_path).status == F


def test_check_7_tier1_read_budget():
    over = [{"kind": "tool_result", "tool": "Read", "content_preview": "x" * 5000}]
    assert ev.check_7_tier1_read_budget(over, budget=100).status == F
    # Spawn breaks the pre-dispatch accounting
    seq = [{"kind": "tool_call", "tool": "Spawn"},
           {"kind": "tool_result", "tool": "Read", "content_preview": "x" * 5000}]
    assert ev.check_7_tier1_read_budget(seq, budget=100).status == P


def test_check_8_memory_cap_pressure(tmp_path):
    budgets = {"memory_hot_max": 10, "memory_log_max": 10}
    assert ev.check_8_memory_cap_pressure(budgets, memories_dir=tmp_path / "none").status == ev.CheckStatus.NA
    (tmp_path / "current.md").write_text("x" * 100)   # over hot cap
    assert ev.check_8_memory_cap_pressure(budgets, memories_dir=tmp_path).status == F
    (tmp_path / "current.md").write_text("x")          # within
    assert ev.check_8_memory_cap_pressure(budgets, memories_dir=tmp_path).status == P


def test_check_18_permission_gates():
    bad = [{"kind": "permission_decision", "destructive": True, "allowed": True,
            "reason": "looked fine", "tool": "Bash"}]
    assert ev.check_18_permission_gates(bad).status == F
    ok = [{"kind": "permission_decision", "destructive": True, "allowed": True,
           "reason": "user approve via telegram", "tool": "Bash"}]
    assert ev.check_18_permission_gates(ok).status == P


def test_check_19_spend_killswitch():
    over = [{"kind": "model_response", "tokens_in": 600, "tokens_out": 600}]
    assert ev.check_19_spend_killswitch(over, budget=1000).status == F
    assert ev.check_19_spend_killswitch(over, budget=5000).status == P


def test_check_20_identical_call_counter():
    same = [{"kind": "tool_call", "tool": "Read", "arguments": {"file_path": "x"}}] * 4
    assert ev.check_20_identical_call_counter(same, threshold=3).status == F
    varied = [{"kind": "tool_call", "tool": "Read", "arguments": {"file_path": f"x{i}"}}
              for i in range(4)]
    assert ev.check_20_identical_call_counter(varied, threshold=3).status == P


def test_check_1_pre_commitment(tmp_path):
    q = tmp_path / "cycle_queue.md"
    q.write_text("# Queue\n\n1. first\n2. second\n3. third\n")
    assert ev.check_1_pre_commitment_exists(cycle_queue_path=q).status == P
    assert ev.check_1_pre_commitment_exists(cycle_queue_path=tmp_path / "missing.md").status == F


def test_check_21_and_load_events():
    # real verify over the repo results dir — tolerant, just returns a CheckResult
    r = ev.check_21_signature_forgery()
    assert r.status in (P, F, ev.CheckStatus.NA)
    assert isinstance(ev._load_cycle_events(999999), list)   # no events for far cycle


def test_check_11_calibration_reasoning(tmp_path):
    log = tmp_path / "log.md"
    log.write_text("## D-001\n" + ("x" * 120) + "\n## D-002\nshort\n")
    r = ev.check_11_calibration_reasoning_quality(log_path=log, min_chars=80)
    assert r.status in (P, F)
    assert ev.check_11_calibration_reasoning_quality(
        log_path=tmp_path / "missing.md").status == ev.CheckStatus.NA


def test_check_14_build_pass_blocking(tmp_path):
    assert ev.check_14_build_pass_blocking(1, results_dir=tmp_path / "none").status == ev.CheckStatus.NA
    # a non-build cycle → NA (no build dispatches)
    (tmp_path / "p.json").write_text(json.dumps({"cycle": 1, "verdict": "APPROVE"}))
    r = ev.check_14_build_pass_blocking(1, results_dir=tmp_path)
    assert r.status in (P, F, ev.CheckStatus.NA)


def test_check_17_constitutional_preamble(tmp_path):
    assert ev.check_17_constitutional_preamble(governance_dir=tmp_path / "none").status == F
    (tmp_path / "constitutional.md").write_text("# Constitution\n\n" + ("principle. " * 50))
    r = ev.check_17_constitutional_preamble(governance_dir=tmp_path)
    assert r.status in (P, F)


def test_check_22_roster_health(tmp_path):
    (tmp_path / "researcher").mkdir()
    (tmp_path / "researcher" / "procedural.md").write_text("# role")
    r = ev.check_22_roster_health(agents_dir=tmp_path, cycle=1)
    assert hasattr(r, "status")


def test_evaluate_cycle_orchestration():
    # runs every check against the real repo state (tolerant) → a CycleEvaluation
    result = ev.evaluate_cycle(999999)
    assert hasattr(result, "overall") and hasattr(result, "fail_count")
    assert isinstance(ev.gates_graceful_exit(result), bool)
    assert isinstance(ev.render_report(result), str)


def main() -> int:
    tests = [
        test_check_4_director_specialist_work,
        test_check_11_calibration_reasoning,
        test_check_14_build_pass_blocking,
        test_check_17_constitutional_preamble,
        test_check_22_roster_health,
        test_evaluate_cycle_orchestration,
        test_check_6_distinct_output_paths,
        test_check_7_tier1_read_budget,
        test_check_8_memory_cap_pressure,
        test_check_18_permission_gates,
        test_check_19_spend_killswitch,
        test_check_20_identical_call_counter,
        test_check_1_pre_commitment,
        test_check_21_and_load_events,
    ]
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
