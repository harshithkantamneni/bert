"""Smoke: core/observability.py — JSONL event store (was 57%).

Drives the whole emit/rotate/archive/query surface against a temp
OBS_DIR: emit() + the cycle_outcome / background_invocation / model_call
helpers, calibration_count (with predicate + malformed-line handling),
size-triggered _maybe_rotate + read_archived round-trip, rotate_all, and
the _cli summary. ROTATION_THRESHOLD_BYTES is monkeypatched low so
rotation fires deterministically without writing megabytes.
"""

from __future__ import annotations

import inspect
import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import observability as obs  # noqa: E402


class _MP:
    def __init__(self):
        self._u = []
    def setattr(self, obj, name, val):
        self._u.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)
    def undo(self):
        for o, n, v in reversed(self._u):
            setattr(o, n, v)
        self._u.clear()


def test_emit_and_calibration_count(monkeypatch, tmp_path):
    monkeypatch.setattr(obs, "OBS_DIR", tmp_path)
    obs.emit("verdict", {"verdict": "APPROVE"})
    obs.emit("verdict", {"verdict": "SCOPE_STOP"})
    assert (tmp_path / "verdict.jsonl").exists()
    assert obs.calibration_count("verdict") == 2
    assert obs.calibration_count("verdict", {"verdict": "SCOPE_STOP"}) == 1
    assert obs.calibration_count("nonexistent_class") == 0
    # malformed line → counted defensively (no crash)
    with (tmp_path / "verdict.jsonl").open("a") as f:
        f.write("not valid json\n")
    assert obs.calibration_count("verdict") >= 0


def test_emit_cycle_outcome_and_background(monkeypatch, tmp_path):
    monkeypatch.setattr(obs, "OBS_DIR", tmp_path)
    obs.emit_cycle_outcome(
        7, lab="demo", success=True, elapsed_secs=12.5,
        dispatches_total=4, dispatches_valid=3, verdicts=["APPROVE", "REVISE"],
        findings_produced=2, artifacts_accepted=1,
        concerns_raised=3, concerns_resolved=1, extra={"note": "x"})
    obs.emit_background_invocation(
        "falsifier_baseline", args={"n": 5}, duration_ms=42.0,
        findings_produced=["f1", "f2"], success=True, extra={"k": "v"})
    assert (tmp_path / "cycle_outcome.jsonl").exists()
    assert (tmp_path / "background_invocation.jsonl").exists()
    rec = obs.calibration_count("cycle_outcome", {"success": True})
    assert rec == 1


def test_emit_model_call(monkeypatch, tmp_path):
    monkeypatch.setattr(obs, "OBS_DIR", tmp_path)
    obs.emit_model_call(
        provider="nvidia", model="llama-3.3-70b", input_tokens=100,
        output_tokens=40, cached_tokens=10, thinking_tokens=5,
        elapsed_ms=850, role="researcher", cycle=3)
    assert obs.calibration_count("model_call", {"provider": "nvidia"}) == 1


def test_rotation_and_read_archived(monkeypatch, tmp_path):
    monkeypatch.setattr(obs, "OBS_DIR", tmp_path)
    # no archive yet → empty
    assert obs.read_archived("tool_call") == []
    obs.emit("tool_call", {"tool": "Read", "ok": True})  # creates file
    # shrink threshold so the next emit rotates the existing file
    monkeypatch.setattr(obs, "ROTATION_THRESHOLD_BYTES", 5)
    obs.emit("tool_call", {"tool": "Write", "ok": True})  # triggers rotate
    archived = obs.read_archived("tool_call")
    assert archived, "rotated events should be readable from the archive"
    assert (tmp_path / "archive").exists()


def test_rotate_all(monkeypatch, tmp_path):
    monkeypatch.setattr(obs, "OBS_DIR", tmp_path)
    obs.emit("verdict", {"verdict": "APPROVE"})
    obs.emit("model_call", {"provider": "groq"})
    result = obs.rotate_all(threshold_bytes=1)  # force-rotate everything
    assert isinstance(result, dict)
    assert any(result.values()), "at least one file should rotate at threshold=1"


def test_cli(monkeypatch, tmp_path):
    monkeypatch.setattr(obs, "OBS_DIR", tmp_path)
    obs.emit("verdict", {"verdict": "APPROVE"})
    # usage error
    monkeypatch.setattr(sys, "argv", ["prog"])
    assert obs._cli() == 1
    # calibration summary
    monkeypatch.setattr(sys, "argv", ["prog", "calibration"])
    assert obs._cli() == 0


def test_emit_oversized_event(monkeypatch, tmp_path):
    monkeypatch.setattr(obs, "OBS_DIR", tmp_path)
    # payload > PIPE_BUF (4096) → advisory-lock chunked write path
    obs.emit("tool_call", {"blob": "x" * 6000})
    assert (tmp_path / "tool_call.jsonl").exists()
    assert obs.calibration_count("tool_call") == 1


def test_get_otel_tracer_callable():
    # returns a tracer or None depending on whether the OTel SDK is present;
    # either way the entry + cache path must not crash
    t = obs._get_otel_tracer()
    assert t is None or hasattr(t, "start_as_current_span")


def main() -> int:
    tests = [
        test_emit_and_calibration_count,
        test_emit_cycle_outcome_and_background,
        test_emit_model_call,
        test_rotation_and_read_archived,
        test_rotate_all,
        test_cli,
        test_emit_oversized_event,
        test_get_otel_tracer_callable,
    ]
    for t in tests:
        mp = _MP()
        td = Path(tempfile.mkdtemp())
        try:
            params = inspect.signature(t).parameters
            kwargs = {}
            if "tmp_path" in params:
                kwargs["tmp_path"] = td
            if "monkeypatch" in params:
                kwargs["monkeypatch"] = mp
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
            mp.undo()
            shutil.rmtree(td, ignore_errors=True)
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
