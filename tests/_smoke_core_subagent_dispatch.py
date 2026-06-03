"""Smoke: core/subagent.py run_subagent dispatch loop (was 50%).

Stubs the one network seam (core.agent.run_role) + pins _result_path_for to
a temp path, so the real dispatch flow runs offline: spec validation,
observability emit, scoped-task build, packet read + validate, verdict
mapping, telemetry overwrite, the crash path, the no-packet failure
synthesis, and the verification_command path.
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

from core import subagent  # noqa: E402

_SPEC = {
    "dispatch_altitude": "INFRA",
    "role": "researcher",
    "cycle": 42,
    "task": "Write a substantive finding. " * 4,
    "success_criterion": "ResultPacket exists and validates.",
    "output_path": "findings/sub_smoke.md",
    "model": "nvidia/meta/llama-3.3-70b-instruct",
    "process_hygiene": "Smoke: minimal compliant output.",
    "confidence_required": True,
    "falsifier_text": "Fails if the packet is missing or schema-invalid.",
}


def _valid_packet():
    return {
        "verdict": "APPROVE",
        "role": "researcher",
        "cycle": 42,
        "dispatch_id": "researcher_C42",
        "findings_count": {"high": 1, "med": 0, "low": 0, "nit": 0},
        "confidence_1to10": 8,
        "calibration_reasoning": "Confident: the finding is well-sourced and falsifiable. " * 2,
        "telemetry": {"tokens_in": 100, "tokens_out": 40, "latency_secs": 1.0,
                      "model_used": "nvidia/meta/llama-3.3-70b-instruct"},
    }


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


def _pin(monkeypatch, tmp_path):
    packet_path = tmp_path / "packet.json"
    monkeypatch.setattr(subagent, "_result_path_for", lambda role, cycle, tag: packet_path)
    monkeypatch.setattr(subagent, "_active_results_dir", lambda: tmp_path)
    monkeypatch.setattr(subagent.observability, "emit", lambda *a, **k: None)
    return packet_path


def test_invalid_spec_returns_other():
    out = subagent.run_subagent({"role": "researcher"})  # missing required fields
    assert out["spec_valid"] is False and out["verdict"]


def test_happy_path(monkeypatch, tmp_path):
    from core import agent as _agent
    packet_path = _pin(monkeypatch, tmp_path)

    def fake_run_role(role, *, cycle=1, task=None, telemetry_sink=None, **kw):
        if telemetry_sink is not None:
            telemetry_sink.update({"model_used": "nvidia/real", "tokens_in": 100,
                                   "tokens_out": 40, "provider": "nvidia"})
        packet_path.write_text(json.dumps(_valid_packet()))
        return 0
    monkeypatch.setattr(_agent, "run_role", fake_run_role)
    out = subagent.run_subagent(_SPEC)
    assert out["spec_valid"] is True and out["result_valid"] is True
    assert out["role"] == "researcher" and out["verdict"] == "APPROVE"


def test_crash_path(monkeypatch, tmp_path):
    from core import agent as _agent
    _pin(monkeypatch, tmp_path)

    def boom(role, **kw):
        raise RuntimeError("agent exploded")
    monkeypatch.setattr(_agent, "run_role", boom)
    out = subagent.run_subagent(_SPEC)
    # crash → no packet written → honest failure packet, never raises
    assert out["spec_valid"] is True and isinstance(out["verdict"], str)
    assert out["errors"]


def test_no_packet_failure(monkeypatch, tmp_path):
    from core import agent as _agent
    _pin(monkeypatch, tmp_path)
    # run_role returns rc=1 without writing a packet, no verification → failure synth
    monkeypatch.setattr(_agent, "run_role", lambda role, **kw: 1)
    out = subagent.run_subagent(_SPEC)
    assert out["spec_valid"] is True and out["errors"]


def test_verification_command_path(monkeypatch, tmp_path):
    from core import agent as _agent
    packet_path = _pin(monkeypatch, tmp_path)
    monkeypatch.setattr(_agent, "run_role",
                        lambda role, **kw: (packet_path.write_text(json.dumps(_valid_packet())), 0)[1])
    spec = {**_SPEC, "verification_command": "exit 0", "verification_timeout_secs": 10}
    out = subagent.run_subagent(spec)
    assert out["spec_valid"] is True


def test_verification_spec_path(monkeypatch, tmp_path):
    import types

    from core import agent as _agent
    from core import verify_engine
    packet_path = _pin(monkeypatch, tmp_path)
    monkeypatch.setattr(_agent, "run_role",
                        lambda role, **kw: (packet_path.write_text(json.dumps(_valid_packet())), 0)[1])
    # Python-native verification_spec → verify_engine.verify_artifact (stubbed)
    monkeypatch.setattr(verify_engine, "verify_artifact",
                        lambda spec, path, **k: types.SimpleNamespace(
                            ok=True, exit_code=0, checks_passed=["c1"], checks_failed=[],
                            elapsed_ms=5, timed_out=False))
    spec = {**_SPEC, "verification_spec": {"checks": [{"kind": "file_exists"}]},
            "verification_timeout_secs": 10}
    out = subagent.run_subagent(spec)
    assert out["spec_valid"] is True


def test_output_landed_synthesis(monkeypatch, tmp_path):
    # run_role writes NO packet but DOES write the output_path file →
    # run_subagent synthesizes a BUILD_PASS packet from evidence-of-work.
    from core import agent as _agent
    from core import lab_context
    _pin(monkeypatch, tmp_path)
    monkeypatch.setattr(lab_context, "get_active_lab_path", lambda: tmp_path)
    out_file = tmp_path / _SPEC["output_path"]   # findings/sub_smoke.md
    out_file.parent.mkdir(parents=True, exist_ok=True)

    def write_output_not_packet(role, **kw):
        out_file.write_text("# Finding\n\nThe agent wrote its output but not the packet.\n")
        return 0
    monkeypatch.setattr(_agent, "run_role", write_output_not_packet)
    out = subagent.run_subagent(_SPEC)
    assert out["spec_valid"] is True and out["result_valid"] is True
    assert "synthesized" in " ".join(out.get("errors", []))


def test_dispatch_chain(monkeypatch, tmp_path):
    from core import agent as _agent
    packet_path = _pin(monkeypatch, tmp_path)
    monkeypatch.setattr(_agent, "run_role",
                        lambda role, **kw: (packet_path.write_text(json.dumps(_valid_packet())), 0)[1])
    assert subagent.dispatch_chain([]) == []
    results = subagent.dispatch_chain([_SPEC, {**_SPEC, "cycle": 43}])
    assert len(results) == 2 and all(r["spec_valid"] for r in results)
    # an invalid spec in the chain → its summary marks spec_valid False
    mixed = subagent.dispatch_chain([{"role": "researcher"}])
    assert mixed and mixed[0]["spec_valid"] is False


def main() -> int:
    tests = [
        test_invalid_spec_returns_other,
        test_happy_path,
        test_crash_path,
        test_no_packet_failure,
        test_verification_command_path,
        test_verification_spec_path,
        test_output_landed_synthesis,
        test_dispatch_chain,
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
