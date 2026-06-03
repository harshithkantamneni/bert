"""Smoke: core/subagent.py dispatch-engine logic (was 42%).

Drives the pure, network-free surface of the dispatch engine: dispatch-
spec + result-packet validation, provider/model parsing, model-family
classification + cross-family evaluator selection, scoped-task + result-
path construction, the JSON-Schema registry, and the Python-native
external verification (_run_verification) against real temp commands.
The LLM dispatch (run_subagent → agent.run_role) is out of scope here —
that path is covered by the live _smoke_spawn / E2E suites.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import subagent  # noqa: E402

_VALID_SPEC = {
    "dispatch_altitude": "INFRA",
    "role": "researcher",
    "cycle": 99,
    "task": "Write a substantive finding to output_path. " * 4,
    "success_criterion": "output file + ResultPacket exist and validate.",
    "output_path": "findings/test_smoke.md",
    "model": "nvidia/meta/llama-3.3-70b-instruct",
    "process_hygiene": "Smoke: no real crawling; minimal compliant output.",
    "confidence_required": True,
    "falsifier_text": "Fails if the output file is missing or schema-invalid.",
}


def test_validate_dispatch_spec_valid_and_invalid():
    ok, errs = subagent.validate_dispatch_spec(_VALID_SPEC)
    assert ok, f"valid spec rejected: {errs}"
    # unknown role rejected
    bad_role = {**_VALID_SPEC, "role": "not_a_real_role"}
    ok2, errs2 = subagent.validate_dispatch_spec(bad_role)
    assert not ok2 and errs2
    # custom- prefix allowed
    custom = {**_VALID_SPEC, "role": "custom-special"}
    ok3, _ = subagent.validate_dispatch_spec(custom)
    assert ok3
    # missing required field
    missing = {k: v for k, v in _VALID_SPEC.items() if k != "task"}
    ok4, errs4 = subagent.validate_dispatch_spec(missing)
    assert not ok4 and errs4


def test_validate_result_packet():
    ok, errs = subagent.validate_result_packet({})  # empty → invalid
    assert not ok and errs


def test_parse_provider_model():
    assert subagent._parse_provider_model("nvidia/meta/llama-3.3-70b-instruct")[0] == "nvidia"
    p, m = subagent._parse_provider_model("groq")
    assert p == "groq" and m is None
    p2, m2 = subagent._parse_provider_model("cerebras/qwen-3-235b")
    assert p2 == "cerebras" and m2 == "qwen-3-235b"


def test_family_and_evaluator_selection():
    fam = subagent.family_of("nvidia")
    assert isinstance(fam, str) and fam
    slot = subagent.slot_family_of("nvidia", "meta/llama-3.3-70b-instruct")
    assert isinstance(slot, str) and slot
    ev = subagent.pick_evaluator_model("nvidia/meta/llama-3.3-70b-instruct")
    assert isinstance(ev, str) and "/" in ev


def test_scoped_task_and_result_path():
    rp = subagent._result_path_for("researcher", 99, "smoke")
    assert isinstance(rp, Path)
    task = subagent._scoped_task(_VALID_SPEC, rp)
    assert isinstance(task, str) and _VALID_SPEC["output_path"] in task


def test_schemas_build():
    assert isinstance(subagent._dispatch_schema(), dict)
    assert isinstance(subagent._result_schema(), dict)
    reg = subagent._build_schema_registry()
    assert reg is not None


def test_run_verification_pass_and_fail():
    ok = subagent._run_verification("exit 0", timeout=10)
    assert ok.get("ok") is True and ok.get("exit_code") == 0
    bad = subagent._run_verification("exit 3", timeout=10)
    assert bad.get("ok") is False and bad.get("exit_code") == 3


def main() -> int:
    tests = [
        test_validate_dispatch_spec_valid_and_invalid,
        test_validate_result_packet,
        test_parse_provider_model,
        test_family_and_evaluator_selection,
        test_scoped_task_and_result_path,
        test_schemas_build,
        test_run_verification_pass_and_fail,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
