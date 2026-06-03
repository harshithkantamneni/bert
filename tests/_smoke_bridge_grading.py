"""Smoke + TDD: the Claude CLI bridge must (1) honor the router's resolved
tier (opus/sonnet/haiku — not hardcoded opus) and (2) grade host-Opus output
with the SAME verification_spec the standard loop enforces.

Before this, the bridge returned APPROVE on "file exists + >=100 bytes" and
always ran --model opus. Now that the bridge is the PRIMARY path for every
role in a host context (not a rare researcher override), trusting host output
blindly while rigorously grading free-tier output is backwards, and ignoring
the router's sonnet/haiku tier wastes budget. Both fixed via two pure-ish
helpers tested here directly (no `claude -p` shell-out).
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

import tools.bert_run as br  # noqa: E402


def test_model_flag_maps_resolved_tier():
    f = br._anthropic_cli_model_flag
    assert f("anthropic-cli/claude-opus-4-7") == "opus"
    assert f("anthropic-cli/claude-sonnet-4-6") == "sonnet"
    assert f("anthropic-cli/claude-haiku-4-5") == "haiku"
    # Unknown / bare anthropic-cli -> safe default of opus (highest tier).
    assert f("anthropic-cli/something-new") == "opus"
    assert f("anthropic-cli") == "opus"
    assert f("") == "opus"


def _spec(vspec=None, role="writer"):
    return {"role": role, "cycle": 3, "output_path": "findings/x.md",
            "task": "t", "verification_spec": vspec}


def _cli_out():
    return {"session_id": "abcd1234", "duration_ms": 5000,
            "total_cost_usd": 0.12, "usage": {"output_tokens": 800}}


def test_grade_passes_when_verify_ok(tmp_path):
    art = tmp_path / "findings" / "x.md"
    art.parent.mkdir(parents=True)
    art.write_text("# Title\n\nThis is a sufficiently long deliverable body.\n" * 3)
    spec = _spec({"output_required": True, "min_chars": 50})
    out = br._grade_bridge_artifact(spec, "writer:3", art, "findings/x.md",
                                    _cli_out(), 0.0)
    assert out["verdict"] == "APPROVE"
    assert out["result_valid"] is True
    assert out["errors"] == []


def test_grade_changes_requested_when_verify_fails(tmp_path):
    art = tmp_path / "findings" / "x.md"
    art.parent.mkdir(parents=True)
    art.write_text("too short")
    spec = _spec({"output_required": True, "min_chars": 5000})
    out = br._grade_bridge_artifact(spec, "writer:3", art, "findings/x.md",
                                    _cli_out(), 0.0)
    # Opus wrote SOMETHING, so the cycle keeps the artifact (result_valid=True)
    # and does NOT downgrade to free-tier; but it's honestly not APPROVE.
    assert out["verdict"] == "CHANGES_REQUESTED"
    assert out["result_valid"] is True
    assert any("min_chars" in e for e in out["errors"])


def test_telemetry_splits_cache_read_from_fresh(tmp_path):
    # The 17-47x token "overhead" is inflated by counting cache-reads (an
    # agentic loop re-reads its context every turn). Telemetry must expose the
    # split so real cost (fresh) is separable from cheap re-reads (cache).
    art = tmp_path / "findings" / "x.md"
    art.parent.mkdir(parents=True)
    art.write_text("x" * 200)
    cli = {"session_id": "s", "duration_ms": 1000, "total_cost_usd": 0.1,
           "usage": {"input_tokens": 1000, "cache_creation_input_tokens": 500,
                     "cache_read_input_tokens": 8000, "output_tokens": 400}}
    out = br._grade_bridge_artifact(_spec(None), "writer:3", art, "findings/x.md",
                                    cli, 0.0)
    t = out["telemetry"]
    assert t["tokens_in"] == 1000 + 500 + 8000        # gross unchanged (back-compat)
    assert t["tokens_in_fresh"] == 1000 + 500         # input + cache_creation
    assert t["tokens_cache_read"] == 8000             # re-reads (~10% price)
    assert t["tokens_cache_read"] > t["tokens_in_fresh"]  # gross overstates real cost


def test_grade_no_spec_approves_on_existence(tmp_path):
    art = tmp_path / "findings" / "x.md"
    art.parent.mkdir(parents=True)
    art.write_text("x" * 200)
    out = br._grade_bridge_artifact(_spec(None), "writer:3", art, "findings/x.md",
                                    _cli_out(), 0.0)
    assert out["verdict"] == "APPROVE"
    assert out["result_valid"] is True


def main() -> int:
    import inspect
    import tempfile
    tests = [
        test_model_flag_maps_resolved_tier,
        test_grade_passes_when_verify_ok,
        test_grade_changes_requested_when_verify_fails,
        test_telemetry_splits_cache_read_from_fresh,
        test_grade_no_spec_approves_on_existence,
    ]
    for t in tests:
        try:
            if "tmp_path" in inspect.signature(t).parameters:
                with tempfile.TemporaryDirectory() as d:
                    t(Path(d))
            else:
                t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
