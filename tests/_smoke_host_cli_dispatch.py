"""Smoke + TDD: host-Opus dispatch for ALL roles (the MCP-first tier-1 intent).

The router resolves dispatches to anthropic-cli/claude-opus-4-7 (host tier1) when
running inside Claude Code, but _safe_dispatch only routed to the Claude CLI for
role==researcher + BERT_RESEARCHER_VIA_CLAUDE=1 — so every other role fell into the
standard loop, couldn't call "anthropic-cli", and (with the failover) degraded to
free-tier llama. That's backwards from the pivot: in a host context, bert's
dispatches should run on the user's Opus via `claude -p`. This routes ANY
anthropic-cli-resolved dispatch to the CLI bridge, for all roles, with the
standard free-tier path as the LAST resort (only if the CLI bridge fails).
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "tools"))

import tools.bert_run as br  # noqa: E402


def _spec(model, role="writer"):
    return {"role": role, "cycle": 1, "model": model,
            "output_path": "findings/x.md", "task": "t"}


def _patch(monkeypatch, calls):
    monkeypatch.setattr(br, "_dispatch_via_claude_cli",
                        lambda spec, label, lab: (calls.append("cli"),
                                                  {"result_valid": calls.count("cli_ok") > 0,
                                                   "verdict": "APPROVE", "role": spec["role"],
                                                   "cycle": spec["cycle"]})[1])
    from core import subagent
    monkeypatch.setattr(subagent, "run_subagent",
                        lambda spec: (calls.append("std"),
                                      {"verdict": "BUILD_PASS", "role": spec["role"],
                                       "cycle": spec["cycle"], "result_valid": True})[1])


def test_host_opus_routes_to_cli_for_any_role(monkeypatch):
    calls = ["cli_ok"]  # make the CLI bridge "succeed"
    _patch(monkeypatch, calls)
    out = br._safe_dispatch(_spec("anthropic-cli/claude-opus-4-7", role="writer"),
                            "writer:1", Path("/tmp/lab"))
    assert "cli" in calls and "std" not in calls   # host Opus, not free-tier
    assert out["verdict"] == "APPROVE"


def test_cli_failure_falls_through_to_standard(monkeypatch):
    calls = []  # CLI bridge returns result_valid=False (count('cli_ok')==0)
    _patch(monkeypatch, calls)
    br._safe_dispatch(_spec("anthropic-cli/claude-opus-4-7"), "writer:1", Path("/tmp/lab"))
    assert calls == ["cli", "std"]   # tried host Opus, fell through to standard (free-tier)


def test_non_host_provider_uses_standard_directly(monkeypatch):
    calls = ["cli_ok"]
    _patch(monkeypatch, calls)
    br._safe_dispatch(_spec("nvidia/meta/llama-3.3-70b-instruct"), "writer:1", Path("/tmp/lab"))
    assert "cli" not in calls and "std" in calls   # no host CLI when not host-resolved


def test_no_lab_path_uses_standard(monkeypatch):
    calls = ["cli_ok"]
    _patch(monkeypatch, calls)
    br._safe_dispatch(_spec("anthropic-cli/claude-opus-4-7"), "writer:1", None)
    assert "cli" not in calls and "std" in calls   # CLI bridge needs a lab path


def test_legacy_flag_still_routes_researcher(monkeypatch):
    calls = ["cli_ok"]
    _patch(monkeypatch, calls)
    monkeypatch.setenv("BERT_RESEARCHER_VIA_CLAUDE", "1")
    br._safe_dispatch(_spec("nvidia/x", role="researcher"), "researcher:1", Path("/tmp/lab"))
    assert "cli" in calls   # legacy override preserved


class _MP:
    def __init__(self):
        self._u = []
        self._e = []

    def setattr(self, obj, name, val):
        self._u.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    def setenv(self, k, v):
        import os
        self._e.append((k, os.environ.get(k)))
        os.environ[k] = v

    def undo(self):
        import os
        for o, n, v in reversed(self._u):
            setattr(o, n, v)
        for k, v in reversed(self._e):
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        self._u.clear()
        self._e.clear()


def main() -> int:
    import inspect
    tests = [
        test_host_opus_routes_to_cli_for_any_role,
        test_cli_failure_falls_through_to_standard,
        test_non_host_provider_uses_standard_directly,
        test_no_lab_path_uses_standard,
        test_legacy_flag_still_routes_researcher,
    ]
    mp = _MP()
    for t in tests:
        try:
            if "monkeypatch" in inspect.signature(t).parameters:
                t(mp)
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
        finally:
            mp.undo()
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
