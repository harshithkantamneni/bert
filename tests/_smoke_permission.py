"""Smoke test for core/permission.py — destructive-op gate + Telegram hook.

Tests:
  1. Destructive Bash patterns flagged
  2. Destructive path fragments flagged on Write/Edit
  3. Non-destructive ops not flagged
  4. PLAN mode blocks Write/Bash/Edit
  5. DEFAULT mode logs + allows
  6. Unknown tool name → deny
  7. register_telegram_approver hooks the override path
  8. request_approval no-ops without registered approver
  9. agent.py re-exports _is_destructive and _permission_gate

Run: `.venv/bin/python tests/_smoke_permission.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import agent, permission  # noqa: E402
from core.types import PermissionDecision, PermissionMode, ToolCall  # noqa: E402


class _StubTool:
    permission_mode = PermissionMode.AUTO


def _stub_lookup(name: str):
    if name in {"Read", "Write", "Edit", "Bash"}:
        return _StubTool()
    return None


def _call(name: str, **args) -> ToolCall:
    return ToolCall(id="t1", name=name, arguments=args)


def test_destructive_bash_patterns() -> None:
    for cmd in ["rm -rf /tmp/x", "git push --force origin main", "drop table users",
                "docker volume rm vol1"]:
        assert permission.is_destructive(_call("Bash", command=cmd)), f"{cmd!r} should be destructive"


def test_destructive_path_fragments() -> None:
    for path in ["/Users/foo/.bert-lab/credentials.json", "/etc/hosts",
                 "/Users/foo/.ssh/id_rsa", "/Users/foo/.aws/config"]:
        assert permission.is_destructive(_call("Write", file_path=path, content="x")), f"{path!r} should be destructive"


def test_non_destructive_not_flagged() -> None:
    assert not permission.is_destructive(_call("Bash", command="ls -la"))
    assert not permission.is_destructive(_call("Write", file_path="/tmp/safe.md", content="x"))
    assert not permission.is_destructive(_call("Read", file_path="/tmp/x"))


def test_plan_mode_blocks_writes() -> None:
    d = permission.permission_gate(_call("Write", file_path="/tmp/x", content="x"),
                                    PermissionMode.PLAN, tool_lookup=_stub_lookup)
    assert not d.allowed
    assert "plan mode" in d.reason.lower()


def test_default_mode_logs_and_allows() -> None:
    d = permission.permission_gate(_call("Read", file_path="/tmp/x"),
                                    PermissionMode.DEFAULT, tool_lookup=_stub_lookup)
    assert d.allowed
    assert d.reason == "ok"


def test_unknown_tool_denied() -> None:
    d = permission.permission_gate(_call("Mystery"),
                                    PermissionMode.DEFAULT, tool_lookup=_stub_lookup)
    assert not d.allowed
    assert "unknown" in d.reason.lower()


def test_destructive_blocked_with_approval_flag() -> None:
    d = permission.permission_gate(_call("Bash", command="rm -rf /tmp"),
                                    PermissionMode.AUTO, tool_lookup=_stub_lookup)
    assert not d.allowed
    assert d.requires_telegram_approval
    assert d.is_destructive


def test_telegram_approver_can_override() -> None:
    def approver(call: ToolCall, decision: PermissionDecision) -> PermissionDecision:
        return PermissionDecision(allowed=True, reason="PI approved via telegram",
                                  requires_telegram_approval=False, is_destructive=True)
    permission.register_telegram_approver(approver)
    try:
        deny = PermissionDecision(allowed=False, reason="P-011", requires_telegram_approval=True,
                                  is_destructive=True)
        out = permission.request_approval(_call("Bash", command="rm -rf x"), deny)
        assert out.allowed
        assert "PI approved" in out.reason
    finally:
        permission._telegram_approver = None  # reset for other tests


def test_request_approval_noop_without_approver() -> None:
    permission._telegram_approver = None  # ensure unset
    deny = PermissionDecision(allowed=False, reason="P-011", requires_telegram_approval=True,
                              is_destructive=True)
    out = permission.request_approval(_call("Bash", command="rm -rf x"), deny)
    assert not out.allowed
    assert out is deny  # no-op returns same instance


def test_agent_reexports() -> None:
    assert hasattr(agent, "_is_destructive")
    assert hasattr(agent, "_permission_gate")
    assert agent._is_destructive is permission.is_destructive
    assert agent._permission_gate is permission.permission_gate


def main() -> int:
    tests = [
        test_destructive_bash_patterns,
        test_destructive_path_fragments,
        test_non_destructive_not_flagged,
        test_plan_mode_blocks_writes,
        test_default_mode_logs_and_allows,
        test_unknown_tool_denied,
        test_destructive_blocked_with_approval_flag,
        test_telegram_approver_can_override,
        test_request_approval_noop_without_approver,
        test_agent_reexports,
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
