"""bert-sandbox MCP server — sandboxed code execution gateway.

Tools:
  - run_python(code, timeout) → execute code in RESTRICTED tier
  - run_shell(command, timeout) → execute shell in RESTRICTED tier
  - validate_skill(skill_path) → run validate_skill helper

All tools route through core.sandbox.* with deny-by-default profiles.
Permission gated: caller must include `approver` field.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core.mcp_server import MCPServer  # noqa: E402


def _gate(args: dict) -> str | None:
    """Permission gate. Return error message if denied, None if allowed."""
    if not args.get("approver"):
        return "approver field required (P-005 permission gate)"
    return None


def _run_python(args: dict) -> dict:
    err = _gate(args)
    if err:
        return {"ok": False, "error": err}
    code = args.get("code", "")
    timeout = int(args.get("timeout", 30))
    if not code:
        return {"ok": False, "error": "code required"}
    from core import sandbox
    r = sandbox.run(
        ["python3", "-c", code],
        tier=sandbox.Tier.RESTRICTED,
        timeout_secs=timeout,
    )
    return {
        "ok": r.exit_code == 0,
        "exit_code": r.exit_code,
        "stdout": r.stdout[:8000],
        "stderr": r.stderr[:4000],
        "elapsed_ms": r.elapsed_ms,
        "tier_used": r.tier_used.value,
        "timed_out": r.timed_out,
    }


def _run_shell(args: dict) -> dict:
    err = _gate(args)
    if err:
        return {"ok": False, "error": err}
    command = args.get("command", "")
    timeout = int(args.get("timeout", 30))
    if not command:
        return {"ok": False, "error": "command required"}
    from core import sandbox
    r = sandbox.run(
        ["bash", "-c", command],
        tier=sandbox.Tier.RESTRICTED,
        timeout_secs=timeout,
    )
    return {
        "ok": r.exit_code == 0,
        "exit_code": r.exit_code,
        "stdout": r.stdout[:8000],
        "stderr": r.stderr[:4000],
        "elapsed_ms": r.elapsed_ms,
        "tier_used": r.tier_used.value,
    }


def _validate_skill(args: dict) -> dict:
    skill_path = args.get("skill_path", "")
    if not skill_path:
        return {"ok": False, "error": "skill_path required"}
    p = Path(skill_path)
    if not p.is_absolute():
        p = LAB_ROOT / p
    if not p.exists():
        return {"ok": False, "error": f"not found: {skill_path}"}
    from core import sandbox
    # validate_skill expects the SKILL.md file
    if p.is_dir():
        p = p / "SKILL.md"
    r = sandbox.validate_skill(p, timeout_secs=int(args.get("timeout", 30)))
    return {
        "ok": r.exit_code == 0,
        "exit_code": r.exit_code,
        "tier_used": r.tier_used.value,
        "stdout": r.stdout[:4000],
        "stderr": r.stderr[:2000],
        "elapsed_ms": r.elapsed_ms,
    }


def make_server() -> MCPServer:
    srv = MCPServer(name="bert-sandbox", version="0.1")
    srv.register_tool(
        "run_python",
        description=("Execute Python code in bert's RESTRICTED sandbox tier. "
                     "Requires `approver` field per P-005."),
        input_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "approver": {"type": "string"},
                "timeout": {"type": "integer", "default": 30},
            },
            "required": ["code", "approver"],
        },
        handler=_run_python,
    )
    srv.register_tool(
        "run_shell",
        description=("Execute shell command in bert's RESTRICTED sandbox tier. "
                     "Requires `approver` field per P-005."),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "approver": {"type": "string"},
                "timeout": {"type": "integer", "default": 30},
            },
            "required": ["command", "approver"],
        },
        handler=_run_shell,
    )
    srv.register_tool(
        "validate_skill",
        description="Run skill validation (sandbox.validate_skill) on a skill manifest.",
        input_schema={
            "type": "object",
            "properties": {
                "skill_path": {"type": "string"},
                "timeout": {"type": "integer", "default": 30},
            },
            "required": ["skill_path"],
        },
        handler=_validate_skill,
    )
    return srv


if __name__ == "__main__":
    sys.exit(make_server().serve_stdio())
