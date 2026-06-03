"""Production skill driver (task #69) — the engine entry point that was missing.

skill_executor is a pure executor; nothing in production ever built the
ExecutionContext it needs (the agent loop drives raw tools, not skills). run_skill
is that bridge: register the tool suite, load the skill registry, and execute a
named skill against the REAL tool_registry via tool_registry.make_invoker(). This
is what lets a live trigger (the lab_finalize MCP tool; later a CLI) actually run
a skill like finalize_project end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

from core import log

LOG = log.get_logger("bert.skill_runner")


def _quality_contract_for_lab(lab_path: str | Path | None) -> dict | None:
    """Read the lab's declared QualityContract (as a dict) from its persisted
    lab_schema.json. Read-only — never triggers synthesis. None if absent/empty.
    """
    if lab_path is None:
        return None
    schema_file = Path(lab_path) / "lab_schema.json"
    if not schema_file.exists():
        return None
    try:
        data = json.loads(schema_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    qc = data.get("quality_contract")
    return qc if isinstance(qc, dict) and qc else None


def _maybe_inject_contract(args: dict, skill, lab_path: str | Path | None) -> dict:
    """Inject the lab's mission QualityContract into skill args when the skill
    declares a `quality_contract` input and the caller didn't supply one. This
    is what makes finalize grade against the mission's contract rather than the
    balanced default. No-op (returns args unchanged) otherwise."""
    if "quality_contract" not in getattr(skill, "inputs", {}):
        return args
    if args.get("quality_contract"):
        return args
    qc = _quality_contract_for_lab(lab_path)
    if not qc:
        return args
    LOG.info("skill_runner: injected mission quality_contract into %s",
             getattr(skill, "name", "?"))
    return {**args, "quality_contract": qc}


def run_skill(skill_name: str, args: dict, *, lab_path: str | Path | None = None) -> dict:
    """Execute a named skill against the real tool registry.

    Returns {ok, outputs, errors, steps_executed}. When `lab_path` is given, all
    relative file I/O (findings, artifact, gaps, ledger) routes into that lab
    (via core.lab_context) — same rule as the Write tool.
    """
    import core.tools  # noqa: F401 — registers the full tool suite on import
    from core import lab_context, skill_executor, skill_registry, tool_registry

    tok = None
    if lab_path is not None:
        tok = lab_context.set_active_lab_path(Path(lab_path))
    try:
        skill_registry.load_all()
        reg = skill_registry.snapshot()
        skill = reg.get(skill_name)
        if skill is None:
            return {"ok": False, "error": f"skill not found: {skill_name!r}",
                    "available": sorted(reg)[:25]}
        args = _maybe_inject_contract(args, skill, lab_path)
        ctx = skill_executor.ExecutionContext(
            tool_invoker=tool_registry.make_invoker(), skill_registry=reg)
        result = skill_executor.execute_skill(skill, args, ctx)
        return {
            "ok": result.ok,
            "outputs": dict(result.outputs),
            "errors": list(result.errors),
            "steps_executed": list(result.steps_executed),
        }
    finally:
        if tok is not None:
            lab_context.reset_active_lab_path(tok)
