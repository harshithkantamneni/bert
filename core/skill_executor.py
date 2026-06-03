"""Skill executor — interprets a parsed Skill against an ExecutionContext.

Sprint 2 commit 14: walks a Skill's steps in order, evaluating
template expressions, dispatching to tools or sub-skills, handling
foreach (sequential and parallel), conditional steps, and declared
failure modes.

Template expressions use Jinja2-sandboxed evaluation. Args can
reference:
  - skill inputs:   "{{topic}}"
  - prior captures: "{{prior_step.output_field}}"
  - loop items:     "{{item}}" (inside foreach)
  - filters:        "{{topic | comma}}", "{{x | default('foo')}}"

Failure mode handlers supported:
  - "retry"                : retry the step in place
  - "retry_after_<N>s"     : sleep then retry (Round 2 C-12 friendly)
  - "fallback:<skill_name>": invoke the named sub-skill instead
  - "emit_<state>"         : emit named state in captures + skip step
  - "fail"                 : raise (default for unhandled failures)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from core.skill_dsl import (
    _NAME_AT_VERSION,
    Skill,
    SkillStep,
)

LOG = logging.getLogger("bert.skill_executor")


# ── Public types ────────────────────────────────────────────────────


@dataclass
class ExecutionContext:
    """All the side-effects + lookups a skill might need.

    Designed for dependency injection — tests pass fakes; production
    passes real tool registry + skill registry + observability hooks.
    """
    tool_invoker: Callable[[str, dict], Any] | None = None
    """Sync callable: tool_invoker(tool_name, args) → any result."""

    async_tool_invoker: Callable[[str, dict], Awaitable[Any]] | None = None
    """Async equivalent for parallel foreach."""

    skill_registry: dict[str, Skill] = field(default_factory=dict)
    """Snapshot of available skills, name → Skill (per Round 2 C-5)."""

    cycle_id: int | None = None
    lab: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def invoke_tool(self, tool_name: str, args: dict) -> Any:
        if self.tool_invoker is None:
            raise SkillExecutionError(
                f"no tool_invoker configured in ExecutionContext "
                f"(tried to invoke {tool_name!r})"
            )
        return self.tool_invoker(tool_name, args)

    async def invoke_tool_async(self, tool_name: str, args: dict) -> Any:
        if self.async_tool_invoker is not None:
            return await self.async_tool_invoker(tool_name, args)
        # Fall back to sync invoker wrapped in async
        return await asyncio.get_event_loop().run_in_executor(
            None, self.tool_invoker, tool_name, args,
        )


@dataclass
class SkillResult:
    """Outcome of executing a skill."""
    ok: bool
    outputs: dict[str, Any] = field(default_factory=dict)
    captures: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0
    steps_executed: list[str] = field(default_factory=list)
    steps_failed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class SkillExecutionError(Exception):
    """Raised when a step fails and no failure_mode handler covers it."""


# ── Template expansion (Jinja2-sandboxed) ───────────────────────────


_jinja_env = None


def _get_jinja():
    """Lazy-init a Jinja2 SandboxedEnvironment with our filters."""
    global _jinja_env
    if _jinja_env is not None:
        return _jinja_env
    try:
        from jinja2.sandbox import SandboxedEnvironment
    except ImportError:
        # Fallback: simple {{var}} substitution without filters
        return None
    env = SandboxedEnvironment(autoescape=False)
    # Custom filters
    env.filters["comma"] = lambda items: ", ".join(map(str, items))
    env.filters["slug"] = lambda s: re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")
    env.filters["percent"] = lambda f: f"{float(f) * 100:.0f}%"
    _jinja_env = env
    return env


def _resolve_value(value: Any, bindings: dict[str, Any]) -> Any:
    """Recursively expand any string templates in `value` against `bindings`.

    Lists, dicts, and primitives are passed through with nested
    template expansion. Strings are run through Jinja2 (sandbox).
    """
    if isinstance(value, str):
        return _resolve_string(value, bindings)
    if isinstance(value, list):
        return [_resolve_value(v, bindings) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_value(v, bindings) for k, v in value.items()}
    return value


def _resolve_string(s: str, bindings: dict[str, Any]) -> Any:
    """Render `s` as a Jinja2 template against `bindings`.

    Special case: if the entire string is exactly `{{var}}` (no
    surrounding text), preserve the underlying type instead of
    stringifying (so a list binding stays a list).
    """
    # Exact-match short-circuit for typed pass-through
    m = re.fullmatch(r"\s*\{\{\s*([a-zA-Z_][a-zA-Z0-9_.\[\]]*)\s*\}\}\s*", s)
    if m:
        key = m.group(1)
        return _deep_lookup(bindings, key)
    env = _get_jinja()
    if env is None:
        # Naive {{var}} substitution — limited; no filters
        out = s
        for k, v in bindings.items():
            out = out.replace(f"{{{{{k}}}}}", str(v))
        return out
    try:
        template = env.from_string(s)
        return template.render(**bindings)
    except Exception as e:  # noqa: BLE001
        LOG.debug("template render failed (%s) on %r; passthrough", e, s)
        return s


def _deep_lookup(bindings: dict[str, Any], key: str) -> Any:
    """Look up `a.b.c` or `a[0]` in nested bindings."""
    # Strip indexing
    cur: Any = bindings
    for part in re.split(r"\.", key):
        # Handle list indexing like items[0]
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\[(\d+)\]$", part)
        if m:
            attr, idx = m.group(1), int(m.group(2))
            cur = cur.get(attr) if isinstance(cur, dict) else getattr(cur, attr, None)
            if isinstance(cur, (list, tuple)) and 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                return None
        else:
            cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _eval_condition(expr: str, bindings: dict[str, Any]) -> bool:
    """Evaluate an `if_` expression using Jinja2's expression engine."""
    env = _get_jinja()
    if env is None:
        # Naive: if expr is exactly a binding name, check truthiness
        return bool(bindings.get(expr.strip("{}").strip()))
    try:
        # Wrap in `{{ ... }}` form so Jinja2 renders the boolean
        template = env.from_string(f"{{{{ ({expr}) }}}}")
        rendered = template.render(**bindings)
        return rendered.strip().lower() in {"true", "1", "yes"}
    except Exception:  # noqa: BLE001
        return False


# ── The executor ────────────────────────────────────────────────────


def execute_skill(
    skill: Skill,
    inputs: dict[str, Any],
    context: ExecutionContext,
) -> SkillResult:
    """Run a skill end-to-end, sequentially. Returns a SkillResult.

    Validates inputs against skill.inputs first; missing required
    inputs → SkillResult.ok=False with errors. Otherwise interprets
    steps in order. Async parallel foreach (`foreach_parallel`) is
    handled by spawning an asyncio loop transparently.
    """
    start = time.monotonic()

    # ── Validate inputs ──────────────────────────────────────────
    bindings: dict[str, Any] = {}
    errors: list[str] = []
    for in_name, in_spec in skill.inputs.items():
        if in_name in inputs:
            bindings[in_name] = inputs[in_name]
        elif in_spec.required:
            errors.append(f"required input {in_name!r} missing")
        elif in_spec.default is not None:
            bindings[in_name] = in_spec.default
    if errors:
        return SkillResult(
            ok=False, elapsed_ms=int((time.monotonic() - start) * 1000),
            errors=errors,
        )

    steps_executed: list[str] = []
    steps_failed: list[str] = []

    for step in skill.steps:
        # Conditional skip
        if step.if_:
            try:
                if not _eval_condition(step.if_, bindings):
                    LOG.debug("skill %s step %s: if-condition false; skip", skill.name, step.id)
                    continue
            except Exception as e:  # noqa: BLE001
                errors.append(f"step {step.id}: if-condition errored: {e}")
                steps_failed.append(step.id)
                continue

        try:
            result = _run_step(step, bindings, context, skill)
        except SkillExecutionError as e:
            handled = _apply_failure_mode(skill, step, e, bindings, context)
            if not handled:
                errors.append(f"step {step.id}: {e}")
                steps_failed.append(step.id)
                # Stop on unhandled error
                break
            result = handled.get("captures_override")
        if step.capture and result is not None:
            bindings[step.capture] = result
        steps_executed.append(step.id)

    # ── Build outputs from bindings according to skill.outputs ──
    outputs: dict[str, Any] = {}
    for out_name in skill.outputs:
        if out_name in bindings:
            outputs[out_name] = bindings[out_name]

    return SkillResult(
        ok=not steps_failed and not errors,
        outputs=outputs,
        captures=bindings,
        elapsed_ms=int((time.monotonic() - start) * 1000),
        steps_executed=steps_executed,
        steps_failed=steps_failed,
        errors=errors,
    )


def _run_step(
    step: SkillStep,
    bindings: dict[str, Any],
    context: ExecutionContext,
    parent_skill: Skill,
) -> Any:
    """Execute one step. Returns the value to be captured (or None)."""
    # foreach / foreach_parallel
    if step.foreach or step.foreach_parallel:
        return _run_foreach(step, bindings, context, parent_skill)

    # Resolve args from templates
    resolved_args = _resolve_value(step.args, bindings)
    if not isinstance(resolved_args, dict):
        raise SkillExecutionError(
            f"step {step.id}: args resolved to non-dict ({type(resolved_args).__name__})"
        )

    if step.tool:
        try:
            return context.invoke_tool(step.tool, resolved_args)
        except Exception as e:  # noqa: BLE001
            raise SkillExecutionError(f"tool {step.tool} raised: {e}") from e

    if step.skill:
        return _run_sub_skill(step.skill, resolved_args, context)

    raise SkillExecutionError(f"step {step.id}: neither tool nor skill set")


def _run_foreach(
    step: SkillStep,
    bindings: dict[str, Any],
    context: ExecutionContext,
    parent_skill: Skill,
) -> list[Any]:
    """Run a step over each item in a list, sequential or parallel."""
    iter_key = step.foreach or step.foreach_parallel
    items = _deep_lookup(bindings, iter_key)
    if items is None:
        items = []
    if not isinstance(items, (list, tuple)):
        raise SkillExecutionError(
            f"step {step.id}: foreach target {iter_key!r} is not iterable ({type(items).__name__})"
        )

    if step.foreach:
        # Sequential
        results = []
        for item in items:
            scope = dict(bindings, item=item)
            resolved_args = _resolve_value(step.args, scope)
            if step.tool:
                results.append(context.invoke_tool(step.tool, resolved_args))
            elif step.skill:
                results.append(_run_sub_skill(step.skill, resolved_args, context))
        return results

    # foreach_parallel — use asyncio with bounded concurrency
    sem_max = max(1, step.foreach_max_concurrent)

    async def _run_all():
        sem = asyncio.Semaphore(sem_max)
        async def _one(item):
            async with sem:
                scope = dict(bindings, item=item)
                resolved_args = _resolve_value(step.args, scope)
                if step.tool:
                    return await context.invoke_tool_async(step.tool, resolved_args)
                if step.skill:
                    # Sub-skills run sync inside the executor; wrap in thread
                    return await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: _run_sub_skill(step.skill, resolved_args, context),
                    )
                return None
        return await asyncio.gather(*[_one(it) for it in items])

    try:
        return asyncio.run(_run_all())
    except RuntimeError as e:
        # Running inside an existing event loop — fall back to sequential
        LOG.debug("foreach_parallel: %s; falling back to sequential", e)
        return _run_foreach(
            SkillStep(
                id=step.id, tool=step.tool, skill=step.skill, args=step.args,
                capture=step.capture, foreach=iter_key,
            ),
            bindings, context, parent_skill,
        )


def _run_sub_skill(skill_ref: str, args: dict, context: ExecutionContext) -> Any:
    """Invoke a sub-skill via the registry."""
    m = _NAME_AT_VERSION.match(skill_ref)
    if not m:
        raise SkillExecutionError(f"invalid skill ref {skill_ref!r}")
    sub_name = m.group(1)
    sub = context.skill_registry.get(sub_name)
    if sub is None:
        raise SkillExecutionError(f"sub-skill {sub_name!r} not in registry")
    result = execute_skill(sub, args, context)
    if not result.ok:
        raise SkillExecutionError(
            f"sub-skill {sub_name} failed: {'; '.join(result.errors)}"
        )
    return result.outputs


def _apply_failure_mode(
    skill: Skill, step: SkillStep, error: Exception,
    bindings: dict[str, Any], context: ExecutionContext,
) -> dict[str, Any] | None:
    """Match an error against skill.failure_modes and apply the handler.

    Returns a dict {"captures_override": ...} if handled, or None if
    the error should propagate.

    Resolution order:
      1. `retry` handlers (with max_retries) — try the step again
      2. `fallback:<skill>` handlers — invoke the named sub-skill
      3. `emit_<state>` handlers — ONLY if the failure_mode.condition
         text matches the error message (substring, case-insensitive).
         This prevents emit_* from catching generic Python exceptions
         the skill author never intended to handle.

    Match against the UNDERLYING error (error.__cause__), NOT the
    "tool <name> raised: ..." wrapper — otherwise a token in the tool
    name (e.g. 'rubric' in evaluate_artifact_rubric) spuriously matches
    a condition like 'Rubric file missing' and silently swallows a real
    error (Sprint 7 integrity fix).
    """
    underlying = error.__cause__ if error.__cause__ is not None else error
    err_text = str(underlying).lower()

    # Phase 1: retry handlers
    for fm in skill.failure_modes:
        handler = fm.handler.strip().lower()
        if handler.startswith("retry"):
            max_retries = max(1, fm.max_retries)
            for _attempt in range(max_retries):
                time.sleep(0.05)
                try:
                    return {"captures_override": _run_step(step, bindings, context, skill)}
                except Exception:  # noqa: BLE001
                    continue

    # Phase 2: fallback handlers
    for fm in skill.failure_modes:
        handler = fm.handler.strip()
        if handler.startswith("fallback:"):
            fallback_skill = handler.split(":", 1)[1].strip()
            sub = context.skill_registry.get(fallback_skill)
            if sub is not None:
                try:
                    sub_result = execute_skill(sub, _resolve_value(step.args, bindings), context)
                    if sub_result.ok:
                        return {"captures_override": sub_result.outputs}
                except Exception:  # noqa: BLE001
                    continue

    # Phase 3: emit_* — only if condition text matches the error
    for fm in skill.failure_modes:
        handler = fm.handler.strip()
        if handler.startswith("emit_"):
            cond_text = (fm.condition or "").lower()
            # Match if at least one significant keyword from condition
            # appears in error text. Skip tiny words.
            cond_tokens = [t for t in re.split(r"\W+", cond_text) if len(t) >= 4]
            if not cond_tokens or any(t in err_text for t in cond_tokens):
                state = handler[len("emit_"):]
                return {"captures_override": {"state": state, "reason": str(error)}}
    return None
