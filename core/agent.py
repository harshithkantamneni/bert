"""Agent loop — the 9-step pipeline.

Settings → State init → Context assembly → 5 shapers → Model call →
Tool dispatch → Permission gate → Tool execution → Stop condition.

Wired end-to-end with NVIDIA llama-3.3-70b as the default substrate,
Read/Write/Bash + Spawn (subagent dispatch) + Memory tools as
built-in. 5-shaper compaction via `core/compact.py`. Permission gate
in `core/permission.py` with Telegram-approver hook for P-011
destructive-op approval (auto-registered when bot/approval is
available). Constitutional preamble prepended.

Runtime wiring integrates quota + watchdog + evaluator. Lifecycle
wiring connects hooks + indexer + brief_assembler + consolidator +
session.

Returns process exit code; writes session_exit.md as the FINAL action
in a finally: block (P-014 + cycle-1 trace gap fix).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import core.tools  # noqa: F401 — side-effect import: registers built-in tools
from core import (
    brief_assembler,
    config,
    consolidator,
    evaluator,
    hooks,
    indexer,
    log,
    observability,
    permission,
    provider,
    quota,
    tool_registry,
    watchdog,
)
from core import (
    session as _session_mod,
)
from core.types import (
    AgentMessage,
    ExitReason,
    PermissionMode,
    ProviderResponse,
    ToolCall,
    ToolResult,
)

LAB_ROOT = Path(__file__).resolve().parent.parent
SESSION_EXIT = LAB_ROOT / "state" / "session_exit.md"
SESSION_START = LAB_ROOT / "state" / "session_start.md"
LOG = log.get_logger("bert.agent")

# Destructive-op patterns + permission gate live in core/permission.py.
# Re-exported here for backwards compatibility with any external callers.
_is_destructive = permission.is_destructive
_permission_gate = permission.permission_gate

# Module-level indexer daemon singleton — started on the first non-subagent
# run_role and persists across cycles (no per-cycle stop). Real-time
# memory.db freshness via fs-watcher between cycles.
_INDEXER_DAEMON: indexer.IndexerDaemon | None = None


def _ensure_indexer_daemon_running() -> None:
    """Start the indexer fs-watcher daemon if not already running.
    Idempotent — a no-op after the first successful start. Wrapped in
    try/except so a daemon-start failure never breaks a cycle."""
    global _INDEXER_DAEMON
    if _INDEXER_DAEMON is not None:
        return
    try:
        d = indexer.IndexerDaemon()
        d.start()
        _INDEXER_DAEMON = d
        LOG.info("agent: indexer daemon autostarted (fs-watcher live)")
    except Exception as e:  # noqa: BLE001
        LOG.warning("agent: indexer daemon autostart failed (%s); "
                    "memory.db will only update on cycle close", e)


def _execute_tool(call: ToolCall) -> ToolResult:
    """Run the tool's handler; wrap result with P-016 sentinel."""
    td = tool_registry.get(call.name)
    if td is None:
        return ToolResult(
            tool_call_id=call.id,
            content=log.wrap_tool_output(f"[bert] unknown tool: {call.name}"),
            error="unknown_tool",
        )
    start = time.monotonic()
    try:
        result = td.handler(**call.arguments)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        if isinstance(result, dict):
            import json as _j
            content = _j.dumps(result, indent=2, default=str)
        else:
            content = str(result)
        # Inline absurd-size guard. Per-tool 8 KB cap; the 5-shaper
        # compaction in core/compact.py handles message-stream-level
        # context budgeting separately at the model-call boundary.
        truncated = False
        if len(content) > 8000:
            content = content[:6000] + "\n... (truncated " + str(len(content) - 6000) + " chars) ..."
            truncated = True
        return ToolResult(
            tool_call_id=call.id,
            content=log.wrap_tool_output(content),
            truncated=truncated,
            elapsed_ms=elapsed_ms,
        )
    except Exception as e:  # noqa: BLE001
        # A tool handler is invoked with model-generated arguments and must NEVER
        # crash the agent loop — any exception (AttributeError from a str passed
        # where a dict was expected, IndexError, etc.) degrades to a tool-error
        # the model sees and can recover from. (Was a narrow tuple that missed
        # AttributeError, so a buggy/misused tool killed the whole dispatch.)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return ToolResult(
            tool_call_id=call.id,
            content=log.wrap_tool_output(f"[bert] tool error: {type(e).__name__}: {e}"),
            error=str(e),
            elapsed_ms=elapsed_ms,
        )


def _load_system_prompt(role: str) -> str:
    """Constitutional preamble + role-specific procedural prompt."""
    constitutional = (LAB_ROOT / "memories" / "governance" / "constitutional.md").read_text()
    role_path = LAB_ROOT / "prompts" / f"{role}.md"
    role_prompt = role_path.read_text() if role_path.exists() else f"# {role.title()}\n\n(role prompt missing)"
    return constitutional + "\n\n---\n\n" + role_prompt


# How many times to nudge an agent that stopped without writing its required
# deliverable before giving up (bounded — avoids spinning to max_iterations).
_MAX_COMPLETION_NUDGES = 2


def _deliverable_failures(output_path: str, verification_spec: dict | None) -> list[str]:
    """Verification failures for the required deliverable (empty list = OK).

    When a `verification_spec` is given, runs the REAL verify_engine — so the
    in-loop completion nudge mirrors the post-dispatch grade and can tell the
    agent the exact gaps (min_chars, missing headers, missing citation). Falls
    back to a bare existence check otherwise. Resolved against the active lab."""
    try:
        from core.lab_context import get_active_lab_path
        p = Path(output_path)
        if not p.is_absolute():
            p = (get_active_lab_path() or LAB_ROOT) / p
    except Exception:  # noqa: BLE001
        return []
    if verification_spec:
        try:
            from core import verify_engine
            res = verify_engine.verify_artifact(verification_spec, p)
            return [] if res.ok else list(res.checks_failed)
        except Exception:  # noqa: BLE001
            pass  # fall through to existence check
    return [] if p.exists() else [f"deliverable not written: {output_path}"]


def run_role(role: str, *, cycle: int = 1, task: str | None = None,
             provider_name: str = "groq", model: str | None = None,
             max_iterations: int = 30, is_subagent: bool = False,
             telemetry_sink: dict | None = None,
             output_path: str | None = None,
             verification_spec: dict | None = None) -> int:
    """Run one cycle of the agent loop. Returns process exit code (0 OK, 1 error).

    When `is_subagent=True`, the heartbeat and session_exit.md writes are
    skipped — those belong to the parent (Director) cycle. Sub-agents write
    their own ResultPacket via Write to state/results/<role>_C<cycle>.json
    (Spawn reads + validates that file post-loop).

    When `telemetry_sink` is provided, it is mutated in-place to contain real
    accumulated telemetry (tokens_in, tokens_out, latency_secs, model_used,
    provider, retry_count, fallback_chain) when the cycle exits — needed by
    Spawn to overwrite the agent's hallucinated ResultPacket telemetry.
    """
    # Step 1: Settings
    cfg = config.load()
    LOG.info("cycle=%d role=%s%s provider=%s model=%s",
             cycle, role, " (subagent)" if is_subagent else "",
             provider_name, model or "default")

    # Heartbeat — only the top-level run owns these files
    if not is_subagent:
        SESSION_START.parent.mkdir(parents=True, exist_ok=True)
        SESSION_START.write_text(
            f"PID: <pid>\nStart: {datetime.now(UTC).isoformat()}\nCycle: {cycle}\nRole: {role}\n"
        )

    # P-011 destructive-op approver — best-effort wire to Telegram bot
    # if available. Idempotent (skips if already registered).
    permission.maybe_register_default_approver()

    # Telemetry accumulators
    tokens_in_total = 0
    tokens_out_total = 0
    last_model_seen = ""
    cycle_start_t = time.monotonic()

    # Register the session with the watchdog for hang
    # detection + holding-loop counting. Both top-level cycles and
    # sub-agent dispatches register; the holding-loop heuristic counts
    # all sessions equally.
    try:
        _watchdog_session_id = watchdog.record_start(
            pid=__import__("os").getpid(), role=role, cycle=cycle,
        )
    except Exception:  # noqa: BLE001
        _watchdog_session_id = 0  # advisory; never break the cycle

    # Top-level (non-subagent) cycles start the lab-wide infrastructure:
    #   - indexer fs-watcher daemon (real-time memory.db freshness)
    #   - context_brief.md regenerated for the Director's first read
    #   - append-only JSONL session log opened
    # Sub-agent dispatches inherit the parent's daemons and don't repeat.
    _session_handle = None
    if not is_subagent:
        _ensure_indexer_daemon_running()
        try:
            brief_assembler.assemble_brief()
        except Exception as e:  # noqa: BLE001
            LOG.warning("agent: brief_assembler skipped (%s)", e)
        try:
            _session_handle = _session_mod.start_session(role=role, cycle=cycle)
        except Exception as e:  # noqa: BLE001
            LOG.warning("agent: session.start_session failed (%s)", e)

    # Lifecycle event: RoleStart fires user-registered hooks. Hooks are
    # advisory; failures logged but never break the cycle.
    try:
        hooks.fire("RoleStart", {
            "role": role, "cycle": cycle, "is_subagent": is_subagent,
            "provider": provider_name, "model": model,
        })
    except Exception:  # noqa: BLE001
        pass

    exit_reason = ExitReason.GRACEFUL_CHECKPOINT
    try:
        # Step 2: State init
        system_prompt = _load_system_prompt(role)
        # Append cycle context
        cycle_context = f"\n\n---\n\n## Cycle context\n\n- cycle: {cycle}\n- role: {role}\n"
        if task:
            cycle_context += f"- inline task: {task}\n"

        # Step 3: Context assembly
        messages: list[AgentMessage] = [
            AgentMessage(role="system", content=system_prompt + cycle_context),
            AgentMessage(role="user", content=task or "Begin your cycle. Read state, then act."),
        ]

        # Step 4: shapers (lazy — only when cumulative context exceeds budget).
        # P-013 caps usage at cfg.context_usage_cap (70%) of the model window;
        # we approximate with a fixed token target until we have per-model windows.
        from core import compact as _compact
        target_tokens = int(80_000 * cfg.context_usage_cap)  # ~56k for 80k window models

        tools_schema = tool_registry.schemas_for_model()
        # Researcher / strategist / evaluator / implementer should NOT
        # be able to Spawn sub-agents — that's a director-altitude
        # capability. We saw a researcher emit `Spawn` with a literal
        # `provider: provider` placeholder and crash CATASTROPHICally.
        # Filter Spawn out of the tools list for non-director roles.
        if role not in ("director", "custom-director"):
            tools_schema = [t for t in tools_schema
                            if t.get("function", {}).get("name") != "Spawn"]

        from core import provider_fallback
        # Lanes already attempted this dispatch (the initial one + any failovers),
        # so cross-provider failover never loops back to a dead lane.
        _tried_lanes: set[tuple[str, str]] = {(provider_name, model)}
        _completion_nudges = 0  # times we've nudged a premature stop (see Step 9)

        for iteration in range(max_iterations):
            # Apply shapers BEFORE the model call so we send a compacted prompt
            messages = _compact.apply_shapers(
                messages,
                target_tokens=target_tokens,
                provider_name=provider_name,
                model=model,
                cycle=cycle,
            )
            LOG.info("iter=%d msg_count=%d est_tokens=%d",
                     iteration, len(messages), _compact.total_tokens(messages))

            # Pre-flight quota check (advisory; logged not blocked).
            try:
                _q_ok, _q_reason = quota.check_quota(provider_name)
                if not _q_ok:
                    LOG.warning("quota: %s pre-flight failed: %s (continuing — advisory)",
                                provider_name, _q_reason)
            except Exception:  # noqa: BLE001
                pass  # quota is advisory; never break the call path

            # Step 5: Model call
            resp: ProviderResponse = provider.call(
                provider_name,
                messages,
                tools=tools_schema,
                model=model,
                max_tokens=cfg.max_tokens_default,
            )
            tokens_in_total += resp.usage_prompt_tokens or 0
            tokens_out_total += resp.usage_completion_tokens or 0
            if resp.model:
                last_model_seen = resp.model
            log.append_session_event(cycle, {
                "iteration": iteration, "kind": "model_response",
                "finish_reason": resp.finish_reason,
                "tool_calls": len(resp.tool_calls),
                "tokens_in": resp.usage_prompt_tokens,
                "tokens_out": resp.usage_completion_tokens,
                "elapsed_ms": resp.elapsed_ms,
                "model": resp.model,
                "text_preview": (resp.text or "")[:500],
            })
            # Falsifier observability — dual-emit JSONL + OTel.
            try:
                observability.emit_model_call(
                    provider=provider_name,
                    model=resp.model or (model or ""),
                    input_tokens=resp.usage_prompt_tokens or 0,
                    output_tokens=resp.usage_completion_tokens or 0,
                    cached_tokens=resp.usage_cached_tokens or 0,
                    thinking_tokens=resp.usage_thinking_tokens or 0,
                    elapsed_ms=resp.elapsed_ms or 0,
                    role=role, cycle=cycle,
                )
            except Exception:  # noqa: BLE001
                pass  # observability is advisory; never break the loop
            # Lifecycle event: ModelCall hook fires user-registered scripts
            # with the call telemetry. PI can drop a script in
            # hooks/ModelCall/ to (e.g.) ping when latency spikes.
            try:
                hooks.fire("ModelCall", {
                    "provider": provider_name,
                    "model": resp.model or (model or ""),
                    "input_tokens": resp.usage_prompt_tokens or 0,
                    "output_tokens": resp.usage_completion_tokens or 0,
                    "elapsed_ms": resp.elapsed_ms or 0,
                    "finish_reason": resp.finish_reason,
                    "role": role, "cycle": cycle, "iteration": iteration,
                })
            except Exception:  # noqa: BLE001
                pass

            if resp.finish_reason == "error":
                # Cross-provider failover: a quota/rate-limit/too-large error
                # (gemini 429, groq 413) OR an unrunnable provider (the router
                # picked a lane the executor can't call, e.g. anthropic-cli host
                # tier -> "unknown provider") is survivable — retry the same step
                # on a different lane instead of dying CATASTROPHIC. provider.call
                # has already exhausted its within-provider retries by this point.
                if provider_fallback.is_failoverable_error(resp):
                    lane = provider_fallback.next_fallback_lane(exclude=_tried_lanes)
                    if lane is not None:
                        _tried_lanes.add(lane)
                        LOG.warning("provider %s/%s quota/limit-exhausted (%s); "
                                    "failing over to %s/%s",
                                    provider_name, model, (resp.text or "")[:120],
                                    lane[0], lane[1])
                        provider_name, model = lane
                        continue  # re-attempt the same messages on the fallback lane
                    LOG.error("provider error: %s (no fallback lane left)", resp.text)
                else:
                    LOG.error("provider error: %s", resp.text)
                exit_reason = ExitReason.CATASTROPHIC
                break

            # Append assistant message
            messages.append(AgentMessage(
                role="assistant", content=resp.text, tool_calls=resp.tool_calls,
            ))

            # Step 9: Stop condition
            if resp.finish_reason != "tool_use":
                # Completion check: the model says it's done — but if it owes a
                # required deliverable (output_path) it never wrote, it almost
                # certainly summarized in chat instead of Write-ing the file (a
                # common, model-agnostic failure). Nudge it to actually produce
                # the deliverable before accepting completion, bounded so a model
                # that keeps refusing doesn't spin to max_iterations.
                if output_path and _completion_nudges < _MAX_COMPLETION_NUDGES:
                    _failures = _deliverable_failures(output_path, verification_spec)
                    if _failures:
                        _completion_nudges += 1
                        LOG.warning("agent stopped but deliverable %s fails "
                                    "verification %s; nudging (attempt %d/%d)",
                                    output_path, _failures[:3], _completion_nudges,
                                    _MAX_COMPLETION_NUDGES)
                        messages.append(AgentMessage(
                            role="user",
                            content=(f"Your deliverable at `{output_path}` does NOT "
                                     f"yet meet its required verification — you are "
                                     f"graded on these and the dispatch fails without "
                                     f"them. Use the Write tool now to rewrite the "
                                     f"complete deliverable, fixing: "
                                     f"{'; '.join(_failures[:5])}."),
                        ))
                        continue
                LOG.info("model finished (reason=%s)", resp.finish_reason)
                break

            # Step 6: Tool dispatch
            for call in resp.tool_calls:
                LOG.info("tool_call: %s(%s)", call.name, list(call.arguments.keys()))

                # Lifecycle event: PreToolUse fires before permission gate.
                # PI can register hooks to (e.g.) audit calls, redact
                # arguments before logging, or veto via stderr signal.
                try:
                    hooks.fire("PreToolUse", {
                        "tool": call.name,
                        "arguments": call.arguments,
                        "role": role, "cycle": cycle, "iteration": iteration,
                    })
                except Exception:  # noqa: BLE001
                    pass

                # tool_call observability — fills the documented
                # event_class slot that was previously unwired.
                try:
                    observability.emit("tool_call", {
                        "tool": call.name,
                        "arguments_keys": list(call.arguments.keys()),
                        "role": role, "cycle": cycle, "iteration": iteration,
                    })
                except Exception:  # noqa: BLE001
                    pass

                # Step 7: Permission gate
                decision = _permission_gate(call, cfg.permission_mode if isinstance(cfg.permission_mode, PermissionMode) else PermissionMode.DEFAULT)
                log.append_session_event(cycle, {
                    "iteration": iteration, "kind": "permission_decision",
                    "tool": call.name, "allowed": decision.allowed,
                    "reason": decision.reason, "destructive": decision.is_destructive,
                })

                if not decision.allowed:
                    messages.append(AgentMessage(
                        role="tool",
                        content=log.wrap_tool_output(f"[bert] permission_denied: {decision.reason}"),
                        tool_call_id=call.id, name=call.name,
                    ))
                    continue

                # Step 8: Tool execution
                result = _execute_tool(call)
                log.append_session_event(cycle, {
                    "iteration": iteration, "kind": "tool_result",
                    "tool": call.name,
                    "elapsed_ms": result.elapsed_ms,
                    "error": result.error,
                    "truncated": result.truncated,
                    "content_preview": result.content[:500],
                })

                # Lifecycle event: PostToolUse fires after tool execution
                # with the result envelope (truncated content, error
                # status). PI hooks can act on tool failures.
                try:
                    hooks.fire("PostToolUse", {
                        "tool": call.name,
                        "elapsed_ms": result.elapsed_ms,
                        "error": result.error,
                        "truncated": result.truncated,
                        "role": role, "cycle": cycle, "iteration": iteration,
                    })
                except Exception:  # noqa: BLE001
                    pass

                # memory_write observability — wired here for any tool
                # whose primary effect is writing to memories/ or
                # findings/. Approximate: catch tool=Write with
                # file_path under those dirs. Per-tool emit is the
                # canonical surface; this fills the documented
                # event_class for now.
                if call.name == "Write" and isinstance(call.arguments, dict):
                    fp = str(call.arguments.get("file_path") or "")
                    if "/memories/" in fp or "/findings/" in fp or fp.startswith("memories/") or fp.startswith("findings/"):
                        try:
                            observability.emit("memory_write", {
                                "path": fp, "role": role, "cycle": cycle,
                                "elapsed_ms": result.elapsed_ms,
                                "truncated": result.truncated,
                            })
                        except Exception:  # noqa: BLE001
                            pass

                messages.append(AgentMessage(
                    role="tool", content=result.content,
                    tool_call_id=call.id, name=call.name,
                ))

        else:
            # Loop exhausted without breaking
            LOG.warning("max_iterations=%d reached without stop_reason!=tool_use", max_iterations)
            exit_reason = ExitReason.CONTEXT_FULL

    except (KeyboardInterrupt, SystemExit):
        exit_reason = ExitReason.PIVOT
        raise
    except Exception as e:  # noqa: BLE001
        LOG.exception("agent loop crashed: %s", e)
        exit_reason = ExitReason.CATASTROPHIC
    finally:
        # Close out the watchdog session record.
        if _watchdog_session_id:
            try:
                watchdog.record_end(_watchdog_session_id, exit_reason=exit_reason.value)
            except Exception:  # noqa: BLE001
                pass

        # P-014 + cycle-1 trace gap fix: session_exit.md is the FINAL action,
        # in a finally: block so it fires even on exception.
        # Sub-agents skip this — that file belongs to the parent cycle.
        if not is_subagent:
            SESSION_EXIT.parent.mkdir(parents=True, exist_ok=True)
            SESSION_EXIT.write_text(
                f"{exit_reason.value}\n\n"
                f"# Cycle {cycle} ({role}) exited\n\n"
                f"Timestamp: {datetime.now(UTC).isoformat()}\n"
            )

            # End-of-cycle Python-side evaluator. Writes the
            # mechanical-checks report alongside the agent-side evaluator
            # findings; runner reads both for the GRACEFUL_CHECKPOINT gate.
            try:
                _eval = evaluator.evaluate_cycle(cycle)
                _eval_path = LAB_ROOT / "findings" / f"evaluator_python_C{cycle}.md"
                _eval_path.parent.mkdir(parents=True, exist_ok=True)
                _eval_path.write_text(evaluator.render_report(_eval))
                if not evaluator.gates_graceful_exit(_eval):
                    LOG.warning(
                        "evaluator: cycle %d FAILED %d mechanical checks — "
                        "GRACEFUL_CHECKPOINT exit blocked",
                        cycle, _eval.fail_count,
                    )
                # Lifecycle event: EvaluatorVerdict fires user-registered
                # hooks with the gate result + fail count. PI can hook
                # this to (e.g.) auto-pause on repeated mechanical FAIL.
                try:
                    hooks.fire("EvaluatorVerdict", {
                        "cycle": cycle, "role": role,
                        "overall": _eval.overall.value,
                        "fail_count": _eval.fail_count,
                        "gates_graceful_exit": evaluator.gates_graceful_exit(_eval),
                    })
                except Exception:  # noqa: BLE001
                    pass
            except Exception as e:  # noqa: BLE001
                LOG.warning("evaluator: skipped (error: %s)", e)

            # KM agent runs after the evaluator gates. The
            # consolidator's should_run() check throttles automatically
            # so calling it every cycle is cheap.
            try:
                consolidator.consolidate(cycle=cycle)
            except Exception as e:  # noqa: BLE001
                LOG.warning("consolidator: skipped (error: %s)", e)
        # Real telemetry — overwrites whatever the model hallucinated about itself.
        if telemetry_sink is not None:
            telemetry_sink.update({
                "tokens_in": tokens_in_total,
                "tokens_out": tokens_out_total,
                "latency_secs": round(time.monotonic() - cycle_start_t, 2),
                "model_used": last_model_seen or (model or ""),
                "provider": provider_name,
                "retry_count": 0,           # provider.call retries internally; not surfaced to caller
                "fallback_chain": [],       # core/router.py populates when smart-routing fires
            })
        LOG.info("cycle=%d role=%s exit_reason=%s tokens=%d/%d",
                 cycle, role, exit_reason.value,
                 tokens_in_total, tokens_out_total)

        # Close out the JSONL session log (top-level only; sub-agents
        # don't own the session).
        if _session_handle is not None:
            try:
                _session_mod.end_session(_session_handle, exit_reason=exit_reason.value)
            except Exception as e:  # noqa: BLE001
                LOG.warning("agent: session.end_session failed (%s)", e)

        # Lifecycle event: RoleEnd fires last. PI can hook this to
        # (e.g.) push status to Slack, archive run artifacts, etc.
        try:
            hooks.fire("RoleEnd", {
                "role": role, "cycle": cycle, "is_subagent": is_subagent,
                "exit_reason": exit_reason.value,
                "tokens_in": tokens_in_total, "tokens_out": tokens_out_total,
                "elapsed_secs": round(time.monotonic() - cycle_start_t, 2),
            })
        except Exception:  # noqa: BLE001
            pass

    # Process exit code: VICTORY=0, CATASTROPHIC=1, others=0 (runner restarts)
    return 1 if exit_reason == ExitReason.CATASTROPHIC else 0


__all__ = ["run_role"]
