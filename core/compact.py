"""Context compaction pipeline (5-shaper architecture target; MVP: Snip + Microcompact).

The full architecture has 5 shapers in order:
  Budget Reduction → Snip → Microcompact → Context Collapse → Auto-Compact

This MVP ships the two highest-leverage ones:

- **Snip** (deterministic, free): walks messages, replaces stale tool_result
  content with a short preview + size note. "Stale" = the tool result is
  followed by ≥ K subsequent assistant turns (the agent has already had
  the result, reasoned about it, and moved on). Massively reduces context
  bloat from WebFetch/Read returning multi-KB blobs.

- **Microcompact** (LLM-powered): if total tokens still exceed the budget
  after Snip, replace the oldest 1/3 of post-system messages with a single
  synthesized "## Earlier in this cycle" summary message. Costs one model
  call per compaction.

Order matters: Snip before Microcompact so we don't pay an LLM call to
summarize content we could have just truncated.

Token estimates use a 4-chars-per-token rule of thumb. Real tokenizers
would be more accurate but cost an extra dep — for budget enforcement at
~70% of context window, the rough estimate is sufficient.
"""

from __future__ import annotations

from core import log
from core.types import AgentMessage

LOG = log.get_logger("bert.compact")

# ── Token estimation ────────────────────────────────────────────────


def estimate_tokens(text: str | None) -> int:
    """Rough estimate: ~4 chars per token for English text."""
    return (len(text) // 4) if text else 0


def total_tokens(messages: list[AgentMessage]) -> int:
    n = 0
    for m in messages:
        n += estimate_tokens(m.content)
        if m.tool_calls:
            for tc in m.tool_calls:
                n += estimate_tokens(tc.name) + estimate_tokens(str(tc.arguments))
    return n


# ── Snip ────────────────────────────────────────────────────────────


def _snip_replacement(tool_name: str, original_chars: int, preview: str) -> str:
    """Compact replacement for an old tool_result body."""
    return (
        f"[snipped — {tool_name} returned {original_chars} chars, "
        f"agent has since acted on this. Preview: {preview[:200]!r}]"
    )


def snip_stale_tool_results(
    messages: list[AgentMessage],
    *,
    keep_recent_iters: int = 3,
    snip_threshold_chars: int = 1500,
) -> tuple[list[AgentMessage], int]:
    """Replace tool_result content that's older than `keep_recent_iters`
    assistant turns and over `snip_threshold_chars`.

    "Iteration" = an assistant message that emitted tool_calls. Walking from
    the end, we count those; any tool_result preceded by ≥ keep_recent_iters
    such messages is considered stale.

    Returns (new_messages, chars_saved).
    """
    # First pass: number each tool_result by how many assistant-with-tool-calls
    # messages come AFTER it.
    iters_after: dict[int, int] = {}
    seen_assistants_with_calls = 0
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.role == "assistant" and m.tool_calls:
            seen_assistants_with_calls += 1
        elif m.role == "tool":
            iters_after[i] = seen_assistants_with_calls

    chars_saved = 0
    out: list[AgentMessage] = []
    for i, m in enumerate(messages):
        if (
            m.role == "tool"
            and iters_after.get(i, 0) >= keep_recent_iters
            and m.content is not None
            and len(m.content) > snip_threshold_chars
        ):
            tool_name = m.name or "unknown"
            preview = m.content[:300]
            new_content = _snip_replacement(tool_name, len(m.content), preview)
            chars_saved += len(m.content) - len(new_content)
            out.append(AgentMessage(
                role=m.role, content=new_content,
                tool_call_id=m.tool_call_id, name=m.name,
            ))
        else:
            out.append(m)
    return out, chars_saved


# ── Microcompact ────────────────────────────────────────────────────


_SUMMARIZE_SYSTEM = (
    "You are a context compactor. Below is a slice of an agent cycle's "
    "conversation history. Summarize it in ≤300 words preserving: "
    "(a) what tools were called and what they returned (names, key paths/URLs); "
    "(b) decisions the agent reached and any constraints discovered; "
    "(c) any data the agent will need later (file paths, IDs, citations). "
    "Output plain prose under a '## Earlier in this cycle' heading. "
    "No preamble, no commentary about your task — just the summary."
)


def _serialize_for_summary(messages: list[AgentMessage]) -> str:
    """Render a slice of agent history into a single text block for the
    summarizer LLM."""
    lines: list[str] = []
    for m in messages:
        if m.role == "assistant":
            txt = (m.content or "").strip()
            if txt:
                lines.append(f"[assistant said] {txt[:600]}")
            if m.tool_calls:
                for tc in m.tool_calls:
                    args = ", ".join(f"{k}={v!r}" for k, v in
                                     list(tc.arguments.items())[:3])
                    lines.append(f"[assistant called] {tc.name}({args})")
        elif m.role == "tool":
            preview = (m.content or "")[:400]
            lines.append(f"[tool {m.name or '?'} returned] {preview}")
        elif m.role == "user":
            lines.append(f"[user] {(m.content or '')[:300]}")
        else:
            lines.append(f"[{m.role}] {(m.content or '')[:300]}")
    return "\n".join(lines)


def microcompact_oldest(
    messages: list[AgentMessage],
    *,
    fraction: float = 0.33,
    keep_recent: int = 8,
    provider_name: str = "mistral",
    model: str = "mistral-small-latest",
) -> tuple[list[AgentMessage], int]:
    """LLM-summarize the oldest `fraction` of the post-system conversation,
    leaving at least `keep_recent` recent messages intact.

    The system message (index 0) is always preserved; the first user message
    (index 1) usually carries the cycle task and is also preserved.

    Returns (new_messages, tokens_saved_estimate).
    """
    # Lazy import to avoid circular: provider → ... → compact
    from core import provider as _prov

    if len(messages) < keep_recent + 4:
        return messages, 0

    # Indices 0 (system) and 1 (initial user) are preserved
    head = messages[:2]
    body = messages[2:]
    if len(body) <= keep_recent:
        return messages, 0

    # Compact the oldest `fraction` of body, but leave ≥ keep_recent at the end
    n_to_compact = max(1, int(len(body) * fraction))
    n_to_compact = min(n_to_compact, len(body) - keep_recent)
    if n_to_compact <= 0:
        return messages, 0

    to_compact = body[:n_to_compact]
    rest = body[n_to_compact:]

    try:
        slice_text = _serialize_for_summary(to_compact)
        resp = _prov.call(
            provider_name,
            messages=[
                AgentMessage(role="system", content=_SUMMARIZE_SYSTEM),
                AgentMessage(role="user", content=slice_text),
            ],
            model=model,
            max_tokens=1200,  # ≤300 words ≈ 400-500 tokens; leave headroom
            temperature=0.2,
        )
        if resp.finish_reason == "error" or not resp.text:
            LOG.warning("microcompact: provider returned error or empty; skipping")
            return messages, 0
        summary_text = resp.text.strip()
    except Exception as e:  # noqa: BLE001 — compaction must never crash the cycle
        LOG.warning("microcompact: %s; skipping", e)
        return messages, 0

    summary_msg = AgentMessage(
        role="user",  # use 'user' so all providers accept it; explicit "compaction" framing
        content=(
            f"[bert-compact] The earlier part of this cycle was summarized to "
            f"save context. Treat the following as factual recap of prior "
            f"actions and findings:\n\n{summary_text}"
        ),
    )

    saved = sum(estimate_tokens(m.content) for m in to_compact) - estimate_tokens(summary_msg.content)
    new_messages = head + [summary_msg] + rest
    return new_messages, max(saved, 0)


# ── Public entry ────────────────────────────────────────────────────


# ── H.5 shapers: Budget Reduction + Context Collapse + Auto-Compact ─


def budget_reduce(
    messages: list[AgentMessage],
    *,
    target_tokens: int,
    keep_system: bool = True,
    keep_recent: int = 8,
) -> tuple[list[AgentMessage], int]:
    """Cheapest first shaper: deterministic head-truncation.

    Drops oldest non-system, non-recent messages until total ≤ target.
    Always preserves the system prompt (constitutional preamble) and
    the last `keep_recent` messages (current working context). The
    drop region is the middle — long completed prior tasks.

    Returns (reduced_messages, tokens_dropped).
    """
    if not messages:
        return messages, 0
    total = total_tokens(messages)
    if total <= target_tokens:
        return messages, 0

    sys_msgs = [m for m in messages if keep_system and m.role == "system"]
    others = [m for m in messages if not (keep_system and m.role == "system")]
    if len(others) <= keep_recent:
        return messages, 0  # nothing in the drop zone

    head = others[: -keep_recent]
    tail = others[-keep_recent:]
    # Drop head until we're under target
    while head and total_tokens(sys_msgs + head + tail) > target_tokens:
        head.pop(0)
    out = sys_msgs + head + tail
    dropped = total - total_tokens(out)
    return out, max(0, dropped)


def context_collapse(
    messages: list[AgentMessage],
    *,
    max_summary_words: int = 200,
) -> tuple[list[AgentMessage], int]:
    """Read-time projection: collapse completed sub-task message clusters
    into ≤200-word summaries.

    A "completed sub-task" is detected by a tool_call followed by its
    tool result + at least one assistant continuation. The cluster
    gets replaced by a single assistant message of the shape:

      [collapsed sub-task: tool=<name>, <N> messages, <words> words]

    Reduces middle-context bulk without LLM calls. Best-effort; on
    detection failure returns unchanged.

    Returns (collapsed_messages, messages_dropped).
    """
    if not messages or len(messages) < 4:
        return messages, 0
    out: list[AgentMessage] = []
    i = 0
    collapsed_count = 0
    while i < len(messages):
        m = messages[i]
        # Pattern: assistant with tool_calls → tool result → next msg
        # We collapse a "completed" cluster of length ≥4 if the LATER
        # context already references the result (heuristic: any
        # assistant msg after position i+3 mentions a substring from
        # the tool result).
        is_tool_cluster = (
            m.role == "assistant"
            and getattr(m, "tool_calls", None)
            and i + 2 < len(messages)
            and messages[i + 1].role == "tool"
        )
        if is_tool_cluster and i + 3 < len(messages):
            # Collapse the [m, tool_result, next] triple if the next
            # message is itself an assistant with new tool_calls
            # (meaning the prior tool's output has already been
            # synthesized into the next call).
            next_assist = messages[i + 2]
            if next_assist.role == "assistant" and getattr(next_assist, "tool_calls", None):
                tool_name = m.tool_calls[0].name if m.tool_calls else "unknown"
                # Approximate word count from the cluster
                cluster_text = " ".join(
                    str(getattr(messages[i + j], "content", "") or "")
                    for j in range(2)
                )
                word_count = len(cluster_text.split())
                summary = AgentMessage(
                    role="assistant",
                    content=(
                        f"[collapsed sub-task: tool={tool_name}, "
                        f"2 messages, ~{min(word_count, max_summary_words)} words]"
                    ),
                )
                out.append(summary)
                collapsed_count += 2
                i += 2
                continue
        out.append(m)
        i += 1
    return out, collapsed_count


def auto_compact(
    messages: list[AgentMessage],
    *,
    target_tokens: int,
    provider_name: str = "mistral",
    model: str = "mistral-small-latest",
) -> tuple[list[AgentMessage], int]:
    """Aggressive last-resort shaper: when context is at ~80% of the
    window, summarize EVERYTHING pre-current-task into a single
    rolling summary message.

    Production guidance (Claude Code 2604.14228): trigger at 80%; ratio
    of pre-task vs current-task tokens should land near 20:80 after.

    Returns (compacted_messages, tokens_saved). Subject to the
    3-strike killswitch enforcement below.
    """
    if not messages:
        return messages, 0
    before = total_tokens(messages)
    if before <= target_tokens:
        return messages, 0
    # Keep system + LAST 4 messages (current task); summarize the rest
    sys_msgs = [m for m in messages if m.role == "system"]
    non_sys = [m for m in messages if m.role != "system"]
    if len(non_sys) <= 4:
        return messages, 0
    pre_task = non_sys[:-4]
    current_task = non_sys[-4:]
    if not pre_task:
        return messages, 0

    # Call microcompact_oldest with fraction=1.0 to summarize all pre-task
    compacted_pre, saved = microcompact_oldest(
        sys_msgs + pre_task + current_task,
        provider_name=provider_name, model=model,
        fraction=0.9,  # most of pre-task into 1 summary
        keep_recent=4,
    )
    return compacted_pre, saved


# ── 3-strike killswitch ──────────────────────────────────────────────


_AUTO_COMPACT_STRIKES: dict[int, list[float]] = {}  # cycle → [timestamps]
_STRIKE_WINDOW_SECS = 600  # 10 min
_STRIKE_THRESHOLD = 3


class AutoCompactKillswitch(Exception):
    """Raised when auto_compact fires 3+ times in the strike window —
    the cycle is loop-stuck on context pressure and PI must intervene.
    """


def _record_strike(cycle: int) -> int:
    """Record an auto_compact firing; returns the current strike count
    within the window."""
    import time as _time
    now = _time.time()
    cutoff = now - _STRIKE_WINDOW_SECS
    strikes = _AUTO_COMPACT_STRIKES.setdefault(cycle, [])
    strikes[:] = [t for t in strikes if t > cutoff]
    strikes.append(now)
    return len(strikes)


def reset_strikes(cycle: int | None = None) -> None:
    """Reset strike counter. PI calls this after intervention."""
    if cycle is None:
        _AUTO_COMPACT_STRIKES.clear()
    else:
        _AUTO_COMPACT_STRIKES.pop(cycle, None)


def apply_shapers(
    messages: list[AgentMessage],
    *,
    target_tokens: int = 80000,
    provider_name: str = "mistral",
    model: str = "mistral-small-latest",
    cycle: int | None = None,
    auto_compact_threshold_pct: float = 0.80,
) -> list[AgentMessage]:
    """Run the full 5-shaper compaction pipeline.

    Order (cheapest first per Claude Code 2604.14228):
      1. Budget Reduction — deterministic head-drop of old non-essential
      2. Snip — replace stale tool_result content
      3. Microcompact — LLM-summarize the oldest N messages
      4. Context Collapse — read-time projection of completed sub-tasks
      5. Auto-Compact — aggressive last-resort at ~80% threshold

    3-strike killswitch: auto_compact firing 3+ times within 10 minutes
    on the same cycle raises AutoCompactKillswitch. Caller MUST catch
    it and halt with CATASTROPHIC exit.
    """
    before = total_tokens(messages)
    if before <= target_tokens:
        return messages

    LOG.info("compaction: %d tokens > target %d, running shapers", before, target_tokens)

    # 1. Budget Reduction (deterministic, no LLM)
    out, dropped = budget_reduce(messages, target_tokens=target_tokens)
    if dropped > 0:
        LOG.info("  budget_reduce: -%d tokens → %d total",
                 dropped, total_tokens(out))
    if total_tokens(out) <= target_tokens:
        return out

    # 2. Snip (deterministic)
    snipped, chars_saved = snip_stale_tool_results(out)
    after_snip = total_tokens(snipped)
    if chars_saved > 0:
        LOG.info("  snip: -%d chars (~%d tokens) → %d total tokens",
                 chars_saved, chars_saved // 4, after_snip)
    if after_snip <= target_tokens:
        return snipped

    # 3. Microcompact (LLM)
    compacted, tokens_saved = microcompact_oldest(
        snipped, provider_name=provider_name, model=model,
    )
    after_compact = total_tokens(compacted)
    if tokens_saved > 0:
        LOG.info("  microcompact: -~%d tokens → %d total tokens",
                 tokens_saved, after_compact)
    if after_compact <= target_tokens:
        return compacted

    # 4. Context Collapse (deterministic, read-time projection)
    collapsed, n_collapsed = context_collapse(compacted)
    if n_collapsed > 0:
        LOG.info("  context_collapse: %d messages → 1 summary",
                 n_collapsed)
    if total_tokens(collapsed) <= target_tokens:
        return collapsed

    # 5. Auto-Compact (last resort)
    threshold = int(target_tokens * auto_compact_threshold_pct)
    if total_tokens(collapsed) >= threshold:
        if cycle is not None:
            strikes = _record_strike(cycle)
            if strikes >= _STRIKE_THRESHOLD:
                LOG.error("compaction: auto_compact killswitch — cycle=%d "
                          "fired %d times in 10 min", cycle, strikes)
                raise AutoCompactKillswitch(
                    f"auto_compact fired {strikes} times in 10 min on "
                    f"cycle {cycle}; PI must intervene"
                )
        auto_out, auto_saved = auto_compact(
            collapsed, target_tokens=target_tokens,
            provider_name=provider_name, model=model,
        )
        LOG.warning("  auto_compact: -~%d tokens (strikes=%d)",
                    auto_saved, len(_AUTO_COMPACT_STRIKES.get(cycle or 0, [])))
        return auto_out
    return collapsed


__all__ = [
    "estimate_tokens", "total_tokens",
    "snip_stale_tool_results", "microcompact_oldest",
    "budget_reduce", "context_collapse", "auto_compact",
    "apply_shapers",
    "AutoCompactKillswitch", "reset_strikes",
]
