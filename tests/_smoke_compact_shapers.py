"""Smoke test for H.5 — 5-shaper compaction + 3-strike killswitch."""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import compact  # noqa: E402
from core.types import AgentMessage, ToolCall  # noqa: E402


def _msg(role: str, content: str = "x" * 200,
         tool_calls: list[ToolCall] | None = None) -> AgentMessage:
    return AgentMessage(role=role, content=content,
                          tool_calls=tool_calls or [])


def test_budget_reduce_under_target_is_noop() -> None:
    msgs = [_msg("system"), _msg("user", "y" * 100)]
    out, dropped = compact.budget_reduce(msgs, target_tokens=10000)
    assert out == msgs
    assert dropped == 0


def test_budget_reduce_drops_middle_keeps_system_and_tail() -> None:
    sys_m = _msg("system", "S")
    # 12 user/assistant messages of ~50 tokens each = ~600 tokens
    middle = [_msg("user", "m" * 200) for _ in range(12)]
    msgs = [sys_m] + middle
    # Target 200 tokens — must drop most of middle
    out, dropped = compact.budget_reduce(msgs, target_tokens=200, keep_recent=4)
    # System preserved
    assert out[0].role == "system"
    # Last 4 messages preserved
    assert len(out) >= 1 + 4  # system + 4 tail
    # Some tokens dropped
    assert dropped > 0


def test_context_collapse_with_tool_cluster() -> None:
    tc = ToolCall(id="tc-1", name="Read", arguments={})
    msgs = [
        _msg("system", "s"),
        _msg("assistant", "calling Read", tool_calls=[tc]),
        _msg("tool", "read result body"),
        # Followed by ANOTHER assistant with tool_calls — signal that
        # the prior result has been synthesized
        _msg("assistant", "next step", tool_calls=[ToolCall(id="tc-2", name="Write", arguments={})]),
        _msg("tool", "write result"),
    ]
    out, dropped = compact.context_collapse(msgs)
    assert dropped >= 2
    # A collapsed-summary message should appear
    summaries = [m for m in out if m.content and "collapsed sub-task" in (m.content or "")]
    assert len(summaries) >= 1


def test_context_collapse_short_input_noop() -> None:
    msgs = [_msg("system"), _msg("user")]
    out, dropped = compact.context_collapse(msgs)
    assert dropped == 0
    assert out == msgs


def test_3_strike_killswitch_fires() -> None:
    compact.reset_strikes()
    cycle = 999
    # Fire 3 strikes within 10 min
    s1 = compact._record_strike(cycle)
    s2 = compact._record_strike(cycle)
    s3 = compact._record_strike(cycle)
    assert s1 == 1 and s2 == 2 and s3 == 3
    # Strike count for the cycle should hit threshold
    assert s3 >= compact._STRIKE_THRESHOLD


def test_reset_strikes_clears() -> None:
    cycle = 42
    compact._record_strike(cycle)
    compact._record_strike(cycle)
    compact.reset_strikes(cycle)
    s = compact._record_strike(cycle)
    assert s == 1  # reset, then 1 new


def test_apply_shapers_under_budget_is_noop() -> None:
    """Tokens already under target → return messages unchanged."""
    msgs = [_msg("system"), _msg("user", "tiny")]
    out = compact.apply_shapers(msgs, target_tokens=10000)
    assert out == msgs


def test_apply_shapers_triggers_budget_reduce_first() -> None:
    """Many small messages → Budget Reduction first (cheapest)."""
    sys_m = _msg("system", "S")
    middle = [_msg("user", "m" * 400) for _ in range(20)]  # ~2000 tokens
    msgs = [sys_m] + middle
    # Pick a tight target so budget_reduce must trim
    out = compact.apply_shapers(msgs, target_tokens=500,
                                  provider_name="mistral",
                                  model="mistral-small-latest")
    # Must be shorter
    assert len(out) < len(msgs)
    # System preserved
    assert out[0].role == "system"


def test_killswitch_path_documented() -> None:
    """Verify the AutoCompactKillswitch class is wired + reset_strikes
    exists. The full integration test would need an LLM call so we
    only verify the surface here."""
    assert hasattr(compact, "AutoCompactKillswitch")
    assert callable(compact.reset_strikes)
    assert issubclass(compact.AutoCompactKillswitch, Exception)


def main() -> int:
    tests = [
        test_budget_reduce_under_target_is_noop,
        test_budget_reduce_drops_middle_keeps_system_and_tail,
        test_context_collapse_with_tool_cluster,
        test_context_collapse_short_input_noop,
        test_3_strike_killswitch_fires,
        test_reset_strikes_clears,
        test_apply_shapers_under_budget_is_noop,
        test_apply_shapers_triggers_budget_reduce_first,
        test_killswitch_path_documented,
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
