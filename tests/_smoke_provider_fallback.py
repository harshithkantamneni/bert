"""Smoke + TDD: core/provider_fallback.py — cross-provider failover on quota.

Surfaced while generating organic data: provider.call retries ONE provider on
429/413 then returns finish_reason="error", and the agent loop turns that
straight into CATASTROPHIC — no cross-provider fallback. So a gemini 429 (quota
exhausted) or a groq 413 (prompt > 12K TPM) kills the dispatch even though other
lanes (nvidia, cerebras, ...) are available. This module decides when an error is
a quota/limit exhaustion and picks the next available fallback lane.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import provider_fallback as pf  # noqa: E402


def _resp(text, finish="error"):
    return SimpleNamespace(text=text, finish_reason=finish)


def test_is_failoverable_error():
    # quota / rate-limit / too-large -> another lane might serve it
    assert pf.is_failoverable_error(_resp("[bert] rate-limited (429) by gemini after 5 attempts")) is True
    assert pf.is_failoverable_error(_resp("groq HTTP 413: Request too large ... tokens per minute")) is True
    assert pf.is_failoverable_error(_resp("RESOURCE_EXHAUSTED: quota")) is True
    # unrunnable provider (router picked a lane the executor can't call, e.g.
    # anthropic-cli host tier) / missing credential -> another lane can
    assert pf.is_failoverable_error(_resp("[bert] unknown provider: anthropic-cli")) is True
    assert pf.is_failoverable_error(_resp("[bert] missing credential GROQ_API_KEY for groq")) is True
    # a genuine content/parse error is NOT failoverable (another lane won't help)
    assert pf.is_failoverable_error(_resp("[bert] failed to parse response")) is False
    # a successful response is not an error at all
    assert pf.is_failoverable_error(_resp("hello", finish="stop")) is False


def test_next_fallback_lane_picks_first_available(monkeypatch):
    # only nvidia + cerebras have credentials
    monkeypatch.setattr(pf, "_has_credential", lambda p: p in {"nvidia", "cerebras"})
    lane = pf.next_fallback_lane(exclude=set())
    assert lane is not None and lane[0] in {"nvidia", "cerebras"}


def test_next_fallback_lane_excludes_tried(monkeypatch):
    monkeypatch.setattr(pf, "_has_credential", lambda p: True)
    first = pf.next_fallback_lane(exclude=set())
    second = pf.next_fallback_lane(exclude={first})
    assert second is not None and second != first


def test_next_fallback_lane_none_when_no_credentials(monkeypatch):
    monkeypatch.setattr(pf, "_has_credential", lambda p: False)
    assert pf.next_fallback_lane(exclude=set()) is None


def test_next_fallback_lane_none_when_all_excluded(monkeypatch):
    monkeypatch.setattr(pf, "_has_credential", lambda p: True)
    assert pf.next_fallback_lane(exclude=set(pf.FALLBACK_LANES)) is None


def test_fallback_order_prefers_large_tpm_over_groq(monkeypatch):
    # groq (12K TPM, 413s on big prompts) must be LAST among the cloud lanes
    monkeypatch.setattr(pf, "_has_credential", lambda p: True)
    order = [lane[0] for lane in pf.FALLBACK_LANES]
    assert order.index("groq") == len(order) - 1 or order.index("groq") > order.index("nvidia")


class _MP:
    def __init__(self):
        self._u = []

    def setattr(self, obj, name, val):
        self._u.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    def undo(self):
        for o, n, v in reversed(self._u):
            setattr(o, n, v)
        self._u.clear()


def main() -> int:
    import inspect
    tests = [
        test_is_failoverable_error,
        test_next_fallback_lane_picks_first_available,
        test_next_fallback_lane_excludes_tried,
        test_next_fallback_lane_none_when_no_credentials,
        test_next_fallback_lane_none_when_all_excluded,
        test_fallback_order_prefers_large_tpm_over_groq,
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
