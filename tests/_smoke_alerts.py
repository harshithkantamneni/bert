"""Smoke test for bot/alerts.py — structured alerts + rate limit + digest."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from bot import alerts  # noqa: E402


def _isolate() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="bert_alerts_")) / "alert_state.db"
    alerts.ALERT_DB = tmp
    return tmp


class _Captor:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def __call__(self, text: str) -> None:
        self.sent.append(text)


def test_send_alert_emits_text() -> None:
    _isolate()
    captor = _Captor()
    r = alerts.send_alert("holding_loop", priority="warn",
                          summary="5 short cycles in 2h",
                          short_cycles=5, window_mins=120,
                          send_fn=captor)
    assert r["sent"] is True
    assert len(captor.sent) == 1
    text = captor.sent[0]
    assert "HOLDING_LOOP" in text
    assert "5 short cycles" in text
    assert "short_cycles: 5" in text


def test_rate_limit_suppresses_second_alert_within_window() -> None:
    _isolate()
    captor = _Captor()
    r1 = alerts.send_alert("holding_loop", priority="warn",
                            summary="first", send_fn=captor)
    r2 = alerts.send_alert("holding_loop", priority="warn",
                            summary="second", send_fn=captor)
    assert r1["sent"] is True
    assert r2["sent"] is False
    assert "rate-limited" in r2["reason"]
    assert len(captor.sent) == 1


def test_critical_priority_bypasses_rate_limit() -> None:
    _isolate()
    captor = _Captor()
    alerts.send_alert("forgery", priority="critical",
                       summary="first", send_fn=captor)
    r2 = alerts.send_alert("forgery", priority="critical",
                            summary="second", send_fn=captor)
    assert r2["sent"] is True
    assert len(captor.sent) == 2


def test_explicit_cooldown_zero_bypasses_rate_limit() -> None:
    _isolate()
    captor = _Captor()
    alerts.send_alert("holding_loop", summary="first", send_fn=captor)
    r2 = alerts.send_alert("holding_loop", summary="second",
                            cooldown_secs=0, send_fn=captor)
    assert r2["sent"] is True
    assert len(captor.sent) == 2


def test_format_alert_includes_priority_icon() -> None:
    a_info = alerts.Alert("test", "info", "x")
    a_warn = alerts.Alert("test", "warn", "x")
    a_crit = alerts.Alert("test", "critical", "x")
    assert "ℹ" in alerts.format_alert(a_info)
    assert "⚠" in alerts.format_alert(a_warn)
    assert "🛑" in alerts.format_alert(a_crit)


def test_daily_summary_returns_string() -> None:
    text = alerts.daily_summary()
    assert isinstance(text, str)
    assert "bert" in text.lower()
    assert "seasoning unrevived" in text


def test_send_daily_summary_skips_rate_limit() -> None:
    _isolate()
    captor = _Captor()
    r1 = alerts.send_daily_summary(send_fn=captor)
    r2 = alerts.send_daily_summary(send_fn=captor)
    assert r1["sent"] is True
    assert r2["sent"] is True
    assert len(captor.sent) == 2


def main() -> int:
    tests = [
        test_send_alert_emits_text,
        test_rate_limit_suppresses_second_alert_within_window,
        test_critical_priority_bypasses_rate_limit,
        test_explicit_cooldown_zero_bypasses_rate_limit,
        test_format_alert_includes_priority_icon,
        test_daily_summary_returns_string,
        test_send_daily_summary_skips_rate_limit,
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
