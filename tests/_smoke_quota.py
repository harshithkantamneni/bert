"""Smoke test for core/quota.py — RPM/RPD/daily-tokens windowed checks.

Tests:
  1. Empty DB → check_quota returns ok for every provider
  2. record_call within RPM window → still ok until limit hit
  3. record_call past RPM ceiling → check_quota returns False with reason
  4. context_max gate fires before any DB query
  5. record_probe → stats includes probe counts
  6. Two providers' counters are independent
  7. prune_old removes old rows

Run: `.venv/bin/python tests/_smoke_quota.py`
Exit 0 = pass; non-zero = fail.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

# Use a temp DB so the smoke doesn't pollute lab state.
import tempfile

TMP_DB = Path(tempfile.mkdtemp(prefix="bert_quota_smoke_")) / "quota.db"

from core import quota as quota_mod  # noqa: E402

quota_mod.QUOTA_DB = TMP_DB


def _reset() -> None:
    # Re-pin module path each test; sibling tests (e.g.
    # _smoke_h4_wiring.test_provider_call_records_quota_on_success)
    # mutate quota_mod.QUOTA_DB to their own tempdir, which would
    # otherwise poison subsequent assertions here.
    quota_mod.QUOTA_DB = TMP_DB
    if TMP_DB.exists():
        TMP_DB.unlink()


def test_empty_db_allows_all() -> None:
    _reset()
    for p in ["nvidia", "cerebras", "groq", "gemini", "mistral", "openrouter"]:
        ok, reason = quota_mod.check_quota(p)
        assert ok, f"{p} should be ok on empty db; got reason={reason}"


def test_rpm_ceiling() -> None:
    _reset()
    # cerebras has rpm=30; record 30 then expect block
    for _ in range(30):
        quota_mod.record_call("cerebras", prompt_tokens=10, completion_tokens=5)
    ok, reason = quota_mod.check_quota("cerebras")
    assert not ok, f"30 calls should hit cerebras rpm=30 ceiling; got ok={ok}"
    assert "rpm" in reason, f"reason should mention rpm; got {reason!r}"


def test_context_max_pre_db() -> None:
    _reset()
    # cerebras context_max=8192; sending 8193 prompt tokens must fail
    ok, reason = quota_mod.check_quota("cerebras", prompt_tokens=8193)
    assert not ok
    assert "context_max" in reason


def test_daily_tokens_ceiling() -> None:
    _reset()
    # cerebras daily_tokens=1_000_000; one massive event should exhaust
    quota_mod.record_call("cerebras", prompt_tokens=999_900, completion_tokens=200)
    ok, reason = quota_mod.check_quota("cerebras", prompt_tokens=10)
    assert not ok, f"daily token cap should fire; got ok={ok} reason={reason}"
    assert "daily_tokens" in reason


def test_provider_isolation() -> None:
    _reset()
    for _ in range(30):
        quota_mod.record_call("groq", prompt_tokens=1, completion_tokens=1)
    # groq is at rpm cap, but nvidia must still be free
    nvidia_ok, _ = quota_mod.check_quota("nvidia")
    assert nvidia_ok, "nvidia should not be affected by groq saturation"
    groq_ok, groq_reason = quota_mod.check_quota("groq")
    assert not groq_ok
    assert "rpm" in groq_reason


def test_probe_recorded_in_stats() -> None:
    _reset()
    quota_mod.record_probe("mistral", ok=True, latency_ms=120)
    quota_mod.record_probe("mistral", ok=False, latency_ms=5000, error="timeout")
    s = quota_mod.stats("mistral")
    assert "mistral" in s
    assert s["mistral"]["probes_24h_total"] == 2
    assert s["mistral"]["probes_24h_ok"] == 1


def test_cached_tokens_recorded_and_surfaced() -> None:
    """cached_tokens flow record_call → stats."""
    _reset()
    # Simulate two Gemini dispatches: first cold (cached=0), second warm (cached=900).
    quota_mod.record_call("gemini", prompt_tokens=1000, completion_tokens=100, cached_tokens=0)
    quota_mod.record_call("gemini", prompt_tokens=1000, completion_tokens=100, cached_tokens=900)
    s = quota_mod.stats("gemini")
    assert s["gemini"]["cached_tokens_24h"] == 900
    # cache hit % = 900 / 2000 prompt tokens = 45.0
    assert s["gemini"]["cache_hit_pct_24h"] == 45.0


def test_cache_hit_zero_when_no_prompt_tokens() -> None:
    _reset()
    # Edge case: cached_tokens defaults to 0; cache_hit_pct should be 0.0 not NaN.
    quota_mod.record_call("groq", prompt_tokens=0, completion_tokens=5)
    s = quota_mod.stats("groq")
    assert s["groq"]["cache_hit_pct_24h"] == 0.0
    assert s["groq"]["cached_tokens_24h"] == 0


def test_prune_drops_old_rows() -> None:
    _reset()
    # Insert a row, then mutate ts in-place to simulate age
    quota_mod.record_call("openrouter", prompt_tokens=1, completion_tokens=1)
    import sqlite3
    with sqlite3.connect(TMP_DB) as conn:
        conn.execute("UPDATE events SET ts = ? WHERE provider='openrouter'",
                     (time.time() - 60 * 86400,))  # 60 days old
        conn.commit()
    deleted = quota_mod.prune_old(days=30)
    assert deleted >= 1, f"expected at least 1 row deleted; got {deleted}"


def main() -> int:
    tests = [
        test_empty_db_allows_all,
        test_rpm_ceiling,
        test_context_max_pre_db,
        test_daily_tokens_ceiling,
        test_provider_isolation,
        test_probe_recorded_in_stats,
        test_cached_tokens_recorded_and_surfaced,
        test_cache_hit_zero_when_no_prompt_tokens,
        test_prune_drops_old_rows,
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
