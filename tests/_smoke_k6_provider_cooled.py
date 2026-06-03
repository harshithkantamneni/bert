"""Smoke test for K.6: cooled_until_ts field in /api/quota response.

K.6 shipped without a dedicated smoke. This covers:
- core.quota.stats() includes cooled_until_ts in each provider dict
- Field is None when no recent 429
- Field is set to err_ts + 120s when there's a recent 429
- Cooldown expires (returns None) when the 429 is old enough
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import quota


def _isolate_quota_db() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="bert_k6_")) / "quota.db"
    quota.QUOTA_DB = tmp
    return tmp


def _insert_event(provider: str, status_code: int, ts: float | None = None) -> None:
    if ts is None:
        ts = time.time()
    with sqlite3.connect(quota.QUOTA_DB) as conn:
        # Match schema from quota.record_call
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
                provider TEXT,
                ts REAL,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                cached_tokens INTEGER,
                status_code INTEGER,
                latency_ms INTEGER
            )
        """)
        conn.execute("""
            INSERT INTO events (provider, ts, prompt_tokens, completion_tokens,
                                cached_tokens, status_code, latency_ms)
            VALUES (?, ?, 100, 50, 0, ?, 200)
        """, (provider, ts, status_code))
        conn.commit()


def test_cooled_until_ts_field_present() -> None:
    _isolate_quota_db()
    _insert_event("groq", 200)
    s = quota.stats()
    assert "groq" in s
    assert "cooled_until_ts" in s["groq"], \
        "K.6 — cooled_until_ts must be present in stats() output"


def test_no_recent_429_returns_null() -> None:
    _isolate_quota_db()
    _insert_event("groq", 200)
    s = quota.stats()
    assert s["groq"]["cooled_until_ts"] is None


def test_recent_429_sets_cooldown_to_ts_plus_120() -> None:
    _isolate_quota_db()
    err_ts = time.time() - 30  # 30s ago
    _insert_event("groq", 429, ts=err_ts)
    s = quota.stats()
    cooled_until = s["groq"]["cooled_until_ts"]
    assert cooled_until is not None
    # Should be err_ts + 120 (within float-tolerance)
    assert abs(cooled_until - (err_ts + 120)) < 0.1


def test_old_429_does_not_set_cooldown() -> None:
    """A 429 from 5 minutes ago is past the 120s window — no cooldown."""
    _isolate_quota_db()
    err_ts = time.time() - 300  # 5 min ago
    _insert_event("groq", 429, ts=err_ts)
    s = quota.stats()
    assert s["groq"]["cooled_until_ts"] is None, \
        "5-min-old 429 shouldn't trigger cooldown (window is 120s)"


def test_cooldown_independent_per_provider() -> None:
    """A 429 on groq shouldn't cool nvidia."""
    _isolate_quota_db()
    _insert_event("groq", 429, ts=time.time() - 30)
    _insert_event("nvidia", 200)
    s = quota.stats()
    assert s["groq"]["cooled_until_ts"] is not None
    assert s["nvidia"]["cooled_until_ts"] is None


def test_cooldown_uses_most_recent_429() -> None:
    """If there are multiple 429s, the LATEST one determines cooldown."""
    _isolate_quota_db()
    _insert_event("groq", 429, ts=time.time() - 200)  # too old
    most_recent = time.time() - 10
    _insert_event("groq", 429, ts=most_recent)
    s = quota.stats()
    cooled_until = s["groq"]["cooled_until_ts"]
    assert cooled_until is not None
    assert abs(cooled_until - (most_recent + 120)) < 0.1


def main() -> int:
    tests = [
        test_cooled_until_ts_field_present,
        test_no_recent_429_returns_null,
        test_recent_429_sets_cooldown_to_ts_plus_120,
        test_old_429_does_not_set_cooldown,
        test_cooldown_independent_per_provider,
        test_cooldown_uses_most_recent_429,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
