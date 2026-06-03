"""Smoke test for core/mcp_replay.py + MCPServer integration."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import mcp_replay, mcp_server  # noqa: E402


def _isolate() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_replay_")) / "mcp_nonces.db"
    mcp_replay.DB_PATH = tmp


def test_no_nonce_not_replay() -> None:
    _isolate()
    assert mcp_replay.is_replay("", "any_tool") is False
    assert mcp_replay.is_replay(None, "any_tool") is False  # type: ignore[arg-type]


def test_record_then_is_replay() -> None:
    _isolate()
    assert mcp_replay.is_replay("abc123", "echo") is False
    mcp_replay.record_nonce("abc123", "echo")
    assert mcp_replay.is_replay("abc123", "echo") is True


def test_different_tools_independent() -> None:
    """Same nonce on different tools should NOT collide."""
    _isolate()
    mcp_replay.record_nonce("nonce-x", "echo")
    assert mcp_replay.is_replay("nonce-x", "echo") is True
    assert mcp_replay.is_replay("nonce-x", "other_tool") is False


def test_expired_nonces_pruned() -> None:
    _isolate()
    mcp_replay.record_nonce("old-nonce", "echo")
    # Force old timestamp
    import sqlite3
    with sqlite3.connect(mcp_replay.DB_PATH) as conn:
        conn.execute("UPDATE nonces SET ts = ? WHERE nonce='old-nonce'",
                      (time.time() - 7200,))  # 2h ago
        conn.commit()
    # Default 1h window — should be considered fresh-eligible (not replay)
    assert mcp_replay.is_replay("old-nonce", "echo") is False


def test_stats_returns_counts() -> None:
    _isolate()
    mcp_replay.record_nonce("n1", "echo")
    mcp_replay.record_nonce("n2", "echo")
    mcp_replay.record_nonce("n3", "search")
    s = mcp_replay.stats()
    assert s["active_nonces"] == 3
    assert s["by_tool"]["echo"] == 2
    assert s["by_tool"]["search"] == 1


def test_clear_drops_all() -> None:
    _isolate()
    mcp_replay.record_nonce("n1", "echo")
    mcp_replay.record_nonce("n2", "echo")
    removed = mcp_replay.clear()
    assert removed == 2
    assert mcp_replay.stats()["active_nonces"] == 0


# ── Integration: MCPServer.handle() catches replays ─────────────────


def test_mcp_server_accepts_first_nonce() -> None:
    _isolate()
    srv = mcp_server.make_echo_server()
    resp = srv.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {
            "name": "echo",
            "arguments": {"text": "hi"},
            "_meta": {"nonce": "fresh-nonce-001"},
        },
    })
    assert "result" in resp
    assert "error" not in resp


def test_mcp_server_rejects_replayed_nonce() -> None:
    _isolate()
    srv = mcp_server.make_echo_server()
    params = {
        "name": "echo",
        "arguments": {"text": "hi"},
        "_meta": {"nonce": "replay-test-001"},
    }
    # First call accepted
    r1 = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params})
    assert "result" in r1
    # Second call with same nonce → rejected with REPLAY_REJECTED code
    r2 = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": params})
    assert "error" in r2
    assert r2["error"]["code"] == mcp_server.REPLAY_REJECTED
    assert "nonce" in r2["error"]["message"].lower()


def test_mcp_server_no_nonce_works_unchanged() -> None:
    """Backward compatibility: calls without a nonce continue to work."""
    _isolate()
    srv = mcp_server.make_echo_server()
    resp = srv.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "echo", "arguments": {"text": "hi"}},
    })
    assert "result" in resp


def main() -> int:
    tests = [
        test_no_nonce_not_replay,
        test_record_then_is_replay,
        test_different_tools_independent,
        test_expired_nonces_pruned,
        test_stats_returns_counts,
        test_clear_drops_all,
        test_mcp_server_accepts_first_nonce,
        test_mcp_server_rejects_replayed_nonce,
        test_mcp_server_no_nonce_works_unchanged,
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
