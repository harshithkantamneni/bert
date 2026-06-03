"""Smoke test for bot/approval.py — destructive-op approval round-trip.

Per FINAL_implementation_plan_2026-05-07.md §5.4 H4 + P-011 hard gate.

Tests (no real Telegram round-trip — that's an end-to-end test):
  1. request() writes pending/<id>.json with the right shape
  2. request() polls decided/<id>.json and returns approval when present
  3. request() returns deny on timeout
  4. record_decision writes decided/<id>.json (only if pending exists)
  5. record_decision returns False for unknown id (idempotency)
  6. list_pending filters out already-decided entries
  7. format_pending_for_telegram has all required fields
  8. permission.permission_gate routes to approver when destructive +
     approver registered
  9. core.agent imports permission.maybe_register_default_approver and
     calls it in run_role startup

Run: `.venv/bin/python tests/_smoke_telegram_approval.py`
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(LAB_ROOT / "bot"))

import approval  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="bert_approval_smoke_"))
approval.APPROVAL_DIR = TMP / "approvals"
approval.PENDING_DIR = approval.APPROVAL_DIR / "pending"
approval.DECIDED_DIR = approval.APPROVAL_DIR / "decided"

from core.types import PermissionDecision, ToolCall  # noqa: E402


def _call(name: str, **args) -> ToolCall:
    return ToolCall(id="t1", name=name, arguments=args)


def _deny() -> PermissionDecision:
    return PermissionDecision(
        allowed=False, reason="P-011 hard gate", requires_telegram_approval=True,
        is_destructive=True,
    )


def _reset() -> None:
    for d in (approval.PENDING_DIR, approval.DECIDED_DIR):
        if d.exists():
            for p in d.glob("*"):
                p.unlink()


def _spawn_responder(verdict: str) -> tuple[threading.Thread, threading.Event]:
    """Spawn a thread that records the chosen verdict on the next
    pending entry it sees, then exits. Returns (thread, stop_event);
    callers should call stop_event.set() to ensure the thread exits
    even if no pending entry was seen, preventing test pollution."""
    stop = threading.Event()
    handled: list[str] = []

    def responder():
        while not stop.is_set():
            for p in approval.PENDING_DIR.glob("*.json"):
                rec = json.loads(p.read_text())
                aid = rec["id"]
                if aid in handled:
                    continue
                if approval.record_decision(aid, verdict, "test"):
                    handled.append(aid)
                    stop.set()
                    return
            time.sleep(0.05)

    t = threading.Thread(target=responder, daemon=True)
    t.start()
    return t, stop


def test_request_writes_pending_file() -> None:
    _reset()
    t, stop = _spawn_responder("approve")
    try:
        out = approval.request(_call("Bash", command="rm -rf /tmp/x"), _deny(),
                               timeout_secs=3)
        assert out.allowed
        assert "PI approved" in out.reason or "Telegram" in out.reason
    finally:
        stop.set()
        t.join(timeout=1)


def test_request_returns_deny_on_explicit_deny() -> None:
    _reset()
    t, stop = _spawn_responder("deny")
    try:
        deny = _deny()
        out = approval.request(_call("Bash", command="rm -rf /tmp/x"), deny,
                               timeout_secs=3)
        assert not out.allowed
        assert out is deny
    finally:
        stop.set()
        t.join(timeout=1)


def test_request_times_out() -> None:
    _reset()
    deny = _deny()
    t0 = time.monotonic()
    out = approval.request(_call("Bash", command="rm -rf /tmp/x"), deny,
                           timeout_secs=2)
    elapsed = time.monotonic() - t0
    assert not out.allowed
    assert 1.5 <= elapsed < 4.0, f"expected ~2s timeout; got {elapsed:.1f}s"


def test_record_decision_idempotent() -> None:
    _reset()
    # Without a matching pending — should return False
    assert not approval.record_decision("nonexistent", "approve", "test")
    # With pending — first call True, second call False (already decided)
    pending_path = approval.PENDING_DIR / "abc12345.json"
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_text(json.dumps({"id": "abc12345"}))
    assert approval.record_decision("abc12345", "approve", "test")
    assert not approval.record_decision("abc12345", "approve", "test")


def test_list_pending_filters_decided() -> None:
    _reset()
    approval._ensure_dirs()
    (approval.PENDING_DIR / "open1.json").write_text(json.dumps({"id": "open1"}))
    (approval.PENDING_DIR / "closed1.json").write_text(json.dumps({"id": "closed1"}))
    (approval.DECIDED_DIR / "closed1.json").write_text(json.dumps({"id": "closed1", "verdict": "approve"}))
    pending = approval.list_pending()
    ids = {p["id"] for p in pending}
    assert "open1" in ids and "closed1" not in ids


def test_format_pending_has_required_fields() -> None:
    text = approval.format_pending_for_telegram({
        "id": "abc12345", "tool": "Bash",
        "arguments": {"command": "rm -rf /tmp"},
        "rationale": "P-011 hard gate",
    })
    assert "abc12345" in text
    assert "Bash" in text
    assert "rm -rf /tmp" in text
    assert "/approve abc12345" in text
    assert "/deny abc12345" in text


def test_permission_gate_routes_to_approver() -> None:
    """When destructive AND approver registered, gate calls request_approval."""
    from core import permission

    captured = {}

    def stub_approver(call, decision):
        captured["called"] = True
        captured["call_name"] = call.name
        return PermissionDecision(
            allowed=True, reason="stub-approved",
            requires_telegram_approval=False, is_destructive=True,
        )

    orig = permission._telegram_approver
    try:
        permission.register_telegram_approver(stub_approver)
        out = permission.permission_gate(
            _call("Bash", command="rm -rf /tmp"),
            mode_name="AUTO",  # no-op; just need a value
            tool_lookup=lambda n: type("TD", (), {"permission_mode": None})(),
        ) if False else None  # Need PermissionMode instance, use real one:
        from core.types import PermissionMode
        out = permission.permission_gate(
            _call("Bash", command="rm -rf /tmp"),
            PermissionMode.AUTO,
            tool_lookup=lambda n: type("TD", (), {"permission_mode": None})(),
        )
        assert captured.get("called"), "approver should have been invoked"
        assert out.allowed, f"approver returned approve; gate should propagate: {out}"
    finally:
        permission._telegram_approver = orig


def test_agent_calls_register_default_approver() -> None:
    src = (LAB_ROOT / "core" / "agent.py").read_text()
    assert "maybe_register_default_approver" in src, (
        "agent.py should call permission.maybe_register_default_approver "
        "in run_role startup"
    )


def test_maybe_register_finds_bot_approval() -> None:
    """Best-effort import — should succeed since bot/approval.py exists."""
    from core import permission
    orig = permission._telegram_approver
    try:
        permission._telegram_approver = None  # reset
        ok = permission.maybe_register_default_approver()
        assert ok, "should find bot/approval.py"
        assert permission._telegram_approver is not None
    finally:
        permission._telegram_approver = orig


def main() -> int:
    tests = [
        test_request_writes_pending_file,
        test_request_returns_deny_on_explicit_deny,
        test_request_times_out,
        test_record_decision_idempotent,
        test_list_pending_filters_decided,
        test_format_pending_has_required_fields,
        test_permission_gate_routes_to_approver,
        test_agent_calls_register_default_approver,
        test_maybe_register_finds_bot_approval,
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
