"""Smoke test for A2A POST endpoints (F.9)."""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


def test_send_endpoint_returns_accepted_for_known_skill() -> None:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.post("/a2a/v0/tasks/send", json={
        "skill_id": "bert-memory",
        "message": {"role": "user", "parts": [{"text": "hi"}]},
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["skill_id"] == "bert-memory"
    assert body["state"] == "accepted"
    assert "task_id" in body
    assert body["message_echo"]["role"] == "user"


def test_send_endpoint_rejects_unknown_skill() -> None:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.post("/a2a/v0/tasks/send", json={
        "skill_id": "nonexistent-skill",
    })
    assert r.status_code == 404


def test_send_endpoint_requires_skill_id() -> None:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.post("/a2a/v0/tasks/send", json={})
    assert r.status_code == 400


def test_task_status_endpoint_returns_completed_stub() -> None:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.get("/a2a/v0/tasks/task-fake-id")
    assert r.status_code == 200
    body = r.json()
    assert body["task_id"] == "task-fake-id"
    assert body["state"] == "completed"


def main() -> int:
    tests = [
        test_send_endpoint_returns_accepted_for_known_skill,
        test_send_endpoint_rejects_unknown_skill,
        test_send_endpoint_requires_skill_id,
        test_task_status_endpoint_returns_completed_stub,
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
