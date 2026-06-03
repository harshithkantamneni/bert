"""Smoke test for core/agntcy.py + the AGNTCY API endpoints."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import agntcy  # noqa: E402


def _require_api() -> None:
    """The A2A FastAPI app (`api.main`) is a live-lab runtime surface not
    shipped in the public retrieval-MCP repo. Skip endpoint tests when it
    is absent."""
    if importlib.util.find_spec("api") is None:
        pytest.skip("requires lab runtime artifact: api.main (FastAPI app not shipped in the public repo)")


def test_agent_id_is_stable() -> None:
    a = agntcy._agent_id()
    b = agntcy._agent_id()
    assert a == b
    assert a.startswith("bert-")


def test_agent_card_extensions_shape() -> None:
    ext = agntcy.agent_card_agntcy_extensions(skills=[
        {"id": "bert-memory"}, {"id": "playwright"},
    ])
    assert "agntcy" in ext
    a = ext["agntcy"]
    assert a["spec_version"] == "0.1"
    assert a["agent_id"].startswith("bert-")
    assert a["agent_did"].startswith("did:agntcy:bert-lab:")
    # Transports
    transports = a["transports"]
    assert any(t["kind"] == "http+json-rpc" for t in transports)
    mcp = next(t for t in transports if t["kind"] == "mcp-stdio")
    assert "bert-memory" in mcp["available_servers"]
    # Filter: external (non-bert) skills should NOT appear in mcp servers
    assert "playwright" not in mcp["available_servers"]
    # Observability
    assert "observability" in a
    assert "verdict" in a["observability"]["event_classes"]
    # Governance
    assert a["governance"]["permission_model"] == "P-005 PI-blessed"


def test_directory_entry_shape() -> None:
    entry = agntcy.agntcy_directory_entry()
    assert entry["display_name"] == "bert-lab"
    assert entry["spec_version"] == "agntcy-0.1"
    assert "agent_id" in entry
    assert entry["agent_did"].startswith("did:agntcy:bert-lab:")
    assert "registration_ts" in entry


def test_parse_slim_envelope_returns_none_without_sender() -> None:
    assert agntcy.parse_slim_envelope({}) is None
    assert agntcy.parse_slim_envelope({"x-other-header": "x"}) is None


def test_parse_slim_envelope_full_headers() -> None:
    headers = {
        "x-agntcy-sender-did": "did:agntcy:other-lab:abc",
        "x-agntcy-receiver-did": "did:agntcy:bert-lab:def",
        "x-agntcy-correlation-id": "corr-123",
        "x-agntcy-trace-id": "trace-456",
        "x-agntcy-span-id": "span-789",
        "x-agntcy-ts": "2026-05-13T17:00:00Z",
        "x-agntcy-custom": "extra-value",
        "authorization": "Bearer token-xyz",
    }
    env = agntcy.parse_slim_envelope(headers)
    assert env is not None
    assert env.sender_did == "did:agntcy:other-lab:abc"
    assert env.receiver_did == "did:agntcy:bert-lab:def"
    assert env.correlation_id == "corr-123"
    assert env.trace_id == "trace-456"
    assert env.auth_token == "Bearer token-xyz"
    assert env.extras.get("custom") == "extra-value"


def test_slim_response_headers_echoes_correlation() -> None:
    env = agntcy.SLIMEnvelope(
        sender_did="A", receiver_did="B",
        correlation_id="c-1", ts="2026-05-13T17:00:00Z",
        trace_id="t-99",
    )
    resp = agntcy.slim_response_headers(env)
    # Receiver and sender swap
    assert resp["x-agntcy-sender-did"] == "B"
    assert resp["x-agntcy-receiver-did"] == "A"
    assert resp["x-agntcy-correlation-id"] == "c-1"
    assert resp["x-agntcy-trace-id"] == "t-99"


def test_emit_agntcy_event_writes_jsonl() -> None:
    tmp = Path(tempfile.mkdtemp()) / "agntcy_event.jsonl"
    agntcy.AGNTCY_OBS_PATH = tmp
    agntcy.emit_agntcy_event(
        "test_event",
        correlation_id="corr-test",
        payload={"x": 1, "y": "two"},
    )
    assert tmp.exists()
    lines = tmp.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["event_class"] == "test_event"
    assert rec["correlation_id"] == "corr-test"
    assert rec["payload"]["x"] == 1
    assert rec["agent_id"].startswith("bert-")


def test_agent_card_endpoint_includes_agntcy_extensions() -> None:
    _require_api()
    from api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    r = client.get("/.well-known/agent.json")
    assert r.status_code == 200
    card = r.json()
    # A2A fields still present
    assert card["name"] == "bert-lab"
    assert "skills" in card
    # AGNTCY extensions merged in
    assert "agntcy" in card
    assert card["agntcy"]["spec_version"] == "0.1"
    assert "transports" in card["agntcy"]
    assert "observability" in card["agntcy"]
    assert "governance" in card["agntcy"]


def test_agntcy_directory_endpoint() -> None:
    _require_api()
    from api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    r = client.get("/.well-known/agntcy-directory.json")
    assert r.status_code == 200
    entry = r.json()
    assert entry["display_name"] == "bert-lab"
    assert "agent_did" in entry


def test_observability_inbound_endpoint() -> None:
    _require_api()
    from api.main import app
    from fastapi.testclient import TestClient
    # Isolate event log
    tmp = Path(tempfile.mkdtemp()) / "agntcy_event.jsonl"
    agntcy.AGNTCY_OBS_PATH = tmp
    client = TestClient(app)
    r = client.post("/a2a/v0/observability/event", json={
        "event_class": "external_dispatch",
        "correlation_id": "corr-99",
        "sender_did": "did:agntcy:peer:xyz",
        "status": "completed",
    })
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert tmp.exists()
    rec = json.loads(tmp.read_text().splitlines()[0])
    assert rec["event_class"] == "external_dispatch"
    assert rec["correlation_id"] == "corr-99"


def main() -> int:
    tests = [
        test_agent_id_is_stable,
        test_agent_card_extensions_shape,
        test_directory_entry_shape,
        test_parse_slim_envelope_returns_none_without_sender,
        test_parse_slim_envelope_full_headers,
        test_slim_response_headers_echoes_correlation,
        test_emit_agntcy_event_writes_jsonl,
        test_agent_card_endpoint_includes_agntcy_extensions,
        test_agntcy_directory_endpoint,
        test_observability_inbound_endpoint,
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
