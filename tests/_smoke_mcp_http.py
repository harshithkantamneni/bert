"""Smoke test for MCPHttpClient (F.8).

We can't easily spin up a real HTTP MCP server in the smoke (would need
an aiohttp/Starlette dependency for the test). Instead we verify the
client structure with monkeypatched httpx.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import mcp_client  # noqa: E402


def _fake_post_factory(resp_json, status=200):
    class FakeResp:
        status_code = status
        text = json.dumps(resp_json) if status >= 400 else ""

        def json(self):
            return resp_json

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def post(self, url, json=None, headers=None):
            return FakeResp()

    return FakeClient


def test_http_initialize_sets_capabilities() -> None:
    client = mcp_client.MCPHttpClient(endpoint="http://example.com/mcp")
    init_result = {
        "protocolVersion": "2025-06-18",
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": "test", "version": "0.1"},
    }
    fake_client = _fake_post_factory({"jsonrpc": "2.0", "id": 1, "result": init_result})
    with mock.patch("httpx.Client", fake_client):
        result = client.initialize(client_name="bert-test")
    assert result["serverInfo"]["name"] == "test"
    assert client.server_capabilities == init_result["capabilities"]
    assert client._initialized is True


def test_http_list_tools_returns_array() -> None:
    client = mcp_client.MCPHttpClient(endpoint="http://example.com/mcp")
    fake = _fake_post_factory({
        "jsonrpc": "2.0", "id": 1,
        "result": {"tools": [{"name": "echo", "description": "x", "inputSchema": {}}]},
    })
    with mock.patch("httpx.Client", fake):
        tools = client.list_tools()
    assert tools == [{"name": "echo", "description": "x", "inputSchema": {}}]


def test_http_call_tool_returns_content() -> None:
    client = mcp_client.MCPHttpClient(endpoint="http://example.com/mcp")
    fake = _fake_post_factory({
        "jsonrpc": "2.0", "id": 1,
        "result": {"content": [{"type": "text", "text": "ok"}]},
    })
    with mock.patch("httpx.Client", fake):
        r = client.call_tool("echo", {"text": "hi"})
    assert r["content"][0]["text"] == "ok"


def test_http_request_error_raises_mcperror() -> None:
    client = mcp_client.MCPHttpClient(endpoint="http://example.com/mcp")
    fake = _fake_post_factory({
        "jsonrpc": "2.0", "id": 1,
        "error": {"code": -32601, "message": "method not found"},
    })
    with mock.patch("httpx.Client", fake):
        try:
            client.list_tools()
        except mcp_client.MCPError as e:
            assert e.code == -32601
            assert "method not found" in e.message
            return
    raise AssertionError("expected MCPError")


def test_http_close_is_noop() -> None:
    client = mcp_client.MCPHttpClient(endpoint="http://example.com/mcp")
    assert client.close() == 0


def main() -> int:
    tests = [
        test_http_initialize_sets_capabilities,
        test_http_list_tools_returns_array,
        test_http_call_tool_returns_content,
        test_http_request_error_raises_mcperror,
        test_http_close_is_noop,
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
