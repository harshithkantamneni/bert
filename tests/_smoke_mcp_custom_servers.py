"""Smoke tests for the 7 bert-* custom MCP servers.

Per E.1 deferred scope. Each server has a make_server() factory; this
suite confirms each one builds, registers its tools, and handles the
initialize handshake + tools/list + a representative tool call.

End-to-end stdio handshake is covered by _smoke_mcp.py for bert-echo;
here we do in-process handle() round-trips for speed.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

SERVER_NAMES = [
    "bert-orchestrator", "bert-memory", "bert-queue", "bert-mission",
    "bert-search", "bert-evaluator", "bert-sandbox",
]


def _factory(name: str):
    modname = name.replace("bert-", "bert_", 1)
    mod = __import__(f"tools.mcp.{modname}", fromlist=["make_server"])
    return mod.make_server


def test_each_server_builds() -> None:
    for name in SERVER_NAMES:
        factory = _factory(name)
        srv = factory()
        assert srv.name == name, f"{name}: server.name mismatch {srv.name!r}"
        assert len(srv.tools) >= 1, f"{name} has no tools registered"


def test_each_server_initialize_handshake() -> None:
    for name in SERVER_NAMES:
        srv = _factory(name)()
        resp = srv.handle({"jsonrpc": "2.0", "id": 1,
                           "method": "initialize", "params": {}})
        assert resp["result"]["serverInfo"]["name"] == name
        assert "tools" in resp["result"]["capabilities"]


def test_each_server_tools_list_nonempty() -> None:
    for name in SERVER_NAMES:
        srv = _factory(name)()
        resp = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        tools = resp["result"]["tools"]
        assert len(tools) >= 1, f"{name} tools/list returned empty"
        for t in tools:
            assert "name" in t and "description" in t and "inputSchema" in t


def test_bert_memory_tail_events() -> None:
    """Concrete tool exercise: tail_events on bert-memory."""
    srv = _factory("bert-memory")()
    resp = srv.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "tail_events", "arguments": {"limit": 3}},
    })
    # tool returned a result (content blocks)
    content = resp["result"]["content"]
    assert isinstance(content, list) and len(content) == 1
    text = content[0]["text"]
    assert "events" in text  # JSON-encoded dict containing 'events' key


def test_bert_search_grep_finds_a_known_string() -> None:
    """bert-search.grep should find a string we know exists."""
    srv = _factory("bert-search")()
    resp = srv.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "grep", "arguments": {"query": "ts"}},  # ubiquitous token
    })
    content = resp["result"]["content"]
    text = content[0]["text"]
    assert "hits" in text


def test_bert_queue_submit_requires_approver() -> None:
    """bert-queue.submit_pending should refuse without an approver."""
    srv = _factory("bert-queue")()
    resp = srv.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "submit_pending", "arguments": {"task": "test"}},
    })
    text = resp["result"]["content"][0]["text"]
    assert "approver" in text.lower()


def test_bert_sandbox_run_python_requires_approver() -> None:
    """bert-sandbox.run_python should refuse without an approver."""
    srv = _factory("bert-sandbox")()
    resp = srv.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "run_python",
                   "arguments": {"code": "print(1)"}},
    })
    text = resp["result"]["content"][0]["text"]
    assert "approver" in text.lower()


def main() -> int:
    tests = [
        test_each_server_builds,
        test_each_server_initialize_handshake,
        test_each_server_tools_list_nonempty,
        test_bert_memory_tail_events,
        test_bert_search_grep_finds_a_known_string,
        test_bert_queue_submit_requires_approver,
        test_bert_sandbox_run_python_requires_approver,
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
