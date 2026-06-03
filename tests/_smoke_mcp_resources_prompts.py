"""Smoke + TDD: core/mcp_server.py resources + prompts primitives (Sprint 4 A1).

Asserts the MCP-spec-verified result shapes (modelcontextprotocol official
spec, 2026-05-28): resources/list + resources/read ({contents:[{uri,mimeType,
text}]}), prompts/list (arguments:[{name,description,required}]), and
prompts/get ({description, messages:[{role, content:{type:"text",text}}]} —
content is a STRUCTURED object, not a bare string). Plus the unknown-uri /
unknown-prompt error paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import mcp_server as ms  # noqa: E402


def _rpc(method, params=None, req_id=1):
    msg = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def _server_with_resource_and_prompt():
    srv = ms.MCPServer(name="t", version="0.1")
    srv.register_resource(
        uri="bert://lab/test01/seed_brief", name="seed_brief",
        description="The lab's seed brief", mime_type="text/markdown",
        reader=lambda: "# Seed\n\nInvestigate vector DB recall.")
    srv.register_prompt(
        name="code_review", description="Ask the LLM to review code",
        arguments=[{"name": "code", "description": "the code", "required": True}],
        builder=lambda a: [{"role": "user",
                            "content": {"type": "text", "text": f"Review:\n{a['code']}"}}])
    return srv


def test_resources_list():
    srv = _server_with_resource_and_prompt()
    out = srv.handle(_rpc("resources/list"))["result"]["resources"]
    r = next(x for x in out if x["uri"] == "bert://lab/test01/seed_brief")
    assert r["name"] == "seed_brief" and r["mimeType"] == "text/markdown"


def test_resources_read():
    srv = _server_with_resource_and_prompt()
    out = srv.handle(_rpc("resources/read", {"uri": "bert://lab/test01/seed_brief"}))
    c = out["result"]["contents"][0]
    assert c["uri"] == "bert://lab/test01/seed_brief"
    assert c["mimeType"] == "text/markdown" and "vector DB recall" in c["text"]


def test_resources_read_unknown():
    srv = _server_with_resource_and_prompt()
    out = srv.handle(_rpc("resources/read", {"uri": "bert://nope"}))
    assert "error" in out


def test_prompts_list():
    srv = _server_with_resource_and_prompt()
    out = srv.handle(_rpc("prompts/list"))["result"]["prompts"]
    p = next(x for x in out if x["name"] == "code_review")
    assert p["description"] and p["arguments"][0]["name"] == "code"
    assert p["arguments"][0]["required"] is True


def test_prompts_get_structured_content():
    srv = _server_with_resource_and_prompt()
    out = srv.handle(_rpc("prompts/get", {"name": "code_review",
                                          "arguments": {"code": "x = 1"}}))["result"]
    assert "description" in out                      # top-level description
    msg = out["messages"][0]
    assert msg["role"] == "user"
    # content is a STRUCTURED object {type, text}, not a bare string
    assert isinstance(msg["content"], dict)
    assert msg["content"]["type"] == "text" and "x = 1" in msg["content"]["text"]


def test_prompts_get_unknown():
    srv = _server_with_resource_and_prompt()
    out = srv.handle(_rpc("prompts/get", {"name": "no_such_prompt"}))
    assert "error" in out


def main() -> int:
    tests = [
        test_resources_list,
        test_resources_read,
        test_resources_read_unknown,
        test_prompts_list,
        test_prompts_get_structured_content,
        test_prompts_get_unknown,
    ]
    for t in tests:
        try:
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
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
