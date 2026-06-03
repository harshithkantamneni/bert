"""MCP server scaffolding for bert's custom servers.

Replaces the 10-LoC `# Implementation pending` stub.

bert hosts a small set of custom MCP servers that expose lab
internals to other agents (A2A interop, E.5):

  bert-orchestrator — dispatch + plan-state queries
  bert-memory       — read access to memories/* and lab/sor/events
  bert-queue        — work_queue + cycle_queue read/write
  bert-mission      — current mission, candidates, decisions
  bert-search       — semantic search over memory tier
  bert-evaluator    — falsifier baseline + verdict history
  bert-sandbox      — sandboxed code execution gateway

This module is the framework — `MCPServer` provides JSON-RPC 2.0
stdio plumbing identical in shape to `core.mcp_client.MCPClient`.
Each named server is a subclass that registers its tools / resources.

The minimal-viable implementation in this commit:
  - JSON-RPC stdio loop with initialize / initialized handshake
  - tools/list + tools/call methods
  - One example server: `BertEchoServer` (for smoke tests)

Individual `bert-*` servers are scaffolded in tools/mcp/<name>.py and
land as separate operational work (Phase E.5 surfaces them on the
canvas; not all need to ship before launch).
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

LOG = logging.getLogger("bert.mcp_server")

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
# bert-defined: replay-protection rejection
REPLAY_REJECTED = -32004


class _ReplayRejected(Exception):
    """Raised when a tool call carries an already-used nonce.
    Caught in handle() and translated to a JSON-RPC error with code
    REPLAY_REJECTED."""


@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict], dict]


@dataclass
class ResourceDef:
    """An MCP resource — a read-only addressable blob (a lab artifact)."""
    uri: str
    name: str
    description: str
    mime_type: str
    reader: Callable[[], str]
    title: str | None = None


@dataclass
class PromptDef:
    """An MCP prompt — a reusable, argument-templated message list.

    `builder(arguments) -> list[{role, content:{type,text}}]` per the MCP
    spec (content is a structured object, not a bare string).
    """
    name: str
    description: str
    arguments: list[dict[str, Any]]  # [{name, description, required}]
    builder: Callable[[dict], list[dict]]
    title: str | None = None


@dataclass
class MCPServer:
    """JSON-RPC 2.0 stdio server. Subclass and register tools in __init__."""

    name: str
    version: str = "0.1.0"
    # Optional dotted namespace (e.g. "bert.lab") — when set, list responses
    # emit qualified ids (collision-safe when a host loads >1 bert server),
    # while calls still accept the bare name (back-compat for CLI + tests).
    namespace: str | None = None
    tools: dict[str, ToolDef] = field(default_factory=dict)
    resources: dict[str, ResourceDef] = field(default_factory=dict)
    prompts: dict[str, PromptDef] = field(default_factory=dict)

    def _qualified(self, bare: str) -> str:
        return f"{self.namespace}.{bare}" if self.namespace else bare

    def _strip_ns(self, name: str) -> str:
        prefix = f"{self.namespace}." if self.namespace else ""
        return name[len(prefix):] if prefix and name.startswith(prefix) else name

    def register_tool(self, name: str, *, description: str,
                      input_schema: dict[str, Any],
                      handler: Callable[[dict], dict]) -> None:
        self.tools[name] = ToolDef(name=name, description=description,
                                    input_schema=input_schema, handler=handler)

    def register_resource(self, *, uri: str, name: str, description: str,
                          mime_type: str, reader: Callable[[], str],
                          title: str | None = None) -> None:
        self.resources[uri] = ResourceDef(
            uri=uri, name=name, description=description, mime_type=mime_type,
            reader=reader, title=title)

    def register_prompt(self, name: str, *, description: str,
                        arguments: list[dict[str, Any]],
                        builder: Callable[[dict], list[dict]],
                        title: str | None = None) -> None:
        self.prompts[name] = PromptDef(
            name=name, description=description, arguments=arguments,
            builder=builder, title=title)

    # ── JSON-RPC handlers ─────────────────────────────────────────────

    def handle(self, msg: dict) -> dict | None:
        """Dispatch one JSON-RPC message. Returns response dict, or None
        for notifications (no response expected)."""
        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") or {}
        is_notification = req_id is None

        try:
            if method == "initialize":
                result = self._initialize(params)
            elif method == "notifications/initialized":
                # Notification — no response.
                return None
            elif method == "tools/list":
                result = self._list_tools()
            elif method == "tools/call":
                result = self._call_tool(params)
            elif method == "resources/list":
                result = self._list_resources()
            elif method == "resources/read":
                result = self._read_resource(params)
            elif method == "prompts/list":
                result = self._list_prompts()
            elif method == "prompts/get":
                result = self._get_prompt(params)
            else:
                if is_notification:
                    return None
                return _err(req_id, METHOD_NOT_FOUND, f"unknown method: {method}")
        except _ReplayRejected as e:
            # H.1 — distinct error code so callers can retry with a
            # fresh nonce, not fall into generic INTERNAL_ERROR.
            if is_notification:
                return None
            return _err(req_id, REPLAY_REJECTED, str(e))
        except (ValueError, KeyError) as e:
            # Client-side bad params (unknown tool/resource/prompt name) →
            # INVALID_PARAMS, not INTERNAL_ERROR (which signals a server bug).
            # MCP/JSON-RPC convention: -32602 for a malformed/unresolvable
            # request. (recheck 2026-05-28)
            if is_notification:
                return None
            return _err(req_id, INVALID_PARAMS, str(e))
        except Exception as e:  # noqa: BLE001
            if is_notification:
                LOG.exception("mcp_server: handler raised in notification %s", method)
                return None
            return _err(req_id, INTERNAL_ERROR, str(e))

        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": "2025-06-18",
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": {"name": self.name, "version": self.version},
        }

    def _list_tools(self) -> dict:
        return {
            "tools": [
                {
                    "name": self._qualified(t.name),
                    "description": t.description,
                    "inputSchema": t.input_schema,
                }
                for t in self.tools.values()
            ],
        }

    # ── Resources (MCP spec: server/resources) ────────────────────────

    def _list_resources(self) -> dict:
        out = []
        for r in self.resources.values():
            entry = {"uri": r.uri, "name": r.name,
                     "description": r.description, "mimeType": r.mime_type}
            if r.title:
                entry["title"] = r.title
            out.append(entry)
        return {"resources": out}

    def _read_resource(self, params: dict) -> dict:
        uri = params.get("uri")
        rd = self.resources.get(uri) if uri else None
        if rd is None:
            raise ValueError(f"unknown resource: {uri!r}")
        return {"contents": [{"uri": rd.uri, "mimeType": rd.mime_type,
                              "text": rd.reader()}]}

    # ── Prompts (MCP spec: server/prompts) ────────────────────────────

    def _list_prompts(self) -> dict:
        out = []
        for p in self.prompts.values():
            entry = {"name": self._qualified(p.name), "description": p.description,
                     "arguments": p.arguments}
            if p.title:
                entry["title"] = p.title
            out.append(entry)
        return {"prompts": out}

    def _get_prompt(self, params: dict) -> dict:
        name = self._strip_ns(params.get("name") or "")
        pd = self.prompts.get(name) if name else None
        if pd is None:
            raise ValueError(f"unknown prompt: {name!r}")
        messages = pd.builder(params.get("arguments") or {})
        return {"description": pd.description, "messages": messages}

    def _call_tool(self, params: dict) -> dict:
        name = self._strip_ns(params.get("name") or "")
        if not name or name not in self.tools:
            raise ValueError(f"unknown tool: {name!r}")
        # H.1 — nonce + replay protection. Per OWASP Top-10-for-
        # Agentic-Apps 2026 LLM07 (insecure-plugin-design): tool calls
        # without nonce can be replayed. Nonce travels via _meta.nonce
        # at the params level (MCP convention for client metadata).
        meta = params.get("_meta") or {}
        nonce = meta.get("nonce") if isinstance(meta, dict) else None
        if nonce:
            try:
                from core import mcp_replay
                if mcp_replay.is_replay(nonce, name):
                    raise _ReplayRejected(
                        f"nonce already used for tool {name!r}"
                    )
                mcp_replay.record_nonce(nonce, name)
            except _ReplayRejected:
                raise
            except Exception as e:  # noqa: BLE001
                # Replay subsystem unavailable → fail open with WARN
                # (advisory; never break dispatch on observability).
                LOG.warning("mcp_replay unavailable (%s); accepting call", e)
        args = params.get("arguments") or {}
        result = self.tools[name].handler(args)
        # MCP expects content blocks in the response
        if isinstance(result, dict) and "content" in result:
            return result
        return {
            "content": [{"type": "text", "text": json.dumps(result, default=str)}],
            "isError": False,
        }

    # ── stdio main loop ───────────────────────────────────────────────

    def serve_stdio(self) -> int:
        """Block until stdin closes; respond to each JSON-RPC line."""
        LOG.info("mcp_server: %s serving on stdio", self.name)
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as e:
                    err = _err(None, PARSE_ERROR, f"parse error: {e}")
                    sys.stdout.write(json.dumps(err) + "\n")
                    sys.stdout.flush()
                    continue
                resp = self.handle(msg)
                if resp is not None:
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()
        except KeyboardInterrupt:
            LOG.info("mcp_server: interrupted")
        return 0


def _err(req_id: Any, code: int, message: str, data: Any = None) -> dict:
    out: dict = {"jsonrpc": "2.0", "id": req_id, "error": {
        "code": code, "message": message,
    }}
    if data is not None:
        out["error"]["data"] = data
    return out


# ── Example / smoke-test server ───────────────────────────────────────


def make_echo_server() -> MCPServer:
    """Tiny server with one tool, used by the smoke test."""
    srv = MCPServer(name="bert-echo")

    def _echo(args: dict) -> dict:
        return {"echoed": args.get("text", "")}

    srv.register_tool(
        "echo",
        description="Echo back the input text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=_echo,
    )
    return srv


def run(name: str) -> int:
    """Entry point invoked from `lab.py mcp <name>`."""
    name = (name or "").lower()
    if name == "bert-echo":
        return make_echo_server().serve_stdio()
    # Custom servers live in tools/mcp/bert_<name>.py with a
    # make_server() factory each. Resolve via dynamic import.
    if name.startswith("bert-"):
        modname = name.replace("bert-", "bert_", 1)
        try:
            mod = __import__(f"tools.mcp.{modname}", fromlist=["make_server"])
            server = mod.make_server()
            return server.serve_stdio()
        except (ImportError, AttributeError) as e:
            LOG.error("mcp_server: %s not available (%s)", name, e)
            return 2
    KNOWN = {"bert-echo", "bert-orchestrator", "bert-memory", "bert-queue",
             "bert-mission", "bert-search", "bert-evaluator", "bert-sandbox"}
    LOG.error("mcp_server: unknown server %r (known: %s)",
              name, sorted(KNOWN))
    return 2
