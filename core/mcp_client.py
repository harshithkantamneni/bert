"""Native MCP client (JSON-RPC 2.0).

Replaces the 10-LoC `# Implementation pending` stub.

Supports stdio-launched MCP servers (the predominant transport). HTTP
transport is supported by passing a URL via `endpoint=` to the client
factory; we use httpx for that path.

The client implements the methods bert actually calls:
  - initialize / initialized
  - tools/list, tools/call
  - resources/list, resources/read
  - prompts/list, prompts/get

It does NOT implement the full MCP spec — server capability
negotiation, sampling, roots — those land in operational follow-ups
when bert's MCP integration actually demands them. The minimal
surface gets E.2 (agent upskilling) and E.5 (Playwright integration)
unblocked.

Wire-format: JSON-RPC 2.0 with newline-delimited framing over stdio.

Usage:

  client = MCPClient.spawn(["npx", "-y", "@modelcontextprotocol/server-everything"])
  client.initialize(client_name="bert-lab", client_version="0.1")
  tools = client.list_tools()
  result = client.call_tool("echo", {"message": "hi"})
  client.close()
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOG = logging.getLogger("bert.mcp_client")

# JSON-RPC error codes (per spec)
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


@dataclass
class MCPError(Exception):
    code: int
    message: str
    data: Any = None

    def __str__(self) -> str:
        return f"MCP error {self.code}: {self.message}"


@dataclass
class MCPClient:
    """Minimal MCP client over stdio.

    Spawn a server via classmethod `spawn(argv)`; clean up with `close()`.
    """

    proc: subprocess.Popen
    _next_id: int = 1
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _initialized: bool = False
    server_capabilities: dict = field(default_factory=dict)
    server_info: dict = field(default_factory=dict)

    @classmethod
    def spawn(cls, argv: list[str], *, cwd: str | Path | None = None,
              env: dict[str, str] | None = None) -> MCPClient:
        """Launch an MCP server subprocess connected via stdio."""
        LOG.info("mcp: spawning %s", argv)
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        return cls(proc=proc)

    # ── JSON-RPC plumbing ─────────────────────────────────────────────

    def _request(self, method: str, params: dict | None = None,
                 *, timeout: float = 30.0) -> Any:
        """Send a JSON-RPC request, wait for matching response.

        Raises MCPError on protocol error. Times out via stdout poll.
        """
        with self._lock:
            req_id = self._next_id
            self._next_id += 1
            req = {"jsonrpc": "2.0", "id": req_id, "method": method}
            if params is not None:
                req["params"] = params
            line = json.dumps(req) + "\n"
            assert self.proc.stdin is not None
            self.proc.stdin.write(line)
            self.proc.stdin.flush()

            deadline = time.monotonic() + timeout
            while True:
                if time.monotonic() > deadline:
                    raise MCPError(INTERNAL_ERROR, f"timeout waiting for response to {method}")
                assert self.proc.stdout is not None
                raw = self.proc.stdout.readline()
                if not raw:
                    # Process may have exited; bubble up
                    if self.proc.poll() is not None:
                        stderr = self.proc.stderr.read() if self.proc.stderr else ""
                        raise MCPError(INTERNAL_ERROR,
                                       f"server exited (code={self.proc.returncode}): {stderr[:500]}")
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError as e:
                    LOG.warning("mcp: non-JSON line on stdout: %r", raw[:200])
                    raise MCPError(PARSE_ERROR, f"non-JSON line: {e}") from None
                if msg.get("id") != req_id:
                    # Server may emit notifications between request/response
                    if "method" in msg and "id" not in msg:
                        LOG.debug("mcp: notification %s ignored", msg.get("method"))
                        continue
                    LOG.debug("mcp: unmatched id=%s (expected %s); buffering not implemented", msg.get("id"), req_id)
                    continue
                if "error" in msg:
                    err = msg["error"]
                    raise MCPError(err.get("code", INTERNAL_ERROR),
                                   err.get("message", "unknown"),
                                   err.get("data"))
                return msg.get("result")

    def _notify(self, method: str, params: dict | None = None) -> None:
        """Send a notification (no response expected)."""
        with self._lock:
            req = {"jsonrpc": "2.0", "method": method}
            if params is not None:
                req["params"] = params
            line = json.dumps(req) + "\n"
            assert self.proc.stdin is not None
            self.proc.stdin.write(line)
            self.proc.stdin.flush()

    # ── MCP methods bert actually calls ───────────────────────────────

    def initialize(self, *, client_name: str = "bert-lab",
                   client_version: str = "0.1",
                   protocol_version: str = "2025-06-18") -> dict:
        """Negotiate session. Must be called before any other method."""
        result = self._request("initialize", {
            "protocolVersion": protocol_version,
            "capabilities": {
                "roots": {"listChanged": False},
                "sampling": {},
            },
            "clientInfo": {"name": client_name, "version": client_version},
        })
        self.server_capabilities = result.get("capabilities", {}) if result else {}
        self.server_info = result.get("serverInfo", {}) if result else {}
        # Per spec: client must send `initialized` notification after.
        self._notify("notifications/initialized")
        self._initialized = True
        return result or {}

    def list_tools(self) -> list[dict]:
        result = self._request("tools/list")
        return (result or {}).get("tools", []) if isinstance(result, dict) else []

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        return self._request("tools/call", {
            "name": name,
            "arguments": arguments or {},
        }) or {}

    def list_resources(self) -> list[dict]:
        result = self._request("resources/list")
        return (result or {}).get("resources", []) if isinstance(result, dict) else []

    def read_resource(self, uri: str) -> dict:
        return self._request("resources/read", {"uri": uri}) or {}

    def list_prompts(self) -> list[dict]:
        result = self._request("prompts/list")
        return (result or {}).get("prompts", []) if isinstance(result, dict) else []

    def get_prompt(self, name: str, arguments: dict | None = None) -> dict:
        return self._request("prompts/get", {
            "name": name, "arguments": arguments or {},
        }) or {}

    def close(self, *, timeout: float = 5.0) -> int:
        """Terminate the server. Returns exit code."""
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        return self.proc.returncode or 0


# ── HTTP transport ─────────────────────────────────────────────


@dataclass
class MCPHttpClient:
    """MCP client over HTTP+JSON-RPC.

    For MCP servers that expose `/mcp` as a POST endpoint accepting
    a JSON-RPC 2.0 envelope. Convention follows the spec's streamable
    HTTP transport (single POST request, response is the JSON-RPC reply,
    optional Server-Sent Events for streaming).

    Use when the server is remote or unwilling to speak stdio. The
    initialize handshake + tools/list + tools/call API are identical
    to MCPClient; the difference is which transport carries the
    request.
    """

    endpoint: str
    _next_id: int = 1
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _initialized: bool = False
    server_capabilities: dict = field(default_factory=dict)
    server_info: dict = field(default_factory=dict)
    timeout: float = 30.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def _request(self, method: str, params: dict | None = None,
                 *, timeout: float | None = None) -> Any:
        import httpx
        to = timeout if timeout is not None else self.timeout
        with self._lock:
            req_id = self._next_id
            self._next_id += 1
            req = {"jsonrpc": "2.0", "id": req_id, "method": method}
            if params is not None:
                req["params"] = params
            headers = {"Content-Type": "application/json",
                       "Accept": "application/json"}
            headers.update(self.extra_headers)
            try:
                with httpx.Client(timeout=to) as client:
                    resp = client.post(self.endpoint, json=req, headers=headers)
                if resp.status_code >= 400:
                    raise MCPError(INTERNAL_ERROR,
                                   f"HTTP {resp.status_code}: {resp.text[:200]}")
                msg = resp.json()
            except httpx.HTTPError as e:
                raise MCPError(INTERNAL_ERROR, f"http error: {e}") from None
            if "error" in msg:
                err = msg["error"]
                raise MCPError(err.get("code", INTERNAL_ERROR),
                               err.get("message", "unknown"),
                               err.get("data"))
            return msg.get("result")

    def _notify(self, method: str, params: dict | None = None) -> None:
        import httpx
        req = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            req["params"] = params
        try:
            with httpx.Client(timeout=self.timeout) as client:
                client.post(self.endpoint, json=req,
                            headers={"Content-Type": "application/json"})
        except httpx.HTTPError:
            pass  # Notifications are fire-and-forget

    def initialize(self, *, client_name: str = "bert-lab",
                   client_version: str = "0.1",
                   protocol_version: str = "2025-06-18") -> dict:
        result = self._request("initialize", {
            "protocolVersion": protocol_version,
            "capabilities": {"roots": {"listChanged": False}, "sampling": {}},
            "clientInfo": {"name": client_name, "version": client_version},
        })
        self.server_capabilities = (result or {}).get("capabilities", {})
        self.server_info = (result or {}).get("serverInfo", {})
        self._notify("notifications/initialized")
        self._initialized = True
        return result or {}

    def list_tools(self) -> list[dict]:
        result = self._request("tools/list")
        return (result or {}).get("tools", []) if isinstance(result, dict) else []

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        return self._request("tools/call", {
            "name": name, "arguments": arguments or {},
        }) or {}

    def list_resources(self) -> list[dict]:
        result = self._request("resources/list")
        return (result or {}).get("resources", []) if isinstance(result, dict) else []

    def read_resource(self, uri: str) -> dict:
        return self._request("resources/read", {"uri": uri}) or {}

    def close(self) -> int:
        # HTTP is stateless from the client's view — nothing to close.
        return 0
