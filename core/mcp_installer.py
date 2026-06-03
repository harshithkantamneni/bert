"""MCP server installer + config loader.

Per FINAL_implementation_plan_amendment_2026-05-13.md §A1 E.1.
Replaces the 10-LoC `# Implementation pending` stub.

Reads MCP server configuration from `state/mcp_servers.json`. The
schema matches the Claude Desktop / Cursor / VSCode convention so bert
can adopt MCP server entries from those configs without reformatting.

  {
    "mcpServers": {
      "playwright": {
        "command": "npx",
        "args": ["-y", "@playwright/mcp@latest"],
        "env": {"PWDEBUG": "0"}
      },
      "fetch": {
        "command": "uvx",
        "args": ["mcp-server-fetch"]
      }
    }
  }

The installer exposes:
  - list_configured() — names of configured servers
  - load_spec(name) — full spec for one server
  - spawn(name) — spawn the server via MCPClient
  - install(name, command, args, env) — add to registry
  - uninstall(name) — remove from registry
  - registry_path() — where the config file lives
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from core.mcp_client import MCPClient

LOG = logging.getLogger("bert.mcp_installer")
LAB_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = LAB_ROOT / "state" / "mcp_servers.json"


@dataclass
class MCPServerSpec:
    name: str
    command: str
    args: list[str]
    env: dict[str, str]
    description: str = ""

    @classmethod
    def from_dict(cls, name: str, d: dict) -> MCPServerSpec:
        return cls(
            name=name,
            command=str(d.get("command", "")),
            args=list(d.get("args", [])),
            env=dict(d.get("env", {})),
            description=str(d.get("description", "")),
        )

    def argv(self) -> list[str]:
        return [self.command, *self.args]


def registry_path() -> Path:
    """Return the configured registry path (override via BERT_MCP_REGISTRY env)."""
    override = os.environ.get("BERT_MCP_REGISTRY")
    return Path(override) if override else DEFAULT_REGISTRY


def _load_registry(path: Path | None = None) -> dict[str, MCPServerSpec]:
    p = path or registry_path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        LOG.warning("mcp_installer: %s is not valid JSON (%s); treating as empty", p, e)
        return {}
    servers = raw.get("mcpServers", {})
    out: dict[str, MCPServerSpec] = {}
    for name, d in servers.items():
        if not isinstance(d, dict):
            LOG.warning("mcp_installer: skipping invalid entry %r", name)
            continue
        out[name] = MCPServerSpec.from_dict(name, d)
    return out


def list_configured(path: Path | None = None) -> list[str]:
    return sorted(_load_registry(path).keys())


def load_spec(name: str, *, path: Path | None = None) -> MCPServerSpec | None:
    return _load_registry(path).get(name)


def spawn(name: str, *, path: Path | None = None) -> MCPClient:
    """Launch the named MCP server and return an initialized client.

    Raises FileNotFoundError if the name isn't configured.
    The returned client has already completed the initialize handshake.
    """
    spec = load_spec(name, path=path)
    if spec is None:
        configured = list_configured(path)
        raise FileNotFoundError(
            f"MCP server {name!r} not configured (registry: {registry_path()}; "
            f"available: {configured})"
        )
    if not spec.command:
        raise ValueError(f"MCP server {name!r} has empty command")
    env = dict(os.environ)
    env.update(spec.env)
    client = MCPClient.spawn(spec.argv(), env=env)
    client.initialize(client_name=f"bert-lab/{name}")
    return client


def install(name: str, command: str, args: list[str],
            *, env: dict[str, str] | None = None, description: str = "",
            path: Path | None = None) -> None:
    """Add or replace an MCP server entry in the registry."""
    p = path or registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = json.loads(p.read_text()) if p.exists() else {"mcpServers": {}}
    except json.JSONDecodeError:
        raw = {"mcpServers": {}}
    raw.setdefault("mcpServers", {})[name] = {
        "command": command,
        "args": args,
        "env": env or {},
        "description": description,
    }
    p.write_text(json.dumps(raw, indent=2) + "\n")
    LOG.info("mcp_installer: installed %s → %s %s", name, command, args)


def uninstall(name: str, *, path: Path | None = None) -> bool:
    """Remove an MCP server entry. Returns True if removed."""
    p = path or registry_path()
    if not p.exists():
        return False
    raw = json.loads(p.read_text())
    servers = raw.setdefault("mcpServers", {})
    if name not in servers:
        return False
    del servers[name]
    p.write_text(json.dumps(raw, indent=2) + "\n")
    LOG.info("mcp_installer: uninstalled %s", name)
    return True
