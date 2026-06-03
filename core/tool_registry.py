"""Dynamic tool registry.

Registers built-in tools + skill-derived tools + MCP-derived tools +
bert-generated tools. Lookup by name; schema dump for model context;
hot-reload (Phase 3+).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.types import PermissionMode, SandboxTier, ToolDescriptor

_registry: dict[str, ToolDescriptor] = {}


def register(td: ToolDescriptor) -> None:
    """Register a tool. Last-write-wins on name collision (with warning)."""
    if td.name in _registry:
        # Hot-reload semantics for skill / MCP / creator-generated tools
        existing = _registry[td.name]
        if existing.source == "builtin" and td.source != "builtin":
            raise ValueError(
                f"Cannot override builtin tool '{td.name}' with {td.source} tool"
            )
    _registry[td.name] = td


def register_function(
    name: str,
    description: str,
    parameters_schema: dict[str, Any],
    handler: Callable[..., Any],
    *,
    permission_mode: PermissionMode = PermissionMode.AUTO,
    sandbox_tier: SandboxTier = SandboxTier.TRUSTED,
    is_destructive: bool = False,
    source: str = "builtin",
) -> None:
    """Convenience wrapper around register()."""
    register(ToolDescriptor(
        name=name,
        description=description,
        parameters_schema=parameters_schema,
        handler=handler,
        permission_mode=permission_mode,
        sandbox_tier=sandbox_tier,
        is_destructive=is_destructive,
        source=source,  # type: ignore[arg-type]
    ))


def get(name: str) -> ToolDescriptor | None:
    return _registry.get(name)


def all_tools() -> list[ToolDescriptor]:
    return list(_registry.values())


def make_invoker():
    """Return a `(tool_name, args: dict) -> result` callable backed by this
    registry — the bridge skill_executor.ExecutionContext.tool_invoker expects.

    This is the production engine that lets DSL skills run against real tools
    (the agent loop calls tool handlers directly and never used skills). Raises
    KeyError for an unregistered tool so a typo in a skill surfaces loudly rather
    than silently no-op'ing.
    """
    def invoke(tool_name: str, args: dict | None = None):
        td = get(tool_name)
        if td is None:
            raise KeyError(f"tool not registered: {tool_name!r}")
        return td.handler(**(args or {}))
    return invoke


def schemas_for_model() -> list[dict[str, Any]]:
    """Build the OpenAI-compatible `tools` array for an LLM call."""
    return [
        {
            "type": "function",
            "function": {
                "name": td.name,
                "description": td.description,
                "parameters": td.parameters_schema,
            },
        }
        for td in _registry.values()
    ]


def clear() -> None:
    """Reset registry (used in tests + hot reload)."""
    _registry.clear()


def count() -> int:
    return len(_registry)
