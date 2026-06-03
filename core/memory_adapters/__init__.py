"""MemoryAdapter — abstract base for all data-shape adapters.

Phase B1 of the v3 plan. Every data shape (document_corpus,
code_repo, time_series, tabular, conversational, knowledge_graph,
multimodal, numeric_simulation) implements this contract.

Locked-in contract (L-10 per plan v3 §11):
  - All adapters store under `lab_path/memory/<adapter_name>/`
  - All adapters own migrations under `core/migrations/<adapter_name>/`
  - All search() returns SearchResult objects (uniform host rendering)
  - All adapters declare `data_shape` + `name` as class attrs

Why each method exists:
  ingest(source)   — sole entry; idempotent invariant prevents drift
  search(...)      — agent's primary memory access; method=None picks adapter's best
  related(...)     — graph traversal; adapters do this differently but expose same call
  get(item_id)     — direct fetch for proof packets, citations, agent re-read
  delete(item_id)  — soft delete with history; needed for kill-list + GDPR
  stats()          — health for lab_status MCP, consolidator triggers, budgets
  consolidate()    — adapter-specific maintenance the consolidator calls
  schema_version() — version reporting for migrations
  export_for_packet() — what this adapter contributes to a proof packet

Adapters auto-register via `__subclasses__()` walk. Synthesizer
discovers them by data_shape via `find_adapter_for_shape(...)`.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOG = logging.getLogger("bert.memory.adapter")


# ── Result dataclasses (shared shape across adapters) ────────────────


@dataclass
class IngestResult:
    """Result of ingesting one source (file, URL, repo, stream chunk)."""
    source_id: str             # canonical id for what was ingested
    bytes_in: int              # raw input size
    items_added: int           # adapter-specific count (chunks/symbols/events)
    duration_ms: int
    warnings: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """One match from search(). Adapters fill what's relevant;
    consumers handle missing fields gracefully."""
    id: str                              # adapter-specific item id
    score: float                         # higher = better; comparable WITHIN one query
    content: str | None = None           # textual representation for the agent
    metadata: dict = field(default_factory=dict)  # adapter-specific
    source_path: str | None = None       # for citation/provenance
    snippet: str | None = None           # ≤300 chars context


@dataclass
class RelatedResult:
    """Item related to a seed via graph/structural traversal."""
    id: str
    relation_kind: str                   # 'cites'|'calls'|'references'|'derived_from'|...
    distance: int                        # graph distance (1 = direct)
    content: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class AdapterStats:
    items_total: int
    items_added_last_24h: int
    bytes_on_disk: int
    last_ingest_ts: int | None
    health: str                          # 'ok'|'degraded'|'down'
    notes: tuple[str, ...] = ()


# ── Abstract base ────────────────────────────────────────────────────


class MemoryAdapter(ABC):
    """Every data-shape adapter implements this interface.

    Invariants:
      - Idempotent ingest: same source twice = no duplicates (use content_hash)
      - Adapter owns DB files under lab_path/memory/<name>/
      - Migrations live at core/migrations/<name>/
      - Subclass MUST set `data_shape` and `name` as class attributes
    """

    data_shape: str = ""    # subclass MUST override (e.g. 'document_corpus')
    name: str = ""          # subclass MUST override (e.g. 'document_corpus')

    def __init__(self, lab_path: Path):
        if not self.data_shape or not self.name:
            raise NotImplementedError(
                f"{type(self).__name__} must declare class attrs "
                f"`data_shape` and `name`"
            )
        self.lab_path = Path(lab_path)
        self.db_dir = self.lab_path / "memory" / self.name
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # ── Required overrides ──

    @abstractmethod
    def _ensure_schema(self) -> None:
        """Run pending migrations from core/migrations/<name>/ to bring
        DB to current schema version. Updates schema_versions table.
        Idempotent."""

    @abstractmethod
    def ingest(self, source: Any, **opts) -> IngestResult:
        """Ingest one source. Idempotent by content hash."""

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 8,
        filters: dict | None = None,
        method: str | None = None,
    ) -> list[SearchResult]:
        """Search the adapter's data. method=None lets adapter pick its best."""

    @abstractmethod
    def related(
        self,
        item_id: str,
        depth: int = 2,
        k: int = 8,
        relation_kinds: tuple[str, ...] | None = None,
    ) -> list[RelatedResult]:
        """Graph/structural traversal from a seed item."""

    @abstractmethod
    def get(self, item_id: str) -> dict | None:
        """Fetch one item by id with full content + metadata."""

    @abstractmethod
    def delete(self, item_id: str) -> bool:
        """Soft-delete (mark invalidated, keep history). True if found."""

    @abstractmethod
    def stats(self) -> AdapterStats:
        """Health + size info. Cheap; should not require full scan."""

    # ── Optional hooks (default no-op) ──

    def consolidate(self) -> dict:
        """Adapter-specific consolidation (dedup, rollup, archive).
        Called by core.consolidator. Returns telemetry dict."""
        return {}

    def schema_version(self) -> int:
        """Current schema version from migrations meta. 0 = uninitialized."""
        try:
            from core import migrations
            st = migrations.status(self.lab_path, self.name)
            return st.current_version
        except Exception as e:  # noqa: BLE001
            LOG.warning("schema_version lookup failed: %s", e)
            return 0

    def export_for_packet(self) -> dict:
        """What this adapter contributes to a proof packet export.
        Returns paths to files to include + manifest entries."""
        return {"files": [], "manifest": {}}


# ── Registry ─────────────────────────────────────────────────────────


def find_adapter_for_shape(data_shape: str) -> type[MemoryAdapter]:
    """Return the MemoryAdapter subclass whose data_shape matches.
    Discovers all subclasses by importing the package's modules.

    Raises ValueError if no adapter is registered for the shape.
    """
    # Trigger import of all adapter modules so subclasses register
    _import_all_adapter_modules()
    for cls in _all_subclasses(MemoryAdapter):
        if getattr(cls, "data_shape", None) == data_shape:
            return cls
    raise ValueError(
        f"no MemoryAdapter registered for data_shape={data_shape!r}; "
        f"known shapes: {[c.data_shape for c in _all_subclasses(MemoryAdapter)]}"
    )


def list_registered_adapters() -> list[dict]:
    """Inventory of adapters that have imported into the registry."""
    _import_all_adapter_modules()
    return [
        {
            "name": cls.name,
            "data_shape": cls.data_shape,
            "module": cls.__module__,
        }
        for cls in _all_subclasses(MemoryAdapter)
    ]


def _all_subclasses(cls):
    """Recursively walk subclass tree."""
    out = set()
    for sub in cls.__subclasses__():
        out.add(sub)
        out.update(_all_subclasses(sub))
    return out


_imported = False


def _import_all_adapter_modules() -> None:
    """Import every module in core.memory_adapters.* so subclasses register."""
    global _imported
    if _imported:
        return
    import core.memory_adapters as pkg
    for _finder, modname, _ispkg in pkgutil.iter_modules(pkg.__path__):
        if modname == "__init__":
            continue
        try:
            importlib.import_module(f"core.memory_adapters.{modname}")
        except Exception as e:  # noqa: BLE001
            LOG.warning(
                "failed to import core.memory_adapters.%s: %s", modname, e
            )
    _imported = True
