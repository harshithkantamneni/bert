"""DocumentCorpusAdapter — first concrete adapter.

Phase B1 of the v3 plan. Wraps the existing
  - core.memory          (sqlite-vec dense vector index)
  - core.graph_store     (typed entity/edge graph)
  - core.retrieval       (hybrid retrieval — vector + graph + cache + RRF)

behind the MemoryAdapter interface so the synthesizer can instantiate
it generically from a profile and the agent loop can use it without
caring about backend specifics.

This adapter is intentionally THIN — most logic lives in the existing
modules. The adapter's job is to:
  1. own the adapter-specific DB directory + migration runner
  2. translate MemoryAdapter calls into the existing API shapes
  3. uniform-format the responses as SearchResult / RelatedResult / etc.

Future-proofing: when BM25 + PPR layers land (B2/B3), they slot into
core/retrieval.py via hybrid_retrieve(); this adapter automatically
inherits them with no changes needed here.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from core.memory_adapters import (
    AdapterStats,
    IngestResult,
    MemoryAdapter,
    RelatedResult,
    SearchResult,
)

LOG = logging.getLogger("bert.memory.document_corpus")


class DocumentCorpusAdapter(MemoryAdapter):
    """Documents → chunks → vectors + graph + cache.

    Ingest sources: file paths (Path), URLs (str), or raw dicts
    {"text": "...", "metadata": {...}}.
    """

    data_shape = "document_corpus"
    name = "document_corpus"

    def _ensure_schema(self) -> None:
        """Run pending migrations from core/migrations/document_corpus/."""
        try:
            from core import migrations
            result = migrations.apply_pending(self.lab_path, self.name)
            if result.errors:
                LOG.warning(
                    "document_corpus migrations had %d errors: %s",
                    len(result.errors), result.errors[:2],
                )
            elif result.applied:
                LOG.info(
                    "applied %d migration(s) to document_corpus: %s",
                    len(result.applied), result.applied,
                )
        except Exception as e:  # noqa: BLE001
            # Don't break adapter instantiation on migration runner failure.
            # The legacy core.memory + core.graph_store create their own
            # tables on demand, so the adapter remains usable.
            LOG.warning("schema migration deferred: %s", e)

    # ── Ingest ──

    def ingest(self, source: Any, **opts) -> IngestResult:
        """Ingest one document. `source` can be:
          - Path to a file under memories/ or findings/
          - str URL (fetched + cached locally)
          - dict {"text": ..., "metadata": ...} for direct insertion

        Delegates to core.memory.create() for the existing pipeline
        (chunking + embedding + sqlite-vec insert + FTS5 insert).
        """
        t0 = time.monotonic()
        warnings: list[str] = []
        items_added = 0
        source_id = ""
        bytes_in = 0
        metadata: dict = {}

        try:
            from core import memory as _legacy_memory
        except ImportError as e:
            return IngestResult(
                source_id="", bytes_in=0, items_added=0,
                duration_ms=int((time.monotonic() - t0) * 1000),
                warnings=(f"core.memory unavailable: {e}",),
                metadata={},
            )

        if isinstance(source, (str, Path)) and (
            isinstance(source, Path) or "/" in str(source)
        ):
            # Path or path-string
            p = Path(source).expanduser()
            if not p.exists():
                warnings.append(f"file not found: {p}")
            else:
                bytes_in = p.stat().st_size
                source_id = str(p)
                # core.memory.create takes (path, content)
                content = p.read_text(errors="replace")
                rel_path = self._relativize(p)
                resp = _legacy_memory.create(rel_path, content)
                if resp.get("ok"):
                    items_added = 1
                    metadata["path"] = rel_path
                else:
                    warnings.append(
                        f"create returned not-ok: {resp.get('error', '?')}"
                    )
        elif isinstance(source, dict) and "text" in source:
            # Direct insertion — write to a temp finding file
            text = source["text"]
            bytes_in = len(text.encode())
            src_meta = source.get("metadata", {})
            fname = src_meta.get("filename", f"ingested_{int(t0)}.md")
            rel_path = f"findings/{fname}"
            resp = _legacy_memory.create(rel_path, text)
            if resp.get("ok"):
                items_added = 1
                source_id = rel_path
                metadata.update(src_meta)
                metadata["path"] = rel_path
            else:
                warnings.append(
                    f"create returned not-ok: {resp.get('error', '?')}"
                )
        else:
            warnings.append(
                f"unsupported source type: {type(source).__name__}"
            )

        return IngestResult(
            source_id=source_id,
            bytes_in=bytes_in,
            items_added=items_added,
            duration_ms=int((time.monotonic() - t0) * 1000),
            warnings=tuple(warnings),
            metadata=metadata,
        )

    def _relativize(self, p: Path) -> str:
        """Best-effort relative path under lab_path."""
        try:
            return str(p.relative_to(self.lab_path))
        except ValueError:
            return str(p)

    # ── Search ──

    def search(
        self,
        query: str,
        k: int = 8,
        filters: dict | None = None,
        method: str | None = None,
    ) -> list[SearchResult]:
        """Hybrid search via core.retrieval (vector + graph + cache + RRF).
        On hybrid failure, falls back to vector-only via core.memory.search.

        `method` may be one of {None, "hybrid", "vector"}. None lets the
        adapter pick the best available.
        """
        method = (method or "hybrid").lower()
        if method not in ("hybrid", "vector"):
            method = "hybrid"

        results: list[SearchResult] = []
        if method == "hybrid":
            try:
                from core import retrieval as _ret
                rr = _ret.hybrid_retrieve(
                    query, k_per_source=max(k * 4, 20), top_n=k,
                )
                for r in rr:
                    meta = r.metadata or {}
                    results.append(SearchResult(
                        id=r.id,
                        score=r.final_score,
                        content=r.text,
                        metadata={
                            "sources": r.sources,
                            "chunk_idx": meta.get("chunk_idx"),
                            **meta,
                        },
                        source_path=meta.get("path"),
                        snippet=(r.text or "")[:300] if r.text else None,
                    ))
                return results
            except Exception as e:  # noqa: BLE001
                LOG.warning("hybrid search failed; falling back to vector: %s", e)
                # Fall through to vector

        # Vector fallback (or explicit method='vector')
        try:
            from core import memory as _legacy_memory
            hits = _legacy_memory.search(query, k=k)
            for h in hits:
                results.append(SearchResult(
                    id=f"{h.get('path', '')}#{h.get('chunk_idx', 0)}",
                    score=1.0 - float(h.get("distance", 1.0)),
                    content=h.get("content"),
                    metadata={
                        "chunk_idx": h.get("chunk_idx"),
                        "distance": h.get("distance"),
                    },
                    source_path=h.get("path"),
                    snippet=(h.get("content") or "")[:300] if h.get("content") else None,
                ))
        except Exception as e:  # noqa: BLE001
            LOG.warning("vector fallback also failed: %s", e)
        return results

    # ── Related ──

    def related(
        self,
        item_id: str,
        depth: int = 2,
        k: int = 8,
        relation_kinds: tuple[str, ...] | None = None,
    ) -> list[RelatedResult]:
        """Graph traversal via core.graph_store. The seed `item_id`
        should be a node id (e.g., 'find_abc123' or 'Paper:arxiv:2312.00752')."""
        results: list[RelatedResult] = []
        try:
            from core import graph_store as gs
            # core.graph_store.neighbors expects a node id + optional edge_type
            # Just do 1-hop for v1; multi-hop is future work.
            for edge_type in (relation_kinds or (None,)):
                neighbors = gs.neighbors(
                    item_id,
                    edge_type=edge_type if edge_type else None,
                )
                for node in neighbors[:k]:
                    results.append(RelatedResult(
                        id=node.get("id", ""),
                        relation_kind=node.get("edge_type", "unknown"),
                        distance=1,
                        content=node.get("label", ""),
                        metadata=node,
                    ))
        except Exception as e:  # noqa: BLE001
            LOG.warning("related() failed: %s", e)
        return results[:k]

    # ── Get / Delete ──

    def get(self, item_id: str) -> dict | None:
        """Fetch one item by id. item_id formats:
          - "<rel_path>" — return file content
          - "<rel_path>#<chunk_idx>" — return one chunk
        """
        if not item_id:
            return None
        # Resolve relative or absolute
        if "#" in item_id:
            path_part, chunk_idx_str = item_id.split("#", 1)
        else:
            path_part, chunk_idx_str = item_id, None
        full_path = self.lab_path / path_part
        if not full_path.exists():
            return None
        try:
            content = full_path.read_text(errors="replace")
        except OSError:
            return None
        return {
            "id": item_id,
            "path": path_part,
            "content": content,
            "size_bytes": full_path.stat().st_size,
            "chunk_idx": int(chunk_idx_str) if chunk_idx_str else None,
        }

    def delete(self, item_id: str) -> bool:
        """Soft-delete: rename to `.deleted/<ts>_<name>`. Preserves history."""
        if not item_id:
            return False
        path_part = item_id.split("#", 1)[0] if "#" in item_id else item_id
        src = self.lab_path / path_part
        if not src.exists():
            return False
        bin_dir = self.lab_path / ".deleted"
        bin_dir.mkdir(parents=True, exist_ok=True)
        dst = bin_dir / f"{int(time.time())}_{src.name}"
        try:
            src.rename(dst)
            return True
        except OSError as e:
            LOG.warning("delete failed: %s", e)
            return False

    # ── Stats ──

    def stats(self) -> AdapterStats:
        """Cheap stats: count files under findings/ + memories/, total bytes,
        most recent mtime."""
        items_total = 0
        bytes_on_disk = 0
        last_ts: int | None = None
        cutoff_24h = time.time() - 86400
        items_24h = 0
        for root in (self.lab_path / "findings", self.lab_path / "memories"):
            if not root.exists():
                continue
            for p in root.rglob("*.md"):
                if not p.is_file():
                    continue
                items_total += 1
                try:
                    st = p.stat()
                    bytes_on_disk += st.st_size
                    if last_ts is None or st.st_mtime > last_ts:
                        last_ts = int(st.st_mtime)
                    if st.st_mtime > cutoff_24h:
                        items_24h += 1
                except OSError:
                    continue
        # Health: if there are 0 items, that's a fresh lab (ok); if the
        # vector DB exists but is empty for a non-fresh lab, degraded
        health = "ok"
        notes: list[str] = []
        if items_total == 0:
            notes.append("no documents ingested yet")
        return AdapterStats(
            items_total=items_total,
            items_added_last_24h=items_24h,
            bytes_on_disk=bytes_on_disk,
            last_ingest_ts=last_ts,
            health=health,
            notes=tuple(notes),
        )

    # ── Proof packet contribution ──

    def export_for_packet(self) -> dict:
        """Document corpus contributes: all findings/*.md, the chunks.db,
        the kùzu graph snapshot. Returns manifest dict."""
        files = []
        for sub in ("findings", "memories", "knowledge"):
            d = self.lab_path / sub
            if d.exists():
                for p in d.rglob("*.md"):
                    files.append(str(p.relative_to(self.lab_path)))
        # Adapter's own DB
        adapter_db = self.db_dir / f"{self.name}.db"
        if adapter_db.exists():
            files.append(str(adapter_db.relative_to(self.lab_path)))
        return {
            "files": files,
            "manifest": {
                "adapter": self.name,
                "data_shape": self.data_shape,
                "items_total": self.stats().items_total,
            },
        }
