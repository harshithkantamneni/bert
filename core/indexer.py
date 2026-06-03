"""File-watcher + re-embed daemon — keeps memory.db fresh between cycles.

`core.memory._index_corpus()` does the one-shot walk: it diffs mtimes
against indexed_mtime and re-embeds anything that changed. Runs at
cycle close. Indexer.py is the *between-cycle* freshness mechanism: a
daemon thread that watches memories/ + agents/ + findings/ + channels/
via the PyPI `watchdog` package, debounces a burst of modify events
into a single re-index, and writes stats to
`lab/state/indexer.last_run.json` after each pass.

Two operating modes:
  - Foreground daemon: `python -m core.indexer watch` runs forever.
  - Background thread: callers create IndexerDaemon, call start() and
    later stop() — useful when the agent loop wants live re-indexing
    during a long cycle.

Naming note: the PyPI `watchdog` package and bert's own
`core/watchdog.py` (holding-loop detector) coexist. Inside this module
we import PyPI watchdog as `_pypi_watchdog` to keep the namespace
clear.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core import log

LOG = log.get_logger("bert.indexer")
LAB_ROOT = Path(__file__).resolve().parent.parent
WATCH_DIRS = [
    LAB_ROOT / "memories",
    LAB_ROOT / "agents",
    LAB_ROOT / "findings",
    LAB_ROOT / "channels",
]
LAST_RUN_PATH = LAB_ROOT / "lab" / "state" / "indexer.last_run.json"


@dataclass
class IndexerStats:
    files_seen: int = 0
    files_reindexed: int = 0
    chunks_reindexed: int = 0
    last_event_ts: float = 0.0
    last_run_ts: float = 0.0
    last_run_elapsed_ms: int = 0


class _DebouncedHandler:
    """File-event handler that schedules a single re-index after a burst.

    Used both by the live PyPI-watchdog Observer and by smoke tests
    (which inject events directly via on_event() without a real
    Observer). Keeps the debounce + state-machine logic out of the
    Observer subclass and easy to unit-test.
    """

    def __init__(
        self,
        *,
        reindex_fn: Callable[[], int],
        debounce_secs: float = 2.0,
        stats_path: Path | None = None,
    ):
        self._reindex_fn = reindex_fn
        self._debounce_secs = float(debounce_secs)
        self._stats_path = stats_path or LAST_RUN_PATH
        self.stats = IndexerStats()
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def on_event(self, *, src_path: str, kind: str = "modified") -> None:
        if not _is_indexable(src_path):
            return
        self.stats.files_seen += 1
        self.stats.last_event_ts = time.time()
        # Debounce — cancel pending timer; schedule a fresh one
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_secs, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        t0 = time.monotonic()
        try:
            n_chunks = self._reindex_fn()
        except Exception as e:  # noqa: BLE001
            LOG.exception("indexer: reindex crashed: %s", e)
            n_chunks = 0
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        self.stats.chunks_reindexed += n_chunks
        if n_chunks:
            self.stats.files_reindexed += 1
        self.stats.last_run_ts = time.time()
        self.stats.last_run_elapsed_ms = elapsed_ms
        self._write_stats()
        LOG.info("indexer: reindex done chunks=%d elapsed=%dms", n_chunks, elapsed_ms)

    def _write_stats(self) -> None:
        try:
            self._stats_path.parent.mkdir(parents=True, exist_ok=True)
            self._stats_path.write_text(json.dumps({
                "files_seen": self.stats.files_seen,
                "files_reindexed": self.stats.files_reindexed,
                "chunks_reindexed": self.stats.chunks_reindexed,
                "last_event_ts": self.stats.last_event_ts,
                "last_run_ts": self.stats.last_run_ts,
                "last_run_elapsed_ms": self.stats.last_run_elapsed_ms,
            }, indent=2))
        except OSError as e:
            LOG.warning("indexer: cannot write stats: %s", e)

    def flush_now(self) -> None:
        """Force-fire any pending debounced re-index. Used by stop() and tests."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        self._fire()


def _is_indexable(path: str) -> bool:
    """Only .md files in the watch tree, excluding hidden / venv / cache."""
    p = Path(path)
    if p.suffix.lower() != ".md":
        return False
    parts = p.parts
    excluded = {".venv", "__pycache__", ".git", "logs", "state"}
    return not any(part in excluded or part.startswith(".") for part in parts)


@dataclass
class IndexerDaemon:
    """Long-running fs-watcher + re-embed loop. Use start()/stop() to
    control its lifetime. Backed by PyPI watchdog Observer."""
    debounce_secs: float = 2.0
    watch_dirs: list[Path] | None = None

    def __post_init__(self):
        self._handler = _DebouncedHandler(
            reindex_fn=self._reindex,
            debounce_secs=self.debounce_secs,
        )
        self._observer = None  # PyPI Observer instance
        self._started = False

    def _reindex(self) -> int:
        # Lazy import — memory is heavy; only load when re-indexing.
        from core import memory as _memory
        return _memory._index_corpus()

    def start(self) -> None:
        if self._started:
            return
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        handler = self._handler

        class _Hook(FileSystemEventHandler):
            def on_modified(self, event) -> None:
                if not getattr(event, "is_directory", False):
                    handler.on_event(src_path=event.src_path, kind="modified")

            def on_created(self, event) -> None:
                if not getattr(event, "is_directory", False):
                    handler.on_event(src_path=event.src_path, kind="created")

        self._observer = Observer()
        for d in (self.watch_dirs or WATCH_DIRS):
            if d.exists():
                self._observer.schedule(_Hook(), str(d), recursive=True)
                LOG.info("indexer: watching %s", d)
        self._observer.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
        # Force any pending debounced re-index to fire so we don't lose state
        self._handler.flush_now()
        self._started = False
        LOG.info("indexer: stopped (chunks reindexed lifetime=%d)",
                 self._handler.stats.chunks_reindexed)

    @property
    def stats(self) -> IndexerStats:
        return self._handler.stats


def cli(op: str = "watch") -> int:
    """`python -m core.indexer watch` — run forever; Ctrl-C to stop."""
    if op == "watch":
        d = IndexerDaemon()
        d.start()
        LOG.info("indexer daemon started; Ctrl-C to stop")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            LOG.info("indexer: SIGINT received")
        finally:
            d.stop()
        return 0
    if op == "status":
        if LAST_RUN_PATH.exists():
            print(LAST_RUN_PATH.read_text())
            return 0
        print("(no indexer.last_run.json yet)")
        return 1
    print(f"unknown op: {op}; use 'watch' or 'status'")
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(cli(sys.argv[1] if len(sys.argv) > 1 else "watch"))
