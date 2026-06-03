"""Per-adapter schema migration framework.

Phase A6 of the v3 plan. Each MemoryAdapter (DocumentCorpus,
CodeRepo, TimeSeries, etc.) ships its own ordered migration sequence.
When bert opens a lab, the runner compares the lab's current schema
version (recorded in a meta table) against the adapter's current
version (the highest migration file number) and runs any pending
migrations in order.

Directory layout (one subdirectory per adapter):

    core/migrations/
    ├── __init__.py             ← this file (runner + meta-table helpers)
    ├── document_corpus/
    │   ├── 0001_initial.sql
    │   ├── 0002_add_section_path.sql
    │   └── 0003_add_language.sql
    ├── code_repo/
    │   ├── 0001_initial.sql
    │   └── 0002_add_test_coverage.sql
    └── time_series/
        └── 0001_initial.sql

Migration file conventions:
  - Filename: `<NNNN>_<short_description>.sql`
  - First 4 chars are zero-padded sequence number; later = higher
  - Contents: pure SQLite SQL (DDL + optional INSERT for seed data)
  - Idempotency NOT required by file; runner records what's applied

Per-lab state lives at `<lab>/state/bert_meta.db`:

    CREATE TABLE schema_versions (
      adapter        TEXT PRIMARY KEY,
      version        INTEGER NOT NULL,
      applied_at_ts  REAL NOT NULL,
      migration_file TEXT NOT NULL
    );

The runner is invoked by `MemoryAdapter._ensure_schema()` (B1 lands
this hook). Standalone CLI also available for ops/debugging:

    python -m core.migrations status <lab_path>
    python -m core.migrations apply <lab_path> [adapter]
"""

from __future__ import annotations

import logging
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger("bert.migrations")

LAB_ROOT = Path(__file__).resolve().parent.parent.parent
MIGRATIONS_DIR = LAB_ROOT / "core" / "migrations"
_FILE_PATTERN = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


@dataclass(frozen=True)
class MigrationStatus:
    adapter: str
    current_version: int        # in the lab right now
    available_version: int      # latest migration in the codebase
    pending: tuple[Path, ...]   # migrations to apply, in order


@dataclass(frozen=True)
class MigrationResult:
    adapter: str
    applied: tuple[str, ...]    # filenames applied this run
    final_version: int
    errors: tuple[str, ...]


# ── Meta-table helpers ────────────────────────────────────────────────


def _meta_db_path(lab_path: Path) -> Path:
    return lab_path / "state" / "bert_meta.db"


def _ensure_meta_table(con: sqlite3.Connection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS schema_versions (
            adapter        TEXT PRIMARY KEY,
            version        INTEGER NOT NULL,
            applied_at_ts  REAL    NOT NULL,
            migration_file TEXT    NOT NULL
        )
    """)
    con.commit()


def _get_current_version(con: sqlite3.Connection, adapter: str) -> int:
    row = con.execute(
        "SELECT version FROM schema_versions WHERE adapter = ?",
        (adapter,),
    ).fetchone()
    return int(row[0]) if row else 0


def _set_current_version(
    con: sqlite3.Connection, adapter: str, version: int,
    migration_file: str,
) -> None:
    con.execute(
        """
        INSERT INTO schema_versions (adapter, version, applied_at_ts, migration_file)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(adapter) DO UPDATE SET
            version = excluded.version,
            applied_at_ts = excluded.applied_at_ts,
            migration_file = excluded.migration_file
        """,
        (adapter, version, time.time(), migration_file),
    )
    con.commit()


# ── Migration discovery ──────────────────────────────────────────────


def _adapter_dir(adapter: str) -> Path | None:
    p = MIGRATIONS_DIR / adapter
    return p if p.is_dir() else None


def _list_migrations(adapter: str) -> list[tuple[int, Path]]:
    """List all migration files for `adapter`, sorted by sequence.
    Returns [(version, path), ...]. Empty if adapter dir missing."""
    d = _adapter_dir(adapter)
    if d is None:
        return []
    out: list[tuple[int, Path]] = []
    for f in sorted(d.iterdir()):
        if not f.is_file() or not f.name.endswith(".sql"):
            continue
        m = _FILE_PATTERN.match(f.name)
        if not m:
            LOG.warning("ignoring non-conforming migration file: %s", f.name)
            continue
        out.append((int(m.group(1)), f))
    return out


def list_known_adapters() -> list[str]:
    """All adapter names with migration directories on disk."""
    if not MIGRATIONS_DIR.is_dir():
        return []
    return sorted(
        p.name for p in MIGRATIONS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith("_") and not p.name.startswith(".")
    )


# ── SQL splitting (transaction-safe migration apply) ─────────────────


def _split_sql_statements(sql: str) -> list[str]:
    """Split SQL into individual statements on `;` boundaries, respecting
    SQL string literals + line comments. Our migration files are simple
    DDL (CREATE TABLE / INDEX / TRIGGER) so this is sufficient — but it
    correctly handles semicolons inside quoted strings."""
    statements: list[str] = []
    buf: list[str] = []
    i = 0
    in_squote = False
    in_dquote = False
    in_line_comment = False
    in_block_comment = False
    while i < len(sql):
        c = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""
        if in_line_comment:
            buf.append(c)
            if c == "\n":
                in_line_comment = False
        elif in_block_comment:
            buf.append(c)
            if c == "*" and nxt == "/":
                buf.append(nxt)
                i += 1
                in_block_comment = False
        elif in_squote:
            buf.append(c)
            if c == "'" and nxt == "'":
                buf.append(nxt)
                i += 1
            elif c == "'":
                in_squote = False
        elif in_dquote:
            buf.append(c)
            if c == '"':
                in_dquote = False
        else:
            if c == "-" and nxt == "-":
                in_line_comment = True
                buf.append(c)
            elif c == "/" and nxt == "*":
                in_block_comment = True
                buf.append(c)
            elif c == "'":
                in_squote = True
                buf.append(c)
            elif c == '"':
                in_dquote = True
                buf.append(c)
            elif c == ";":
                stmt = "".join(buf).strip()
                if stmt:
                    statements.append(stmt)
                buf = []
            else:
                buf.append(c)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


# ── Public API ───────────────────────────────────────────────────────


def status(lab_path: Path, adapter: str) -> MigrationStatus:
    """What's the migration state for this (lab, adapter) pair?"""
    available = _list_migrations(adapter)
    available_version = max((v for v, _ in available), default=0)
    db = _meta_db_path(lab_path)
    if not db.exists():
        return MigrationStatus(
            adapter=adapter, current_version=0,
            available_version=available_version,
            pending=tuple(p for _, p in available),
        )
    with sqlite3.connect(db) as con:
        _ensure_meta_table(con)
        current = _get_current_version(con, adapter)
    pending = tuple(p for v, p in available if v > current)
    return MigrationStatus(
        adapter=adapter, current_version=current,
        available_version=available_version, pending=pending,
    )


def apply_pending(lab_path: Path, adapter: str) -> MigrationResult:
    """Run all pending migrations for (lab, adapter) in order.

    Idempotent: re-running after success is a no-op. Each migration
    runs in its own transaction; if one fails, subsequent ones are
    skipped and the result.errors list contains the failure detail.

    The adapter's actual database file is owned by the adapter; this
    runner connects to it via a convention path
    (lab/memory/<adapter>/<adapter>.db) — adapters that store
    elsewhere need to call this differently (B1 handles per-adapter
    nuances)."""
    st = status(lab_path, adapter)
    if not st.pending:
        return MigrationResult(
            adapter=adapter, applied=(), final_version=st.current_version,
            errors=(),
        )

    # Adapter DB lives under lab/memory/<adapter>/
    adapter_db_dir = lab_path / "memory" / adapter
    adapter_db_dir.mkdir(parents=True, exist_ok=True)
    adapter_db = adapter_db_dir / f"{adapter}.db"

    applied: list[str] = []
    errors: list[str] = []
    final_version = st.current_version

    meta_db = _meta_db_path(lab_path)
    meta_db.parent.mkdir(parents=True, exist_ok=True)

    for path in st.pending:
        try:
            sql = path.read_text()
        except OSError as e:
            errors.append(f"{path.name}: read failed — {e}")
            break
        try:
            with sqlite3.connect(adapter_db) as con:
                # Use explicit transaction so ANY failure rolls back the
                # entire migration file (executescript leaks partial state
                # because it implicitly commits between statements).
                con.isolation_level = None  # we control transactions
                con.execute("BEGIN")
                try:
                    for stmt in _split_sql_statements(sql):
                        if stmt.strip():
                            con.execute(stmt)
                    con.execute("COMMIT")
                except Exception:
                    con.execute("ROLLBACK")
                    raise
        except sqlite3.Error as e:
            errors.append(f"{path.name}: SQL error — {e}")
            break
        m = _FILE_PATTERN.match(path.name)
        if not m:
            errors.append(f"{path.name}: filename pattern broken (no version)")
            break
        version = int(m.group(1))
        try:
            with sqlite3.connect(meta_db) as con:
                _ensure_meta_table(con)
                _set_current_version(con, adapter, version, path.name)
        except sqlite3.Error as e:
            errors.append(f"{path.name}: meta-table write failed — {e}")
            break
        applied.append(path.name)
        final_version = version
        LOG.info("applied migration %s on adapter=%s (lab=%s)",
                 path.name, adapter, lab_path.name)

    return MigrationResult(
        adapter=adapter, applied=tuple(applied),
        final_version=final_version, errors=tuple(errors),
    )


def apply_all_pending(lab_path: Path) -> list[MigrationResult]:
    """Apply pending migrations for ALL known adapters in this lab."""
    return [
        apply_pending(lab_path, adapter)
        for adapter in list_known_adapters()
    ]


# ── CLI ──────────────────────────────────────────────────────────────


def _cli(argv: list[str]) -> int:
    """python -m core.migrations status <lab_path>
    python -m core.migrations apply <lab_path> [<adapter>]
    python -m core.migrations list-adapters
    """
    import json
    if len(argv) < 2:
        print("usage: migrations status|apply|list-adapters ...",
              file=sys.stderr)
        return 2
    cmd = argv[1]
    if cmd == "list-adapters":
        print(json.dumps(list_known_adapters()))
        return 0
    if cmd == "status":
        if len(argv) < 3:
            print("usage: migrations status <lab_path>", file=sys.stderr)
            return 2
        lab = Path(argv[2]).expanduser()
        for adapter in list_known_adapters():
            st = status(lab, adapter)
            print(
                f"{adapter:25s} current={st.current_version:4d} "
                f"available={st.available_version:4d} pending={len(st.pending)}"
            )
        return 0
    if cmd == "apply":
        if len(argv) < 3:
            print("usage: migrations apply <lab_path> [<adapter>]",
                  file=sys.stderr)
            return 2
        lab = Path(argv[2]).expanduser()
        if len(argv) >= 4:
            r = apply_pending(lab, argv[3])
            print(json.dumps({
                "adapter": r.adapter,
                "applied": list(r.applied),
                "final_version": r.final_version,
                "errors": list(r.errors),
            }, indent=2))
        else:
            results = apply_all_pending(lab)
            print(json.dumps([
                {
                    "adapter": r.adapter,
                    "applied": list(r.applied),
                    "final_version": r.final_version,
                    "errors": list(r.errors),
                } for r in results
            ], indent=2))
        return 0
    print(f"unknown cmd: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(_cli(sys.argv))
