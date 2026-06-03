"""Error-path robustness tests — what happens when things go wrong.

Quality-first: smoke tests that verify GRACEFUL DEGRADATION on
realistic production failure modes. Each test deliberately breaks
something and asserts that the agent loop's primary path is
preserved — log + continue, never crash.

Failure modes covered (drawn from real production patterns, not
contrived edge cases):

  1. memory.db is corrupted (truncated / not-a-sqlite-db)
  2. quota.db is locked by another writer (concurrent SQLite access)
  3. observability OBS_DIR is read-only (permissions / disk full)
  4. cycle log JSONL has malformed lines
  5. result_packet schema file missing
  6. Python hook script has a syntax error
  7. brief_assembler input files are unreadable
  8. consolidator runs against a tree with no procedures.md

These all SHOULD already work (the modules wrap risky paths in
try/except + LOG.warning per quality-first audit). This suite is
the regression net — if a future refactor removes the safety net,
these tests catch it.

Run: `.venv/bin/python tests/_smoke_robustness.py`
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


# ── Memory.db corruption ────────────────────────────────────────────


def test_memory_handles_corrupted_db() -> None:
    """memory_search / index against a corrupted db file should NOT
    crash. The function should either rebuild or return empty.

    We stub the embedder before importing memory so this test stays
    fast (~5s) on a cold sentence-transformers cache. The test is
    about db-corruption handling, not embedder behavior — using a
    8-dim deterministic stub is sufficient and avoids the ~30s
    transformers cold-load on disk-pressured machines."""
    import struct
    def fake_embed_batch(texts):
        # Deterministic zero embedding, packed in the format memory
        # expects (list of float32 byte-blobs of EMBED_DIM floats).
        from core.memory import EMBED_DIM
        return [struct.pack(f"{EMBED_DIM}f", *([0.0] * EMBED_DIM))
                for _ in texts]
    # Pre-stub: replace the embedder before sentence_transformers
    # cold-load can run. Avoids the ~30s import on cold cache.
    with mock.patch("core.memory._embed_batch", side_effect=fake_embed_batch):
        from core import memory
        tmp = Path(tempfile.mkdtemp(prefix="bert_robust_mem_"))
        bad_db = tmp / "memory.db"
        # Write garbage that's not a valid SQLite database
        bad_db.write_bytes(b"this is not a sqlite db" * 200)
        orig_db = memory.MEMORY_DB if hasattr(memory, "MEMORY_DB") else None
        try:
            try:
                memory.MEMORY_DB = bad_db
            except Exception:
                pass
            try:
                results = memory.search("anything", k=3)
                assert isinstance(results, list)
            except (sqlite3.DatabaseError, sqlite3.OperationalError):
                pass  # clean error is acceptable
        finally:
            if orig_db is not None:
                try:
                    memory.MEMORY_DB = orig_db
                except Exception:
                    pass


# ── Quota.db sqlite locking ─────────────────────────────────────────


def test_quota_survives_concurrent_writer() -> None:
    """When two processes contend for quota.db, the sqlite WAL mode
    should let them both succeed (or fail cleanly with a timeout, not
    deadlock). Simulate by holding an exclusive transaction in one
    thread and calling check_quota concurrently."""
    from core import quota
    tmp = Path(tempfile.mkdtemp(prefix="bert_robust_quota_"))
    quota.QUOTA_DB = tmp / "quota.db"
    # Initialize the schema by recording one event
    quota.record_call("test", prompt_tokens=1, completion_tokens=1)

    # Hold an exclusive transaction in a worker thread
    holder_done = threading.Event()
    worker_in = threading.Event()

    def hold_lock():
        with sqlite3.connect(quota.QUOTA_DB, timeout=10.0) as conn:
            conn.execute("BEGIN EXCLUSIVE")
            conn.execute("INSERT INTO events(provider, ts) VALUES (?, ?)",
                         ("blocker", time.time()))
            worker_in.set()
            # Hold for 0.5s
            time.sleep(0.5)
            conn.commit()
            holder_done.set()

    t = threading.Thread(target=hold_lock, daemon=True)
    t.start()
    worker_in.wait(2.0)  # wait until the worker has the lock

    # check_quota should NOT hang — it has a 5s timeout in _connect.
    # It might block briefly but should return within 1s or so given
    # WAL mode lets readers proceed.
    t0 = time.monotonic()
    try:
        ok, reason = quota.check_quota("test")
        elapsed = time.monotonic() - t0
        assert elapsed < 6.0, f"check_quota hung; took {elapsed:.1f}s"
    except sqlite3.OperationalError:
        # Acceptable: clean timeout error, not a hang
        pass

    holder_done.wait(2.0)
    t.join(timeout=1.0)


# ── Observability OBS_DIR not writable ──────────────────────────────


def test_observability_emit_handles_readonly_dir() -> None:
    """When OBS_DIR is not writable (read-only mount, permission
    denied), emit should log a warning and continue, not crash."""
    from core import observability
    tmp = Path(tempfile.mkdtemp(prefix="bert_robust_obs_"))
    # Make the dir read-only AFTER creating it
    obs_dir = tmp / "obs"
    obs_dir.mkdir()
    obs_dir.chmod(0o555)  # r-x only, no write
    observability.OBS_DIR = obs_dir
    try:
        # emit() should not raise even though it can't write
        observability.emit("circuit_breaker_event", {"provider": "test", "kind": "rpm"})
        # No assertion on side effect — primary requirement is no exception
    finally:
        obs_dir.chmod(0o755)  # restore so cleanup works


# ── Cycle log JSONL has malformed lines ─────────────────────────────


def test_evaluator_skips_malformed_jsonl_lines() -> None:
    """Cycle log files can have malformed lines (truncated last write,
    encoding issues). _load_cycle_events should skip and continue."""
    from core import evaluator
    tmp = Path(tempfile.mkdtemp(prefix="bert_robust_eval_"))
    logs_dir = tmp / "logs"
    logs_dir.mkdir()
    bad_log = logs_dir / "cycle_42_20260508.jsonl"
    bad_log.write_text(
        '{"kind": "model_response", "tokens_in": 100, "tokens_out": 50}\n'
        'this is not json — corrupt line\n'
        '{"kind": "tool_result", "tool": "Read"}\n'
        '{"truncated json with no closing brace\n'
    )
    orig = evaluator.LOGS_DIR
    evaluator.LOGS_DIR = logs_dir
    try:
        events = evaluator._load_cycle_events(42)
        # Should get the 2 valid events; skip the 2 malformed ones
        assert len(events) == 2, f"expected 2 valid events; got {len(events)}: {events}"
        assert events[0]["kind"] == "model_response"
        assert events[1]["kind"] == "tool_result"
    finally:
        evaluator.LOGS_DIR = orig


# ── Schema file missing ─────────────────────────────────────────────


def test_schema_correction_returns_none_when_schema_missing() -> None:
    """If schemas/result_packet.json is missing/unreadable, the
    schema-correction retry must return None, not crash."""
    from core import subagent
    orig_schemas = subagent.SCHEMAS_DIR
    tmp = Path(tempfile.mkdtemp(prefix="bert_robust_schema_"))
    subagent.SCHEMAS_DIR = tmp / "nonexistent"
    try:
        out = subagent._attempt_schema_correction(
            {"model": "nvidia/x"},
            {"role": "researcher", "cycle": 1},
            ["some-error"],
        )
        assert out is None
    finally:
        subagent.SCHEMAS_DIR = orig_schemas


# ── Hook script with Python syntax error ────────────────────────────


def test_python_hook_with_syntax_error_doesnt_break_fire() -> None:
    """A hook script that's broken Python should fail with a clean
    non-zero exit, not crash hooks.fire()."""
    from core import hooks
    tmp = Path(tempfile.mkdtemp(prefix="bert_robust_hook_"))
    hooks.HOOKS_DIR = tmp / "hooks"
    d = hooks.HOOKS_DIR / "PreToolUse"
    d.mkdir(parents=True)
    bad = d / "broken.py"
    bad.write_text("#!/usr/bin/env python3\nthis is not valid python\n")
    bad.chmod(0o755)

    rep = hooks.fire("PreToolUse", {"tool": "Read"}, timeout_secs=5)
    assert len(rep.outcomes) == 1
    o = rep.outcomes[0]
    assert o.exit_code != 0, "broken Python hook should exit non-zero"
    assert "Error" in o.stderr or "syntax" in o.stderr.lower() or o.stderr, (
        f"expected python error in stderr; got {o.stderr!r}"
    )
    assert not rep.all_passed


# ── brief_assembler with unreadable inputs ──────────────────────────


def test_brief_assembler_handles_missing_files() -> None:
    """When the memory files don't exist, brief_assembler should
    write a brief with placeholders instead of crashing."""
    from core import brief_assembler
    tmp = Path(tempfile.mkdtemp(prefix="bert_robust_brief_"))
    orig_lab = brief_assembler.LAB_ROOT
    orig_mem = brief_assembler.MEMORIES_DIR
    orig_state = brief_assembler.STATE_DIR
    orig_brief = brief_assembler.BRIEF_PATH
    brief_assembler.LAB_ROOT = tmp
    brief_assembler.MEMORIES_DIR = tmp / "memories"
    brief_assembler.STATE_DIR = tmp / "state"
    brief_assembler.BRIEF_PATH = tmp / "context_brief.md"
    try:
        # Don't create any input files
        path, stats = brief_assembler.assemble_brief()
        assert path.exists()
        text = path.read_text()
        # Should contain placeholders for missing files
        assert "missing" in text or "_(no" in text
        assert stats["total_chars"] > 0
    finally:
        brief_assembler.LAB_ROOT = orig_lab
        brief_assembler.MEMORIES_DIR = orig_mem
        brief_assembler.STATE_DIR = orig_state
        brief_assembler.BRIEF_PATH = orig_brief


# ── consolidator with missing tier files ────────────────────────────


def test_consolidator_handles_missing_procedures() -> None:
    """When memories/procedures.md doesn't exist, promote_statuses
    should return [] not crash."""
    from core import consolidator
    tmp = Path(tempfile.mkdtemp(prefix="bert_robust_cons_"))
    orig = consolidator.MEMORIES_DIR
    consolidator.MEMORIES_DIR = tmp / "memories"
    try:
        promotions = consolidator.promote_statuses()
        assert promotions == []
    finally:
        consolidator.MEMORIES_DIR = orig


# ── Telegram approver registered but bot offline ────────────────────


def test_telegram_approver_timeout_returns_deny() -> None:
    """Quality-first: when the Telegram bot doesn't respond (timeout),
    request_approval must return the original deny — never hang
    indefinitely."""
    sys.path.insert(0, str(LAB_ROOT / "bot"))
    import approval  # type: ignore

    from core.types import PermissionDecision, ToolCall

    tmp = Path(tempfile.mkdtemp(prefix="bert_robust_approval_"))
    approval.APPROVAL_DIR = tmp / "approvals"
    approval.PENDING_DIR = approval.APPROVAL_DIR / "pending"
    approval.DECIDED_DIR = approval.APPROVAL_DIR / "decided"

    deny = PermissionDecision(
        allowed=False, reason="P-011", requires_telegram_approval=True,
        is_destructive=True,
    )
    call = ToolCall(id="t", name="Bash", arguments={"command": "rm -rf /"})

    t0 = time.monotonic()
    out = approval.request(call, deny, timeout_secs=2)
    elapsed = time.monotonic() - t0

    assert not out.allowed, "timeout must result in deny"
    assert 1.5 <= elapsed < 4.0, f"timeout not enforced; took {elapsed:.1f}s"


def main() -> int:
    tests = [
        test_memory_handles_corrupted_db,
        test_quota_survives_concurrent_writer,
        test_observability_emit_handles_readonly_dir,
        test_evaluator_skips_malformed_jsonl_lines,
        test_schema_correction_returns_none_when_schema_missing,
        test_python_hook_with_syntax_error_doesnt_break_fire,
        test_brief_assembler_handles_missing_files,
        test_consolidator_handles_missing_procedures,
        test_telegram_approver_timeout_returns_deny,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {t.__name__} (exception)")
            print(f"        {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
    print(f"\nAll {len(tests)} robustness tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
