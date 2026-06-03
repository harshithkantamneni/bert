"""Tests for note.capture — cycle 1 acceptance."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

# Import the sibling module
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import note  # noqa: E402


def test_capture_writes_text() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.md"
        note.capture("hello world", path=p)
        content = p.read_text()
        assert "hello world" in content
        assert "ts:" in content
        assert "tags: []" in content


def test_capture_with_tags() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.md"
        note.capture("thought", tags=["deep", "idea"], path=p)
        content = p.read_text()
        assert "tags: [deep, idea]" in content


def test_capture_rejects_empty() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.md"
        for empty in ("", "   ", "\t\n"):
            try:
                note.capture(empty, path=p)
                raise AssertionError(f"expected ValueError for {empty!r}")
            except ValueError:
                pass


def test_capture_appends() -> None:
    """Multiple captures append to the same file."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.md"
        note.capture("first", path=p)
        note.capture("second", path=p)
        content = p.read_text()
        assert "first" in content
        assert "second" in content


def test_capture_latency_under_100ms() -> None:
    """The success metric: capture in <100ms."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.md"
        # Warm up (first call may import stdlib lazily)
        note.capture("warmup", path=p)
        # Measure
        t0 = time.perf_counter()
        note.capture("measure me", path=p)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 100, f"capture took {elapsed_ms:.1f}ms; target <100ms"


def main() -> int:
    tests = [
        test_capture_writes_text,
        test_capture_with_tags,
        test_capture_rejects_empty,
        test_capture_appends,
        test_capture_latency_under_100ms,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
