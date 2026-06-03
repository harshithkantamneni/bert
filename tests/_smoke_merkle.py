"""Smoke test for core/merkle.py — Merkle hash tooling.

Per FINAL_implementation_plan_2026-05-07.md §5.1 H1 day 4 + L-01.

Verifies:
  1. Empty leaf list returns canonical empty-tree root
  2. Single leaf tree root is leaf-prefixed SHA-256
  3. Two-leaf tree matches RFC 6962 spec (node-prefix + concat)
  4. Odd-length tree duplicates last node (RFC 6962 canonical)
  5. file_root over JSONL handles missing/empty files
  6. file_root strips line whitespace before hashing (no editor drift)
  7. verify() accepts both bytes and hex string
  8. CLI helper returns hex root

Run: `.venv/bin/python tests/_smoke_merkle.py`
"""

import hashlib
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import merkle  # noqa: E402


def test_empty_leaves_returns_empty_string_hash() -> None:
    expected = hashlib.sha256(b"").digest()
    assert merkle.tree_root([]) == expected


def test_single_leaf_root_is_leaf_hash() -> None:
    leaf = merkle.leaf_hash(b"hello")
    assert merkle.tree_root([leaf]) == leaf


def test_leaf_hash_uses_leaf_prefix() -> None:
    """RFC 6962: leaf hash = sha256(0x00 || data)."""
    leaf = merkle.leaf_hash(b"x")
    expected = hashlib.sha256(b"\x00x").digest()
    assert leaf == expected


def test_two_leaf_tree() -> None:
    """RFC 6962: parent = sha256(0x01 || left || right)."""
    a = merkle.leaf_hash(b"a")
    b = merkle.leaf_hash(b"b")
    expected = hashlib.sha256(b"\x01" + a + b).digest()
    assert merkle.tree_root([a, b]) == expected


def test_odd_leaves_duplicates_last() -> None:
    """3-leaf tree: third leaf is duplicated to make pair at first level."""
    a = merkle.leaf_hash(b"a")
    b = merkle.leaf_hash(b"b")
    c = merkle.leaf_hash(b"c")
    # Level 1: pair(a,b) + pair(c,c)
    ab = hashlib.sha256(b"\x01" + a + b).digest()
    cc = hashlib.sha256(b"\x01" + c + c).digest()
    # Level 2: pair(ab, cc)
    expected = hashlib.sha256(b"\x01" + ab + cc).digest()
    assert merkle.tree_root([a, b, c]) == expected


def test_file_root_missing_file() -> None:
    expected = merkle.tree_root([])
    assert merkle.file_root("/nonexistent/path.jsonl") == expected


def test_file_root_empty_file() -> None:
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        assert merkle.file_root(path) == merkle.tree_root([])
    finally:
        Path(path).unlink(missing_ok=True)


def test_file_root_skips_blank_lines_and_strips() -> None:
    """Trailing newlines / blank lines should not affect the root —
    different editors save with/without trailing newline; the hash must
    be insensitive to that."""
    content_a = b'{"a": 1}\n{"b": 2}\n'                     # trailing \n
    content_b = b'{"a": 1}\n{"b": 2}'                        # no trailing \n
    content_c = b'{"a": 1}\n\n{"b": 2}\n'                    # blank line between
    content_d = b'  {"a": 1}\n{"b": 2}  \n'                 # whitespace around lines

    roots = []
    for content in (content_a, content_b, content_c, content_d):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            roots.append(merkle.file_root(path))
        finally:
            Path(path).unlink(missing_ok=True)

    assert all(r == roots[0] for r in roots), (
        "file_root should produce same hash for editor save-variants; "
        f"got {[r.hex()[:16] for r in roots]}"
    )


def test_verify_accepts_bytes_and_hex() -> None:
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        f.write(b'{"x": 1}\n')
        path = f.name
    try:
        root_bytes = merkle.file_root(path)
        root_hex = root_bytes.hex()
        assert merkle.verify(path, root_bytes)
        assert merkle.verify(path, root_hex)
        # Wrong root rejected
        assert not merkle.verify(path, b"\x00" * 32)
        assert not merkle.verify(path, "00" * 32)
    finally:
        Path(path).unlink(missing_ok=True)


def test_file_root_hex_returns_64_char_string() -> None:
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        f.write(b'{"x": 1}\n')
        path = f.name
    try:
        h = merkle.file_root_hex(path)
        assert isinstance(h, str)
        assert len(h) == 64
        # is hex
        bytes.fromhex(h)
    finally:
        Path(path).unlink(missing_ok=True)


def main() -> int:
    tests = [
        test_empty_leaves_returns_empty_string_hash,
        test_single_leaf_root_is_leaf_hash,
        test_leaf_hash_uses_leaf_prefix,
        test_two_leaf_tree,
        test_odd_leaves_duplicates_last,
        test_file_root_missing_file,
        test_file_root_empty_file,
        test_file_root_skips_blank_lines_and_strips,
        test_verify_accepts_bytes_and_hex,
        test_file_root_hex_returns_64_char_string,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
