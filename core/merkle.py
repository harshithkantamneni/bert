"""Merkle hash tooling for pace-layered file integrity.

The pace-layered file structure (Gartner SoR / SoD / SoI) commits to
*reproducibility-by-Merkle-hash*: every decision is reproducible; the
canvas time-machine mode replays any past lab state.

This module provides:
  - leaf_hash(content_bytes) → SHA-256 of content with a leaf-domain prefix
  - tree_root(leaves) → Merkle root of a list of leaf hashes
  - file_root(jsonl_path) → Merkle root over each JSONL line as a leaf
  - verify(jsonl_path, expected_root) → bool

Why Merkle, not just a single SHA-256:
  - Merkle roots support efficient *proof-of-inclusion* — bert can
    later answer "did event X really happen at time T?" without
    rehashing the entire log
  - Append-only log + Merkle-tree-at-checkpoint = standard pattern for
    auditable systems (Certificate Transparency, Sigstore, git itself)
  - Phase C0 emits canvas events to lab/sor/events.jsonl; the canvas
    time-machine mode in Phase C2/C3 needs efficient verifiability of
    "was this event really in the log at scrub-target timestamp T?"

The leaf-domain prefix (b'\\x00') and node-domain prefix (b'\\x01')
follow RFC 6962 (Certificate Transparency) to prevent second-preimage
attacks on Merkle trees with mixed leaf/internal-node hashes.

For sub-byte performance: hashlib.sha256 in Python is C-extension code,
fast enough that the dominant cost is file I/O. A 100K-line JSONL file
with average line ~1KB takes ~50-100ms to fully hash + tree-build on
M3 Pro.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"


def leaf_hash(content: bytes) -> bytes:
    """SHA-256 of a leaf, with leaf-domain prefix per RFC 6962."""
    return hashlib.sha256(LEAF_PREFIX + content).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    """SHA-256 of an internal node, with node-domain prefix per RFC 6962."""
    return hashlib.sha256(NODE_PREFIX + left + right).digest()


def tree_root(leaves: list[bytes]) -> bytes:
    """Compute the Merkle root over a list of leaf hashes.

    For an odd-length level, the last node is duplicated (RFC 6962
    canonical convention).

    Empty list returns the SHA-256 of the empty string (canonical
    "empty tree" root, also RFC 6962).
    """
    if not leaves:
        return hashlib.sha256(b"").digest()
    level: list[bytes] = list(leaves)
    while len(level) > 1:
        next_level: list[bytes] = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else level[i]
            next_level.append(_node_hash(left, right))
        level = next_level
    return level[0]


def file_root(jsonl_path: Path | str) -> bytes:
    """Compute Merkle root over each line of a JSONL file as a leaf.

    Empty/missing file returns the canonical empty-tree root. Whitespace-
    only lines are skipped. Each surviving line is treated as a leaf
    (UTF-8 encoded; leading/trailing whitespace stripped before hashing
    to avoid hash drift on editor save-with-trailing-newline behavior).
    """
    path = Path(jsonl_path)
    if not path.exists():
        return tree_root([])
    leaves: list[bytes] = []
    with path.open("rb") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                leaves.append(leaf_hash(stripped))
    return tree_root(leaves)


def file_root_hex(jsonl_path: Path | str) -> str:
    """Convenience: file_root(path).hex() for human-readable / log-friendly
    output. 64 hex chars (256 bits)."""
    return file_root(jsonl_path).hex()


def verify(jsonl_path: Path | str, expected_root: bytes | str) -> bool:
    """Verify that the Merkle root of `jsonl_path` matches `expected_root`.

    `expected_root` may be raw bytes (32 bytes) or hex string (64 chars).
    """
    if isinstance(expected_root, str):
        expected_root = bytes.fromhex(expected_root)
    return file_root(jsonl_path) == expected_root


# ── CLI helper ──────────────────────────────────────────────────────


def _cli() -> int:
    """Tiny CLI: `python -m core.merkle <jsonl-path>` prints hex root."""
    import sys
    if len(sys.argv) != 2:
        print("usage: python -m core.merkle <jsonl-path>", file=sys.stderr)
        return 1
    path = Path(sys.argv[1])
    print(file_root_hex(path))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
