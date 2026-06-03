"""Smoke test for H.3 — KG validity-window edges (Graphiti pattern)."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import graph_store as gs  # noqa: E402


def _isolate() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_validity_")) / "graph.db"
    gs.DB_PATH = tmp


def test_edge_no_validity_returns_for_all_times() -> None:
    _isolate()
    gs.add_node("A", "Decision", "A")
    gs.add_node("B", "Decision", "B")
    gs.add_edge("A", "B", "REFERENCES")
    now = time.time()
    # All `at` times should return the edge — no validity window set
    assert len(gs.neighbors("A", at=now - 86400)) == 1
    assert len(gs.neighbors("A", at=now)) == 1
    assert len(gs.neighbors("A", at=now + 86400)) == 1
    # No `at` filter → edge visible
    assert len(gs.neighbors("A")) == 1


def test_edge_with_validity_window() -> None:
    _isolate()
    gs.add_node("A", "Decision", "A")
    gs.add_node("B", "Decision", "B")
    now = time.time()
    # Edge valid from 1h ago, still valid (valid_to None)
    gs.add_edge("A", "B", "REFERENCES", valid_from=now - 3600)
    # Query at "now" → visible
    assert len(gs.neighbors("A", at=now)) == 1
    # Query at "2h ago" → before valid_from → not visible
    assert len(gs.neighbors("A", at=now - 7200)) == 0


def test_invalidate_edge_closes_window() -> None:
    _isolate()
    gs.add_node("A", "Decision", "A")
    gs.add_node("B", "Decision", "B")
    now = time.time()
    gs.add_edge("A", "B", "REFERENCES", valid_from=now - 3600)
    assert len(gs.neighbors("A", at=now)) == 1
    # Invalidate
    assert gs.invalidate_edge("A", "B", "REFERENCES", at=now) is True
    # Query "now" → just-closed (valid_to == now, predicate is >, so not visible)
    assert len(gs.neighbors("A", at=now + 1)) == 0
    # Query "30 min ago" (within window) → STILL visible (historical query)
    assert len(gs.neighbors("A", at=now - 1800)) == 1


def test_invalidate_returns_false_when_already_closed() -> None:
    _isolate()
    gs.add_node("A", "Decision", "A")
    gs.add_node("B", "Decision", "B")
    now = time.time()
    gs.add_edge("A", "B", "REFERENCES", valid_from=now - 3600,
                 valid_to=now - 1800)
    # Already closed in the past; re-invalidating "now" is a no-op
    assert gs.invalidate_edge("A", "B", "REFERENCES", at=now) is False


def test_subgraph_respects_at_filter() -> None:
    _isolate()
    now = time.time()
    for nid in ("A", "B", "C", "D"):
        gs.add_node(nid, "Decision", nid)
    # A→B always valid
    gs.add_edge("A", "B", "REFERENCES")
    # B→C only valid in the past (already closed)
    gs.add_edge("B", "C", "REFERENCES", valid_from=now - 3600,
                 valid_to=now - 1800)
    # C→D currently valid
    gs.add_edge("C", "D", "REFERENCES", valid_from=now - 1000)

    # Query "now": A→B reachable, B→C closed (won't traverse), so D unreachable
    nodes, edges = gs.subgraph(["A"], hops=4, at=now)
    nids = {n.id for n in nodes}
    assert "A" in nids and "B" in nids
    # Because we can't traverse B→C at `now`, C and D should not appear
    assert "C" not in nids
    assert "D" not in nids

    # Query "33 min ago" (within B→C's valid window [now-3600, now-1800)):
    # A→B and B→C both valid → reach C; C→D was not yet valid → don't reach D
    nodes_past, _ = gs.subgraph(["A"], hops=4, at=now - 2000)
    past_ids = {n.id for n in nodes_past}
    assert "C" in past_ids
    assert "D" not in past_ids


def test_no_at_filter_returns_all_edges() -> None:
    """Backward compat: queries without `at` return everything,
    closed or not."""
    _isolate()
    now = time.time()
    gs.add_node("X", "Decision", "X")
    gs.add_node("Y", "Decision", "Y")
    gs.add_edge("X", "Y", "REFERENCES", valid_from=now - 3600,
                 valid_to=now - 100)  # Closed
    # No `at` → still visible
    assert len(gs.neighbors("X")) == 1


def test_validity_columns_added_via_alter_table() -> None:
    """Forward-compat: ALTER TABLE on first connect adds the cols."""
    _isolate()
    # First connect creates with cols
    with gs._connect() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(edges)").fetchall()}
    assert "valid_from" in cols
    assert "valid_to" in cols


def main() -> int:
    tests = [
        test_edge_no_validity_returns_for_all_times,
        test_edge_with_validity_window,
        test_invalidate_edge_closes_window,
        test_invalidate_returns_false_when_already_closed,
        test_subgraph_respects_at_filter,
        test_no_at_filter_returns_all_edges,
        test_validity_columns_added_via_alter_table,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
