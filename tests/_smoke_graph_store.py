"""Smoke test for core/graph_store.py."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import graph_store  # noqa: E402


def _isolate() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bert_graph_")) / "graph.db"
    graph_store.DB_PATH = tmp


def test_add_node_and_get() -> None:
    _isolate()
    graph_store.add_node("D-10", "Decision", "Quaker pipeline ratification",
                          props={"date": "2026-05-07"})
    n = graph_store.get_node("D-10")
    assert n is not None
    assert n.type == "Decision"
    assert n.props["date"] == "2026-05-07"


def test_unknown_node_type_raises() -> None:
    _isolate()
    try:
        graph_store.add_node("X-1", "NotAType", "x")
    except ValueError as e:
        assert "unknown node type" in str(e)
        return
    raise AssertionError("expected ValueError")


def test_add_edge_neighbors_both_directions() -> None:
    _isolate()
    for nid in ("D-10", "FALS-9-1", "P-001"):
        kind = {"D-10": "Decision", "FALS-9-1": "Falsifier", "P-001": "Heuristic"}[nid]
        graph_store.add_node(nid, kind, nid)
    graph_store.add_edge("D-10", "FALS-9-1", "EVIDENCED_BY")
    graph_store.add_edge("P-001", "D-10", "REFERENCES")

    out_d10 = graph_store.neighbors("D-10", direction="out")
    in_d10 = graph_store.neighbors("D-10", direction="in")
    both_d10 = graph_store.neighbors("D-10", direction="both")
    assert len(out_d10) == 1 and out_d10[0].dst == "FALS-9-1"
    assert len(in_d10) == 1 and in_d10[0].src == "P-001"
    assert len(both_d10) == 2


def test_edge_type_filter() -> None:
    _isolate()
    for nid in ("D-1", "D-2", "D-3"):
        graph_store.add_node(nid, "Decision", nid)
    graph_store.add_edge("D-1", "D-2", "SUPERSEDES")
    graph_store.add_edge("D-1", "D-3", "REFERENCES")
    out_sup = graph_store.neighbors("D-1", edge_type="SUPERSEDES", direction="out")
    assert len(out_sup) == 1
    assert out_sup[0].dst == "D-2"


def test_subgraph_bfs() -> None:
    _isolate()
    for nid in ("A", "B", "C", "D"):
        graph_store.add_node(nid, "Decision", nid)
    graph_store.add_edge("A", "B", "REFERENCES")
    graph_store.add_edge("B", "C", "REFERENCES")
    graph_store.add_edge("C", "D", "REFERENCES")
    nodes, edges = graph_store.subgraph(["A"], hops=2)
    ids = {n.id for n in nodes}
    assert {"A", "B", "C"} <= ids
    assert "D" not in ids  # 3 hops away, beyond max


def test_shortest_path() -> None:
    _isolate()
    for nid in ("X", "Y", "Z"):
        graph_store.add_node(nid, "Decision", nid)
    graph_store.add_edge("X", "Y", "REFERENCES")
    graph_store.add_edge("Y", "Z", "REFERENCES")
    path = graph_store.shortest_path("X", "Z", max_hops=3)
    assert path == ["X", "Y", "Z"]


def test_count_stats() -> None:
    _isolate()
    graph_store.add_node("D-1", "Decision", "D-1")
    graph_store.add_node("P-1", "Heuristic", "P-1")
    graph_store.add_edge("P-1", "D-1", "REFERENCES")
    s = graph_store.count()
    assert s["nodes_total"] == 2
    assert s["edges_total"] == 1
    assert s["nodes_by_type"]["Decision"] == 1
    assert s["edges_by_type"]["REFERENCES"] == 1


def main() -> int:
    tests = [
        test_add_node_and_get,
        test_unknown_node_type_raises,
        test_add_edge_neighbors_both_directions,
        test_edge_type_filter,
        test_subgraph_bfs,
        test_shortest_path,
        test_count_stats,
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
