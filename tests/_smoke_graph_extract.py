"""Smoke test for F11 — findings → graph_store ingestion.

When a Write tool call lands a finding file, an extractor walks the
markdown and upserts entity nodes + relation edges into the active
lab's graph.db. Atlas strata ring reads from this store.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


def test_finding_emits_candidate_and_mission_nodes() -> None:
    from core.lab_context import set_active_lab_path, reset_active_lab_path
    from core.tools import _write
    from core import graph_store

    with tempfile.TemporaryDirectory() as td:
        lab = Path(td) / "scratch-lab"
        lab.mkdir(parents=True)
        token = set_active_lab_path(lab)
        try:
            _write(
                "findings/bert_run_C7_researcher.md",
                "# Researcher Finding\n\nThe seed asks about post-transformer "
                "models. Three candidates: **Mamba**, **RWKV**, **Hyena**.\n",
            )
            counts = graph_store.count()
            assert counts["nodes_total"] >= 4, \
                f"expected ≥4 nodes (1 candidate + 1 mission + 3 named); got {counts}"
            ntypes = counts.get("nodes_by_type", {})
            assert ntypes.get("Mission", 0) == 1, \
                f"need exactly 1 Mission node for the cycle; got {ntypes}"
            assert ntypes.get("Candidate", 0) >= 4, \
                f"need 1 finding candidate + 3 named candidates; got {ntypes}"
        finally:
            reset_active_lab_path(token)


def test_finding_emits_evidenced_by_edge() -> None:
    from core.lab_context import set_active_lab_path, reset_active_lab_path
    from core.tools import _write
    from core import graph_store

    with tempfile.TemporaryDirectory() as td:
        lab = Path(td) / "scratch-lab"
        lab.mkdir(parents=True)
        token = set_active_lab_path(lab)
        try:
            _write(
                "findings/bert_run_C9_strategist.md",
                "# Strategist Recommendation\n\nGo with **Mamba** for now.\n",
            )
            counts = graph_store.count()
            etypes = counts.get("edges_by_type", {})
            assert etypes.get("EVIDENCED_BY", 0) >= 1, \
                f"finding → Mission edge missing; got {etypes}"
        finally:
            reset_active_lab_path(token)


def test_finding_with_arxiv_link_creates_source_node() -> None:
    from core.lab_context import set_active_lab_path, reset_active_lab_path
    from core.tools import _write
    from core import graph_store

    with tempfile.TemporaryDirectory() as td:
        lab = Path(td) / "scratch-lab"
        lab.mkdir(parents=True)
        token = set_active_lab_path(lab)
        try:
            _write(
                "findings/bert_run_C11_researcher.md",
                "# Researcher Finding\n\nFor Mamba see "
                "[the paper](https://arxiv.org/abs/2312.00752).\n",
            )
            counts = graph_store.count()
            ntypes = counts.get("nodes_by_type", {})
            etypes = counts.get("edges_by_type", {})
            assert ntypes.get("Source", 0) >= 1, \
                f"arxiv link should create Source node; got {ntypes}"
            assert etypes.get("REFERENCES", 0) >= 1, \
                f"finding → Source REFERENCES edge missing; got {etypes}"
        finally:
            reset_active_lab_path(token)


def test_graph_store_routes_per_lab() -> None:
    """Two different active-lab contexts must yield two different
    graph.db files — no cross-lab contamination."""
    from core.lab_context import set_active_lab_path, reset_active_lab_path
    from core.tools import _write
    from core import graph_store

    with tempfile.TemporaryDirectory() as td:
        lab_a = Path(td) / "lab-a"
        lab_a.mkdir()
        lab_b = Path(td) / "lab-b"
        lab_b.mkdir()

        t1 = set_active_lab_path(lab_a)
        try:
            _write("findings/lab_a_finding.md", "# A\n\nA finding for lab A.")
            counts_a = graph_store.count()
        finally:
            reset_active_lab_path(t1)

        t2 = set_active_lab_path(lab_b)
        try:
            counts_b = graph_store.count()
        finally:
            reset_active_lab_path(t2)

        assert counts_a["nodes_total"] >= 1, "lab-a should have nodes"
        assert counts_b["nodes_total"] == 0, \
            f"lab-b should be empty; got {counts_b}"


def test_extractor_idempotent_on_same_finding_id() -> None:
    """Re-writing the same finding path should not duplicate nodes."""
    from core.lab_context import set_active_lab_path, reset_active_lab_path
    from core.tools import _write
    from core import graph_store

    with tempfile.TemporaryDirectory() as td:
        lab = Path(td) / "scratch-lab"
        lab.mkdir(parents=True)
        token = set_active_lab_path(lab)
        try:
            _write("findings/repeat.md", "# Repeat\n\nFirst write — **X**.")
            c1 = graph_store.count()["nodes_total"]
            _write("findings/repeat.md", "# Repeat\n\nSecond write — **X**.")
            c2 = graph_store.count()["nodes_total"]
            assert c1 == c2, \
                f"node count should stay equal on re-write; {c1} → {c2}"
        finally:
            reset_active_lab_path(token)


def main() -> int:
    tests = [
        test_finding_emits_candidate_and_mission_nodes,
        test_finding_emits_evidenced_by_edge,
        test_finding_with_arxiv_link_creates_source_node,
        test_graph_store_routes_per_lab,
        test_extractor_idempotent_on_same_finding_id,
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
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
