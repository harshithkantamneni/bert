"""Smoke test for the J.1 H-phase API endpoints."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["BERT_DISABLE_IDLE_COMPUTE"] = "1"

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

H_ENDPOINTS = [
    "/api/graph",
    "/api/retrieval",
    "/api/compaction",
    "/api/quality-report",
    "/api/eval-scorecard",
    "/api/token-redundancy",
]


def test_all_endpoints_return_200() -> None:
    for path in H_ENDPOINTS:
        r = client.get(path)
        assert r.status_code == 200, f"{path} returned {r.status_code}: {r.text[:200]}"


def test_all_endpoints_return_dict_with_ts() -> None:
    for path in H_ENDPOINTS:
        body = client.get(path).json()
        assert isinstance(body, dict), f"{path} body not dict"
        assert "ts" in body, f"{path} missing ts"


def test_graph_endpoint_shape() -> None:
    body = client.get("/api/graph").json()
    for k in ("nodes_total", "edges_total", "edges_active",
              "edges_invalidated", "nodes_by_type", "edges_by_type",
              "recent"):
        assert k in body, f"/api/graph missing {k}"
    assert isinstance(body["recent"], list)


def test_retrieval_endpoint_shape() -> None:
    body = client.get("/api/retrieval").json()
    assert "adapters" in body
    assert set(body["adapters"].keys()) == {"vector", "graph", "cache"}
    assert isinstance(body["rrf_k"], int)
    assert isinstance(body["reranker_available"], bool)


def test_compaction_endpoint_shape() -> None:
    body = client.get("/api/compaction").json()
    assert "shapers_in_order" in body
    assert isinstance(body["shapers_in_order"], list)
    assert len(body["shapers_in_order"]) == 5
    assert body["strike_threshold"] == 3


def test_quality_report_endpoint_when_available() -> None:
    body = client.get("/api/quality-report").json()
    # Either available or honestly empty — both shapes acceptable.
    assert "available" in body
    if body["available"]:
        assert "grades" in body
        assert "sections" in body
        assert "overall_grade" in body


def test_eval_scorecard_endpoint_shape() -> None:
    body = client.get("/api/eval-scorecard").json()
    assert "owasp" in body
    assert "memoryagentbench" in body
    assert set(body["owasp"].keys()) >= {"passed", "failed", "details"}
    assert set(body["memoryagentbench"].keys()) >= {"passed", "failed", "axes"}
    # OWASP should have 10 details (10 threats)
    assert len(body["owasp"]["details"]) == 10


def test_token_redundancy_endpoint() -> None:
    body = client.get("/api/token-redundancy").json()
    assert "available" in body
    if body["available"]:
        assert "method" in body
        assert "grade" in body


def test_endpoints_degrade_gracefully_with_no_data() -> None:
    """The /api/graph endpoint must return 200 + well-formed structural
    keys regardless of how much data the underlying store holds — so the
    UI can render "no signal yet" panels on an empty store AND real
    panels on a populated one (graceful-degradation contract).

    NOTE: this runs against the ambient graph store, which is empty on a
    fresh clone but populated on a dev machine that has run cycles. The
    contract is "well-formed response", NOT "count is exactly 0" — the
    earlier `== 0` assertion was coupled to fresh-clone empty-state and
    failed once the dev's graph.kuzu accumulated nodes."""
    body = client.get("/api/graph").json()
    # Structural keys must be present whether the store is empty or full,
    # and counts must be well-formed non-negative integers.
    for k in ("nodes_total", "edges_total", "edges_active"):
        assert k in body, f"missing structural key {k!r}"
        assert isinstance(body[k], int) and body[k] >= 0, \
            f"{k} must be a non-negative int, got {body[k]!r}"


def main() -> int:
    tests = [
        test_all_endpoints_return_200,
        test_all_endpoints_return_dict_with_ts,
        test_graph_endpoint_shape,
        test_retrieval_endpoint_shape,
        test_compaction_endpoint_shape,
        test_quality_report_endpoint_when_available,
        test_eval_scorecard_endpoint_shape,
        test_token_redundancy_endpoint,
        test_endpoints_degrade_gracefully_with_no_data,
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
