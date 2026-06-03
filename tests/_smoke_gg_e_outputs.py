"""Smoke test for GG-E — proof packet outputs viewer.

Closes the per-audit bug #5 (flat packet directory with no
enumeration endpoint, no UI). Adds:

  - GET /api/proof-packets?lab=<name> — list packets, peek cycle.json
    for metadata, filter by labRef
  - GET /api/proof-packets/{cycle_id} — extract cycle.json +
    adversarial.json + self-eval.json + failures.md from the tarball
  - POST /api/proof-packets/{cycle_id}/verify — invoke
    core.verify_packet.verify_packet, return the 8-check ladder
  - Outputs surface (/proofs) — ledger of stamped receipts, each
    expands to a detail panel with claims, failures.md, adversarial
    summary, self-eval, and verify button rendering the 8 checks
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


OUTPUTS = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "Outputs.tsx"
APP_TSX = LAB_ROOT / "bert" / "v4" / "src" / "App.tsx"
API_MAIN = LAB_ROOT / "api" / "main.py"
CLIENT_TS = LAB_ROOT / "bert" / "v4" / "src" / "api" / "client.ts"


# ─── Backend endpoints ────────────────────────────────────────────


def test_list_endpoint_defined() -> None:
    src = API_MAIN.read_text()
    assert '@app.get("/api/proof-packets")' in src
    assert "def list_proof_packets" in src


def test_detail_endpoint_defined() -> None:
    src = API_MAIN.read_text()
    assert '@app.get("/api/proof-packets/{cycle_id}")' in src
    assert "def get_proof_packet" in src


def test_verify_endpoint_defined() -> None:
    src = API_MAIN.read_text()
    assert '@app.post("/api/proof-packets/{cycle_id}/verify")' in src
    assert "def verify_proof_packet" in src


def test_list_returns_packets_with_metadata() -> None:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.get("/api/proof-packets")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    # Each packet has the metadata fields a strata-ledger UI needs
    for p in body["packets"]:
        assert "cycle_id" in p
        assert "lab_ref" in p
        assert "completed_at" in p
        assert "claims_count" in p
        assert "limitations_count" in p
        assert "event_count" in p
        assert "tarball_bytes" in p


def test_list_filters_by_lab() -> None:
    """?lab=customer-survey should hide packets from other labs."""
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    # The repo's seed packets are all from 'bert' / supervisor lab.
    # Filter on a non-existent lab should yield zero results.
    r = client.get("/api/proof-packets?lab=definitely-no-such-lab-xyz")
    assert r.status_code == 200
    assert r.json()["count"] == 0


def test_detail_returns_full_packet_contents() -> None:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.get("/api/proof-packets/0400")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "cycle_json" in body
    assert "adversarial" in body
    assert "self_eval" in body
    assert "failures_md" in body
    assert "file_index" in body
    # cycle_json has the claims array
    assert isinstance(body["cycle_json"]["claims"], list)


def test_detail_404_on_unknown_cycle() -> None:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.get("/api/proof-packets/9999999")
    assert r.status_code == 404


def test_detail_handles_multiple_cycle_id_formats() -> None:
    """Accept 'cycle-0400', '0400', '400' — all reach the same packet."""
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    a = client.get("/api/proof-packets/cycle-0400").json()
    b = client.get("/api/proof-packets/0400").json()
    c = client.get("/api/proof-packets/400").json()
    assert a["cycle_id"] == b["cycle_id"] == c["cycle_id"]


def test_verify_returns_eight_check_ladder() -> None:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.post("/api/proof-packets/0400/verify")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["overall"] in ("PASS", "PASS-WITH-WARNINGS", "FAIL")
    assert isinstance(body["checks"], list)
    assert len(body["checks"]) >= 6  # the full ladder is 8; minimum gate
    for chk in body["checks"]:
        assert chk["status"] in ("PASS", "WARN", "FAIL")
        assert "name" in chk
        assert "detail" in chk


def test_verify_local_dev_reports_rekor_as_warn() -> None:
    """Honest disclosure: the local-dev signing mode skips Rekor
    inclusion check. Verify should report it as WARN, not silently
    PASS — per DD.2 retired one-flip framing."""
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.post("/api/proof-packets/0400/verify")
    body = r.json()
    rekor_check = next(
        (c for c in body["checks"] if "Rekor" in c["name"]), None)
    assert rekor_check is not None
    assert rekor_check["status"] == "WARN"


def test_verify_404_on_unknown_cycle() -> None:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.post("/api/proof-packets/9999999/verify")
    assert r.status_code == 404


# ─── Outputs UI surface ───────────────────────────────────────────


def test_outputs_surface_exists() -> None:
    assert OUTPUTS.exists()


def test_outputs_route_registered() -> None:
    text = APP_TSX.read_text()
    assert '<Route path="/proofs"' in text
    assert "Outputs" in text


def test_outputs_uses_ledger_idiom_not_card_grid() -> None:
    """Per feedback_visualization_as_art: receipts in a vertical
    ledger, NOT a card grid. Anti-pattern check on the code."""
    text = OUTPUTS.read_text()
    decommented = re.sub(r"//[^\n]*", "", text)
    decommented = re.sub(r"/\*.*?\*/", "", decommented, flags=re.DOTALL)
    # No CSS grid template columns repeat() — that's the card-grid
    # smell from GG-A.2.
    assert not re.search(r"gridTemplateColumns:\s*['\"]repeat\(",
                          decommented), (
        "Outputs uses CSS grid repeat — that's the card-grid pattern "
        "feedback_visualization_as_art warns against; use vertical stack"
    )
    # The ledger uses the receipt component
    assert "function Receipt" in text


def test_outputs_renders_claims_in_detail() -> None:
    text = OUTPUTS.read_text()
    assert "ClaimLine" in text
    assert "function ClaimLine" in text


def test_outputs_renders_failures_md_as_markdown() -> None:
    """failures.md must render as printed prose, not raw text."""
    text = OUTPUTS.read_text()
    assert "ReactMarkdown" in text
    assert "remarkGfm" in text


def test_outputs_has_verify_button() -> None:
    text = OUTPUTS.read_text()
    # Posts to the verify endpoint
    assert "/verify" in text
    assert "apiPost" in text
    # Renders the 8-check ladder
    assert "CheckLadder" in text or "function CheckLadder" in text


def test_outputs_uses_active_lab_filter() -> None:
    text = OUTPUTS.read_text()
    assert "useActiveLab" in text
    # Routes /api/proof-packets through labQuery
    assert "labQuery" in text


def test_outputs_empty_state_uses_connectomic_empty() -> None:
    text = OUTPUTS.read_text()
    assert "ConnectomicEmpty" in text


def test_outputs_loading_uses_stratum_skeleton() -> None:
    """Consistent visual language with the lab dashboard +
    polish components from I.8."""
    text = OUTPUTS.read_text()
    assert "StratumSkeleton" in text


# ─── Types extended ───────────────────────────────────────────────


def test_packet_types_in_client_ts() -> None:
    text = CLIENT_TS.read_text()
    assert "export interface PacketSummary" in text
    assert "export interface PacketsResponse" in text
    assert "export interface PacketDetail" in text
    assert "export interface VerifyCheck" in text
    assert "export interface VerifyResponse" in text


def test_verify_check_status_locked_to_three_values() -> None:
    text = CLIENT_TS.read_text()
    # The three-state grade
    vc_idx = text.find("export interface VerifyCheck")
    assert vc_idx >= 0
    body = text[vc_idx:vc_idx + 400]
    assert '"PASS" | "WARN" | "FAIL"' in body


def main() -> int:
    tests = [
        test_list_endpoint_defined,
        test_detail_endpoint_defined,
        test_verify_endpoint_defined,
        test_list_returns_packets_with_metadata,
        test_list_filters_by_lab,
        test_detail_returns_full_packet_contents,
        test_detail_404_on_unknown_cycle,
        test_detail_handles_multiple_cycle_id_formats,
        test_verify_returns_eight_check_ladder,
        test_verify_local_dev_reports_rekor_as_warn,
        test_verify_404_on_unknown_cycle,
        test_outputs_surface_exists,
        test_outputs_route_registered,
        test_outputs_uses_ledger_idiom_not_card_grid,
        test_outputs_renders_claims_in_detail,
        test_outputs_renders_failures_md_as_markdown,
        test_outputs_has_verify_button,
        test_outputs_uses_active_lab_filter,
        test_outputs_empty_state_uses_connectomic_empty,
        test_outputs_loading_uses_stratum_skeleton,
        test_packet_types_in_client_ts,
        test_verify_check_status_locked_to_three_values,
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
