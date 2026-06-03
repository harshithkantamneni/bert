"""AGNTCY Agent Card extensions + SLIM-transport headers.

Per Linux Foundation AGNTCY project (Cisco/Salesforce/Dell/Google/
Oracle/Red Hat). AGNTCY sits ABOVE A2A + MCP, providing:

  - directory (where to find an agent)
  - identity (who is this agent)
  - observability (standardized event surface)
  - SLIM transport (Secure Low-latency Inter-agent Messaging)

A2A is the wire format. AGNTCY is the connective tissue. By May 2026,
150+ orgs are on A2A and AGNTCY is consolidating around the directory
+ observability layer.

This module extends bert's A2A Agent Card (api/main.py
/.well-known/agent.json) with the AGNTCY-shaped extra fields, and
exposes the AGNTCY directory registration endpoint shape so external
agents can discover bert via a directory rather than a known URL.

What's implemented
==================

  agent_card_agntcy_extensions()  — extra fields per AGNTCY 0.1 spec:
                                     identity, observability, SLIM
                                     endpoints, governance
  agntcy_directory_entry()        — bert as a directory-listable
                                     entry; PI can paste this into
                                     an AGNTCY directory's registration
                                     form
  parse_slim_envelope(headers)    — read AGNTCY SLIM transport headers
                                     from an inbound request
  emit_slim_observability(event)  — write AGNTCY-shaped observability
                                     events into events.jsonl so the
                                     AGNTCY consumer can pick them up

This is the *forward-compatible* surface. AGNTCY's wire spec is still
moving (Linux Foundation Q1 2026 v0.1); the parts that look stable
are implemented; the parts still moving (formal SLIM grpc service)
are stubbed with explicit notes.

References
==========
  https://docs.agntcy.org/
  https://www.linuxfoundation.org/press/linux-foundation-welcomes-the-agntcy-project-to-standardize-open-multi-agent-system-infrastructure-and-break-down-ai-agent-silos
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOG = logging.getLogger("bert.agntcy")
LAB_ROOT = Path(__file__).resolve().parent.parent

# AGNTCY-specific event-class for the observability stream. Distinct
# from the canonical events.jsonl event_class enum so consumers can
# filter for AGNTCY-shaped events.
AGNTCY_OBS_PATH = LAB_ROOT / "state" / "observability" / "agntcy_event.jsonl"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _agent_id() -> str:
    """Deterministic AGNTCY agent_id derived from bert-lab + host.
    PI can override by setting BERT_LAB_AGNTCY_ID env var to a stable
    UUID once they're registered in a directory."""
    import os
    override = os.environ.get("BERT_LAB_AGNTCY_ID")
    if override:
        return override
    seed = f"bert-lab|{LAB_ROOT}"
    return "bert-" + hashlib.sha256(seed.encode()).hexdigest()[:12]


# ── Agent Card extension fields ──────────────────────────────────────


def agent_card_agntcy_extensions(skills: list[dict] | None = None) -> dict[str, Any]:
    """Extra fields to merge into bert's A2A Agent Card so AGNTCY
    consumers can discover bert's identity + observability surface.

    Per AGNTCY 0.1: identity has agent_id + DID-ish form; observability
    declares which event classes are exposed; SLIM declares the
    transport endpoint(s).
    """
    agent_id = _agent_id()
    return {
        # Identity (AGNTCY 0.1 §3.2)
        "agntcy": {
            "spec_version": "0.1",
            "agent_id": agent_id,
            "agent_did": f"did:agntcy:bert-lab:{agent_id}",
            "ratification_status": "self-attested",
            "agent_family": "discernment-pipeline-autonomous-lab",
            # SLIM transport (AGNTCY 0.1 §4.1) — bert speaks JSON-RPC
            # over HTTPS (the A2A + MCP transports); when AGNTCY
            # finalizes its grpc SLIM service, bert can add it here.
            "transports": [
                {
                    "kind": "http+json-rpc",
                    "endpoint": "http://127.0.0.1:5174/a2a/v0/tasks/send",
                    "spec": "a2a-v0.1",
                },
                {
                    "kind": "mcp-stdio",
                    "endpoint": "subprocess",
                    "spec": "mcp-2025-06-18",
                    "available_servers": [s["id"] for s in (skills or []) if s["id"].startswith("bert-")],
                },
            ],
            # Observability (AGNTCY 0.1 §5) — declares the event
            # classes bert emits + the read endpoint for them
            "observability": {
                "events_endpoint": "http://127.0.0.1:5174/api/events",
                "stream_endpoint": "http://127.0.0.1:5174/api/events/stream",
                "agntcy_event_log": "state/observability/agntcy_event.jsonl",
                "event_classes": [
                    "verdict", "stand_aside_verdict",
                    "threshing_dispatch",
                    "clearness_phase1_dispatch", "clearness_phase2_dispatch",
                    "seasoning_entry", "seasoning_revive",
                    "concern_raised", "concern_propagated",
                    "concern_addressed", "revival_proposed",
                    "circuit_breaker_event",
                ],
            },
            # Governance (AGNTCY 0.1 §6) — declares permission gates
            "governance": {
                "permission_model": "P-005 PI-blessed",
                "approval_endpoint": "http://127.0.0.1:5174/api/approve",
                "pending_approvals_endpoint": "http://127.0.0.1:5174/api/pending",
                "audit_log": "lab/sor/events.jsonl",
                "merkle_attestation": "core/merkle.py",
            },
        },
    }


def agntcy_directory_entry() -> dict[str, Any]:
    """Compose bert's directory-registration payload. PI submits this
    to an AGNTCY directory (or self-hosts via docs.agntcy.org tooling).
    """
    agent_id = _agent_id()
    return {
        "agent_id": agent_id,
        "agent_did": f"did:agntcy:bert-lab:{agent_id}",
        "display_name": "bert-lab",
        "registration_ts": _now_iso(),
        "agent_family": "discernment-pipeline-autonomous-lab",
        "agent_card_url": "http://127.0.0.1:5174/.well-known/agent.json",
        "tags": [
            "autonomous-lab",
            "quaker-discernment",
            "free-tier-cascade",
            "p-vs-02-cross-family",
        ],
        "spec_version": "agntcy-0.1",
    }


# ── SLIM transport ───────────────────────────────────────────────────


@dataclass
class SLIMEnvelope:
    sender_did: str
    receiver_did: str
    correlation_id: str
    ts: str
    trace_id: str | None = None
    span_id: str | None = None
    auth_token: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def parse_slim_envelope(headers: dict[str, str]) -> SLIMEnvelope | None:
    """Read AGNTCY SLIM headers off an inbound HTTP request.

    AGNTCY's transport layer attaches:
      x-agntcy-sender-did     — caller's agent_did
      x-agntcy-receiver-did   — must match bert's did
      x-agntcy-correlation-id — request correlation id
      x-agntcy-trace-id       — distributed trace id (OTel-compatible)
      x-agntcy-span-id        — parent span id
      authorization           — bearer token (if auth-required policy)
      x-agntcy-ts             — RFC3339 timestamp

    Returns None when the inbound request isn't an AGNTCY envelope —
    that's how the A2A endpoint distinguishes raw A2A calls from
    AGNTCY-wrapped ones.
    """
    sender = headers.get("x-agntcy-sender-did")
    if not sender:
        return None
    return SLIMEnvelope(
        sender_did=sender,
        receiver_did=headers.get("x-agntcy-receiver-did", ""),
        correlation_id=headers.get("x-agntcy-correlation-id", ""),
        ts=headers.get("x-agntcy-ts", _now_iso()),
        trace_id=headers.get("x-agntcy-trace-id"),
        span_id=headers.get("x-agntcy-span-id"),
        auth_token=headers.get("authorization"),
        extras={
            k.replace("x-agntcy-", ""): v
            for k, v in headers.items()
            if k.startswith("x-agntcy-") and k not in {
                "x-agntcy-sender-did", "x-agntcy-receiver-did",
                "x-agntcy-correlation-id", "x-agntcy-trace-id",
                "x-agntcy-span-id", "x-agntcy-ts",
            }
        },
    )


def slim_response_headers(envelope: SLIMEnvelope) -> dict[str, str]:
    """Compose the response-side SLIM headers for an inbound request.

    The receiver echoes correlation_id + trace_id so distributed
    tracing closes the loop.
    """
    out = {
        "x-agntcy-sender-did": envelope.receiver_did,
        "x-agntcy-receiver-did": envelope.sender_did,
        "x-agntcy-correlation-id": envelope.correlation_id,
        "x-agntcy-ts": _now_iso(),
    }
    if envelope.trace_id:
        out["x-agntcy-trace-id"] = envelope.trace_id
    return out


# ── Observability ────────────────────────────────────────────────────


def emit_agntcy_event(
    event_class: str,
    *,
    correlation_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Write an AGNTCY-shaped observability record alongside bert's
    native observability stream. The format follows AGNTCY 0.1 §5.2:
      {ts, agent_id, event_class, correlation_id, payload}
    """
    AGNTCY_OBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": _now_iso(),
        "agent_id": _agent_id(),
        "event_class": event_class,
        "correlation_id": correlation_id,
        "payload": payload or {},
    }
    try:
        with AGNTCY_OBS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError as e:
        LOG.warning("agntcy: obs write failed: %s", e)
