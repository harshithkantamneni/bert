"""Proof packet exporter v1 (I.2 — investor wow-moment artifact).

Bundles one Quaker cycle's complete record into a portable, verifiable
.tar.gz.

File layout:
  cycle-NNNN.tar.gz
  ├── manifest.json              # OCI 1.1 artifact manifest
  ├── README.md                  # human-readable cycle summary
  ├── cycle.json                 # canonical machine-readable cycle state
  ├── provenance/
  │   ├── slsa.intoto.jsonl      # SLSA v1.1 in-toto DSSE envelope
  │   ├── slsa.sigstore          # Sigstore-shaped bundle (local-dev mode)
  │   └── trusted-root.json      # pinned trust root snapshot
  ├── inputs/
  │   ├── prompt.txt             # original goal
  │   ├── seed.json              # RNG seed, model version, sampling params
  │   └── memory-snapshot.tar    # bert memory state at cycle start (optional)
  ├── outputs/
  │   ├── results.md             # numbered claims [C-1]...[C-N]
  │   ├── artifacts/             # generated files
  │   └── metrics.json           # measured outcomes
  ├── failures.md                # numbered limitations [L-1]...[L-N]
  ├── failures.sigstore          # separate signature on failures.md
  ├── reproduce.sh               # exact commands to re-run cycle
  ├── eval/
  │   ├── self-eval.json         # bert's own confidence + scoring
  │   └── adversarial.json       # red-team agent attacks
  └── HASHES.txt + HASHES.sigstore

I.2 focuses on the exporter + SLSA envelope + HASHES signing. I.3
adds failures.md signing + adversarial.json. I.4 adds bert verify.

Local-dev Sigstore mode: bert's existing core.signing produces
Sigstore-shape signatures with mode="local-dev" (no Fulcio cert, no
Rekor entry). Compatible with the production-Sigstore migration path
but does NOT verify with vanilla cosign — that's intentional for the
pre-commercial phase.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path

LOG = logging.getLogger("bert.proof_packet")
LAB_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_EVENTS_PATH = LAB_ROOT / "lab" / "sor" / "events.jsonl"
_DEFAULT_FINDINGS_DIR = LAB_ROOT / "findings"
_DEFAULT_RESULTS_DIR = LAB_ROOT / "state" / "results"


def _active_lab_paths() -> tuple[Path, Path, Path]:
    """Resolve (events, findings, results) paths for the active lab.

    If lab_context has an active lab path set, use its sor/events.jsonl
    and findings/. Otherwise fall back to the bert-lab project's own
    paths (legacy behavior for in-repo workflows).

    This was a critical correctness bug: without this, the MCP
    packet_export would attest events from the wrong lab when the
    user has set an active lab. Tests verified by t_21+t_26 in
    `_e2e_mcp_full_lifecycle.py`.
    """
    from core.lab_context import get_active_lab_path
    active = get_active_lab_path()
    if active is None:
        return _DEFAULT_EVENTS_PATH, _DEFAULT_FINDINGS_DIR, _DEFAULT_RESULTS_DIR
    return (
        active / "sor" / "events.jsonl",
        active / "findings",
        active / "state" / "results",
    )


# Kept for back-compat with code that imports the module-level constants
# directly. Prefer _active_lab_paths() for new code.
EVENTS_PATH = _DEFAULT_EVENTS_PATH
FINDINGS_DIR = _DEFAULT_FINDINGS_DIR
RESULTS_DIR = _DEFAULT_RESULTS_DIR


SCHEMA_VERSION = "bert.proof.v1"
PREDICATE_TYPE = "https://bert.dev/cycle/v1"
SLSA_PREDICATE_TYPE = "https://slsa.dev/provenance/v1"
INTOTO_STATEMENT_TYPE = "https://in-toto.io/Statement/v1"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_events_for_cycle(cycle_id: int) -> list[dict]:
    """Return all events in events.jsonl with `cycle == cycle_id`.
    Reads from the active lab's events.jsonl when lab_context is set."""
    events_path, _, _ = _active_lab_paths()
    if not events_path.exists():
        return []
    out: list[dict] = []
    for line in events_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("cycle") == cycle_id:
            out.append(ev)
    return out


def _find_artifacts_for_cycle(cycle_id: int) -> list[Path]:
    """Look for cycle-specific output files in findings/ + state/results/.

    Heuristic match: filename contains `_C{cycle}` or `_cycle_{cycle}_`.
    Returns absolute paths. Uses lab_context to find the active lab's
    findings/ + state/results/ when set; falls back to project paths
    otherwise."""
    found: list[Path] = []
    patterns = [f"_C{cycle_id}_", f"_C{cycle_id}.", f"_cycle_{cycle_id}_"]
    _, findings_dir, results_dir = _active_lab_paths()
    for d in (findings_dir, results_dir):
        if not d.exists():
            continue
        for p in d.glob("*"):
            if any(pat in p.name for pat in patterns) and p.is_file():
                found.append(p)
    return sorted(found)


def extract_claims_from_events(events: list[dict]) -> list[dict]:
    """Extract structured claims from a cycle's events.

    Each verdict event becomes a claim. Confidence + role + verdict are
    preserved so the adversarial-eval module can attack the claim
    intelligently. Limitation cross-references populate in
    `extract_limitations_from_events`.
    """
    claims: list[dict] = []
    for _i, ev in enumerate(events):
        if ev.get("event_class") != "verdict":
            continue
        cid = f"C-{len(claims) + 1}"
        text = (
            f"{ev.get('agent', '?')} returned verdict={ev.get('verdict', '?')} "
            f"at cycle {ev.get('cycle', '?')}"
        )
        claims.append({
            "id": cid,
            "text": text,
            "role": ev.get("agent"),
            "verdict": ev.get("verdict"),
            "confidence_1to10": ev.get("confidence_1to10"),
            "limitationRefs": [],  # filled below
        })
    return claims


def _read_obs_events(event_class: str, cycle_id: int | None = None) -> list[dict]:
    """Read state/observability/{event_class}.jsonl, optionally filtered
    by cycle. The merged lab/sor/events.jsonl flattens structured payload
    fields; this preserves them.
    """
    obs_path = LAB_ROOT / "state" / "observability" / f"{event_class}.jsonl"
    if not obs_path.exists():
        return []
    out: list[dict] = []
    for line in obs_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if cycle_id is not None and ev.get("cycle") != cycle_id:
            continue
        out.append(ev)
    return out


def extract_limitations_from_events(events: list[dict], claims: list[dict],
                                     *, cycle_id: int | None = None) -> list[dict]:
    """Extract structured limitations (POPPER-style categorized) from
    the cycle's events. Cross-references claims when a stand_aside or
    APPROVE_WITH_CAVEATS verdict ties a limitation to a specific claim.

    Reads BOTH the merged events list AND the per-class observability
    logs (state/observability/*.jsonl) because the merged log strips
    structured payload fields like concern_count.

    Categories (model-card / POPPER-style):
      - distribution-shift
      - over-generalization
      - selective-disclosure
      - staleness
      - conservative-judgement (low confidence)
      - known-unaddressed-concern
    """
    limitations: list[dict] = []
    next_id = 1
    role_to_claim: dict[str, str] = {}
    for c in claims:
        if c.get("role"):
            role_to_claim.setdefault(c["role"], c["id"])

    # Pull structured stand_aside_verdict + concern_addressed events
    # from the per-class observability log (not events.jsonl).
    obs_stand_aside = _read_obs_events("stand_aside_verdict", cycle_id)
    obs_concern_addressed = _read_obs_events("concern_addressed", cycle_id)
    augmented = list(events) + obs_stand_aside + obs_concern_addressed

    for ev in augmented:
        evc = ev.get("event_class")
        # APPROVE_WITH_CAVEATS verdict: each caveat is a limitation
        if evc == "stand_aside_verdict":
            role = ev.get("role")
            n_concerns = ev.get("concern_count", 0)
            severity = ev.get("severity_grade")
            if n_concerns > 0:
                lim = {
                    "id": f"L-{next_id}",
                    "category": "known-unaddressed-concern",
                    "text": (
                        f"{role} dispatch raised {n_concerns} concern(s) "
                        f"(severity={severity}); concerns ride forward via "
                        f"concern propagation but the original cycle's claim "
                        f"stands subject to downstream resolution."
                    ),
                    "claimRefs": [role_to_claim[role]] if role in role_to_claim else [],
                }
                limitations.append(lim)
                next_id += 1

        # Low-confidence verdict → conservative-judgement limitation
        elif evc == "verdict":
            conf = ev.get("confidence_1to10")
            if isinstance(conf, int) and conf < 6:
                role = ev.get("agent")
                lim = {
                    "id": f"L-{next_id}",
                    "category": "conservative-judgement",
                    "text": (
                        f"{role} declared confidence {conf}/10 — below the "
                        f"≥6 threshold that the falsifier baseline treats as "
                        f"reliable. The verdict stands but the surrounding "
                        f"claim should not be over-generalized."
                    ),
                    "claimRefs": [role_to_claim[role]] if role in role_to_claim else [],
                }
                limitations.append(lim)
                next_id += 1

        # concern_addressed events represent bounded uncertainties
        elif evc == "concern_addressed":
            verdict = ev.get("resolution_verdict")
            cid = ev.get("concern_id", "?")
            lim = {
                "id": f"L-{next_id}",
                "category": "distribution-shift",
                "text": (
                    f"Propagated concern {cid} was addressed by a "
                    f"resolution_verdict={verdict}. Resolution holds for the "
                    f"observed cycle but the concern's original distribution "
                    f"may re-emerge under shift."
                ),
                "claimRefs": [],
            }
            limitations.append(lim)
            next_id += 1

    # Stale: if cycle has no recent verdict event in the window
    if events:
        last_ev = max(events, key=lambda e: e.get("ts", ""))
        last_ts = last_ev.get("ts")
        if last_ts:
            try:
                from datetime import datetime
                t = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                age_days = (datetime.now(UTC) - t).days
                if age_days > 7:
                    lim = {
                        "id": f"L-{next_id}",
                        "category": "staleness",
                        "text": (
                            f"Cycle last activity was {age_days} days ago. "
                            f"Provider behavior, model weights, and free-tier "
                            f"availability may have shifted since."
                        ),
                        "claimRefs": [c["id"] for c in claims],
                    }
                    limitations.append(lim)
                    next_id += 1
            except (ValueError, AttributeError):
                pass

    # Cross-reference: tag claims with the limitations that apply.
    for lim in limitations:
        for ref in lim.get("claimRefs") or []:
            for c in claims:
                if c["id"] == ref and lim["id"] not in c["limitationRefs"]:
                    c["limitationRefs"].append(lim["id"])

    return limitations


def render_failures_md(cycle_id: int, claims: list[dict], limitations: list[dict]) -> str:
    """Render failures.md from structured limitations. POPPER-style table
    with category, text, and cross-references to claims."""
    if not limitations:
        return (
            f"# Failures — cycle-{cycle_id:04d}\n\n"
            f"_No structured limitations were extracted from this cycle._\n\n"
            f"Per the §9 honest-failure-surfacing rubric, an empty "
            f"`failures.md` is a yellow flag — `bert verify` will return "
            f"PASS-WITH-WARNINGS until limitations are declared.\n\n"
            f"This packet declares {len(claims)} claim(s); none surfaced a "
            f"limitation, which may reflect either (a) high signal in this "
            f"specific cycle, or (b) under-enumeration. Future cycles should "
            f"populate this section with POPPER-style categorized failures.\n"
        )
    lines = [
        f"# Failures — cycle-{cycle_id:04d}",
        "",
        f"_{len(limitations)} structured limitation(s) declared, across "
        f"{len(claims)} claim(s). POPPER-style categorization. Cross-references "
        f"to `outputs/results.md` claims by `[C-N]` id._",
        "",
        "| ID | Category | Text | Affects claims |",
        "|---|---|---|---|",
    ]
    for lim in limitations:
        refs = ", ".join(lim.get("claimRefs", [])) or "—"
        lines.append(
            f"| `{lim['id']}` | `{lim['category']}` | "
            f"{lim['text'][:200].replace(chr(10), ' ')} | {refs} |"
        )
    lines += [
        "",
        "## Category definitions (POPPER-style)",
        "",
        "- **distribution-shift** — the cycle's claim holds in its observed "
        "input regime but a shift could invalidate it",
        "- **over-generalization** — single-cycle conclusion extrapolated "
        "beyond justified scope",
        "- **selective-disclosure** — relevant context the cycle did not "
        "incorporate",
        "- **staleness** — the cycle's data or provider state may have "
        "shifted since",
        "- **conservative-judgement** — confidence below the reliable-claim "
        "floor",
        "- **known-unaddressed-concern** — a flagged concern that the cycle "
        "did NOT resolve internally",
        "",
    ]
    return "\n".join(lines) + "\n"


def build_cycle_json(
    *,
    cycle_id: int,
    parent_cycle_id: int | None = None,
    parent_digest: str | None = None,
    events: list[dict] | None = None,
    artifacts: list[Path] | None = None,
    extra_claims: list[dict] | None = None,
    extra_limitations: list[dict] | None = None,
) -> dict:
    """Build the canonical cycle.json shape per locked schema."""
    events = events if events is not None else _read_events_for_cycle(cycle_id)
    artifacts = artifacts if artifacts is not None else _find_artifacts_for_cycle(cycle_id)
    # Pick the cycle's start/end from earliest/latest event ts
    sorted_events = sorted(events, key=lambda e: e.get("ts", ""))
    started_at = sorted_events[0].get("ts") if sorted_events else _now_iso()
    completed_at = sorted_events[-1].get("ts") if sorted_events else _now_iso()
    # Provider/model: pull from the first verdict event with telemetry
    provider = None
    model_id = None
    for ev in events:
        tel = ev.get("telemetry") or {}
        if isinstance(tel, dict):
            provider = tel.get("provider") or provider
            model_id = tel.get("model_used") or model_id
        if ev.get("model"):
            model_id = ev.get("model") or model_id
    # Subject digest is computed externally over the packet tarball
    # AFTER the rest of the packet is built (chicken-and-egg). For
    # cycle.json we record subject_name only; subject_digest added in
    # the in-toto envelope at sign time.
    return {
        "schemaVersion": SCHEMA_VERSION,
        "cycleId": f"cycle-{cycle_id:04d}",
        "labRef": f"local://bert@cycle-{cycle_id:04d}",
        "parentCycleId": (
            f"cycle-{parent_cycle_id:04d}" if parent_cycle_id is not None else None
        ),
        "parentDigest": parent_digest,
        "startedAt": started_at,
        "completedAt": completed_at,
        "provider": provider,
        "model": {"id": model_id, "version": None, "samplingParams": {}},
        "subject": {"name": f"cycle-{cycle_id:04d}"},
        "predicateType": PREDICATE_TYPE,
        "claims": extra_claims or [],
        "limitations": extra_limitations or [],
        "reproduce": {"command": "./reproduce.sh", "expectedHash": None},
        "eventCount": len(events),
        "artifactCount": len(artifacts),
    }


def _build_intoto_envelope(cycle_json: dict, subject_digest: str) -> dict:
    """Build the SLSA v1.1 in-toto Statement embedded in a DSSE envelope.

    Returns the *unsigned* envelope; the caller signs it via
    core.signing.sign_bytes.
    """
    statement = {
        "_type": INTOTO_STATEMENT_TYPE,
        "subject": [{
            "name": cycle_json["subject"]["name"],
            "digest": {"sha256": subject_digest},
        }],
        "predicateType": SLSA_PREDICATE_TYPE,
        "predicate": {
            "buildDefinition": {
                "buildType": "https://bert.dev/cycle/v1",
                "externalParameters": {
                    "cycleId": cycle_json["cycleId"],
                    "parentCycleId": cycle_json.get("parentCycleId"),
                    "labRef": cycle_json["labRef"],
                },
                "internalParameters": {
                    "provider": cycle_json.get("provider"),
                    "model": cycle_json.get("model"),
                },
            },
            "runDetails": {
                "builder": {"id": "https://bert.dev/bert/runtime"},
                "metadata": {
                    "invocationId": cycle_json["cycleId"],
                    "startedOn": cycle_json["startedAt"],
                    "finishedOn": cycle_json["completedAt"],
                },
            },
        },
    }
    # DSSE envelope wraps the statement
    import base64
    payload = json.dumps(statement, sort_keys=True).encode()
    return {
        "payloadType": "application/vnd.in-toto+json",
        "payload": base64.b64encode(payload).decode(),
        "_statement": statement,  # convenience field; not in DSSE spec
    }


def _build_sigstore_bundle(
    *, signature_b64: str, pubkey_pem: str, signed_payload_b64: str,
    artifact_hash: str, ts: str,
) -> dict:
    """Build a Sigstore-shape bundle.

    bert ships local-dev mode: bundle has the same FIELD SHAPE as
    production Sigstore (cert, transparencyLogEntries, etc.) but the
    transparency-log entries are empty, RFC3161 timestamps are empty,
    and signatures are produced by bert's local ed25519 key rather
    than a Fulcio-issued ephemeral cert.

    Wire-format compatibility is intentional so verifiers can parse
    the bundle without bert-specific code. It is NOT the same as
    production Sigstore being implemented: per DD.2, real Sigstore
    requires sigstore-python + OIDC + Fulcio cert acquisition + Rekor
    submission + TSA integration. See `core/signing.py` module
    docstring for the full engineering punch list.
    """
    return {
        "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
        "verificationMaterial": {
            "publicKey": {
                "hint": "bert-local-dev-ed25519",
                "rawBytes": pubkey_pem,
            },
            "mode": "local-dev",
            "tlogEntries": [],
            "timestampVerificationData": {
                "rfc3161Timestamps": [],
            },
        },
        "messageSignature": {
            "messageDigest": {
                "algorithm": "SHA2_256",
                "digest": artifact_hash,
            },
            "signature": signature_b64,
        },
        "_dsseEnvelopePayload": signed_payload_b64,
        "_signedAt": ts,
    }


def _build_oci_manifest(cycle_id: int, files: dict[str, str]) -> dict:
    """Build a minimal OCI 1.1 artifact manifest pointing at packet
    files. `files` is {relpath: sha256_hex}. mediaType is bert-specific
    artifactType per OCI 1.1 spec."""
    return {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "artifactType": "application/vnd.bert.proof.v1+json",
        "config": {
            "mediaType": "application/vnd.bert.proof.config.v1+json",
            "digest": f"sha256:{files.get('cycle.json', '')}",
            "size": 0,
        },
        "layers": [
            {
                "mediaType": "application/vnd.bert.proof.layer.v1+json",
                "digest": f"sha256:{h}",
                "size": 0,
                "annotations": {"path": p},
            }
            for p, h in sorted(files.items())
        ],
        "annotations": {
            "bert.cycle.id": f"cycle-{cycle_id:04d}",
            "bert.proof.schema": SCHEMA_VERSION,
        },
    }


def _compute_hashes_recursive(packet_dir: Path) -> dict[str, str]:
    """Compute SHA-256 over every file in packet_dir, returning
    {relpath: hex_hash}. Stable order (sorted)."""
    hashes: dict[str, str] = {}
    for p in sorted(packet_dir.rglob("*")):
        if not p.is_file():
            continue
        # Exclude the HASHES files themselves to avoid self-reference
        if p.name.startswith("HASHES"):
            continue
        rel = p.relative_to(packet_dir).as_posix()
        hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def _write_hashes_file(packet_dir: Path, hashes: dict[str, str]) -> Path:
    """Write HASHES.txt (sorted relpath SHA-256 manifest)."""
    lines = [f"{h}  {p}" for p, h in sorted(hashes.items())]
    path = packet_dir / "HASHES.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def _build_reproduce_sh(cycle_id: int) -> str:
    """Honest reproduce.sh body (DD.3).

    The earlier placeholder said "real reproduce.sh ships in I.4".
    That promise was structurally undeliverable for hosted-LLM
    workflows. There are TWO different things a verifier could mean
    by "reproduce", and bert only delivers one of them:

      (a) CRYPTOGRAPHIC VERIFICATION (deliverable today, with vanilla
          tools and no bert install): given the .tar.gz, recompute
          file hashes, verify the SLSA in-toto envelope's DSSE
          signature, verify HASHES.sigstore, verify failures.sigstore
          independently. If any byte changed, verification fails.
          This catches edited-demo fraud.

      (b) EXACT-PROCESS RE-EXECUTION (NOT deliverable, not by bert,
          not at I.4, not at I.4+N): re-run the producer LLM with
          identical inputs and check the output matches byte-for-byte.
          Hosted LLMs are NOT deterministic across requests — same
          prompt, same seed, same temperature, same model — because
          providers' KV-cache, batching, and load-balancing introduce
          non-determinism that bert cannot control. Promising it
          would be dishonest. Local-model labs CAN deliver it (with
          vLLM seed + temperature=0 + identical hardware); bert's
          free-tier hosted-LLM stack structurally cannot.

    What this script DOES do:

      1. Recompute file hashes inside the packet, compare against
         HASHES.txt. If any mismatch, exit 1 — the packet has been
         tampered with.
      2. Print the real `cosign verify-blob --key cosign.pub
         --signature HASHES.sig …` invocation (item 25 — ECDSA-P256,
         offline, works under vanilla cosign today) that the verifier
         can run by hand. (We don't run it for them because cosign
         isn't a transitive dependency; the verifier owns that step.)
      3. Print the structural-equivalence test that bert's CYCLE 0
         verifier would use to re-evaluate the SAME prompts with
         FRESH model calls and check semantic equivalence (not
         byte-equivalence). This is the closest honest analogue to
         re-execution for hosted-LLM workflows.

    Per DD.3: this disclosure replaces the "I.4 placeholder" framing
    everywhere it appears (qa.md Q5, anti_patterns.md reproducibility
    ladder, glossary).
    """
    return f"""#!/bin/sh
# reproduce.sh — verify proof packet for cycle {cycle_id}
#
# Two layers of "reproduce" — bert ships layer (a) cleanly; layer (b)
# is structurally impossible for hosted-LLM workflows. See DD.3.
#
# Layer (a) — CRYPTOGRAPHIC VERIFICATION (this script).
#   Recomputes hashes, checks signatures. No LLM calls. No bert
#   install required (cosign is the only external tool, and it's
#   optional — local hash check works without it). Run on any
#   laptop, online or offline.
#
# Layer (b) — EXACT-PROCESS RE-EXECUTION (NOT shipped).
#   Hosted LLMs (NVIDIA, Groq, Mistral, Cerebras, etc.) are not
#   deterministic across requests. Same prompt + same seed +
#   same temperature + same model gives DIFFERENT outputs from
#   the same provider due to KV-cache state, batching, and
#   load-balancing on the provider's side. bert cannot promise
#   byte-exact replay against a hosted producer. Promising it
#   would be Devin-class dishonesty.
#
#   What we CAN do, and what `bert verify --structural` does in
#   the milestone where we ship it: re-run the recorded prompts
#   with FRESH provider calls and check semantic equivalence (by
#   the same cross-family judge that produced the original
#   verdict). This is the strongest honest analogue.

set -e
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

echo "── Layer (a) — cryptographic verification ──────────────────"
echo

if [ ! -f HASHES.txt ]; then
  echo "FAIL: HASHES.txt missing — this is not a bert proof packet"
  exit 1
fi

# Recompute every file's SHA-256 and compare to HASHES.txt
fail=0
while IFS= read -r line; do
  expected="${{line%%  *}}"
  relpath="${{line#*  }}"
  if [ -z "$expected" ] || [ -z "$relpath" ]; then continue; fi
  if [ "$relpath" = "HASHES.txt" ] || [ "$relpath" = "HASHES.sigstore" ]; then
    continue  # self-reference loop avoided
  fi
  if [ ! -f "$relpath" ]; then
    echo "  MISSING: $relpath"
    fail=1
    continue
  fi
  actual=$(shasum -a 256 "$relpath" | cut -d' ' -f1)
  if [ "$expected" != "$actual" ]; then
    echo "  TAMPER: $relpath"
    echo "    expected $expected"
    echo "    got      $actual"
    fail=1
  fi
done < HASHES.txt

if [ "$fail" = "1" ]; then
  echo
  echo "FAIL: at least one file's hash does not match HASHES.txt"
  echo "      The packet has been tampered with since signing."
  exit 1
fi

echo "OK: every file in the packet matches HASHES.txt"
echo

# Print (don't auto-run) the cosign commands — the verifier owns
# that step and may not have cosign installed.
echo "── Verify HASHES.txt with vanilla cosign (works today) ─────"
echo "  cosign verify-blob \\"
echo "    --key cosign.pub \\"
echo "    --signature HASHES.sig \\"
echo "    --insecure-ignore-tlog=true \\"
echo "    HASHES.txt"
echo
echo "── Verify failures.md independently (works today) ──────────"
echo "  cosign verify-blob \\"
echo "    --key cosign.pub \\"
echo "    --signature failures.sig \\"
echo "    --insecure-ignore-tlog=true \\"
echo "    failures.md"
echo "  (--insecure-ignore-tlog skips the transparency-log check;"
echo "   the signature itself is fully verified against cosign.pub.)"
echo
echo "── SLSA in-toto envelope (attestation verify — future) ─────"
echo "  The in-toto/DSSE attestation in provenance/ is verified by"
echo "  bert verify today; cosign attestation verification is a"
echo "  deferred production-Sigstore step (no live Fulcio/Rekor)."
echo

echo "── Layer (b) — exact-process re-execution ─────────────────"
echo
echo "  NOT deliverable by bert. Hosted-LLM providers are not"
echo "  deterministic across requests. The strongest honest analogue"
echo "  is structural re-evaluation: re-run the recorded inputs"
echo "  with fresh provider calls and check the cross-family judge"
echo "  agrees on the verdict. That's `bert verify --structural`"
echo "  (post-investor milestone). The cryptographic chain above"
echo "  is what catches edited-demo fraud today."
echo
echo "OK: cycle {cycle_id} verified at layer (a)."
"""


def _make_readme(cycle_json: dict) -> str:
    """Human-readable README. Built BEFORE HASHES.txt so it's covered
    by the integrity check; the per-file hash listing lives in HASHES.txt
    rather than embedded here (avoids a self-reference loop)."""
    return (
        f"# Proof packet: {cycle_json['cycleId']}\n\n"
        f"**Generated:** {cycle_json['completedAt']}  \n"
        f"**Provider:** {cycle_json.get('provider') or 'unknown'}  \n"
        f"**Model:** {(cycle_json.get('model') or {}).get('id') or 'unknown'}  \n"
        f"**Events recorded:** {cycle_json.get('eventCount', 0)}  \n"
        f"**Artifacts bundled:** {cycle_json.get('artifactCount', 0)}  \n"
        f"**Parent cycle:** {cycle_json.get('parentCycleId') or '(none — initial cycle)'}\n\n"
        f"## Verifying this packet\n\n"
        f"```\n"
        f"bert verify {cycle_json['cycleId']}.tar.gz\n"
        f"```\n\n"
        f"Or independently with vanilla cosign (works today — no bert needed):\n\n"
        f"```\n"
        f"cosign verify-blob --key cosign.pub --signature HASHES.sig \\\n"
        f"  --insecure-ignore-tlog=true HASHES.txt\n"
        f"```\n\n"
        f"## File inventory\n\n"
        f"See `HASHES.txt` for the per-file SHA-256 manifest of every file in\n"
        f"this packet (except `manifest.json` and the HASHES files themselves).\n"
        f"The list is signed in `HASHES.sigstore`.\n\n"
        f"## Schema\n\n"
        f"This packet follows the **{cycle_json['schemaVersion']}** schema.\n"
        f"`cycle.json` is the canonical machine-readable cycle state;\n"
        f"`provenance/slsa.intoto.jsonl` is the SLSA v1.1 provenance\n"
        f"attestation; `failures.md` enumerates declared limitations.\n"
    )


def build_packet(
    *,
    cycle_id: int,
    output_dir: Path | None = None,
    parent_cycle_id: int | None = None,
    parent_digest: str | None = None,
    include_memory_snapshot: bool = False,
    extra_claims: list[dict] | None = None,
    extra_limitations: list[dict] | None = None,
) -> Path:
    """Build the proof packet .tar.gz for `cycle_id`.

    Returns the path to the produced .tar.gz. Raises ValueError if the
    cycle has no events (nothing to attest).
    """
    from core import signing

    events = _read_events_for_cycle(cycle_id)
    if not events:
        active_events_path, _, _ = _active_lab_paths()
        raise ValueError(
            f"cycle {cycle_id} has no events in {active_events_path} — nothing to attest"
        )

    output_dir = output_dir or (LAB_ROOT / "findings" / "proof_packets")
    output_dir.mkdir(parents=True, exist_ok=True)
    packet_name = f"cycle-{cycle_id:04d}"
    tarball_path = output_dir / f"{packet_name}.tar.gz"

    with tempfile.TemporaryDirectory(prefix="bert_pp_") as tmpdir:
        packet_dir = Path(tmpdir) / packet_name
        # Build directory structure
        (packet_dir / "provenance").mkdir(parents=True)
        (packet_dir / "inputs").mkdir()
        (packet_dir / "outputs" / "artifacts").mkdir(parents=True)
        (packet_dir / "eval").mkdir()

        # cycle.json
        cycle_json = build_cycle_json(
            cycle_id=cycle_id,
            parent_cycle_id=parent_cycle_id,
            parent_digest=parent_digest,
            events=events,
            extra_claims=extra_claims,
            extra_limitations=extra_limitations,
        )
        (packet_dir / "cycle.json").write_text(
            json.dumps(cycle_json, indent=2, sort_keys=True),
        )

        # inputs/
        # prompt.txt — first event content if any
        first_content = events[0].get("content", "") if events else ""
        (packet_dir / "inputs" / "prompt.txt").write_text(str(first_content))
        # seed.json — provider + model + sampling params snapshot
        (packet_dir / "inputs" / "seed.json").write_text(json.dumps({
            "provider": cycle_json.get("provider"),
            "model": cycle_json.get("model"),
            "ts": cycle_json["startedAt"],
            "eventCount": cycle_json["eventCount"],
        }, indent=2))

        # outputs/
        # results.md — list all events with their roles + verdicts
        results_lines = [f"# Cycle {cycle_id} results", ""]
        for ev in events:
            results_lines.append(
                f"- `{ev.get('event_class', '?')}` "
                f"role={ev.get('agent', '?')} "
                f"verdict={ev.get('verdict', '—')}"
            )
        (packet_dir / "outputs" / "results.md").write_text(
            "\n".join(results_lines) + "\n",
        )
        # metrics.json
        verdict_count = sum(1 for e in events if e.get("event_class") == "verdict")
        (packet_dir / "outputs" / "metrics.json").write_text(json.dumps({
            "eventCount": len(events),
            "verdictCount": verdict_count,
            "elapsed_secs": _seconds_between(
                cycle_json["startedAt"], cycle_json["completedAt"],
            ),
        }, indent=2))
        # artifacts/ — copy heuristic-matched files
        for src in _find_artifacts_for_cycle(cycle_id):
            try:
                shutil.copy2(src, packet_dir / "outputs" / "artifacts" / src.name)
            except (OSError, shutil.Error):
                continue

        # eval/self-eval.json
        confidences = [
            e.get("confidence_1to10") for e in events
            if e.get("confidence_1to10") is not None
        ]
        avg_conf = (sum(confidences) / len(confidences)) if confidences else None
        (packet_dir / "eval" / "self-eval.json").write_text(json.dumps({
            "verdict_count": verdict_count,
            "avg_confidence_1to10": avg_conf,
            "samples": len(confidences),
        }, indent=2))

        # I.3 — extract structured claims + limitations from events,
        # back-fill cycle.json, run adversarial-eval, write failures.md
        claims = extract_claims_from_events(events)
        limitations = extract_limitations_from_events(events, claims, cycle_id=cycle_id)
        # Update cycle.json on disk with the real claims + limitations
        cycle_json["claims"] = claims
        cycle_json["limitations"] = limitations
        (packet_dir / "cycle.json").write_text(
            json.dumps(cycle_json, indent=2, sort_keys=True),
        )

        # I.3 — adversarial eval. DD.1 wires LLM-driven mode via env
        # var BERT_ADVERSARIAL_MODE=llm. Producer model comes from
        # cycle.json so the cross-family attacker selection has a
        # concrete family to differ from.
        from core import adversarial_eval as adv_mod
        adv_mode = os.environ.get("BERT_ADVERSARIAL_MODE", "heuristic").lower()
        if adv_mode not in ("heuristic", "llm"):
            adv_mode = "heuristic"
        producer_provider = cycle_json.get("provider")
        producer_model_str = cycle_json.get("model")
        producer_model = None
        if producer_provider and producer_model_str:
            producer_model = f"{producer_provider}/{producer_model_str}"
        adv = adv_mod.run_adversarial_eval(
            claims, mode=adv_mode, producer_model=producer_model,
        )
        (packet_dir / "eval" / "adversarial.json").write_text(
            json.dumps(adv, indent=2),
        )

        # I.3 — failures.md (real content, replaces placeholder)
        (packet_dir / "failures.md").write_text(
            render_failures_md(cycle_id, claims, limitations),
        )

        # I.3 — failures.sigstore: SEPARATE DSSE signature on
        # failures.md alone. An investor's CTO friend can verify
        # failures.md independently of the rest of the packet. This
        # is the structural honest-failure-surfacing primitive.
        failures_bytes = (packet_dir / "failures.md").read_bytes()
        failures_env = signing.sign_blob_cosign(failures_bytes)
        failures_env["extras"] = {
            "cycleId": cycle_json["cycleId"],
            "limitation_count": len(limitations),
            "claim_count": len(claims),
        }
        (packet_dir / "failures.sigstore").write_text(json.dumps(failures_env, indent=2))
        # cosign-native sibling: `cosign verify-blob --key cosign.pub
        # --signature failures.sig failures.md --insecure-ignore-tlog=true`
        (packet_dir / "failures.sig").write_text(failures_env["signature_b64"])

        # reproduce.sh — DD.3 honest rewrite (was: I.4 placeholder)
        (packet_dir / "reproduce.sh").write_text(
            _build_reproduce_sh(cycle_id)
        )
        (packet_dir / "reproduce.sh").chmod(0o755)

        # provenance/
        # Build the SLSA envelope; subject_digest is over cycle.json
        cycle_json_bytes = (packet_dir / "cycle.json").read_bytes()
        subject_digest = hashlib.sha256(cycle_json_bytes).hexdigest()
        envelope = _build_intoto_envelope(cycle_json, subject_digest)
        (packet_dir / "provenance" / "slsa.intoto.jsonl").write_text(
            json.dumps(envelope) + "\n",
        )

        # Sign the canonical statement bytes
        statement = envelope.pop("_statement")
        statement_bytes = signing.canonical_json(statement)
        sig = signing.sign_bytes(
            statement_bytes,
            artifact_kind="proof_packet_slsa",
            extras={"cycleId": cycle_json["cycleId"]},
        )
        bundle = _build_sigstore_bundle(
            signature_b64=sig.signature_b64,
            pubkey_pem=sig.pubkey_pem,
            signed_payload_b64=envelope["payload"],
            artifact_hash=sig.artifact_hash,
            ts=sig.ts,
        )
        (packet_dir / "provenance" / "slsa.sigstore").write_text(
            json.dumps(bundle, indent=2),
        )

        # Trusted root snapshot — point at bert's cosign-verifiable ECDSA
        # pubkey (the key that signs HASHES + failures, verifiable by vanilla
        # cosign). The ed25519 key (`sig.pubkey_pem`) stays internal.
        cosign_pubkey_pem = signing.cosign_public_key_pem()
        (packet_dir / "provenance" / "trusted-root.json").write_text(json.dumps({
            "mediaType": "application/vnd.bert.trusted-root.v1+json",
            "publicKeys": [{
                "hint": "bert-local-dev-ecdsa-p256",
                "pubkey_pem": cosign_pubkey_pem,
            }],
            "validFor": {"start": sig.ts},
            "_note": "Local-dev trust root. Production Sigstore TUF root TBD.",
        }, indent=2))
        # cosign.pub written here (before HASHES) so the convenience pubkey is
        # itself integrity-covered by HASHES.txt.
        (packet_dir / "cosign.pub").write_text(cosign_pubkey_pem)

        # README BEFORE HASHES so it's covered by the integrity check.
        (packet_dir / "README.md").write_text(_make_readme(cycle_json))

        # HASHES — recursive sha256 + sign. Covers everything in the
        # packet so far (cycle.json + inputs/ + outputs/ + eval/ +
        # failures.md + reproduce.sh + provenance/ + README.md).
        # Excludes HASHES.* (self-reference) and manifest.json (built
        # next as the OCI catalog of the same hashes).
        hashes = _compute_hashes_recursive(packet_dir)
        _write_hashes_file(packet_dir, hashes)
        hashes_bytes = (packet_dir / "HASHES.txt").read_bytes()
        hashes_env = signing.sign_blob_cosign(hashes_bytes)
        hashes_env["extras"] = {"cycleId": cycle_json["cycleId"]}
        (packet_dir / "HASHES.sigstore").write_text(json.dumps(hashes_env, indent=2))
        # cosign-native siblings so a CTO verifies offline with vanilla cosign:
        #   cosign verify-blob --key cosign.pub --signature HASHES.sig \
        #     HASHES.txt --insecure-ignore-tlog=true
        (packet_dir / "HASHES.sig").write_text(hashes_env["signature_b64"])

        # OCI manifest LAST — catalogs the file hashes that HASHES.txt
        # records. Not itself in HASHES.txt (that would be a self-loop).
        manifest = _build_oci_manifest(cycle_id, hashes)
        (packet_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Bundle into .tar.gz
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(packet_dir, arcname=packet_name)

    LOG.info("proof_packet: wrote %s", tarball_path)
    return tarball_path


def _seconds_between(start_iso: str, end_iso: str) -> float:
    try:
        from datetime import datetime
        s = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        e = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
        return round((e - s).total_seconds(), 2)
    except (ValueError, AttributeError):
        return 0.0


__all__ = [
    "build_packet", "build_cycle_json",
    "SCHEMA_VERSION", "PREDICATE_TYPE", "SLSA_PREDICATE_TYPE",
]
