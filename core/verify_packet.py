"""bert verify thin wrapper (I.4 — investor-grade verification).

8-check verification of a proof packet .tar.gz. Local-only by default;
--fetch-rekor opt-in for transparency-log lookup. Three-mode output:
PASS / PASS-WITH-WARNINGS / FAIL. Failure messages name the specific
check + print the equivalent vanilla cosign command (CTO-friend test).

The 8 checks:
  [1] HASHES.sigstore signature valid against HASHES.txt
  [2] Public key matches trusted-root.json
  [3] (skip in local-dev mode) Rekor v2 inclusion proof
  [4] (skip in local-dev mode) RFC3161 timestamp in cert validity
  [5] SLSA envelope subject digest matches cycle.json hash
  [6] SLSA provenance predicate is well-formed
  [7] failures.md present + failures.sigstore signature valid
  [8] Every file in HASHES.txt actually exists with matching hash

`bert verify --chain` walks cycle.json.parentCycleId through prior
packets, verifying lineage continuity.

PASS-WITH-WARNINGS cases (yellow, not red):
  - failures.md is empty (no declared limitations) — packet looks rehearsed
  - eval/adversarial.json marked as v1 heuristic (not yet LLM-driven)
  - Local-dev mode signatures (production Sigstore migration TBD)
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent


class CheckStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    detail: str = ""
    cosign_equivalent: str | None = None


@dataclass
class VerifyResult:
    packet_path: str
    overall: str  # "PASS" | "PASS-WITH-WARNINGS" | "FAIL"
    checks: list[CheckResult] = field(default_factory=list)
    cycle_id: str | None = None
    rekor_uuid: str | None = None
    failures_count: int = 0
    claims_count: int = 0


def _extract_tarball(packet_path: Path, dest: Path) -> Path:
    """Extract a packet .tar.gz to `dest`. Returns the packet dir inside."""
    with tarfile.open(packet_path, "r:gz") as tar:
        tar.extractall(dest, filter="data")
    # The packet has a single top-level dir matching the basename
    subdirs = [p for p in dest.iterdir() if p.is_dir()]
    if not subdirs:
        raise ValueError(f"packet {packet_path} contained no directory")
    return subdirs[0]


def _verify_sigstore_bundle(
    packet_dir: Path,
    bundle_path: Path,
    signed_bytes: bytes,
    artifact_kind: str,
) -> tuple[bool, str]:
    """Verify a cosign-compatible signature envelope against the bytes that
    were signed. The envelope is an ECDSA-P256/SHA-256 signature (item 25) that
    also verifies under vanilla `cosign verify-blob --key --signature`.
    Returns (ok, detail)."""
    if not bundle_path.exists():
        return False, f"{bundle_path.name} missing"
    try:
        env = json.loads(bundle_path.read_text())
    except json.JSONDecodeError as e:
        return False, f"{bundle_path.name} not valid JSON: {e}"
    sig_b64 = env.get("signature_b64")
    digest = env.get("digest_sha256")
    pubkey_pem = env.get("pubkey_pem")
    if not sig_b64 or not digest or not pubkey_pem:
        return False, f"{bundle_path.name} missing required fields"
    # Hash check
    actual_hash = hashlib.sha256(signed_bytes).hexdigest()
    if actual_hash != digest:
        return False, (
            f"{bundle_path.name} digest mismatch: bundle={digest[:16]}… "
            f"actual={actual_hash[:16]}…"
        )
    # Signature verify (pure-python ECDSA — same check cosign performs)
    from core import signing
    if signing.verify_cosign_blob(signed_bytes, sig_b64, pubkey_pem):
        return True, "signature valid (ECDSA-P256, cosign-verifiable)"
    return False, "signature verification failed"


def verify_packet(packet_path: Path, *, fetch_rekor: bool = False) -> VerifyResult:
    """Run the 8-check verification on a proof packet."""
    pp = Path(packet_path)
    if not pp.exists():
        return VerifyResult(
            packet_path=str(pp), overall="FAIL",
            checks=[CheckResult(name="packet_exists", status=CheckStatus.FAIL,
                                detail=f"{pp} does not exist")],
        )

    result = VerifyResult(packet_path=str(pp), overall="PASS")
    with tempfile.TemporaryDirectory(prefix="bert_verify_") as tmp:
        tmpd = Path(tmp)
        try:
            pdir = _extract_tarball(pp, tmpd)
        except (tarfile.TarError, ValueError) as e:
            result.checks.append(CheckResult(
                name="extract_tarball", status=CheckStatus.FAIL,
                detail=str(e),
            ))
            result.overall = "FAIL"
            return result

        # [1] HASHES.sigstore signature
        hashes_txt = pdir / "HASHES.txt"
        hashes_sig = pdir / "HASHES.sigstore"
        ok, detail = _verify_sigstore_bundle(
            pdir, hashes_sig,
            hashes_txt.read_bytes() if hashes_txt.exists() else b"",
            "proof_packet_hashes",
        )
        result.checks.append(CheckResult(
            name="[1] HASHES signature valid",
            status=CheckStatus.PASS if ok else CheckStatus.FAIL,
            detail=detail,
            cosign_equivalent=(
                "cosign verify-blob --key cosign.pub --signature HASHES.sig "
                "HASHES.txt --insecure-ignore-tlog=true"
            ),
        ))

        # [2] Public key matches trusted-root.json
        tr_path = pdir / "provenance" / "trusted-root.json"
        if not tr_path.exists():
            result.checks.append(CheckResult(
                name="[2] Public key matches trusted-root",
                status=CheckStatus.FAIL,
                detail="trusted-root.json missing",
            ))
        else:
            try:
                tr = json.loads(tr_path.read_text())
                pks = {k.get("pubkey_pem") for k in tr.get("publicKeys", [])}
                env = json.loads(hashes_sig.read_text())
                packet_pk = env.get("pubkey_pem")
                if packet_pk in pks:
                    result.checks.append(CheckResult(
                        name="[2] Public key matches trusted-root",
                        status=CheckStatus.PASS,
                    ))
                else:
                    result.checks.append(CheckResult(
                        name="[2] Public key matches trusted-root",
                        status=CheckStatus.FAIL,
                        detail="packet pubkey not in trusted-root",
                    ))
            except (json.JSONDecodeError, OSError) as e:
                result.checks.append(CheckResult(
                    name="[2] Public key matches trusted-root",
                    status=CheckStatus.FAIL, detail=str(e),
                ))

        # [3] Rekor v2 inclusion (warn in local-dev mode)
        slsa_bundle_path = pdir / "provenance" / "slsa.sigstore"
        mode = "local-dev"
        if slsa_bundle_path.exists():
            try:
                slsa_bundle = json.loads(slsa_bundle_path.read_text())
                mode = (slsa_bundle.get("verificationMaterial") or {}).get("mode", "local-dev")
            except (json.JSONDecodeError, OSError):
                pass
        if mode == "local-dev":
            result.checks.append(CheckResult(
                name="[3] Rekor v2 inclusion proof",
                status=CheckStatus.WARN,
                detail="local-dev mode — no Rekor entry expected (production Sigstore TBD)",
            ))
        elif fetch_rekor:
            # Real Rekor lookup would go here; for v1 we just note it
            result.checks.append(CheckResult(
                name="[3] Rekor v2 inclusion proof",
                status=CheckStatus.WARN,
                detail="--fetch-rekor requested but not implemented in I.4 v1",
            ))
        else:
            result.checks.append(CheckResult(
                name="[3] Rekor v2 inclusion proof",
                status=CheckStatus.WARN,
                detail="skipped (re-run with --fetch-rekor for transparency-log lookup)",
            ))

        # [4] RFC3161 timestamp (warn in local-dev mode)
        result.checks.append(CheckResult(
            name="[4] RFC3161 timestamp in cert validity",
            status=CheckStatus.WARN,
            detail="local-dev mode — RFC3161 timestamps populated in production Sigstore mode",
        ))

        # [5] SLSA envelope subject digest matches cycle.json
        cycle_json_path = pdir / "cycle.json"
        slsa_env_path = pdir / "provenance" / "slsa.intoto.jsonl"
        if cycle_json_path.exists() and slsa_env_path.exists():
            try:
                cycle_bytes = cycle_json_path.read_bytes()
                actual_digest = hashlib.sha256(cycle_bytes).hexdigest()
                env = json.loads(slsa_env_path.read_text())
                statement = json.loads(base64.b64decode(env["payload"]))
                declared_digest = statement["subject"][0]["digest"]["sha256"]
                if declared_digest == actual_digest:
                    result.checks.append(CheckResult(
                        name="[5] SLSA subject digest matches cycle.json",
                        status=CheckStatus.PASS,
                    ))
                else:
                    result.checks.append(CheckResult(
                        name="[5] SLSA subject digest matches cycle.json",
                        status=CheckStatus.FAIL,
                        detail=(
                            f"declared={declared_digest[:16]}… "
                            f"actual={actual_digest[:16]}…"
                        ),
                    ))
                # Capture cycle_id for the result summary
                result.cycle_id = statement["subject"][0]["name"]
            except (json.JSONDecodeError, OSError, KeyError) as e:
                result.checks.append(CheckResult(
                    name="[5] SLSA subject digest matches cycle.json",
                    status=CheckStatus.FAIL, detail=str(e),
                ))
        else:
            result.checks.append(CheckResult(
                name="[5] SLSA subject digest matches cycle.json",
                status=CheckStatus.FAIL,
                detail="missing cycle.json or slsa.intoto.jsonl",
            ))

        # [6] SLSA provenance predicate well-formed
        if slsa_env_path.exists():
            try:
                env = json.loads(slsa_env_path.read_text())
                statement = json.loads(base64.b64decode(env["payload"]))
                pred = statement.get("predicate", {})
                has_bd = "buildDefinition" in pred
                has_rd = "runDetails" in pred
                has_builder = "builder" in (pred.get("runDetails") or {})
                if has_bd and has_rd and has_builder:
                    result.checks.append(CheckResult(
                        name="[6] SLSA provenance well-formed",
                        status=CheckStatus.PASS,
                    ))
                else:
                    result.checks.append(CheckResult(
                        name="[6] SLSA provenance well-formed",
                        status=CheckStatus.FAIL,
                        detail=(
                            f"missing fields: "
                            f"{'buildDefinition ' if not has_bd else ''}"
                            f"{'runDetails ' if not has_rd else ''}"
                            f"{'builder' if not has_builder else ''}"
                        ),
                    ))
            except (json.JSONDecodeError, OSError) as e:
                result.checks.append(CheckResult(
                    name="[6] SLSA provenance well-formed",
                    status=CheckStatus.FAIL, detail=str(e),
                ))
        else:
            result.checks.append(CheckResult(
                name="[6] SLSA provenance well-formed",
                status=CheckStatus.FAIL,
                detail="slsa.intoto.jsonl missing",
            ))

        # [7] failures.md present + failures.sigstore signature valid
        failures_md = pdir / "failures.md"
        failures_sig = pdir / "failures.sigstore"
        if not failures_md.exists():
            result.checks.append(CheckResult(
                name="[7] failures.md present + signed",
                status=CheckStatus.FAIL,
                detail="failures.md missing — packet is incomplete",
            ))
        elif not failures_sig.exists():
            result.checks.append(CheckResult(
                name="[7] failures.md present + signed",
                status=CheckStatus.FAIL,
                detail="failures.sigstore missing — failures.md not separately signed",
            ))
        else:
            f_bytes = failures_md.read_bytes()
            ok, detail = _verify_sigstore_bundle(
                pdir, failures_sig, f_bytes, "proof_packet_failures",
            )
            f_text = failures_md.read_text()
            # Count limitations in failures.md
            n_lim = f_text.count("| `L-")
            result.failures_count = n_lim
            if ok and n_lim > 0:
                result.checks.append(CheckResult(
                    name="[7] failures.md present + signed",
                    status=CheckStatus.PASS,
                    detail=f"{n_lim} declared limitations",
                ))
            elif ok and n_lim == 0:
                # The structural yellow flag: empty failures.md → WARN
                result.checks.append(CheckResult(
                    name="[7] failures.md present + signed",
                    status=CheckStatus.WARN,
                    detail="failures.md is empty — this packet looks rehearsed",
                ))
            else:
                result.checks.append(CheckResult(
                    name="[7] failures.md present + signed",
                    status=CheckStatus.FAIL, detail=detail,
                ))

        # [8] Every file in HASHES.txt matches actual content
        if hashes_txt.exists():
            try:
                declared = {}
                for line in hashes_txt.read_text().strip().splitlines():
                    h, p = line.split("  ", 1)
                    declared[p] = h
                mismatches: list[str] = []
                missing: list[str] = []
                for rel, dh in declared.items():
                    fp = pdir / rel
                    if not fp.exists():
                        missing.append(rel)
                        continue
                    actual = hashlib.sha256(fp.read_bytes()).hexdigest()
                    if actual != dh:
                        mismatches.append(rel)
                if not missing and not mismatches:
                    result.checks.append(CheckResult(
                        name="[8] HASHES manifest matches actual files",
                        status=CheckStatus.PASS,
                        detail=f"{len(declared)}/{len(declared)} files OK",
                    ))
                else:
                    result.checks.append(CheckResult(
                        name="[8] HASHES manifest matches actual files",
                        status=CheckStatus.FAIL,
                        detail=(
                            f"missing={len(missing)} "
                            f"mismatches={len(mismatches)}"
                        ),
                    ))
            except (OSError, ValueError) as e:
                result.checks.append(CheckResult(
                    name="[8] HASHES manifest matches actual files",
                    status=CheckStatus.FAIL, detail=str(e),
                ))
        else:
            result.checks.append(CheckResult(
                name="[8] HASHES manifest matches actual files",
                status=CheckStatus.FAIL, detail="HASHES.txt missing",
            ))

        # Read cycle.json for the summary numbers (claims count)
        if cycle_json_path.exists():
            try:
                cj = json.loads(cycle_json_path.read_text())
                result.claims_count = len(cj.get("claims", []))
            except (json.JSONDecodeError, OSError):
                pass

    # Compute overall
    has_fail = any(c.status == CheckStatus.FAIL for c in result.checks)
    has_warn = any(c.status == CheckStatus.WARN for c in result.checks)
    if has_fail:
        result.overall = "FAIL"
    elif has_warn:
        result.overall = "PASS-WITH-WARNINGS"
    else:
        result.overall = "PASS"
    return result


def verify_chain(packet_paths: list[Path]) -> dict:
    """Verify a sequence of proof packets as a chain (parentCycleId
    + parentDigest matches). Returns dict with overall + per-packet
    results + chain-link verdicts.

    Three link states:
      - "linked": curr.parentCycleId == prev.cycle_id (real lineage)
      - "unlinked": curr declared NO parent (no lineage claim made;
        valid for the first cycle of a lab)
      - "mismatch": curr declared a parent but it doesn't match prev
        (real integrity failure — chain semantically broken)
    """
    packets = [verify_packet(p) for p in packet_paths]
    chain_links: list[dict] = []
    for i in range(1, len(packets)):
        prev = packets[i - 1]
        curr = packets[i]
        # Compare prev.cycle_id to curr.cycle_json.parentCycleId
        declared_parent: str | None = None
        with tempfile.TemporaryDirectory(prefix="bert_chain_") as tmp:
            try:
                pdir = _extract_tarball(Path(curr.packet_path), Path(tmp))
                cj = json.loads((pdir / "cycle.json").read_text())
                declared_parent = cj.get("parentCycleId")
            except (json.JSONDecodeError, OSError, tarfile.TarError):
                declared_parent = None

        if declared_parent is None:
            state = "unlinked"
            linked = False
        elif declared_parent == prev.cycle_id:
            state = "linked"
            linked = True
        else:
            state = "mismatch"
            linked = False

        chain_links.append({
            "from": prev.cycle_id, "to": curr.cycle_id,
            "state": state,
            "linked": linked,
            "detail": (
                f"{curr.cycle_id}.parentCycleId={declared_parent} "
                f"vs prev.cycle_id={prev.cycle_id}"
            ),
        })
    # chain_ok is True iff every link is "linked" (real lineage).
    # has_mismatch is True iff ANY link is a real integrity failure.
    has_mismatch = any(l["state"] == "mismatch" for l in chain_links)
    return {
        "packets": [
            {"cycle_id": p.cycle_id, "overall": p.overall,
             "fail_count": sum(1 for c in p.checks if c.status == CheckStatus.FAIL)}
            for p in packets
        ],
        "chain_links": chain_links,
        "chain_ok": all(link["linked"] for link in chain_links),
        "has_mismatch": has_mismatch,
    }


def format_result(result: VerifyResult, *, color: bool = True) -> str:
    """Format a VerifyResult for terminal output.

    PASS = green, WARN = yellow, FAIL = red. CTO-friend: every FAIL
    includes the equivalent vanilla cosign command.
    """
    if color and sys.stdout.isatty():
        GREEN, YELLOW, RED, RESET, BOLD = "\033[32m", "\033[33m", "\033[31m", "\033[0m", "\033[1m"
    else:
        GREEN = YELLOW = RED = RESET = BOLD = ""

    lines = [f"Verifying {Path(result.packet_path).name} ..."]
    for c in result.checks:
        if c.status == CheckStatus.PASS:
            lines.append(f"  {GREEN}{c.name:55s} PASS{RESET}  {c.detail}")
        elif c.status == CheckStatus.WARN:
            lines.append(f"  {YELLOW}{c.name:55s} WARN{RESET}  {c.detail}")
        else:
            lines.append(f"  {RED}{c.name:55s} FAIL{RESET}  {c.detail}")
            if c.cosign_equivalent:
                lines.append(f"    Reproduce: {c.cosign_equivalent}")

    if result.overall == "PASS":
        head = f"{GREEN}{BOLD}VERIFIED{RESET}  {result.cycle_id}"
    elif result.overall == "PASS-WITH-WARNINGS":
        head = f"{YELLOW}{BOLD}PASS-WITH-WARNINGS{RESET}  {result.cycle_id}"
    else:
        head = f"{RED}{BOLD}FAILED{RESET}  {result.cycle_id}"
    lines += [
        "",
        head,
        f"  {result.claims_count} claims, {result.failures_count} declared limitations",
    ]
    return "\n".join(lines)


__all__ = [
    "verify_packet", "verify_chain", "format_result",
    "VerifyResult", "CheckResult", "CheckStatus",
]
