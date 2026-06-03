"""Production implementations of the finalize_project tool suite (task #68).

Before this, finalize_project's sub-skills (gather -> synthesize -> disclose ->
grade_and_sign -> record_in_ledger) referenced ~10 tools that existed ONLY as
test stubs, so the skill could not run in production. This module registers real
implementations on import (core/tools.py imports it, so `import core.tools`
registers everything). The 7th finalize tool, evaluate_artifact_rubric, lives in
core/tools.py as the grader bridge.

Each tool reads REAL lab data (the SoR ledger + findings dir) or the provider
cascade — no fabricated values. The two LLM-backed tools (synthesize /
analyze_holes) reuse the grader's provider-cascade resilience: an LLM outage
degrades to a safe, honest result (low honest_score / synthesis-failed note),
never a crash or a silent fabrication.

NOTE (deferred, surfaced to PI): the skill-execution layer has no production
TRIGGER yet — the agent loop drives raw tools, not skills. `tool_registry.make_invoker()`
+ skill_executor is the engine; how finalize_project gets invoked live (MCP tool
/ agent primitive / CLI) is a separate architecture decision. These tools + the
invoker make finalize_project runnable through the executor today (proven by the
e2e test), independent of that trigger choice.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core import log, tool_registry
from core.types import PermissionMode

LOG = log.get_logger("bert.finalize_tools")
LAB_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEDGER = "lab/sor/events.jsonl"


def _lab_root() -> Path:
    """Resolve relative paths against the active lab (if set via
    core.lab_context) else the bert-lab repo root — same rule as the Write tool,
    so finalize artifacts land where the rest of the lab writes."""
    from core.lab_context import get_active_lab_path
    return get_active_lab_path() or LAB_ROOT


# ── identity (passthrough used by every skill's pluck steps) ─────────


def _identity(**kwargs):
    """Return the `value` arg unchanged (the skill DSL's pluck primitive)."""
    return kwargs.get("value", kwargs)


# ── list_findings ────────────────────────────────────────────────────


def _list_findings(dir: str = "findings/", min_quality: float = 0.5) -> dict:
    """List the .md findings under `dir` with a quality_score, filtered by
    min_quality. quality_score defaults to 1.0 (neutral) — findings carry no
    declared score, and we do NOT fabricate one (per-finding scoring from the
    SoR ledger's confidence_1to10 is a deferred enrichment to avoid a full-ledger
    scan per call). min_quality is honored mechanically against the score."""
    d = Path(dir)
    if not d.is_absolute():
        d = _lab_root() /d
    files: list[dict] = []
    if d.exists():
        for p in sorted(d.rglob("*.md")):
            if not p.is_file():
                continue
            quality_score = 1.0
            if quality_score >= min_quality:
                root = _lab_root()
                files.append({"path": str(p.relative_to(root)) if p.is_relative_to(root)
                              else str(p), "quality_score": quality_score})
    return {"files": files}


# ── read_ledger_rows ─────────────────────────────────────────────────


def _read_ledger_rows(path: str = DEFAULT_LEDGER,
                      event_types: list[str] | None = None) -> dict:
    """Read the SoR jsonl ledger, returning rows whose event_class/event_type is
    in `event_types` (all rows if None). Each row -> {event_type, cycle_id, payload}."""
    p = Path(path)
    if not p.is_absolute():
        p = _lab_root() /p
    wanted = set(event_types) if event_types else None
    rows: list[dict] = []
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = ev.get("event_type") or ev.get("event_class")
                if wanted is not None and etype not in wanted:
                    continue
                cyc = ev.get("cycle_id", ev.get("cycle"))
                try:
                    cyc = int(cyc) if cyc is not None else None
                except (TypeError, ValueError):
                    cyc = None
                rows.append({"event_type": etype, "cycle_id": cyc,
                             "payload": ev.get("payload", ev)})
    return {"rows": rows}


# ── assemble_evidence_bundle ─────────────────────────────────────────


def _assemble_evidence_bundle(findings: list[dict] | None = None,
                              ledger_rows: list[dict] | None = None,
                              objective: str = "") -> dict:
    """Combine findings (with content read from disk) + ledger rows into a
    deduped evidence list, each carrying provenance + quality_score."""
    findings = findings or []
    ledger_rows = ledger_rows or []
    evidence: list[dict] = []
    cycles: set[int] = set()
    seen_paths: set[str] = set()
    for f in findings:
        path = f.get("path")
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        fp = Path(path)
        if not fp.is_absolute():
            fp = _lab_root() /fp
        content = ""
        if fp.exists() and fp.suffix == ".md":
            content = fp.read_text(encoding="utf-8", errors="replace")[:2000]
        evidence.append({
            "type": "finding", "source_path": path, "content": content,
            "provenance": {"source_path": path, "cycle": None, "agent": None},
            "quality_score": float(f.get("quality_score", 1.0)),
        })
    for r in ledger_rows:
        cyc = r.get("cycle_id")
        if isinstance(cyc, int):
            cycles.add(cyc)
        evidence.append({
            "type": "ledger", "source_path": DEFAULT_LEDGER,
            "content": json.dumps(r.get("payload", {}))[:1000],
            "provenance": {"cycle": cyc, "event_type": r.get("event_type")},
            "quality_score": 1.0,
        })
    return {"evidence": evidence, "count": len(evidence),
            "cycles_covered": sorted(cycles)}


# ── synthesize_artifact_body (LLM, cascade) ──────────────────────────

_SYNTH_SYS = (
    "You are bert's artifact synthesizer. Given an objective and an evidence "
    "list (each item has a source_path + content), write ONE polished markdown "
    "artifact that answers the objective, citing evidence inline with [^n] "
    "footnotes (n = 1-based evidence index). Do not invent claims beyond the "
    "evidence. Return ONLY JSON: {\"body\": \"<markdown>\", \"citations_used\": "
    "<int>, \"uncited_evidence\": [<0-based indices not cited>]}."
)


def _synthesize_artifact_body(evidence: list[dict] | None = None, objective: str = "",
                              target_grade: str = "A", max_words: int = 4000,
                              cascade=None) -> dict:
    from core.grader import DEFAULT_CASCADE, _cascade_json
    evidence = evidence or []
    lanes = cascade if cascade is not None else DEFAULT_CASCADE
    ev_text = "\n".join(
        f"[{i}] {e.get('source_path', '?')}: {str(e.get('content', ''))[:500]}"
        for i, e in enumerate(evidence))
    obj = _cascade_json([
        {"role": "system", "content": _SYNTH_SYS},
        {"role": "user", "content": (
            f"OBJECTIVE: {objective}\nTARGET GRADE: {target_grade}\n"
            f"MAX WORDS: {max_words}\n\nEVIDENCE:\n{ev_text}\n\nWrite now. JSON only.")},
    ], lanes, max_tokens=3000)
    if obj is None or not isinstance(obj.get("body"), str):
        return {"body": "(synthesis unavailable: all provider lanes failed)",
                "word_count": 0, "citations_used": 0,
                "uncited_evidence": list(range(len(evidence)))}
    body = obj["body"]
    uncited = obj.get("uncited_evidence") or []
    if not isinstance(uncited, list):
        uncited = []
    cited = obj.get("citations_used")
    if not isinstance(cited, int):
        cited = body.count("[^")
    return {"body": body, "word_count": len(body.split()),
            "citations_used": cited, "uncited_evidence": uncited}


# ── analyze_evidence_holes (LLM, cascade) ────────────────────────────

_HOLES_SYS = (
    "You are bert's gap auditor. Given the objective, the evidence, and the "
    "synthesized artifact, identify what the artifact does NOT answer or support: "
    "missing data, single-source claims, unaddressed parts of the objective. "
    "Write an honest gaps.md. Return ONLY JSON: {\"gaps_md\": \"<markdown>\", "
    "\"gap_count\": <int>, \"unanswered_questions\": [<str>], \"honest_score\": "
    "<0.0-1.0, higher = more candid about limitations>}."
)


def _analyze_evidence_holes(evidence: list[dict] | None = None, artifact: str = "",
                            objective: str = "", cascade=None) -> dict:
    from core.grader import DEFAULT_CASCADE, _cascade_json
    evidence = evidence or []
    lanes = cascade if cascade is not None else DEFAULT_CASCADE
    obj = _cascade_json([
        {"role": "system", "content": _HOLES_SYS},
        {"role": "user", "content": (
            f"OBJECTIVE: {objective}\n\nARTIFACT:\n{artifact[:6000]}\n\n"
            f"EVIDENCE COUNT: {len(evidence)}\n\nFind the gaps. JSON only.")},
    ], lanes, max_tokens=1500)
    if obj is None:
        return {"gaps_md": "# Gaps\n\n(gap analysis unavailable: provider lanes failed)",
                "gap_count": 0, "unanswered_questions": [], "honest_score": 0.0}
    uq = obj.get("unanswered_questions") or []
    return {
        "gaps_md": str(obj.get("gaps_md", "# Gaps\n")),
        "gap_count": int(obj["gap_count"]) if isinstance(obj.get("gap_count"), int) else 0,
        "unanswered_questions": uq if isinstance(uq, list) else [],
        "honest_score": float(obj["honest_score"]) if isinstance(obj.get("honest_score"), (int, float)) else 0.0,
    }


# ── claim-level contradiction flag (B-9, Sprint 6) ───────────────────


def _detect_claim_contradictions(claims: list[str] | None = None,
                                 artifact: str = "", cascade=None) -> dict:
    """Flag claim-vs-claim contradictions for the grader/PI. NOT a block (PI
    decision 2026-05-29): a contradiction may be legitimate scope/temporal
    nuance, so this informs the gaps.md + grade, it does not auto-fail.

    Pass `claims` directly, or `artifact` text to have claims extracted first.
    LLM outage -> is_inconclusive (NOT a silent clean pass), surfaced in gaps."""
    from core import contradiction
    if claims:
        res = contradiction.detect_contradictions(claims, cascade=cascade)
    else:
        res = contradiction.detect_in_artifact(artifact, cascade=cascade)
    out = res.to_dict()
    out["summary_md"] = _contradiction_summary_md(res)
    return out


def _contradiction_summary_md(res) -> str:
    if res.is_inconclusive:
        return ("# Claim contradictions\n\n(inconclusive: contradiction check "
                f"unavailable — {res.error}. Treat as unverified, not clean.)\n")
    if not res.has_contradictions:
        return "# Claim contradictions\n\nNone detected across the artifact's claims.\n"
    lines = [f"# Claim contradictions ({len(res.pairs)})\n"]
    for p in res.pairs:
        lines.append(
            f"- **[{p['severity']}/{p['kind']}]** \"{p['a']}\" vs \"{p['b']}\""
            + (f" — {p['rationale']}" if p["rationale"] else ""))
    return "\n".join(lines) + "\n"


# ── sha256_envelope (pure) ───────────────────────────────────────────


def _sha256_envelope(artifact: str = "", gaps: str = "", grade: str = "",
                     components: dict | None = None) -> dict:
    """SHA-256 over (artifact || gaps || canonical envelope JSON). The envelope
    is the signed anchor for the proof packet's grade."""
    from core import signing
    envelope = {"grade": grade, "components": components or {}}
    env_bytes = signing.canonical_json(envelope)
    h = hashlib.sha256(artifact.encode("utf-8") + gaps.encode("utf-8") + env_bytes)
    return {"hash": h.hexdigest(), "envelope": envelope}


# ── ledger row tools ─────────────────────────────────────────────────


def _validate_ledger_row(event_type: str = "", cycle_id=None, agent: str = "",
                         payload=None) -> dict:
    """Validate a ledger row has the required, well-typed fields."""
    errors: list[str] = []
    if not isinstance(event_type, str) or not event_type:
        errors.append("event_type must be a non-empty string")
    if not isinstance(cycle_id, int):
        errors.append("cycle_id must be an int")
    if not isinstance(agent, str) or not agent:
        errors.append("agent must be a non-empty string")
    if not isinstance(payload, dict):
        errors.append("payload must be an object")
    return {"ok": not errors, "errors": errors}


def _append_jsonl_atomic(path: str = DEFAULT_LEDGER, row: dict | None = None) -> dict:
    """Append one JSON row to a jsonl file. Returns the pre-append byte offset, a
    stable row_id derived from (cycle, type, payload hash), and an ISO timestamp."""
    from datetime import UTC, datetime
    row = dict(row or {})
    p = Path(path)
    if not p.is_absolute():
        p = _lab_root() /p
    p.parent.mkdir(parents=True, exist_ok=True)
    offset = p.stat().st_size if p.exists() else 0
    payload_hash = hashlib.sha256(
        json.dumps(row.get("payload", {}), sort_keys=True).encode()).hexdigest()[:8]
    row_id = f"evt-{row.get('cycle_id', 0)}-{row.get('event_type', 'row')}-{payload_hash}"
    appended_at = datetime.now(UTC).isoformat()
    row.setdefault("row_id", row_id)
    row.setdefault("ts", appended_at)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {"offset": offset, "row_id": row_id, "appended_at": appended_at}


def _finalize_ready_check(grade: str = "F", gaps_path: str = "",
                          ledger_row_id=None) -> bool:
    """Ready iff grade is A or B, gaps.md exists, and a ledger row was written."""
    gp = Path(gaps_path)
    if not gp.is_absolute() and gaps_path:
        gp = _lab_root() /gaps_path
    gaps_ok = bool(gaps_path) and gp.exists()
    return grade in ("A", "B") and gaps_ok and bool(ledger_row_id)


# ── registration ─────────────────────────────────────────────────────


def _reg(name: str, desc: str, schema: dict, handler, mode=PermissionMode.AUTO) -> None:
    tool_registry.register_function(name=name, description=desc,
                                    parameters_schema=schema, handler=handler,
                                    permission_mode=mode)


_OBJ = {"type": "object"}

_reg("identity", "Passthrough — returns the `value` arg unchanged (skill pluck primitive).",
     {"type": "object", "properties": {"value": {}}}, _identity)
_reg("list_findings", "List .md findings under a dir with quality_score, filtered by min_quality.",
     {"type": "object", "properties": {"dir": {"type": "string"},
      "min_quality": {"type": "number"}}}, _list_findings)
_reg("read_ledger_rows", "Read the SoR jsonl ledger; filter by event_types; return {event_type,cycle_id,payload} rows.",
     {"type": "object", "properties": {"path": {"type": "string"},
      "event_types": {"type": "array", "items": {"type": "string"}}}}, _read_ledger_rows)
_reg("assemble_evidence_bundle", "Combine findings (content read from disk) + ledger rows into a deduped evidence bundle with provenance.",
     {"type": "object", "properties": {"findings": {"type": "array"},
      "ledger_rows": {"type": "array"}, "objective": {"type": "string"}}},
     _assemble_evidence_bundle)
_reg("synthesize_artifact_body", "LLM-synthesize a polished, inline-cited markdown artifact from evidence (provider cascade).",
     {"type": "object", "properties": {"evidence": {"type": "array"},
      "objective": {"type": "string"}, "target_grade": {"type": "string"},
      "max_words": {"type": "integer"}}}, _synthesize_artifact_body)
_reg("analyze_evidence_holes", "LLM gap auditor: find what the artifact doesn't answer/support; emit honest gaps.md (provider cascade).",
     {"type": "object", "properties": {"evidence": {"type": "array"},
      "artifact": {"type": "string"}, "objective": {"type": "string"}}},
     _analyze_evidence_holes)
_reg("detect_claim_contradictions", "Flag claim-vs-claim contradictions (B-9) for the grader/PI — informs gaps.md, does NOT block. Pass `claims` or `artifact`.",
     {"type": "object", "properties": {"claims": {"type": "array", "items": {"type": "string"}},
      "artifact": {"type": "string"}}}, _detect_claim_contradictions)
_reg("sha256_envelope", "SHA-256 over (artifact || gaps || canonical grade-envelope) — the proof-packet grade anchor.",
     {"type": "object", "properties": {"artifact": {"type": "string"},
      "gaps": {"type": "string"}, "grade": {"type": "string"},
      "components": {"type": "object"}}}, _sha256_envelope)
_reg("validate_ledger_row", "Validate a ledger row has required, well-typed fields.",
     {"type": "object", "properties": {"event_type": {"type": "string"},
      "cycle_id": {"type": "integer"}, "agent": {"type": "string"},
      "payload": {"type": "object"}}}, _validate_ledger_row)
_reg("append_jsonl_atomic", "Append one JSON row to a jsonl file; return pre-append offset + stable row_id + timestamp.",
     {"type": "object", "properties": {"path": {"type": "string"},
      "row": {"type": "object"}}}, _append_jsonl_atomic, PermissionMode.DEFAULT)
_reg("finalize_ready_check", "True iff grade is A/B, gaps.md exists, and a ledger row was written.",
     {"type": "object", "properties": {"grade": {"type": "string"},
      "gaps_path": {"type": "string"}, "ledger_row_id": {"type": "string"}}},
     _finalize_ready_check)


__all__ = ["_identity", "_list_findings", "_read_ledger_rows",
           "_assemble_evidence_bundle", "_synthesize_artifact_body",
           "_analyze_evidence_holes", "_detect_claim_contradictions",
           "_sha256_envelope", "_validate_ledger_row",
           "_append_jsonl_atomic", "_finalize_ready_check"]
