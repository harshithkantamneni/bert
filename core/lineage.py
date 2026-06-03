"""Finalization lineage check (Sprint 5 Q-4).

Before an artifact is finalized, >=80% of its claims must trace back to recorded
findings — otherwise the artifact asserts things the lab never actually
established. This is a hard finalization gate, distinct from
artifact_acceptance.py (which tracks the acceptance RATE across artifacts, not
claim-level traceability within one).

Quality-first (P-8): claim extraction and claim->finding matching are LLM-judged
(constrained JSON), not regex keyword overlap — a regex would count a claim
"traced" because a word coincides, which is exactly the false-confidence this
gate exists to prevent. The grader's provider-cascade resilience pattern is
reused: an LLM failure degrades to a non-crashing BLOCK (we cannot verify
lineage), never a silent pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from core import log

LOG = log.get_logger("bert.lineage")

DEFAULT_THRESHOLD = 0.80
# Reuse the grader's free-tier cascade default.
try:
    from core.grader import DEFAULT_CASCADE
except Exception:  # noqa: BLE001 — grader optional at import time
    DEFAULT_CASCADE = [("groq", "llama-3.3-70b-versatile")]


@dataclass
class LineageResult:
    traceability: float
    traced: int
    total: int
    passes: bool
    threshold: float
    claim_traces: list[dict] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "traceability": self.traceability,
            "traced": self.traced,
            "total": self.total,
            "passes": self.passes,
            "threshold": self.threshold,
            "claim_traces": self.claim_traces,
            "error": self.error,
        }


def _score(claim_traces: list[dict], threshold: float) -> tuple[float, int, int, bool]:
    """Pure: traceability ratio + gate. Zero claims is a vacuous pass (nothing
    unsupported), but the caller flags total==0 separately."""
    total = len(claim_traces)
    traced = sum(1 for t in claim_traces if t.get("supported"))
    if total == 0:
        return 1.0, 0, 0, threshold <= 1.0
    ratio = traced / total
    return ratio, traced, total, ratio >= threshold


# ── LLM helpers (provider cascade) ───────────────────────────────────


def _llm_json(messages: list[dict],
              cascade: list[tuple[str, str | None]]) -> dict | None:
    """Call the provider cascade; return the first lane's parsed JSON object,
    or None if every lane fails / is unparseable."""
    from core import provider as _prov  # lazy — avoid import cycle
    for prov_name, model in cascade:
        try:
            resp = _prov.call(prov_name, messages, model=model, max_tokens=1200,
                              temperature=0.1, response_format={"type": "json_object"},
                              timeout=30.0)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("lineage: lane %s raised: %s", prov_name, exc)
            continue
        if resp.finish_reason == "error" or resp.text.startswith("[bert]"):
            LOG.warning("lineage: lane %s failed: %s", prov_name, resp.text[:120])
            continue
        try:
            obj = json.loads(resp.text)
        except (json.JSONDecodeError, TypeError):
            LOG.warning("lineage: lane %s unparseable", prov_name)
            continue
        if isinstance(obj, dict):
            return obj
    return None


_EXTRACT_SYS = (
    "You extract the verifiable factual CLAIMS made by an artifact. A claim is a "
    "declarative assertion that could be checked true or false (a result, a "
    "measurement, a causal statement). Ignore hedges, questions, and meta-text. "
    "Return ONLY JSON: {\"claims\": [\"claim 1\", \"claim 2\", ...]}."
)

_TRACE_SYS = (
    "You judge whether each numbered claim is SUPPORTED by the findings corpus "
    "below. Supported means the findings provide direct evidence for the claim — "
    "not merely a topical word overlap. Return ONLY JSON: {\"traces\": "
    "[{\"claim_index\": 0, \"supported\": true, \"evidence\": \"<which finding>\"}, ...]} "
    "with one entry per claim."
)


def _extract_claims(artifact: str,
                    cascade: list[tuple[str, str | None]]) -> list[str] | None:
    obj = _llm_json([
        {"role": "system", "content": _EXTRACT_SYS},
        {"role": "user", "content": f"ARTIFACT:\n{artifact}\n\nExtract the claims. JSON only."},
    ], cascade)
    if obj is None:
        return None
    claims = obj.get("claims")
    if not isinstance(claims, list):
        return None
    return [str(c) for c in claims]


def _trace_claims(claims: list[str], findings: str,
                  cascade: list[tuple[str, str | None]]) -> list[dict] | None:
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims))
    obj = _llm_json([
        {"role": "system", "content": _TRACE_SYS},
        {"role": "user", "content": (
            f"CLAIMS:\n{numbered}\n\nFINDINGS CORPUS:\n{findings}\n\n"
            f"Judge each claim. JSON only.")},
    ], cascade)
    if obj is None:
        return None
    raw = obj.get("traces")
    if not isinstance(raw, list):
        return None
    # Map by claim_index; a claim with no/invalid entry counts as unsupported.
    supported_by_idx: dict[int, dict] = {}
    for entry in raw:
        if isinstance(entry, dict) and isinstance(entry.get("claim_index"), int):
            supported_by_idx[entry["claim_index"]] = entry
    traces: list[dict] = []
    for i, claim in enumerate(claims):
        entry = supported_by_idx.get(i, {})
        traces.append({
            "claim": claim,
            "supported": bool(entry.get("supported", False)),
            "evidence": str(entry.get("evidence", "")),
        })
    return traces


def check_lineage(artifact: str, findings: str, *,
                  threshold: float = DEFAULT_THRESHOLD,
                  cascade: list[tuple[str, str | None]] | None = None) -> LineageResult:
    """Extract the artifact's claims, trace each against `findings`, and gate on
    >=`threshold` traceability. LLM failure -> non-crashing BLOCK with an error,
    never a silent pass."""
    lanes = cascade if cascade is not None else DEFAULT_CASCADE
    claims = _extract_claims(artifact, lanes)
    if claims is None:
        return LineageResult(0.0, 0, 0, False, threshold,
                             error="claim extraction failed (all provider lanes)")
    if not claims:
        # genuinely no claims — vacuous pass, but surface total==0
        tr, traced, total, passes = _score([], threshold)
        return LineageResult(tr, traced, total, passes, threshold, claim_traces=[])
    traces = _trace_claims(claims, findings, lanes)
    if traces is None:
        return LineageResult(0.0, 0, len(claims), False, threshold,
                             error="claim tracing failed (all provider lanes)")
    tr, traced, total, passes = _score(traces, threshold)
    return LineageResult(tr, traced, total, passes, threshold, claim_traces=traces)
