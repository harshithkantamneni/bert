"""B-9: claim-level contradiction detection (Sprint 6 — organic growth).

Detects when two CLAIMS in the same artifact/corpus logically conflict. The
spec is explicit: claim-vs-claim reasoning, NOT whole-document embedding
similarity. This is the dual of lineage.py — lineage traces claims->findings
(is each claim supported?); this traces claims->claims (do any two conflict?).

Decision (PI, 2026-05-29): a detected contradiction is a FLAG that informs the
grader / PI, NOT a hard finalization block. A "contradiction" can be a
legitimate scope, magnitude, or temporal nuance ("latency rose under load" vs
"latency fell at idle"), so a hard block would false-fail honest artifacts. The
finalize grader consumes this signal; the human decides.

Quality-first (P-8): detection is LLM-judged over the actual claim text, not
regex/keyword antonym matching — antonyms ("up"/"down") are neither necessary
nor sufficient for a logical contradiction. The grader's provider-cascade
resilience is reused: an LLM failure degrades to a non-crashing INCONCLUSIVE
result (method="unavailable"), never a silent "no contradictions" clean pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from core import log

LOG = log.get_logger("bert.contradiction")

# Reuse the grader's free-tier cascade default (same as lineage.py).
try:
    from core.grader import DEFAULT_CASCADE
except Exception:  # noqa: BLE001 — grader optional at import time
    DEFAULT_CASCADE = [("groq", "llama-3.3-70b-versatile")]

# Contradiction taxonomy. An unrecognized kind from the judge normalizes to
# "unspecified" rather than being dropped — the pair still matters.
KINDS = frozenset({"direct", "scope", "magnitude", "temporal", "definitional"})
SEVERITIES = frozenset({"low", "medium", "high"})
DEFAULT_SEVERITY = "medium"


@dataclass
class ContradictionResult:
    pairs: list[dict] = field(default_factory=list)
    n_claims: int = 0
    method: str = "llm-v1"  # "llm-v1" | "trivial" | "unavailable"
    error: str | None = None

    @property
    def has_contradictions(self) -> bool:
        return len(self.pairs) > 0

    @property
    def is_inconclusive(self) -> bool:
        """True when we could NOT determine contradictions (LLM unreachable).
        Distinct from has_contradictions=False, which means we checked and
        found none. The grader must treat inconclusive != clean."""
        return self.method == "unavailable"

    def to_dict(self) -> dict:
        return {
            "pairs": self.pairs,
            "n_claims": self.n_claims,
            "method": self.method,
            "has_contradictions": self.has_contradictions,
            "is_inconclusive": self.is_inconclusive,
            "error": self.error,
        }


# ── pure parsing layer ───────────────────────────────────────────────


def _parse_pairs(obj: dict, claims: list[str]) -> list[dict]:
    """Turn the judge's JSON into normalized, validated, deduped pairs.

    Validation: a, b must be distinct in-range integer indices into `claims`.
    Normalization: order each pair (a<b); dedupe by (a,b); default unknown
    kind->"unspecified" and bad severity->"medium". Invalid entries are dropped.
    """
    raw = obj.get("contradictions")
    if not isinstance(raw, list):
        return []
    n = len(claims)
    seen: set[tuple[int, int]] = set()
    pairs: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        a, b = entry.get("a"), entry.get("b")
        if not (isinstance(a, int) and isinstance(b, int)):
            continue
        if a == b or not (0 <= a < n) or not (0 <= b < n):
            continue
        lo, hi = (a, b) if a < b else (b, a)
        if (lo, hi) in seen:
            continue
        seen.add((lo, hi))
        kind = entry.get("kind")
        kind = kind if kind in KINDS else "unspecified"
        severity = entry.get("severity")
        severity = severity if severity in SEVERITIES else DEFAULT_SEVERITY
        pairs.append({
            "a_index": lo,
            "b_index": hi,
            "a": claims[lo],
            "b": claims[hi],
            "kind": kind,
            "severity": severity,
            "rationale": str(entry.get("rationale", "")),
        })
    return pairs


# ── LLM layer (provider cascade) ─────────────────────────────────────


def _llm_json(messages: list[dict],
              cascade: list[tuple[str, str | None]]) -> dict | None:
    """Call the provider cascade; return the first lane's parsed JSON object,
    or None if every lane fails / is unparseable. (Same shape as lineage.)"""
    from core import provider as _prov  # lazy — avoid import cycle
    for prov_name, model in cascade:
        try:
            resp = _prov.call(prov_name, messages, model=model, max_tokens=1500,
                              temperature=0.1, response_format={"type": "json_object"},
                              timeout=30.0)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("contradiction: lane %s raised: %s", prov_name, exc)
            continue
        if resp.finish_reason == "error" or resp.text.startswith("[bert]"):
            LOG.warning("contradiction: lane %s failed: %s", prov_name, resp.text[:120])
            continue
        try:
            parsed = json.loads(resp.text)
        except (json.JSONDecodeError, TypeError):
            LOG.warning("contradiction: lane %s unparseable", prov_name)
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


_DETECT_SYS = (
    "You find pairs of CLAIMS that logically CONTRADICT each other. Two claims "
    "contradict when they cannot both be true of the same subject under the same "
    "conditions. A mere topical overlap, a different subject, or a different "
    "scope/time is NOT a contradiction. Classify each contradicting pair: "
    "kind is one of direct|scope|magnitude|temporal|definitional; severity is "
    "low|medium|high. Return ONLY JSON: {\"contradictions\": [{\"a\": <index>, "
    "\"b\": <index>, \"kind\": \"...\", \"severity\": \"...\", \"rationale\": "
    "\"...\"}, ...]}. Use the numbered indices. Empty list if none contradict."
)


def detect_contradictions(
    claims: list[str], *,
    cascade: list[tuple[str, str | None]] | None = None,
) -> ContradictionResult:
    """Detect claim-vs-claim contradictions in a single batched LLM call.

    Fewer than 2 claims -> trivial (no provider call). LLM unreachable ->
    method="unavailable" + error (inconclusive, NOT a clean pass).
    """
    n = len(claims)
    if n < 2:
        return ContradictionResult(pairs=[], n_claims=n, method="trivial")
    lanes = cascade if cascade is not None else DEFAULT_CASCADE
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims))
    obj = _llm_json([
        {"role": "system", "content": _DETECT_SYS},
        {"role": "user", "content": (
            f"CLAIMS:\n{numbered}\n\nReturn the contradicting pairs. JSON only.")},
    ], lanes)
    if obj is None:
        return ContradictionResult(
            pairs=[], n_claims=n, method="unavailable",
            error="contradiction check failed (all provider lanes)")
    pairs = _parse_pairs(obj, claims)
    return ContradictionResult(pairs=pairs, n_claims=n, method="llm-v1")


def detect_in_artifact(
    artifact: str, *,
    cascade: list[tuple[str, str | None]] | None = None,
) -> ContradictionResult:
    """Convenience: extract claims from raw artifact text (reusing the lineage
    LLM extractor) then detect contradictions among them."""
    from core import lineage
    lanes = cascade if cascade is not None else DEFAULT_CASCADE
    claims = lineage._extract_claims(artifact, lanes)
    if claims is None:
        return ContradictionResult(
            pairs=[], n_claims=0, method="unavailable",
            error="claim extraction failed (all provider lanes)")
    return detect_contradictions(claims, cascade=lanes)
