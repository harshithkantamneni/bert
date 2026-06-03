"""Effort-triage: a cheap up-front classifier that decides how much machinery a
task warrants, so bert stops running a 4-fetch, 1500-char, multi-role research
ritual on a trivia lookup (the case that cost 253K tokens in the B7 benchmark).

classify(text) -> (effort, needs_grounding, confidence)
  effort: "trivial" | "standard" | "deep"
  needs_grounding: bool  — time-sensitive, must hit the web even if short
  confidence: float 0..1

Design (see benchmarks/B8_EFFICIENCY_AND_RAG_PLAN.md, WS0c):
- Stage 1 is a FREE deterministic heuristic over a FROZEN lexicon
  (core/library/effort_lexicon.yaml) — it only fast-paths the unambiguous
  trivia; everything with any complexity signal escalates.
- Stage 2 (optional `model_classify` hook) is consulted ONLY for the ambiguous
  middle, honouring the quality-first rule that a heuristic alone never decides
  a borderline case. Default None = heuristic-only (deterministic, testable).
- Quality-first guard: a judgment ask (ALWAYS_A keyword: review/judge/propose/
  decide/falsify/paper) is NEVER down-triaged below deep.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_LIB = Path(__file__).resolve().parent / "library" / "effort_lexicon.yaml"

# Judgment asks that must never be down-triaged (kept in sync with
# router.ALWAYS_A_KEYWORDS; duplicated here to avoid an import cycle).
_ALWAYS_DEEP = ("review", "judge", "propose", "verdict", "falsify",
                "falsifier", "red_team", "red team", "paper", "decide")


@lru_cache(maxsize=1)
def _lexicon() -> dict:
    import yaml
    with open(_LIB) as fh:
        return yaml.safe_load(fh)


def _has_any(text: str, markers: list[str]) -> bool:
    return any(m in text for m in markers)


def classify(text: str, *, model_classify=None) -> tuple[str, bool, float]:
    """Classify a task's effort tier. `model_classify(text) -> (effort,
    needs_grounding, confidence)` is an optional escalation consulted only for
    ambiguous cases; None keeps this fully deterministic."""
    lex = _lexicon()
    t = (text or "").lower().strip()

    needs_grounding = _has_any(t, lex.get("grounding_markers", []))

    # Quality-first: judgment asks are always deep, regardless of phrasing/length.
    if _has_any(t, _ALWAYS_DEEP):
        return ("deep", needs_grounding, 1.0)

    # Any complexity/quality signal -> deep.
    if _has_any(t, lex.get("deep_markers", [])):
        return ("deep", needs_grounding, 0.9)

    is_short = len(text or "") <= int(lex.get("trivial_max_chars", 220))
    # A single, non-compound question: one '?' and not a stack of sentences.
    single_clause = t.count("?") <= 1 and t.count(". ") <= 1
    has_lookup = _has_any(t, lex.get("trivial_lookup_markers", []))

    # Short, single-fact lookup that is NOT time-sensitive -> trivial fast-path.
    if is_short and single_clause and has_lookup and not needs_grounding:
        return ("trivial", False, 0.85)

    # Short but time-sensitive -> standard at least (a stale answer is wrong).
    if needs_grounding and is_short:
        return ("standard", True, 0.7)

    # Ambiguous middle: consult the model escalation if available, else standard.
    if model_classify is not None:
        try:
            eff, ground, conf = model_classify(text)
            if eff in ("trivial", "standard", "deep"):
                # Never let the model down-triage a grounding-required ask.
                if needs_grounding and eff == "trivial":
                    eff = "standard"
                return (eff, bool(ground or needs_grounding), float(conf))
        except Exception:  # noqa: BLE001 — escalation must never crash dispatch
            pass

    return ("standard", needs_grounding, 0.6)
