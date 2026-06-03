"""4-judge median+variance artifact grader (Sprint 5 item 24).

Four judge personas each score a finalized artifact on ALL 8 quality dimensions
(0-5), calibrated against core/library/grading_rubric.yaml. The grader takes the
MEDIAN per dimension (B-5: median, not max — a single generous or harsh judge
cannot swing the grade) and reports the population VARIANCE per dimension as a
judge-disagreement signal. The medians collapse through the mission's
QualityContract to a weighted 0-1 score that gates acceptance.

Resilience (S-10): each judge runs against a provider CASCADE. A judge that
fails every lane is DROPPED (recorded), and the grade is computed over the
survivors rather than crashing — a provider outage degrades the grade's
confidence (fewer judges, higher effective variance), it does not block
finalization.

The judges (B-6 — the 4th covers reproducibility + efficiency):
  correctness       — factual/logical correctness + completeness
  gap_finder        — what is missing / unsupported; completeness + defensibility
  honesty           — overclaim detection; honesty + provenance
  repro_efficiency  — reproducibility + efficiency + usability
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from core import log, quality

LOG = log.get_logger("bert.grader")
LAB_ROOT = Path(__file__).resolve().parent.parent
RUBRIC_PATH = LAB_ROOT / "core" / "library" / "grading_rubric.yaml"

JUDGES: tuple[str, ...] = ("correctness", "gap_finder", "honesty", "repro_efficiency")

_PERSONAS = {
    "correctness": "scrutinize factual and logical correctness and completeness above all else",
    "gap_finder": "hunt for what is missing, unstated, or unsupported; weigh completeness and defensibility",
    "honesty": "detect overclaiming; reward surfaced limitations and checkable provenance",
    "repro_efficiency": "judge whether the result is reproducible and was achieved without waste",
}

# Default free-tier provider cascade (provider, model). Ordered by the
# free-tier landscape: fast lanes first, cross-family fallbacks after. Callers
# (and BYO-key users) can override.
DEFAULT_CASCADE: list[tuple[str, str | None]] = [
    ("groq", "llama-3.3-70b-versatile"),
    ("cerebras", "llama-3.3-70b"),
    ("nvidia", "meta/llama-3.3-70b-instruct"),
    ("gemini", "gemini-2.0-flash"),
]


@dataclass
class JudgeScore:
    judge: str
    dimensions: dict[str, int]   # dim -> 0-5
    provider: str
    rationale: str = ""


@dataclass
class GradeResult:
    medians: dict[str, float]
    variances: dict[str, float]
    weighted_score: float
    passes: bool
    overall_variance: float
    judges: list[JudgeScore] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "medians": self.medians,
            "variances": self.variances,
            "weighted_score": self.weighted_score,
            "passes": self.passes,
            "overall_variance": self.overall_variance,
            "judges": [
                {"judge": j.judge, "provider": j.provider,
                 "dimensions": j.dimensions, "rationale": j.rationale}
                for j in self.judges
            ],
            "dropped": self.dropped,
        }


# ── Rubric ───────────────────────────────────────────────────────────


def _load_rubric(path: Path = RUBRIC_PATH) -> dict:
    import yaml
    if not path.exists():
        raise FileNotFoundError(f"grading rubric not found: {path}")
    rubric = yaml.safe_load(path.read_text())
    missing = set(quality.DIMENSIONS) - set(rubric.get("dimensions", {}))
    if missing:
        raise ValueError(f"rubric missing dimensions: {sorted(missing)}")
    return rubric


def _rubric_text(rubric: dict) -> str:
    lines: list[str] = []
    for dim in quality.DIMENSIONS:
        body = rubric["dimensions"][dim]
        lines.append(f"{dim} — {body['description']}")
        for level in range(6):
            lines.append(f"  {level}: {body['anchors'][level]}")
    return "\n".join(lines)


# ── Prompts ──────────────────────────────────────────────────────────


def _judge_system_prompt(judge: str, rubric: dict) -> str:
    dims = ", ".join(quality.DIMENSIONS)
    return (
        f"You are bert's {judge} judge. Your lens: {_PERSONAS[judge]}. "
        f"Grade the artifact on ALL 8 dimensions ({dims}) using the rubric "
        f"anchors below — score every dimension 0-5, including ones outside "
        f"your lens. An all-5 or all-identical score is almost never real; "
        f"differentiate.\n\n"
        f"Return ONLY a JSON object with an integer 0-5 for each of the 8 "
        f"dimensions plus a short \"rationale\" string. No prose outside JSON.\n\n"
        f"RUBRIC:\n{_rubric_text(rubric)}"
    )


def _judge_user_prompt(artifact: str, gaps: str, evidence_count: int) -> str:
    return (
        f"ARTIFACT (the finalized deliverable):\n{artifact}\n\n"
        f"DECLARED GAPS / LIMITATIONS:\n{gaps}\n\n"
        f"RECORDED EVIDENCE COUNT: {evidence_count}\n\n"
        f"Score now. JSON only."
    )


def _parse_scores(text: str) -> dict[str, int] | None:
    """Parse a judge's JSON response into validated 0-5 dimension scores.
    Returns None if unparseable or any dimension is missing / out of range."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    scores: dict[str, int] = {}
    for dim in quality.DIMENSIONS:
        v = obj.get(dim)
        if not isinstance(v, int) or isinstance(v, bool) or not (0 <= v <= 5):
            return None
        scores[dim] = v
    return scores


# ── Judge execution (provider cascade) ───────────────────────────────


def _run_one_judge(judge: str, rubric: dict, artifact: str, gaps: str,
                   evidence_count: int,
                   cascade: list[tuple[str, str | None]],
                   system_prompt_fn=None) -> JudgeScore | None:
    """Run one judge against the provider cascade. Returns a JudgeScore from
    the first lane that answers with a valid score vector, or None if every
    lane fails (the judge is then dropped by the caller).

    `system_prompt_fn(judge, rubric) -> str` optionally replaces the default
    house-framed judge prompt (e.g. a neutral evaluator persona for an
    A/B benchmark). None keeps the production default unchanged."""
    from core import provider as _prov  # lazy — avoid import cycle
    build_system = system_prompt_fn or _judge_system_prompt
    messages = [
        {"role": "system", "content": build_system(judge, rubric)},
        {"role": "user", "content": _judge_user_prompt(artifact, gaps, evidence_count)},
    ]
    for prov_name, model in cascade:
        try:
            resp = _prov.call(
                prov_name, messages, model=model, max_tokens=700,
                temperature=0.2, response_format={"type": "json_object"},
                timeout=30.0,
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("grader: %s judge lane %s raised: %s", judge, prov_name, exc)
            continue
        if resp.finish_reason == "error" or resp.text.startswith("[bert]"):
            LOG.warning("grader: %s judge lane %s failed: %s",
                        judge, prov_name, resp.text[:120])
            continue
        scores = _parse_scores(resp.text)
        if scores is None:
            LOG.warning("grader: %s judge lane %s unparseable", judge, prov_name)
            continue
        rationale = ""
        try:
            rationale = str(json.loads(resp.text).get("rationale", ""))[:400]
        except (json.JSONDecodeError, TypeError):
            pass
        return JudgeScore(judge=judge, dimensions=scores,
                          provider=prov_name, rationale=rationale)
    return None


# ── Aggregation (pure — no network) ──────────────────────────────────


def aggregate(judge_scores: list[JudgeScore],
              contract: quality.QualityContract) -> GradeResult:
    """Collapse per-judge dimension scores into medians + variances, then a
    weighted 0-1 score + pass/fail via the QualityContract. Pure function:
    the median/variance/passes invariants are provable without any LLM."""
    if not judge_scores:
        zero = dict.fromkeys(quality.DIMENSIONS, 0.0)
        return GradeResult(medians=zero, variances=zero, weighted_score=0.0,
                           passes=contract.passes({}), overall_variance=0.0,
                           judges=[], dropped=[])
    medians: dict[str, float] = {}
    variances: dict[str, float] = {}
    for dim in quality.DIMENSIONS:
        vals = [j.dimensions[dim] for j in judge_scores]
        medians[dim] = float(statistics.median(vals))
        variances[dim] = float(statistics.pvariance(vals)) if len(vals) > 1 else 0.0
    weighted = contract.weighted_score(medians)
    overall_var = statistics.fmean(variances.values()) if variances else 0.0
    return GradeResult(
        medians=medians, variances=variances, weighted_score=weighted,
        passes=weighted >= contract.pass_threshold,
        overall_variance=overall_var, judges=list(judge_scores), dropped=[],
    )


def grade_artifact(artifact: str, gaps: str, *,
                   contract: quality.QualityContract,
                   rubric_path: Path = RUBRIC_PATH,
                   cascade: list[tuple[str, str | None]] | None = None,
                   evidence_count: int = 0,
                   system_prompt_fn=None) -> GradeResult:
    """Grade `artifact` with the 4 judges (each over the provider cascade) and
    aggregate. Judges that fail every lane are recorded in `.dropped` and the
    grade is computed over the survivors.

    `system_prompt_fn(judge, rubric) -> str` optionally overrides the default
    judge framing for every judge (used by the B7 benchmark for a neutral
    evaluator persona); None preserves production behavior."""
    rubric = _load_rubric(rubric_path)
    lanes = cascade if cascade is not None else DEFAULT_CASCADE
    survivors: list[JudgeScore] = []
    dropped: list[str] = []
    for judge in JUDGES:
        js = _run_one_judge(judge, rubric, artifact, gaps, evidence_count, lanes,
                            system_prompt_fn=system_prompt_fn)
        if js is None:
            dropped.append(judge)
        else:
            survivors.append(js)
    res = aggregate(survivors, contract)
    res.dropped = dropped
    return res


# ── gaps.md content-quality validator (Q-5) ──────────────────────────

GAPS_THRESHOLD = 0.6

_GAPS_SYS = (
    "You are bert's gaps.md auditor. Score the gaps / limitations text on three "
    "0-5 sub-dimensions: completeness (are the real limitations covered, or are "
    "obvious ones missing?), specificity (concrete and falsifiable, not "
    "hand-wavy?), honesty (does it own hard failures rather than minimize them?). "
    "A bare \"no known gaps\" or a vague disclaimer scores low on all three. "
    "Return ONLY JSON: {\"completeness\": int, \"specificity\": int, "
    "\"honesty\": int, \"rationale\": str}."
)


@dataclass
class GapsValidation:
    completeness: int
    specificity: int
    honesty: int
    score: float          # mean of the 3 sub-dims, normalized 0-1
    passes: bool
    rationale: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "completeness": self.completeness, "specificity": self.specificity,
            "honesty": self.honesty, "score": self.score, "passes": self.passes,
            "rationale": self.rationale, "error": self.error,
        }


def _cascade_json(messages: list[dict], cascade: list[tuple[str, str | None]],
                  *, max_tokens: int = 500) -> dict | None:
    """Call the provider cascade; return the first lane's parsed JSON dict, or
    None if every lane fails/unparses."""
    from core import provider as _prov  # lazy — avoid import cycle
    for prov_name, model in cascade:
        try:
            resp = _prov.call(prov_name, messages, model=model,
                              max_tokens=max_tokens, temperature=0.2,
                              response_format={"type": "json_object"}, timeout=30.0)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("grader: gaps lane %s raised: %s", prov_name, exc)
            continue
        if resp.finish_reason == "error" or resp.text.startswith("[bert]"):
            continue
        try:
            obj = json.loads(resp.text)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def validate_gaps(gaps_text: str, *, threshold: float = GAPS_THRESHOLD,
                  cascade: list[tuple[str, str | None]] | None = None) -> GapsValidation:
    """LLM-judge the QUALITY of a gaps.md (completeness/specificity/honesty),
    not just that the file exists. LLM failure -> non-crashing BLOCK with an
    error, never a silent pass."""
    lanes = cascade if cascade is not None else DEFAULT_CASCADE
    obj = _cascade_json([
        {"role": "system", "content": _GAPS_SYS},
        {"role": "user", "content": f"GAPS / LIMITATIONS TEXT:\n{gaps_text}\n\nScore now. JSON only."},
    ], lanes)
    if obj is None:
        return GapsValidation(0, 0, 0, 0.0, False,
                              error="gaps validation failed (all provider lanes)")

    def _clamp(v: object) -> int:
        return v if isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 5 else 0

    c, s, h = (_clamp(obj.get("completeness")), _clamp(obj.get("specificity")),
               _clamp(obj.get("honesty")))
    score = (c + s + h) / 15.0   # mean of 3 dims, normalized to 0-1
    return GapsValidation(c, s, h, score, score >= threshold,
                          rationale=str(obj.get("rationale", ""))[:400])
