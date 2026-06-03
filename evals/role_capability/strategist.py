"""Strategist capability battery — option-tree completeness.

15-task corpus: each task gives a research finding and asks for a
ranked candidate list (≥3 options, each with rationale + tradeoff).
"""

from __future__ import annotations

import re

from . import _common
from ._common import BatteryRunResult, Task

ROLE = "strategist"
REFERENCE_SET = "strategist_battery_v1"


TASKS: list[Task] = [
    Task(
        id=f"strategist_{i:02d}",
        prompt=(
            "Given this research finding, generate a ranked list of 3-5 "
            "product / direction candidates. For each: name, rationale, "
            "and the strongest tradeoff. Finding: " + topic
        ),
        rubric_hint="≥3 candidates with name + rationale + tradeoff each",
    )
    for i, topic in enumerate(
        [
            "L-23 AutoSkill training-free skill library lands first-attempt routing wins on local Ollama by 70%",
            "Cross-family adversarial review (P-VS-02) catches 4× more verdict drift than same-family",
            "Free-tier model providers shift rosters monthly; capability-per-role matrix needed",
            "KV-cache passing between local Ollama processes yields 5-10× speedup on same-family chains",
            "Pre-registered falsifiers reduce post-hoc rationalization rate from 40% to 8%",
            "Memory tier promotion (recall → core) is most accurate when delayed 7+ cycles",
            "Stand-aside concerns propagate downstream 70%+ but only 40% get addressed within 5 cycles",
            "RouteLLM saves 85% cost at 95% quality on general-purpose benchmarks",
            "Pace-layered SoR/SoD/SoI structure cuts file-corruption risk by 95%",
            "LLMLingua-2 compresses 4-10× with BERTScore F1 ≥ 0.92 on cross-family judge legs",
            "Seasoning queue depth correlates with mission focus drift; threshold 25 unrevived",
            "Daily /now page generation improves PI mental model continuity",
            "Telegram-side approval gates reduce friction vs file-watch by 10×",
            "AGNTCY conventions diverge from A2A; pick one or build a bridge",
            "Cross-family Qwen-via-NVIDIA is the strongest free-tier evaluator slot",
        ], start=1)
]


def score(task: Task, response: str) -> float:
    text = (response or "").strip()
    if not text:
        return 0.0
    # Count candidate markers (numbered list, bullets, or "Option N")
    candidates = max(
        len(re.findall(r"^\s*\d+\.", text, re.M)),
        len(re.findall(r"^\s*[-*]", text, re.M)),
        len(re.findall(r"\b[Oo]ption \d+", text)),
    )
    has_rationale = bool(re.search(r"\b(rationale|because|since|reason)\b", text, re.I))
    has_tradeoff = bool(re.search(r"\b(tradeoff|trade-off|but|however|cost|drawback|risk)\b", text, re.I))
    return (
        min(0.50, 0.15 * candidates)
        + (0.25 if has_rationale else 0.0)
        + (0.25 if has_tradeoff else 0.0)
    )


def run(provider: str, model: str, *, live: bool = False,
        sample: int | None = None) -> BatteryRunResult:
    return _common.run_battery(
        ROLE, TASKS, score,
        provider=provider, model=model,
        live=live, sample=sample,
    )
