"""Researcher capability battery — ArXiv synthesis quality.

20-task corpus: each task asks the model to summarize a fictional
ArXiv abstract into a one-paragraph synthesis with claim, evidence,
caveat. Scorer (live mode) checks for those three structural elements
+ length window (80-260 words).
"""

from __future__ import annotations

import re

from . import _common
from ._common import Task, BatteryRunResult

ROLE = "researcher"
REFERENCE_SET = "researcher_battery_v1"


TASKS: list[Task] = [
    Task(
        id=f"researcher_{i:02d}",
        prompt=(
            f"Synthesize the following paper into one paragraph: claim, "
            f"evidence, caveat. Topic {i}: {topic}"
        ),
        rubric_hint="paragraph must include claim, evidence, caveat",
    )
    for i, topic in enumerate(
        [
            "training-free skill libraries for autonomous agents",
            "KV-cache passing between LLM agents (KVComm)",
            "LLMLingua-2 prompt compression",
            "Agent2Agent (A2A) protocol",
            "verifiable AI via cross-family adversarial review",
            "Quaker-style discernment pipelines in agent dispatch",
            "free-tier multi-provider LLM cascades",
            "Coconut latent reasoning vs LatentMAS",
            "RouteLLM cost-quality smart routing",
            "Stafford Beer's Viable System Model in AI orchestration",
            "Sigstore-style append-only event-stream auditability",
            "MemGPT/Letta tiered memory function-calling APIs",
            "ColBERT v2 reranking on hybrid vector + graph retrieval",
            "Inspect AI vs deepeval for LLM evaluation",
            "RouteLLM weights divergence across role-specific tasks",
            "PolyKV / Q-KVComm adaptive layer-wise quantization",
            "consolidator agents for memory-tier promotion",
            "permission gates for self-modifying AI (P-005 pattern)",
            "Apache AGE vs FalkorDB for agent knowledge graphs",
            "AGNTCY collective conventions vs A2A vs MCP",
        ], start=1)
]


def score(task: Task, response: str) -> float:
    """Structural scorer. Three components, each 0..0.34, sum capped at 1.0."""
    text = (response or "").strip()
    if not text:
        return 0.0
    word_count = len(text.split())
    length_ok = 60 <= word_count <= 320
    has_claim = bool(re.search(r"\b(claim|propose|argue|find|report)\b", text, re.I))
    has_evidence = bool(re.search(r"\b(show|demonstrate|experiment|measure|benchmark|result|evaluation)\b", text, re.I))
    has_caveat = bool(re.search(r"\b(caveat|limitation|however|but|though|qualifies|risk)\b", text, re.I))
    return (
        (0.34 if length_ok else 0.10)
        + (0.22 if has_claim else 0.0)
        + (0.22 if has_evidence else 0.0)
        + (0.22 if has_caveat else 0.0)
    )


def run(provider: str, model: str, *, live: bool = False,
        sample: int | None = None) -> BatteryRunResult:
    return _common.run_battery(
        ROLE, TASKS, score,
        provider=provider, model=model,
        live=live, sample=sample,
    )
