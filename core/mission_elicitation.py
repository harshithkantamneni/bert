"""Mission elicitation — clarifying questions before spend (recheck Q-1).

When the classifier is uncertain (confidence < 0.7) or the mission text
is vague, bert asks 0-3 targeted clarifying questions BEFORE running a
cycle, rather than burning tokens on a guess. Each question names the
ambiguous field, the question itself, and why it's being asked.

This is the front-half of the multi-turn elicitation loop: the host
surfaces these questions, the user answers, and the (now-specific)
mission is re-classified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as dc_field

CONFIDENCE_FLOOR = 0.7   # below this the classification is too shaky to run on
MAX_QUESTIONS = 3

# Filler tokens that signal a mission hasn't been pinned to anything concrete.
_VAGUE_TOKENS = {
    "stuff", "things", "thing", "something", "anything", "whatever", "etc",
}
_STOPWORDS = {
    "a", "an", "the", "of", "to", "for", "on", "in", "into", "and", "or",
    "do", "please", "some", "this", "that", "it", "with", "about", "look",
}
# Words that hint a research mission so we can ask about recency.
_RESEARCH_HINTS = {"survey", "papers", "research", "literature", "compare", "review"}
_TIME_HINTS = {"month", "months", "year", "years", "recent", "latest", "since", "2024", "2025", "2026", "q1", "q2", "q3", "q4"}


@dataclass
class ClarifyingQuestion:
    field: str        # the ambiguous dimension (scope / intent / time_horizon / ...)
    question: str     # the question text shown to the user (ends with '?')
    why: str          # rationale — why bert is asking
    options: tuple[str, ...] = dc_field(default_factory=tuple)


def _content_words(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9@]+", text.lower())
    return [w for w in words if w not in _STOPWORDS]


def is_vague(mission_text: str) -> bool:
    """True when the mission is too thin to scaffold against — too few
    content words, or dominated by filler tokens."""
    words = re.findall(r"[a-z0-9@]+", mission_text.lower())
    content = _content_words(mission_text)
    if len(content) < 4:
        return True
    return any(t in words for t in _VAGUE_TOKENS) and len(content) < 6


def elicit(mission_text: str, profile) -> list[ClarifyingQuestion]:
    """Return 0-3 clarifying questions. Empty when the mission is both
    specific and confidently classified."""
    questions: list[ClarifyingQuestion] = []
    words = set(re.findall(r"[a-z0-9]+", mission_text.lower()))

    # 1. Vague mission → ask for scope (highest priority; without scope the
    #    other answers don't help).
    if is_vague(mission_text):
        questions.append(ClarifyingQuestion(
            field="scope",
            question=(
                "Your mission looks broad — can you narrow the scope to a "
                "specific topic, system, or question?"
            ),
            why="mission text is too vague to scaffold a roster + workflow against",
        ))

    # 2. Low classifier confidence → confirm the kind of work.
    confidence = getattr(profile, "classifier_confidence", 0.0) or 0.0
    if confidence < CONFIDENCE_FLOOR:
        questions.append(ClarifyingQuestion(
            field="intent",
            question=(
                "What kind of work is this — research, build, audit, or a "
                "decision?"
            ),
            why=f"classifier confidence {confidence:.2f} is below {CONFIDENCE_FLOOR}",
            options=("research", "build", "audit", "decision"),
        ))

    # 3. Research-shaped mission with no time window → ask about recency.
    if (words & _RESEARCH_HINTS) and not (words & _TIME_HINTS):
        questions.append(ClarifyingQuestion(
            field="time_horizon",
            question="How recent should sources be (e.g. the last 12 months)?",
            why="research mission with no time window detected",
        ))

    # Dedupe by field (defensive) and cap at MAX_QUESTIONS.
    seen: set[str] = set()
    unique: list[ClarifyingQuestion] = []
    for q in questions:
        if q.field in seen:
            continue
        seen.add(q.field)
        unique.append(q)
    return unique[:MAX_QUESTIONS]
