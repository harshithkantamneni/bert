"""Sprint 3 c7: mission elicitation (recheck Q-1).

When the mission classifier is uncertain (confidence < 0.7) or the
mission text is vague, bert asks 0-3 clarifying questions BEFORE
spending tokens, rather than running on a guess.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import mission_elicitation, mission_profile  # noqa: E402


def _profile(text: str, confidence: float):
    p = mission_profile.default_profile(text)
    return dataclasses.replace(p, classifier_confidence=confidence)


def test_confident_and_specific_mission_needs_no_questions():
    text = "Survey vector database papers from 2026 comparing recall and latency"
    qs = mission_elicitation.elicit(text, _profile(text, 0.85))
    assert qs == []


def test_low_confidence_triggers_intent_question():
    text = "Survey vector database papers from 2026 comparing recall and latency"
    qs = mission_elicitation.elicit(text, _profile(text, 0.4))
    # specifically the 'intent' clarifier fires (text is specific, not vague,
    # so scope shouldn't fire; it has a time window, so time_horizon shouldn't)
    fields = {q.field for q in qs}
    assert "intent" in fields
    assert "scope" not in fields  # mission is specific
    intent_q = next(q for q in qs if q.field == "intent")
    assert intent_q.options == ("research", "build", "audit", "decision")


def test_vague_mission_asks_for_scope():
    text = "research stuff"
    qs = mission_elicitation.elicit(text, _profile(text, 0.4))
    fields = {q.field for q in qs}
    assert "scope" in fields, f"vague mission should fire a scope question; got {fields}"
    scope_q = next(q for q in qs if q.field == "scope")
    joined = scope_q.question.lower()
    assert "scope" in joined or "specific" in joined or "narrow" in joined


def test_never_more_than_three_questions():
    text = "do things"  # maximally vague + low confidence
    qs = mission_elicitation.elicit(text, _profile(text, 0.1))
    assert len(qs) <= 3


def test_each_question_has_field_text_and_why():
    text = "look into the thing"
    qs = mission_elicitation.elicit(text, _profile(text, 0.3))
    assert qs, "expected questions for a vague low-confidence mission"
    for q in qs:
        assert q.field
        assert q.question.endswith("?")
        assert q.why


def test_questions_are_unique_by_field():
    text = "stuff"
    qs = mission_elicitation.elicit(text, _profile(text, 0.2))
    fields = [q.field for q in qs]
    assert len(fields) == len(set(fields)), "no duplicate clarifying fields"


def test_is_vague_detector():
    assert mission_elicitation.is_vague("research stuff") is True
    assert mission_elicitation.is_vague("do things") is True
    assert mission_elicitation.is_vague(
        "Survey vector database papers from 2026 comparing recall and latency"
    ) is False
