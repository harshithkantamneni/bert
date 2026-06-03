"""Shared enrichment helper for canvas v2 events.

Originally lived inside `tools/enrich_events_jsonl.py` (the batch
backfill tool). Hoisted here so the live-emit hook in
`core/canvas_emit.py` can call the same function without a
tools/ → core/ import inversion. Both consumers share the schema,
prompt, and clip/dedupe behavior — single source of truth.

Public surface:
  ENRICHMENT_SCHEMA       — JSON schema the LLM must satisfy
  enrich_one(event, ...)  — sync. Returns {tags, lineage, provenance}.
                            provenance ∈ {"llm", "heuristic"}.
  heuristic_fallback(ev)  — exposed so callers can short-circuit when
                            they know the LLM call would be wasted.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import decode, log

LOG = log.get_logger("bert.enrichment")
LAB_ROOT = Path(__file__).resolve().parent.parent
SOURCE_TRUNCATE_CHARS = 8000

# Defaults match what the v8 Mistral run actually used (see
# canvas_v2_phase1_plan and the 7fb5f43 commit). The batch tool
# overrides these via CLI flags; live callers can override via kwargs.
DEFAULT_PROVIDER = "mistral"
DEFAULT_MODEL = "mistral-small-latest"

ENRICHMENT_SCHEMA: dict = {
    "type": "object",
    "required": ["tags", "lineage"],
    "additionalProperties": False,
    "properties": {
        "tags": {
            "type": "array",
            "minItems": 1,
            "maxItems": 7,
            "items": {"type": "string", "minLength": 2, "maxLength": 60},
            "description": "1-7 hashtag-style tags. Hyphenated for multi-word.",
        },
        "lineage": {
            "type": "array",
            "maxItems": 12,
            "items": {"type": "string", "minLength": 1, "maxLength": 200},
            "description": (
                "Paths or D-N IDs this work derived from. Empty array if "
                "the event is foundational (doesn't derive from prior work)."
            ),
        },
    },
}

_EXTRACTION_PROMPT = """\
Read the artifact below and extract semantic metadata for the Trail
visualization. Output a JSON object with exactly two fields.

CRITICAL: tag values must come from THIS artifact's content, not from
any patterns described here. If the artifact discusses "cross-family
judging" use "#cross-family-judge"; if it discusses something else,
use tags reflecting that other thing. Tags that don't appear as
concepts in the artifact's actual text are a bug.

Field 1: tags — array of 1 to 7 hashtag-style strings extracted from
the artifact's distinctive concepts. Hyphenated for multi-word.
Specific to THIS artifact, not generic. Aim for 3-5 tags when the
artifact has rich content; emit 1-2 when content is sparse (a one-
line verdict, a short status row). Better to emit fewer accurate
tags than to invent. If the artifact is research on memory
architecture, the tags should be about memory architecture. If it's
a build report, the tags should be about what was built.

Field 2: lineage — array (possibly empty) of up to 10 file paths or
D-N IDs that this artifact explicitly references as having built on
or derived from. Look for hyperlinks, "see X.md", "per D-N", "builds
on the Y analysis." If the artifact is foundational (no explicit
references to prior bert work), lineage MUST be []. Do not invent
references that aren't in the text.

Output ONLY the JSON. No prose, no code fences, no commentary.

Artifact metadata:
  event_class: {event_class}
  source_path: {source_path}
  agent: {agent}
  cycle: {cycle}

----- ARTIFACT CONTENT BEGINS -----
{content}
----- ARTIFACT CONTENT ENDS -----
"""


def _read_source(source_path: str) -> str:
    """Best-effort read of the source artifact. Returns truncated text
    or empty string if unreadable.

    Path conventions:
      foo.md                  → read whole file
      foo.md#D-7              → read whole file then extract `## D-7` block
      foo.jsonl#L42           → SKIP (observability row; structured already)
    """
    file_part, _, anchor = source_path.partition("#")
    if anchor.startswith("L") and anchor[1:].isdigit():
        return ""
    p = LAB_ROOT / file_part
    if not p.exists() or not p.is_file():
        return ""
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if anchor.startswith("D-") or anchor.startswith("d-"):
        m = re.search(rf"^##\s+{anchor}\b.*?(?=^##\s|\Z)", text, re.MULTILINE | re.DOTALL)
        if m:
            text = m.group(0)
    if len(text) > SOURCE_TRUNCATE_CHARS:
        text = text[:SOURCE_TRUNCATE_CHARS] + f"\n\n... [truncated; full content {len(text)} chars]"
    return text


def heuristic_fallback(event: dict[str, Any]) -> dict[str, Any]:
    """When LLM extraction can't fire (observability row, missing
    source file), synthesize tags from structured metadata.

    Emits only tags that carry real semantic content (event_class,
    verdict, agent role, severity). If a record has none of those,
    falls back to a single #unknown-event tag rather than padding
    with #auto-0/1/2 — the meaningless padding pollutes the tag
    namespace (Mind L0's region clustering would treat #auto-0 as
    a real territory). Schema requires minItems: 1, so 1 tag is
    legal.
    """
    tags: list[str] = []
    ec = event.get("event_class")
    if ec:
        tags.append(f"#{ec.replace('_', '-')}")
    if event.get("verdict"):
        tags.append(f"#verdict-{str(event['verdict']).lower()}")
    if event.get("agent"):
        tags.append(f"#role-{event['agent'].lower()}")
    if event.get("severity_grade"):
        tags.append(f"#severity-{event['severity_grade']}")
    if not tags:
        tags.append("#unknown-event")
    return {"tags": tags[:7], "lineage": [], "provenance": "heuristic"}


def enrich_one(
    event: dict[str, Any],
    *,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    max_retries: int = 2,
    timeout_secs: float | None = None,
) -> dict[str, Any] | None:
    """Returns {tags, lineage, provenance} dict or None on hard failure.

    Best-effort: any provider error or schema-correction exhaustion
    yields a heuristic fallback rather than None — the canvas always
    gets *something* renderable. Returns None only when the input event
    is so malformed we can't even synthesize heuristic tags.
    """
    content = _read_source(event.get("source_path", ""))
    if not content:
        embedded = (event.get("content") or "").strip()
        if not embedded:
            return heuristic_fallback(event)
        verdict = event.get("verdict") or ""
        severity = event.get("severity_grade") or ""
        provider_meta = event.get("judge_provider") or ""
        meta_lines: list[str] = []
        if verdict:
            meta_lines.append(f"verdict: {verdict}")
        if severity:
            meta_lines.append(f"severity: {severity}")
        if provider_meta:
            meta_lines.append(f"judge: {provider_meta}")
        content = "\n".join(meta_lines + [embedded])
        if len(content) > SOURCE_TRUNCATE_CHARS:
            content = content[:SOURCE_TRUNCATE_CHARS] + "\n\n... [truncated]"

    prompt = _EXTRACTION_PROMPT.format(
        event_class=event.get("event_class", "?"),
        source_path=event.get("source_path", "?"),
        agent=event.get("agent") or "?",
        cycle=event.get("cycle") or "?",
        content=content,
    )
    try:
        result = decode.call_with_schema(
            provider, [{"role": "user", "content": prompt}],
            schema=ENRICHMENT_SCHEMA,
            model=model,
            schema_name="event_enrichment",
            max_retries=max_retries,
            max_tokens=600,
            temperature=0.4,
        )
    except Exception as e:  # noqa: BLE001 — provider errors must not kill the caller
        LOG.warning("enrich: %s — provider exception: %s", event.get("id"), str(e)[:140])
        return heuristic_fallback(event)

    if result.parsed is None:
        LOG.warning(
            "enrich: %s — decode failed after %d attempts: %s",
            event.get("id"), result.attempts, result.last_error[:140],
        )
        return heuristic_fallback(event)

    raw_tags = result.parsed.get("tags") or []
    raw_lineage = result.parsed.get("lineage") or []
    seen_t: set[str] = set()
    tags: list[str] = []
    for t in raw_tags:
        if t and t not in seen_t:
            seen_t.add(t)
            tags.append(t)
        if len(tags) >= 7:
            break
    seen_l: set[str] = set()
    lineage: list[str] = []
    for l in raw_lineage:
        if l and l not in seen_l:
            seen_l.add(l)
            lineage.append(l)
        if len(lineage) >= 10:
            break
    return {"tags": tags, "lineage": lineage, "provenance": "llm"}
