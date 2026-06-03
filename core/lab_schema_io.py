"""Load or synthesize the LabSchema for a lab.

The wire between `mission_profile` + `schema_synthesizer` (the design
layer) and `bert_run.py` (the cycle runner). Pre-Sprint-1, bert_run.py
hardcoded `researcher → strategist` regardless of mission. Post-
Sprint-1, the roster comes from the synthesizer.

Persistence model:
- First lab_start: classify seed_brief → synthesize LabSchema → write
  lab/lab_schema.json (atomic temp+rename)
- Subsequent cycle runs: read lab/lab_schema.json (skips re-classification,
  saves ~$0.01 + 3s per cycle)
- Corrupt schema file: warn + re-synthesize from seed_brief
- Missing seed_brief: actionable error

Per v1.0 spec section 1.1 of `docs/BERT_V1_IMPLEMENTATION_SPEC_PART1.md`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path

from core import mission_profile, schema_synthesizer

LOG = logging.getLogger("bert.lab_schema_io")
SCHEMA_FILENAME = "lab_schema.json"


class SchemaLoadError(Exception):
    """Raised when neither a persisted schema can be read nor a fresh
    schema can be synthesized from seed_brief."""


def load_or_synthesize(
    lab_path: Path,
    *,
    use_llm_classifier: bool = True,
    force_resynthesize: bool = False,
) -> schema_synthesizer.LabSchema:
    """Read persisted schema if present; else classify + synthesize.

    Args:
      lab_path: the lab's root directory (contains seed_brief.md)
      use_llm_classifier: whether to use the LLM-based classifier
        (default True). False → heuristic only (faster, no LLM cost).
      force_resynthesize: skip the cache, always re-classify. Useful
        when the seed_brief has been edited.

    Returns the LabSchema.

    Raises:
      SchemaLoadError: missing seed_brief or unrecoverable synthesis failure.
    """
    schema_file = lab_path / SCHEMA_FILENAME
    seed_brief_path = lab_path / "seed_brief.md"

    # Step 1: try the persisted cache (unless forced OR seed_brief.md
    # has been modified after the schema was synthesized). The latter
    # is the multi-mission case — when run_mission_suite.sh swaps
    # seed_brief.md between missions, we must re-classify per mission.
    if not force_resynthesize and schema_file.exists():
        seed_newer = (
            seed_brief_path.exists()
            and seed_brief_path.stat().st_mtime > schema_file.stat().st_mtime
        )
        if seed_newer:
            LOG.info(
                "seed_brief.md mtime > lab_schema.json mtime; re-synthesizing"
            )
        else:
            try:
                data = json.loads(schema_file.read_text())
                return _from_dict(data)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                LOG.warning(
                    "lab_schema.json at %s corrupt (%s); re-synthesizing",
                    schema_file, e,
                )
                # fall through to synthesize

    # Step 2: read seed_brief.
    seed_brief_path = lab_path / "seed_brief.md"
    if not seed_brief_path.exists():
        raise SchemaLoadError(
            f"missing seed_brief.md at {seed_brief_path}. "
            f"Create a mission file describing what bert should do."
        )
    seed_brief = seed_brief_path.read_text()

    # Step 3: classify.
    try:
        profile = mission_profile.classify_mission(
            seed_brief, use_llm=use_llm_classifier,
        )
    except Exception as e:  # noqa: BLE001
        LOG.warning(
            "classifier failed (%s: %s); falling back to heuristic profile",
            type(e).__name__, e,
        )
        profile = mission_profile.default_profile(seed_brief)

    # Step 3b: record the classification for feature auto-promotion (Sprint 6 #29).
    # Best-effort — must never break schema synthesis.
    try:
        from core import feature_promoter
        feature_promoter.record_mission_classified(
            profile.to_dict(), seed_excerpt=seed_brief[:240])
    except Exception as e:  # noqa: BLE001
        LOG.debug("mission_classified record skipped (advisory): %s", e)

    # Step 4: synthesize.
    try:
        schema = schema_synthesizer.synthesize(profile)
    except ValueError as e:
        raise SchemaLoadError(
            f"synthesizer found no matching rule for profile: {e}. "
            f"Check core/library/synthesizer_rules.yaml — "
            f"default rule should always match."
        ) from e

    # Step 5: persist (atomic).
    _persist_atomic(schema_file, schema)

    LOG.info(
        "lab_schema synthesized: rule=%s roster_initial=%s workflow=%s",
        schema.rule_id, schema.roster_initial, schema.workflow,
    )
    return schema


def _persist_atomic(
    schema_file: Path,
    schema: schema_synthesizer.LabSchema,
) -> None:
    """Write schema to file atomically via temp + rename."""
    schema_file.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".lab_schema_", suffix=".tmp",
        dir=str(schema_file.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                schema.to_dict(), f, indent=2,
                default=_json_serializer,
                ensure_ascii=False,
            )
        os.replace(tmp_path, schema_file)
    except Exception:
        # Cleanup tmp file on failure
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def _json_serializer(obj):
    """Help json.dump handle tuples (LabSchema uses them) + sets."""
    if isinstance(obj, (tuple, set)):
        return list(obj)
    raise TypeError(
        f"object of type {type(obj).__name__} is not JSON serializable"
    )


def _from_dict(data: dict) -> schema_synthesizer.LabSchema:
    """Convert persisted dict back to LabSchema.

    Tuples were serialized as JSON lists; we re-tuple them so the
    dataclass invariant holds (downstream code may rely on tuple-ness).
    """
    # v1.0 fields are default-safe: an old persisted schema lacks them
    # and loads fine; a new one rehydrates quality_contract to a typed
    # QualityContract.
    from core.quality import QualityContract
    qc_raw = data.get("quality_contract")
    quality_contract = (
        QualityContract.from_dict(qc_raw) if isinstance(qc_raw, dict) else None
    )
    return schema_synthesizer.LabSchema(
        profile_id=data["profile_id"],
        rule_id=data["rule_id"],
        roster_core=tuple(data["roster_core"]),
        roster_initial=tuple(data["roster_initial"]),
        memory_adapters=tuple(data.get("memory_adapters", [])),
        knowledge_files=tuple(data.get("knowledge_files", [])),
        graph_schema=data.get("graph_schema", ""),
        workflow=data.get("workflow", ""),
        output_format=data.get("output_format", ""),
        routing_overrides=dict(data.get("routing_overrides", {})),
        skill_plan=tuple(data.get("skill_plan", ())),
        quality_contract=quality_contract,
        fitness_command=data.get("fitness_command"),
        output_path_pattern=data.get("output_path_pattern", ""),
        estimated_cost_usd=data.get("estimated_cost_usd"),
        estimated_time_minutes=data.get("estimated_time_minutes"),
        classifier_confidence=float(data.get("classifier_confidence", 0.0) or 0.0),
    )
