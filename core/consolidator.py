"""Async memory KM agent — between-cycle maintenance.

The cycle Evaluator (core/evaluator.py + prompts/evaluator.md) runs at
the END of each cycle and gates the GRACEFUL_CHECKPOINT exit. The
consolidator runs AFTER the Evaluator, asynchronously, to do the
heavier maintenance work that doesn't need to block exit:

  • Status promotion in procedures.md / heuristics.md based on
    cycle-count + cross-link evidence (PROPOSED → VALIDATED →
    ACCEPTED → STABILIZED → ARCHIVED).
  • Archival to memories/archive/<date>/ when a tier file exceeds
    its soft cap (Hot 40 KB / Wiki 15 KB per file / Log 30 KB
    rolling).
  • INDEX.md refresh per memories/ subdirectory.
  • Stale-entry flagging: entries untouched ≥N cycles get a
    `**STALE — review needed**` marker.

Triggers — `should_run()` returns True when ANY of:
  • At least N cycles have passed since last run (default 1 — every cycle)
  • At least M hours have elapsed (default 6h)
  • At least K new D-N entries have been appended (default 10)
  • The caller explicitly forces (e.g., phase-end manual trigger)

LLM-bound operation (optional, opt-in via summarize=True):
  • summarize_log_head() — when log.md exceeds soft cap, the oldest
    third is summarized via core.provider.call() (Cerebras
    llama3.1-8b on free tier — fast, cheap, no Anthropic paid).

Last-run state is in `lab/state/consolidator.last_run.json` so
triggers can compare against it.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from core import log

LOG = log.get_logger("bert.consolidator")
LAB_ROOT = Path(__file__).resolve().parent.parent
MEMORIES_DIR = LAB_ROOT / "memories"
ARCHIVE_DIR = MEMORIES_DIR / "archive"
LAST_RUN_PATH = LAB_ROOT / "lab" / "state" / "consolidator.last_run.json"

# Soft caps per layer (bytes). Crossing the cap fires archival.
_DEFAULT_CAPS = {
    "memories/current.md": 40_000,
    "memories/log.md": 30_000,
    "memories/procedures.md": 100_000,
    "memories/heuristics.md": 100_000,
    "memories/killed.md": 100_000,
    "memories/shared.md": 50_000,
}

# Status promotion thresholds (cycles since proposal).
_DEFAULT_PROMOTION_THRESHOLDS = {
    "PROPOSED": 0,        # immediately eligible to be VALIDATED
    "VALIDATED": 3,       # needs 3 cycles to reach ACCEPTED
    "ACCEPTED": 10,       # needs 10 cycles to reach STABILIZED
    "STABILIZED": 50,     # 50 cycles = ARCHIVED candidate
}


@dataclass
class ConsolidatorReport:
    cycle: int
    started_ts: float
    finished_ts: float = 0.0
    promotions: list[str] = field(default_factory=list)
    archived_paths: list[str] = field(default_factory=list)
    indexes_refreshed: list[str] = field(default_factory=list)
    stale_flagged: list[str] = field(default_factory=list)
    summarized_log: bool = False
    skipped: bool = False
    skip_reason: str = ""
    # C5 — list of newly-promoted inline specializations (e.g.
    # 'researcher__literature_hunter') that now have permanent
    # agents/<role>/procedural.md files.
    specialization_promotions: list[str] = field(default_factory=list)
    # Sprint 6/7 — feature-promotion candidate ids surfaced this pass
    # (repeated mission patterns -> feature suggestions for PI review).
    feature_suggestions: list[str] = field(default_factory=list)

    @property
    def elapsed_secs(self) -> float:
        if self.finished_ts == 0:
            return 0.0
        return self.finished_ts - self.started_ts


def _read_last_run() -> dict:
    if not LAST_RUN_PATH.exists():
        return {"last_ts": 0, "last_cycle": -1, "log_dn_count": 0}
    try:
        return json.loads(LAST_RUN_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {"last_ts": 0, "last_cycle": -1, "log_dn_count": 0}


def _write_last_run(data: dict) -> None:
    LAST_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN_PATH.write_text(json.dumps(data, indent=2))


def _count_dn_entries(log_md: Path) -> int:
    if not log_md.exists():
        return 0
    txt = log_md.read_text(encoding="utf-8", errors="replace")
    return len(re.findall(r"^##\s+D-\d+\b", txt, flags=re.MULTILINE))


def should_run(
    *,
    cycle: int,
    min_hours_elapsed: float = 6.0,
    min_new_dn: int = 10,
    force: bool = False,
) -> tuple[bool, str]:
    """Decide whether to run the consolidator now."""
    if force:
        return True, "force"
    last = _read_last_run()
    now_ts = time.time()
    hours_elapsed = (now_ts - last.get("last_ts", 0)) / 3600
    if hours_elapsed >= min_hours_elapsed:
        return True, f"{hours_elapsed:.1f}h since last run"
    new_dn = _count_dn_entries(MEMORIES_DIR / "log.md") - last.get("log_dn_count", 0)
    if new_dn >= min_new_dn:
        return True, f"{new_dn} new D-N entries"
    if cycle - last.get("last_cycle", -1) >= 1:
        # Default: run every cycle. Override with min_hours to throttle.
        if hours_elapsed >= 0.25:  # 15-min throttle
            return True, "new cycle + throttle window passed"
    return False, "trigger thresholds not met"


def promote_statuses(
    *,
    procedures_path: Path | None = None,
    cycle: int = 0,
    thresholds: dict | None = None,
) -> list[str]:
    """Walk a wiki file and promote any entry whose status's threshold has elapsed.

    Status convention in bert: a `**STATUS:**` line followed by either
    `PROPOSED on YYYY-MM-DD`, `FROZEN on YYYY-MM-DD`, or chains thereof.
    For now we only promote PROPOSED → VALIDATED based on cycle count
    in the entry's metadata; full chain is reserved for a richer
    implementation. This first pass: any PROPOSED entry that's been
    referenced in another file gets promoted to VALIDATED.
    """
    pp = procedures_path or (MEMORIES_DIR / "procedures.md")
    if not pp.exists():
        return []
    txt = pp.read_text(encoding="utf-8", errors="replace")
    promotions: list[str] = []

    # Find entries with **STATUS:** PROPOSED
    pattern = re.compile(r"^(##\s+(P-\S+)\b[^\n]*\n.*?\*\*STATUS:\*\*\s+)(PROPOSED)([^\n]*)",
                          re.MULTILINE | re.DOTALL)

    def _maybe_promote(m: re.Match) -> str:
        prefix, pid, _status, rest = m.group(1), m.group(2), m.group(3), m.group(4)
        # Promotion condition (v1 heuristic): has the entry been
        # referenced from at least one other memory file?
        ref_count = _count_references(pid, exclude=pp)
        if ref_count >= 1:
            promotions.append(f"{pid} PROPOSED → VALIDATED ({ref_count} cross-refs)")
            return prefix + "VALIDATED" + rest
        return m.group(0)

    new_txt = pattern.sub(_maybe_promote, txt)
    if new_txt != txt:
        pp.write_text(new_txt, encoding="utf-8")
    return promotions


def _count_references(token: str, *, exclude: Path | None = None) -> int:
    count = 0
    for p in MEMORIES_DIR.rglob("*.md"):
        if exclude is not None and p == exclude:
            continue
        try:
            if token in p.read_text(encoding="utf-8", errors="replace"):
                count += 1
        except OSError:
            continue
    return count


def archive_oversized(
    *, caps: dict | None = None, archive_root: Path | None = None,
) -> list[str]:
    """For each tracked file, if it exceeds its soft cap, copy the
    OLDEST third of entries to memories/archive/<date>/ and trim
    the live file. Returns archived file paths.

    For now, "oldest third" is a simple byte-slice — files maintain
    newest-first or chronological convention so the tail/head matches.
    A future refinement would parse entry boundaries.
    """
    cmap = caps or _DEFAULT_CAPS
    ar = archive_root or ARCHIVE_DIR
    archived: list[str] = []
    today = time.strftime("%Y-%m-%d", time.gmtime())
    target_dir = ar / today
    for rel, cap in cmap.items():
        p = LAB_ROOT / rel
        if not p.exists():
            continue
        size = p.stat().st_size
        if size <= cap:
            continue
        # Move oldest 1/3 of the file to archive
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / p.name
        text = p.read_text(encoding="utf-8", errors="replace")
        cutoff = (size * 2) // 3  # keep newest 2/3
        oldest = text[cutoff:]
        newest = text[:cutoff]
        # Append the older content to the archive file (don't overwrite)
        with target.open("a", encoding="utf-8") as f:
            f.write(f"\n\n# Archived from {rel} on {today}\n\n")
            f.write(oldest)
        p.write_text(newest, encoding="utf-8")
        archived.append(rel)
        LOG.info("consolidator: archived %d bytes of %s → %s", len(oldest), rel, target)
    return archived


def refresh_indexes(*, dirs: list[Path] | None = None) -> list[str]:
    """Refresh INDEX.md in each directory under memories/programs/ etc.
    INDEX.md is just a list of relative paths in the directory."""
    targets = dirs or [d for d in MEMORIES_DIR.iterdir() if d.is_dir()]
    refreshed: list[str] = []
    for d in targets:
        if d.name == "archive":
            continue  # archives don't get an index
        idx = d / "INDEX.md"
        entries = sorted(d.glob("**/*.md"))
        # Don't include the index file in itself
        entries = [e for e in entries if e.name != "INDEX.md"]
        if not entries:
            continue
        lines = [f"# {d.name}/ index", "",
                 f"_Auto-refreshed by core.consolidator on {time.strftime('%Y-%m-%d', time.gmtime())}_",
                 ""]
        for e in entries:
            rel = e.relative_to(d)
            sz = e.stat().st_size
            lines.append(f"- [{rel}]({rel}) — {sz:,} bytes")
        idx.write_text("\n".join(lines) + "\n", encoding="utf-8")
        refreshed.append(str(idx.relative_to(LAB_ROOT)))
    return refreshed


def flag_stale(
    *, max_age_days: int = 30,
    target_paths: list[Path] | None = None,
) -> list[str]:
    """Mark entries whose last-modified time is older than `max_age_days`
    with a `**STALE — review needed**` annotation at the top.
    Idempotent — re-running on an already-flagged file is a no-op.
    """
    flagged: list[str] = []
    cutoff = time.time() - max_age_days * 86400
    targets = target_paths or [
        MEMORIES_DIR / "procedures.md",
        MEMORIES_DIR / "heuristics.md",
        MEMORIES_DIR / "shared.md",
    ]
    for p in targets:
        if not p.exists():
            continue
        if p.stat().st_mtime >= cutoff:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if "**STALE — review needed**" in text:
            continue
        new = f"**STALE — review needed** _(consolidator: untouched ≥{max_age_days}d)_\n\n{text}"
        p.write_text(new, encoding="utf-8")
        flagged.append(str(p.relative_to(LAB_ROOT)))
        LOG.info("consolidator: flagged stale: %s", p)
    return flagged


def summarize_log_head(
    *, log_path: Path | None = None,
    target_provider: str = "cerebras",
    target_model: str = "llama3.1-8b",
    cap_bytes: int = 30_000,
) -> bool:
    """When log.md > cap_bytes, send the oldest third to a free-tier
    LLM for summarization. Replaces that section in-place with the
    summary. Returns True if summarization fired."""
    p = log_path or (MEMORIES_DIR / "log.md")
    if not p.exists():
        return False
    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) <= cap_bytes:
        return False
    # Newest-first convention: oldest content is at the END.
    cutoff = (len(text) * 2) // 3  # keep newest 2/3 verbatim
    newest = text[:cutoff]
    oldest = text[cutoff:]
    if not oldest.strip():
        return False
    try:
        from core import provider as _provider
        msgs = [
            {"role": "system", "content": (
                "You are a memory-consolidation summarizer for the bert-lab "
                "autonomous research lab. Your job is to summarize a section "
                "of decision log entries (D-N format) into a single concise "
                "summary preserving: (a) the decision IDs that were made, "
                "(b) the major themes / programs they advanced, (c) any "
                "still-open caveats. Keep entry IDs verbatim; collapse the "
                "narrative. Output a single markdown block under '## Summary "
                "of archived entries'.")},
            {"role": "user", "content": oldest},
        ]
        resp = _provider.call(target_provider, msgs, model=target_model, max_tokens=1500)
        if resp.finish_reason == "error":
            LOG.warning("consolidator: summarization failed: %s", resp.text[:200])
            return False
        summary = resp.text.strip()
    except Exception as e:  # noqa: BLE001
        LOG.exception("consolidator: summarization crashed: %s", e)
        return False
    new_text = newest + "\n\n---\n\n" + summary + "\n"
    p.write_text(new_text, encoding="utf-8")
    LOG.info("consolidator: summarized %d bytes of log.md head", len(oldest))
    return True


def _promote_specializations(
    *, cycle: int, labs_dir: Path | None = None,
) -> list[str]:
    """C5 — Promote inline specializations the director has spawned ≥3
    times to permanent roles.

    For each candidate, write `agents/<template>__<inline>/procedural.md`
    with the template body + inline header + promotion-stamp footer.
    Then call roster.mark_promoted() so the candidate doesn't re-fire
    on next consolidate run.

    Discovers labs by scanning labs_dir (default ~/.bert/labs/) for
    `agents/_spawn_tracker.json`. `labs_dir` override is mainly for
    tests. Returns list of newly-promoted role names (across all labs).
    """
    promoted: list[str] = []
    try:
        from core import roster
    except ImportError as e:
        LOG.warning("roster import failed: %s", e)
        return promoted

    labs_dir = labs_dir or (Path.home() / ".bert" / "labs")
    if not labs_dir.exists():
        return promoted

    for lab in labs_dir.iterdir():
        if not lab.is_dir():
            continue
        try:
            cands = roster.candidates_for_promotion(lab, threshold=3)
        except Exception as e:  # noqa: BLE001
            LOG.warning("consolidator: roster scan failed for %s: %s",
                         lab.name, e)
            continue
        for cand in cands:
            role_name = f"{cand.template}__{cand.inline_name}"
            role_dir = lab / "agents" / role_name
            role_dir.mkdir(parents=True, exist_ok=True)
            proc_path = role_dir / "procedural.md"
            if proc_path.exists():
                # Already promoted; just mark in tracker
                roster.mark_promoted(lab, cand.template, cand.inline_name)
                continue
            # Render the procedural by re-using roster.spawn_inline's
            # template+inline-header composition
            result = roster.spawn_inline(
                lab_path=lab, template=cand.template,
                inline_name=cand.inline_name, cycle=cycle,
            )
            if not result.get("ok"):
                continue
            body = result["procedural"]
            footer = (
                f"\n\n---\n\n## C5 promotion footer\n"
                f"*This role was promoted from inline specialization to "
                f"permanent at cycle C{cycle} after {cand.use_count} uses "
                f"(first seen C{cand.first_seen_cycle}, last seen "
                f"C{cand.last_seen_cycle}).*\n"
            )
            proc_path.write_text(body + footer)
            # Create the standard sub-directories
            (role_dir / "episodic").mkdir(parents=True, exist_ok=True)
            (role_dir / "semantic.md").touch()
            roster.mark_promoted(lab, cand.template, cand.inline_name)
            promoted.append(f"{lab.name}:{role_name}")
            LOG.info("consolidator: promoted %s in lab=%s after %d uses",
                     role_name, lab.name, cand.use_count)
    return promoted


def _run_feature_promotion() -> list[str]:
    """Best-effort feature auto-promotion (Sprint 6 #29) on the consolidator's
    periodic cadence. Surfaces repeated mission patterns as PI-review feature
    suggestions. Never raises — advisory maintenance must not break a cycle."""
    try:
        from core import feature_promoter
        return feature_promoter.run()
    except Exception as e:  # noqa: BLE001
        LOG.warning("consolidator: feature_promotion failed: %s", e)
        return []


def consolidate(
    *, cycle: int, force: bool = False, summarize: bool = False,
) -> ConsolidatorReport:
    """One full pass. Caller invokes after the Evaluator landing
    (post-GRACEFUL_CHECKPOINT). Set summarize=True to enable the
    LLM-bound log summarization."""
    rep = ConsolidatorReport(cycle=cycle, started_ts=time.time())
    ok, reason = should_run(cycle=cycle, force=force)
    if not ok:
        rep.skipped = True
        rep.skip_reason = reason
        rep.finished_ts = time.time()
        return rep

    rep.promotions = promote_statuses(cycle=cycle)
    rep.archived_paths = archive_oversized()
    rep.indexes_refreshed = refresh_indexes()
    rep.stale_flagged = flag_stale()
    if summarize:
        rep.summarized_log = summarize_log_head()

    # C5 — promote organic-spawn specializations to permanent roles.
    # When the director has spawned `researcher__literature_hunter`
    # 3+ times, the consolidator writes agents/<role>/procedural.md
    # (template content + inline header) so future cycles can dispatch
    # directly to the role without re-instantiating.
    rep.specialization_promotions = _promote_specializations(cycle=cycle)

    # Sprint 6/7 — periodic organic growth: surface repeated mission patterns as
    # feature suggestions. Runs on the consolidator's existing periodic cadence
    # (only when should_run passed). Best-effort; never breaks consolidation.
    rep.feature_suggestions = _run_feature_promotion()

    # Quality-first scaling: rotate observability JSONLs above
    # threshold + prune old quota events. These were previously
    # built but unwired — the consolidator's periodic-maintenance
    # role is the right home for them.
    rotated = 0
    pruned = 0
    try:
        from core import observability as _obs
        rot_results = _obs.rotate_all()
        rotated = sum(1 for v in rot_results.values() if v)
    except Exception as e:  # noqa: BLE001
        LOG.warning("consolidator: observability rotate failed: %s", e)
    try:
        from core import quota as _quota
        pruned = _quota.prune_old(days=30)
    except Exception as e:  # noqa: BLE001
        LOG.warning("consolidator: quota.prune_old failed: %s", e)

    rep.finished_ts = time.time()
    _write_last_run({
        "last_ts": rep.finished_ts,
        "last_cycle": cycle,
        "log_dn_count": _count_dn_entries(MEMORIES_DIR / "log.md"),
        "promotions": len(rep.promotions),
        "archived": len(rep.archived_paths),
        "stale_flagged": len(rep.stale_flagged),
        "obs_rotated": rotated,
        "quota_pruned": pruned,
    })
    LOG.info(
        "consolidator: cycle=%d promotions=%d archived=%d indexes=%d stale=%d "
        "rotated=%d pruned=%d elapsed=%.1fs",
        cycle, len(rep.promotions), len(rep.archived_paths),
        len(rep.indexes_refreshed), len(rep.stale_flagged),
        rotated, pruned, rep.elapsed_secs,
    )
    return rep
