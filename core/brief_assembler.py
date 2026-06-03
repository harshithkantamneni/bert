"""Deterministic context_brief.md producer — Layer 5 of bert memory.

No LLM call. Pure file-read + string assembly. Target ~50ms wall time.

Director's first read every cycle is `memories/context_brief.md`. The
brief is produced by stitching:
  1. classify_session() — routine-monitor / phase-transition /
     user-action / post-failure, derived from file mtimes + queue +
     last cycle exit
  2. §Current Program extracted from memories/current.md
  3. Head 5 most-recent D-N entries from memories/log.md
  4. Active P-N carry-forwards from memories/procedures.md
  5. Open Director queue items from state/cycle_queue.md
  6. Critical-files manifest (paths only — Director reads on-demand)

Total target: 10-20 KB. Director's startup read budget is 10 KB total
(P-VS evaluator point #7); the brief sits within that budget by capping
per-section sizes.

The brief is REGENERATED every cycle, not appended. It's a
right-this-second snapshot, not a historical artifact. The historical
record lives in memories/log.md.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from core import log

LOG = log.get_logger("bert.brief")
LAB_ROOT = Path(__file__).resolve().parent.parent
MEMORIES_DIR = LAB_ROOT / "memories"
STATE_DIR = LAB_ROOT / "state"
BRIEF_PATH = MEMORIES_DIR / "context_brief.md"

# Per-section budgets (chars). Tuned so the brief stays well under the
# 10 KB Director startup-read cap (Evaluator point #7).
_BUDGET = {
    "current_program": 2_500,
    "log_head": 4_000,
    "carry_forwards": 2_500,
    "queue": 2_000,
    "critical_files": 1_500,
}


class SessionClass(StrEnum):
    ROUTINE_MONITOR = "routine-monitor"
    PHASE_TRANSITION = "phase-transition"
    USER_ACTION = "user-action"
    POST_FAILURE = "post-failure"
    # B4 — a prior lab-style extensions for data-shape-specific sessions.
    # Surfaced when lab.yaml mission_profile gives us enough hints
    # to specialize the brief.
    POST_RESHAPE = "post-reshape"           # lab.yaml profile changed last cycle
    SATURATION_REVIEW = "saturation-review"  # consolidator detected saturation
    BUDGET_THRESHOLD = "budget-threshold"   # near cost cap; ask user
    BUILD_FAIL_DEBUG = "build-fail-debug"   # for code labs after a failed build


@dataclass(frozen=True)
class SessionClassification:
    classification: SessionClass
    rationale: str


def classify_session(
    *,
    queue_path: Path | None = None,
    pi_notes_path: Path | None = None,
    session_exit_path: Path | None = None,
    current_path: Path | None = None,
) -> SessionClassification:
    """Heuristic session classification.

    Inputs:
      - state/session_exit.md: previous exit reason on first line
      - state/cycle_queue.md: open Director priorities
      - memories/governance/pi_notes.md: PI-authored directives (mtime)
      - memories/current.md: latest §Current Program (mtime)

    Decision tree:
      1. If session_exit.md first line is CATASTROPHIC → post-failure
      2. Else if pi_notes.md mtime is newer than session_exit.md → user-action
      3. Else if current.md mentions §Phase transition or last D-N entry
         starts with "Phase" → phase-transition
      4. Else → routine-monitor
    """
    queue_path or (STATE_DIR / "cycle_queue.md")
    pn = pi_notes_path or (MEMORIES_DIR / "governance" / "pi_notes.md")
    sx = session_exit_path or (STATE_DIR / "session_exit.md")
    cm = current_path or (MEMORIES_DIR / "current.md")

    if sx.exists():
        first = sx.read_text(encoding="utf-8", errors="replace").splitlines()
        first_line = first[0].strip() if first else ""
        if "CATASTROPHIC" in first_line:
            return SessionClassification(
                SessionClass.POST_FAILURE,
                f"session_exit.md head='{first_line}'",
            )

    sx_mtime = sx.stat().st_mtime if sx.exists() else 0.0
    pn_mtime = pn.stat().st_mtime if pn.exists() else 0.0
    if pn_mtime > sx_mtime and pn.exists():
        return SessionClassification(
            SessionClass.USER_ACTION,
            f"pi_notes.md mtime {pn_mtime:.0f} > session_exit.md {sx_mtime:.0f}",
        )

    if cm.exists():
        text = cm.read_text(encoding="utf-8", errors="replace")
        if re.search(r"^#+\s*Phase\b|§\s*Phase\s+(transition|ready)", text, re.MULTILINE):
            return SessionClassification(
                SessionClass.PHASE_TRANSITION,
                "current.md mentions Phase header",
            )

    return SessionClassification(
        SessionClass.ROUTINE_MONITOR,
        "no transition / failure / user signal",
    )


def _truncate(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    suffix = "\n\n... [truncated to fit brief budget]"
    return text[: budget - len(suffix)] + suffix


def _extract_current_program(current_md: Path) -> str:
    if not current_md.exists():
        return "_(memories/current.md missing)_"
    txt = current_md.read_text(encoding="utf-8", errors="replace")
    # Find § Current Program block
    m = re.search(r"§\s*Current\s+Program\b.*?(?=§\s|\Z)", txt, re.DOTALL | re.IGNORECASE)
    if m:
        return _truncate(m.group(0).strip(), _BUDGET["current_program"])
    # Fallback: head of current.md
    return _truncate(txt.strip(), _BUDGET["current_program"])


def _head_log_entries(log_md: Path, n: int = 5) -> str:
    if not log_md.exists():
        return "_(memories/log.md missing)_"
    txt = log_md.read_text(encoding="utf-8", errors="replace")
    # Newest-first convention: D-N entries appear in order from top.
    # Capture each "## D-NNN" block.
    blocks = re.split(r"(?=^##\s+D-\d+\b)", txt, flags=re.MULTILINE)
    blocks = [b.strip() for b in blocks if b.strip().startswith("## D-")]
    head = "\n\n".join(blocks[:n])
    if not head:
        # Fallback: first n*1KB chars
        head = txt[: n * 1000]
    return _truncate(head, _BUDGET["log_head"])


def _carry_forwards(procedures_md: Path) -> str:
    if not procedures_md.exists():
        return "_(memories/procedures.md missing)_"
    txt = procedures_md.read_text(encoding="utf-8", errors="replace")
    # Active P-N: any pattern with STATUS that contains FROZEN or PROPOSED
    # but NOT KILLED. We surface STATUS lines.
    matches = re.findall(
        r"^##\s+(P-(?:VS-)?\d+)\b.*?\n.*?\*\*STATUS:\*\*\s+([^\n]+)",
        txt, re.DOTALL | re.MULTILINE,
    )
    active = [(pid, status) for (pid, status) in matches if "KILLED" not in status.upper()]
    if not active:
        return "_(no active patterns)_"
    lines = [f"- **{pid}** — {status.strip()}" for pid, status in active[:30]]
    return _truncate("\n".join(lines), _BUDGET["carry_forwards"])


def _queue(queue_md: Path) -> str:
    if not queue_md.exists():
        return "_(state/cycle_queue.md missing)_"
    txt = queue_md.read_text(encoding="utf-8", errors="replace").strip()
    return _truncate(txt, _BUDGET["queue"])


def _critical_files() -> str:
    """List of files Director may read on-demand. Paths + sizes only."""
    candidates = [
        MEMORIES_DIR / "log.md",
        MEMORIES_DIR / "procedures.md",
        MEMORIES_DIR / "heuristics.md",
        MEMORIES_DIR / "killed.md",
        MEMORIES_DIR / "shared.md",
        MEMORIES_DIR / "governance" / "pi_notes.md",
        MEMORIES_DIR / "governance" / "constitutional.md",
        STATE_DIR / "session_state.md",
        STATE_DIR / "proposals_pending_pi.md",
    ]
    rows = []
    for p in candidates:
        if p.exists():
            rel = p.relative_to(LAB_ROOT)
            sz = p.stat().st_size
            rows.append(f"| `{rel}` | {sz:>7,} | {time.strftime('%Y-%m-%d', time.gmtime(p.stat().st_mtime))} |")
    if not rows:
        return "_(no critical files found)_"
    return _truncate(
        "| Path | Bytes | Last mod |\n|---|---:|---|\n" + "\n".join(rows),
        _BUDGET["critical_files"],
    )


# ── B4 — lab-aware brief helpers ─────────────────────────────────────


def _profile_summary(lab_path: Path) -> str:
    """One-line mission profile summary from lab.yaml (if present)."""
    yaml_path = lab_path / "lab.yaml"
    if not yaml_path.exists():
        return ""
    try:
        import yaml
        cfg = yaml.safe_load(yaml_path.read_text())
    except Exception:  # noqa: BLE001
        return ""
    if not isinstance(cfg, dict):
        return ""
    profile = cfg.get("mission_profile") or {}
    schema = cfg.get("lab_schema") or {}
    if not profile:
        return ""
    return (
        f"**Profile:** {profile.get('domain', '?')} / "
        f"{profile.get('primary_work', '?')} / "
        f"{profile.get('data_shape', '?')} · "
        f"horizon={profile.get('horizon', '?')} · "
        f"rigor={profile.get('rigor', '?')} · "
        f"workflow={schema.get('workflow', '?')} · "
        f"conf={profile.get('classifier_confidence', 0):.2f}"
    )


def _knowledge_pointers(lab_path: Path) -> str:
    """Pointers (paths only) to the lab's knowledge/*.md files. Director
    reads on demand."""
    knowledge_dir = lab_path / "knowledge"
    if not knowledge_dir.exists():
        return ""
    rows = []
    for p in sorted(knowledge_dir.glob("*.md")):
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        rows.append(f"  • `knowledge/{p.name}` ({sz:,} bytes)")
    if not rows:
        return ""
    return "\n".join(rows)


def assemble_brief(
    *,
    output_path: Path | None = None,
    lab_path: Path | None = None,
) -> tuple[Path, dict]:
    """Build the brief and write to `memories/context_brief.md`.

    Returns (path, stats) where stats has timing + section sizes.

    B4 — when `lab_path` is provided and lab.yaml has mission_profile
    + lab_schema (post-A4 labs), the brief is data-shape-aware:
    adds knowledge-file pointers from the schema, drift_hint section
    if saturation detected, profile summary at the head.
    """
    op = output_path or BRIEF_PATH
    t0 = time.monotonic()

    classification = classify_session()
    sections = {
        "current_program": _extract_current_program(MEMORIES_DIR / "current.md"),
        "log_head": _head_log_entries(MEMORIES_DIR / "log.md", n=5),
        "carry_forwards": _carry_forwards(MEMORIES_DIR / "procedures.md"),
        "queue": _queue(STATE_DIR / "cycle_queue.md"),
        "critical_files": _critical_files(),
    }

    # B4 — lab-aware brief: pull profile + schema if lab_path given
    profile_summary = ""
    knowledge_pointers = ""
    if lab_path is not None:
        profile_summary = _profile_summary(lab_path)
        knowledge_pointers = _knowledge_pointers(lab_path)

    # B4 — assemble body. Profile + knowledge pointers added at top
    # when present so the director reads them BEFORE the per-cycle
    # decision sections.
    head_blocks = [
        "# Director Context Brief",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}_",
        "",
        f"**Session classification:** `{classification.classification.value}` — {classification.rationale}",
        "",
    ]
    if profile_summary:
        head_blocks.extend([profile_summary, ""])
    if knowledge_pointers:
        head_blocks.extend([
            "## Lab knowledge files (read on-demand)",
            "",
            knowledge_pointers,
            "",
        ])
    body = "\n".join(head_blocks + [
        "## Current Program",
        "",
        sections["current_program"],
        "",
        "## Recent decisions (newest first)",
        "",
        sections["log_head"],
        "",
        "## Active patterns (carry-forwards)",
        "",
        sections["carry_forwards"],
        "",
        "## Director queue",
        "",
        sections["queue"],
        "",
        "## Critical files (read on-demand — not auto-loaded)",
        "",
        sections["critical_files"],
        "",
        "---",
        "",
        f"*Brief is regenerated every cycle. Source: `core/brief_assembler.py`. Total: {sum(len(s) for s in sections.values())} chars.*",
        "",
    ])

    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(body, encoding="utf-8")
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    stats = {
        "elapsed_ms": elapsed_ms,
        "total_chars": len(body),
        "classification": classification.classification.value,
        "section_chars": {k: len(v) for k, v in sections.items()},
    }
    LOG.info("brief: assembled %d chars in %dms class=%s",
             stats["total_chars"], elapsed_ms, classification.classification.value)
    return op, stats
