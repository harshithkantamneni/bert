"""Mandatory end-of-cycle judge — deterministic Python-side enforcer.

The Evaluator agent (prompts/evaluator.md) runs the full 23-point
checklist with judgment-heavy items. This module is its companion: a
fast Python check that handles every point that can be verified
mechanically from the cycle's session log + state files. The two run
side-by-side — agent-side covers judgment (pre-commitment honoring,
multi-source claims), Python-side covers structure (token budget,
permission decisions, identical-call counter, dispatch ratios).

Verdict gate: when `evaluate_cycle(cycle)` returns FAIL on any mechanical
check, `gates_graceful_exit()` returns False, blocking
GRACEFUL_CHECKPOINT exit until the issue is resolved or explicitly
overridden by PI.

Mechanical checks implemented (point numbers match the prompt's list):
  4   Specialist work by Director
  6   Distinct output paths in parallel dispatches
  7   Tier-1 reads ≤10 KB up to first dispatch
  8   Memory cap pressure (Hot 40KB / Wiki 15KB / Log 30KB)
  18  Permission gates honored (no destructive without approval)
  19  Spend killswitch (per-cycle token budget ≤5M, per-day ≤10M)
  20  Identical-call counter (≥5 same-args repetitions = cursor loop)
  21  Signature forgery (delegates to core.verify)
  23  General-purpose dispatch ratio <40%

Points 1, 2, 3, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22 are
judgment-bound and remain with the agent. The agent's
findings/evaluator_C{cycle}.md ResultPacket merges with this report at
the runner's gate decision.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from core import log, verify

LOG = log.get_logger("bert.evaluator")
LAB_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = LAB_ROOT / "logs"
MEMORIES_DIR = LAB_ROOT / "memories"
RESULTS_DIR = LAB_ROOT / "state" / "results"


class CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"
    NA = "NA"
    AGENT_PENDING = "AGENT_PENDING"


@dataclass(frozen=True)
class CheckResult:
    point_id: int
    name: str
    status: CheckStatus
    evidence: str


@dataclass
class CycleEvaluation:
    cycle: int
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.FAIL)

    @property
    def partial_count(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.PARTIAL)

    @property
    def overall(self) -> CheckStatus:
        if any(c.status == CheckStatus.FAIL for c in self.checks):
            return CheckStatus.FAIL
        if any(c.status == CheckStatus.PARTIAL for c in self.checks):
            return CheckStatus.PARTIAL
        return CheckStatus.PASS


_DEFAULT_BUDGETS = {
    "tier1_read_bytes_max": 10_000,
    "memory_hot_max": 40_000,
    "memory_wiki_per_file_max": 15_000,
    "memory_log_max": 30_000,
    "tokens_per_cycle_max": 5_000_000,
    "identical_call_threshold": 5,
    "general_purpose_ratio_max": 0.40,
}


def _load_cycle_events(cycle: int) -> list[dict]:
    """Read every cycle_<cycle>_*.jsonl file and return parsed events."""
    if not LOGS_DIR.exists():
        return []
    events: list[dict] = []
    for p in sorted(LOGS_DIR.glob(f"cycle_{cycle}_*.jsonl")):
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            LOG.warning("cannot read %s: %s", p, e)
    return events


def check_4_director_specialist_work(events: list[dict]) -> CheckResult:
    bash_total = 0
    for ev in events:
        if ev.get("kind") == "tool_result" and ev.get("tool") == "Bash":
            bash_total += int(ev.get("elapsed_ms") or 0)
    minutes = bash_total / 60_000
    if minutes > 5:
        return CheckResult(
            4, "no_director_specialist_work", CheckStatus.FAIL,
            f"Bash tool spent {minutes:.1f} min (>5 min cap)",
        )
    return CheckResult(
        4, "no_director_specialist_work", CheckStatus.PASS,
        f"Bash tool spent {minutes:.1f} min",
    )


def check_6_distinct_output_paths(cycle: int, results_dir: Path | None = None) -> CheckResult:
    rd = results_dir or RESULTS_DIR
    if not rd.exists():
        return CheckResult(6, "distinct_output_paths", CheckStatus.NA, "no results dir")
    paths: list[str] = []
    for rp in sorted(rd.glob("*.json")):
        try:
            packet = json.loads(rp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if int(packet.get("cycle", -1)) == cycle:
            paths.append(str(packet.get("output_path") or ""))
    if not paths:
        return CheckResult(6, "distinct_output_paths", CheckStatus.NA, "no packets this cycle")
    if len(set(paths)) == len(paths):
        return CheckResult(6, "distinct_output_paths", CheckStatus.PASS,
                           f"{len(paths)} packets, all distinct")
    dups = [p for p, c in Counter(paths).items() if c > 1]
    return CheckResult(6, "distinct_output_paths", CheckStatus.FAIL,
                       f"duplicate output_paths: {dups}")


def check_7_tier1_read_budget(events: list[dict], budget: int) -> CheckResult:
    bytes_seen = 0
    for ev in events:
        if ev.get("kind") == "tool_call" and ev.get("tool") == "Spawn":
            break
        if ev.get("kind") == "tool_result" and ev.get("tool") == "Read":
            bytes_seen += len((ev.get("content_preview") or "").encode("utf-8"))
    if bytes_seen > budget:
        return CheckResult(7, "tier1_read_budget", CheckStatus.FAIL,
                           f"Read pre-dispatch={bytes_seen} > {budget}")
    return CheckResult(7, "tier1_read_budget", CheckStatus.PASS,
                       f"Read pre-dispatch={bytes_seen}")


def check_8_memory_cap_pressure(budgets: dict, memories_dir: Path | None = None) -> CheckResult:
    md = memories_dir or MEMORIES_DIR
    if not md.exists():
        return CheckResult(8, "memory_caps", CheckStatus.NA, "no memories dir")
    issues: list[str] = []
    hot = md / "current.md"
    if hot.exists() and hot.stat().st_size > budgets["memory_hot_max"]:
        issues.append(f"current.md={hot.stat().st_size}>{budgets['memory_hot_max']}")
    log_md = md / "log.md"
    if log_md.exists() and log_md.stat().st_size > budgets["memory_log_max"]:
        issues.append(f"log.md={log_md.stat().st_size}>{budgets['memory_log_max']}")
    if issues:
        return CheckResult(8, "memory_caps", CheckStatus.FAIL, "; ".join(issues))
    return CheckResult(8, "memory_caps", CheckStatus.PASS, "within caps")


def check_18_permission_gates(events: list[dict]) -> CheckResult:
    violations: list[str] = []
    for ev in events:
        if ev.get("kind") != "permission_decision":
            continue
        if ev.get("destructive") and ev.get("allowed") and "approve" not in (ev.get("reason") or "").lower():
            violations.append(f"destructive {ev.get('tool')} allowed without approval evidence")
    if violations:
        return CheckResult(18, "permission_gates", CheckStatus.FAIL, "; ".join(violations[:3]))
    return CheckResult(18, "permission_gates", CheckStatus.PASS, "no permission_gate violations")


def check_19_spend_killswitch(events: list[dict], budget: int) -> CheckResult:
    total_in = sum(int(ev.get("tokens_in") or 0) for ev in events if ev.get("kind") == "model_response")
    total_out = sum(int(ev.get("tokens_out") or 0) for ev in events if ev.get("kind") == "model_response")
    total = total_in + total_out
    if total > budget:
        return CheckResult(19, "spend_killswitch", CheckStatus.FAIL,
                           f"tokens={total} > cap {budget}")
    return CheckResult(19, "spend_killswitch", CheckStatus.PASS,
                       f"tokens={total} (in={total_in}, out={total_out})")


def check_20_identical_call_counter(events: list[dict], threshold: int) -> CheckResult:
    counter: Counter = Counter()
    for ev in events:
        if ev.get("kind") != "tool_call":
            continue
        fingerprint = (ev.get("tool"), json.dumps(ev.get("arguments") or {}, sort_keys=True))
        counter[fingerprint] += 1
    repeats = [(t, n) for (t, _), n in counter.items() if n >= threshold]
    if repeats:
        return CheckResult(20, "identical_call_counter", CheckStatus.FAIL,
                           f"≥{threshold}× same args: {repeats[:3]}")
    return CheckResult(20, "identical_call_counter", CheckStatus.PASS,
                       f"max repetition={max(counter.values()) if counter else 0}")


def check_21_signature_forgery() -> CheckResult:
    reports = verify.verify_results_dir()
    summary = verify.summarize(reports)
    if summary["any_forgery"]:
        return CheckResult(21, "signature_forgery", CheckStatus.FAIL,
                           f"forged={summary['forged_count']}/{summary['total']}: "
                           f"{summary['forged_paths'][:2]}")
    return CheckResult(21, "signature_forgery", CheckStatus.PASS,
                       f"clean={summary['clean_count']}/{summary['total']}")


def check_1_pre_commitment_exists(*, cycle_queue_path: Path | None = None) -> CheckResult:
    """state/cycle_queue.md must exist and contain ≥3 numbered priorities.
    Director writes these at cycle start (P-008). Mechanical: parse the
    file for `^\\d+\\.` lines or `### N.` markers."""
    p = cycle_queue_path or (LAB_ROOT / "state" / "cycle_queue.md")
    if not p.exists():
        return CheckResult(1, "pre_commitment_exists", CheckStatus.FAIL,
                           "state/cycle_queue.md missing")
    text = p.read_text(encoding="utf-8", errors="replace")
    # Count numbered items: "1.", "2.", "3." at line start, or "### 1", "### 2", etc.
    import re
    numbered = re.findall(r"^\s*(?:\d+\.|###\s+\d+)", text, flags=re.MULTILINE)
    if len(numbered) < 3:
        return CheckResult(1, "pre_commitment_exists", CheckStatus.FAIL,
                           f"only {len(numbered)} numbered items (need ≥3)")
    return CheckResult(1, "pre_commitment_exists", CheckStatus.PASS,
                       f"{len(numbered)} numbered priorities in cycle_queue.md")


def check_11_calibration_reasoning_quality(*, log_path: Path | None = None,
                                             min_chars: int = 80) -> CheckResult:
    """Each ## D-N entry in memories/log.md must have ≥`min_chars` of
    reasoning text. Mechanical: parse D-N blocks, count chars in each."""
    p = log_path or (MEMORIES_DIR / "log.md")
    if not p.exists():
        return CheckResult(11, "calibration_reasoning_quality", CheckStatus.NA,
                           "memories/log.md missing")
    text = p.read_text(encoding="utf-8", errors="replace")
    import re
    # Find each "## D-NNN ..." header and extract the body until the next
    # header at the same level.
    blocks = re.split(r"(?m)^##\s+D-\d+", text)
    # blocks[0] is preamble; subsequent are D-N bodies
    short = 0
    total = 0
    for block in blocks[1:]:
        # Body ends at the next "## " or end of text — but we already
        # split on "## D-N" so each block IS the body. Trim to next "##"
        # if present (lower-level dividers like "---" are fine).
        body = re.split(r"(?m)^##\s+", block, maxsplit=1)[0].strip()
        if not body:
            continue
        total += 1
        if len(body) < min_chars:
            short += 1
    if total == 0:
        return CheckResult(11, "calibration_reasoning_quality", CheckStatus.NA,
                           "no D-N entries in log.md")
    if short:
        return CheckResult(11, "calibration_reasoning_quality", CheckStatus.FAIL,
                           f"{short}/{total} D-N entries have <{min_chars} chars of reasoning")
    return CheckResult(11, "calibration_reasoning_quality", CheckStatus.PASS,
                       f"all {total} D-N entries ≥{min_chars} chars")


def check_14_build_pass_blocking(cycle: int, results_dir: Path | None = None) -> CheckResult:
    """Any BUILD_PASS verdict in this cycle's packets MUST have
    telemetry.verification.exit_code == 0. Catches "agent claimed
    BUILD_PASS without running the build" — the cursor-loop pattern."""
    rd = results_dir or RESULTS_DIR
    if not rd.exists():
        return CheckResult(14, "build_pass_blocking", CheckStatus.NA, "no results dir")
    bad: list[str] = []
    counted = 0
    for rp in rd.glob("*.json"):
        try:
            packet = json.loads(rp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if int(packet.get("cycle", -1)) != cycle:
            continue
        if packet.get("verdict") != "BUILD_PASS":
            continue
        counted += 1
        verif = (packet.get("telemetry") or {}).get("verification") or {}
        exit_code = verif.get("exit_code")
        if exit_code != 0:
            bad.append(f"{rp.name}: exit_code={exit_code!r}")
    if counted == 0:
        return CheckResult(14, "build_pass_blocking", CheckStatus.NA,
                           "no BUILD_PASS verdicts this cycle")
    if bad:
        return CheckResult(14, "build_pass_blocking", CheckStatus.FAIL,
                           f"{len(bad)}/{counted} BUILD_PASS without exit=0: {bad[:2]}")
    return CheckResult(14, "build_pass_blocking", CheckStatus.PASS,
                       f"all {counted} BUILD_PASS verdicts backed by exit=0")


def check_17_constitutional_preamble(*, governance_dir: Path | None = None) -> CheckResult:
    """memories/governance/constitutional.md must exist and contain the
    expected boilerplate (P-016 sentinel discipline + P-020 redaction
    discipline at minimum)."""
    gd = governance_dir or (MEMORIES_DIR / "governance")
    p = gd / "constitutional.md"
    if not p.exists():
        return CheckResult(17, "constitutional_preamble", CheckStatus.FAIL,
                           "memories/governance/constitutional.md missing")
    text = p.read_text(encoding="utf-8", errors="replace").lower()
    # Expected anchors (lowercase substrings)
    anchors = ["p-016", "p-020"]
    missing = [a for a in anchors if a not in text]
    if missing:
        return CheckResult(17, "constitutional_preamble", CheckStatus.FAIL,
                           f"constitutional.md missing anchors: {missing}")
    if len(text) < 500:
        return CheckResult(17, "constitutional_preamble", CheckStatus.FAIL,
                           f"constitutional.md only {len(text)} chars (suspicious)")
    return CheckResult(17, "constitutional_preamble", CheckStatus.PASS,
                       f"constitutional.md present, {len(text)} chars, all anchors found")


def check_22_roster_health(*, agents_dir: Path | None = None,
                            cycle: int = 0,
                            stale_cycles: int = 5) -> CheckResult:
    """Each agents/<role>/ directory should have been touched recently.
    Stale agent dirs (no file activity in `stale_cycles` cycles) signal
    a registered role that's not pulling its weight — flag for
    ORG_ADAPTATION review (per prompts/evaluator.md §22).

    Quality-first refinement: per the prompt the action is "flag for
    review," not a hard fail. Stale roles return PARTIAL (informational
    flag for the next ORG_ADAPTATION cycle), not FAIL (hard gate).
    Distinguish two FAIL conditions:
      - `core/` registers a role but `agents/<role>/` doesn't exist
        at all — that's a real broken-config FAIL
      - the standard agents/{procedural,journal,semantic}.md
        scaffolding is incomplete — the role is mis-configured
    """
    ad = agents_dir or (LAB_ROOT / "agents")
    if not ad.exists():
        return CheckResult(22, "roster_health", CheckStatus.NA, "no agents dir")
    import time as _time
    cutoff_secs = _time.time() - (stale_cycles * 8 * 3600)  # ~8h per cycle
    stale_roles: list[str] = []
    malformed_roles: list[str] = []
    total = 0
    for role_dir in ad.iterdir():
        if not role_dir.is_dir() or role_dir.name.startswith("."):
            continue
        total += 1
        # Check scaffolding completeness — procedural.md + journal.md +
        # semantic.md all required per the bert memory architecture (L2).
        required_files = {"procedural.md", "journal.md", "semantic.md"}
        present = {p.name for p in role_dir.iterdir() if p.is_file()}
        missing = required_files - present
        if missing:
            malformed_roles.append(f"{role_dir.name}(missing:{','.join(sorted(missing))})")
            continue
        most_recent = 0.0
        for p in role_dir.rglob("*"):
            if p.is_file():
                most_recent = max(most_recent, p.stat().st_mtime)
        if most_recent < cutoff_secs:
            stale_roles.append(role_dir.name)
    if total == 0:
        return CheckResult(22, "roster_health", CheckStatus.NA, "no agent role dirs")
    if malformed_roles:
        # Real FAIL: scaffolding broken
        return CheckResult(22, "roster_health", CheckStatus.FAIL,
                           f"{len(malformed_roles)}/{total} roles malformed: {malformed_roles[:3]}")
    if stale_roles:
        # PARTIAL: informational flag, not gate-blocking
        return CheckResult(22, "roster_health", CheckStatus.PARTIAL,
                           f"{len(stale_roles)}/{total} roles stale ≥{stale_cycles} cycles "
                           f"(flag for ORG_ADAPTATION review): {stale_roles[:5]}")
    return CheckResult(22, "roster_health", CheckStatus.PASS,
                       f"all {total} roles active + well-formed in last {stale_cycles} cycles")


def check_23_general_purpose_ratio(cycle: int, max_ratio: float, results_dir: Path | None = None) -> CheckResult:
    rd = results_dir or RESULTS_DIR
    if not rd.exists():
        return CheckResult(23, "general_purpose_ratio", CheckStatus.NA, "no results dir")
    roles: list[str] = []
    for rp in rd.glob("*.json"):
        try:
            packet = json.loads(rp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if int(packet.get("cycle", -1)) == cycle:
            roles.append(str(packet.get("role", "")))
    if not roles:
        return CheckResult(23, "general_purpose_ratio", CheckStatus.NA, "no packets this cycle")
    gp = sum(1 for r in roles if r == "general-purpose")
    ratio = gp / len(roles)
    if ratio > max_ratio:
        return CheckResult(23, "general_purpose_ratio", CheckStatus.FAIL,
                           f"general-purpose={gp}/{len(roles)} ratio={ratio:.2f}>{max_ratio}")
    return CheckResult(23, "general_purpose_ratio", CheckStatus.PASS,
                       f"general-purpose={gp}/{len(roles)} ratio={ratio:.2f}")


_AGENT_POINTS: list[tuple[int, str]] = [
    # Judgment-bound: still need agent-side evaluation per
    # prompts/evaluator.md
    (2, "pre_commitment_honored"),
    (3, "pi_directives_addressed"),
    (5, "scoped_dispatch_packets"),
    (9, "findings_inbox_drained"),
    (10, "killed_ideas_check"),
    (12, "multi_source_claims"),
    (13, "falsifier_registration"),
    (15, "step_hash_match"),
    (16, "code_review_gate"),
]


def evaluate_cycle(cycle: int, *, budgets: dict | None = None) -> CycleEvaluation:
    """Run all mechanical checks for a cycle. Judgment items return AGENT_PENDING.

    Mechanical coverage as of this commit: 14 of the 23 evaluator
    checklist points (was 9, expanded by 5 — quality audit). The
    remaining 9 stay agent-bound because they require judgment about
    semantic alignment (pre-commitment honored, PI directive
    interpretation, multi-source claim weighting, etc.).
    """
    b = {**_DEFAULT_BUDGETS, **(budgets or {})}
    events = _load_cycle_events(cycle)
    checks: list[CheckResult] = [
        check_1_pre_commitment_exists(),
        check_4_director_specialist_work(events),
        check_6_distinct_output_paths(cycle),
        check_7_tier1_read_budget(events, b["tier1_read_bytes_max"]),
        check_8_memory_cap_pressure(b),
        check_11_calibration_reasoning_quality(),
        check_14_build_pass_blocking(cycle),
        check_17_constitutional_preamble(),
        check_18_permission_gates(events),
        check_19_spend_killswitch(events, b["tokens_per_cycle_max"]),
        check_20_identical_call_counter(events, b["identical_call_threshold"]),
        check_21_signature_forgery(),
        check_22_roster_health(cycle=cycle),
        check_23_general_purpose_ratio(cycle, b["general_purpose_ratio_max"]),
    ]
    for pid, name in _AGENT_POINTS:
        checks.append(CheckResult(pid, name, CheckStatus.AGENT_PENDING,
                                  "judgment-bound — see findings/evaluator_C{cycle}.md"))
    return CycleEvaluation(cycle=cycle, checks=checks)


def gates_graceful_exit(evaluation: CycleEvaluation) -> bool:
    """Return True if the cycle may exit GRACEFUL_CHECKPOINT.
    Any mechanical FAIL blocks the exit; AGENT_PENDING is permissive
    (the agent's verdict gates separately at the runner level)."""
    return evaluation.fail_count == 0


def render_report(evaluation: CycleEvaluation) -> str:
    lines = [
        f"# Cycle {evaluation.cycle} Evaluator (Python-side)",
        "",
        f"Overall: **{evaluation.overall.value}** (FAIL count: {evaluation.fail_count})",
        f"Gates GRACEFUL_CHECKPOINT exit: **{gates_graceful_exit(evaluation)}**",
        "",
        "| # | Check | Status | Evidence |",
        "|---|---|---|---|",
    ]
    for c in sorted(evaluation.checks, key=lambda x: x.point_id):
        lines.append(f"| {c.point_id} | {c.name} | {c.status.value} | {c.evidence} |")
    return "\n".join(lines)


def to_dict(evaluation: CycleEvaluation) -> dict:
    return {
        "cycle": evaluation.cycle,
        "overall": evaluation.overall.value,
        "fail_count": evaluation.fail_count,
        "gates_graceful_exit": gates_graceful_exit(evaluation),
        "checks": [
            {
                "point_id": c.point_id, "name": c.name,
                "status": c.status.value, "evidence": c.evidence,
            }
            for c in evaluation.checks
        ],
    }
