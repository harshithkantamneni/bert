"""bert init — 5-question lab-bootstrap wizard.

Python questionary + rich. Matches the bert warm-dark-paper aesthetic.
Resume-on-interrupt at ~/.bert/init-resume.json so a re-run picks up
where the founder left off. Smart defaults pulled from environment
(git config, $BERT_PROVIDER_KEY, directory basename).

The five questions (locked from May-2026 research):
  1. Lab archetype       — Product / Research / Strategy
  2. Lab name            — human label, default = directory basename
  3. Model provider      — Groq / NVIDIA / OpenRouter / Ollama
  4. Autonomy level      — Assistant / Collaborator / Pilot
  5. First-cycle seed    — what should bert work on first?

After the wizard, the lab is OPENED to FirstLight — NOT auto-run. The
founder presses `r` on camera to start cycle 1 (feels intentional).

Usage:
  bert init                       # interactive
  bert init --resume              # pick up where we left off
  bert init --non-interactive --archetype product --name foo --provider groq
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent.parent

# Resume file lives under ~/.bert for cross-directory continuity.
HOME_BERT = Path(os.path.expanduser("~/.bert"))
RESUME_PATH = HOME_BERT / "init-resume.json"
LABS_DIR = HOME_BERT / "labs"


# ── Constants ─────────────────────────────────────────────────────────

ARCHETYPES = [
    ("Product",  "ships features end-to-end (code + tests + proof packets)"),
    ("Research", "writes briefs, runs studies, produces signed findings"),
    ("Strategy", "decides moves — option sets, tradeoffs, recommendations"),
]

PROVIDERS = [
    ("Groq",       "groq/llama-3.3-70b-versatile",        "free, fastest (~300 tok/s)"),
    ("NVIDIA",     "nvidia/meta/llama-3.3-70b-instruct",  "free tier, evaluation-only — see provider AUP"),
    ("OpenRouter", "openrouter/auto",                     "bring-your-own-key, all major models"),
    ("Ollama",     "ollama/qwen3:8b",                     "local, offline, lower throughput"),
]

AUTONOMY_LEVELS = [
    ("Assistant",    "confirm every step"),
    ("Collaborator", "confirm major moves; small ones run autonomously"),
    ("Pilot",        "run unattended; PI is notified, not asked"),
]

# Lab-name validation: 2-40 chars, alnum + underscore + dash, alpha-first
NAME_RX = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]{1,39}$")


# ── Smart-default helpers ─────────────────────────────────────────────


def _git_user_name() -> str | None:
    try:
        out = subprocess.run(
            ["git", "config", "--get", "user.name"],
            capture_output=True, text=True, timeout=3,
        )
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _detect_provider() -> str | None:
    for env, prov in (
        ("GROQ_API_KEY", "Groq"),
        ("NVIDIA_API_KEY", "NVIDIA"),
        ("OPENROUTER_API_KEY", "OpenRouter"),
    ):
        if os.environ.get(env):
            return prov
    # Ollama detected via socket presence (cheap check)
    try:
        import socket
        s = socket.socket()
        s.settimeout(0.2)
        s.connect(("127.0.0.1", 11434))
        s.close()
        return "Ollama"
    except OSError:
        pass
    return "Groq"  # safe default


def _default_lab_name(cwd: Path) -> str:
    base = cwd.name
    # Title-case (allow underscores & dashes through)
    return base.replace("_", " ").replace("-", " ").strip().title()[:40] or "bert lab"


# ── Resume management ─────────────────────────────────────────────────


def _load_resume() -> dict | None:
    if not RESUME_PATH.exists():
        return None
    try:
        return json.loads(RESUME_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _save_resume(answers: dict) -> None:
    HOME_BERT.mkdir(parents=True, exist_ok=True)
    RESUME_PATH.write_text(json.dumps(answers, indent=2))


def _clear_resume() -> None:
    with contextlib.suppress(FileNotFoundError):
        RESUME_PATH.unlink()


# ── Wizard ────────────────────────────────────────────────────────────


def _ask_questions(
    *,
    defaults: dict | None = None,
    non_interactive: bool = False,
    cli_args: argparse.Namespace | None = None,
) -> dict:
    """Ask the 5 questions. Persists answers progressively to the
    resume file. Returns the full answers dict."""
    defaults = defaults or {}
    answers: dict[str, Any] = dict(defaults)

    if non_interactive:
        if not cli_args:
            raise ValueError("non_interactive requires cli_args")
        answers["archetype"] = cli_args.archetype or defaults.get("archetype") or "Product"
        answers["name"] = cli_args.name or defaults.get("name") or _default_lab_name(Path.cwd())
        answers["provider"] = cli_args.provider or defaults.get("provider") or _detect_provider()
        answers["autonomy"] = cli_args.autonomy or defaults.get("autonomy") or "Collaborator"
        answers["seed"] = cli_args.seed or defaults.get("seed") or "first cycle: tour the lab"
        return answers

    import questionary
    from rich.console import Console

    console = Console()
    console.print()
    console.print("[bold #FFE0A8]bert · bert lab init[/]")
    console.print("[#9C8B6F]five questions. press [bold]ctrl-c[/bold] to pause; "
                  "the wizard remembers where you stopped.[/]")
    console.print()

    # Q1: archetype
    if "archetype" not in answers or answers.get("archetype") is None:
        a = questionary.select(
            "What kind of lab is this?",
            choices=[f"{name}   — {desc}" for name, desc in ARCHETYPES],
            default=f"Product   — {ARCHETYPES[0][1]}",
        ).ask()
        if a is None:
            raise KeyboardInterrupt()
        answers["archetype"] = a.split(" ", 1)[0]
        _save_resume(answers)

    # Q2: lab name
    if not answers.get("name"):
        default_name = _default_lab_name(Path.cwd())
        n = questionary.text(
            "Name this lab.",
            default=default_name,
            validate=lambda v: True if NAME_RX.match(v or "")
                          else "2-40 chars, letters/numbers/spaces/_/-, alpha first",
        ).ask()
        if n is None:
            raise KeyboardInterrupt()
        answers["name"] = n.strip()
        _save_resume(answers)

    # Q3: provider
    if not answers.get("provider"):
        detected = _detect_provider()
        choices = [f"{name}   — {desc}" for name, model, desc in PROVIDERS]
        default = next(
            (c for c in choices if c.startswith(detected)),
            choices[0],
        )
        p = questionary.select(
            "Primary model provider?",
            choices=choices, default=default,
        ).ask()
        if p is None:
            raise KeyboardInterrupt()
        answers["provider"] = p.split(" ", 1)[0]
        _save_resume(answers)

    # Q4: autonomy
    if not answers.get("autonomy"):
        choices = [f"{name}   — {desc}" for name, desc in AUTONOMY_LEVELS]
        au = questionary.select(
            "How autonomous should bert run?",
            choices=choices,
            default=choices[1],  # Collaborator
        ).ask()
        if au is None:
            raise KeyboardInterrupt()
        answers["autonomy"] = au.split(" ", 1)[0]
        _save_resume(answers)

    # Q5: first-cycle seed
    if not answers.get("seed"):
        s = questionary.text(
            "In one sentence, what should bert work on first?",
            default="Ship a tiny CLI for local Markdown note-taking",
            validate=lambda v: True if 10 <= len(v or "") <= 200
                          else "10-200 characters",
        ).ask()
        if s is None:
            raise KeyboardInterrupt()
        answers["seed"] = s.strip()
        _save_resume(answers)

    return answers


def _render_preview(answers: dict) -> str:
    """Render the end-of-wizard summary screen (Tufte-spare table)."""
    from rich.console import Console
    from rich.table import Table

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="#9C8B6F", justify="right")
    table.add_column(style="#E8DDC4")
    table.add_row("archetype", f"{answers['archetype']} Lab")
    table.add_row("name", answers["name"])
    # Resolve provider → model id
    model_id = next(
        (m for name, m, _ in PROVIDERS if name == answers["provider"]),
        answers["provider"],
    )
    table.add_row("provider", f"{answers['provider']} · {model_id}")
    table.add_row("autonomy", answers["autonomy"])
    lab_dir = LABS_DIR / answers["name"].replace(" ", "_").lower()
    table.add_row("memory", f"{lab_dir}/memory.db")
    table.add_row("seed", answers["seed"][:60] + ("…" if len(answers["seed"]) > 60 else ""))

    from io import StringIO
    buf = StringIO()
    Console(file=buf, force_terminal=False, width=80).print(table)
    return buf.getvalue()


TEMPLATES_ROOT = LAB_ROOT / "templates"


def _scaffold_lab(answers: dict, *, from_template: str | None = None,
                  user_provided_seed: bool = False) -> Path:
    """Materialize the lab directory + lab.yaml. Returns the lab dir.

    If `from_template` is given (one of: product / research / strategy /
    demo_note_cli), the archetype template is copied into the lab dir
    before the customized lab.yaml is written.

    If `user_provided_seed` is True, the template's seed_brief.md is
    rewritten so the user's mission leads (and the template's content
    appears as a context appendix). Without this flag, a user passing
    --seed alongside --from-template would have their mission silently
    overwritten by the template — the exact bug W.3 closes.
    """
    import shutil
    lab_dir = LABS_DIR / answers["name"].replace(" ", "_").lower()
    lab_dir.mkdir(parents=True, exist_ok=True)

    if from_template:
        src = TEMPLATES_ROOT / from_template
        if not src.exists():
            raise ValueError(f"unknown template: {from_template}")
        for fname in ("README.md", "EXPECTED_FIRST_CYCLE.md", "seed_brief.md"):
            src_f = src / fname
            if src_f.exists():
                shutil.copy2(src_f, lab_dir / fname)
        # L.1 — skip Python bytecode + OS clutter that may have
        # accumulated in the template from test runs. Without this
        # ignore-list, `bert init` ships __pycache__ into every
        # scaffolded user lab.
        cycles_src = src / "cycles"
        if cycles_src.exists():
            ignore = shutil.ignore_patterns(
                "__pycache__", "*.pyc", "*.pyo", ".DS_Store",
                ".pytest_cache", ".mypy_cache",
            )
            shutil.copytree(cycles_src, lab_dir / "cycles",
                             dirs_exist_ok=True, ignore=ignore)

    # W.3 — user's --seed must reach seed_brief.md, where bert_run reads
    # the mission from. With a template, prepend the user's seed and
    # keep the template's content as context. Without a template, write
    # a minimal seed_brief.md from the user's seed.
    if user_provided_seed:
        template_content = ""
        seed_file = lab_dir / "seed_brief.md"
        if seed_file.exists():
            template_content = seed_file.read_text()
        rewritten = (
            "# Mission\n\n"
            f"{answers['seed']}\n\n"
        )
        if template_content:
            rewritten += (
                "---\n\n"
                "*Template context (preserved for reference; the mission "
                "above takes precedence):*\n\n"
                + template_content
            )
        seed_file.write_text(rewritten)
    elif not from_template:
        # No template + no user seed override: still write a minimal
        # seed_brief.md so bert_run has something to read.
        (lab_dir / "seed_brief.md").write_text(
            f"# Mission\n\n{answers['seed']}\n"
        )

    # N.2 — ensure scaffolded lab has the bert-lab-shape subdirs the
    # API server expects: sor/ for events, state/ for runtime state.
    # Without this, the bert UI can't route to the lab via
    # ?lab=<name> because /api/status would 404 on missing events.jsonl.
    (lab_dir / "sor").mkdir(exist_ok=True)
    events_file = lab_dir / "sor" / "events.jsonl"
    if not events_file.exists():
        events_file.write_text("")
    (lab_dir / "state").mkdir(exist_ok=True)

    # GG-A.1 — FF-A-aware lab.yaml. Pre-GG this wrote a minimal yaml
    # with no focus_areas, so the director fell back to bert-internal
    # routing/memory/discipline/ux at runtime — wrong taxonomy for
    # a customer lab. Now ship the right archetype-aware areas at
    # scaffold time so the director's bounded decision space reflects
    # the lab's actual mission from cycle 1.
    archetype_lc = answers["archetype"].lower()
    default_areas_by_archetype = {
        "research": ["methodology", "evidence", "synthesis",
                       "consequences", "unspecified"],
        "product":  ["architecture", "implementation", "testing",
                       "operations", "unspecified"],
        "strategy": ["options", "tradeoffs", "risk", "timing",
                       "unspecified"],
    }
    focus_areas = answers.get("focus_areas") or \
        default_areas_by_archetype.get(archetype_lc) or \
        list(default_areas_by_archetype["research"])
    yaml_text = (
        f"# bert lab config — generated by `bert init`\n"
        f"lab_schema_version: 1\n"
        f"name: {answers['name']!r}\n"
        f"archetype: {archetype_lc}\n"
        f"role: standard\n"
        f"share_with_supervisor: true\n"
        f"focus_areas:\n"
        + "".join(f"  - {a}\n" for a in focus_areas)
        + f"provider: {answers['provider'].lower()}\n"
        f"autonomy: {answers['autonomy'].lower()}\n"
        f"first_cycle_seed: |\n"
        f"  {answers['seed']}\n"
    )
    if from_template:
        yaml_text += f"template_origin: {from_template}\n"
    (lab_dir / "lab.yaml").write_text(yaml_text)
    (lab_dir / ".bert").mkdir(exist_ok=True)
    (lab_dir / ".bert" / "answers.yml").write_text(yaml_text)
    return lab_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="bert init — 5-question wizard")
    ap.add_argument("--resume", action="store_true",
                    help="resume from ~/.bert/init-resume.json if present")
    ap.add_argument("--non-interactive", action="store_true",
                    help="don't prompt; require all answers via flags")
    ap.add_argument("--archetype", choices=["Product", "Research", "Strategy"])
    ap.add_argument("--name")
    ap.add_argument("--provider", choices=["Groq", "NVIDIA", "OpenRouter", "Ollama"])
    ap.add_argument("--autonomy", choices=["Assistant", "Collaborator", "Pilot"])
    ap.add_argument("--seed")
    ap.add_argument("--from-template",
                    choices=["product", "research", "strategy", "demo_note_cli"],
                    help="seed the lab from a packaged template")
    ap.add_argument("--run-first-cycle", action="store_true",
                    help="After scaffold, immediately invoke tools/bert_run.py "
                         "for one cycle against the new lab. The 'press r' "
                         "affordance shown in the success screen, made real.")
    args = ap.parse_args()

    defaults: dict | None = None
    if args.resume:
        defaults = _load_resume()
        if defaults:
            from rich.console import Console
            Console().print(
                f"[#9C8B6F]resuming from prior wizard run ({len(defaults)} answers)[/]"
            )

    try:
        answers = _ask_questions(
            defaults=defaults,
            non_interactive=args.non_interactive,
            cli_args=args,
        )
    except KeyboardInterrupt:
        from rich.console import Console
        Console().print()
        Console().print("[#A88542]paused. resume with `bert init --resume`.[/]")
        return 130

    # Scaffold (optionally from template)
    # W.3 — pass user_provided_seed through so template's seed_brief.md
    # doesn't silently overwrite the user's mission.
    user_provided = bool(args.seed) and args.seed != "first cycle: tour the lab"
    _scaffold_lab(answers, from_template=args.from_template,
                            user_provided_seed=user_provided)
    _clear_resume()

    # End-of-wizard preview screen
    from rich.console import Console
    console = Console()
    console.print()
    console.print(_render_preview(answers))
    console.print()
    console.print(
        "  [#A88542]▸ press [bold]enter[/bold] to open the lab "
        "(recommended)[/]"
    )
    console.print(
        "  [#9C8B6F]▸ press [bold]r[/bold] to run first cycle now[/]"
    )
    console.print(
        f"  [#9C8B6F]▸ press [bold]q[/bold] to exit; resume with "
        f"`bert open {answers['name'].replace(' ', '_').lower()}`[/]"
    )
    console.print()

    # W.2 — when --run-first-cycle is set (or 'r' chosen interactively),
    # immediately invoke bert_run.py for one cycle against the new lab.
    # Without this branch, the "press r" affordance above was a literal lie.
    if getattr(args, "run_first_cycle", False):
        lab_name = answers["name"].replace(" ", "_").lower()
        lab_path = Path.home() / ".bert" / "labs" / lab_name
        console.print(
            f"  [#A88542]▸ --run-first-cycle: invoking bert_run.py against "
            f"{lab_path}[/]"
        )
        import subprocess
        rc = subprocess.run(
            [sys.executable,
             str(Path(__file__).parent / "bert_run.py"),
             "--lab", str(lab_path),
             "--max-cycles", "1"],
            check=False,
        ).returncode
        console.print(
            f"  [#A88542]▸ bert run exited rc={rc}[/]"
        )
        return rc

    return 0


if __name__ == "__main__":
    sys.exit(main())
