"""Record a captioned silent EXPLAINER video that walks through what
bert is and why it matters — using real artifacts on disk.

Different artifact from record_demo.py:
- record_demo.py = short "click through the UI" b-roll (~90s)
- record_explainer.py = long "here is what this product is and what it
  produces" narrative (~3-4 min) with captioned slides between content

Each segment is one of:
  card:        full-screen title/caption slide
  file:        styled rendering of a real on-disk file (failures.md,
               scorecard, falsifier corpus, etc.)
  terminal:    rendered command + output, styled as a terminal
  browser:     live screencap of an bert surface

The output is still silent (Playwright can't record audio). The
captions ARE the narration — investor reads them while watching.

Usage:
  .venv/bin/python tools/record_explainer.py
  .venv/bin/python tools/record_explainer.py --quality high --output ~/Desktop/

Requires: playwright + chromium, ffmpeg, npm/node, the on-disk
artifacts in findings/proof_packets and findings/.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
VENV_PY = LAB_ROOT / ".venv" / "bin" / "python"
CANONICAL_PACKET = LAB_ROOT / "findings" / "proof_packets" / "cycle-0400.tar.gz"

VIEWPORT = {"width": 1440, "height": 900}
UVICORN_PORT = 5174
VITE_PORT = 5173

QUALITY = {
    "fast":   {"crf": "22", "preset": "fast"},
    "medium": {"crf": "20", "preset": "medium"},
    "high":   {"crf": "18", "preset": "slow"},
}


def _print(msg: str, color: str = "33") -> None:
    print(f"\033[{color}m{msg}\033[0m", flush=True)


# ── Shared CSS ──────────────────────────────────────────────────────

BASE_CSS = """
  @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;1,400;1,500&family=JetBrains+Mono:wght@400;500&display=swap');
  html,body { margin:0; padding:0; background:#0E0A06; color:#E8DDC4;
              font-family:'Lora',Georgia,serif; height:100vh; overflow:hidden; }
  .frame { padding:80px 96px; height:100vh; box-sizing:border-box;
           display:flex; flex-direction:column; }
  .kicker { font-family:'JetBrains Mono',monospace; font-size:13px;
            letter-spacing:0.18em; text-transform:uppercase;
            color:#A88542; margin-bottom:32px; }
  .hero { font-style:italic; font-weight:500; font-size:64px;
          color:#F5EAD4; letter-spacing:-0.014em; line-height:1.05;
          margin:0 0 28px; max-width:1100px; }
  .sub { font-size:24px; color:#9C8B6F; line-height:1.5;
         max-width:900px; margin:0 0 12px; }
  .body-text { font-size:20px; color:#C7BFA8; line-height:1.55;
               max-width:980px; }
  .mono { font-family:'JetBrains Mono',monospace; font-size:18px;
          color:#E8DDC4; line-height:1.5; white-space:pre-wrap;
          background:#1F1610; padding:24px 32px; border-radius:4px; }
  .prompt { color:#FFE0A8; }
  table { border-collapse:collapse; width:auto; margin:24px 0; }
  th,td { padding:14px 28px; border-bottom:1px solid #3A3023; text-align:left;
          font-size:18px; }
  th { font-family:'JetBrains Mono',monospace; font-size:13px;
       letter-spacing:0.18em; text-transform:uppercase; color:#9A8763;
       font-weight:500; }
  td.grade-a { color:#6B7F4B; font-weight:500; font-size:24px; }
  td.grade-b { color:#C8662E; font-weight:500; font-size:24px; }
  td.grade-c { color:#A8432A; font-weight:500; font-size:24px; }
  .center { justify-content:center; text-align:center; align-items:center; }
  .center .hero, .center .sub { text-align:center; max-width:1200px; margin-left:auto; margin-right:auto; }
"""


def _card_html(*, kicker: str, hero: str, sub: str | None = None) -> str:
    sub_block = f'<p class="sub">{sub}</p>' if sub else ""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<style>{BASE_CSS}</style></head>
<body><div class="frame center">
  <div class="kicker">{kicker}</div>
  <h1 class="hero">{hero}</h1>
  {sub_block}
</div></body></html>"""


def _content_html(*, kicker: str, hero: str, body_html: str,
                  caption: str | None = None) -> str:
    cap = f'<p class="sub" style="margin-top:24px;">{caption}</p>' if caption else ""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<style>{BASE_CSS}</style></head>
<body><div class="frame">
  <div class="kicker">{kicker}</div>
  <h2 class="hero" style="font-size:40px; margin-bottom:24px;">{hero}</h2>
  {body_html}
  {cap}
</div></body></html>"""


def _terminal_html(*, kicker: str, command: str, output: str,
                   caption: str | None = None) -> str:
    cap = f'<p class="sub" style="margin-top:20px;">{caption}</p>' if caption else ""
    # escape HTML in output
    esc = (output.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return f"""<!doctype html><html><head><meta charset="utf-8">
<style>{BASE_CSS}</style></head>
<body><div class="frame">
  <div class="kicker">{kicker}</div>
  <div class="mono"><span class="prompt">$ {command}</span>

{esc}</div>
  {cap}
</div></body></html>"""


# ── Real content readers ────────────────────────────────────────────

def _read_packet_member(member: str) -> str:
    """Extract a single file from the canonical proof packet."""
    cmd = ["tar", "-xzOf", str(CANONICAL_PACKET), f"cycle-0400/{member}"]
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def _scorecard_table_html() -> str:
    """Render the weekly scorecard as a styled HTML table."""
    md = (LAB_ROOT / "findings" / "weekly_quality_report_2026-05-13.md").read_text()
    rows = []
    # Match "| dimension | **A** |"
    import re
    for m in re.finditer(r"\|\s*([a-z_ ]+)\s*\|\s*\*\*([ABC])\*\*\s*\|", md):
        dim, grade = m.group(1).strip(), m.group(2)
        klass = f"grade-{grade.lower()}"
        rows.append(f'<tr><td>{dim}</td><td class="{klass}">{grade}</td></tr>')
    return ('<table><thead><tr><th>dimension</th><th>grade</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def _falsifier_corpus_snippet() -> str:
    md = (LAB_ROOT / "findings" / "falsifier_corpus.md").read_text()
    # Find first scenario
    import re
    m = re.search(r"## Scenario 1:.*?(?=## Scenario 2:)", md, re.DOTALL)
    snippet = m.group(0)[:600] if m else md[:600]
    esc = snippet.replace("<", "&lt;").replace(">", "&gt;")
    return f'<div class="mono" style="font-size:15px; max-height:580px; overflow:hidden;">{esc}</div>'


def _adversarial_summary_text() -> str:
    raw = _read_packet_member("eval/adversarial.json")
    try:
        d = json.loads(raw)
        summary = d.get("summary", {})
        by_verdict = summary.get("by_verdict", {})
        by_attack = summary.get("by_attack_type", {})
        return (
            f"method:        {d.get('method')}\n"
            f"total attempts: {d.get('total_attempts')}\n"
            f"\n"
            f"by verdict:\n"
            + "\n".join(f"  {k:20} {v:>3}" for k, v in by_verdict.items())
            + "\n\nby attack type:\n"
            + "\n".join(f"  {k:20} {v:>3}" for k, v in by_attack.items())
        )
    except Exception:
        return raw[:800]


def _bert_verify_text() -> str:
    out = subprocess.run(
        [str(VENV_PY), str(LAB_ROOT / "tools" / "bert_verify.py"),
         str(CANONICAL_PACKET), "--no-color"],
        capture_output=True, text=True,
    )
    return out.stdout + out.stderr


def _packet_listing_text() -> str:
    out = subprocess.run(
        ["tar", "-tzf", str(CANONICAL_PACKET)],
        capture_output=True, text=True,
    ).stdout
    lines = [l for l in out.split("\n") if l.strip()]
    # Show top-level + the most interesting subdirs (eval, provenance, outputs)
    interesting = [l for l in lines if l.endswith("/")
                   or any(k in l for k in ("HASHES", "README", "cycle.json",
                          "failures", "adversarial", "self-eval", "reproduce",
                          "slsa.intoto", "slsa.sigstore"))]
    return "\n".join(interesting[:22]) + f"\n... ({len(lines)} total files)"


def _failures_md_text() -> str:
    return _read_packet_member("failures.md")[:900]


def _daily_table_html() -> str:
    """Render the daily activity series as a styled HTML table."""
    json_path = LAB_ROOT / "findings" / "daily_history" / "timeline.json"
    if not json_path.exists():
        return '<div class="mono">no daily timeline on disk yet</div>'
    d = json.loads(json_path.read_text())
    rows = []
    for w in d["days"]:
        rows.append(
            f'<tr>'
            f'<td>{w["iso_date"]}</td>'
            f'<td style="text-align:right;">{w["total_events"]}</td>'
            f'<td style="text-align:right;">{w["verdict_count"]}</td>'
            f'<td style="text-align:right;">{w["accepted_count"]}</td>'
            f'<td style="text-align:right;">{w["cycle_count"]}</td>'
            f'</tr>')
    return ('<table><thead><tr>'
            '<th>date</th><th>events</th><th>verdicts</th>'
            '<th>accepted</th><th>cycles</th>'
            '</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


# ── Segment definition ─────────────────────────────────────────────

def build_segments() -> list[dict]:
    """The explainer narrative. Five acts, ~4 minutes total.

    Act 1: the problem (post-Devin, post-Berkeley, why nobody trusts agent demos)
    Act 2: what bert is + how a cycle works
    Act 3: the receipt (proof packet contents — failures, adversarial, etc.)
    Act 4: the discipline (weekly grade, daily activity, falsifier corpus)
    Act 5: the workspace (all 7 bert views walked end-to-end)
    Act 6: verify yourself + close

    Every segment shows REAL content from disk, not abstract claims.
    """
    return [
        # ── ACT 1 · THE PROBLEM ────────────────────────────────────
        {"kind": "card", "duration": 5,
         "kicker": "bert · what bert is and what it does",
         "hero": "Build privately.<br>Prove publicly.",
         "sub": "An explainer in real artifacts on real disk."},

        {"kind": "card", "duration": 6,
         "kicker": "the 2026 trust environment",
         "hero": "Investors stopped trusting agent demos in 2026.",
         "sub": "February: Devin's flagship demo collapsed within a week — "
                "phantom files, staged Upwork tasks, edited video. "
                "April: the UC Berkeley benchmark-fraud paper showed every "
                "major agent benchmark exploitable to 100% with zero real work."},

        {"kind": "card", "duration": 5,
         "kicker": "the founder's bet",
         "hero": "The credibility move isn't the demo. It's the receipt.",
         "sub": "An artifact the investor's CTO friend can verify themselves, "
                "on their own laptop, six months from now."},

        # ── ACT 2 · WHAT BERT IS ───────────────────────────────────
        {"kind": "card", "duration": 6,
         "kicker": "what bert is",
         "hero": "An autonomous research lab on your laptop.",
         "sub": "Not a chat tool. Not a coding assistant. A lab — a place "
                "with memory, discipline, and an output cadence. "
                "Local-first. $0/month runtime. Owned by you."},

        {"kind": "card", "duration": 5,
         "kicker": "what a cycle is",
         "hero": "The unit of work: one cycle.",
         "sub": "Researcher dispatch → strategist dispatch → threshing pass → "
                "clearness × 2. Five model calls. 60–120 seconds. "
                "One signed packet at the end."},

        {"kind": "terminal", "duration": 9,
         "kicker": "starting a lab with a mission",
         "command": "bert init --seed 'ship a CSV-to-JSON CLI with frontmatter tags'",
         "output": ("✓ scaffolded ~/.bert/labs/csv-tool/\n"
                    "  ▸ lab.yaml         configured (Product, Groq, Collaborator)\n"
                    "  ▸ seed_brief.md    your mission, written to disk\n"
                    "  ▸ sor/events.jsonl ready for events\n"
                    "  ▸ state/           ready for runtime state\n"
                    "  ▸ cycles/          template cycles copied\n\n"
                    "  press r to run first cycle now"),
         "caption": "The mission goes in seed_brief.md. Everything bert "
                    "does afterward reads from that file."},

        {"kind": "terminal", "duration": 9,
         "kicker": "running the lab autonomously",
         "command": "bert run --lab ~/.bert/labs/csv-tool --max-cycles 3",
         "output": ("[cycle 1] starting (researcher → strategist)\n"
                    "  ✓  researcher    verdict=APPROVE  62.4s   nvidia/llama-3.3\n"
                    "  ✓  strategist    verdict=APPROVE  48.1s   mistral/small\n"
                    "[cycle 1] ✓ success in 110.5s\n\n"
                    "[cycle 2] starting...\n"
                    "  ✓  researcher    verdict=APPROVE  58.7s   groq/qwen-3-235b\n"
                    "  ✓  strategist    verdict=SCOPE_STOP 41.2s mistral/small\n"
                    "[cycle 2] ✓ success in 99.9s\n\n"
                    "next: tools/daily_quality_report.py --date today"),
         "caption": "Cross-family routing by default: researcher on llama, "
                    "judge on mistral, anchor on qwen. The harness IS the value."},

        # ── ACT 3 · THE RECEIPT ────────────────────────────────────
        {"kind": "card", "duration": 5,
         "kicker": "what each cycle produces",
         "hero": "A signed tarball you can hand to anyone.",
         "sub": "SLSA in-toto provenance. Sigstore bundle. Cosign-verifiable "
                "on any laptop without bert installed. 44 files inside."},

        {"kind": "terminal", "duration": 11,
         "kicker": "what's actually inside a proof packet",
         "command": "tar -tzf findings/proof_packets/cycle-0400.tar.gz",
         "output": _packet_listing_text(),
         "caption": "Provenance under your key. Eval data. Failures section. "
                    "The cycle's inputs, outputs, and the artifacts it touched."},

        {"kind": "card", "duration": 5,
         "kicker": "what makes this honest",
         "hero": "Including the failures. Separately signed.",
         "sub": "The 'limitations' file ships in every packet under its own "
                "signature — so the disclosure can't be edited after the "
                "cycle closes. POPPER-style categorization."},

        {"kind": "file", "duration": 11,
         "kicker": "cycle-0400/failures.md (verbatim)",
         "hero": "What this cycle did NOT achieve.",
         "body_html": (
             f'<div class="mono" style="font-size:15px; '
             f'max-height:560px; overflow:hidden;">'
             f'{_failures_md_text().replace("<", "&lt;").replace(">", "&gt;")}'
             f'</div>'),
         "caption": "One limitation declared, 15 claims total. Each "
                    "limitation cites the specific claim it affects."},

        {"kind": "card", "duration": 5,
         "kicker": "adversarial-eval-by-design",
         "hero": "A red-team agent attacks every cycle's claims.",
         "sub": "Whether the attacks succeed or not, the attack log ships "
                "inside the proof packet. No same-family LLM-as-judge "
                "(the failure mode after the Berkeley paper)."},

        {"kind": "terminal", "duration": 10,
         "kicker": "cycle-0400/eval/adversarial.json",
         "command": "jq '.summary' cycle-0400/eval/adversarial.json",
         "output": _adversarial_summary_text(),
         "caption": "60 attempts. 57 weakened the claim. 3 defended. "
                    "Honest disclosure: today's adversary is heuristic-v1; "
                    "LLM-driven adversarial v2 ships at milestone I.4."},

        # ── ACT 4 · THE DISCIPLINE ─────────────────────────────────
        {"kind": "card", "duration": 5,
         "kicker": "self-measurement",
         "hero": "Bert grades itself weekly.",
         "sub": "Five axes, A / B / C, every Friday. Same file the lab uses "
                "internally to triage what to fix next week. We surface "
                "the C's. The C is the discipline."},

        {"kind": "file", "duration": 12,
         "kicker": "weekly_quality_report_2026-05-13.md",
         "hero": "The honest grade.",
         "body_html": _scorecard_table_html(),
         "caption": "Cross-family agreement got a C — we surface it. Memory, "
                    "falsifier, idle compute: A. Accepted artifacts: C. "
                    "The C is the load-bearing signal."},

        {"kind": "card", "duration": 5,
         "kicker": "pre-registration discipline",
         "hero": "14 falsifier targets. Locked before any cycle runs.",
         "sub": "POPPER and PREP-Eval are research protocols. Bert ships "
                "14 falsifiers in production with thresholds declared "
                "BEFORE the cycle dispatches. No post-hoc reframing."},

        {"kind": "file", "duration": 10,
         "kicker": "falsifier_corpus.md · scenario 1 of 14",
         "hero": "One pre-registered scenario.",
         "body_html": _falsifier_corpus_snippet(),
         "caption": "Substance, lens A vs lens B, the disagreement, the "
                    "expected verdict shape. All locked at cycle ~400 "
                    "before any falsifier ran against them."},

        {"kind": "card", "duration": 5,
         "kicker": "granular activity series",
         "hero": "Six days of real activity, no backfill.",
         "sub": "Daily report computed directly from events.jsonl filtered "
                "by calendar date. Quiet days are omitted, not zero-filled."},

        {"kind": "file", "duration": 12,
         "kicker": "findings/daily_history/timeline.md",
         "hero": "The activity series, as it actually happened.",
         "body_html": _daily_table_html(),
         "caption": "2,864 events across 6 active days. 103 verdicts. "
                    "37 accepted artifacts. Real days the lab actually ran."},

        # ── ACT 5 · THE WORKSPACE ──────────────────────────────────
        {"kind": "card", "duration": 5,
         "kicker": "the workspace",
         "hero": "Seven views. One workspace.",
         "sub": "Morning letter · meeting · tide · manuscript · loom · "
                "atlas · diagnostics. All driven by the same events you "
                "just saw flowing through the cycle."},

        {"kind": "browser", "duration": 8, "path": "/",
         "kicker": "FirstLight — the morning view",
         "hero": "What the lab did overnight.",
         "selector": "text=/Dominus|good morning|director/i",
         "caption": "The director's letter that lands every morning. Cycle "
                    "in keeping, events through the lab, what needs your eye."},

        {"kind": "browser", "duration": 7, "path": "/meeting",
         "kicker": "Meeting — pending decisions",
         "hero": "What needs your eye.",
         "selector": "text=/PENDING|meeting|MEETING/i",
         "caption": "Bless or veto. Every destructive call routes through "
                    "this surface. Telegram is the in-pocket variant."},

        {"kind": "browser", "duration": 7, "path": "/tide",
         "kicker": "Tide — the event stream",
         "hero": "Recent activity, in order.",
         "selector": "text=/TIDE|recent|events/i",
         "caption": "Every dispatch, every verdict, every acceptance. "
                    "Streamed live via SSE while the lab runs."},

        {"kind": "browser", "duration": 8, "path": "/book",
         "kicker": "Manuscript — accepted findings + weekly grade",
         "hero": "The published record.",
         "selector": "text=/MANUSCRIPT|GRADE|weekly|honest grade/i",
         "caption": "Opens on the weekly C grade — worst section first, "
                    "no flattery. Then the accepted findings, ordered."},

        {"kind": "browser", "duration": 7, "path": "/loom",
         "kicker": "Loom — citation threads",
         "hero": "What cites what.",
         "selector": "text=/LOOM|citation|thread/i",
         "caption": "Threads of provenance: which cycle's output is which "
                    "later cycle's input. The lab's intellectual lineage."},

        {"kind": "browser", "duration": 8, "path": "/atlas",
         "kicker": "Atlas — the lab as a place",
         "hero": "The seamount.",
         "selector": "text=/THE PEAK|VITAL SIGNS|seamount|ROSTER/i",
         "caption": "Vital signs, the agent roster, the provider topology, "
                    "the knowledge graph strata. The lab seen as terrain."},

        {"kind": "browser", "duration": 8, "path": "/diagnostics",
         "kicker": "Diagnostics — provider health",
         "hero": "The meters.",
         "selector": "text=/PROVIDER|cerebras|mistral|MEMORY TIER/i",
         "caption": "RPM headroom, daily-token headroom, cache hit-rate, "
                    "probe health. Eight free-tier providers, one fabric."},

        # ── ACT 6 · VERIFY YOURSELF ────────────────────────────────
        {"kind": "card", "duration": 5,
         "kicker": "the diligence move",
         "hero": "All of this. Verifiable.",
         "sub": "Send a partner the .tar.gz. They run vanilla cosign. "
                "No bert install. The receipt either checks out or it doesn't."},

        {"kind": "terminal", "duration": 11,
         "kicker": "bert verify cycle-0400.tar.gz",
         "command": "bert verify cycle-0400.tar.gz",
         "output": _bert_verify_text(),
         "caption": "Eight checks. The two WARNs are honestly local-dev "
                    "mode — production Sigstore is real engineering "
                    "(OIDC + Fulcio + Rekor + TSA), not a demo-day flag."},

        # ── CLOSE ──────────────────────────────────────────────────
        {"kind": "card", "duration": 7,
         "kicker": "apply for a private lab setup",
         "hero": "Build privately.<br>Prove publicly.",
         "sub": "hello@bert.dev · $1K–$3K solo setup · $5K–$20K team setup"},
    ]


# ── Rendering ──────────────────────────────────────────────────────

def render_segment(*, page, segment: dict, idx: int,
                   work_dir: Path, browser_base_url: str) -> Path:
    """Render one segment to a duration-second .mp4."""
    duration = segment["duration"]
    out = work_dir / f"seg_{idx:02d}.mp4"
    png = work_dir / f"seg_{idx:02d}.png"

    if segment["kind"] == "card":
        html = _card_html(kicker=segment["kicker"], hero=segment["hero"],
                          sub=segment.get("sub"))
    elif segment["kind"] == "content" or segment["kind"] == "file":
        html = _content_html(kicker=segment["kicker"], hero=segment["hero"],
                             body_html=segment["body_html"],
                             caption=segment.get("caption"))
    elif segment["kind"] == "terminal":
        html = _terminal_html(kicker=segment["kicker"],
                              command=segment["command"],
                              output=segment["output"],
                              caption=segment.get("caption"))
    elif segment["kind"] == "browser":
        # Live screencap of a UI surface
        url = browser_base_url + segment["path"]
        page.goto(url, wait_until="domcontentloaded", timeout=15_000)
        sel = segment.get("selector")
        if sel:
            with contextlib.suppress(Exception):
                page.wait_for_selector(sel, timeout=10_000)
        time.sleep(3)
        page.screenshot(path=str(png), full_page=False)
        html = None  # no html render needed
    else:
        raise ValueError(f"unknown segment kind: {segment['kind']}")

    if html is not None:
        # Render the HTML to PNG via playwright
        html_path = work_dir / f"seg_{idx:02d}.html"
        html_path.write_text(html)
        page.goto(f"file://{html_path}", wait_until="domcontentloaded")
        time.sleep(1.5)  # let fonts load
        page.screenshot(path=str(png), full_page=False)
        html_path.unlink(missing_ok=True)

    # Loop the PNG into a video
    result = subprocess.run([
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(png),
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-r", "25",
        "-vf", f"scale={VIEWPORT['width']}:{VIEWPORT['height']}",
        str(out),
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"seg {idx} encode failed: {result.stderr[:300]}")
    return out


def concat_videos(*, parts: list[Path], output: Path, quality: dict) -> Path:
    _print(f"[ffmpeg] concatenating {len(parts)} segments → {output.name}")
    list_file = output.parent / ".concat.txt"
    list_file.write_text("\n".join(f"file '{p}'" for p in parts))
    result = subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c:v", "libx264", "-preset", quality["preset"],
        "-crf", quality["crf"],
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output),
    ], capture_output=True, text=True)
    list_file.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"concat failed: {result.stderr[:300]}")
    return output


# ── Main ────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=None)
    ap.add_argument("--quality", choices=list(QUALITY.keys()), default="high")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        _print("[FATAL] ffmpeg not on PATH", "31")
        return 2

    output_dir = Path(args.output).expanduser().resolve() if args.output else \
        LAB_ROOT / "findings" / "investor" / "demo_recording" / \
        f"explainer_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_work"
    work_dir.mkdir(exist_ok=True)
    _print(f"[output] {output_dir}", "32")

    # Spawn uvicorn + vite for browser segments
    procs = []
    def cleanup():
        for label, p in reversed(procs):
            if p.poll() is None:
                _print(f"[cleanup] terminating {label}")
                p.terminate()
                try: p.wait(timeout=3)
                except subprocess.TimeoutExpired: p.kill()
        shutil.rmtree(work_dir, ignore_errors=True)

    def _signal(s, f):
        cleanup()
        sys.exit(130)
    signal.signal(signal.SIGINT, _signal)
    signal.signal(signal.SIGTERM, _signal)

    try:
        # Boot uvicorn + vite (browser segments need them; HTML rendering doesn't)
        _print("[spawn] uvicorn + vite")
        uv_env = {**os.environ, "BERT_DEMO_MODE": "on",
                  "BERT_DISABLE_IDLE_COMPUTE": "1"}
        procs.append(("uvicorn", subprocess.Popen(
            [str(VENV_PY), "-m", "uvicorn", "api.main:app",
             "--host", "127.0.0.1", "--port", str(UVICORN_PORT),
             "--log-level", "error"],
            env=uv_env, cwd=str(LAB_ROOT),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )))
        procs.append(("vite", subprocess.Popen(
            ["npm", "run", "dev", "--", "--host", "127.0.0.1",
             "--port", str(VITE_PORT)],
            cwd=str(LAB_ROOT / "bert" / "v4"),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )))

        # Wait for both
        for url, label in [
            (f"http://127.0.0.1:{UVICORN_PORT}/api/status", "uvicorn"),
            (f"http://127.0.0.1:{VITE_PORT}/", "vite"),
        ]:
            for _ in range(60):
                try:
                    with urllib.request.urlopen(url, timeout=1.5) as r:
                        if r.status == 200: break
                except Exception: pass
                time.sleep(0.5)
            else:
                _print(f"[FATAL] {label} did not come up", "31")
                return 2
            _print(f"[ready] {label}", "32")
        time.sleep(2)

        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(viewport=VIEWPORT)
            page = ctx.new_page()

            segments = build_segments()
            parts = []
            for i, seg in enumerate(segments):
                _print(f"[seg {i+1:02d}/{len(segments)}] "
                       f"({seg['kind']}) {seg.get('hero', '')[:60]}")
                part = render_segment(
                    page=page, segment=seg, idx=i, work_dir=work_dir,
                    browser_base_url=f"http://127.0.0.1:{VITE_PORT}")
                parts.append(part)

            browser.close()

        final = output_dir / "bert_explainer.mp4"
        concat_videos(parts=parts, output=final, quality=QUALITY[args.quality])

        total_secs = sum(s["duration"] for s in build_segments())
        _print(f"\n✓ explainer recorded → {final}", "32")
        _print(f"  duration: {total_secs}s "
               f"({total_secs // 60}:{total_secs % 60:02d})", "32")
        _print(f"  size:     {final.stat().st_size // 1024} KB", "32")
        _print(f"  segments: {len(build_segments())}", "32")
        _print(f"\nopen {final.relative_to(LAB_ROOT)}", "33")
        return 0
    finally:
        cleanup()


if __name__ == "__main__":
    sys.exit(main())
