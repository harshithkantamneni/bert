"""Record a silent demo of the bert UI using Playwright.

Boots uvicorn + vite, drives a Chromium instance through the FirstLight
→ Lab Picker → Atlas → Diagnostics → Manuscript walkthrough, captures
a .webm video plus key still frames, prepends a title card + terminal
segment, ffmpeg-stitches everything into a high-quality .mp4.

The output is **silent** — Playwright cannot record audio. You dub
narration.md over the result in QuickTime / iMovie / Resolve.

Usage:
  .venv/bin/python tools/record_demo.py
  .venv/bin/python tools/record_demo.py --quality high   # ~10MB / 2 Mbps
  .venv/bin/python tools/record_demo.py --with-live-cycle  # adds real Groq run
  .venv/bin/python tools/record_demo.py --output ~/Desktop/demo/

Requirements: playwright + chromium (one-time `playwright install chromium`),
ffmpeg, npm + node for vite. The script aborts cleanly if any are missing.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
VENV_PY = LAB_ROOT / ".venv" / "bin" / "python"

UVICORN_PORT = 5174
VITE_PORT = 5173

VIEWPORT = {"width": 1440, "height": 900}

# Per-surface dwell time (seconds the camera lingers). Tune for the
# narration cadence — narration.md sections are ~15-45s each.
DWELL = {
    "FirstLight": 14,
    "LabPicker": 8,
    "Atlas": 16,
    "Diagnostics": 14,    # restored: bug in MemoryTier API field-mapping fixed
    "Manuscript": 18,     # extra dwell since this carries the wow beat
    "ProofPacket": 10,
}

# Quality presets for the final ffmpeg encode
QUALITY_PRESETS = {
    "fast":   {"crf": "22", "preset": "fast",   "bitrate": "auto"},
    "medium": {"crf": "20", "preset": "medium", "bitrate": "auto"},
    "high":   {"crf": "18", "preset": "slow",   "bitrate": "auto"},
}

# Brand palette tokens (must match bert/v4/src/tokens/palette.ts)
PALETTE = {
    "night":      "0x0E0A06",   # base background
    "bone":       "0xE8DDC4",
    "bone3":      "0x9C8B6F",
    "candle":     "0xFFE0A8",
    "candle3":    "0xA88542",
    "imprimatur": "0xF5EAD4",
}


def _print(msg: str, color: str = "33") -> None:
    print(f"\033[{color}m{msg}\033[0m", flush=True)


# ── Process management ─────────────────────────────────────────────

class ProcessGroup:
    """Track background processes; cleanup on exit."""
    def __init__(self) -> None:
        self.procs: list[tuple[str, subprocess.Popen]] = []

    def spawn(self, label: str, cmd: list[str], *, env: dict | None = None,
              cwd: Path | None = None) -> subprocess.Popen:
        _print(f"[spawn] {label}: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            env={**os.environ, **(env or {})},
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.procs.append((label, proc))
        return proc

    def cleanup(self) -> None:
        for label, proc in reversed(self.procs):
            if proc.poll() is None:
                _print(f"[cleanup] terminating {label} (pid {proc.pid})")
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()


def _wait_for_url(url: str, *, timeout_secs: float = 30.0) -> bool:
    """Poll URL until it returns 2xx or times out."""
    deadline = time.monotonic() + timeout_secs
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as r:
                if 200 <= r.status < 300:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# ── Demo scaffold ──────────────────────────────────────────────────

def scaffold_demo_lab(home: Path) -> Path:
    """Use bert init to create a clean demo-pitch lab."""
    _print(f"[scaffold] creating demo-pitch lab in {home}")
    home.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(VENV_PY), str(LAB_ROOT / "tools" / "bert_init.py"),
         "--non-interactive",
         "--archetype", "Product",
         "--name", "demo-pitch",
         "--provider", "Groq",
         "--autonomy", "Collaborator",
         "--seed", "Ship a tiny Markdown note-capture CLI with frontmatter tags",
         "--from-template", "demo_note_cli"],
        capture_output=True, text=True,
        env={**os.environ, "HOME": str(home)},
        cwd=str(LAB_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"bert init failed: {result.stderr[:400]}")
    return home / ".bert" / "labs" / "demo-pitch"


# ── Recording ──────────────────────────────────────────────────────

def record_browser_walkthrough(*, output_dir: Path) -> Path:
    """Drive Chromium through the surfaces, return path to .webm."""
    from playwright.sync_api import sync_playwright

    output_dir.mkdir(parents=True, exist_ok=True)
    stills_dir = output_dir / "stills"
    stills_dir.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(output_dir),
            record_video_size=VIEWPORT,
        )
        page = context.new_page()

        _print(f"[browser] opening http://127.0.0.1:{VITE_PORT}")
        page.goto(f"http://127.0.0.1:{VITE_PORT}", wait_until="domcontentloaded",
                  timeout=20_000)

        # Scene 1: FirstLight (director letter)
        _print(f"[scene] FirstLight ({DWELL['FirstLight']}s)")
        time.sleep(2)
        page.screenshot(path=str(stills_dir / "01_firstlight.png"),
                        full_page=False)
        time.sleep(DWELL["FirstLight"] - 2)

        # Scene 2: Try clicking the Lab Indicator chip (Lab Picker)
        _print(f"[scene] LabPicker ({DWELL['LabPicker']}s)")
        try:
            chip = page.locator('[aria-haspopup="listbox"]').first
            if chip.count() > 0:
                chip.click(timeout=3_000)
                time.sleep(2)
                page.screenshot(path=str(stills_dir / "02_lab_picker.png"))
        except Exception as exc:
            _print(f"  (lab picker not visible: {exc})", "31")
        time.sleep(DWELL["LabPicker"])

        # Scene 3: navigate to /atlas
        _print(f"[scene] Atlas ({DWELL['Atlas']}s)")
        try:
            page.goto(f"http://127.0.0.1:{VITE_PORT}/atlas",
                      wait_until="domcontentloaded", timeout=15_000)
            time.sleep(2)
            page.screenshot(path=str(stills_dir / "03_atlas.png"))
        except Exception as exc:
            _print(f"  (atlas nav failed: {exc})", "31")
        time.sleep(DWELL["Atlas"] - 2)

        # Scene 4: /diagnostics — provider meters + h-phase health
        # Y.1 fix: MemoryTier API field mapping bug resolved, so this
        # surface now renders end-to-end.
        _print(f"[scene] Diagnostics ({DWELL['Diagnostics']}s)")
        try:
            page.goto(f"http://127.0.0.1:{VITE_PORT}/diagnostics",
                      wait_until="domcontentloaded", timeout=15_000)
            try:
                page.wait_for_selector("text=/PROVIDER|MEMORY TIER|MCP REPLAY/i",
                                       timeout=15_000)
            except Exception as e:
                _print(f"  (diagnostics selector timeout: {e})", "31")
            time.sleep(4)
            page.screenshot(path=str(stills_dir / "04_diagnostics.png"))
        except Exception as exc:
            _print(f"  (diagnostics nav failed: {exc})", "31")
        time.sleep(DWELL["Diagnostics"] - 4)

        # Scene 5: /book (Manuscript surface — weekly grade card, findings)
        _print(f"[scene] Manuscript ({DWELL['Manuscript']}s)")
        try:
            page.goto(f"http://127.0.0.1:{VITE_PORT}/book",
                      wait_until="domcontentloaded", timeout=15_000)
            # Wait for the manuscript content (findings, weekly grade) to load
            with contextlib.suppress(Exception):
                page.wait_for_selector("text=/MANUSCRIPT|FINDING|WEEKLY|GRADE/i",
                                       timeout=10_000)
            time.sleep(3)
            page.screenshot(path=str(stills_dir / "05_manuscript.png"))
        except Exception as exc:
            _print(f"  (manuscript nav failed: {exc})", "31")
        time.sleep(DWELL["Manuscript"] - 3)

        # Close
        context.close()
        browser.close()

    # Find the .webm Playwright wrote
    webms = list(output_dir.glob("*.webm"))
    if not webms:
        raise RuntimeError(f"no .webm written to {output_dir}")
    # Rename the most recent to a stable name
    latest = max(webms, key=lambda p: p.stat().st_mtime)
    target = output_dir / "bert_browser.webm"
    latest.rename(target)
    _print(f"[browser] recording saved: {target}")
    return target


def record_terminal_segment(*, output_dir: Path) -> Path | None:
    """Use macOS `script` to record a terminal session running
    bert verify. Returns a .png screenshot of the final state (we
    convert that into a static slide for the video composite).
    """
    _print("[terminal] capturing bert verify output to a typescript")
    typescript = output_dir / "bert_verify.txt"
    cmd = [
        str(VENV_PY), str(LAB_ROOT / "tools" / "bert_verify.py"),
        str(LAB_ROOT / "findings" / "proof_packets" / "cycle-0400.tar.gz"),
        "--no-color",
    ]
    with typescript.open("w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                cwd=str(LAB_ROOT))
    _print(f"[terminal] saved: {typescript} (rc={result.returncode})")
    return typescript


_TITLE_HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@1,500&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
  html,body { margin:0; padding:0; background:#0E0A06; height:100vh;
              display:flex; flex-direction:column; justify-content:center;
              align-items:center; font-family:'Lora',Georgia,serif; }
  .kicker { font-family:'JetBrains Mono',monospace; font-size:14px;
            color:#A88542; letter-spacing:0.18em; text-transform:uppercase;
            margin-bottom:36px; }
  .hero { font-style:italic; font-weight:500; font-size:68px;
          color:#F5EAD4; letter-spacing:-0.014em; line-height:1.05;
          text-align:center; max-width:1100px; margin:0 0 28px; }
  .sub  { font-style:italic; font-size:24px; color:#9C8B6F;
          text-align:center; max-width:900px; line-height:1.5; }
</style></head><body>
  <div class="kicker">bert · for the investor meeting</div>
  <h1 class="hero">Build privately. Prove publicly.</h1>
  <p class="sub">The only autonomous lab that grades itself weekly and pre-registers what would prove it wrong.</p>
</body></html>"""


def _TERMINAL_HTML(verify_text: str) -> str:
    # HTML-escape the body
    escaped = (verify_text.replace("&", "&amp;").replace("<", "&lt;")
               .replace(">", "&gt;"))
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap">
<style>
  html,body {{ margin:0; padding:80px; background:#0E0A06;
               font-family:'JetBrains Mono',ui-monospace,monospace; color:#E8DDC4;
               font-size:18px; line-height:1.5; }}
  .kicker {{ font-size:13px; color:#A88542; letter-spacing:0.18em;
             text-transform:uppercase; margin-bottom:32px; }}
  .prompt {{ color:#FFE0A8; margin-bottom:14px; }}
  pre {{ white-space:pre-wrap; margin:0; color:#C7BFA8; }}
</style></head><body>
  <div class="kicker">terminal · diligence check</div>
  <div class="prompt">$ bert verify cycle-0400.tar.gz</div>
  <pre>{escaped}</pre>
</body></html>"""


def _render_card_to_video(*, html: str, output: Path, duration: float,
                          viewport: dict = VIEWPORT) -> Path:
    """Y.3 — render HTML to a PNG via Playwright, then loop-encode it
    into a duration-second silent video with ffmpeg.

    This sidesteps the missing libfreetype/drawtext in this ffmpeg
    build and gives us CSS-grade typography for the title cards.
    """
    from playwright.sync_api import sync_playwright

    # Write HTML to a temp file
    html_path = output.with_suffix(".html")
    html_path.write_text(html)

    # Screenshot the HTML at the target viewport
    png_path = output.with_suffix(".png")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport=viewport)
        page.goto(f"file://{html_path}", wait_until="domcontentloaded")
        # Give fonts a beat to download from Google Fonts
        time.sleep(1.5)
        page.screenshot(path=str(png_path), full_page=False)
        browser.close()

    # Loop the PNG into a video
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(png_path),
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-r", "25",
        "-vf", f"scale={viewport['width']}:{viewport['height']}",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"card encode failed: {result.stderr[:500]}")

    # Keep the PNG (useful as a still for the deck) but drop the HTML
    html_path.unlink(missing_ok=True)
    return output


def _generate_title_card(*, output: Path, duration: float = 5.0,
                         viewport: dict = VIEWPORT) -> Path:
    _print(f"[card] generating title card → {output.name}")
    return _render_card_to_video(html=_TITLE_HTML, output=output,
                                 duration=duration, viewport=viewport)


def _generate_terminal_card(*, verify_txt: Path, output: Path,
                            duration: float = 10.0,
                            viewport: dict = VIEWPORT) -> Path:
    _print(f"[card] generating terminal card → {output.name}")
    verify_text = verify_txt.read_text().rstrip()
    return _render_card_to_video(html=_TERMINAL_HTML(verify_text),
                                 output=output, duration=duration,
                                 viewport=viewport)


def _concat_videos(*, parts: list[Path], output: Path) -> Path:
    """Y.3 — concat title + terminal + browser cuts into one .mp4."""
    _print(f"[ffmpeg] concatenating {len(parts)} parts → {output.name}")
    list_file = output.parent / ".concat_list.txt"
    list_file.write_text("\n".join(f"file '{p}'" for p in parts))
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    list_file.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"concat ffmpeg failed: {result.stderr[:400]}")
    return output


def stitch_video(*, browser_webm: Path, output_dir: Path,
                 quality: str = "high",
                 verify_txt: Path | None = None) -> Path:
    """Y.2 + Y.3 — build the final .mp4 from title + terminal + browser.

    quality preset controls the encode (fast / medium / high).
    If verify_txt is provided, a terminal segment is inserted between
    the title card and the browser cut.
    """
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["high"])

    # Build the constituent parts
    title_mp4 = output_dir / "_title.mp4"
    _generate_title_card(output=title_mp4)

    parts: list[Path] = [title_mp4]
    if verify_txt and verify_txt.exists():
        terminal_mp4 = output_dir / "_terminal.mp4"
        _generate_terminal_card(verify_txt=verify_txt, output=terminal_mp4)
        parts.append(terminal_mp4)

    # Convert the browser .webm to .mp4 first (concat needs uniform codecs)
    browser_mp4 = output_dir / "_browser.mp4"
    _print(f"[ffmpeg] transcoding browser webm → mp4 ({quality} quality)")
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", str(browser_webm),
        "-c:v", "libx264", "-preset", preset["preset"], "-crf", preset["crf"],
        "-pix_fmt", "yuv420p",
        "-r", "25",
        str(browser_mp4),
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"browser transcode failed: {result.stderr[:400]}")
    parts.append(browser_mp4)

    # Concat into the final
    target_mp4 = output_dir / "bert_demo.mp4"
    _concat_videos(parts=parts, output=target_mp4)

    # Clean up intermediate fragments
    for p in (title_mp4, browser_mp4):
        p.unlink(missing_ok=True)
    if verify_txt:
        (output_dir / "_terminal.mp4").unlink(missing_ok=True)

    _print(f"[ffmpeg] wrote {target_mp4} "
           f"({target_mp4.stat().st_size // 1024} KB)")
    return target_mp4


# ── Main ────────────────────────────────────────────────────────────

def run_live_cycle_segment(*, output_dir: Path) -> Path | None:
    """Y.4 — if GROQ_API_KEY is in env, run bert_demo_cycle.py and
    capture its output as the live-cycle terminal segment. Otherwise
    return None and continue silently."""
    if not os.environ.get("GROQ_API_KEY"):
        _print("[live-cycle] GROQ_API_KEY not set — skipping live segment",
               "31")
        return None
    _print("[live-cycle] running bert_demo_cycle.py (real Groq calls, ~60-120s)")
    out_path = output_dir / "bert_live_cycle.txt"
    with out_path.open("w") as f:
        result = subprocess.run(
            [str(VENV_PY), str(LAB_ROOT / "tools" / "bert_demo_cycle.py"),
             "--scenario", "1", "--cycle", "999"],
            stdout=f, stderr=subprocess.STDOUT,
            timeout=180, cwd=str(LAB_ROOT),
        )
    _print(f"[live-cycle] saved: {out_path} (rc={result.returncode})")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=None,
                    help="Output directory (default: tmp + auto-rename).")
    ap.add_argument("--quality", choices=list(QUALITY_PRESETS.keys()),
                    default="high",
                    help="Encode quality: fast / medium / high (default: high).")
    ap.add_argument("--with-live-cycle", action="store_true",
                    help="Run bert_demo_cycle.py during recording (requires "
                         "GROQ_API_KEY). Captures real model dispatches as an "
                         "additional terminal segment.")
    ap.add_argument("--no-cleanup", action="store_true",
                    help="Keep uvicorn / vite running after capture.")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        _print("[FATAL] ffmpeg not on PATH — install ffmpeg first.", "31")
        return 2
    if shutil.which("npm") is None:
        _print("[FATAL] npm not on PATH — install node/npm first.", "31")
        return 2

    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        output_dir = LAB_ROOT / "findings" / "investor" / "demo_recording" / f"recording_{ts}"

    output_dir.mkdir(parents=True, exist_ok=True)
    _print(f"[output] {output_dir}", "32")

    pg = ProcessGroup()
    demo_home = Path(tempfile.mkdtemp(prefix="bert_record_"))

    def _signal_cleanup(signum, frame):
        _print(f"[signal] caught {signum} — cleaning up", "31")
        pg.cleanup()
        shutil.rmtree(demo_home, ignore_errors=True)
        sys.exit(130)
    signal.signal(signal.SIGINT, _signal_cleanup)
    signal.signal(signal.SIGTERM, _signal_cleanup)

    try:
        # 1. Scaffold demo lab
        scaffold_demo_lab(demo_home)

        # 2. Start uvicorn (BERT_DEMO_MODE=on so dev surfaces hidden)
        pg.spawn(
            "uvicorn",
            [str(VENV_PY), "-m", "uvicorn", "api.main:app",
             "--host", "127.0.0.1", "--port", str(UVICORN_PORT),
             "--log-level", "error"],
            env={"BERT_DEMO_MODE": "on",
                 "BERT_DISABLE_IDLE_COMPUTE": "1",
                 "HOME": str(demo_home)},
            cwd=LAB_ROOT,
        )

        if not _wait_for_url(f"http://127.0.0.1:{UVICORN_PORT}/api/status",
                             timeout_secs=15):
            _print("[FATAL] uvicorn did not come up within 15s", "31")
            return 2
        _print(f"[uvicorn] ready on :{UVICORN_PORT}", "32")

        # 3. Start vite dev (serves UI on :5173, proxies /api to :5174)
        pg.spawn(
            "vite",
            ["npm", "run", "dev", "--", "--host", "127.0.0.1",
             "--port", str(VITE_PORT)],
            cwd=LAB_ROOT / "bert" / "v4",
        )
        if not _wait_for_url(f"http://127.0.0.1:{VITE_PORT}/",
                             timeout_secs=30):
            _print("[FATAL] vite did not come up within 30s", "31")
            return 2
        _print(f"[vite] ready on :{VITE_PORT}", "32")

        # Give vite a moment to settle (sometimes returns 200 before the
        # client bundle is fully ready)
        time.sleep(2)

        # 4. Record terminal segment (bert verify on canonical packet)
        verify_txt = record_terminal_segment(output_dir=output_dir)

        # 4b. Optional Y.4 — run a live cycle and capture its output too
        if args.with_live_cycle:
            run_live_cycle_segment(output_dir=output_dir)

        # 5. Record the browser walkthrough
        webm = record_browser_walkthrough(output_dir=output_dir)

        # 6. Stitch title + terminal + browser into the final mp4
        mp4 = stitch_video(browser_webm=webm, output_dir=output_dir,
                           quality=args.quality, verify_txt=verify_txt)

        _print("", "32")
        _print(f"✓ silent demo recorded → {mp4}", "32")
        _print(f"  ({mp4.stat().st_size // 1024} KB)", "32")
        _print("", "32")
        _print("next steps:", "33")
        _print("  1. open the .mp4 in QuickTime to preview", "33")
        _print("  2. record narration.md aloud in Voice Memos", "33")
        _print("  3. drop both into iMovie, align audio under video", "33")
        _print("  4. export as bert_demo_5min.mp4", "33")
        return 0
    finally:
        if not args.no_cleanup:
            pg.cleanup()
            shutil.rmtree(demo_home, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
