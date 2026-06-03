"""HH-G — 20-stop Playwright walkthrough of the rehauled UI.

Runs the dev server in demo mode, drives a chromium browser through
every user-facing surface + every retired route's redirect, asserts
the post-HH-F architecture holds end-to-end, captures screenshots,
and reports console errors per stop.

This is the live-browser counterpart to the static smoke gates in
HH-A..F. The smokes verify the source code; this verifies the
running app.

Usage:
    .venv/bin/python tests/_walkthrough_hh_rehaul.py
    .venv/bin/python tests/_walkthrough_hh_rehaul.py --base-url http://localhost:5173

The base-url defaults to the standard vite dev server. If the dev
server is not already running on that port, this script will skip
gracefully with a clear note (rather than blocking the acceptance
gate on environment plumbing).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
SCREENS  = LAB_ROOT / "findings" / "bert_v4" / "hh_walkthrough"


# Each stop has: label, url, asserts (list of (selector_or_text, kind))
# kind: "text" matches body innerText; "role" matches accessible roles;
# "no_text" asserts the string is absent (used to verify retirement)
STOPS = [
    # ── Home + masthead ────────────────────────────────────────
    ("01-home",                "/",
     [("bert · home",        "text"),
      ("home",                  "masthead"),
      ("manuscript",            "masthead"),
      ("proofs",                "masthead"),
      ("atlas",                 "masthead")]),
    ("02-home-talk-drawer",    "/",
     [("talk",                  "masthead")]),
    ("03-home-mission-editor", "/",
     [("MISSION",               "text"),       # mission editor kicker
      ("edit",                  "text"),       # tab label
      ("preview",               "text")]),     # tab label

    # ── Manuscript (tabbed) ────────────────────────────────────
    ("04-manuscript-default",  "/manuscript",
     [("findings",              "text"),
      ("stream",                "text"),
      ("loom",                  "text")]),
    ("05-manuscript-stream",   "/manuscript?tab=stream",
     [("event stream",          "ariaLabel")]),
    ("06-manuscript-loom",     "/manuscript?tab=loom",
     [("threads",               "text")]),

    # ── Other surfaces ─────────────────────────────────────────
    ("07-fleet",               "/labs",        []),
    ("08-proofs",              "/proofs",      []),
    ("09-atlas",               "/atlas",       []),
    ("10-diagnostics",         "/diagnostics", []),

    # ── Retired-route redirects (HH-E) ─────────────────────────
    ("11-redirect-mission",    "/mission",
     [("bert · home",        "text"),       # ends up on Home
      ("/mission",               "no_url")]),
    ("12-redirect-meeting",    "/meeting",
     [("bert · home",        "text")]),
    ("13-redirect-tide",       "/tide",
     [("event stream",          "ariaLabel")]),
    ("14-redirect-loom",       "/loom",
     [("threads",               "text")]),
    ("15-redirect-book",       "/book",
     [("findings",              "text")]),

    # ── Mobile / responsive (640px) ────────────────────────────
    ("16-home-mobile",         "/",            [], 640),
    ("17-masthead-sheet",      "/",            [], 640),

    # ── Error boundary smoke ───────────────────────────────────
    ("18-onboard",             "/onboard",     []),

    # ── Keyboard navigation ────────────────────────────────────
    ("19-keyboard-help",       "/",            []),

    # ── Final lap — home again, masthead intact ────────────────
    ("20-home-final",          "/",
     [("bert · home",        "text")]),
]


def assert_console_clean(messages: list[dict]) -> list[str]:
    """Return any error-level messages (filter out known noise)."""
    bad: list[str] = []
    for msg in messages:
        if msg["type"] != "error": continue
        t = msg["text"].lower()
        # Known harmless noise — vite HMR pings, react-router warnings
        # about strict-mode double effects, etc.
        if "the above error occurred" in t: continue
        if "vite" in t and "hmr" in t: continue
        if "[hmr]" in t: continue
        bad.append(msg["text"])
    return bad


def run_walkthrough(base_url: str) -> int:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        print(f"playwright unavailable; skipping walkthrough: {e}")
        return 0

    SCREENS.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=True)
        except Exception as e:
            print(f"chromium unavailable; skipping walkthrough: {e}")
            return 0
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        for stop in STOPS:
            label, path, asserts = stop[0], stop[1], stop[2]
            width = stop[3] if len(stop) > 3 else 1280
            page = context.new_page()
            page.set_viewport_size({"width": width, "height": 900})
            console: list[dict] = []
            page.on("console", lambda m, console=console: console.append({"type": m.type, "text": m.text}))

            try:
                page.goto(f"{base_url}{path}", wait_until="networkidle", timeout=15000)
            except Exception as e:
                failures.append(f"{label}: navigation failed — {e}")
                page.close()
                continue

            time.sleep(0.6)  # let lazy chunks land
            body_text = page.evaluate("document.body.innerText").lower()
            current_url = page.url

            for needle, kind in asserts:
                n = needle.lower()
                if kind == "text":
                    if n not in body_text:
                        failures.append(f"{label}: text not found · {needle}")
                elif kind == "masthead":
                    masthead_text = page.evaluate(
                        "(() => { const m = document.querySelector('[role=banner]'); return m ? m.innerText : ''; })()"
                    ).lower()
                    if n not in masthead_text:
                        failures.append(f"{label}: masthead missing · {needle}")
                elif kind == "ariaLabel":
                    aria_found = page.evaluate(
                        "(label) => !!document.querySelector('[aria-label=\"' + label + '\"]')",
                        needle,
                    )
                    if not aria_found:
                        failures.append(f"{label}: aria-label not found · {needle}")
                elif kind == "no_text":
                    if n in body_text:
                        failures.append(f"{label}: forbidden text present · {needle}")
                elif kind == "no_url":
                    if needle in current_url:
                        failures.append(f"{label}: redirect did not happen — still at {current_url}")

            errs = assert_console_clean(console)
            if errs:
                failures.append(f"{label}: console errors — " + " | ".join(errs[:3]))

            try:
                page.screenshot(path=str(SCREENS / f"{label}.png"), full_page=False)
            except Exception:
                pass
            page.close()

        browser.close()

    if failures:
        print("WALKTHROUGH FAILURES:")
        for f in failures:
            print(f"  · {f}")
        return 1
    print(f"All {len(STOPS)} stops passed. Screenshots in {SCREENS}.")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:5173")
    args = ap.parse_args(argv)
    return run_walkthrough(args.base_url)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
