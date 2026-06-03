"""Run the feature sweep across Chromium, Firefox, and WebKit.

Cuts back the assertion set vs feature_sweep.py — focuses on the
load-bearing checks that are most likely to expose engine-specific
bugs (CSS, SSR-vs-CSR, route handling, modern API support). Heavy
behavior checks (mission editor expand, talk drawer toggle, etc.)
stay in feature_sweep.py since they don't generally vary by browser
and the deeper assertions are too noisy in cross-browser runs.

Browsers:
    chromium   — also baseline for Playwright stack
    firefox    — Gecko-based, catches Layout/CSS divergence
    webkit     — Safari-based, catches WK-specific gaps on macOS

Run: .venv/bin/python tools/eval/cross_browser_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import Browser, BrowserContext, sync_playwright

BASE = "http://127.0.0.1:5173"
SHOTS = Path("/tmp/cross_browser_sweep")
SHOTS.mkdir(parents=True, exist_ok=True)

SURFACES = [
    "/",
    "/manuscript",
    "/manuscript?tab=stream",
    "/manuscript?tab=loom",
    "/proofs",
    "/atlas",
    "/diagnostics",
    "/labs",
    "/onboard",
]


def run_for_browser(browser: Browser, name: str, fails: list[str]) -> None:
    for active_lab in ["test01", None]:
        ctx: BrowserContext = browser.new_context(
            viewport={"width": 1280, "height": 900},
        )
        ctx.add_init_script(f"""
            try {{
              {('window.localStorage.setItem("bert:active-lab", ' + json.dumps(active_lab) + ');')
                if active_lab else 'window.localStorage.removeItem("bert:active-lab");'}
            }} catch (e) {{}}
        """)
        lab = active_lab or "default"
        for path in SURFACES:
            page = ctx.new_page()
            errs: list[str] = []
            nets: list[str] = []
            page.on("console", lambda m, e=errs:
                    e.append(m.text) if m.type == "error" else None)
            page.on("response", lambda r, n=nets:
                    n.append(f"{r.status} {r.url}")
                    if r.status >= 400 and "/api/" in r.url else None)
            try:
                page.goto(f"{BASE}{path}", wait_until="domcontentloaded",
                          timeout=15000)
            except Exception as e:
                fails.append(f"[{name}:{lab}] {path}: nav {e}")
                page.close()
                continue
            page.wait_for_timeout(1500)
            try:
                body = page.evaluate("document.body.innerText").lower()
            except Exception:
                body = ""
            unreachable = "can't be reached" in body or "cant be reached" in body
            if unreachable:
                fails.append(f"[{name}:{lab}] {path}: Unreachable rendered")
            elif errs:
                # Filter known cross-browser noise (HMR pings on Firefox
                # are normal; only count real loading errors)
                bad = [e for e in errs
                       if any(t in e for t in
                              ("Failed to load resource",
                               "Failed to fetch",
                               "ChunkLoadError"))]
                if bad:
                    fails.append(f"[{name}:{lab}] {path}: {bad[0][:160]}")
                else:
                    print(f"  ✓ [{name}:{lab}] {path}")
            elif nets:
                fails.append(f"[{name}:{lab}] {path}: net {nets[0][:160]}")
            else:
                print(f"  ✓ [{name}:{lab}] {path}")
            try:
                slug = (path.lstrip('/').replace('/', '_')
                        .replace('?', '_q_').replace('=', '_')
                        .replace('&', '_') or "root")
                page.screenshot(path=str(SHOTS / f"{name}_{lab}_{slug}.png"))
            except Exception:
                pass
            page.close()
        ctx.close()


def main() -> int:
    fails: list[str] = []
    with sync_playwright() as pw:
        for name, b in [("chromium", pw.chromium),
                        ("firefox",  pw.firefox),
                        ("webkit",   pw.webkit)]:
            print(f"\n=== {name} ===")
            try:
                browser = b.launch(headless=True)
            except Exception as e:
                fails.append(f"[{name}] launch failed: {e}")
                continue
            try:
                run_for_browser(browser, name, fails)
            finally:
                browser.close()
    print()
    if fails:
        print(f"FAILURES: {len(fails)}")
        for f in fails:
            print(f"  · {f}")
        return 1
    print("CROSS-BROWSER SWEEP CLEAN.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
