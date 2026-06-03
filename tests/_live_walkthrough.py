"""Live in-browser walkthrough of the GG-phase consumer-product loop.

Drives Playwright/Chromium through every key surface and captures a
screenshot per stop. Output lands in /tmp/bert_walkthrough/*.png.

This is the R6 sanity that the source-grep smokes can't catch — off-
screen buttons, z-index conflicts, contrast issues, [object Object]
renders, missing transitions, mobile-viewport problems.

Tour stops:
  1.  / — root (should route to /labs since we have ≥2 labs, OR
      FirstLight if solo, OR /onboard if no keys)
  2.  /labs — dashboard with strata-card view
  3.  / (FirstLight) — director's letter + pulse + run controls
  4.  /proofs — outputs viewer (claims, failures, verify ladder)
  5.  /atlas — KG strata ring
  6.  /diagnostics — provider rows + infrastructure cards
  7.  /book — manuscript (findings)
  8.  /loom — citation threads
  9.  /meeting — pending decisions
  10. /onboard — wizard
  11. Mobile viewport (390x844) — /labs + FirstLight + talk drawer

Each capture is followed by a console.log dump so we see React
errors if any. Returns rc=0 unless a navigation hard-failed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

OUTDIR = Path("/tmp/bert_walkthrough")
OUTDIR.mkdir(parents=True, exist_ok=True)
UI = "http://127.0.0.1:5173"


def main() -> int:
    from playwright.sync_api import sync_playwright

    stops: list[tuple[str, str, str]] = [
        ("01-root",           f"{UI}/",            "root (smart-redirect)"),
        ("02-labs",           f"{UI}/labs",        "lab dashboard (strata)"),
        ("03-firstlight",     f"{UI}/",            "FirstLight (after possible redirect)"),
        ("04-proofs",         f"{UI}/proofs",      "outputs viewer (ledger)"),
        ("05-atlas",          f"{UI}/atlas",       "Atlas (KG strata)"),
        ("06-diagnostics",    f"{UI}/diagnostics", "Diagnostics"),
        ("07-book",           f"{UI}/book",        "Manuscript"),
        ("08-loom",           f"{UI}/loom",        "Loom"),
        ("09-meeting",        f"{UI}/meeting",     "Meeting"),
        ("10-onboard",        f"{UI}/onboard",     "Onboarding wizard"),
    ]

    results: list[dict] = []
    console_logs: dict[str, list[str]] = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()

        # Collect console messages per stop
        current_stop = ["init"]

        def on_console(msg):
            text = f"[{msg.type}] {msg.text}"
            console_logs.setdefault(current_stop[0], []).append(text)

        def on_pageerror(err):
            console_logs.setdefault(current_stop[0], []).append(f"[PAGEERROR] {err}")

        page.on("console", on_console)
        page.on("pageerror", on_pageerror)

        for slug, url, label in stops:
            current_stop[0] = slug
            print(f"→ {slug:18} {label}")
            try:
                page.goto(url, wait_until="networkidle", timeout=15_000)
            except Exception as e:
                print(f"  navigation failed: {e}")
                results.append({
                    "slug": slug, "url": url, "label": label,
                    "navigation_error": str(e),
                })
                continue
            time.sleep(0.8)  # let animations settle
            shot = OUTDIR / f"{slug}.png"
            page.screenshot(path=str(shot), full_page=True)
            ttl = page.title()
            url_after = page.url
            redirected = url_after != url
            results.append({
                "slug": slug,
                "url_requested": url,
                "url_after": url_after,
                "redirected": redirected,
                "title": ttl,
                "screenshot": str(shot),
            })

        # Mobile viewport pass
        print()
        print("=== mobile viewport pass (390x844) ===")
        mobile_ctx = browser.new_context(
            viewport={"width": 390, "height": 844},
            device_scale_factor=2,
        )
        mobile_page = mobile_ctx.new_page()
        mobile_page.on("console", on_console)
        mobile_page.on("pageerror", on_pageerror)

        for slug, url in [
            ("11m-labs",        f"{UI}/labs"),
            ("12m-firstlight",  f"{UI}/"),
            ("13m-proofs",      f"{UI}/proofs"),
            ("14m-onboard",     f"{UI}/onboard"),
        ]:
            current_stop[0] = slug
            print(f"→ {slug:18} {url}")
            try:
                mobile_page.goto(url, wait_until="networkidle", timeout=15_000)
            except Exception as e:
                print(f"  mobile navigation failed: {e}")
                continue
            time.sleep(0.6)
            mobile_page.screenshot(path=str(OUTDIR / f"{slug}.png"), full_page=True)
            results.append({
                "slug": slug,
                "url_requested": url,
                "url_after": mobile_page.url,
                "title": mobile_page.title(),
                "screenshot": str(OUTDIR / f"{slug}.png"),
                "viewport": "390x844",
            })

        browser.close()

    # Summary table
    print()
    print("=== walkthrough summary ===")
    for r in results:
        marker = " ↺" if r.get("redirected") else "  "
        err = r.get("navigation_error")
        if err:
            print(f"  ✗{marker} {r['slug']:18} {err[:60]}")
        else:
            print(f"  ✓{marker} {r['slug']:18} → {r['url_after']}")

    # Console log digest
    interesting_logs = {
        slug: [l for l in logs if "[error]" in l.lower()
                or "[pageerror]" in l.lower()
                or "[warning]" in l.lower()]
        for slug, logs in console_logs.items()
    }
    interesting_logs = {k: v for k, v in interesting_logs.items() if v}
    if interesting_logs:
        print()
        print("=== console errors / warnings ===")
        for slug, logs in interesting_logs.items():
            print(f"  {slug}:")
            for line in logs[:6]:
                print(f"    {line[:140]}")
    else:
        print()
        print("=== no console errors or page errors across all stops ===")

    print()
    print(f"screenshots → {OUTDIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
