"""Run axe-core against every user-facing surface; report violations.

Uses the axe-core JS bundle injected into the page (no Playwright
JS API needed). Sets minimum severity to "serious" to filter noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:5173"
AXE_JS = Path("/path/to/Desktop/bert-lab/bert/v4/node_modules/axe-core/axe.min.js").read_text()

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


def audit(path: str) -> list[dict]:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        page = ctx.new_page()
        try:
            page.goto(f"{BASE}{path}", wait_until="domcontentloaded", timeout=10000)
        except Exception as e:
            browser.close()
            return [{"id": "nav-fail", "impact": "critical",
                     "description": str(e), "nodes": []}]
        page.wait_for_timeout(1500)
        # Inject + run axe
        page.evaluate(AXE_JS)
        result = page.evaluate("""
            () => axe.run(document, {
              runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa'] }
            }).then(r => r.violations)
        """)
        browser.close()
        # Filter to serious or critical only
        out = [v for v in result if v.get("impact") in ("serious", "critical")]
        return out


def main() -> int:
    print("Running axe-core (WCAG 2.0 A + AA, severity ≥ serious)…\n")
    total = 0
    for path in SURFACES:
        violations = audit(path)
        if violations:
            print(f"  ✗ {path}: {len(violations)} violation(s)")
            for v in violations[:5]:
                tgt_count = len(v.get("nodes", []))
                print(f"      [{v['impact']}] {v['id']} · {tgt_count} node(s)")
                print(f"          {v['description'][:160]}")
                # First target's HTML snippet
                if v.get("nodes"):
                    snip = v["nodes"][0].get("html", "")[:120]
                    print(f"          example: {snip}")
            total += len(violations)
        else:
            print(f"  ✓ {path}")
    print()
    if total:
        print(f"AXE: {total} violation(s) at severity ≥ serious")
        return 1
    print("AXE: no serious or critical violations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
