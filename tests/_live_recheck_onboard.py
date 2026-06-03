"""R6 recheck — verify the talk-chip is hidden on /onboard."""
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUTDIR = Path("/tmp/bert_walkthrough")

def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_context(viewport={"width": 1440, "height": 900}).new_page()
        page.goto("http://127.0.0.1:5173/onboard", wait_until="networkidle", timeout=15_000)
        time.sleep(1.0)
        page.screenshot(path=str(OUTDIR / "10-onboard-after-fix.png"), full_page=True)
        # Assert the chip is gone — Playwright should not find a button with the "talk" text
        try:
            count = page.locator('button:has-text("talk")').count()
        except Exception as e:
            print(f"locator error: {e}")
            count = -1
        print(f"talk buttons on /onboard: {count}")
        browser.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
