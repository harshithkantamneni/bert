"""G9 — Visual regression against committed baselines.

For each user-facing surface, capture a Playwright/Chromium
screenshot and compare it to the baseline in findings/visual_
baselines/. Fail if diff > 2% of pixels.

First run with --record-baseline writes the baselines.

Surfaces are captured for both default + test01 labs, but at a
deterministic viewport (1280×900) and after a fixed wait for
SSE/lazy chunks to land. Animations (framer-motion) are noisy so
the script disables them by injecting CSS that sets transition-
duration: 0s globally.

Run:
    .venv/bin/python tools/eval/visual_regression.py
    .venv/bin/python tools/eval/visual_regression.py --record-baseline
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

from PIL import Image
from pixelmatch.contrib.PIL import pixelmatch
from playwright.sync_api import sync_playwright

REPO = Path("/path/to/Desktop/bert-lab")
BASELINES = REPO / "findings" / "visual_baselines"
DIFFS = REPO / "findings" / "visual_diffs"
BASE = "http://127.0.0.1:5173"

# (path, label, lab) — both lab states for every user-facing surface
SHOTS = [
    ("/",                       "home",                 "default"),
    ("/manuscript",             "manuscript_findings",  "default"),
    ("/manuscript?tab=stream",  "manuscript_stream",    "default"),
    ("/manuscript?tab=loom",    "manuscript_loom",      "default"),
    ("/proofs",                 "proofs",               "default"),
    ("/atlas",                  "atlas",                "default"),
    ("/diagnostics",            "diagnostics",          "default"),
    ("/labs",                   "fleet",                "default"),
    ("/onboard",                "onboard",              "default"),
    ("/",                       "home",                 "test01"),
    ("/atlas",                  "atlas",                "test01"),
    ("/manuscript",             "manuscript_findings",  "test01"),
]

DIFF_THRESHOLD_PCT = 2.0  # > 2% mismatched pixels = visual regression


def disable_animations(page):
    page.add_style_tag(content="""
        *, *::before, *::after {
            transition-duration: 0s !important;
            animation-duration: 0s !important;
            animation-delay: 0s !important;
            scroll-behavior: auto !important;
        }
    """)


def take_shot(ctx, path, label, lab) -> bytes:
    page = ctx.new_page()
    page.goto(f"{BASE}{path}", wait_until="domcontentloaded", timeout=15000)
    disable_animations(page)
    page.wait_for_timeout(2500)  # let SSE land + chunks resolve
    img_bytes = page.screenshot(full_page=False)
    page.close()
    return img_bytes


def compare(baseline_path: Path, fresh_bytes: bytes,
            diff_path: Path) -> tuple[bool, float]:
    """True if within threshold. Returns (ok, diff_pct)."""
    bl = Image.open(baseline_path).convert("RGBA")
    fr = Image.open(io.BytesIO(fresh_bytes)).convert("RGBA")
    if bl.size != fr.size:
        return False, 100.0
    diff = Image.new("RGBA", bl.size)
    n_diff = pixelmatch(bl, fr, diff, threshold=0.1, includeAA=False)
    total = bl.size[0] * bl.size[1]
    pct = (n_diff / total) * 100
    if pct > DIFF_THRESHOLD_PCT:
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff.save(diff_path)
    return pct <= DIFF_THRESHOLD_PCT, pct


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-baseline", action="store_true",
                    help="overwrite baselines with current capture")
    args = ap.parse_args(argv)

    BASELINES.mkdir(parents=True, exist_ok=True)

    fails: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        for path, label, lab in SHOTS:
            ctx = browser.new_context(viewport={"width": 1280, "height": 900})
            if lab == "default":
                ctx.add_init_script(
                    'window.localStorage.removeItem("bert:active-lab")'
                )
            else:
                ctx.add_init_script(
                    f'window.localStorage.setItem("bert:active-lab", "{lab}")'
                )

            try:
                fresh = take_shot(ctx, path, label, lab)
            except Exception as e:
                fails.append(f"{lab}/{label}: capture failed — {e}")
                ctx.close()
                continue
            ctx.close()

            baseline = BASELINES / f"{lab}__{label}.png"
            if args.record_baseline or not baseline.exists():
                baseline.write_bytes(fresh)
                print(f"  ✏  {lab}/{label}: baseline written ({len(fresh)//1024}KB)")
                continue

            diff_path = DIFFS / f"{lab}__{label}.diff.png"
            ok, pct = compare(baseline, fresh, diff_path)
            if ok:
                print(f"  ✓ {lab}/{label}: diff {pct:.2f}% (≤ {DIFF_THRESHOLD_PCT}%)")
            else:
                fails.append(f"{lab}/{label}: diff {pct:.2f}% > {DIFF_THRESHOLD_PCT}% — see {diff_path}")
                print(f"  ✗ {lab}/{label}: diff {pct:.2f}%")
        browser.close()

    print()
    if fails:
        print(f"VISUAL REGRESSION: {len(fails)} failures")
        for f in fails:
            print(f"  · {f}")
        return 1
    if args.record_baseline:
        print(f"VISUAL BASELINES RECORDED to {BASELINES}")
    else:
        print(f"VISUAL REGRESSION CLEAN across {len(SHOTS)} surfaces.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
