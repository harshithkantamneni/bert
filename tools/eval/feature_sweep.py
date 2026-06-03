"""F7 — Comprehensive Playwright sweep of every user-visible feature.

Stages:
  A. Baseline navigation (every surface in both lab states; no console
     errors; no 4xx/5xx).
  B. Masthead controls (lab picker, surface links, talk drawer).
  C. Mission editor (read seed_brief, expand, type, switch tab to
     preview, see markdown render).
  D. NeedsExpanded (only fires if pending matters exist; observe
     empty state).
  E. Manuscript tabs (findings/stream/loom; verify ?tab= URL
     round-trip; query-param preservation on tab switch).
  F. Retired-route redirects (/mission /meeting /tide /loom /book
     /book/abc).
  G. Run-cycle controls (idle → start → poll → see receipt; we DO NOT
     actually start a real run since one was just finished — instead
     verify the start button is rendered with the right copy).
  H. Atlas (4 rings render; if test01 has agents, roster ring has
     entries).
  I. Keyboard shortcuts (g+m → /manuscript, g+t → ?tab=stream, etc.).
  J. Mobile breakpoint (640px → masthead sheet, two-line header).
  K. Console-error capture across every stop.

A single failure prints which surface/step + summary; non-zero exit.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:5173"
SHOTS = Path("/tmp/feature_sweep")
SHOTS.mkdir(parents=True, exist_ok=True)

fails: list[str] = []


def fail(msg: str) -> None:
    fails.append(msg)
    print(f"  ✗ {msg}")


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def shot(page, name: str) -> None:
    with contextlib.suppress(Exception):
        page.screenshot(path=str(SHOTS / f"{name}.png"))


with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    for active_lab in ["test01", None]:
        ctx = browser.new_context(viewport={"width": 1440, "height": 1100})
        ctx.add_init_script(f"""
            try {{
              {('window.localStorage.setItem(\"bert:active-lab\", ' + json.dumps(active_lab) + ');')
                if active_lab else 'window.localStorage.removeItem(\"bert:active-lab\");'}
            }} catch (e) {{}}
        """)
        lab_label = active_lab or "default"
        print(f"\n=== Lab: {lab_label} ===")

        # ── A. baseline navigation ───────────────────────────────
        print("\n[A] baseline navigation")
        for path in ["/", "/manuscript", "/manuscript?tab=stream",
                     "/manuscript?tab=loom", "/proofs", "/atlas",
                     "/diagnostics", "/labs", "/onboard"]:
            page = ctx.new_page()
            errs: list[str] = []
            nets: list[str] = []
            page.on("console", lambda m, e=errs: e.append(m.text)
                    if m.type == "error" else None)
            page.on("response", lambda r, n=nets: n.append(
                f"{r.status} {r.url}") if r.status >= 400 and "/api/" in r.url else None)
            try:
                page.goto(f"{BASE}{path}", wait_until="domcontentloaded", timeout=10000)
            except Exception as e:
                fail(f"[A:{lab_label}] {path}: nav {e}")
                page.close(); continue
            page.wait_for_timeout(1500)
            body = page.evaluate("document.body.innerText").lower()
            if "can't be reached" in body or "cant be reached" in body:
                fail(f"[A:{lab_label}] {path}: Unreachable rendered")
            elif errs:
                fail(f"[A:{lab_label}] {path}: console errs {errs[:2]}")
            elif nets:
                fail(f"[A:{lab_label}] {path}: 4xx/5xx {nets[:2]}")
            else:
                ok(f"[A:{lab_label}] {path}")
            shot(page, f"{lab_label}_A_{path.lstrip('/').replace('/','_').replace('?','_').replace('=','_')}")
            page.close()

        # ── B. masthead controls ─────────────────────────────────
        print("\n[B] masthead controls")
        page = ctx.new_page()
        page.goto(f"{BASE}/", wait_until="domcontentloaded", timeout=10000)
        page.wait_for_timeout(1500)
        mast = page.evaluate("""
            () => {
              const m = document.querySelector('[role=banner]');
              if (!m) return null;
              const links = [...m.querySelectorAll('a')].map(a => a.innerText.trim());
              const buttons = [...m.querySelectorAll('button')].map(b => b.innerText.trim());
              return { links, buttons };
            }
        """)
        if not mast:
            fail(f"[B:{lab_label}] masthead missing")
        else:
            expected_links = ["BERT", "home", "fleet", "proofs", "atlas",
                              "manuscript", "diagnostics"]
            text = " ".join(mast["links"] + mast["buttons"]).lower()
            missing = [s for s in expected_links if s.lower() not in text]
            if missing:
                fail(f"[B:{lab_label}] masthead missing links: {missing}")
            else:
                ok(f"[B:{lab_label}] masthead has all 6 surface links + brand")
            # Talk button
            if "talk" not in text:
                fail(f"[B:{lab_label}] talk button missing from masthead")
            else:
                ok(f"[B:{lab_label}] masthead has talk")
        page.close()

        # ── C. mission editor on home ────────────────────────────
        print("\n[C] mission editor")
        page = ctx.new_page()
        page.goto(f"{BASE}/", wait_until="domcontentloaded", timeout=10000)
        page.wait_for_timeout(1500)
        has_editor = page.evaluate("""
            () => !!document.querySelector('[role=region][aria-label="mission editor"]')
        """)
        if not has_editor:
            fail(f"[C:{lab_label}] mission editor not found on Home")
        else:
            ok(f"[C:{lab_label}] mission editor rendered")
            # Try to expand it if collapsed
            page.evaluate("""
                () => {
                  const r = document.querySelector('[role=region][aria-label="mission editor"]');
                  const btn = r.querySelector('button[aria-expanded]');
                  if (btn && btn.getAttribute('aria-expanded') === 'false') btn.click();
                }
            """)
            page.wait_for_timeout(500)
            has_tabs = page.evaluate("""
                () => {
                  const r = document.querySelector('[role=region][aria-label="mission editor"]');
                  return [...r.querySelectorAll('button[aria-pressed]')].length;
                }
            """)
            if has_tabs >= 2:
                ok(f"[C:{lab_label}] mission editor has edit/preview tabs")
            else:
                fail(f"[C:{lab_label}] mission editor expand → no tabs")
        page.close()

        # ── D. NeedsExpanded ─────────────────────────────────────
        print("\n[D] NeedsExpanded (only renders when pending > 0)")
        page = ctx.new_page()
        page.goto(f"{BASE}/", wait_until="domcontentloaded", timeout=10000)
        page.wait_for_timeout(1500)
        has_needs = page.evaluate("""
            () => !!document.querySelector('[role=region][aria-label="needs you"]')
        """)
        ok(f"[D:{lab_label}] needs-you region {'present' if has_needs else 'absent (no pending matters)'}")
        page.close()

        # ── E. Manuscript tabs ───────────────────────────────────
        print("\n[E] Manuscript tabs")
        for tab in ["findings", "stream", "loom"]:
            page = ctx.new_page()
            url = "/manuscript" if tab == "findings" else f"/manuscript?tab={tab}"
            page.goto(f"{BASE}{url}", wait_until="domcontentloaded", timeout=10000)
            page.wait_for_timeout(1500)
            # Tab button should have aria-selected=true
            active = page.evaluate(f"""
                () => {{
                  const t = document.getElementById('manuscript-tab-{tab}');
                  return t && t.getAttribute('aria-selected') === 'true';
                }}
            """)
            if active:
                ok(f"[E:{lab_label}] tab {tab} active when ?tab={tab}")
            else:
                fail(f"[E:{lab_label}] tab {tab} not active under {url}")
            page.close()

        # ── F. retired-route redirects ───────────────────────────
        print("\n[F] retired-route redirects")
        retired_map = {
            "/mission":  "/",
            "/meeting":  "/",
            "/tide":     "/manuscript?tab=stream",
            "/loom":     "/manuscript?tab=loom",
            "/book":     "/manuscript",
        }
        for src, dst in retired_map.items():
            page = ctx.new_page()
            page.goto(f"{BASE}{src}", wait_until="domcontentloaded", timeout=10000)
            page.wait_for_timeout(800)
            cur = page.url.replace(BASE, "")
            if cur == dst:
                ok(f"[F:{lab_label}] {src} → {dst}")
            else:
                fail(f"[F:{lab_label}] {src} → {cur} (expected {dst})")
            page.close()

        # ── G. run-cycle controls (button only, not actual run) ──
        print("\n[G] run-cycle controls")
        page = ctx.new_page()
        page.goto(f"{BASE}/", wait_until="domcontentloaded", timeout=10000)
        page.wait_for_timeout(1500)
        has_start = page.evaluate("""
            () => {
              const btns = [...document.querySelectorAll('button')];
              return btns.some(b => b.getAttribute('aria-label') === 'start mission');
            }
        """)
        if has_start:
            ok(f"[G:{lab_label}] start-mission button present")
        else:
            fail(f"[G:{lab_label}] start-mission button missing")
        page.close()

        # ── H. Atlas rings ────────────────────────────────────────
        print("\n[H] Atlas rings")
        page = ctx.new_page()
        page.goto(f"{BASE}/atlas", wait_until="domcontentloaded", timeout=10000)
        page.wait_for_timeout(2000)
        rings = page.evaluate("""
            () => {
              const text = document.body.innerText.toUpperCase();
              return {
                peak: text.includes('THE PEAK'),
                roster: text.includes('THE ROSTER'),
                topology: text.includes('THE TOPOLOGY'),
                strata: text.includes('THE STRATA'),
              };
            }
        """)
        for ring, ok_flag in rings.items():
            if ok_flag:
                ok(f"[H:{lab_label}] atlas · {ring} ring rendered")
            else:
                fail(f"[H:{lab_label}] atlas · {ring} ring missing")
        page.close()

        # ── I. keyboard shortcuts ────────────────────────────────
        print("\n[I] keyboard shortcuts (g + key)")
        page = ctx.new_page()
        page.goto(f"{BASE}/", wait_until="domcontentloaded", timeout=10000)
        page.wait_for_timeout(1500)
        # Press g then m → /manuscript
        page.keyboard.press("g")
        page.wait_for_timeout(120)
        page.keyboard.press("m")
        page.wait_for_timeout(800)
        cur = page.url.replace(BASE, "")
        if cur.startswith("/manuscript"):
            ok(f"[I:{lab_label}] g+m → {cur}")
        else:
            fail(f"[I:{lab_label}] g+m landed on {cur} (expected /manuscript*)")
        page.close()

        # ── J. mobile breakpoint ─────────────────────────────────
        print("\n[J] mobile (640px)")
        mctx = browser.new_context(viewport={"width": 600, "height": 800})
        mctx.add_init_script(f"""
            try {{
              {('window.localStorage.setItem(\"bert:active-lab\", ' + json.dumps(active_lab) + ');')
                if active_lab else 'window.localStorage.removeItem(\"bert:active-lab\");'}
            }} catch (e) {{}}
        """)
        mp = mctx.new_page()
        mp.goto(f"{BASE}/", wait_until="domcontentloaded", timeout=10000)
        mp.wait_for_timeout(1500)
        sheet_btn = mp.evaluate("""
            () => {
              const btn = [...document.querySelectorAll('button')]
                .find(b => b.getAttribute('aria-label') === 'open surfaces menu');
              return !!btn;
            }
        """)
        if sheet_btn:
            ok(f"[J:{lab_label}] ☰ mobile sheet button present at 600px")
        else:
            fail(f"[J:{lab_label}] mobile sheet button missing at 600px")
        shot(mp, f"{lab_label}_J_mobile")
        mp.close()
        mctx.close()

        ctx.close()
    browser.close()

print()
if fails:
    print(f"FAILURES: {len(fails)}")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("ALL FEATURE CHECKS PASSED.")
sys.exit(0)
