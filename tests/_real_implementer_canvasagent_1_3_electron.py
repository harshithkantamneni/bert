"""Phase 1.3 — Electron packaging for CanvasAgent.

Wraps the existing Vite-built React frontend in an Electron shell so we
get a Mac/Win desktop binary and stop fighting browser CORS. The frontend
keeps working standalone (current behavior); Electron just changes the
shell.

Why this dispatch matters:
  - Onboarding completion is observable from a
    real downloadable .dmg / .exe, not from `npm run dev`. We need the
    binary to test fresh-install funnels at all.
  - The README currently says "set OLLAMA_ORIGINS=* or run browser with
    --disable-web-security". Electron's main process makes its own
    HTTP requests so CORS isn't even an issue — that workaround
    evaporates.
  - LangGraph backend (1.1b) and replay+branch (1.2) both need a real
    package surface to land into.

Run: PYTHONPATH=. uv run python tests/_real_implementer_canvasagent_1_3_electron.py
"""
from __future__ import annotations

import json
import sys

from core import subagent

DISPATCH = {
    "dispatch_altitude": "IMPL",
    "role": "implementer",
    "cycle": 12,
    "task": (
        "PHASE 1.3 — package CanvasAgent as an Electron desktop app.\n\n"
        "READ FIRST:\n"
        "- memories/mission.md (Phase 1.3 spec at the bottom)\n"
        "- phase1/canvasagent/package.json (current Vite/React/RF setup)\n"
        "- phase1/canvasagent/README.md (current dev workflow + CORS note)\n"
        "- phase1/canvasagent/vite.config.ts (existing Vite config)\n\n"
        "GOAL: produce a Mac .dmg + Win NSIS .exe build artifact pipeline\n"
        "for CanvasAgent. The existing Vite-bundled frontend gets loaded\n"
        "in an Electron BrowserWindow. Backend integration comes later\n"
        "(1.1b.2 dispatch); for now Electron is purely a shell around\n"
        "the existing browser-only flow.\n\n"
        "REQUIREMENTS:\n\n"
        "1. ADD electron + electron-builder to package.json devDependencies\n"
        "   (electron@^32, electron-builder@^25). Add scripts:\n"
        "     'electron': 'electron .'\n"
        "     'electron:dev': 'concurrently \"npm run dev\" \"wait-on http://localhost:5173 && electron .\"'\n"
        "     'electron:pack': 'npm run build && electron-builder --dir'\n"
        "     'electron:dist': 'npm run build && electron-builder -mw'\n"
        "   Plus 'concurrently' and 'wait-on' as devDeps.\n\n"
        "2. ADD a top-level 'main' field to package.json pointing to\n"
        "   electron/main.js (don't put main process in src/ — keep it\n"
        "   separate from the React renderer).\n\n"
        "3. CREATE electron/main.js — Electron main process:\n"
        "   - Creates BrowserWindow at 1280x800 with nodeIntegration: false\n"
        "     and contextIsolation: true (security baseline)\n"
        "   - In dev (NODE_ENV=development or process.env.ELECTRON_DEV=1):\n"
        "     loads http://localhost:5173 (Vite dev server)\n"
        "   - In production: loads file://<__dirname>/../dist/index.html\n"
        "     (the Vite build output)\n"
        "   - Mac: hide menu bar; reasonable default app menu\n"
        "   - Closes app when last window closes (except Mac, where it\n"
        "     stays in dock)\n"
        "   - Sets app.setName('CanvasAgent')\n\n"
        "4. CREATE electron/preload.js — empty preload, just a placeholder\n"
        "   for future IPC. One line: console.log('CanvasAgent preload').\n\n"
        "5. ADD electron-builder config to package.json under 'build' key:\n"
        "     appId: 'ai.bert-lab.canvasagent'\n"
        "     productName: 'CanvasAgent'\n"
        "     directories: { output: 'release' }\n"
        "     files: ['dist/**/*', 'electron/**/*', 'package.json']\n"
        "     mac: { target: 'dmg', category: 'public.app-category.developer-tools' }\n"
        "     win: { target: 'nsis' }\n"
        "     linux: { target: 'AppImage' }\n"
        "     asar: true\n\n"
        "6. UPDATE README.md — replace the 'CORS' section with a 'Run' section\n"
        "   containing both browser-dev and Electron-dev workflows. Drop\n"
        "   the OLLAMA_ORIGINS=* / --disable-web-security workaround text\n"
        "   (Electron main process makes its own HTTP requests; no CORS).\n"
        "   Add 'Build a desktop binary' section showing\n"
        "   `npm run electron:dist` for Mac+Win .dmg/.exe in release/.\n\n"
        "7. UPDATE .gitignore — add 'release/' (build output), 'out/'\n"
        "   (electron-forge default if anyone runs that), keep dist/.\n\n"
        "8. CREATE electron/icon.png — placeholder 512x512 icon. Use a\n"
        "   simple SVG-style geometric shape (canvas + arrow node motif)\n"
        "   converted to a 512x512 PNG. If converting is hard, use the\n"
        "   `convert` ImageMagick tool via Bash. If neither is available,\n"
        "   create a minimal 512x512 solid-color PNG via a small Python\n"
        "   script using PIL or via base64-decoding a hand-built minimal\n"
        "   PNG header. Worst-case: skip the icon for now and reference\n"
        "   the path in build config; electron-builder will fall back.\n\n"
        "BUDGET: total LoC under 700 across all phase1/canvasagent/ files\n"
        "(was 1214 after 1.1c; +~50 for electron/ scaffolding + +~30 for\n"
        "package.json edits is realistic). Don't blow up the LoC by\n"
        "over-styling the main.js.\n\n"
        "DO NOT touch src/* — frontend stays exactly as 1.1c left it.\n\n"
        "DO write build report to drafts/canvasagent_1_3_electron_report.md\n"
        "with file diff summary, package.json changes, and the exact\n"
        "Quick Start steps for: dev workflow, building a Mac .dmg, building\n"
        "a Windows .exe."
    ),
    "success_criterion": (
        "electron/main.js + electron/preload.js exist; package.json has "
        "main field, electron + electron-builder + concurrently + wait-on "
        "devDeps, build config, electron:dev/pack/dist scripts. README "
        "drops the CORS workaround. drafts/canvasagent_1_3_electron_report.md "
        "exists. ResultPacket schema-validates."
    ),
    "output_path": "drafts/canvasagent_1_3_electron_report.md",
    "model": "mistral/mistral-small-latest",
    "process_hygiene": (
        "SECURITY BASELINE: Electron main.js MUST set nodeIntegration: false "
        "and contextIsolation: true. Don't expose require() or fs to the "
        "renderer. The preload bridge stays empty for now; future IPC "
        "lands in 1.1b.2. PRIVACY: don't add any network call from the "
        "main process — the renderer's existing fetch-to-Ollama is the "
        "only network surface. PACKAGE.JSON: keep dependencies vs "
        "devDependencies clean — only RUNTIME deps in deps; tooling like "
        "electron-builder/concurrently/wait-on is devDeps."
    ),
    "confidence_required": True,
    "falsifier_text": (
        "Failure if any of: (a) any of the 4 new/modified files missing; "
        "(b) `npm install` fails; (c) `npm run electron:pack` fails to "
        "produce a build; (d) main.js sets nodeIntegration:true or "
        "contextIsolation:false; (e) ResultPacket schema-invalid."
    ),
    # Verify Electron config syntax + package.json shape + actual unpacked
    # Mac build via electron-builder --dir. Skip the full .dmg/.exe pass
    # (slow, requires code-signing). The unpacked .app bundle in release/
    # is enough to verify the toolchain works.
    # Crucially: NO `|| true` — past dispatches had `node -e require(main.js) || true`
    # which silently swallowed real missing-dependency errors. If a require fails,
    # the verification SHOULD fail too.
    "verification_command": (
        "set -eo pipefail && cd phase1/canvasagent && "
        "npm install --silent --no-audit --no-fund 2>&1 | tail -3 && "
        "npm run build 2>&1 | tail -3 && "
        "node --check electron/main.js && echo 'main.js: syntax OK' && "
        "node --check electron/preload.js && echo 'preload.js: syntax OK' && "
        "node -e \"const p = require('./package.json'); "
        "  if (!p.main) throw new Error('missing main field'); "
        "  if (!p.scripts['electron:dist']) throw new Error('missing electron:dist'); "
        "  if (!p.devDependencies.electron) throw new Error('missing electron dep'); "
        "  if (!p.build) throw new Error('missing build config'); "
        "  console.log('package.json: main=' + p.main + ', appId=' + p.build.appId);\" && "
        "echo '--- electron-builder --dir (unpacked) ---' && "
        "npx electron-builder --dir --mac 2>&1 | tail -5 && "
        "test -d release/mac-arm64/CanvasAgent.app/Contents/MacOS && "
        "echo 'CanvasAgent.app: bundled OK'"
    ),
    "verification_timeout_secs": 360,
}


def main() -> int:
    print("=" * 72)
    print("PHASE 1.3 — Implementer packages CanvasAgent for Electron")
    print("=" * 72)
    print()

    summary = subagent.run_subagent(DISPATCH)

    print()
    print("=" * 72)
    print("Implementer return")
    print("=" * 72)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "calibration_reasoning"},
                     indent=2, default=str))
    print()
    print("--- calibration_reasoning ---")
    print(summary.get("calibration_reasoning", "")[:1500])
    print()
    print("--- verification ---")
    tel = summary.get("telemetry", {})
    verify = tel.get("verification") if isinstance(tel, dict) else None
    if verify:
        print(f"  ok={verify['ok']} exit={verify['exit_code']} elapsed={verify['elapsed_ms']}ms")
        print(f"  stdout tail:\n{verify.get('stdout', '')[-800:]}")
        if verify.get("stderr"):
            print(f"  stderr tail:\n{verify['stderr'][-400:]}")
    else:
        print("  (no verification block)")
    return 0 if summary["spec_valid"] and summary["result_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
