"""G6 — License scan across npm + python dep trees.

Asserts every dep ships under a permissive license (MIT / BSD /
Apache-2.0 / ISC / 0BSD / MPL-2.0 / LGPL-2.1+ for dynamically-linked
Python; CC0 / Public Domain / Python-2.0 explicitly allowed).

Bert is a proprietary product — copyleft (GPL-3.0+, AGPL) deps would
force source release. Permissive-only audit catches that before
release. LGPL is allowed for Python because Python's import is
dynamic linking by definition.

Run: .venv/bin/python tools/eval/license_scan.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/path/to/Desktop/bert-lab")


PERMISSIVE_LICENSES = {
    # MIT family
    "MIT", "MIT License", "MIT-CMU",
    # BSD family
    "BSD", "BSD License",
    "BSD-2-Clause", "BSD-3-Clause", "0BSD",
    "3-Clause BSD License",
    # space-form variants pip-licenses emits for some packages
    "BSD 3-Clause", "BSD 2-Clause", "BSD 3-Clause License",
    # Apache
    "Apache 2.0", "Apache-2.0", "Apache 2.0 License",
    "Apache License 2.0", "Apache Software License", "Apache2.0",
    # ISC
    "ISC", "ISC License (ISCL)",
    # Python ecosystem
    "Python Software Foundation License",
    "Python-2.0", "PSF-2.0", "CNRI-Python",
    # Mozilla — weakly copyleft but compatible with proprietary
    "Mozilla Public License 2.0 (MPL 2.0)", "MPL-2.0",
    # Public domain / no-rights
    "Unlicense", "The Unlicense (Unlicense)",
    "CC0-1.0", "CC0 (Public Domain)",
    "CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
    "Public Domain", "Public Domain (CC0)",
    # CC-BY: attribution-only, used by data-only deps like caniuse-lite.
    # Not viral — fine to ship alongside proprietary code.
    "CC-BY-4.0", "CC-BY-3.0",
    "Historical Permission Notice and Disclaimer (HPND)", "HPND",
    "Zlib", "WTFPL", "BlueOak-1.0.0",
    # Eclipse — allowed for our deploy model
    "Eclipse Public License 2.0 (EPL-2.0)",
    # LGPL — allowed for Python because Python's import is dynamic
    # linking, which keeps us compatible with LGPL §6.
    "GNU Lesser General Public License v2 or later (LGPLv2+)",
    "GNU Lesser General Public License v3 or later (LGPLv3+)",
    "LGPL-2.1-or-later", "LGPL-3.0-only", "LGPL-3.0-or-later",
}


# Python deps whose pip-licenses returns UNKNOWN because the license
# is shipped as a separate LICENSE file rather than in metadata. We
# verify each by hand and pin here.
ALLOWLISTED_UNKNOWN_PY = {
    "caio",            # MIT (per pypi.org/project/caio)
    "sigstore-models", # Apache-2.0 (per github.com/sigstore/sigstore-python)
}


# Copyleft (GPL) deps that are BENCHMARK/DEV-ONLY and never shipped in
# the proprietary product, so they don't trigger GPL source-disclosure
# (which only fires on DISTRIBUTION of the GPL'd code). All three are
# transitive deps of `ir_datasets`, used solely by
# benchmarks/b2_beir_scifact.py to download + parse the BEIR scifact
# corpus. They are not imported by core/ or tools/ and are not bundled
# in any release artifact. Re-evaluate if ir_datasets ever moves onto a
# shipped code path.
BENCHMARK_ONLY_COPYLEFT_PY = {
    "warc3-wet",            # GPLv2 — WARC parsing for ir_datasets
    "warc3-wet-clueweb09",  # GPLv2 — WARC parsing for ir_datasets
}


def _is_permissive(raw: str) -> bool:
    """SPDX-aware permissive check. Some packages stuff the full
    license body into the field — take the first line. Then split on
    ' OR ' / ' AND ' / ';' / ',' and accept if ANY component is
    permissive (most-permissive interpretation of dual licensing)."""
    first_line = raw.split("\n", 1)[0].strip()
    parts = re.split(r" OR | AND |;|,", first_line)
    parts = [p.strip() for p in parts if p.strip()]
    return any(p in PERMISSIVE_LICENSES for p in parts)


def scan_npm() -> tuple[int, list[tuple[str, str]]]:
    """Run license-checker-rseidelsohn on bert/v4 production deps."""
    out_path = "/tmp/npm_licenses.json"
    subprocess.run(
        ["npx", "license-checker-rseidelsohn", "--json",
         "--production", "--excludePrivatePackages",
         "--out", out_path],
        cwd=str(REPO / "bert" / "v4"),
        check=True, capture_output=True, timeout=300,
    )
    d = json.loads(Path(out_path).read_text())
    problems: list[tuple[str, str]] = []
    for pkg, info in d.items():
        raw = info.get("licenses", "").strip()
        if raw.startswith("("):
            raw = raw[1:-1]
        if not _is_permissive(raw):
            problems.append((pkg, raw))
    return len(d), problems


def scan_py() -> tuple[int, list[tuple[str, str]]]:
    """Run pip-licenses against the active venv."""
    out_path = "/tmp/py_licenses.json"
    # 300s (was 120s): pip-licenses reads license metadata for ~260
    # packages from site-packages. It's disk-I/O-bound (~115s wall /
    # ~3s CPU on a disk-pressured macOS), which tipped over the 120s
    # ceiling when run after 17 other eval stages. Matches the 300s used
    # by the sibling subprocess call above.
    with open(out_path, "w") as _out:
        subprocess.run(
            [str(REPO / ".venv" / "bin" / "pip-licenses"),
             "--format=json"],
            stdout=_out,
            check=True, timeout=300,
        )
    d = json.loads(Path(out_path).read_text())
    problems: list[tuple[str, str]] = []
    for entry in d:
        raw = entry.get("License", "UNKNOWN").strip()
        name = entry.get("Name", "?")
        if raw == "UNKNOWN" and name in ALLOWLISTED_UNKNOWN_PY:
            continue
        if name in BENCHMARK_ONLY_COPYLEFT_PY:
            # GPL but benchmark-only + never distributed — see allowlist note
            continue
        if not _is_permissive(raw):
            problems.append((name, raw[:80]))
    return len(d), problems


def main() -> int:
    npm_total, npm_problems = scan_npm()
    py_total, py_problems = scan_py()

    print(f"npm packages: {npm_total} scanned, {len(npm_problems)} problematic")
    for p in npm_problems[:25]:
        print(f"  · {p[0]:40s} {p[1]}")

    print(f"py packages:  {py_total} scanned, {len(py_problems)} problematic")
    for p in py_problems[:25]:
        print(f"  · {p[0]:40s} {p[1]}")

    if npm_problems or py_problems:
        return 1
    print()
    print(f"LICENSE SCAN CLEAN — {npm_total + py_total} dependencies, "
          f"all permissive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
