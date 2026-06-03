"""Probe every documented API endpoint in api/main.py.

For each GET endpoint without path params, probe bare + ?lab=test01.
Capture status code + first 200 bytes. Flag any 5xx as a code bug,
and any 422 on a presumed-valid request as a schema bug.
"""

from __future__ import annotations

import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

LAB_ROOT = Path("/path/to/Desktop/bert-lab")
MAIN = LAB_ROOT / "api" / "main.py"
BASE = "http://127.0.0.1:5174"


def list_get_endpoints() -> list[str]:
    text = MAIN.read_text()
    endpoints: list[str] = []
    for m in re.finditer(r'@app\.get\(["\']([^"\']+)["\']\)', text):
        path = m.group(1)
        # Skip path-param endpoints (curly braces) and dev-only routes
        if "{" in path or "/.well-known/" in path or path.startswith("/api/dev/"):
            continue
        endpoints.append(path)
    return endpoints


def probe(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            body = r.read(200)
            return r.status, body.decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        try:
            body = e.read(200)
        except Exception:
            body = b""
        return e.code, body.decode("utf-8", "replace")
    except Exception as e:
        return 0, f"NET-ERR {e}"


def main() -> int:
    eps = list_get_endpoints()
    print(f"Probing {len(eps)} GET endpoints …\n")
    bugs: list[str] = []
    for ep in sorted(set(eps)):
        url_bare = f"{BASE}{ep}"
        url_lab = f"{BASE}{ep}{'&' if '?' in ep else '?'}lab=test01"
        code_b, body_b = probe(url_bare)
        code_l, body_l = probe(url_lab)
        flag_b = "BUG" if code_b >= 500 else (
                 "?  " if code_b == 422 else
                 "OK ")
        flag_l = "BUG" if code_l >= 500 else (
                 "?  " if code_l == 422 else
                 "OK ")
        print(f"  [{flag_b}/{flag_l}] {ep:<48} bare={code_b}  lab=test01={code_l}")
        if code_b >= 500:
            bugs.append(f"{ep} bare → {code_b} {body_b[:120]}")
        if code_l >= 500:
            bugs.append(f"{ep} ?lab=test01 → {code_l} {body_l[:120]}")
    print()
    if bugs:
        print(f"\n{len(bugs)} possible BUGS:")
        for b in bugs:
            print(f"  · {b}")
        return 1
    print("API probe: no 5xx anywhere.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
