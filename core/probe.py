"""Provider readiness probe.

Used by `lab.py probe`. Hits each provider's /v1/models endpoint and reports
reachable + first few model IDs. ~5s total for all 8.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from core import config
from core import provider as prov


def run() -> int:
    cfg = config.load()
    print("=== bert-lab provider probe ===")
    print(f"credentials.json keys present: {[k for k, v in cfg.credentials.items() if v]}")
    print()

    targets = list(prov.PROVIDERS.keys())

    def _probe(name: str) -> tuple[str, bool, list[str], str, float]:
        spec = prov.PROVIDERS[name]
        if spec.requires_api_key and not cfg.has(spec.api_key_env):
            return name, False, [], f"missing credential {spec.api_key_env}", 0.0
        start = time.monotonic()
        ok, ids, err = prov.probe_models(name)
        return name, ok, ids, err, time.monotonic() - start

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_probe, targets))

    rc = 0
    for name, ok, ids, err, elapsed in results:
        flag = "✓" if ok else "✗"
        suffix = f"({len(ids)} models, {elapsed*1000:.0f} ms)" if ok else f"({err})"
        print(f"  {flag} {name:12s} {suffix}")
        if ok and ids:
            sample = ", ".join(ids[:3])
            if len(ids) > 3:
                sample += f", … +{len(ids) - 3}"
            print(f"       e.g. {sample}")
        if not ok and "missing credential" not in err:
            rc = 1
    return rc
