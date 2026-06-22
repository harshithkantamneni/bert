"""Assemble the LARGE-SCALE corpus (track-B scale point): a real multi-repo
Python codebase far larger than any context window, so full-context is
infeasible and truncation captures a vanishing fraction — the regime where
retrieval is the only option.

Clones a curated set of large, well-known Python repos (shallow), keeps only
source files, and grows until a token target (or all repos) is reached. Reports
the realized token count. Idempotent: skips repos already cloned.

Usage:
    .venv/bin/python benchmarks/v2_big_corpus.py --target-tokens 30000000
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

BIG_ROOT = Path("/tmp/v2_corpora/big")
MANIFEST = Path(__file__).resolve().parents[1] / "benchmarks/results/v2_corpora_manifest.json"

# Large, well-known Python repos — ordered roughly largest-impact first. Shallow
# cloned; only .py kept. Enough headroom to reach tens of millions of tokens.
REPOS = [
    ("cpython", "https://github.com/python/cpython", "Lib"),       # stdlib only
    ("django", "https://github.com/django/django", "django"),
    ("numpy", "https://github.com/numpy/numpy", "numpy"),
    ("pandas", "https://github.com/pandas-dev/pandas", "pandas"),
    ("scipy", "https://github.com/scipy/scipy", "scipy"),
    ("sklearn", "https://github.com/scikit-learn/scikit-learn", "sklearn"),
    ("transformers", "https://github.com/huggingface/transformers", "src"),
    ("sqlalchemy", "https://github.com/sqlalchemy/sqlalchemy", "lib"),
    ("sympy", "https://github.com/sympy/sympy", "sympy"),
    ("ansible", "https://github.com/ansible/ansible", "lib"),
    ("salt", "https://github.com/saltstack/salt", "salt"),
    ("airflow", "https://github.com/apache/airflow", "airflow-core"),
]
EXTS = {".py"}


def _est_tokens(root: Path) -> int:
    total = 0
    for f in root.rglob("*.py"):
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return total // 4  # ~4 chars/token


def _clone(name: str, url: str, subdir: str) -> Path | None:
    dest = BIG_ROOT / name
    if dest.exists() and any(dest.rglob("*.py")):
        return dest
    BIG_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = BIG_ROOT / f".{name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"  cloning {name}…", flush=True)
    r = subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none",
                        url, str(tmp)], capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        print(f"    clone failed {name}: {r.stderr[:160]}", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    src = tmp / subdir if (tmp / subdir).exists() else tmp
    dest.mkdir(parents=True, exist_ok=True)
    kept = 0
    for f in src.rglob("*.py"):
        rel = f.relative_to(src)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(f, out)
            kept += 1
        except OSError:
            pass
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"    {name}: kept {kept} .py files", flush=True)
    return dest if kept else None


def build_big_corpus(target_tokens: int | None = None) -> dict:
    """Clone repos until target_tokens (or all). Returns a manifest entry and
    appends it to the shared corpora manifest."""
    BIG_ROOT.mkdir(parents=True, exist_ok=True)
    realized = _est_tokens(BIG_ROOT)
    for name, url, subdir in REPOS:
        if target_tokens and realized >= target_tokens:
            break
        _clone(name, url, subdir)
        realized = _est_tokens(BIG_ROOT)
        print(f"  running total: ~{realized:,} tokens", flush=True)
    n_files = sum(1 for _ in BIG_ROOT.rglob("*.py"))
    entry = {"name": "big", "lang": "python", "root": str(BIG_ROOT),
             "n_files": n_files, "est_tokens": realized,
             "repos": [n for n, _, _ in REPOS if (BIG_ROOT / n).exists()]}
    # merge into manifest
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    man = [c for c in man if c.get("name") != "big"] + [entry]
    MANIFEST.write_text(json.dumps(man, indent=2))
    return entry


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=None,
                    help="stop adding repos once this token count is reached (None = all)")
    args = ap.parse_args()
    e = build_big_corpus(args.target_tokens)
    print(f"\nBIG corpus: {e['n_files']:,} files, ~{e['est_tokens']:,} tokens, "
          f"repos={e['repos']}")
