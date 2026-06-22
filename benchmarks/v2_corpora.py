"""v2 benchmark corpora — a varied, multi-language corpus set for the QA +
agentic-grep tracks.

The v1 B9 work ran on a SINGLE corpus (httpx+starlette, ~57 Python files at
/tmp/b9_corpus). A skeptical reviewer's first objection is "you tuned to one
codebase" — a hybrid retriever that wins on one repo's idioms may lose on
another language or domain. This module fixes that by assembling three distinct
corpora and pinning their provenance so the whole QA sweep is reproducible:

  c1  python-web   httpx + starlette  (REUSED from /tmp/b9_corpus — the v1 corpus)
  c2  go-web       gin-gonic/gin      (different LANGUAGE: Go, web framework)
  c3  python-data  pydantic           (different DOMAIN + larger: validation/typing)

Design rules that keep this defensible:
  - Source files only (.py for Python corpora, .go for the Go corpus). No
    vendored deps, no tests, no generated files, no docs — those would inflate
    token counts and muddy "what the model is searching".
  - A per-corpus file cap (CAP_FILES) keeps each tractable for local bge
    indexing on an M3 Pro. We select the cap deterministically (sorted by path)
    so a re-run picks the SAME files.
  - Clones are shallow (--depth 1) and PINNED to a recorded commit, so the
    manifest is a reproducible artifact, not "whatever HEAD was that day".
  - Idempotent: an already-cloned corpus is reused; build_corpora() never
    re-clones if the tree is present.

est_tokens is chars//4 — the same crude-but-consistent proxy the rest of the
B9 suite uses (see benchmarks/b9_rag_stats.est_tokens); it is NOT a real BPE
count, only a relative size signal across corpora.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
MANIFEST_PATH = RESULTS / "v2_corpora_manifest.json"

# Where the new corpora live on disk. c1 is the pre-existing v1 corpus.
V2_ROOT = Path("/tmp/v2_corpora")
C1_ROOT = Path("/tmp/b9_corpus")          # reuse — do NOT re-clone

# Per-corpus source-file cap. Keeps local bge indexing tractable while staying
# in the requested 150-400 file band across the set. c1 is small by nature.
CAP_FILES = 320

CHARS_PER_TOKEN = 4   # crude est_tokens proxy, consistent with b9_rag_stats


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    lang: str
    root: Path
    ext: str                # source extension to KEEP, e.g. ".py" / ".go"
    repo: str | None        # git URL; None for the reused c1
    subdir: str | None      # restrict the kept tree to this subdir (post-clone)
    # path fragments to EXCLUDE (tests, examples, vendored, generated)
    exclude: tuple[str, ...] = ()


# ── corpus definitions ───────────────────────────────────────────────
SPECS: list[CorpusSpec] = [
    CorpusSpec(
        name="c1", lang="python", root=C1_ROOT, ext=".py", repo=None,
        subdir=None,
        # c1 is already curated to package source by the v1 B9 work; keep it
        # byte-for-byte identical (all 57 .py files, incl. starlette's
        # production testclient.py — that is library code, not a test). Only
        # exclude genuine test files / caches, matched PRECISELY (not the bare
        # substring "test", which would wrongly drop testclient.py).
        exclude=("/tests/", "/test_", "_test.py", "__pycache__"),
    ),
    CorpusSpec(
        name="c2", lang="go", root=V2_ROOT / "c2", ext=".go",
        repo="https://github.com/gin-gonic/gin.git",
        subdir=None,
        # Go convention: *_test.go are tests; testdata/ + examples are noise.
        exclude=("_test.go", "/testdata/", "/examples/", "/example/",
                 "/vendor/", "/.git/"),
    ),
    CorpusSpec(
        name="c3", lang="python", root=V2_ROOT / "c3",
        ext=".py",
        repo="https://github.com/pydantic/pydantic.git",
        # keep only the package source, not tests/docs/benchmarks
        subdir="pydantic",
        exclude=("/tests/", "/test_", "/docs/", "__pycache__",
                 "/.git/", "/benchmarks/"),
    ),
]


def _git(*args: str, cwd: Path | None = None, timeout: float = 300.0) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd) if cwd else None,
                          capture_output=True, text=True, timeout=timeout, check=False)


def _is_excluded(rel_posix: str, exclude: tuple[str, ...]) -> bool:
    return any(frag in rel_posix for frag in exclude)


def _source_files(spec: CorpusSpec) -> list[Path]:
    """All kept source files for a corpus, deterministically sorted then capped.

    Kept = right extension, under the (optional) subdir, not matching any
    exclude fragment. Sort-by-path makes the cap selection reproducible.

    The subdir is only meaningful on a FRESHLY cloned tree; after _prune_to_kept
    collapses that subdir up to the corpus root, the subdir no longer exists, so
    we fall back to scanning spec.root directly.
    """
    base = spec.root / spec.subdir if spec.subdir else spec.root
    if spec.subdir and not base.exists():
        base = spec.root          # subdir already collapsed by a prior prune
    if not base.exists():
        return []
    keep: list[Path] = []
    for f in sorted(base.rglob(f"*{spec.ext}")):
        if not f.is_file():
            continue
        rel = f.relative_to(spec.root).as_posix()
        if _is_excluded("/" + rel, spec.exclude):
            continue
        keep.append(f)
    return keep[:CAP_FILES]


def _prune_to_kept(spec: CorpusSpec) -> None:
    """Delete everything in a CLONED corpus tree except the kept source files,
    so the on-disk corpus root contains only what the benchmark will index.
    The .git dir is removed too (we pin the commit in the manifest instead).

    Only runs on cloned corpora (spec.repo is not None) — never touches the
    reused c1 tree.
    """
    if spec.repo is None:
        return
    keep = set(_source_files(spec))
    if not keep:
        return
    # If a subdir was specified, collapse the corpus root to that subdir so the
    # indexed paths are clean (e.g. "validators.py" not "pydantic/validators.py").
    src_base = spec.root / spec.subdir if spec.subdir else spec.root
    # Remove non-kept files first.
    for f in spec.root.rglob("*"):
        if f.is_file() and f not in keep:
            try:
                f.unlink()
            except OSError:
                pass
    # Remove now-empty dirs (deepest first), and drop .git entirely.
    git_dir = spec.root / ".git"
    if git_dir.exists():
        subprocess.run(["rm", "-rf", str(git_dir)], check=False)
    for d in sorted([p for p in spec.root.rglob("*") if p.is_dir()],
                    key=lambda p: len(p.parts), reverse=True):
        try:
            next(d.iterdir())
        except StopIteration:
            try:
                d.rmdir()
            except OSError:
                pass
    # If we restricted to a subdir, move its contents up to the corpus root.
    if spec.subdir and src_base.exists() and src_base != spec.root:
        for item in list(src_base.iterdir()):
            target = spec.root / item.name
            if not target.exists():
                item.rename(target)
        try:
            src_base.rmdir()
        except OSError:
            pass


def _already_built(spec: CorpusSpec) -> bool:
    """True if the corpus tree already has kept source files on disk."""
    return len(_source_files(spec)) > 0


def _clone(spec: CorpusSpec) -> str:
    """Shallow-clone a corpus repo and return the resolved commit SHA.
    Idempotent: if the tree already has kept files, skip the clone and resolve
    the SHA from the manifest if available (or 'reused' if the .git is gone)."""
    if spec.repo is None:
        return "reused-v1-corpus"
    spec.root.parent.mkdir(parents=True, exist_ok=True)
    if _already_built(spec):
        # Already pruned in a prior run; recover the pinned SHA from manifest.
        prior = _load_prior_manifest_sha(spec.name)
        return prior or "prebuilt"
    # Clean any partial dir before cloning.
    if spec.root.exists():
        subprocess.run(["rm", "-rf", str(spec.root)], check=False)
    cp = _git("clone", "--depth", "1", spec.repo, str(spec.root))
    if cp.returncode != 0:
        raise RuntimeError(f"git clone failed for {spec.name} ({spec.repo}): "
                           f"{cp.stderr.strip()[:300]}")
    rev = _git("rev-parse", "HEAD", cwd=spec.root)
    sha = rev.stdout.strip() if rev.returncode == 0 else "unknown"
    _prune_to_kept(spec)
    return sha


def _load_prior_manifest_sha(name: str) -> str | None:
    if not MANIFEST_PATH.exists():
        return None
    try:
        prior = json.loads(MANIFEST_PATH.read_text())
        for row in prior:
            if row.get("name") == name:
                return row.get("commit")
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _corpus_stats(spec: CorpusSpec) -> tuple[int, int]:
    """(n_files, est_tokens) over the kept source files."""
    files = _source_files(spec)
    chars = 0
    for f in files:
        try:
            chars += len(f.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return len(files), chars // CHARS_PER_TOKEN


def build_corpora(*, write_manifest: bool = True) -> list[dict]:
    """Ensure all corpora are present on disk (clone the missing ones, reuse the
    rest), prune cloned trees to source-only, and return the manifest:

        [{name, lang, root, ext, repo, commit, n_files, est_tokens}, ...]

    Idempotent and network-robust: an already-built corpus is never re-cloned.
    """
    manifest: list[dict] = []
    for spec in SPECS:
        commit = _clone(spec)
        n_files, est_tokens = _corpus_stats(spec)
        manifest.append({
            "name": spec.name,
            "lang": spec.lang,
            "root": str(spec.root),
            "ext": spec.ext,
            "repo": spec.repo,
            "commit": commit,
            "n_files": n_files,
            "est_tokens": est_tokens,
        })
    if write_manifest:
        RESULTS.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


def _print_table(manifest: list[dict]) -> None:
    print(f"\n{'name':5} {'lang':8} {'n_files':>8} {'est_tokens':>12}  {'root'}")
    print("-" * 78)
    for m in manifest:
        print(f"{m['name']:5} {m['lang']:8} {m['n_files']:>8} "
              f"{m['est_tokens']:>12,}  {m['root']}")
    tot_f = sum(m["n_files"] for m in manifest)
    tot_t = sum(m["est_tokens"] for m in manifest)
    print("-" * 78)
    print(f"{'TOTAL':5} {'':8} {tot_f:>8} {tot_t:>12,}")


if __name__ == "__main__":
    import sys

    print("[v2_corpora] building corpus set (clone missing, reuse present) ...",
          flush=True)
    manifest = build_corpora()
    _print_table(manifest)

    # ── self-test against REAL on-disk corpora ──────────────────────────
    failures: list[str] = []

    # 1. at least 3 corpora present, each with files on disk
    present = [m for m in manifest if m["n_files"] > 0]
    if len(present) < 3:
        failures.append(f"expected >=3 corpora with n_files>0, got {len(present)}")

    # 2. each corpus root really exists and holds its declared file count
    by_name = {m["name"]: m for m in manifest}
    for spec in SPECS:
        m = by_name[spec.name]
        if not Path(m["root"]).exists():
            failures.append(f"{spec.name}: root {m['root']} does not exist")
            continue
        actual = len(_source_files(spec))
        if actual != m["n_files"]:
            failures.append(f"{spec.name}: manifest n_files={m['n_files']} "
                            f"!= recount {actual}")
        if actual == 0:
            failures.append(f"{spec.name}: zero kept {spec.ext} files on disk")

    # 3. language variety — at least one Go corpus and at least two Python
    langs = [m["lang"] for m in present]
    if "go" not in langs:
        failures.append("no Go corpus present (lang variety requirement)")
    if langs.count("python") < 2:
        failures.append(f"need >=2 python corpora, got {langs.count('python')}")

    # 4. extension purity — every kept file in a corpus matches its ext
    for spec in SPECS:
        bad = [str(f) for f in _source_files(spec) if f.suffix != spec.ext]
        if bad:
            failures.append(f"{spec.name}: {len(bad)} files with wrong ext "
                            f"(e.g. {bad[0]})")

    # 5. cloned corpora must NOT retain a .git dir (commit is pinned in manifest)
    for spec in SPECS:
        if spec.repo and (spec.root / ".git").exists():
            failures.append(f"{spec.name}: .git dir still present after prune")

    # 6. cloned corpora must have a real pinned commit SHA (40 hex) or recovered
    for m in manifest:
        if m["repo"] is not None:
            c = m["commit"]
            ok = (len(c) == 40 and all(ch in "0123456789abcdef" for ch in c)) \
                or c in {"prebuilt", "unknown"}
            if not ok:
                failures.append(f"{m['name']}: commit '{c}' not a valid SHA")

    # 7. manifest file actually written and parses back to the same content
    if not MANIFEST_PATH.exists():
        failures.append(f"manifest not written to {MANIFEST_PATH}")
    else:
        reloaded = json.loads(MANIFEST_PATH.read_text())
        if reloaded != manifest:
            failures.append("written manifest != in-memory manifest")

    # 8. total file count in the tractable band (sanity, not a hard fail floor)
    tot = sum(m["n_files"] for m in manifest)
    print(f"\n[v2_corpora] total kept files: {tot}")

    print(f"\n[v2_corpora] manifest -> {MANIFEST_PATH}")
    if failures:
        print("\nSELF-TEST: FAIL")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nSELF-TEST: PASS  "
          f"({len(present)} corpora, {len(langs)} trees, "
          f"langs={sorted(set(langs))})")
    sys.exit(0)
