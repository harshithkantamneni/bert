"""50-cycle soak + resource-leak detection.

Hammers the MCP+CLI surface in a long-running pattern that mimics how
a busy lab would actually run over a week. We're not testing
correctness here (the other suites cover that) — we're testing that
the system survives 50× the unit-test load WITHOUT:

  - RSS memory growth past a budget (linear or unbounded → leak)
  - File-descriptor growth (open SQLite + open subprocess pipes that
    don't get closed are the most common silent killers)
  - Child process zombification (subprocess.Popen + .terminate but
    no .wait → zombies that pile up under load)
  - Wall-clock per-call regression (p99 budget per tool call)
  - Per-cycle subprocess fork explosion

The MCP server is spawned ONCE and reused across calls (that matches
production: Claude Code keeps the bert MCP server alive for the
whole session, the partner doesn't fork-per-call). Per-call latency
must stay flat — if p99 climbs as N grows, we have linear cleanup
debt.

Hermetic: HOME → temp dir; no LLM; no network.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("BERT_DISABLE_RERANKER", "1")

import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


passed = 0
failed = 0


def check(name: str, fn):
    global passed, failed
    t0 = time.monotonic()
    try:
        fn()
        print(f"  PASS  {name}  ({(time.monotonic()-t0)*1000:.0f}ms)")
        passed += 1
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        failed += 1
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL  {name} UNEXPECTED {type(e).__name__}: {e}")
        failed += 1


def _rss_mb(pid: int | None = None) -> float:
    """Resident set size in MB. None = self."""
    if pid is None:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # On Darwin, ru_maxrss is in BYTES; on Linux, in KB.
        if sys.platform == "darwin":
            return ru.ru_maxrss / (1024 * 1024)
        return ru.ru_maxrss / 1024
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip()) / 1024  # KB → MB
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def _open_fds(pid: int | None = None) -> int:
    """Approximate count of open file descriptors for a process."""
    target = pid if pid is not None else os.getpid()
    try:
        out = subprocess.run(
            ["lsof", "-p", str(target)],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return len(out.stdout.strip().splitlines()) - 1  # minus header
    except Exception:  # noqa: BLE001
        pass
    return -1  # unknown


def _child_proc_count() -> int:
    """Number of direct child processes of this process."""
    try:
        out = subprocess.run(
            ["pgrep", "-P", str(os.getpid())],
            capture_output=True, text=True, timeout=2,
        )
        return len(out.stdout.strip().splitlines()) if out.stdout.strip() else 0
    except Exception:  # noqa: BLE001
        return -1


# ── MCP session driver ───────────────────────────────────────────


class PersistentMCP:
    """A single MCP server that stays up across many calls."""

    def __init__(self, env: dict[str, str]) -> None:
        self._proc: subprocess.Popen | None = None
        self._env = env
        self._next_id = 0

    def start(self) -> None:
        cmd = [sys.executable, "-m", "tools.mcp.bert_lab"]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
            env={**os.environ, **self._env}, cwd=str(LAB_ROOT),
        )
        self._send({"jsonrpc": "2.0", "id": self._gen_id(),
                    "method": "initialize", "params": {}})
        self._recv()
        self._send({"jsonrpc": "2.0",
                    "method": "notifications/initialized"})

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()

    @property
    def pid(self) -> int:
        return self._proc.pid if self._proc else -1

    def _gen_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _send(self, msg: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()

    def _recv(self) -> dict:
        assert self._proc is not None and self._proc.stdout is not None
        line = self._proc.stdout.readline()
        return json.loads(line)

    def call(self, tool: str, args: dict) -> dict:
        self._send({
            "jsonrpc": "2.0", "id": self._gen_id(),
            "method": "tools/call",
            "params": {"name": tool, "arguments": args},
        })
        resp = self._recv()
        text = resp["result"]["content"][0]["text"]
        return json.loads(text)


# ── State ─────────────────────────────────────────────────────────


class SoakState:
    def __init__(self) -> None:
        self.tmpdir: tempfile.TemporaryDirectory | None = None
        self.home: Path | None = None
        self.mcp: PersistentMCP | None = None
        self.lab_name = "soak_lab"
        self.latencies_ms: list[float] = []
        self.rss_samples: list[tuple[int, float]] = []  # (iteration, rss_mb)
        self.fd_samples: list[tuple[int, int]] = []
        self.child_samples: list[tuple[int, int]] = []

    def setup(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory(prefix="bert_soak_")
        self.home = Path(self.tmpdir.name)
        self.mcp = PersistentMCP(env={"HOME": str(self.home)})
        self.mcp.start()

    def teardown(self) -> None:
        if self.mcp:
            self.mcp.stop()
        if self.tmpdir:
            self.tmpdir.cleanup()


S = SoakState()


# ── Tests ─────────────────────────────────────────────────────────


N_CYCLES = 50


def t_01_setup_lab():
    """Create the soak lab once; everything else operates on it."""
    result = S.mcp.call("lab_start", {
        "name": S.lab_name,
        "mission": (
            "Soak test mission — exercise the surface for 50 calls. "
            "No real LLM activity, just hermetic state mutation."
        ),
        "use_llm_classifier": False,
    })
    assert result.get("ok"), f"lab_start failed: {result}"


def t_02_run_50_calls_no_resource_leak():
    """Hammer the MCP server 50× with a rotating set of read ops.
    Sample resource usage every 10 iterations. Assert bounds."""
    tools_rotation = [
        ("lab_list", {}),
        ("lab_status", {"lab": S.lab_name}),
        ("memory_search", {"lab": S.lab_name, "query": "test"}),
        ("lab_list", {"prefix": "soak"}),
        ("lab_reshape", {"lab": S.lab_name}),
    ]
    # Baseline
    S.rss_samples.append((0, _rss_mb(S.mcp.pid)))
    S.fd_samples.append((0, _open_fds(S.mcp.pid)))
    S.child_samples.append((0, _child_proc_count()))

    for i in range(N_CYCLES):
        tool, args = tools_rotation[i % len(tools_rotation)]
        t0 = time.monotonic()
        S.mcp.call(tool, args)
        S.latencies_ms.append((time.monotonic() - t0) * 1000)
        # Every 10 calls, sample resource usage
        if (i + 1) % 10 == 0:
            S.rss_samples.append((i + 1, _rss_mb(S.mcp.pid)))
            S.fd_samples.append((i + 1, _open_fds(S.mcp.pid)))
            S.child_samples.append((i + 1, _child_proc_count()))

    assert len(S.latencies_ms) == N_CYCLES
    print(f"      ↳ ran {N_CYCLES} calls, latencies recorded")


def t_03_latency_distribution():
    """p50, p95, p99 budget check."""
    lats = sorted(S.latencies_ms)
    p50 = lats[len(lats) // 2]
    p95 = lats[int(len(lats) * 0.95)]
    p99 = lats[int(len(lats) * 0.99)]
    mean = statistics.mean(lats)
    print(f"      ↳ latency p50={p50:.1f}ms p95={p95:.1f}ms p99={p99:.1f}ms "
          f"mean={mean:.1f}ms max={max(lats):.1f}ms")
    # Budgets for stdio JSON-RPC on local subprocess
    assert p50 < 50, f"p50 latency too high: {p50:.1f}ms"
    assert p95 < 200, f"p95 latency too high: {p95:.1f}ms"
    assert p99 < 500, f"p99 latency too high: {p99:.1f}ms"


def t_04_rss_bounded():
    """MCP server RSS should not grow unboundedly across 50 calls."""
    if not S.rss_samples or S.rss_samples[0][1] == 0:
        print("      ↳ skipped: rss measurement unavailable")
        return
    iters, rss = zip(*S.rss_samples, strict=False)
    delta = rss[-1] - rss[0]
    print("      ↳ RSS samples: " + " ".join(
        f"i{i}={r:.1f}MB" for i, r in S.rss_samples
    ))
    print(f"      ↳ growth: {delta:+.1f}MB over {iters[-1]} calls")
    # MCP server is light; 50MB growth is generous
    assert delta < 50, f"RSS grew by {delta:.1f}MB — leak suspected"


def t_05_fd_count_stable():
    """File descriptor count should stabilize, not grow per call."""
    iters, fds = zip(*S.fd_samples, strict=False)
    if any(f < 0 for f in fds):
        print("      ↳ skipped: fd measurement unavailable")
        return
    delta = fds[-1] - fds[0]
    print("      ↳ FD samples: " + " ".join(
        f"i{i}={f}" for i, f in S.fd_samples
    ))
    # Tolerate +5 fds for warmup; reject +50 as leak indicator
    assert delta < 50, f"FD count grew by {delta} — leak suspected"


def t_06_no_zombie_children():
    """We spawned ONE MCP child process. Confirm no extras leaked."""
    children = _child_proc_count()
    print(f"      ↳ direct children: {children} (expected 1: MCP server)")
    # Expect exactly 1 (the MCP server). Tolerate 0-2 for race
    # with sampling subprocesses (lsof/ps fired by t_02).
    assert 0 <= children <= 3, (
        f"unexpected child count: {children} — zombies?"
    )


def t_07_adapter_100_ingests_no_corruption():
    """Hammer the code adapter with 100 ingest/search cycles on a
    single lab — verify final state is consistent."""
    from core.memory_adapters import find_adapter_for_shape
    cls = find_adapter_for_shape("code_repo")
    with tempfile.TemporaryDirectory() as tmp:
        lab = Path(tmp)
        ad = cls(lab)
        src = lab / "src"
        src.mkdir()
        for i in range(100):
            (src / f"f_{i}.py").write_text(f"def func_{i}(): return {i}\n")
            ad.ingest(src / f"f_{i}.py")
        st = ad.stats()
        # 100 files, 100 functions
        assert st.items_total == 100, (
            f"expected 100 symbols, got {st.items_total}"
        )


def t_08_migrations_50_labs_no_handle_leak():
    """Apply migrations to 50 fresh labs. Each opens + closes its
    own SQLite handles. Verify no fd leak across all 50."""
    from core import migrations
    fds_before = _open_fds() if _open_fds() > 0 else -1
    for _i in range(50):
        with tempfile.TemporaryDirectory() as tmp:
            lab = Path(tmp)
            r = migrations.apply_pending(lab, "document_corpus")
            assert not r.errors
    fds_after = _open_fds() if _open_fds() > 0 else -1
    if fds_before > 0 and fds_after > 0:
        delta = fds_after - fds_before
        print(f"      ↳ FD delta after 50 migrations: {delta:+d}")
        # We can leak a few fds to OS caching but not 50
        assert delta < 30, f"migration FD leak: +{delta}"
    else:
        print("      ↳ FD measurement skipped (lsof unavailable)")


def t_09_hybrid_retrieval_throughput():
    """100 hybrid_retrieve calls on the project's lab — confirm
    median latency is sane (sub-second).

    Warm-up: the embedder model (bge-base-en-v1.5, ~440 MB) cold-loads
    on first call and adds ~30-60s. That's a production cold-start
    concern (the server should pre-warm at boot), not a per-call
    regression. We discard the first call's latency to measure the
    *hot* path that actually serves user traffic."""
    from core import retrieval as _ret
    # Warm-up call (cold-load embedder + open SQLite handles)
    _ret.hybrid_retrieve("warmup", top_n=5)
    latencies = []
    for _ in range(100):
        t0 = time.monotonic()
        _ret.hybrid_retrieve("test query", top_n=5)
        latencies.append((time.monotonic() - t0) * 1000)
    lats = sorted(latencies)
    p50, p95, p99 = lats[50], lats[95], lats[99]
    print(f"      ↳ retrieval (hot) p50={p50:.1f}ms p95={p95:.1f}ms p99={p99:.1f}ms")
    assert p50 < 500, f"retrieval p50 too high: {p50:.1f}ms"
    # Hot p99 must be stable — no slow tail under steady load
    assert p99 < 2000, f"retrieval p99 too high: {p99:.1f}ms"


def t_10_classifier_throughput():
    """100 default_profile calls. Should be < 1ms each on heuristics."""
    from core import mission_profile
    t0 = time.monotonic()
    for _ in range(100):
        mission_profile.default_profile("test mission with some words")
    elapsed = (time.monotonic() - t0) * 1000
    per_call = elapsed / 100
    print(f"      ↳ {per_call:.2f}ms / call (100 total)")
    assert per_call < 5, f"classifier too slow: {per_call:.2f}ms/call"


# ── Runner ────────────────────────────────────────────────────────


TESTS = [
    t_01_setup_lab,
    t_02_run_50_calls_no_resource_leak,
    t_03_latency_distribution,
    t_04_rss_bounded,
    t_05_fd_count_stable,
    t_06_no_zombie_children,
    t_07_adapter_100_ingests_no_corruption,
    t_08_migrations_50_labs_no_handle_leak,
    t_09_hybrid_retrieval_throughput,
    t_10_classifier_throughput,
]


def main() -> int:
    print(f"Running {len(TESTS)} soak + resource tests "
          f"({N_CYCLES} MCP calls)…\n")
    t0_total = time.monotonic()
    try:
        S.setup()
        for fn in TESTS:
            check(fn.__name__, fn)
    finally:
        S.teardown()
    elapsed = time.monotonic() - t0_total
    print()
    print(f"Soak: pass={passed} fail={failed}  elapsed={elapsed:.1f}s")
    if failed == 0:
        print(f"All {passed} tests passed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
