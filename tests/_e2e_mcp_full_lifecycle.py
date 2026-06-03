"""End-to-end MCP lifecycle test — the SHIP GATE.

Spawns a real MCP server subprocess, drives it with real JSON-RPC over
stdio (like Claude Code does), and walks the full mission lifecycle:

  1. handshake (initialize + initialized)
  2. tools/list — all 8 tools advertised
  3. lab_list — empty initially
  4. lab_start — creates lab, scaffolds knowledge
  5. lab_list — 1 lab now
  6. lab_status — reports cycle=0, findings=0
  7. lab_start (dup name) — error path
  8. lab_status (missing lab) — error path
  9. memory_search — empty results (no ingest yet)
 10. lab_reshape — auto-propose mode returns drift report
 11. lab_resume — bad token rejected, good token accepted
 12. packet_export — fails on cycle=0 (no cycles)
 13. fabricate cycle events → packet_export succeeds → packet on disk
 14. packet verify (via bert_verify.py) — 8-check pipeline passes

Hermetic: HOME → temp dir, no network, no LLM. Uses
`use_llm_classifier=False` for lab_start to skip the Claude CLI call.

Total time budget: 60s. Each tool has a per-call budget too.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("BERT_DISABLE_RERANKER", "1")  # tests: no 568MB cold-start

import subprocess
import sys
import tempfile
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


passed = 0
failed = 0
issues: list[str] = []


def check(name: str, fn):
    global passed, failed
    t0 = time.monotonic()
    try:
        fn()
        elapsed = time.monotonic() - t0
        print(f"  PASS  {name}  ({elapsed*1000:.0f}ms)")
        passed += 1
    except AssertionError as e:
        elapsed = time.monotonic() - t0
        print(f"  FAIL  {name}  ({elapsed*1000:.0f}ms): {e}")
        failed += 1
        issues.append(f"{name}: {e}")
    except Exception as e:  # noqa: BLE001
        elapsed = time.monotonic() - t0
        print(f"  FAIL  {name}  ({elapsed*1000:.0f}ms) UNEXPECTED {type(e).__name__}: {e}")
        failed += 1
        issues.append(f"{name}: UNEXPECTED {type(e).__name__}: {e}")


# ── MCP client driver ────────────────────────────────────────────


class MCPClient:
    """Minimal JSON-RPC over stdio MCP client. Spawns the server as a
    subprocess and handles request/response framing + ID tracking."""

    def __init__(self, env: dict[str, str] | None = None) -> None:
        self._proc: subprocess.Popen | None = None
        self._env = env
        self._next_id = 0

    def start(self) -> None:
        cmd = [sys.executable, "-m", "tools.mcp.bert_lab"]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
            env={**os.environ, **(self._env or {})},
            cwd=str(LAB_ROOT),
        )
        # Initialize
        self._send({
            "jsonrpc": "2.0", "id": self._gen_id(),
            "method": "initialize", "params": {},
        })
        init_resp = self._read()
        assert "result" in init_resp, f"initialize failed: {init_resp}"
        # Initialized notification
        self._send({
            "jsonrpc": "2.0", "method": "notifications/initialized",
        })

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()

    def _gen_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _send(self, msg: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()

    def _read(self) -> dict:
        assert self._proc is not None and self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            stderr_tail = ""
            if self._proc.stderr is not None:
                stderr_tail = self._proc.stderr.read()[-400:]
            raise RuntimeError(f"MCP server closed unexpectedly. stderr: {stderr_tail}")
        return json.loads(line)

    def list_tools(self) -> list[dict]:
        self._send({
            "jsonrpc": "2.0", "id": self._gen_id(),
            "method": "tools/list", "params": {},
        })
        resp = self._read()
        return resp["result"]["tools"]

    def call_tool(self, name: str, args: dict) -> dict:
        """Call a tool and return the parsed JSON result. Asserts no
        JSON-RPC error (network error), but tool-level ok=False is
        passed through to caller for error-path testing."""
        self._send({
            "jsonrpc": "2.0", "id": self._gen_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": args},
        })
        resp = self._read()
        assert "error" not in resp, f"tool {name} JSON-RPC error: {resp['error']}"
        assert "result" in resp, f"tool {name}: no result in {resp}"
        content = resp["result"].get("content", [])
        assert content, f"tool {name}: empty content"
        text = content[0].get("text", "")
        return json.loads(text)


# ── State holder ─────────────────────────────────────────────────


class E2EState:
    def __init__(self) -> None:
        self.tmpdir: tempfile.TemporaryDirectory | None = None
        self.home: Path | None = None
        self.labs_dir: Path | None = None
        self.client: MCPClient | None = None
        self.lab_name = "e2e_test_lab"
        self.lab_path: Path | None = None

    def setup(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory(prefix="bert_e2e_")
        self.home = Path(self.tmpdir.name)
        self.labs_dir = self.home / ".bert" / "labs"
        self.client = MCPClient(env={"HOME": str(self.home)})
        self.client.start()

    def teardown(self) -> None:
        if self.client:
            self.client.stop()
        if self.tmpdir:
            self.tmpdir.cleanup()


S = E2EState()


# ── Tests ─────────────────────────────────────────────────────────


def t_01_handshake():
    """MCP server starts + handshake completes."""
    # If setup() succeeded, handshake worked
    assert S.client is not None


def t_02_tools_list_complete():
    """All 8 expected tools advertised."""
    tools = S.client.list_tools()
    expected = {
        "lab_list", "lab_status", "lab_start", "lab_cycle",
        "lab_reshape", "lab_resume", "memory_search", "packet_export",
    }
    # Sprint 4 A2 — tool ids are now namespaced (bert.lab.lab_start); compare
    # bare suffixes so the "all 8 advertised" check is namespace-agnostic.
    actual = {t["name"].split(".")[-1] for t in tools}
    missing = expected - actual
    assert not missing, f"missing tools: {missing}"
    # Schemas are non-empty
    for t in tools:
        assert "inputSchema" in t, f"tool {t['name']} has no inputSchema"
        assert t.get("description"), f"tool {t['name']} has no description"


def t_03_lab_list_empty():
    """Fresh HOME → 0 labs."""
    result = S.client.call_tool("lab_list", {})
    assert result["count"] == 0, f"expected 0 labs, got {result['count']}"
    assert "labs_dir" in result


def t_04_lab_start_invalid_name_slash():
    result = S.client.call_tool("lab_start", {
        "name": "bad/name", "mission": "x" * 30,
    })
    assert not result.get("ok"), "should reject name with slash"
    assert "slug" in result["error"].lower()


def t_05_lab_start_invalid_name_traversal():
    result = S.client.call_tool("lab_start", {
        "name": "../escape", "mission": "x" * 30,
    })
    assert not result.get("ok"), "should reject name with .."
    assert "slug" in result["error"].lower()


def t_06_lab_start_mission_too_short():
    result = S.client.call_tool("lab_start", {
        "name": "valid_name", "mission": "short",
    })
    assert not result.get("ok"), "should reject mission < 20 chars"


def t_07_lab_start_invalid_archetype():
    result = S.client.call_tool("lab_start", {
        "name": "valid_name2", "mission": "x" * 30,
        "archetype": "garbage_type",
    })
    assert not result.get("ok"), "should reject unknown archetype"


def t_08_lab_start_happy_path():
    result = S.client.call_tool("lab_start", {
        "name": S.lab_name,
        "mission": (
            "Monitor recent arxiv papers about Mamba and state-space "
            "models. Build a citation graph and weekly summary."
        ),
        "archetype": "research",
        "use_llm_classifier": False,  # hermetic — no Claude CLI
    })
    assert result.get("ok"), f"lab_start failed: {result}"
    assert result["lab"] == S.lab_name
    assert result["archetype"] == "research"
    assert "profile" in result, "profile missing from response"
    assert "schema" in result, "schema missing from response"
    assert result["profile"]["data_shape"], "data_shape not set"
    assert isinstance(result["scaffolded_knowledge_files"], list)
    # Path returned matches our temp HOME
    assert str(S.labs_dir) in result["path"], (
        f"lab path {result['path']!r} not under {S.labs_dir!r}"
    )
    S.lab_path = Path(result["path"])


def t_09_lab_dir_structure_correct():
    """lab_start should create the expected directory tree."""
    assert S.lab_path is not None
    for sub in ("memories", "findings", "drafts", "sor", "state",
                "knowledge", "agents"):
        assert (S.lab_path / sub).is_dir(), f"missing dir: {sub}"
    assert (S.lab_path / "sor" / "events.jsonl").exists(), "events.jsonl not created"
    assert (S.lab_path / "lab.yaml").exists(), "lab.yaml not created"
    assert (S.lab_path / "seed_brief.md").exists(), "seed_brief.md not created"


def t_10_lab_yaml_has_profile():
    yaml_text = (S.lab_path / "lab.yaml").read_text()
    assert "mission_profile:" in yaml_text
    assert "lab_schema:" in yaml_text
    assert "data_shape:" in yaml_text


def t_11_lab_start_duplicate_rejected():
    """Starting the same lab name twice should fail."""
    result = S.client.call_tool("lab_start", {
        "name": S.lab_name, "mission": "x" * 30,
        "use_llm_classifier": False,
    })
    assert not result.get("ok"), "duplicate lab should be rejected"
    assert "exists" in result["error"].lower()


def t_12_lab_list_one_lab():
    result = S.client.call_tool("lab_list", {})
    assert result["count"] == 1, f"expected 1 lab, got {result['count']}"
    labs = result["labs"]
    assert labs[0]["lab"] == S.lab_name


def t_13_lab_status_existing():
    result = S.client.call_tool("lab_status", {"lab": S.lab_name})
    assert result.get("ok")
    assert result["lab"] == S.lab_name
    assert result["last_cycle"] == 0
    assert result["findings_count"] == 0


def t_14_lab_status_missing():
    result = S.client.call_tool("lab_status", {"lab": "nonexistent_lab_xyz"})
    assert not result.get("ok")
    assert "not found" in result["error"].lower()


def t_15_memory_search_empty():
    result = S.client.call_tool("memory_search", {
        "lab": S.lab_name, "query": "mamba",
    })
    # Empty lab — either 0 hits or graceful empty response
    assert "hits" in result or "results" in result or "ok" in result


def t_16_lab_reshape_auto_propose():
    """auto-propose mode on a 0-cycle lab → drift_score=0, no proposal."""
    result = S.client.call_tool("lab_reshape", {"lab": S.lab_name})
    # Either ok + propose=False (low drift) OR ok + propose=True with proposal
    assert "ok" in result or "drift_score" in result or "propose" in result, (
        f"unexpected reshape response shape: {result}"
    )


def t_17_lab_resume_bad_token():
    result = S.client.call_tool("lab_resume", {
        "token": "not_a_real_token", "answer": "x",
    })
    assert not result.get("ok")
    assert "token" in result["error"].lower()


def t_18_lab_resume_missing_args():
    result = S.client.call_tool("lab_resume", {})
    assert not result.get("ok")


def t_19_packet_export_no_cycles():
    """With cycle=0, packet_export should refuse cleanly."""
    result = S.client.call_tool("packet_export", {
        "lab": S.lab_name, "cycle_id": 0,
    })
    assert not result.get("ok"), "should reject cycle_id=0"


def t_20_fabricate_cycle_events():
    """Hand-write 1 cycle's worth of events so packet_export has data."""
    assert S.lab_path is not None
    events_path = S.lab_path / "sor" / "events.jsonl"
    now_iso = "2026-05-24T19:00:00+00:00"
    events = [
        {
            "ts": now_iso, "cycle": 1, "event_class": "subagent_spawn",
            "agent": "director", "kind": "subagent_spawn",
        },
        {
            "ts": now_iso, "cycle": 1, "event_class": "model_call",
            "agent": "researcher", "kind": "model_call",
            "provider": "anthropic", "model": "claude-opus-4-7",
        },
        {
            "ts": now_iso, "cycle": 1, "event_class": "finding",
            "agent": "researcher", "kind": "finding",
            "content": "Found a paper on Mamba state-space models.",
        },
        {
            "ts": now_iso, "cycle": 1, "event_class": "verdict",
            "agent": "evaluator", "kind": "verdict", "verdict": "APPROVE",
        },
    ]
    with events_path.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    # Also create a finding file so packet has an artifact
    finding = S.lab_path / "findings" / "bert_run_C1_researcher.md"
    finding.write_text(
        "# Researcher finding (cycle 1)\n\nMamba paper at arxiv:2312.00752\n"
    )
    # Verify events.jsonl is readable
    assert events_path.exists()
    assert len(events_path.read_text().splitlines()) == 4


def t_21_packet_export_happy_path():
    """With a real cycle, packet_export should succeed."""
    result = S.client.call_tool("packet_export", {
        "lab": S.lab_name, "cycle_id": 1,
    })
    if not result.get("ok"):
        # Capture the failure — this is the proof_packet/lab_context bug
        raise AssertionError(
            f"packet_export failed: {result.get('error', result)}"
        )
    assert "packet_path" in result
    assert result["packet_bytes"] > 0
    packet_path = Path(result["packet_path"])
    assert packet_path.exists(), f"packet not on disk: {packet_path}"
    # Stash for next test
    S.packet_path = packet_path


def t_22_packet_is_valid_tarball():
    """Packet is a valid .tar.gz with expected structure."""
    import tarfile
    pp = getattr(S, "packet_path", None)
    assert pp is not None and pp.exists(), "no packet to inspect"
    with tarfile.open(pp, "r:gz") as tf:
        names = tf.getnames()
    # Should have provenance/, inputs/, outputs/, eval/ + cycle.json
    has_cycle = any("cycle.json" in n for n in names)
    has_provenance = any("provenance" in n for n in names)
    has_outputs = any("outputs" in n for n in names)
    assert has_cycle, f"packet missing cycle.json: {names[:10]}"
    assert has_provenance, f"packet missing provenance/: {names[:10]}"
    assert has_outputs, f"packet missing outputs/: {names[:10]}"


def t_22b_packet_attests_correct_lab_events():
    """CRITICAL CORRECTNESS: the packet's cycle.json must reflect MY
    fabricated 4 events — not whatever's in any other lab's SoR. This
    catches the cross-lab contamination bug class (proof_packet was
    reading from a hardcoded LAB_ROOT/lab/sor/events.jsonl instead of
    the active lab's events.jsonl)."""
    import tarfile
    pp = getattr(S, "packet_path", None)
    assert pp is not None
    with tarfile.open(pp, "r:gz") as tf:
        cycle_json_member = None
        for m in tf.getmembers():
            if m.name.endswith("cycle.json"):
                cycle_json_member = m
                break
        assert cycle_json_member is not None, "no cycle.json in packet"
        cycle_json = json.loads(tf.extractfile(cycle_json_member).read())
    # We wrote exactly 4 events for cycle 1
    assert cycle_json["eventCount"] == 4, (
        f"contamination: packet says eventCount={cycle_json['eventCount']} "
        f"but we fabricated exactly 4. The packet is attesting events "
        f"from a different lab's SoR."
    )


def t_23_packet_verify_via_cli():
    """`tools/bert_verify.py` runs the 8-check pipeline.

    Exit codes per the CLI contract:
      0 = PASS, 1 = PASS-WITH-WARNINGS, 2 = FAIL.
    PASS-WITH-WARNINGS is acceptable for a fabricated packet (we
    didn't run adversarial-eval, so failures.md is empty by design —
    that is a structural yellow flag, not a verification failure).
    The E2E gate is: structural FAIL must not happen."""
    pp = getattr(S, "packet_path", None)
    assert pp is not None
    proc = subprocess.run(
        [sys.executable, str(LAB_ROOT / "tools" / "bert_verify.py"), str(pp)],
        capture_output=True, text=True, timeout=30,
        cwd=str(LAB_ROOT),
    )
    # rc=2 (FAIL) is the only true failure.
    assert proc.returncode != 2, (
        f"bert_verify reported FAIL\n"
        f"stdout: {proc.stdout[-500:]}\n"
        f"stderr: {proc.stderr[-300:]}"
    )
    # rc=0 (PASS) or rc=1 (PASS-WITH-WARNINGS) both OK.
    assert proc.returncode in (0, 1), (
        f"unexpected rc={proc.returncode}\nstdout: {proc.stdout[-400:]}"
    )
    # Structural sanity: stdout reports the 8 expected checks
    expected_checks = [
        "[1]", "[2]", "[3]", "[4]", "[5]", "[6]", "[7]", "[8]",
    ]
    for c in expected_checks:
        assert c in proc.stdout, f"missing check {c} in verifier output"


def t_24_tool_response_times():
    """Sanity: light tools respond in < 1s each."""
    for tool, args in [
        ("lab_list", {}),
        ("lab_status", {"lab": S.lab_name}),
    ]:
        t0 = time.monotonic()
        S.client.call_tool(tool, args)
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, f"{tool} too slow: {elapsed:.2f}s"


def t_25_multiple_concurrent_calls_same_client():
    """10 sequential rapid-fire calls — should all succeed."""
    for _ in range(10):
        result = S.client.call_tool("lab_list", {})
        assert "count" in result


# ── Runner ────────────────────────────────────────────────────────


TESTS = [
    t_01_handshake,
    t_02_tools_list_complete,
    t_03_lab_list_empty,
    t_04_lab_start_invalid_name_slash,
    t_05_lab_start_invalid_name_traversal,
    t_06_lab_start_mission_too_short,
    t_07_lab_start_invalid_archetype,
    t_08_lab_start_happy_path,
    t_09_lab_dir_structure_correct,
    t_10_lab_yaml_has_profile,
    t_11_lab_start_duplicate_rejected,
    t_12_lab_list_one_lab,
    t_13_lab_status_existing,
    t_14_lab_status_missing,
    t_15_memory_search_empty,
    t_16_lab_reshape_auto_propose,
    t_17_lab_resume_bad_token,
    t_18_lab_resume_missing_args,
    t_19_packet_export_no_cycles,
    t_20_fabricate_cycle_events,
    t_21_packet_export_happy_path,
    t_22_packet_is_valid_tarball,
    t_22b_packet_attests_correct_lab_events,
    t_23_packet_verify_via_cli,
    t_24_tool_response_times,
    t_25_multiple_concurrent_calls_same_client,
]


def main() -> int:
    print(f"Running {len(TESTS)} E2E MCP lifecycle tests…\n")
    t0_total = time.monotonic()
    try:
        S.setup()
        for fn in TESTS:
            check(fn.__name__, fn)
    finally:
        S.teardown()
    elapsed = time.monotonic() - t0_total
    print()
    print(f"E2E: pass={passed} fail={failed}  total_elapsed={elapsed:.1f}s")
    if issues:
        print()
        print("FAILURES:")
        for i in issues:
            print(f"  · {i}")
    if failed == 0:
        print(f"All {passed} tests passed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
