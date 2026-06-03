"""Smoke: core/tools.py — the agent's built-in tool implementations (was 34%).

Drives every handler network-free:
  - Read/Write/Edit against absolute temp paths (_resolve_relative_path
    returns absolutes as-is, so no active lab needed)
  - the findings auto-emit branch (get_active_lab_path monkeypatched to a
    temp lab so the event lands in a temp sor/events.jsonl, not the repo)
  - _bash real exit codes + the unimplemented-sandbox + timeout branches
  - the pure HTML/DDG parsers (_clean_html, _parse_ddg_results,
    _unwrap_ddg_url) on fixtures
  - _webfetch / _websearch with a faked httpx.Client (html/json/404/empty)
  - _memory_search hybrid-success + vector-fallback, _memory_create +
    _spawn delegation (core.retrieval / core.memory / core.subagent stubbed)
"""

from __future__ import annotations

import inspect
import sys
import types
import urllib.parse
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import tools  # noqa: E402


class _MP:
    def __init__(self):
        self._u = []
    def setattr(self, obj, name, val):
        self._u.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)
    def undo(self):
        for o, n, v in reversed(self._u):
            setattr(o, n, v)
        self._u.clear()


class _FakeResp:
    def __init__(self, text="", status=200, ctype="text/html", url="https://x.test/"):
        self.text = text
        self.status_code = status
        self.headers = {"content-type": ctype}
        self.url = url


class _FakeClient:
    def __init__(self, resp):
        self._r = resp
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def get(self, *a, **k):
        return self._r
    def post(self, *a, **k):
        return self._r


def _fake_httpx(resp):
    return lambda *a, **k: _FakeClient(resp)


# ── file ops ──────────────────────────────────────────────────────────

def test_resolve_relative_path():
    abs_p = LAB_ROOT / "x" / "y.md"
    assert tools._resolve_relative_path(str(abs_p)) == abs_p
    # relative + no active lab → under LAB_ROOT
    assert tools._resolve_relative_path("drafts/z.md") == LAB_ROOT / "drafts" / "z.md"


def test_read_write_edit(tmp_path):
    f = tmp_path / "doc.md"
    assert "wrote" in tools._write(str(f), "line1\nline2\nline3\n")
    assert f.read_text() == "line1\nline2\nline3\n"
    assert tools._read(str(f)) == "line1\nline2\nline3\n"
    assert tools._read(str(tmp_path / "missing.md")) == ""  # empty-state guarantee
    # offset/limit slice
    assert tools._read(str(f), offset=2, limit=1) == "line2\n"
    # edit: happy path
    r = tools._edit(str(f), "line2", "LINE2")
    assert r["ok"] and r["replacements"] == 1
    assert "LINE2" in f.read_text()
    # edit: identical no-op
    assert tools._edit(str(f), "x", "x")["ok"] is False
    # edit: missing file
    assert tools._edit(str(tmp_path / "nope.md"), "a", "b")["ok"] is False
    # edit: not found
    assert tools._edit(str(f), "zzz-not-present", "q")["ok"] is False
    # edit: multiple occurrences requires replace_all
    f.write_text("dup dup dup\n")
    assert tools._edit(str(f), "dup", "X")["ok"] is False
    r_all = tools._edit(str(f), "dup", "X", replace_all=True)
    assert r_all["ok"] and r_all["replacements"] == 3


def test_write_emits_finding_event(monkeypatch, tmp_path):
    lab = tmp_path / "lab"
    monkeypatch.setattr(tools, "get_active_lab_path", lambda: lab)
    fpath = lab / "findings" / "survey_C7_researcher.md"
    msg = tools._write(str(fpath), "# Heading\n\nThe real finding paragraph here.\n")
    assert "wrote" in msg
    events = lab / "sor" / "events.jsonl"
    assert events.exists(), "findings write should emit a finding event"
    assert "finding" in events.read_text()


def test_finding_inference_helpers():
    assert tools._infer_cycle("bert_run_C5_researcher.md") == 5
    assert tools._infer_cycle("no_cycle.md") is None
    assert tools._infer_agent("survey_C7_researcher.md") == "researcher"
    assert tools._first_paragraph("# Title\n\nFirst real para.\n\nSecond.") == "First real para."


# ── bash ──────────────────────────────────────────────────────────────

def test_bash_exit_codes():
    ok = tools._bash("echo hi")
    assert ok["exit_code"] == 0 and "hi" in ok["stdout"]
    bad = tools._bash("exit 3")
    assert bad["exit_code"] == 3
    # unimplemented sandbox tier
    nope = tools._bash("echo x", sandbox="docker")
    assert nope["exit_code"] == 2
    # timeout path (bounded to ~1s)
    to = tools._bash("sleep 3", timeout=1)
    assert to["exit_code"] == 124


# ── html / search parsers ─────────────────────────────────────────────

def test_clean_html():
    html = ("<html><head><title>My Title</title></head>"
            "<body><script>junk()</script><main><p>Real body text.</p></main>"
            "<footer>nav junk</footer></body></html>")
    title, text = tools._clean_html(html)
    assert title == "My Title"
    assert "Real body text." in text and "junk" not in text


def test_unwrap_ddg_url():
    enc = urllib.parse.quote("https://example.com/page?x=1", safe="")
    href = f"//duckduckgo.com/l/?uddg={enc}&rut=abc"
    assert tools._unwrap_ddg_url(href) == "https://example.com/page?x=1"
    # bare absolute href passes through
    assert tools._unwrap_ddg_url("https://plain.example/").startswith("https://")


def test_parse_ddg_results():
    enc = urllib.parse.quote("https://result.example/a", safe="")
    html = (
        '<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg='
        + enc + '">Result Title</a>'
        '<div class="result__snippet">A snippet.</div></div>'
    )
    results = tools._parse_ddg_results(html, max_results=5)
    assert results and results[0]["title"] == "Result Title"
    assert results[0]["url"] == "https://result.example/a"


# ── webfetch / websearch (faked httpx) ────────────────────────────────

def test_webfetch_html_json_and_error(monkeypatch):
    monkeypatch.setattr(tools.httpx, "Client",
                        _fake_httpx(_FakeResp("<html><title>T</title><main><p>Body</p></main></html>")))
    r = tools._webfetch("https://x.test/")
    assert r["ok"] and r["title"] == "T" and "Body" in r["content"]
    # json content-type
    monkeypatch.setattr(tools.httpx, "Client",
                        _fake_httpx(_FakeResp('{"a":1}', ctype="application/json")))
    rj = tools._webfetch("https://x.test/api")
    assert rj["ok"] and rj["content"] == '{"a":1}'
    # http error
    monkeypatch.setattr(tools.httpx, "Client", _fake_httpx(_FakeResp("nope", status=404)))
    re_ = tools._webfetch("https://x.test/missing")
    assert re_["ok"] is False and "404" in re_["error"]


def test_websearch_results_and_empty(monkeypatch):
    enc = urllib.parse.quote("https://r.example/a", safe="")
    good = ('<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg='
            + enc + '">Hit</a><div class="result__snippet">snip</div></div>')
    monkeypatch.setattr(tools.httpx, "Client", _fake_httpx(_FakeResp(good)))
    r = tools._websearch("query terms", max_results=3)
    assert r["ok"] and r["results"][0]["title"] == "Hit"
    # empty/unparsable
    monkeypatch.setattr(tools.httpx, "Client", _fake_httpx(_FakeResp("<html>no results</html>")))
    empty = tools._websearch("q")
    assert empty["ok"] is False


# ── memory_search / memory_create / spawn (stubbed deps) ──────────────

def test_memory_search_hybrid_and_fallback(monkeypatch):
    from core import retrieval as _ret
    fake_hit = types.SimpleNamespace(
        metadata={"path": "findings/x.md", "chunk_idx": 0},
        text="hybrid chunk", final_score=0.9, sources=["vector", "bm25"])
    monkeypatch.setattr(_ret, "hybrid_retrieve", lambda *a, **k: [fake_hit])
    r = tools._memory_search("vector db", k=3)
    assert r["ok"] and r["method"] == "hybrid" and r["hits"][0]["content"] == "hybrid chunk"
    # hybrid raises → vector fallback
    def _boom(*a, **k):
        raise RuntimeError("hybrid down")
    monkeypatch.setattr(_ret, "hybrid_retrieve", _boom)
    from core import memory as _mem
    monkeypatch.setattr(_mem, "search", lambda q, k=5: [{"path": "m.md", "content": "v"}])
    rf = tools._memory_search("q", k=2)
    assert rf["ok"] and rf["method"] == "vector_fallback"


def test_memory_create_and_spawn_delegate(monkeypatch):
    from core import memory as _mem
    monkeypatch.setattr(_mem, "create", lambda p, c: {"ok": True, "path": p, "bytes": len(c)})
    assert tools._memory_create("findings/x.md", "body")["ok"]
    from core import subagent as _sa
    monkeypatch.setattr(_sa, "run_subagent", lambda spec: {"ok": True, "summary": "done"})
    assert tools._spawn({"role": "researcher"})["ok"]


class _RaisingClient:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def get(self, *a, **k):
        import httpx as _h
        raise _h.TimeoutException("slow")
    def post(self, *a, **k):
        import httpx as _h
        raise _h.NetworkError("down")


def test_resolve_relative_path_active_lab(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "get_active_lab_path", lambda: tmp_path)
    assert tools._resolve_relative_path("drafts/z.md") == tmp_path / "drafts" / "z.md"


def test_read_directory_error(tmp_path):
    # reading a directory → caught IsADirectoryError branch
    out = tools._read(str(tmp_path))
    assert out.startswith("[Read error")


def test_emit_skip_branches(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "get_active_lab_path", lambda: tmp_path)
    # non-.md → no emit
    tools._write(str(tmp_path / "findings" / "note.txt"), "x")
    # archived finding → skipped
    tools._write(str(tmp_path / "findings" / "archive" / "old_C1_researcher.md"), "x")
    # nested findings subdir → skipped (keep flat)
    tools._write(str(tmp_path / "findings" / "2026-05" / "deep.md"), "x")
    assert not (tmp_path / "sor" / "events.jsonl").exists()


def test_first_para_and_agent_edges():
    # all-heading block is skipped, then falls through to stripped text
    assert tools._first_paragraph("# only headings\n## still heading") != ""
    assert tools._infer_agent("file_without_role.md") is None


def test_webfetch_text_and_timeout(monkeypatch):
    big = "z" * 40
    monkeypatch.setattr(tools.httpx, "Client",
                        _fake_httpx(_FakeResp(big, ctype="text/plain")))
    r = tools._webfetch("https://x.test/t.txt", max_chars=10)
    assert r["ok"] and r["truncated"] is True
    monkeypatch.setattr(tools.httpx, "Client", lambda *a, **k: _RaisingClient())
    rt = tools._webfetch("https://x.test/slow")
    assert rt["ok"] is False and "TimeoutException" in rt["error"]


def test_websearch_404_and_timeout(monkeypatch):
    monkeypatch.setattr(tools.httpx, "Client", _fake_httpx(_FakeResp("err", status=503)))
    r = tools._websearch("q")
    assert r["ok"] is False and "503" in r["error"]
    monkeypatch.setattr(tools.httpx, "Client", lambda *a, **k: _RaisingClient())
    rt = tools._websearch("q")
    assert rt["ok"] is False and "NetworkError" in rt["error"]


def test_memory_search_bad_k_coercion(monkeypatch):
    from core import retrieval as _ret
    monkeypatch.setattr(_ret, "hybrid_retrieve", lambda *a, **k: [])
    r = tools._memory_search("q", k="not-an-int")  # type: ignore[arg-type]
    assert r["ok"] and r["hits"] == []


def main() -> int:
    import shutil
    import tempfile
    tests = [
        test_resolve_relative_path,
        test_read_write_edit,
        test_write_emits_finding_event,
        test_finding_inference_helpers,
        test_bash_exit_codes,
        test_clean_html,
        test_unwrap_ddg_url,
        test_parse_ddg_results,
        test_webfetch_html_json_and_error,
        test_websearch_results_and_empty,
        test_memory_search_hybrid_and_fallback,
        test_memory_create_and_spawn_delegate,
        test_resolve_relative_path_active_lab,
        test_read_directory_error,
        test_emit_skip_branches,
        test_first_para_and_agent_edges,
        test_webfetch_text_and_timeout,
        test_websearch_404_and_timeout,
        test_memory_search_bad_k_coercion,
    ]
    for t in tests:
        mp = _MP()
        td = Path(tempfile.mkdtemp())
        try:
            params = inspect.signature(t).parameters
            kwargs = {}
            if "tmp_path" in params:
                kwargs["tmp_path"] = td
            if "monkeypatch" in params:
                kwargs["monkeypatch"] = mp
            t(**kwargs)
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
        finally:
            mp.undo()
            shutil.rmtree(td, ignore_errors=True)
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
