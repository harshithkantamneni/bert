"""Smoke + TDD: `bert verify` CLI wiring (item 27) + failures.md immutability (item 26).

Item 27: `bert verify <packet>` must work as a real subcommand of the `bert` CLI
(lab.py), not only as `python tools/bert_verify.py`. We test the shared run()
exit codes and that lab.main() routes the `verify` subcommand.

Item 26: failures.md "cannot be edited post-finalization" — already enforced
cryptographically (failures.sigstore signs it + HASHES.txt covers it, both now
ECDSA/cosign-grade after item 25). This pins the guarantee explicitly: editing
failures.md in a sealed packet makes verify FAIL on the failures signature.
"""

from __future__ import annotations

import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

import lab  # noqa: E402
from core import proof_packet, verify_packet  # noqa: E402
from tools import bert_verify  # noqa: E402

EVENTS_PATH = LAB_ROOT / "lab" / "sor" / "events.jsonl"


def _require(*paths: Path) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "requires lab runtime artifact(s) not shipped in the public repo: "
            + ", ".join(str(m) for m in missing)
        )


def _built_packet(tmp: Path) -> Path:
    # build_packet attests cycle events from lab/sor/events.jsonl (lab runtime).
    # The public repo ships no events for cycle 400 — skip when absent.
    _require(EVENTS_PATH)
    if not proof_packet._read_events_for_cycle(400):
        pytest.skip(
            "requires lab runtime artifact: cycle-400 events in "
            "lab/sor/events.jsonl (not shipped in the public repo)"
        )
    return proof_packet.build_packet(cycle_id=400, output_dir=tmp)


def test_run_passes_on_real_packet():
    tmp = Path(tempfile.mkdtemp(prefix="bert_vcli_"))
    try:
        pkt = _built_packet(tmp)
        code = bert_verify.run([str(pkt)], no_color=True)
        assert code in (0, 1), f"expected PASS/WARN exit, got {code}"
    finally:
        shutil.rmtree(tmp)


def test_run_fails_on_missing_packet():
    assert bert_verify.run(["/no/such/packet.tar.gz"], no_color=True) == 2


def test_lab_cli_routes_verify(monkeypatch):
    # lab.main() must dispatch the `verify` subcommand to bert_verify.run.
    monkeypatch.setattr(sys, "argv", ["bert", "verify", "/no/such/packet.tar.gz"])
    assert lab.main() == 2


def test_failures_md_immutable_after_seal():
    # item 26: tamper failures.md in a built packet -> verify FAILs on the
    # failures.md signature check ([7]) — the file cannot be silently edited.
    tmp = Path(tempfile.mkdtemp(prefix="bert_imm_"))
    extract = Path(tempfile.mkdtemp(prefix="bert_imm_x_"))
    try:
        pkt = _built_packet(tmp)
        with tarfile.open(pkt, "r:gz") as tar:
            tar.extractall(extract, filter="data")
        pdir = extract / "cycle-0400"
        # edit failures.md AFTER sealing
        fpath = pdir / "failures.md"
        fpath.write_text(fpath.read_text() + "\n(silently edited post-finalization)\n")
        tampered = tmp / "tampered.tar.gz"
        with tarfile.open(tampered, "w:gz") as tar:
            tar.add(pdir, arcname="cycle-0400")
        res = verify_packet.verify_packet(tampered)
        assert res.overall == "FAIL"
        check_7 = next(c for c in res.checks if "[7]" in c.name)
        assert check_7.status == verify_packet.CheckStatus.FAIL, (
            f"failures.md edit must fail check [7]; got {check_7.status}: {check_7.detail}"
        )
    finally:
        shutil.rmtree(tmp)
        shutil.rmtree(extract)


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


def main() -> int:
    import inspect
    tests = [
        test_run_passes_on_real_packet,
        test_run_fails_on_missing_packet,
        test_lab_cli_routes_verify,
        test_failures_md_immutable_after_seal,
    ]
    mp = _MP()
    for t in tests:
        try:
            if "monkeypatch" in inspect.signature(t).parameters:
                t(mp)
            else:
                t()
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
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
