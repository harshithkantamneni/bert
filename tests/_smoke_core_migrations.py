"""Smoke: core/migrations/__init__.py — sqlite migration runner (was 69%).

Clean file/sqlite logic. Covers _split_sql_statements (line/block comments
+ quoted-semicolon handling), list_known_adapters, _adapter_dir hit+miss,
_list_migrations hit+miss, status (fresh → all pending), apply_pending
(applies real code_repo migrations + idempotent re-run no-op).
"""

from __future__ import annotations

import inspect
import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from core import migrations as mig  # noqa: E402

_ADAPTER = "code_repo"


def test_split_sql_statements():
    sql = (
        "CREATE TABLE a (id INT);\n"
        "-- a line comment with ; semicolon\n"
        "/* block ; comment */\n"
        "INSERT INTO a VALUES ('x;y');\n"
        "CREATE INDEX i ON a(id)"   # no trailing ;
    )
    stmts = mig._split_sql_statements(sql)
    assert len(stmts) == 3                       # 2 with ;, 1 trailing
    assert any("'x;y'" in s for s in stmts)      # quoted semicolon preserved


def test_adapter_discovery():
    adapters = mig.list_known_adapters()
    assert _ADAPTER in adapters
    assert mig._adapter_dir(_ADAPTER) is not None
    assert mig._adapter_dir("definitely_not_an_adapter") is None
    migs = mig._list_migrations(_ADAPTER)
    assert migs and all(isinstance(v, int) for v, _ in migs)
    assert mig._list_migrations("definitely_not_an_adapter") == []


def test_status_and_apply(tmp_path):
    # fresh lab → no meta db → everything pending
    st0 = mig.status(tmp_path, _ADAPTER)
    assert st0.current_version == 0 and st0.pending
    # apply → all migrations run
    res = mig.apply_pending(tmp_path, _ADAPTER)
    assert res.applied and not res.errors
    st1 = mig.status(tmp_path, _ADAPTER)
    assert st1.current_version == st1.available_version and not st1.pending
    # idempotent re-run → nothing applied
    res2 = mig.apply_pending(tmp_path, _ADAPTER)
    assert not res2.applied and not res2.errors


def main() -> int:
    tests = [
        test_split_sql_statements,
        test_adapter_discovery,
        test_status_and_apply,
    ]
    for t in tests:
        td = Path(tempfile.mkdtemp())
        try:
            kwargs = {"tmp_path": td} if "tmp_path" in inspect.signature(t).parameters else {}
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
            shutil.rmtree(td, ignore_errors=True)
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
