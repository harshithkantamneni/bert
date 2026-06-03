"""Smoke test for GG-A.1 — lab creation flow from the UI.

Pre-GG-A.1 the user could type a "first mission" in the onboarding
wizard but it landed at `memories/mission.md` — a path bert_run.py
never reads. So onboarding completed successfully, but bert had no
mission to run on. The Mission tab in the UI ALSO writes seed_brief.md
into the supervisor lab, which would clobber the self-improvement
mission. Net effect: the consumer-facing path to "give bert a
project" was completely broken (bug #8 from the audit).

GG-A.1 closes this with:
  - /api/labs POST: scaffolds a real customer lab at
    ~/.bert/labs/<slug>/ with seed_brief.md = mission, lab.yaml
    declaring focus_areas + role:standard + share_with_supervisor:true
    + archetype-appropriate defaults
  - /api/labs GET: lists all labs (supervisor + customer) for the
    dashboard. Ignores share_with_supervisor (UI is user-facing,
    not supervisor-facing)
  - tools/bert_init._scaffold_lab gained focus_areas-aware lab.yaml
    generation
  - Onboarding's FirstMission step now POSTs to /api/labs with
    {name, mission, archetype} instead of /api/onboarding/first-mission

Covers:
  - CreateLabRequest schema validates name pattern, archetype enum,
    autonomy enum, focus_areas optional
  - POST /api/labs success: directory exists, lab.yaml ships
    FF-A-aware schema, seed_brief.md == mission, sor/events.jsonl
    initialized
  - POST /api/labs 409 on slug conflict
  - POST /api/labs 422 on bad inputs (name pattern, mission too short)
  - GET /api/labs returns supervisor + customer labs with all FF-A
    metadata
  - bert_init._scaffold_lab writes focus_areas from answers OR
    archetype defaults; lab_schema_version: 1; role: standard
  - Onboarding.tsx FirstMission POSTs to /api/labs (source check)
  - Onboarding.tsx FirstMission no longer hits first-mission endpoint
  - Pre-GG memories/mission.md write path is removed from the wizard
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


# ─── /api/labs POST + GET roundtrip ────────────────────────────────


def test_create_lab_endpoint_exists() -> None:
    src = (LAB_ROOT / "api" / "main.py").read_text()
    assert '@app.post("/api/labs")' in src
    assert "def create_lab" in src
    assert '@app.get("/api/labs")' in src
    # The unified handler (GG-A.1 + L.4 contract preserved)
    assert "def list_labs_unified" in src


def test_create_lab_request_schema_validates_name_pattern() -> None:
    from api.main import CreateLabRequest
    from pydantic import ValidationError
    # Valid name
    r = CreateLabRequest(
        name="My Research Lab",
        mission="x" * 50,
    )
    assert r.archetype == "research"  # default

    # Invalid: starts with non-letter
    try:
        CreateLabRequest(name="9bad", mission="x" * 50)
        raise AssertionError("expected ValidationError")
    except ValidationError:
        pass

    # Invalid: too short
    try:
        CreateLabRequest(name="x", mission="x" * 50)
        raise AssertionError("expected ValidationError")
    except ValidationError:
        pass


def test_create_lab_request_validates_archetype_enum() -> None:
    from api.main import CreateLabRequest
    from pydantic import ValidationError
    for arch in ("research", "product", "strategy"):
        r = CreateLabRequest(name="My Lab", mission="x" * 50, archetype=arch)
        assert r.archetype == arch
    try:
        CreateLabRequest(name="My Lab", mission="x" * 50, archetype="other")
        raise AssertionError("expected ValidationError")
    except ValidationError:
        pass


def test_create_lab_request_validates_mission_min_length() -> None:
    from api.main import CreateLabRequest
    from pydantic import ValidationError
    try:
        CreateLabRequest(name="My Lab", mission="short")
        raise AssertionError("expected ValidationError")
    except ValidationError:
        pass


def test_create_lab_round_trip_via_testclient() -> None:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)

    user_labs = Path.home() / ".bert" / "labs"
    user_labs.mkdir(parents=True, exist_ok=True)
    # The slugifier in tools/bert_init._scaffold_lab is
    # `name.replace(" ", "_").lower()` — spaces become underscores,
    # hyphens stay as-is.
    test_slug = "gg-a-1_create_smoke"
    test_path = user_labs / test_slug
    # Clean any leftover
    if test_path.exists():
        shutil.rmtree(test_path)
    try:
        r = client.post("/api/labs", json={
            "name": "GG-A-1 Create Smoke",
            "mission": ("compare free-tier inference providers across cost, "
                        "latency, and cross-family diversity"),
            "archetype": "research",
        })
        assert r.status_code == 200, f"got {r.status_code}: {r.text}"
        body = r.json()
        assert body["ok"] is True
        assert body["name"] == test_slug
        # The lab dir exists with the FF-A-aware shape
        assert test_path.exists()
        assert (test_path / "lab.yaml").exists()
        assert (test_path / "sor" / "events.jsonl").exists()
        assert (test_path / "state").exists()
        # seed_brief.md has the mission
        seed = (test_path / "seed_brief.md").read_text()
        assert "free-tier inference providers" in seed
        # lab.yaml has the FF-A-aware schema
        ly = (test_path / "lab.yaml").read_text()
        assert "lab_schema_version: 1" in ly
        assert "role: standard" in ly
        assert "share_with_supervisor: true" in ly
        assert "focus_areas:" in ly
        # Research archetype defaults
        assert "methodology" in ly
        assert "evidence" in ly
    finally:
        if test_path.exists():
            shutil.rmtree(test_path)


def test_create_lab_rejects_duplicate_slug() -> None:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)

    user_labs = Path.home() / ".bert" / "labs"
    user_labs.mkdir(parents=True, exist_ok=True)
    test_slug = "gg-a-1_dup_test"
    test_path = user_labs / test_slug
    if test_path.exists():
        shutil.rmtree(test_path)
    try:
        # First create succeeds
        r1 = client.post("/api/labs", json={
            "name": "GG-A-1 Dup Test",
            "mission": "x" * 60,
            "archetype": "research",
        })
        assert r1.status_code == 200
        # Second create with same name → 409
        r2 = client.post("/api/labs", json={
            "name": "GG-A-1 Dup Test",
            "mission": "x" * 60,
            "archetype": "research",
        })
        assert r2.status_code == 409
        assert "already exists" in r2.json()["detail"]
    finally:
        if test_path.exists():
            shutil.rmtree(test_path)


def test_list_labs_preserves_l4_contract() -> None:
    """L.4 shape (active + scaffolded) must still be present so pre-GG
    smoke tests (_smoke_l4_lab_routing, _smoke_p6_end_to_end) pass."""
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.get("/api/labs")
    assert r.status_code == 200
    body = r.json()
    assert "active" in body
    assert "scaffolded" in body
    assert isinstance(body["scaffolded"], list)
    assert "path" in body["active"]
    assert "is_bert_lab_default" in body["active"]


def test_list_labs_includes_supervisor() -> None:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    r = client.get("/api/labs")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    # The supervisor (repo's own lab/) should be in the unified flat list
    supervisor = next((l for l in body["labs"] if l["is_supervisor"]), None)
    assert supervisor is not None, "supervisor lab not in /api/labs response"
    assert supervisor["role"] == "supervisor"
    # The bert-internal focus areas
    assert "routing" in supervisor["focus_areas"]
    assert "memory" in supervisor["focus_areas"]


def test_list_labs_includes_customer_labs() -> None:
    """After creating a customer lab, /api/labs includes it."""
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)

    user_labs = Path.home() / ".bert" / "labs"
    user_labs.mkdir(parents=True, exist_ok=True)
    test_slug = "gg-a-1_list_test"
    test_path = user_labs / test_slug
    if test_path.exists():
        shutil.rmtree(test_path)
    try:
        client.post("/api/labs", json={
            "name": "GG-A-1 List Test",
            "mission": "x" * 60,
            "archetype": "product",
        })
        r = client.get("/api/labs")
        body = r.json()
        customer = next((l for l in body["labs"]
                          if l["name"] == test_slug), None)
        assert customer is not None, (
            f"customer lab {test_slug} missing from /api/labs response"
        )
        assert customer["archetype"] == "product"
        assert "architecture" in customer["focus_areas"]
        assert customer["role"] == "standard"
    finally:
        if test_path.exists():
            shutil.rmtree(test_path)


def test_list_labs_does_not_apply_share_filter() -> None:
    """The UI's lab listing must include opt-out labs (bug #6 fix).
    share_with_supervisor=false hides a lab from the SUPERVISOR's
    cross-lab view but NOT from the USER's lab list."""
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)

    user_labs = Path.home() / ".bert" / "labs"
    user_labs.mkdir(parents=True, exist_ok=True)
    test_slug = "gg-a-1-private-test"
    test_path = user_labs / test_slug
    if test_path.exists():
        shutil.rmtree(test_path)
    try:
        # Scaffold a lab with share_with_supervisor: false
        (test_path / "sor").mkdir(parents=True)
        (test_path / "sor" / "events.jsonl").write_text("")
        (test_path / "state").mkdir()
        (test_path / "seed_brief.md").write_text("# Private mission")
        (test_path / "lab.yaml").write_text(
            "lab_schema_version: 1\n"
            f"name: {test_slug}\n"
            "archetype: research\n"
            "role: standard\n"
            "share_with_supervisor: false\n"
            "focus_areas: [a, b, c]\n"
        )
        r = client.get("/api/labs")
        body = r.json()
        names = [l["name"] for l in body["labs"]]
        assert test_slug in names, (
            "opt-out lab missing from /api/labs — share filter "
            "shouldn't apply to UI lab listing"
        )
    finally:
        if test_path.exists():
            shutil.rmtree(test_path)


# ─── bert_init._scaffold_lab FF-A-awareness ────────────────────────


def test_scaffold_lab_writes_focus_areas() -> None:
    from tools import bert_init
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    # Redirect LABS_DIR for the duration of this test
    original = bert_init.LABS_DIR
    bert_init.LABS_DIR = tmp
    try:
        ld = bert_init._scaffold_lab(
            {"name": "Test Lab", "archetype": "Research",
             "provider": "groq", "autonomy": "Collaborator",
             "seed": "x" * 50, "focus_areas": None},
            user_provided_seed=True,
        )
        yaml_text = (ld / "lab.yaml").read_text()
        assert "focus_areas:" in yaml_text
        assert "methodology" in yaml_text  # research default
        assert "lab_schema_version: 1" in yaml_text
        assert "role: standard" in yaml_text
        assert "share_with_supervisor: true" in yaml_text
    finally:
        bert_init.LABS_DIR = original
        shutil.rmtree(tmp, ignore_errors=True)


def test_scaffold_lab_honors_user_focus_areas() -> None:
    from tools import bert_init
    tmp = Path(tempfile.mkdtemp())
    original = bert_init.LABS_DIR
    bert_init.LABS_DIR = tmp
    try:
        ld = bert_init._scaffold_lab(
            {"name": "Custom Areas", "archetype": "Research",
             "provider": "groq", "autonomy": "Pilot",
             "seed": "x" * 50,
             "focus_areas": ["latency", "cost", "reliability"]},
            user_provided_seed=True,
        )
        yaml_text = (ld / "lab.yaml").read_text()
        assert "latency" in yaml_text
        assert "cost" in yaml_text
        assert "methodology" not in yaml_text  # archetype default shouldn't win
    finally:
        bert_init.LABS_DIR = original
        shutil.rmtree(tmp, ignore_errors=True)


def test_scaffold_lab_product_archetype_defaults() -> None:
    from tools import bert_init
    tmp = Path(tempfile.mkdtemp())
    original = bert_init.LABS_DIR
    bert_init.LABS_DIR = tmp
    try:
        ld = bert_init._scaffold_lab(
            {"name": "Product Lab", "archetype": "Product",
             "provider": "groq", "autonomy": "Collaborator",
             "seed": "x" * 50, "focus_areas": None},
            user_provided_seed=True,
        )
        yaml_text = (ld / "lab.yaml").read_text()
        assert "architecture" in yaml_text
        assert "implementation" in yaml_text
        assert "testing" in yaml_text
    finally:
        bert_init.LABS_DIR = original
        shutil.rmtree(tmp, ignore_errors=True)


# ─── Onboarding UI integration ────────────────────────────────────


def test_onboarding_firstmission_posts_to_api_labs() -> None:
    src = (LAB_ROOT / "bert" / "v4" / "src" / "surfaces" /
           "Onboarding.tsx").read_text()
    assert '"/api/labs"' in src
    # Pre-GG it hit /api/onboarding/first-mission; bug #8 says retire
    # that path. Historical mentions in comments are fine; the live
    # apiPost call must NOT target that endpoint.
    import re
    # Find apiPost call paths
    live_paths = re.findall(r'apiPost<[^>]*>\(\s*"([^"]+)"', src)
    assert "/api/onboarding/first-mission" not in live_paths, (
        "Onboarding still calls the retired /api/onboarding/first-mission "
        f"endpoint. Live apiPost paths: {live_paths}"
    )


def test_onboarding_firstmission_exposes_archetype_picker() -> None:
    src = (LAB_ROOT / "bert" / "v4" / "src" / "surfaces" /
           "Onboarding.tsx").read_text()
    # Three archetypes
    assert 'value="research"' in src
    assert 'value="product"' in src
    assert 'value="strategy"' in src
    assert "ARCHETYPE_BLURBS" in src


def test_onboarding_firstmission_surfaces_errors() -> None:
    src = (LAB_ROOT / "bert" / "v4" / "src" / "surfaces" /
           "Onboarding.tsx").read_text()
    # Error state for 409 / 422 from /api/labs
    assert "setError" in src
    # 409 detail propagation
    assert "could not create the lab" in src or "scaffolding…" in src


def test_no_memory_mission_md_writes_in_onboarding() -> None:
    """bug #8 — the old `memories/mission.md` write path must be gone
    from the wizard's LIVE code. Historical mentions in comments
    that document the retired path are allowed (and useful — they
    explain to future readers why the path changed)."""
    src = (LAB_ROOT / "bert" / "v4" / "src" / "surfaces" /
           "Onboarding.tsx").read_text()
    # Drop comments before searching for the live path
    import re
    # Strip // line comments
    decommented = re.sub(r"//[^\n]*", "", src)
    # Strip /* */ block comments (greedy across newlines)
    decommented = re.sub(r"/\*.*?\*/", "", decommented, flags=re.DOTALL)
    assert "memories/mission.md" not in decommented, (
        "Onboarding live code still references memories/mission.md "
        "— bug #8 says retire that write path"
    )


def main() -> int:
    tests = [
        test_create_lab_endpoint_exists,
        test_create_lab_request_schema_validates_name_pattern,
        test_create_lab_request_validates_archetype_enum,
        test_create_lab_request_validates_mission_min_length,
        test_create_lab_round_trip_via_testclient,
        test_create_lab_rejects_duplicate_slug,
        test_list_labs_preserves_l4_contract,
        test_list_labs_includes_supervisor,
        test_list_labs_includes_customer_labs,
        test_list_labs_does_not_apply_share_filter,
        test_scaffold_lab_writes_focus_areas,
        test_scaffold_lab_honors_user_focus_areas,
        test_scaffold_lab_product_archetype_defaults,
        test_onboarding_firstmission_posts_to_api_labs,
        test_onboarding_firstmission_exposes_archetype_picker,
        test_onboarding_firstmission_surfaces_errors,
        test_no_memory_mission_md_writes_in_onboarding,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            return 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"\nAll {len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
