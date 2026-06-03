"""Smoke test for GG-B — talk-to-lab channel.

Pre-GG-B, /api/steer wrote to two audit ledgers (steers.jsonl +
pi_actions.jsonl) but never to the lab's events.jsonl — so the
director's gather_observation, which reads ONLY from events.jsonl,
never saw PI messages. The director loop ran disconnected from PI
intent unless the operator re-typed their steer into seed_brief.md
by hand. Bug-shaped UX: the UI's "talk" channel existed on paper
but didn't actually connect to bert.

GG-B closes this with three structural fixes:

  GG-B.1 — /api/steer now also appends a pi_message event_class
  entry to <lab>/sor/events.jsonl, alongside the existing
  steers.jsonl + pi_actions.jsonl writes. Same for /api/voice-steer.

  GG-B.2 — core/director._read_recent_events adds "pi_message" to
  keep_classes; Observation gains a `pi_messages` field that
  filters those events for the director's prompt. director_decision.md
  gains a "PI messages this iteration" section with a locked
  address-or-acknowledge rule.

  GG-B.3 — TalkToLab UI component (sidebar drawer) mounted on the
  App shell. Letter-slip aesthetic, persistent across surfaces,
  hidden in demo mode and during onboarding.

Covers:
  - /api/steer triple-writes (steers + pi_actions + events.jsonl)
  - /api/voice-steer triple-writes
  - pi_message events have the expected shape
  - director keep_classes includes pi_message
  - Observation has pi_messages field
  - gather_observation extracts pi_messages from recent events
  - director prompt has "PI messages this iteration" section
  - TalkToLab component exists at canonical path
  - TalkToLab posts to /api/steer with apiPost
  - TalkToLab uses useActiveLab for per-lab routing
  - TalkToLab has a persistent chip + drawer (not a modal)
  - App shell mounts TalkToLab (hidden in demo mode + pre-keys)
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


TALK = LAB_ROOT / "bert" / "v4" / "src" / "components" / "TalkToLab.tsx"
APP_TSX = LAB_ROOT / "bert" / "v4" / "src" / "App.tsx"
PROMPT = LAB_ROOT / "prompts" / "director_decision.md"


# ─── Backend: /api/steer dual-writes ───────────────────────────────


def test_steer_writes_pi_message_to_events_jsonl() -> None:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)

    user_labs = Path.home() / ".bert" / "labs"
    user_labs.mkdir(parents=True, exist_ok=True)
    test_slug = "gg-b_steer_test"
    test_path = user_labs / test_slug
    if test_path.exists():
        shutil.rmtree(test_path)
    try:
        (test_path / "sor").mkdir(parents=True)
        (test_path / "sor" / "events.jsonl").write_text("")
        (test_path / "state").mkdir()
        (test_path / "seed_brief.md").write_text("# x")

        r = client.post(f"/api/steer?lab={test_slug}", json={
            "text": "focus on memory next cycle",
            "modality": "typed",
        })
        assert r.status_code == 200
        # All three files written
        events = (test_path / "sor" / "events.jsonl").read_text().splitlines()
        steers = (test_path / "state" / "steers.jsonl").read_text().splitlines()
        actions = (test_path / "state" / "pi_actions.jsonl").read_text().splitlines()
        assert len(events) == 1
        assert len(steers) == 1
        assert len(actions) == 1
        # The events.jsonl entry has the expected shape
        ev = json.loads(events[0])
        assert ev["event_class"] == "pi_message"
        assert ev["agent"] == "pi"
        assert ev["text"] == "focus on memory next cycle"
        assert ev["modality"] == "typed"
        assert "pi-steer" in ev["tags"]
    finally:
        if test_path.exists():
            shutil.rmtree(test_path)


def test_voice_steer_writes_pi_message_to_events_jsonl() -> None:
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)

    user_labs = Path.home() / ".bert" / "labs"
    user_labs.mkdir(parents=True, exist_ok=True)
    test_slug = "gg-b_voice_test"
    test_path = user_labs / test_slug
    if test_path.exists():
        shutil.rmtree(test_path)
    try:
        (test_path / "sor").mkdir(parents=True)
        (test_path / "sor" / "events.jsonl").write_text("")
        (test_path / "state").mkdir()
        (test_path / "seed_brief.md").write_text("# x")

        # Upload a tiny non-empty blob
        r = client.post(
            f"/api/voice-steer?lab={test_slug}",
            files={"audio": ("test.webm", b"fake audio bytes", "audio/webm")},
        )
        assert r.status_code == 200
        events = (test_path / "sor" / "events.jsonl").read_text().splitlines()
        assert len(events) == 1
        ev = json.loads(events[0])
        assert ev["event_class"] == "pi_message"
        assert ev["modality"] == "whisper"
        assert "voice" in ev["tags"]
    finally:
        if test_path.exists():
            shutil.rmtree(test_path)


# ─── Director consumes pi_message ─────────────────────────────────


def test_director_keep_classes_includes_pi_message() -> None:
    from core import director as dir_mod
    import inspect
    src = inspect.getsource(dir_mod._read_recent_events)
    assert '"pi_message"' in src, (
        "director._read_recent_events keep_classes must include pi_message"
    )


def test_observation_has_pi_messages_field() -> None:
    from core import director as dir_mod
    # The dataclass field exists
    fields = {f.name for f in dir_mod.Observation.__dataclass_fields__.values()}
    assert "pi_messages" in fields, (
        "Observation must have a pi_messages list field"
    )


def test_gather_observation_extracts_pi_messages() -> None:
    """Synthesize a lab with pi_message events and verify
    gather_observation surfaces them as a separate field."""
    from core import director as dir_mod
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "seed_brief.md").write_text("# x")
        (tmp / "sor").mkdir()
        events = [
            {"event_class": "pi_message", "id": "stx_1",
             "text": "focus on memory", "agent": "pi",
             "ts": "2026-05-18T00:00:00Z"},
            {"event_class": "verdict", "id": "v_1",
             "verdict": "APPROVE", "ts": "2026-05-18T00:01:00Z"},
            {"event_class": "pi_message", "id": "stx_2",
             "text": "and check the cooldown frequency", "agent": "pi",
             "ts": "2026-05-18T00:02:00Z"},
        ]
        with (tmp / "sor" / "events.jsonl").open("w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        (tmp / "state").mkdir()

        obs = dir_mod.gather_observation(tmp, iteration=1)
        assert len(obs.pi_messages) == 2, (
            f"expected 2 pi messages; got {len(obs.pi_messages)}"
        )
        assert obs.pi_messages[0]["text"] == "focus on memory"
        assert obs.pi_messages[1]["text"] == "and check the cooldown frequency"
        # The verdict should NOT be in pi_messages
        assert all(m.get("event_class") == "pi_message"
                   for m in obs.pi_messages)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_observation_to_json_surfaces_pi_messages() -> None:
    from core import director as dir_mod
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "seed_brief.md").write_text("# x")
        (tmp / "sor").mkdir()
        with (tmp / "sor" / "events.jsonl").open("w") as f:
            f.write(json.dumps({
                "event_class": "pi_message", "id": "stx_3",
                "text": "test message", "agent": "pi",
                "ts": "2026-05-18T00:00:00Z",
            }) + "\n")
        (tmp / "state").mkdir()

        obs = dir_mod.gather_observation(tmp, iteration=1)
        rendered = obs.to_json()
        assert "pi_messages" in rendered
        assert "test message" in rendered
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── Director prompt section ──────────────────────────────────────


def test_director_prompt_has_pi_messages_section() -> None:
    text = PROMPT.read_text()
    assert "PI messages this iteration" in text
    # The locked rule must be there
    assert "address every unconsumed message" in text.lower()


def test_director_prompt_locks_override_rule() -> None:
    """If PI says X and calibration says Y, PI wins. Lock that rule."""
    text = PROMPT.read_text()
    # The override rule (PI > heuristics)
    assert ("PI messages are HIGHER weight" in text or
            "PI messages are higher weight" in text.lower())


def test_director_prompt_addresses_empty_state() -> None:
    text = PROMPT.read_text()
    # "When pi_messages is empty, this section is a no-op"
    assert "pi_messages` is empty" in text or "pi_messages is empty" in text


# ─── TalkToLab UI component ───────────────────────────────────────


def test_talk_to_lab_component_exists() -> None:
    assert TALK.exists(), f"TalkToLab missing at {TALK}"


def test_talk_to_lab_posts_to_steer_endpoint() -> None:
    text = TALK.read_text()
    # The component uses a template string `/api/steer${qs}` so match
    # the path fragment, not the literal double-quoted string.
    assert "/api/steer" in text
    assert "apiPost" in text


def test_talk_to_lab_uses_active_lab_for_routing() -> None:
    text = TALK.read_text()
    assert "useActiveLab" in text
    # Routes via the labQuery helper
    assert "labQuery" in text


def test_talk_to_lab_is_drawer_not_modal() -> None:
    """Per feedback_visualization_as_art: not a modal popup. The drawer
    slides from the right, persistent chip toggles open/closed."""
    text = TALK.read_text()
    # Drawer position (right edge)
    assert "position: \"fixed\"" in text
    # Slide-from-right transform on AnimatePresence
    assert "x: 480" in text or "x: -480" in text
    # The chip exists as a separate persistent affordance
    assert "talk" in text.lower()


def test_talk_to_lab_fetches_recent_messages() -> None:
    """Drawer shows the last 10 pi_message entries with their status."""
    text = TALK.read_text()
    assert "/api/events" in text
    # Filters by event_class
    assert '"pi_message"' in text
    # Shows consumed_at_cycle status
    assert "consumed_at_cycle" in text


def test_talk_to_lab_does_NOT_use_chat_bubble_or_slack_pattern() -> None:
    """Anti-pattern check. The PI ↔ lab channel is NOT a chat — it's
    a steer ledger. Letter-slip aesthetic, not message bubbles."""
    import re
    text = TALK.read_text()
    decommented = re.sub(r"//[^\n]*", "", text)
    decommented = re.sub(r"/\*.*?\*/", "", decommented, flags=re.DOTALL)
    low = decommented.lower()
    assert "bubble" not in low
    assert "chatbot" not in low
    # No Slack / Discord style indicators
    assert "slack" not in low
    assert "discord" not in low


def test_talk_to_lab_supports_keyboard_send() -> None:
    """⌘/Ctrl + Enter sends the message — standard composer affordance."""
    text = TALK.read_text()
    assert "metaKey" in text or "ctrlKey" in text
    # Hint surfaced in the UI
    assert "Ctrl + Enter" in text or "Cmd + Enter" in text or "⌘" in text


# ─── App shell integration ────────────────────────────────────────


def test_app_shell_mounts_talk_to_lab() -> None:
    text = APP_TSX.read_text()
    assert "<TalkToLab" in text
    assert "import { TalkToLab }" in text


def test_app_shell_hides_talk_in_demo_mode() -> None:
    text = APP_TSX.read_text()
    # Hidden in demo mode (investor flows don't show the steer channel)
    assert "!isDemoMode" in text
    # The TalkToLab line specifically must be gated by isDemoMode
    app_idx = text.find("<TalkToLab")
    assert app_idx >= 0
    # Find the line containing <TalkToLab — must have an isDemoMode check
    line_start = text.rfind("\n", 0, app_idx)
    line_end = text.find("\n", app_idx)
    line = text[line_start:line_end]
    assert "isDemoMode" in line, (
        f"TalkToLab line must be gated by isDemoMode: {line!r}"
    )


def test_app_shell_hides_talk_during_onboarding() -> None:
    """User hasn't entered keys yet → no point showing a talk-to-lab
    drawer because there's no working lab to talk to."""
    text = APP_TSX.read_text()
    app_idx = text.find("<TalkToLab")
    assert app_idx >= 0
    line_start = text.rfind("\n", 0, app_idx)
    line_end = text.find("\n", app_idx)
    line = text[line_start:line_end]
    assert "credsReady" in line, (
        f"TalkToLab line must be gated by credsReady: {line!r}"
    )


def main() -> int:
    tests = [
        test_steer_writes_pi_message_to_events_jsonl,
        test_voice_steer_writes_pi_message_to_events_jsonl,
        test_director_keep_classes_includes_pi_message,
        test_observation_has_pi_messages_field,
        test_gather_observation_extracts_pi_messages,
        test_observation_to_json_surfaces_pi_messages,
        test_director_prompt_has_pi_messages_section,
        test_director_prompt_locks_override_rule,
        test_director_prompt_addresses_empty_state,
        test_talk_to_lab_component_exists,
        test_talk_to_lab_posts_to_steer_endpoint,
        test_talk_to_lab_uses_active_lab_for_routing,
        test_talk_to_lab_is_drawer_not_modal,
        test_talk_to_lab_fetches_recent_messages,
        test_talk_to_lab_does_NOT_use_chat_bubble_or_slack_pattern,
        test_talk_to_lab_supports_keyboard_send,
        test_app_shell_mounts_talk_to_lab,
        test_app_shell_hides_talk_in_demo_mode,
        test_app_shell_hides_talk_during_onboarding,
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
