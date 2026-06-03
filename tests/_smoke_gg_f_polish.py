"""Smoke test for GG-F — polish pass.

Two structural fixes (the others were judged out-of-scope after the
audit found that Mission/Meeting/Tide/Manuscript already have
bespoke loading/empty components — replacing them with shared polish
components would be a regression, not progress):

  GG-F.2 — Skip-path banner (audit bug #9). User who clicks
  "skip — i'll set these up later" on the onboarding wizard's
  Providers panel landed on FirstLight with no provider keys, and
  the GG-A.0 credsReady gate immediately bounced them BACK to
  /onboard. No signal that they had made a deliberate choice. Now
  the skip flag persists in localStorage; credsReady honors it; a
  warm-amber banner explains the state on every surface with a
  "finish onboarding" CTA.

  GG-F.4 — Touch-target sweep (per feedback_consumer_product). All
  interactive buttons in the GG-built components (LiveCycle cancel,
  RunCycleControls fire-single + fire-autonomous, PauseResumeControls
  toggle, TalkToLab chip + send) now have explicit minHeight ≥ 44
  for mobile / tablet touch comfort.

Covers:

  SkipPathBanner component:
    - File exists, exports markOnboardingSkipped, clearOnboardingSkipped,
      didSkipOnboarding helpers
    - Banner reads /api/onboarding/credentials-status
    - Banner hidden on /onboard route + when keys exist + on probe error
    - Banner contains a "finish onboarding" CTA
    - Banner uses the warm-amber palette (NOT rust — it's a nudge,
      not an error)

  Onboarding integration:
    - Welcome.onSkip → markOnboardingSkipped()
    - Providers.onSkip → markOnboardingSkipped()
    - Providers.onNext (after successful save) → clearOnboardingSkipped()

  App.tsx integration:
    - SkipPathBanner imported + mounted (not in demo mode)
    - useCredentialsReady honors didSkipOnboarding (early return true)

  Touch-target sweep:
    - LiveCycle cancel button has minHeight 44
    - RunCycleControls buttons (both modes) have minHeight 44
    - PauseResumeControls button has minHeight 44
    - TalkToLab chip + send button both have minHeight 44
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))


BANNER = LAB_ROOT / "bert" / "v4" / "src" / "components" / "SkipPathBanner.tsx"
APP_TSX = LAB_ROOT / "bert" / "v4" / "src" / "App.tsx"
ONBOARDING = LAB_ROOT / "bert" / "v4" / "src" / "surfaces" / "Onboarding.tsx"
LIVE_CYCLE = LAB_ROOT / "bert" / "v4" / "src" / "components" / "LiveCycle.tsx"
RUN_CTRL = LAB_ROOT / "bert" / "v4" / "src" / "components" / "RunCycleControls.tsx"
PAUSE_CTRL = LAB_ROOT / "bert" / "v4" / "src" / "components" / "PauseResumeControls.tsx"
TALK = LAB_ROOT / "bert" / "v4" / "src" / "components" / "TalkToLab.tsx"


# ─── SkipPathBanner component ─────────────────────────────────────


def test_skip_path_banner_file_exists() -> None:
    assert BANNER.exists()


def test_skip_path_banner_exports_helpers() -> None:
    text = BANNER.read_text()
    assert "export function markOnboardingSkipped" in text
    assert "export function clearOnboardingSkipped" in text
    assert "export function didSkipOnboarding" in text
    assert "export function SkipPathBanner" in text


def test_skip_path_banner_reads_credentials_status() -> None:
    text = BANNER.read_text()
    assert "/api/onboarding/credentials-status" in text


def test_skip_path_banner_hidden_on_onboard_route() -> None:
    """The banner must NOT render while the user is finishing the
    wizard — would be redundant with the surface itself."""
    text = BANNER.read_text()
    assert 'location.pathname.startsWith("/onboard")' in text


def test_skip_path_banner_clears_when_keys_arrive() -> None:
    """Once keys exist (user came back and finished onboarding),
    clear the skip flag + hide the banner so the nudge stops."""
    text = BANNER.read_text()
    assert "has_any_provider" in text
    assert "clearOnboardingSkipped()" in text


def test_skip_path_banner_has_finish_cta() -> None:
    text = BANNER.read_text()
    assert "finish onboarding" in text.lower()
    # CTA navigates to /onboard
    assert 'navigate("/onboard")' in text


def test_skip_path_banner_is_nudge_not_error() -> None:
    """Visual idiom: warm-amber (PALETTE.candle) nudge, NOT rust
    error tint. The user made a deliberate choice; respect it."""
    text = BANNER.read_text()
    # The banner uses candle for color, not rust
    assert "PALETTE.candle" in text
    # Specifically check the banner color (candle, not rust)
    banner_idx = text.find("background: PALETTE.candle")
    assert banner_idx >= 0


def test_skip_path_banner_uses_aria_status_role() -> None:
    """Accessibility: role=status announces the nudge to screen readers."""
    text = BANNER.read_text()
    assert 'role="status"' in text


def test_skip_path_banner_cta_meets_touch_target() -> None:
    """Per feedback_consumer_product: interactive elements ≥44px.
    The CTA button has minHeight in its style block — anchor on the
    `navigate("/onboard")` call which is unique to the button (the
    string "finish onboarding" appears in both the docstring and
    the rendered button text)."""
    text = BANNER.read_text()
    nav_idx = text.find('navigate("/onboard")')
    assert nav_idx >= 0
    # The button's style block follows the onClick; check the next
    # 600 chars for minHeight: 44.
    window = text[nav_idx:nav_idx + 600]
    assert "minHeight: 44" in window


# ─── Onboarding integration ───────────────────────────────────────


def test_onboarding_imports_skip_helpers() -> None:
    text = ONBOARDING.read_text()
    assert "markOnboardingSkipped" in text
    assert "clearOnboardingSkipped" in text


def test_onboarding_welcome_skip_marks_skipped() -> None:
    text = ONBOARDING.read_text()
    # The Welcome step's onSkip must call markOnboardingSkipped before
    # navigating away
    welcome_block = re.search(
        r'step === "welcome".*?onSkip=\{[^}]*?\}', text, re.DOTALL)
    assert welcome_block, "Welcome step block not found"
    assert "markOnboardingSkipped" in welcome_block.group(0)


def test_onboarding_providers_skip_marks_skipped() -> None:
    text = ONBOARDING.read_text()
    providers_block = re.search(
        r'step === "providers".*?(?=\{step ===)', text, re.DOTALL)
    assert providers_block, "Providers step block not found"
    assert "markOnboardingSkipped" in providers_block.group(0)


def test_onboarding_providers_next_clears_skip_flag() -> None:
    """After successful key-save, the skip flag should be cleared so
    the banner doesn't continue nudging on later surfaces."""
    text = ONBOARDING.read_text()
    providers_block = re.search(
        r'step === "providers".*?(?=\{step ===)', text, re.DOTALL)
    assert providers_block
    assert "clearOnboardingSkipped" in providers_block.group(0)


# ─── App.tsx integration ──────────────────────────────────────────


def test_app_imports_skip_banner() -> None:
    text = APP_TSX.read_text()
    assert "SkipPathBanner" in text
    assert "didSkipOnboarding" in text


def test_app_mounts_skip_banner_outside_demo_mode() -> None:
    text = APP_TSX.read_text()
    assert "<SkipPathBanner" in text
    # Banner mount line is gated by !isDemoMode
    mount_idx = text.find("<SkipPathBanner")
    line_start = text.rfind("\n", 0, mount_idx)
    line_end = text.find("\n", mount_idx)
    line = text[line_start:line_end]
    assert "isDemoMode" in line, (
        f"SkipPathBanner mount must be gated by isDemoMode: {line!r}"
    )


def test_app_creds_ready_honors_skip_flag() -> None:
    """When the user has explicitly skipped, useCredentialsReady
    returns true (don't bounce back to /onboard)."""
    text = APP_TSX.read_text()
    creds_idx = text.find("function useCredentialsReady")
    assert creds_idx >= 0
    body = text[creds_idx:creds_idx + 2000]
    assert "didSkipOnboarding()" in body


# ─── Touch-target sweep (≥44px) ───────────────────────────────────


def test_live_cycle_cancel_button_meets_touch_target() -> None:
    text = LIVE_CYCLE.read_text()
    # The cancel button block contains minHeight 44
    cancel_idx = text.find("cancel current cycle")
    assert cancel_idx >= 0
    # Search a window forward of the aria-label
    window = text[max(0, cancel_idx - 200):cancel_idx + 1500]
    assert "minHeight: 44" in window


def test_run_cycle_buttons_meet_touch_target() -> None:
    """Post-rework: primaryBtn (start mission) is 48px; secondaryBtn
    (stop / run again / try again) is 44px. Both ≥ 44 WCAG min."""
    text = RUN_CTRL.read_text()
    pri = text.find("function primaryBtn")
    sec = text.find("function secondaryBtn")
    assert pri >= 0 and sec >= 0
    assert "minHeight: 48" in text[pri:pri + 400]
    assert "minHeight: 44" in text[sec:sec + 400]


def test_pause_resume_button_meets_touch_target() -> None:
    text = PAUSE_CTRL.read_text()
    # The toggle button's style has minHeight: 44
    assert "minHeight: 44" in text


def test_talk_to_lab_chip_meets_touch_target() -> None:
    """The persistent talk-to-lab chip in the bottom-right is a
    primary affordance — must be touch-comfortable on mobile."""
    text = TALK.read_text()
    # Both the chip and the send button have minHeight: 44
    assert text.count("minHeight: 44") >= 2


# ─── No regressions on bound polish components ────────────────────


def test_lab_dashboard_still_uses_stratum_skeleton() -> None:
    """GG-A.2 binding shouldn't have been undone by this phase."""
    text = (LAB_ROOT / "bert" / "v4" / "src" / "surfaces" /
            "LabDashboard.tsx").read_text()
    assert "StratumSkeleton" in text


def test_outputs_still_uses_polish_components() -> None:
    text = (LAB_ROOT / "bert" / "v4" / "src" / "surfaces" /
            "Outputs.tsx").read_text()
    assert "StratumSkeleton" in text
    assert "ConnectomicEmpty" in text


def test_talk_to_lab_hides_on_onboard_route() -> None:
    """R6 walkthrough caught the talk chip leaking into the
    onboarding wizard. The component now early-returns null when
    location.pathname starts with /onboard. App.tsx's gates still
    cover demo mode + pre-keys; this gate adds the in-onboarding
    case.
    """
    text = (LAB_ROOT / "bert" / "v4" / "src" / "components" /
            "TalkToLab.tsx").read_text()
    # Must import useLocation
    assert 'import { useLocation } from "react-router-dom"' in text
    # Must check pathname against /onboard
    assert 'location.pathname.startsWith("/onboard")' in text
    # Must early-return null
    assert "if (hideForOnboarding) return null" in text or \
           "if (location.pathname.startsWith(\"/onboard\")) return null" in text


def main() -> int:
    tests = [
        test_skip_path_banner_file_exists,
        test_skip_path_banner_exports_helpers,
        test_skip_path_banner_reads_credentials_status,
        test_skip_path_banner_hidden_on_onboard_route,
        test_skip_path_banner_clears_when_keys_arrive,
        test_skip_path_banner_has_finish_cta,
        test_skip_path_banner_is_nudge_not_error,
        test_skip_path_banner_uses_aria_status_role,
        test_skip_path_banner_cta_meets_touch_target,
        test_onboarding_imports_skip_helpers,
        test_onboarding_welcome_skip_marks_skipped,
        test_onboarding_providers_skip_marks_skipped,
        test_onboarding_providers_next_clears_skip_flag,
        test_app_imports_skip_banner,
        test_app_mounts_skip_banner_outside_demo_mode,
        test_app_creds_ready_honors_skip_flag,
        test_live_cycle_cancel_button_meets_touch_target,
        test_run_cycle_buttons_meet_touch_target,
        test_pause_resume_button_meets_touch_target,
        test_talk_to_lab_chip_meets_touch_target,
        test_lab_dashboard_still_uses_stratum_skeleton,
        test_outputs_still_uses_polish_components,
        test_talk_to_lab_hides_on_onboard_route,
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
