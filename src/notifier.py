"""
notifier.py
───────────
Multi-channel alert system for kill-switch + trading notifications.

Channels (in priority order):
1. macOS Native Notification — always works, appears as popup + sound
2. Pushover (optional)      — real phone push notification ($5 one-time)
3. Terminal bell             — fallback beep

Usage
─────
  from src.notifier import notify_kill_switch, notify_trade, notify_daily

  # Kill-switch triggered → urgent popup + sound + phone push
  notify_kill_switch("Daily loss > 2%: -$205.00")

  # Trade executed → standard notification
  notify_trade("BUY", 0.1234, 150.00, pnl=None)

  # Daily summary → quiet notification
  notify_daily(sharpe=0.85, pnl=12.50, trades=3)
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from pathlib import Path

# Load .env file for Pushover keys
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

# ── OS Detection ─────────────────────────────────────────────────────────
# Skip macOS-specific notifications on Linux (e.g., Oracle Cloud VM)
IS_MACOS = platform.system() == "Darwin"


# ── Configuration ─────────────────────────────────────────────────────────

# Pushover (optional — set in .env for phone push notifications)
# Sign up at https://pushover.net ($5 one-time) and create an Application
PUSHOVER_USER_KEY = os.getenv("PUSHOVER_USER_KEY", "")
PUSHOVER_APP_TOKEN = os.getenv("PUSHOVER_APP_TOKEN", "")


def _macos_notify(title: str, message: str, sound: str = "default", urgent: bool = False) -> bool:
    """
    Send macOS native notification via osascript.

    Appears as popup banner/alert in Notification Center.
    Works even if Terminal is minimized.
    Silently skipped on non-macOS systems (e.g., Linux VM).
    """
    if not IS_MACOS:
        log.debug(f"Skipping macOS notification (not macOS): {title}")
        return False
    try:
        sound_part = f'sound name "{sound}"' if sound else ""
        script = f'display notification "{message}" with title "{title}" {sound_part}'
        subprocess.run(
            ["osascript", "-e", script],
            timeout=5,
            capture_output=True,
        )
        log.info(f"macOS notification sent: {title}")
        return True
    except Exception as e:
        log.warning(f"macOS notification failed: {e}")
        return False


def _macos_alert_dialog(title: str, message: str) -> bool:
    """
    Show a macOS alert dialog that REQUIRES clicking OK to dismiss.

    Use for critical kill-switch alerts — cannot be missed or swiped away.
    Silently skipped on non-macOS systems (e.g., Linux VM).
    """
    if not IS_MACOS:
        log.debug(f"Skipping macOS alert dialog (not macOS): {title}")
        return False
    try:
        script = (
            f'display alert "{title}" message "{message}" '
            f'as critical buttons {{"OK"}} default button "OK"'
        )
        subprocess.run(
            ["osascript", "-e", script],
            timeout=300,  # Wait up to 5 min for user to click OK
            capture_output=True,
        )
        log.info(f"macOS alert dialog shown: {title}")
        return True
    except Exception as e:
        log.warning(f"macOS alert dialog failed: {e}")
        return False


def _pushover_notify(title: str, message: str, priority: int = 0) -> bool:
    """
    Send push notification via Pushover API (phone notification).

    Priority levels:
      -1 = quiet (no sound)
       0 = normal
       1 = high priority (bypasses quiet hours)
       2 = emergency (repeats until acknowledged)
    """
    if not PUSHOVER_USER_KEY or not PUSHOVER_APP_TOKEN:
        log.debug("Pushover not configured — skipping phone notification")
        return False

    try:
        import urllib.request
        import urllib.parse
        import json

        data = urllib.parse.urlencode({
            "token": PUSHOVER_APP_TOKEN,
            "user": PUSHOVER_USER_KEY,
            "title": title,
            "message": message,
            "priority": priority,
            "sound": "siren" if priority >= 1 else "pushover",
        }).encode()

        # Emergency priority requires retry/expire params
        if priority == 2:
            data = urllib.parse.urlencode({
                "token": PUSHOVER_APP_TOKEN,
                "user": PUSHOVER_USER_KEY,
                "title": title,
                "message": message,
                "priority": 2,
                "retry": 60,    # Repeat every 60 seconds
                "expire": 3600,  # Stop after 1 hour
                "sound": "siren",
            }).encode()

        req = urllib.request.Request(
            "https://api.pushover.net/1/messages.json",
            data=data,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("status") == 1:
                log.info(f"Pushover notification sent: {title}")
                return True
            else:
                log.warning(f"Pushover error: {result}")
                return False

    except Exception as e:
        log.warning(f"Pushover notification failed: {e}")
        return False


def _terminal_bell() -> None:
    """Ring terminal bell as last-resort alert."""
    print("\a" * 3, end="", flush=True)


# ── Public API ────────────────────────────────────────────────────────────

def notify_kill_switch(reason: str) -> None:
    """
    CRITICAL ALERT: Kill-switch triggered.

    Sends notifications via ALL channels:
    - macOS alert dialog (modal, must click OK)
    - macOS notification center (banner + sound)
    - Pushover emergency priority (repeats until acknowledged)
    - Terminal bell
    """
    title = "KILL-SWITCH TRIGGERED"
    message = f"Trading STOPPED. Reason: {reason}"

    log.error(f"KILL-SWITCH ALERT: {reason}")

    # 1. Modal dialog (cannot be missed)
    _macos_alert_dialog(title, message)

    # 2. Notification center (backup)
    _macos_notify(title, message, sound="Basso", urgent=True)

    # 3. Phone push (emergency priority — repeats until acknowledged)
    _pushover_notify(title, message, priority=2)

    # 4. Terminal bell
    _terminal_bell()


def notify_trade(side: str, size: float, price: float, pnl: float | None = None,
                 ticker: str | None = None, project: str = "AGATE") -> None:
    """
    Trade executed notification — phone push + macOS banner.

    Supports all projects: AGATE (crypto), BERYL (equities), CITRINE (portfolio).
    Pass ticker="NVDA" and project="BERYL" for equity trades.
    """
    ticker_str = ticker or "SOL"
    # Equities: 2 decimal places for shares; crypto: 4
    size_fmt = f"{size:.2f}" if ticker and not ticker.startswith("X:") else f"{size:.4f}"

    if pnl is not None:
        emoji = "+" if pnl >= 0 else ""
        message = f"{side} {size_fmt} {ticker_str} @ ${price:.2f} | P&L: {emoji}${pnl:.2f}"
    else:
        message = f"{side} {size_fmt} {ticker_str} @ ${price:.2f}"

    title = f"{project} Trade: {side} {ticker_str}"

    # macOS notification (banner)
    _macos_notify(title, message, sound="Pop")

    # Phone push (normal priority)
    _pushover_notify(title, message, priority=0)


def notify_daily(sharpe: float, pnl: float, trades: int, win_rate: float = 0.0) -> None:
    """
    Daily summary notification.

    Quiet priority — no sound on phone, just appears in notification list.
    """
    emoji = "+" if pnl >= 0 else ""
    title = "AGATE Daily Report"
    message = (
        f"Sharpe: {sharpe:.3f} | P&L: {emoji}${pnl:.2f} | "
        f"Trades: {trades} | WR: {win_rate:.0f}%"
    )

    # macOS notification (banner, default sound)
    _macos_notify(title, message, sound="default")

    # Phone push (quiet — no sound)
    _pushover_notify(title, message, priority=-1)


def notify_signal(signal: str, regime: str, confirmations: int, price: float) -> None:
    """
    Signal generated notification (informational).

    Only macOS notification — no phone push for signals.
    """
    title = f"AGATE Signal: {signal}"
    message = f"{regime} | {confirmations}/8 confirms | ${price:.2f}"

    _macos_notify(title, message, sound="Tink")


def test_notifications() -> None:
    """Test all notification channels."""
    print("Testing notification channels...\n")

    print("1. macOS Notification Center...")
    ok = _macos_notify("AGATE Test", "This is a test notification", sound="default")
    print(f"   {'OK' if ok else 'FAILED'}")

    print("2. Pushover (phone)...")
    if PUSHOVER_USER_KEY and PUSHOVER_APP_TOKEN:
        ok = _pushover_notify("AGATE Test", "This is a test phone notification", priority=0)
        print(f"   {'OK' if ok else 'FAILED'}")
    else:
        print("   SKIPPED (PUSHOVER_USER_KEY / PUSHOVER_APP_TOKEN not set in .env)")
        print("   To enable: sign up at https://pushover.net ($5 one-time)")
        print("   Then add to .env:")
        print("     PUSHOVER_USER_KEY=your_user_key")
        print("     PUSHOVER_APP_TOKEN=your_app_token")

    print("3. Terminal bell...")
    _terminal_bell()
    print("   OK (did you hear a beep?)")

    print("\nDone! Kill-switch alerts will use ALL available channels.")


if __name__ == "__main__":
    test_notifications()
