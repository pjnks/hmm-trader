"""
SPRINT 17a — AGATE Shadow-Only Flag Patch
==========================================

Status: PRE-STAGED — DO NOT APPLY UNTIL 2026-04-25 (post hands-off protocol)
Target: live_trading.py (AGATE)
Approved by: user directive 2026-04-23

PROBLEM
-------
AGATE was suspended Sprint 16 (2026-04-20) with the service stopped + disabled.
All data collection ceased: agate_journal.db last write 2026-04-20T01:02:07 UTC.
Every day dark = a day of Gaussian-baseline training data that cannot be
reconstructed later when the eventual Student's t rewrite needs benchmarking.

The suspension diagnosis ("Gaussian emission misspecification") is also
currently unproven — it's an inductive hypothesis from N=12 trades, not a
statistical measurement of confidence-vs-realized-return calibration.

SOLUTION
--------
A --shadow-only flag that runs the full scan/HMM/journal/status pipeline
but bypasses all live trading paths. Produces the dataset needed for the
Phase 2 Kurtosis Diagnostic (~May 23) without touching paper_trades.db,
the broker, the monitor, or sending any trade notifications.

ZERO RISK
---------
- No order routing (broker.place_market_order never called)
- No paper P&L writes (monitor.log_entry/log_exit never called)
- No trade notifications (notify_trade never called)
- No state mutation (broker.position stays None, held_ticker stays None)
- CAN run alongside BERYL and CITRINE without quota/memory contention
  (AGATE was consuming ~110 MB pre-suspension, same as other traders)

FILES TOUCHED
-------------
live_trading.py (one file, four surgical edits)
  - argparse (add --shadow-only)
  - main() (pass shadow_only through to engine constructor)
  - LiveTradingEngine.__init__ (accept shadow_only param, log startup banner)
  - _enter_position + _exit_position (early-return short-circuits for shadow mode)

=============================================================================
"""

# =============================================================================
# EDIT 1 — argparse (live_trading.py:744-752)
# =============================================================================
# Current:
#
#   def main():
#       parser = argparse.ArgumentParser(description="AGATE multi-ticker crypto rotation")
#       parser.add_argument("--test", action="store_true", help="Run in test mode (no real money)")
#       parser.add_argument("--live", action="store_true", help="RUN IN LIVE MODE (REAL MONEY)")
#       parser.add_argument(
#           "--tickers",
#           type=str,
#           default=None,
#           help="Comma-separated Polygon tickers (default: all 16 from config.AGATE_TICKERS)",
#       )
#
# ADD a new argument after --live:
#
#   parser.add_argument(
#       "--shadow-only",
#       action="store_true",
#       dest="shadow_only",
#       help="Run full scan + HMM + journaling, but bypass broker/monitor/notify. "
#            "For Gaussian-baseline data collection during Sprint 17a.",
#   )


# =============================================================================
# EDIT 2 — main() (live_trading.py:754-784)
# =============================================================================
# Current validation:
#
#   if not args.test and not args.live:
#       parser.print_help()
#       print("\nYou must specify --test or --live")
#       sys.exit(1)
#
# REPLACE WITH (accept shadow-only as valid):
#
#   if not args.test and not args.live and not args.shadow_only:
#       parser.print_help()
#       print("\nYou must specify --test, --live, or --shadow-only")
#       sys.exit(1)
#
#   if args.shadow_only and (args.test or args.live):
#       print("--shadow-only is mutually exclusive with --test/--live")
#       sys.exit(1)
#
# Current engine construction at L783:
#
#   engine = LiveTradingEngine(tickers=tickers, test_mode=args.test)
#
# REPLACE WITH:
#
#   engine = LiveTradingEngine(
#       tickers=tickers,
#       test_mode=args.test,
#       shadow_only=args.shadow_only,
#   )


# =============================================================================
# EDIT 3 — LiveTradingEngine.__init__ (live_trading.py:102-145)
# =============================================================================
# Current signature:
#
#   def __init__(self, tickers: list[str], test_mode: bool = True):
#       self.tickers = tickers
#       self.test_mode = test_mode
#
# REPLACE WITH:
#
#   def __init__(
#       self,
#       tickers: list[str],
#       test_mode: bool = True,
#       shadow_only: bool = False,
#   ):
#       self.tickers = tickers
#       self.test_mode = test_mode
#       self.shadow_only = shadow_only
#
# Current startup banner at L134-146 (log.info lines):
#
#   mode = "TEST" if test_mode else "REAL"
#   log.info(f"Live Trading Engine initialized ({mode} mode)")
#
# REPLACE WITH:
#
#   if shadow_only:
#       mode = "SHADOW-ONLY"
#   elif test_mode:
#       mode = "TEST"
#   else:
#       mode = "REAL"
#   log.info(f"Live Trading Engine initialized ({mode} mode)")
#   if shadow_only:
#       log.info("   [SHADOW] No broker calls, no monitor writes, no notifications.")
#       log.info("   [SHADOW] Purpose: Gaussian baseline data for Phase 2 Kurtosis Diagnostic.")
#       log.info("   [SHADOW] Writes only to agate_journal.db + agate_status.json.")


# =============================================================================
# EDIT 4 — _enter_position (live_trading.py:475-529)
# =============================================================================
# Current:
#
#   def _enter_position(self, signal: dict) -> bool:
#       """Enter a position on the given ticker."""
#       ticker = signal["ticker"]
#       current_price = signal.get("current_price", 0.0)
#       confirmations = signal.get("confirmations", 0)
#
#       if current_price <= 0:
#           log.warning(f"Invalid price for {ticker}: {current_price}")
#           return False
#
#       # Switch broker to this ticker's Coinbase product
#       product_id = config.CRYPTO_PRODUCT_MAP.get(ticker, "SOL-USD")
#       ...
#
# ADD early-return at the top, before any broker interaction:
#
#   def _enter_position(self, signal: dict) -> bool:
#       """Enter a position on the given ticker."""
#       ticker = signal["ticker"]
#       current_price = signal.get("current_price", 0.0)
#       confirmations = signal.get("confirmations", 0)
#
#       if current_price <= 0:
#           log.warning(f"Invalid price for {ticker}: {current_price}")
#           return False
#
#       # Sprint 17a: shadow mode logs the would-be entry and exits early.
#       if self.shadow_only:
#           log.info(
#               f"[SHADOW] Would BUY {ticker.replace('X:','')} @ ${current_price:.4f} "
#               f"(cf={confirmations}). No order placed."
#           )
#           return False  # No trade executed; caller sees executed=False
#
#       # Switch broker to this ticker's Coinbase product
#       ... (rest of function unchanged)


# =============================================================================
# EDIT 5 — _exit_position (live_trading.py:531-~580)
# =============================================================================
# Current:
#
#   def _exit_position(self, current_price: float, signal: dict) -> bool:
#       """Exit current position and apply per-ticker cooldown."""
#       if not self.broker.position:
#           return False
#
#       ticker = self.held_ticker or self.broker.position.ticker
#       ...
#
# ADD early-return (though _exit_position should never be reached in shadow
# mode since no positions are ever opened — but guard anyway for safety):
#
#   def _exit_position(self, current_price: float, signal: dict) -> bool:
#       """Exit current position and apply per-ticker cooldown."""
#       if self.shadow_only:
#           # Shadow mode never opens positions, so this path is unreachable.
#           # Guard anyway in case of state weirdness during transition.
#           log.warning("[SHADOW] _exit_position called in shadow mode — unreachable path")
#           return False
#
#       if not self.broker.position:
#           return False
#       ... (rest of function unchanged)


# =============================================================================
# SYSTEMD DEPLOY STEPS (Apr 25)
# =============================================================================
# 1. Apply code edits on Mac, commit to hmm-trader repo
# 2. rsync or scp live_trading.py to VM /home/ubuntu/HMM-Trader/
# 3. Update systemd unit file:
#
#      sudo systemctl edit agate-trader
#
#    Add override to change ExecStart:
#
#      [Service]
#      ExecStart=
#      ExecStart=/home/ubuntu/miniconda3/bin/python live_trading.py --shadow-only
#
# 4. Reload + enable + start:
#
#      sudo systemctl daemon-reload
#      sudo systemctl enable agate-trader
#      sudo systemctl start agate-trader
#
# 5. Verify within 10 minutes:
#
#      sudo systemctl status agate-trader            # active (running)
#      sudo journalctl -u agate-trader -n 30         # SHADOW-ONLY banner + scan starting
#      ls -la /home/ubuntu/HMM-Trader/agate_journal.db   # mtime should be recent
#      ls -la /home/ubuntu/HMM-Trader/paper_trades.db    # mtime should STILL be Apr 20
#
# 6. After 24h, confirm data accumulation:
#
#      sqlite3 agate_journal.db \
#        "SELECT COUNT(*) FROM signal_journal WHERE timestamp > '2026-04-25';"
#      # Expect ~84 rows/day (14 tickers × 6 cycles)


# =============================================================================
# ROLLBACK (if anything goes wrong)
# =============================================================================
# sudo systemctl stop agate-trader
# sudo systemctl disable agate-trader
# git checkout HEAD~1 -- live_trading.py   # on Mac
# rsync to VM, done. Returns to suspended state, zero BERYL/CITRINE impact.


# =============================================================================
# SUCCESS CRITERIA (48h post-deploy)
# =============================================================================
# [ ] agate-trader service: active (running)
# [ ] agate_journal.db: growing (~84 rows/day)
# [ ] paper_trades.db: frozen (no writes since 2026-04-20)
# [ ] consolidated dashboard AGATE panel: showing live regime data
# [ ] Zero Pushover alerts from AGATE
# [ ] Zero new rows in paper_trades.db::open_positions
# [ ] No errors in `journalctl -u agate-trader --since '24 hours ago'`
