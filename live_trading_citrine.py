"""
live_trading_citrine.py
───────────────────────
CITRINE: Confidence-Weighted NDX100 Portfolio Rotation — Live Trading Engine.

Runs daily after market close:
  1. Scans all tickers with HMM regime detection
  2. Computes CITRINE scores and portfolio weights
  3. Executes rebalance (simulated fills in test mode)
  4. Logs trades and portfolio snapshots to citrine_trades.db
  5. Checks portfolio-level kill-switch rules

Usage
─────
  python live_trading_citrine.py --test                              # full NDX100
  python live_trading_citrine.py --test --tickers AAPL,TSLA,NVDA     # subset
  python live_trading_citrine.py --test --long-only                   # long only
  python live_trading_citrine.py --test --cooldown time               # time-based cooldown
"""

from __future__ import annotations

import argparse
import json
import logging
import signal as signal_mod
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.citrine_scanner import CitrineScanner, TickerScan
from src.citrine_allocator import CitrineAllocator, PortfolioWeight
from src.notifier import _macos_notify, _pushover_notify, _terminal_bell, notify_trade

# ── Logging ────────────────────────────────────────────────────────────────
# Explicitly configure root logger — basicConfig is a no-op if any imported
# module already triggered it.  force=True (Python 3.8+) overrides that.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger("citrine_live")
log.setLevel(logging.INFO)


# ── Position sizing ────────────────────────────────────────────────────────
SLIPPAGE_BPS = config.CITRINE_SLIPPAGE_BPS  # 10bps
TAKER_FEE = config.CITRINE_TAKER_FEE  # 0.04% per side

# Signal check interval (seconds) — once per day after market close
SIGNAL_INTERVAL_S = 24 * 3600


# ── Portfolio Position Tracker ─────────────────────────────────────────────

class CitrinePosition:
    """Track a single open position with risk engine metrics (Sprint 9)."""
    def __init__(self, ticker: str, direction: str, shares: float,
                 entry_price: float, target_weight: float,
                 entry_atr: float = 0.0, entry_confidence: float = 0.0):
        self.ticker = ticker
        self.direction = direction  # "LONG" or "SHORT"
        self.shares = shares
        self.entry_price = entry_price
        self.target_weight = target_weight
        self.entry_time = datetime.now(tz=timezone.utc).isoformat()
        self.notional = shares * entry_price

        # Risk engine fields (Sprint 9)
        self.entry_atr = entry_atr              # ATR at entry — frozen, never updated
        self.entry_confidence = entry_confidence  # HMM confidence at entry
        self.prev_confidence = entry_confidence   # previous cycle's confidence (for velocity)
        self.high_watermark = entry_price         # highest price since entry (LONG)
        self.low_watermark = entry_price          # lowest price since entry (SHORT)
        self.mae_pct = 0.0                        # max adverse excursion (% of entry)
        self.mfe_pct = 0.0                        # max favorable excursion (% of entry)
        self.mae_atr = 0.0                        # MAE in ATR units
        self.mfe_atr = 0.0                        # MFE in ATR units

    def update_excursions(self, current_price: float) -> None:
        """Update MAE/MFE watermarks with latest price."""
        if self.direction == "LONG":
            self.high_watermark = max(self.high_watermark, current_price)
            self.low_watermark = min(self.low_watermark, current_price)
            # MFE = best unrealized gain
            self.mfe_pct = max(self.mfe_pct,
                               (self.high_watermark - self.entry_price) / self.entry_price)
            # MAE = worst unrealized drawdown (negative)
            adverse = (self.low_watermark - self.entry_price) / self.entry_price
            if adverse < self.mae_pct:
                self.mae_pct = adverse
        else:  # SHORT
            self.low_watermark = min(self.low_watermark, current_price)
            self.high_watermark = max(self.high_watermark, current_price)
            self.mfe_pct = max(self.mfe_pct,
                               (self.entry_price - self.low_watermark) / self.entry_price)
            adverse = (self.entry_price - self.high_watermark) / self.entry_price
            if adverse < self.mae_pct:
                self.mae_pct = adverse

        # ATR-unit excursions (if entry_atr is available)
        if self.entry_atr > 0:
            self.mae_atr = self.mae_pct * self.entry_price / self.entry_atr
            self.mfe_atr = self.mfe_pct * self.entry_price / self.entry_atr

    def chandelier_stop(self, multiplier: float = 2.0) -> float | None:
        """Compute Chandelier trailing stop price. Returns None if no ATR."""
        if self.entry_atr <= 0:
            return None
        if self.direction == "LONG":
            return self.high_watermark - (self.entry_atr * multiplier)
        else:  # SHORT
            return self.low_watermark + (self.entry_atr * multiplier)

    def check_confidence_velocity(
        self, current_confidence: float, current_price: float, prev_close: float,
        threshold: float = -0.25,
    ) -> bool:
        """Return True if confidence velocity exit triggers.
        Condition: confidence dropped > threshold from entry AND close < prev close."""
        delta = current_confidence - self.entry_confidence
        if delta < threshold and current_price < prev_close:
            return True
        return False


# ── Live Engine ────────────────────────────────────────────────────────────

class ShadowTracker:
    """
    Shadow/harvesting mode — runs a parallel allocator with relaxed thresholds
    against the same scan results as the live engine.  Logs theoretical
    entries/exits to a ``shadow_trades`` table for offline analysis.

    Zero impact on live trading:
      • No extra API calls (reuses live scan data)
      • Separate allocator state (own holdings, own hysteresis)
      • Only writes to its own DB table
    """

    # Relaxed thresholds (vs live: entry=0.90, exit=0.50, persistence=3)
    SHADOW_ENTRY_CONFIDENCE = 0.70
    SHADOW_EXIT_CONFIDENCE = 0.35
    SHADOW_PERSISTENCE_DAYS = 1
    SHADOW_MAX_POSITIONS = 30  # higher cap → more data points

    def __init__(self, db_path: Path, capital: float, long_only: bool = True):
        self.db_path = db_path
        self.capital = capital
        self.long_only = long_only

        # Shadow allocator with its own state — completely independent of live
        self._allocator = CitrineAllocator(
            capital=capital,
            long_only=long_only,
            cooldown_mode="none",
        )

        # Shadow positions: ticker → {direction, entry_price, entry_date, ...}
        self._positions: dict[str, dict] = {}

        self._init_shadow_table()
        self._restore_shadow_state()

    def _init_shadow_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shadow_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    action TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    price REAL NOT NULL,
                    notional REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    regime TEXT,
                    confidence REAL,
                    persistence INTEGER,
                    confirmations INTEGER,
                    realized_vol REAL,
                    citrine_score REAL,
                    sector TEXT,
                    entry_atr REAL,
                    regime_half_life REAL,
                    alt_boost REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shadow_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    num_positions INTEGER NOT NULL,
                    theoretical_equity REAL,
                    positions_json TEXT
                )
            """)
            conn.commit()

    def _restore_shadow_state(self):
        """Rebuild shadow positions from the latest snapshot on restart."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT positions_json FROM shadow_snapshots "
                    "ORDER BY id DESC LIMIT 1"
                ).fetchone()
            if row and row[0]:
                data = json.loads(row[0])
                if isinstance(data, dict):
                    self._positions = data
                    # Restore allocator holdings so hysteresis works
                    for ticker in self._positions:
                        self._allocator._holdings[ticker] = \
                            self._positions[ticker].get("days_held", 1)
                    log.info(f"[SHADOW] Restored {len(self._positions)} shadow positions")
        except Exception as e:
            log.warning(f"[SHADOW] Could not restore state: {e}")

    def run_shadow_cycle(
        self,
        scans: list[TickerScan],
        alt_data_boosts: dict[str, float] | None = None,
    ) -> None:
        """
        Run one shadow allocation cycle using relaxed thresholds.

        Called from _run_daily_cycle() with the SAME scan results —
        no extra API calls or HMM fits.
        """
        if not scans:
            return

        # Temporarily override config thresholds for shadow allocation
        orig_entry = config.CITRINE_ENTRY_CONFIDENCE
        orig_exit = config.CITRINE_EXIT_CONFIDENCE
        orig_persist = config.CITRINE_PERSISTENCE_DAYS
        orig_max_pos = config.CITRINE_MAX_POSITIONS

        try:
            config.CITRINE_ENTRY_CONFIDENCE = self.SHADOW_ENTRY_CONFIDENCE
            config.CITRINE_EXIT_CONFIDENCE = self.SHADOW_EXIT_CONFIDENCE
            config.CITRINE_PERSISTENCE_DAYS = self.SHADOW_PERSISTENCE_DAYS
            config.CITRINE_MAX_POSITIONS = self.SHADOW_MAX_POSITIONS

            weights, cash_pct = self._allocator.allocate(
                scans, alt_data_boosts=alt_data_boosts)
        finally:
            # Always restore — even if allocate() throws
            config.CITRINE_ENTRY_CONFIDENCE = orig_entry
            config.CITRINE_EXIT_CONFIDENCE = orig_exit
            config.CITRINE_PERSISTENCE_DAYS = orig_persist
            config.CITRINE_MAX_POSITIONS = orig_max_pos

        # Process shadow entries and exits
        now = datetime.now(tz=timezone.utc).isoformat()
        scan_map = {s.ticker: s for s in scans if s.error is None}
        new_entries = 0
        new_exits = 0

        for w in weights:
            scan = scan_map.get(w.ticker)
            if scan is None:
                continue

            if w.action == "ENTER" and w.ticker not in self._positions:
                # Shadow entry
                self._positions[w.ticker] = {
                    "direction": w.direction,
                    "entry_price": scan.current_price,
                    "entry_date": now,
                    "entry_confidence": scan.confidence,
                    "entry_atr": getattr(scan, "current_atr", 0.0) or 0.0,
                    "days_held": 1,
                    "notional": min(w.notional_usd, self.capital / self.SHADOW_MAX_POSITIONS),
                }
                self._log_shadow_trade(
                    now, w.ticker, "ENTER", w.direction, scan,
                    score=w.raw_score,
                    alt_boost=(alt_data_boosts or {}).get(w.ticker, 1.0),
                )
                new_entries += 1

            elif w.action == "EXIT" and w.ticker in self._positions:
                # Shadow exit
                pos = self._positions[w.ticker]
                entry_price = pos["entry_price"]
                exit_price = scan.current_price
                if entry_price > 0 and exit_price > 0:
                    if pos["direction"] == "LONG":
                        pnl_pct = (exit_price - entry_price) / entry_price * 100
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price * 100
                    notional = pos.get("notional", 0)
                    pnl = notional * pnl_pct / 100
                else:
                    pnl, pnl_pct = 0.0, 0.0

                self._log_shadow_trade(
                    now, w.ticker, "EXIT", pos["direction"], scan,
                    pnl=pnl, pnl_pct=pnl_pct,
                    alt_boost=(alt_data_boosts or {}).get(w.ticker, 1.0),
                )
                del self._positions[w.ticker]
                new_exits += 1

        # Age held positions
        for ticker in self._positions:
            self._positions[ticker]["days_held"] = \
                self._positions[ticker].get("days_held", 0) + 1

        # Update allocator state
        self._allocator.update_holdings(weights)

        # Snapshot
        self._log_shadow_snapshot(now, scan_map)

        total = len(self._positions)
        log.info(f"  [SHADOW] {new_entries} enter | {new_exits} exit | "
                 f"{total} held | thresholds: conf≥{self.SHADOW_ENTRY_CONFIDENCE}, "
                 f"persist≥{self.SHADOW_PERSISTENCE_DAYS}")

    def _log_shadow_trade(
        self, timestamp: str, ticker: str, action: str,
        direction: str, scan: TickerScan,
        score: float = 0.0, pnl: float | None = None,
        pnl_pct: float | None = None, alt_boost: float = 1.0,
    ):
        notional = None
        if action == "ENTER":
            notional = min(
                self.capital / self.SHADOW_MAX_POSITIONS,
                config.CITRINE_MAX_NOTIONAL,
            )
        elif action == "EXIT" and ticker in self._positions:
            notional = self._positions[ticker].get("notional", 0)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO shadow_trades
                (timestamp, ticker, action, direction, price, notional,
                 pnl, pnl_pct, regime, confidence, persistence,
                 confirmations, realized_vol, citrine_score, sector,
                 entry_atr, regime_half_life, alt_boost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    timestamp, ticker, action, direction,
                    round(scan.current_price, 4),
                    round(notional, 2) if notional else None,
                    round(pnl, 2) if pnl is not None else None,
                    round(pnl_pct, 2) if pnl_pct is not None else None,
                    scan.regime_cat,
                    round(scan.confidence, 4),
                    scan.persistence,
                    scan.confirmations,
                    round(scan.realized_vol, 4) if scan.realized_vol else None,
                    round(score, 4),
                    scan.sector,
                    round(getattr(scan, "current_atr", 0.0) or 0.0, 4),
                    round(getattr(scan, "regime_half_life", 0.0) or 0.0, 2),
                    round(alt_boost, 3),
                ),
            )
            conn.commit()

    def _log_shadow_snapshot(self, timestamp: str, scan_map: dict):
        """Write a lightweight snapshot of shadow positions."""
        # Mark-to-market
        for ticker, pos in self._positions.items():
            scan = scan_map.get(ticker)
            if scan and scan.current_price > 0:
                pos["current_price"] = scan.current_price

        equity = sum(
            p.get("notional", 0) * (
                1 + (p.get("current_price", p["entry_price"]) - p["entry_price"])
                / p["entry_price"]
            ) if p["entry_price"] > 0 else 0
            for p in self._positions.values()
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO shadow_snapshots
                (timestamp, num_positions, theoretical_equity, positions_json)
                VALUES (?, ?, ?, ?)""",
                (timestamp, len(self._positions), round(equity, 2),
                 json.dumps(self._positions)),
            )
            conn.commit()


class CitrineLiveEngine:
    """
    CITRINE daily portfolio rotation engine.

    Runs once per day:
      1. Scan all tickers with HMM
      2. Allocate portfolio weights
      3. Rebalance positions (enter new, exit old, scale existing)
      4. Log everything to SQLite
      5. Check kill-switch rules
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        test_mode: bool = True,
        long_only: bool = False,
        cooldown_mode: str = "none",
    ):
        self.tickers = tickers or config.CITRINE_UNIVERSE
        self.test_mode = test_mode
        self.long_only = long_only
        self.cooldown_mode = cooldown_mode

        self.capital = config.CITRINE_INITIAL_CAPITAL
        self.cash = self.capital  # start fully in cash
        self.positions: dict[str, CitrinePosition] = {}  # ticker → position

        self.db_path = ROOT / "citrine_trades.db"
        self._init_db()

        self.scanner = CitrineScanner(
            tickers=self.tickers,
            lookback_days=365,
            quiet=False,
        )
        self.allocator = CitrineAllocator(
            capital=self.capital,
            long_only=self.long_only,
            cooldown_mode=self.cooldown_mode,
        )

        # Last scan regime counts (carried forward for intraday snapshots)
        self._last_bull_count = 0
        self._last_bear_count = 0
        self._last_chop_count = 0
        self._last_cash_pct = 0.0

        # Kill-switch grace period: track restart time and completed cycles
        # After a restart, skip kill-switch checks until at least
        # KILL_SWITCH_GRACE_CYCLES daily cycles have completed, giving new
        # trades a chance to improve metrics before re-triggering.
        self._restart_time = datetime.now(tz=timezone.utc)
        self._cycles_since_restart = 0
        self._KILL_SWITCH_GRACE_CYCLES = 3  # Skip kill-switch for first 3 cycles

        # Shadow tracker: relaxed-threshold allocator for data harvesting
        self.shadow = ShadowTracker(
            db_path=self.db_path,
            capital=self.capital,
            long_only=self.long_only,
        )

        # Restore state from DB if previous session exists
        self._restore_state_from_db()

        log.info(f"CITRINE Engine initialized: {len(self.tickers)} tickers "
                 f"({'TEST' if test_mode else 'LIVE'})")
        log.info(f"Long-only: {long_only} | Cooldown: {cooldown_mode}")
        log.info(f"Kill-switch grace period: {self._KILL_SWITCH_GRACE_CYCLES} cycles")
        log.info(f"Shadow tracker: conf≥{ShadowTracker.SHADOW_ENTRY_CONFIDENCE}, "
                 f"persist≥{ShadowTracker.SHADOW_PERSISTENCE_DAYS}, "
                 f"max {ShadowTracker.SHADOW_MAX_POSITIONS} positions")

    # ── Risk Engine Constants (Sprint 9) ─────────────────────────────────────
    CHANDELIER_MULTIPLIER = 2.0     # MAE calibration: 95th pctile = 1.793 ATR × 1.1
    CONF_VELOCITY_THRESHOLD = -0.25 # exit if confidence drops > 0.25 from entry
    ATR_RISK_BUDGET = 0.01          # 1% of capital risked per position via ATR sizing
    ATR_MAX_NOTIONAL_PCT = 0.15     # 15% of capital max per position

    def _init_db(self):
        """Initialize SQLite database for CITRINE trade and snapshot logging."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    action TEXT NOT NULL,
                    shares REAL NOT NULL,
                    price REAL NOT NULL,
                    notional REAL NOT NULL,
                    pnl REAL,
                    pnl_pct REAL,
                    signal_strength INTEGER,
                    portfolio_weight REAL,
                    regime TEXT,
                    confidence REAL
                )
            """)
            # Sprint 9: add risk engine columns (safe if already exist)
            for col, typ in [
                ("entry_atr", "REAL"),
                ("exit_reason", "TEXT"),
                ("mae_pct", "REAL"),
                ("mfe_pct", "REAL"),
                ("mae_atr", "REAL"),
                ("mfe_atr", "REAL"),
                ("entry_confidence", "REAL"),
                ("hold_days", "INTEGER"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {typ}")
                except sqlite3.OperationalError:
                    pass  # column already exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_equity REAL NOT NULL,
                    cash REAL NOT NULL,
                    invested REAL NOT NULL,
                    num_positions INTEGER NOT NULL,
                    num_long INTEGER NOT NULL,
                    num_short INTEGER NOT NULL,
                    bull_count INTEGER NOT NULL,
                    bear_count INTEGER NOT NULL,
                    chop_count INTEGER NOT NULL,
                    positions_json TEXT,
                    cash_pct REAL
                )
            """)
            conn.commit()

    # ── State Restoration ──────────────────────────────────────────────────

    def _restore_state_from_db(self):
        """
        Restore positions and cash from last portfolio snapshot on restart.

        Prevents duplicate ENTER trades when the process is restarted.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT cash, positions_json, total_equity "
                    "FROM portfolio_snapshots ORDER BY id DESC LIMIT 1"
                ).fetchone()

            if row is None:
                log.info("No previous state found — starting fresh")
                return

            saved_cash, positions_json, saved_equity = row

            if not positions_json:
                log.info("Previous snapshot has no positions — starting fresh")
                return

            pos_data = json.loads(positions_json)
            if not pos_data:
                self.cash = saved_cash
                log.info(f"Restored state: 0 positions, ${saved_cash:,.0f} cash")
                return

            # Rebuild positions from stored JSON
            restored = 0
            for ticker, info in pos_data.items():
                direction = info.get("direction", "LONG")
                shares = info.get("shares", 0)
                entry = info.get("entry", 0)
                if shares > 0 and entry > 0:
                    pos = CitrinePosition(
                        ticker=ticker,
                        direction=direction,
                        shares=shares,
                        entry_price=entry,
                        target_weight=0,  # Will be recalculated on next cycle
                        entry_atr=info.get("entry_atr", 0.0),
                        entry_confidence=info.get("entry_confidence", 0.0),
                    )
                    # Override auto-set entry_time and notional
                    pos.notional = shares * entry
                    # Restore watermarks (use current price if available)
                    current = info.get("current", entry)
                    pos.high_watermark = max(entry, current)
                    pos.low_watermark = min(entry, current)
                    self.positions[ticker] = pos
                    restored += 1

            self.cash = saved_cash

            # Restore allocator holdings (days_held) from trade timestamps
            self._restore_allocator_holdings()

            log.info(f"Restored state: {restored} positions, ${saved_cash:,.0f} cash "
                     f"(last equity ${saved_equity:,.0f})")

        except Exception as e:
            log.warning(f"Could not restore state from DB: {e} — starting fresh")

    def _restore_allocator_holdings(self):
        """Restore allocator._holdings from ENTER trades in DB."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get latest ENTER trade per ticker that is still open (no EXIT after it)
                rows = conn.execute("""
                    SELECT t.ticker, t.timestamp
                    FROM trades t
                    WHERE t.action = 'ENTER'
                    AND t.ticker NOT IN (
                        SELECT ticker FROM trades
                        WHERE action = 'EXIT'
                        AND timestamp > t.timestamp
                    )
                    ORDER BY t.timestamp DESC
                """).fetchall()

            for ticker, ts in rows:
                if ticker in self.positions:
                    try:
                        entry_date = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        days_held = (datetime.now(tz=timezone.utc) - entry_date).days
                        self.allocator._holdings[ticker] = max(1, days_held)
                    except (ValueError, TypeError):
                        self.allocator._holdings[ticker] = 1

        except Exception as e:
            log.warning(f"Could not restore allocator holdings: {e}")

    # ── Core Loop ──────────────────────────────────────────────────────────

    def run_forever(self):
        """Main trading loop — runs once per day after market close."""
        log.info(f"\n{'='*60}")
        log.info(f"  CITRINE LIVE TRADING — Portfolio Rotation")
        log.info(f"  Mode: {'TEST (simulated)' if self.test_mode else 'LIVE'}")
        log.info(f"  Tickers: {len(self.tickers)}")
        log.info(f"  Capital: ${self.capital:,.0f}")
        log.info(f"  Max positions: {config.CITRINE_MAX_POSITIONS}")
        log.info(f"  Long-only: {self.long_only}")
        log.info(f"  Cooldown: {self.cooldown_mode}")
        log.info(f"{'='*60}\n")

        _macos_notify(
            "CITRINE Started",
            f"{len(self.tickers)}-ticker portfolio rotation — "
            f"{'TEST' if self.test_mode else 'LIVE'} mode"
        )

        # Immediate scan on restart — don't wait for next scheduled cycle
        if self.positions:
            log.info(f"[CITRINE] {len(self.positions)} positions restored — running immediate scan...")
            try:
                self._run_daily_cycle()
                self._cycles_since_restart += 1
                log.info("[CITRINE] Immediate post-restart scan complete")
            except Exception as e:
                log.warning(f"[CITRINE] WARNING: Post-restart scan failed: {e}")
                log.info("[CITRINE] Will retry on next scheduled cycle")

        while True:
            try:
                # Check kill-switch (with grace period after restart)
                if self._cycles_since_restart >= self._KILL_SWITCH_GRACE_CYCLES:
                    if self._check_kill_switch():
                        log.error("KILL-SWITCH TRIGGERED — stopping CITRINE")
                        break
                else:
                    log.info(f"Kill-switch grace period: cycle {self._cycles_since_restart + 1}"
                             f"/{self._KILL_SWITCH_GRACE_CYCLES} (skipping check)")

                # Run daily scan + rebalance cycle
                self._run_daily_cycle()
                self._cycles_since_restart += 1

                # Sleep until next check — wake every 4h for intraday risk check
                log.info(f"Next cycle in {SIGNAL_INTERVAL_S // 3600}h. Sleeping "
                         f"(intra-day risk checks every 4h)...")
                self._sleep_with_risk_checks(SIGNAL_INTERVAL_S)

            except KeyboardInterrupt:
                log.info("Keyboard interrupt — shutting down CITRINE")
                break
            except Exception as e:
                log.error(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(300)  # Wait 5 min then retry

    def _run_daily_cycle(self):
        """Execute one complete scan → allocate → rebalance cycle."""
        log.info(f"\n{'─'*60}")
        log.info(f"  DAILY SCAN — {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        log.info(f"{'─'*60}")

        # Step 1: Scan all tickers
        log.info("\n[Step 1] Scanning tickers with HMM regime detection...")
        try:
            scans = self.scanner.scan_all()
        except Exception as e:
            log.error(f"[CITRINE] ERROR: Failed to scan tickers: {e}")
            log.info("[CITRINE] Skipping this cycle due to scan failure")
            return

        bull_count = sum(1 for s in scans if s.is_bull and s.error is None)
        bear_count = sum(1 for s in scans if s.is_bear and s.error is None)
        chop_count = sum(1 for s in scans if s.regime_cat == "CHOP" and s.error is None)
        err_count = sum(1 for s in scans if s.error is not None)

        log.info(f"  Scan complete: {bull_count} BULL | {bear_count} BEAR | "
                 f"{chop_count} CHOP | {err_count} errors")

        # Step 1b: Risk engine exits — Chandelier stop + confidence velocity
        # Run BEFORE allocator so the allocator doesn't try to re-enter
        self._check_risk_exits(scans)

        # Step 1c: Fetch alt-data boost (insider trading signals)
        alt_data_boosts = self._fetch_alt_data_boosts(scans)

        # Step 2: Allocate portfolio weights
        log.info("\n[Step 2] Computing CITRINE allocations...")
        try:
            weights, cash_pct = self.allocator.allocate(scans, alt_data_boosts=alt_data_boosts)
        except Exception as e:
            log.error(f"[CITRINE] ERROR: Failed to allocate portfolio: {e}")
            log.info("[CITRINE] Skipping rebalance this cycle due to allocation failure")
            return

        enter_count = sum(1 for w in weights if w.action == "ENTER")
        hold_count = sum(1 for w in weights if w.action in ("HOLD", "SCALE_UP"))
        exit_count = sum(1 for w in weights if w.action == "EXIT")

        log.info(f"  Allocations: {enter_count} ENTER | {hold_count} HOLD/SCALE | "
                 f"{exit_count} EXIT | Cash target: {cash_pct:.0%}")

        # Step 3: Execute rebalance
        log.info("\n[Step 3] Executing rebalance...")
        self._execute_rebalance(weights, scans, cash_pct)

        # Step 4: Update allocator state
        self.allocator.update_holdings(weights)

        # Step 4b: Shadow tracker — run relaxed-threshold allocator on same scans
        try:
            self.shadow.run_shadow_cycle(scans, alt_data_boosts=alt_data_boosts)
        except Exception as e:
            log.warning(f"[SHADOW] Shadow cycle failed (non-fatal): {e}")

        # Step 5: Log portfolio snapshot + cache regime counts for intraday snapshots
        self._last_bull_count = bull_count
        self._last_bear_count = bear_count
        self._last_chop_count = chop_count
        self._last_cash_pct = cash_pct
        self._log_snapshot(scans, cash_pct, bull_count, bear_count, chop_count)

        # Step 6: Send summary notification
        self._notify_summary(weights, cash_pct, bull_count)

        # Print portfolio summary
        self._print_portfolio_summary()

    def _check_risk_exits(self, scans: list[TickerScan]) -> None:
        """
        Sprint 9 risk engine: check Chandelier stops and confidence velocity
        for all held positions. Force-exits before the allocator runs.

        Two independent exit triggers:
          1. Chandelier: price < high_watermark - (entry_atr × 2.0)
          2. Conf velocity: confidence dropped >0.25 from entry AND price declining
        """
        if not self.positions:
            return

        scan_map = {s.ticker: s for s in scans if s.error is None}
        exits_triggered: list[tuple[str, float, str]] = []  # (ticker, price, reason)

        for ticker, pos in list(self.positions.items()):
            scan = scan_map.get(ticker)
            if not scan or scan.current_price <= 0:
                continue

            price = scan.current_price

            # Update MAE/MFE excursions with latest price
            pos.update_excursions(price)

            # 1. Chandelier trailing stop
            stop = pos.chandelier_stop(self.CHANDELIER_MULTIPLIER)
            if stop is not None:
                if pos.direction == "LONG" and price < stop:
                    exits_triggered.append((
                        ticker, price,
                        f"chandelier_stop (stop=${stop:.2f}, "
                        f"HWM=${pos.high_watermark:.2f}, "
                        f"ATR=${pos.entry_atr:.2f})"
                    ))
                    continue
                elif pos.direction == "SHORT" and price > stop:
                    exits_triggered.append((
                        ticker, price,
                        f"chandelier_stop (stop=${stop:.2f}, "
                        f"LWM=${pos.low_watermark:.2f})"
                    ))
                    continue

            # 2. Confidence velocity exit
            prev_close = price  # approximate — use scan price as proxy
            if pos.check_confidence_velocity(
                scan.confidence, price, prev_close,
                self.CONF_VELOCITY_THRESHOLD,
            ):
                exits_triggered.append((
                    ticker, price,
                    f"conf_velocity (entry={pos.entry_confidence:.2f}→"
                    f"now={scan.confidence:.2f}, "
                    f"Δ={scan.confidence - pos.entry_confidence:+.2f})"
                ))
                continue

            # Update prev_confidence for next cycle
            pos.prev_confidence = scan.confidence

        # Execute risk exits
        if exits_triggered:
            log.info(f"\n  [RISK ENGINE] {len(exits_triggered)} exit(s) triggered:")
            for ticker, price, reason in exits_triggered:
                log.warning(f"  🛑 {ticker}: {reason}")
                try:
                    self._force_exit_single(ticker, price, reason=reason)
                except Exception as e:
                    log.error(f"  [RISK] Failed to exit {ticker}: {e}")

            # Notify
            summary = ", ".join(f"{t}" for t, _, _ in exits_triggered)
            _pushover_notify(
                "CITRINE Risk Exit",
                f"{len(exits_triggered)} exits: {summary}",
                priority=0,
            )
        else:
            held_with_atr = sum(1 for p in self.positions.values() if p.entry_atr > 0)
            log.info(f"  [RISK ENGINE] All {len(self.positions)} positions OK "
                     f"({held_with_atr} with Chandelier stops)")

    def _fetch_alt_data_boosts(self, scans) -> dict[str, float]:
        """
        Fetch alt-data signals from two sources and combine multiplicatively:
          1. SEC insider trading (Form 4) — BULL tickers only
          2. DIAMOND Kalshi anomaly bridge — all tickers

        Returns boost multiplier map (0.7–1.5) per ticker.
        """
        all_tickers = [s.ticker for s in scans if s.error is None]
        boosts: dict[str, float] = {}

        # Source 1: SEC insider trading signals
        try:
            from src.alternative_data import AlternativeDataScore
            scorer = AlternativeDataScore()

            bull_tickers = [s.ticker for s in scans
                           if s.regime_cat == "BULL" and s.error is None]

            if bull_tickers:
                log.info(f"\n[Step 1b] Fetching insider data for {len(bull_tickers)} BULL tickers...")
                insider_boosts = scorer.scan_tickers(bull_tickers, days=90)

                boosted = {t: b for t, b in insider_boosts.items() if b != 1.0}
                if boosted:
                    for t, b in sorted(boosted.items(), key=lambda x: x[1]):
                        indicator = "🟢" if b > 1.0 else "🔴"
                        log.info(f"  {indicator} {t}: insider boost {b:.2f}x")
                else:
                    log.info(f"  All {len(bull_tickers)} tickers: neutral insider signal (1.0x)")

                boosts.update(insider_boosts)
        except Exception as e:
            log.warning(f"  [AltData] Failed to fetch insider data: {e}")

        # Source 2: DIAMOND Kalshi anomaly bridge
        try:
            from src.diamond_bridge import CitrineDiamondBridge
            bridge = CitrineDiamondBridge()
            diamond_boosts = bridge.fetch_boosts(all_tickers)

            if diamond_boosts:
                log.info(f"[Step 1c] DIAMOND bridge: {len(diamond_boosts)} equity signals")
                # Combine multiplicatively with insider boosts
                for ticker, d_boost in diamond_boosts.items():
                    existing = boosts.get(ticker, 1.0)
                    combined = max(0.7, min(1.5, existing * d_boost))
                    boosts[ticker] = round(combined, 3)
        except Exception as e:
            log.warning(f"  [AltData] Failed to fetch DIAMOND data: {e}")

        return boosts

    def _execute_rebalance(
        self,
        weights: list[PortfolioWeight],
        scans: list[TickerScan],
        cash_pct: float = 0.30,
    ):
        """Execute all rebalance actions (enter, exit, scale).

        Enforces cash_pct as a HARD FLOOR — entries are skipped if they
        would push cash below the target percentage of equity.
        """
        scan_map = {s.ticker: s for s in scans}

        # Process EXITs first (frees up cash)
        for w in weights:
            if w.action == "EXIT":
                try:
                    self._exit_position(w, scan_map)
                except Exception as e:
                    log.error(f"[CITRINE] ERROR: Failed to exit {w.ticker}: {e}")

        # Compute cash floor AFTER exits (equity = cash + invested)
        equity = self.cash + sum(p.shares * p.entry_price for p in self.positions.values())
        min_cash = equity * cash_pct
        log.info(f"  Cash floor: ${min_cash:.0f} ({cash_pct:.0%} of ${equity:.0f} equity) | "
                 f"Available: ${self.cash:.0f}")

        # Process ENTERs and SCALE_UPs (respecting cash floor)
        for w in weights:
            if w.action == "ENTER":
                notional = min(w.notional_usd, self.cash)
                if self.cash - notional < min_cash:
                    log.info(f"  Skip ENTER {w.ticker}: would breach "
                             f"{cash_pct:.0%} cash floor "
                             f"(${self.cash:.0f} - ${notional:.0f} < ${min_cash:.0f})")
                    continue
                try:
                    self._enter_position(w, scan_map)
                except Exception as e:
                    log.error(f"[CITRINE] ERROR: Failed to enter {w.ticker}: {e}")
            elif w.action == "SCALE_UP":
                try:
                    self._scale_position(w, scan_map)
                except Exception as e:
                    log.error(f"[CITRINE] ERROR: Failed to scale {w.ticker}: {e}")

    def _enter_position(self, w: PortfolioWeight, scan_map: dict):
        """Enter a new position with ATR-based sizing (Sprint 9 risk engine)."""
        # Deduplication guard: skip if already holding this ticker
        if w.ticker in self.positions:
            log.info(f"  Skip {w.ticker}: already held (dedup guard)")
            return

        scan = scan_map.get(w.ticker)
        if not scan or scan.current_price <= 0:
            log.warning(f"  Cannot enter {w.ticker}: no price data")
            return

        price = scan.current_price
        fill_price = self._simulate_fill(price, w.direction)

        # ATR-based position sizing (Sprint 9): dollar_risk / (ATR × multiplier)
        # Falls back to allocator-provided notional if ATR unavailable
        entry_atr = getattr(scan, "current_atr", 0.0) or 0.0
        if entry_atr > 0:
            equity = self.cash + sum(
                p.shares * p.entry_price for p in self.positions.values())
            dollar_risk = equity * self.ATR_RISK_BUDGET
            stop_distance = entry_atr * self.CHANDELIER_MULTIPLIER
            atr_notional = (dollar_risk / stop_distance) * fill_price
            max_notional = equity * self.ATR_MAX_NOTIONAL_PCT
            notional = min(atr_notional, max_notional, w.notional_usd, self.cash)
            log.info(f"  [ATR sizing] {w.ticker}: ATR=${entry_atr:.2f}, "
                     f"risk=${dollar_risk:.0f}, atr_notional=${atr_notional:.0f} "
                     f"→ ${notional:.0f}")
        else:
            notional = min(w.notional_usd, self.cash)

        if notional < 100:  # Minimum position size $100
            log.info(f"  Skip {w.ticker}: notional ${notional:.0f} below minimum")
            return

        shares = notional / fill_price

        try:
            # Create position with risk engine fields
            pos = CitrinePosition(
                ticker=w.ticker,
                direction=w.direction,
                shares=shares,
                entry_price=fill_price,
                target_weight=w.scaled_weight,
                entry_atr=entry_atr,
                entry_confidence=scan.confidence if scan else 0.0,
            )

            # Update cash
            fee = notional * TAKER_FEE
            self.cash -= (notional + fee)

            # Only add to positions after successful order simulation
            self.positions[w.ticker] = pos

            log.info(f"  {'🟢' if w.direction == 'LONG' else '🔴'} ENTER {w.ticker} "
                     f"{w.direction}: {shares:.2f} shares @ ${fill_price:.2f} "
                     f"(${notional:.0f}, weight {w.scaled_weight:.1%})")

            notify_trade("BUY", shares, fill_price, ticker=w.ticker, project="CITRINE")

            # Log trade with entry ATR
            self._log_trade(w.ticker, "ENTER", w.direction, shares, fill_price,
                            notional, None, None, w, scan, position=pos)
        except Exception as e:
            log.error(f"[CITRINE] ERROR: Failed to complete entry for {w.ticker}: {e}")
            # Roll back: remove position if it was added
            self.positions.pop(w.ticker, None)

    def _exit_position(self, w: PortfolioWeight, scan_map: dict,
                       exit_reason: str = "regime_flip"):
        """Exit an existing position with risk metrics (Sprint 9)."""
        pos = self.positions.get(w.ticker)
        if pos is None:
            return

        scan = scan_map.get(w.ticker)
        price = scan.current_price if scan and scan.current_price and scan.current_price > 0 else pos.entry_price

        # CRITICAL: never execute a trade at $0 — indicates a data fetch failure
        if price <= 0:
            log.error(f"[CITRINE] BLOCKED EXIT {w.ticker}: price is ${price:.2f} "
                      f"(data failure). Holding position until next cycle.")
            return

        try:
            fill_price = self._simulate_fill(price,
                                             "SELL" if pos.direction == "LONG" else "BUY")

            # Final excursion update before exit
            pos.update_excursions(fill_price)

            # Calculate P&L
            if pos.direction == "LONG":
                pnl = pos.shares * (fill_price - pos.entry_price)
            else:
                pnl = pos.shares * (pos.entry_price - fill_price)

            notional = pos.shares * fill_price
            fee = (notional + pos.notional) * TAKER_FEE  # Both sides
            pnl -= fee
            pnl_pct = (pnl / pos.notional) * 100

            # Return cash
            self.cash += (notional - fee / 2)  # Only exit-side fee on cash return

            emoji = "✅" if pnl >= 0 else "❌"
            log.info(f"  {emoji} EXIT {w.ticker} {pos.direction}: "
                     f"{pos.shares:.2f} shares @ ${fill_price:.2f} | "
                     f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) | "
                     f"MAE:{pos.mae_pct:+.1%} MFE:{pos.mfe_pct:+.1%} [{exit_reason}]")

            notify_trade("SELL", pos.shares, fill_price, pnl=pnl, ticker=w.ticker, project="CITRINE")

            # Log trade with risk metrics
            self._log_trade(w.ticker, "EXIT", pos.direction, pos.shares,
                            fill_price, notional, pnl, pnl_pct, w, scan,
                            exit_reason=exit_reason, position=pos)

            # Remove position only after successful logging
            del self.positions[w.ticker]
        except Exception as e:
            log.error(f"[CITRINE] ERROR: Failed to complete exit for {w.ticker}: {e}")

    def _scale_position(self, w: PortfolioWeight, scan_map: dict):
        """Scale an existing position up (gradual scaling day 1→3→5)."""
        pos = self.positions.get(w.ticker)
        if pos is None:
            return

        scan = scan_map.get(w.ticker)
        if not scan or not scan.current_price or scan.current_price <= 0:
            return

        # Calculate target notional vs current
        target_notional = min(w.notional_usd, config.CITRINE_MAX_NOTIONAL)
        current_notional = pos.shares * scan.current_price
        delta = target_notional - current_notional

        if delta < 100:  # Not worth scaling less than $100
            log.info(f"  ➡️  HOLD {w.ticker} {pos.direction}: "
                     f"${current_notional:.0f} (scale delta < $100)")
            return

        # Cap delta at available cash (prevent negative cash)
        delta = min(delta, self.cash * 0.95)  # Keep 5% buffer
        if delta < 100:
            log.info(f"  ➡️  HOLD {w.ticker} {pos.direction}: "
                     f"insufficient cash for scale (${self.cash:.0f} avail)")
            return

        # Add to position
        price = scan.current_price
        fill_price = self._simulate_fill(price, pos.direction)
        additional_shares = delta / fill_price

        pos.shares += additional_shares
        fee = delta * TAKER_FEE
        self.cash -= (delta + fee)

        log.info(f"  📈 SCALE {w.ticker} {pos.direction}: +{additional_shares:.2f} shares "
                 f"@ ${fill_price:.2f} (${delta:.0f}, total ${target_notional:.0f})")

        self._log_trade(w.ticker, "SCALE_UP", pos.direction, additional_shares,
                        fill_price, delta, None, None, w, scan)

    def _simulate_fill(self, price: float, side: str) -> float:
        """Add simulated slippage to price."""
        slippage = price * (SLIPPAGE_BPS / 10000)
        if side in ("LONG", "BUY"):
            return price + slippage  # Worse fill for buys
        return price - slippage  # Worse fill for sells

    # ── Logging ────────────────────────────────────────────────────────────

    def _log_trade(
        self, ticker: str, action: str, direction: str,
        shares: float, price: float, notional: float,
        pnl: float | None, pnl_pct: float | None,
        weight: PortfolioWeight, scan: TickerScan | None,
        exit_reason: str = "",
        position: CitrinePosition | None = None,
    ):
        """Log a trade to SQLite with risk engine metrics (Sprint 9)."""
        now = datetime.now(tz=timezone.utc).isoformat()
        side = "BUY" if direction == "LONG" else "SELL"
        if action in ("EXIT", "STOP_EXIT"):
            side = "SELL" if direction == "LONG" else "BUY"

        # Extract risk metrics from position (available on exits)
        entry_atr = getattr(position, "entry_atr", None) if position else None
        mae_pct = getattr(position, "mae_pct", None) if position else None
        mfe_pct = getattr(position, "mfe_pct", None) if position else None
        mae_atr = getattr(position, "mae_atr", None) if position else None
        mfe_atr = getattr(position, "mfe_atr", None) if position else None
        entry_conf = getattr(position, "entry_confidence", None) if position else None

        # Hold duration
        hold_days = None
        if position and hasattr(position, "entry_time"):
            try:
                entry_dt = datetime.fromisoformat(
                    position.entry_time.replace("Z", "+00:00"))
                hold_days = (datetime.now(tz=timezone.utc) - entry_dt).days
            except (ValueError, TypeError):
                pass

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO trades
                (timestamp, ticker, side, direction, action, shares, price,
                 notional, pnl, pnl_pct, signal_strength, portfolio_weight,
                 regime, confidence, entry_atr, exit_reason, mae_pct, mfe_pct,
                 mae_atr, mfe_atr, entry_confidence, hold_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now, ticker, side, direction, action,
                    round(shares, 4), round(price, 2),
                    round(notional, 2),
                    round(pnl, 2) if pnl is not None else None,
                    round(pnl_pct, 2) if pnl_pct is not None else None,
                    weight.confidence if hasattr(weight, 'confidence') else 0,
                    round(weight.scaled_weight, 4) if weight else 0,
                    scan.regime_cat if scan else "",
                    round(scan.confidence, 3) if scan else 0,
                    round(entry_atr, 4) if entry_atr else None,
                    exit_reason or None,
                    round(mae_pct, 6) if mae_pct is not None else None,
                    round(mfe_pct, 6) if mfe_pct is not None else None,
                    round(mae_atr, 4) if mae_atr is not None else None,
                    round(mfe_atr, 4) if mfe_atr is not None else None,
                    round(entry_conf, 4) if entry_conf is not None else None,
                    hold_days,
                ),
            )
            conn.commit()

    def _log_snapshot(
        self, scans: list[TickerScan], cash_pct: float,
        bull_count: int, bear_count: int, chop_count: int,
    ):
        """Log portfolio snapshot to SQLite."""
        try:
            now = datetime.now(tz=timezone.utc).isoformat()

            # Build scan price lookup (O(1) per ticker instead of O(n) nested loop)
            scan_prices = {s.ticker: s.current_price for s in scans
                           if s.current_price and s.current_price > 0}

            # Calculate portfolio equity
            invested = 0.0
            pos_data = {}
            for ticker, pos in self.positions.items():
                # Use scan price; only fall back to entry_price as last resort
                current_price = scan_prices.get(ticker, pos.entry_price)
                if pos.direction == "LONG":
                    value = pos.shares * current_price
                else:
                    value = pos.notional + (pos.entry_price - current_price) * pos.shares
                invested += value
                pos_data[ticker] = {
                    "direction": pos.direction,
                    "shares": round(pos.shares, 4),
                    "entry": round(pos.entry_price, 2),
                    "current": round(current_price, 2),
                    "value": round(value, 2),
                    # Risk engine fields (Sprint 9)
                    "entry_atr": round(pos.entry_atr, 4) if pos.entry_atr else 0.0,
                    "entry_confidence": round(pos.entry_confidence, 4),
                    "high_watermark": round(pos.high_watermark, 2),
                    "low_watermark": round(pos.low_watermark, 2),
                    "mae_pct": round(pos.mae_pct, 6),
                    "mfe_pct": round(pos.mfe_pct, 6),
                }

            total_equity = self.cash + invested
            # Store computed equity for use in notifications
            self._last_computed_equity = total_equity
            num_long = sum(1 for p in self.positions.values() if p.direction == "LONG")
            num_short = sum(1 for p in self.positions.values() if p.direction == "SHORT")

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO portfolio_snapshots
                    (timestamp, total_equity, cash, invested, num_positions,
                     num_long, num_short, bull_count, bear_count, chop_count,
                     positions_json, cash_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        now, round(total_equity, 2), round(self.cash, 2),
                        round(invested, 2), len(self.positions),
                        num_long, num_short, bull_count, bear_count, chop_count,
                        json.dumps(pos_data), round(cash_pct, 4),
                    ),
                )
                conn.commit()

            log.info(f"\n  Portfolio: ${total_equity:,.2f} "
                     f"(cash ${self.cash:,.0f} + invested ${invested:,.0f})")
        except Exception as e:
            log.error(f"[CITRINE] ERROR: Failed to log portfolio snapshot: {e}")

    def _log_intraday_snapshot(self, price_map: dict[str, float]) -> None:
        """Lightweight snapshot using prices already fetched during risk check.

        Reuses cached regime counts from the last full scan (regimes only change
        on daily bar close, so carrying forward is correct). This enables the
        dashboard to show intraday P&L updates without re-running HMMs.
        """
        try:
            now = datetime.now(tz=timezone.utc).isoformat()

            invested = 0.0
            pos_data = {}
            for ticker, pos in self.positions.items():
                current_price = price_map.get(ticker, pos.entry_price)
                if pos.direction == "LONG":
                    value = pos.shares * current_price
                else:
                    value = pos.notional + (pos.entry_price - current_price) * pos.shares
                invested += value
                pos_data[ticker] = {
                    "direction": pos.direction,
                    "shares": round(pos.shares, 4),
                    "entry": round(pos.entry_price, 2),
                    "current": round(current_price, 2),
                    "value": round(value, 2),
                    "entry_atr": round(pos.entry_atr, 4) if pos.entry_atr else 0.0,
                    "entry_confidence": round(pos.entry_confidence, 4),
                    "high_watermark": round(pos.high_watermark, 2),
                    "low_watermark": round(pos.low_watermark, 2),
                    "mae_pct": round(pos.mae_pct, 6),
                    "mfe_pct": round(pos.mfe_pct, 6),
                }

            total_equity = self.cash + invested
            self._last_computed_equity = total_equity
            num_long = sum(1 for p in self.positions.values() if p.direction == "LONG")
            num_short = sum(1 for p in self.positions.values() if p.direction == "SHORT")

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO portfolio_snapshots
                    (timestamp, total_equity, cash, invested, num_positions,
                     num_long, num_short, bull_count, bear_count, chop_count,
                     positions_json, cash_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        now, round(total_equity, 2), round(self.cash, 2),
                        round(invested, 2), len(self.positions),
                        num_long, num_short,
                        self._last_bull_count, self._last_bear_count,
                        self._last_chop_count,
                        json.dumps(pos_data), round(self._last_cash_pct, 4),
                    ),
                )
                conn.commit()

            log.info(f"  [Intraday] Snapshot: ${total_equity:,.2f} "
                     f"(cash ${self.cash:,.0f} + invested ${invested:,.0f})")
        except Exception as e:
            log.error(f"[CITRINE] ERROR: Failed to log intraday snapshot: {e}")

    def _print_portfolio_summary(self):
        """Print current holdings to terminal."""
        if not self.positions:
            log.info("  No open positions")
            return

        log.info("\n  Current Holdings:")
        for ticker, pos in sorted(self.positions.items()):
            emoji = "🟢" if pos.direction == "LONG" else "🔴"
            log.info(f"    {emoji} {ticker:6s} {pos.direction:5s} "
                     f"{pos.shares:.2f} shares @ ${pos.entry_price:.2f} "
                     f"(${pos.notional:.0f})")

    # ── Notifications ──────────────────────────────────────────────────────

    def _notify_summary(
        self, weights: list[PortfolioWeight], cash_pct: float, bull_count: int
    ):
        """Send daily summary notification."""
        enters = sum(1 for w in weights if w.action == "ENTER")
        exits = sum(1 for w in weights if w.action == "EXIT")
        holds = sum(1 for w in weights if w.action in ("HOLD", "SCALE_UP"))

        # Use mark-to-market equity from _log_snapshot if available
        total_equity = getattr(self, "_last_computed_equity", None)
        if total_equity is None:
            total_equity = self.cash + sum(
                p.shares * p.entry_price for p in self.positions.values()
            )

        title = "CITRINE Daily Scan"
        msg = (f"{bull_count} BULL | {len(self.positions)} positions | "
               f"{enters} enter {exits} exit | "
               f"Equity: ${total_equity:,.0f} | Cash: {cash_pct:.0%}")

        _macos_notify(title, msg, sound="Pop")
        _pushover_notify(title, msg, priority=-1)

    # ── Kill-Switch ────────────────────────────────────────────────────────

    # ── Sprint 3.5: Intra-Day Risk Monitoring ──────────────────────────────

    def _sleep_with_risk_checks(self, total_seconds: int) -> None:
        """Sleep for total_seconds, waking for risk checks + intraday snapshots.

        During US market hours (9:30am-4pm ET): check every 1 hour
        Outside market hours: check every 4 hours
        Each check fetches current prices, runs risk logic, and writes a
        snapshot so the dashboard shows intraday P&L updates.
        """
        MARKET_INTERVAL = 1 * 3600     # 1 hour during market hours
        OFF_HOURS_INTERVAL = 4 * 3600  # 4 hours outside market hours
        remaining = total_seconds

        while remaining > 0:
            # Determine interval based on whether US market is open
            now_utc = datetime.now(tz=timezone.utc)
            # US Eastern: UTC-4 (EDT) or UTC-5 (EST)
            # Approximate: EDT Mar-Nov, EST Nov-Mar
            et_offset = -4 if 3 <= now_utc.month <= 10 else -5
            et_hour = (now_utc.hour + et_offset) % 24
            is_market_hours = (9 <= et_hour < 16) and (now_utc.weekday() < 5)

            interval = MARKET_INTERVAL if is_market_hours else OFF_HOURS_INTERVAL
            sleep_time = min(remaining, interval)
            time.sleep(sleep_time)
            remaining -= sleep_time

            if remaining > 0:
                # Still between daily cycles — run intraday risk check + snapshot
                self._check_intraday_risk()

    # ── Trailing stop-loss threshold ─────────────────────────────────────
    TRAILING_STOP_PCT = -0.02  # -2% per-position trailing stop

    def _check_intraday_risk(self) -> None:
        """
        Intra-day risk check — called every 4 hours during sleep.
        Fetches latest prices for held tickers and checks unrealized P&L.

        Per-position trailing stop (Sprint 9):
          - Any position unrealized loss > 2% → force-exit immediately
        Warnings:
          - Portfolio unrealized loss > 3%
        Kill-switch trigger:
          - Total unrealized loss > 5% of capital
        """
        if not self.positions:
            return

        log.info("  [Intraday] Checking unrealized P&L for %d positions...",
                 len(self.positions))

        total_unrealized = 0.0
        warnings = []
        stop_loss_exits = []  # tickers to force-exit
        intraday_prices: dict[str, float] = {}  # for snapshot

        for ticker, pos in list(self.positions.items()):
            from src.data_fetcher import fetch_latest_price
            current_price = fetch_latest_price(ticker)
            if current_price is None or current_price <= 0:
                current_price = pos.entry_price  # fallback
            intraday_prices[ticker] = current_price

            # Update excursion watermarks (Sprint 9)
            pos.update_excursions(current_price)

            # Compute unrealized P&L
            if pos.direction == "LONG":
                unrealized = pos.shares * (current_price - pos.entry_price)
            else:  # SHORT
                unrealized = pos.shares * (pos.entry_price - current_price)

            unrealized_pct = unrealized / pos.notional if pos.notional > 0 else 0
            total_unrealized += unrealized

            # Chandelier stop check (Sprint 9) — takes priority over pct stop
            stop = pos.chandelier_stop(self.CHANDELIER_MULTIPLIER)
            if stop is not None:
                if ((pos.direction == "LONG" and current_price < stop) or
                        (pos.direction == "SHORT" and current_price > stop)):
                    msg = (f"  [CHANDELIER] {ticker} breached stop ${stop:.2f} "
                           f"(price=${current_price:.2f}, ATR=${pos.entry_atr:.2f})")
                    log.warning(msg)
                    warnings.append(msg)
                    stop_loss_exits.append((ticker, current_price))
                    continue

            # Fallback pct trailing stop (for positions without ATR)
            if unrealized_pct < self.TRAILING_STOP_PCT:
                msg = (f"  [STOP-LOSS] {ticker} hit {unrealized_pct:.1%} "
                       f"(${unrealized:.2f}) — breached {self.TRAILING_STOP_PCT:.0%} stop")
                log.warning(msg)
                warnings.append(msg)
                stop_loss_exits.append((ticker, current_price))
            elif unrealized_pct < -0.01:
                # Warn at -1% (early warning before stop triggers)
                msg = (f"  [Intraday] WATCH: {ticker} unrealized {unrealized_pct:.1%} "
                       f"(${unrealized:.2f})")
                log.info(msg)

        # Execute trailing stop-loss exits
        if stop_loss_exits:
            log.warning(f"  [STOP-LOSS] Exiting {len(stop_loss_exits)} positions: "
                        f"{[t for t, _ in stop_loss_exits]}")
            for ticker, price in stop_loss_exits:
                try:
                    self._force_exit_single(ticker, price, reason="trailing_stop")
                except Exception as e:
                    log.error(f"  [STOP-LOSS] Failed to exit {ticker}: {e}")

            _pushover_notify(
                "CITRINE Stop-Loss",
                f"Exited {len(stop_loss_exits)} positions: "
                + ", ".join(f"{t} @ ${p:.2f}" for t, p in stop_loss_exits),
                priority=0,
            )

        # Portfolio-level check (recalculate after any stop-loss exits)
        if self.positions:
            total_unrealized = 0.0
            for ticker, pos in self.positions.items():
                current_price = fetch_latest_price(ticker)
                if current_price is None or current_price <= 0:
                    current_price = pos.entry_price
                if pos.direction == "LONG":
                    total_unrealized += pos.shares * (current_price - pos.entry_price)
                else:
                    total_unrealized += pos.shares * (pos.entry_price - current_price)

        portfolio_pct = total_unrealized / self.capital if self.capital > 0 else 0

        if portfolio_pct < -0.03:
            msg = (f"  [Intraday] WARNING: Portfolio unrealized {portfolio_pct:.1%} "
                   f"(${total_unrealized:.2f})")
            log.warning(msg)
            warnings.append(msg)

        # Kill-switch: total unrealized loss > 5% of capital
        if portfolio_pct < -0.05:
            reason = (f"CITRINE INTRADAY: Unrealized loss {portfolio_pct:.1%} "
                      f"(${total_unrealized:.2f}) exceeds 5% of capital")
            log.error(f"KILL-SWITCH: {reason}")
            self._kill_switch_alert(reason)
            self._emergency_exit_all(reason)
            return

        # Log intraday snapshot for dashboard visibility
        if intraday_prices:
            self._log_intraday_snapshot(intraday_prices)

        # Send notifications for warnings
        if warnings:
            _macos_notify("CITRINE Risk Warning", "\n".join(warnings))
        else:
            log.info("  [Intraday] All positions within risk limits "
                     f"(unrealized: {portfolio_pct:+.2%})")

    def _force_exit_single(self, ticker: str, current_price: float,
                           reason: str = "stop_loss") -> None:
        """Force-exit a single position (Chandelier stop, conf velocity, or risk limit)."""
        pos = self.positions.get(ticker)
        if pos is None:
            return

        fill_price = self._simulate_fill(
            current_price, "SELL" if pos.direction == "LONG" else "BUY")

        # Final excursion update
        pos.update_excursions(fill_price)

        if pos.direction == "LONG":
            pnl = pos.shares * (fill_price - pos.entry_price)
        else:
            pnl = pos.shares * (pos.entry_price - fill_price)

        notional = pos.shares * fill_price
        fee = (notional + pos.notional) * TAKER_FEE
        pnl -= fee
        pnl_pct = (pnl / pos.notional) * 100

        self.cash += (notional - fee / 2)

        log.warning(f"  🛑 STOP-EXIT {ticker}: {pos.shares:.2f} shares @ ${fill_price:.2f} "
                    f"P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) | "
                    f"MAE:{pos.mae_pct:+.1%} MFE:{pos.mfe_pct:+.1%} [{reason}]")

        # Log to DB with risk metrics
        scan_stub = TickerScan(
            ticker=ticker, regime_cat="STOP", confidence=0.0,
            persistence=0, realized_vol=0.0, confirmations=0,
            confirmations_short=0, current_price=current_price,
            sector="", hmm_converged=False,
        )
        w_stub = PortfolioWeight(
            ticker=ticker, direction="FLAT", raw_score=0.0,
            target_weight=0.0, scaled_weight=0.0, notional_usd=0.0,
            days_held=0, action="STOP_EXIT",
        )
        self._log_trade(ticker, "STOP_EXIT", pos.direction, pos.shares,
                        fill_price, notional, pnl, pnl_pct, w_stub, scan_stub,
                        exit_reason=reason, position=pos)

        del self.positions[ticker]

    def _emergency_exit_all(self, reason: str) -> None:
        """Emergency exit all positions when kill-switch triggers."""
        log.error("EMERGENCY EXIT: Closing all %d positions", len(self.positions))
        failed_tickers = []
        for ticker in list(self.positions.keys()):
            try:
                pos = self.positions[ticker]
                # Build a minimal PortfolioWeight for the exit call
                exit_w = PortfolioWeight(
                    ticker=ticker, direction="FLAT", raw_score=0.0,
                    target_weight=0.0, scaled_weight=0.0, notional_usd=0.0,
                    days_held=0, action="EXIT",
                )
                self._exit_position(exit_w, {ticker: TickerScan(
                    ticker=ticker, regime_cat="CHOP", confidence=0.0,
                    persistence=0, realized_vol=0.0, confirmations=0,
                    confirmations_short=0, current_price=pos.entry_price,
                    sector="", hmm_converged=False,
                )})
            except Exception as e:
                log.error("[CITRINE] ERROR: Failed to emergency-exit %s: %s", ticker, e)
                failed_tickers.append(ticker)
        # Clear successfully exited positions (failed ones remain for retry)
        for ticker in list(self.positions.keys()):
            if ticker not in failed_tickers:
                self.positions.pop(ticker, None)
        if failed_tickers:
            log.error("[CITRINE] ERROR: Emergency exit failed for: %s", ", ".join(failed_tickers))

    def _check_kill_switch(self) -> bool:
        """
        Check portfolio-level kill-switch rules.

        Rules:
          1. Total loss > 5% of initial capital → exit all
          2. Rolling 20-trade Sharpe < 0.3 → exit all
          3. 0 wins in last 10 closed trades → stop
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get closed trades (EXITs with P&L)
            rows = conn.execute(
                "SELECT pnl FROM trades WHERE action = 'EXIT' AND pnl IS NOT NULL "
                "ORDER BY timestamp DESC LIMIT 20"
            ).fetchall()

        if not rows:
            return False

        pnls = [r[0] for r in rows]

        # Rule 1: Total P&L loss > 5% of capital
        total_pnl = sum(pnls)
        threshold = self.capital * 0.05
        if total_pnl < -threshold:
            reason = (f"CITRINE: Total loss ${total_pnl:.2f} exceeds 5% "
                      f"of capital (${threshold:.0f})")
            log.error(f"KILL-SWITCH: {reason}")
            self._kill_switch_alert(reason)
            return True

        # Rule 2: Rolling 20-trade Sharpe < 0.3
        if len(pnls) >= 20:
            sharpe = np.mean(pnls) / (np.std(pnls) + 1e-9)
            if sharpe < 0.3:
                reason = f"CITRINE: Rolling 20-trade Sharpe {sharpe:.3f} < 0.3"
                log.error(f"KILL-SWITCH: {reason}")
                self._kill_switch_alert(reason)
                return True

        # Rule 3: 0 wins in last 10 trades
        if len(pnls) >= 10:
            recent = pnls[:10]
            if all(p <= 0 for p in recent):
                reason = "CITRINE: 0/10 winning trades"
                log.error(f"KILL-SWITCH: {reason}")
                self._kill_switch_alert(reason)
                return True

        return False

    def _kill_switch_alert(self, reason: str):
        """Send kill-switch alerts on all channels."""
        from src.notifier import notify_kill_switch
        notify_kill_switch(reason)


# ── Entry Point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CITRINE: NDX100 Portfolio Rotation — Live Trading"
    )
    parser.add_argument("--test", action="store_true", default=True,
                        help="Run in test mode (simulated fills)")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated ticker list (default: full NDX100)")
    parser.add_argument("--long-only", action="store_true", default=False,
                        help="Long-only mode (no SHORT positions)")
    parser.add_argument("--cooldown", type=str, default="none",
                        choices=["none", "time", "threshold"],
                        help="Cooldown mode (default: none)")
    args = parser.parse_args()

    tickers = args.tickers.split(",") if args.tickers else None

    # Handle graceful shutdown
    def _sigterm_handler(signum, frame):
        log.info("SIGTERM received — shutting down CITRINE")
        sys.exit(0)
    signal_mod.signal(signal_mod.SIGTERM, _sigterm_handler)

    engine = CitrineLiveEngine(
        tickers=tickers,
        test_mode=args.test,
        long_only=args.long_only,
        cooldown_mode=args.cooldown,
    )
    engine.run_forever()


if __name__ == "__main__":
    main()
