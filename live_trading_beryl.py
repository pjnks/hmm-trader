"""
live_trading_beryl.py
─────────────────────
BERYL: NDX100 live trading engine (test mode) with multi-ticker rotation.

Scans 82 NDX100 tickers daily, picks the strongest BUY signals.
Holds up to MAX_POSITIONS positions simultaneously (multi-position rotation).

  - Loads per-ticker optimized HMM configs from beryl_per_ticker_configs.json
  - Fetches daily bars from Polygon.io for each ticker (365-day lookback)
  - Fits EnsembleHMM per ticker (3 models with majority voting)
  - Falls back to single HMMRegimeModel(n_states=4, cov_type="diag") if ensemble fails
  - Picks strongest BUY signals across all tickers (by confidence x confirmations)
  - If a held position goes BEAR -> sells, potentially enters different tickers
  - Simulates trades with 10bps slippage
  - Logs to beryl_trades.db

Usage
─────
  python live_trading_beryl.py --test                                 # All 82 tickers (default)
  python live_trading_beryl.py --test --tickers NVDA,TSLA,GOOGL,MSFT,AAPL  # custom subset
  python live_trading_beryl.py --test --ticker NVDA                   # single-ticker (legacy)
  python live_trading_beryl.py --test --min-confirmations 6           # override confirmation threshold

Note: Equities only trade Mon-Fri 9:30am-4pm ET. Signals are checked
once per day after market close (4:30pm ET).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import signal
import sys
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.data_fetcher import build_hmm_features
from src.hmm_model import HMMRegimeModel
from src.ensemble import EnsembleHMM
from src.indicators import (
    attach_all,
    compute_realized_vol_ratio,
    compute_return_autocorr,
    compute_candle_body_ratio,
    compute_bb_width,
    compute_realized_kurtosis,
    compute_volume_return_intensity,
    compute_return_momentum_ratio,
)
from src.strategy import build_signal_series
from src.notifier import notify_trade, notify_signal, notify_kill_switch, _macos_notify

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("beryl_live")


# ── BERYL Default Config (fallback for tickers not in per-ticker JSON) ────
BERYL_DEFAULT_CONFIG = {
    "n_states": 4,
    "feature_set": "base",
    "confirmations": 5,
    "cov_type": "diag",  # diag converges reliably with ~90 daily obs
}

# Trading params (not per-ticker)
BERYL_LEVERAGE = 1.5
BERYL_COOLDOWN_HOURS = 48

# Position sizing (conservative — same philosophy as AGATE)
MAX_NOTIONAL = 2_500  # $2,500 max total notional
MAX_POSITIONS = 3     # Up to 3 simultaneous positions
MAX_NOTIONAL_PER_POS = MAX_NOTIONAL // MAX_POSITIONS  # ~$833 per position
INITIAL_EQUITY = 10_000
SLIPPAGE_BPS = 10  # 10bps simulated slippage
TAKER_FEE = 0.0004  # 0.04% per side (typical equity commission)

# Signal check interval (seconds)
# Equities: check once per day after market close
SIGNAL_INTERVAL_S = 24 * 3600  # 24 hours

# Lookback for HMM training (trading days after warmup)
LOOKBACK_DAYS = 365  # ~250 trading days — robust for ensemble HMMs

# Global confirmation override (set via --min-confirmations CLI arg)
_GLOBAL_MIN_CONFIRMATIONS: int | None = None


class BerylPosition:
    """Track an open equity position."""
    def __init__(self, ticker: str, side: str, size: float, entry_price: float):
        self.ticker = ticker
        self.side = side
        self.size = size
        self.entry_price = entry_price
        self.entry_time = datetime.now(tz=timezone.utc).isoformat()
        self.notional = size * entry_price


class BerylLiveEngine:
    """
    BERYL live trading engine for NDX100 equities.
    Scans 82 tickers daily with per-ticker optimized HMM configs.
    Picks strongest BUY signals, holds up to MAX_POSITIONS positions (rotation).
    Uses EnsembleHMM (3 models, majority voting) with single-model fallback.
    """

    def __init__(self, tickers: list[str] | None = None, test_mode: bool = True):
        self.test_mode = test_mode
        self.positions: dict[str, BerylPosition] = {}
        self.cooldowns: dict[str, datetime] = {}
        self.db_path = ROOT / "beryl_trades.db"
        self._init_db()

        # Kill-switch grace period: skip checks for first N cycles after restart
        self._cycles_since_restart = 0
        self._KILL_SWITCH_GRACE_CYCLES = 3

        # Load per-ticker optimized HMM configs from BERYL optimization
        self._per_ticker_params: dict[str, dict] = {}
        self._load_per_ticker_configs()

        # Set ticker universe: CLI override > NDX100 universe > beryl configs > fallback
        # BERYL scans all NDX100 tickers; beryl_per_ticker_configs has optimized params
        # for a subset, the rest use BERYL_DEFAULT_CONFIG fallback.
        if tickers:
            self.tickers = tickers
        else:
            self.tickers = self._load_ticker_universe()

        # Apply sensible defaults to global config
        config.LEVERAGE = BERYL_LEVERAGE
        config.COOLDOWN_HOURS = BERYL_COOLDOWN_HOURS

        log.info(f"BERYL Engine initialized: {len(self.tickers)} tickers ({'TEST' if test_mode else 'LIVE'})")
        log.info(f"Per-ticker configs loaded: {len(self._per_ticker_params)} tickers")
        log.info(f"Default fallback: {BERYL_DEFAULT_CONFIG}")
        log.info(f"Max positions: {MAX_POSITIONS}, notional per position: ${MAX_NOTIONAL_PER_POS:,}")

    def _load_per_ticker_configs(self) -> None:
        """Load per-ticker optimized HMM configs from BERYL optimization results."""
        config_path = ROOT / "beryl_per_ticker_configs.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    self._per_ticker_params = json.load(f)
                log.info(f"Loaded per-ticker configs for {len(self._per_ticker_params)} tickers "
                         f"from beryl_per_ticker_configs.json")
            except Exception as e:
                log.warning(f"Failed to load per-ticker configs: {e}")
        else:
            log.warning("beryl_per_ticker_configs.json not found — using defaults for all tickers")

    def _load_ticker_universe(self) -> list[str]:
        """Load full NDX100 ticker universe from citrine configs, fall back to beryl configs.

        BERYL scans the full NDX100 universe (same as CITRINE). The beryl_per_ticker_configs
        only covers tickers with positive optimizer results; the rest use default fallback.
        """
        universe_path = ROOT / "citrine_per_ticker_configs.json"
        if universe_path.exists():
            try:
                with open(universe_path) as f:
                    data = json.load(f)
                tickers = sorted(data.keys())
                if tickers:
                    log.info(f"NDX100 universe: {len(tickers)} tickers "
                             f"({len(self._per_ticker_params)} with BERYL-optimized configs)")
                    return tickers
            except Exception as e:
                log.warning(f"Failed to load NDX100 universe: {e}")
        # Fall back to beryl configs keys, then hardcoded defaults
        if self._per_ticker_params:
            return sorted(self._per_ticker_params.keys())
        return ["NVDA", "TSLA", "GOOGL", "MSFT", "AAPL"]

    def _get_ticker_config(self, ticker: str) -> dict:
        """Get HMM config for a specific ticker (per-ticker or fallback).

        Memory-safe override: forces diag covariance on the constrained VM.
        Full covariance with 6 states + n_init=3 causes OOM on 1GB RAM.

        If --min-confirmations was passed via CLI, it overrides per-ticker thresholds.
        """
        tc = self._per_ticker_params.get(ticker, BERYL_DEFAULT_CONFIG).copy()
        # Force diag on VM — full covariance OOMs with 98 tickers on 1GB RAM
        tc["cov_type"] = "diag"
        # Apply global CLI override if set
        if _GLOBAL_MIN_CONFIRMATIONS is not None:
            tc["confirmations"] = _GLOBAL_MIN_CONFIRMATIONS
        return tc

    def _init_db(self):
        """Initialize SQLite database for BERYL trade logging."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    signal_strength INTEGER NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scan_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    confirmations INTEGER NOT NULL,
                    min_confirmations INTEGER NOT NULL,
                    signal TEXT NOT NULL,
                    close_price REAL NOT NULL,
                    config_used TEXT,
                    UNIQUE(scan_date, ticker)
                )
            """)
            conn.commit()

    def _log_scan_journal(self, signals: list[dict]) -> None:
        """Log all scan results to scan_journal table for prediction scoring."""
        scan_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        rows = []
        for s in signals:
            rows.append((
                scan_date,
                s["ticker"],
                s["regime"],
                round(s["confidence"], 4),
                s["confirmations"],
                s.get("min_confirmations", 5),
                s["signal"],
                s.get("current_price", 0.0),
                s.get("config_used", "unknown"),
            ))
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany("""
                    INSERT OR REPLACE INTO scan_journal
                    (scan_date, ticker, regime, confidence, confirmations,
                     min_confirmations, signal, close_price, config_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rows)
                conn.commit()
            log.info(f"  [Journal] Logged {len(rows)} scan results for {scan_date}")
        except Exception as e:
            log.warning(f"  [Journal] Failed to log scan: {e}")

    def _fetch_data(self, ticker: str) -> pd.DataFrame:
        """Fetch recent equity data from Polygon (365-day lookback)."""
        from walk_forward_ndx import fetch_equity_daily
        df = fetch_equity_daily(ticker, years=2)
        if df.empty:
            return df
        # Keep only last LOOKBACK_DAYS
        cutoff = df.index[-1] - pd.Timedelta(days=LOOKBACK_DAYS)
        return df[df.index >= cutoff]

    def _generate_signal_for_ticker(self, ticker: str) -> dict | None:
        """Generate trading signal for a single ticker using EnsembleHMM with fallback."""
        try:
            df = self._fetch_data(ticker)
            if df.empty or len(df) < 60:
                log.warning(f"{ticker}: Insufficient data ({len(df)} bars)")
                return None

            # Get per-ticker HMM config
            tc = self._get_ticker_config(ticker)
            feature_set = tc.get("feature_set", "base")
            n_states = tc.get("n_states", 4)
            cov_type = tc.get("cov_type", "diag")
            min_confirms = tc.get("confirmations", 5)

            # Build HMM features
            df = build_hmm_features(df)

            # Add extended features based on per-ticker config (config-driven, Sprint 3.1)
            needed = set(config.FEATURE_SETS.get(feature_set, []))
            if "realized_vol_ratio" in needed:
                df["realized_vol_ratio"] = compute_realized_vol_ratio(df)
            if "return_autocorr" in needed:
                df["return_autocorr"] = compute_return_autocorr(df)
            if "candle_body_ratio" in needed:
                df["candle_body_ratio"] = compute_candle_body_ratio(df)
            if "bb_width" in needed:
                df["bb_width"] = compute_bb_width(df)
            if "realized_kurtosis" in needed:
                df["realized_kurtosis"] = compute_realized_kurtosis(df)
            if "volume_return_intensity" in needed:
                df["volume_return_intensity"] = compute_volume_return_intensity(df)
            if "return_momentum_ratio" in needed:
                df["return_momentum_ratio"] = compute_return_momentum_ratio(df)

            df = df.dropna()

            # Get feature columns for this ticker's feature set
            feature_cols = config.FEATURE_SETS.get(feature_set, config.FEATURE_SETS["base"])

            # Temporarily set global config for build_signal_series
            saved_confirms = config.MIN_CONFIRMATIONS
            config.MIN_CONFIRMATIONS = min_confirms

            try:
                # Fit EnsembleHMM with per-ticker params (3 models: N-1, N, N+1)
                n_states_list = [max(3, n_states - 1), n_states, n_states + 1]
                model = EnsembleHMM(
                    n_states_list=n_states_list,
                    cov_type=cov_type,
                    feature_cols=feature_cols,
                )
                model.fit(df)

                used_fallback = False
                if not model.converged:
                    # Fallback: single HMMRegimeModel with conservative params
                    log.warning(f"{ticker}: Ensemble did not converge ({n_states_list}/{feature_set}/{cov_type})"
                                f" — retrying with single HMM fallback (4st/diag)")
                    model = HMMRegimeModel(
                        n_states=4,
                        cov_type="diag",
                        feature_cols=feature_cols,
                    )
                    model.fit(df)
                    used_fallback = True

                    if not model.converged:
                        log.warning(f"{ticker}: Fallback HMM also did not converge")
                        return None

                # Predict regimes
                df = model.predict(df)

                # Attach indicators and build signals
                df = attach_all(df)
                df = build_signal_series(df, use_regime_mapper=False)
            finally:
                config.MIN_CONFIRMATIONS = saved_confirms

            # Get latest bar
            last = df.iloc[-1]
            current_price = float(last["Close"])

            # Count confirmations
            check_cols = [c for c in df.columns if c.startswith("check_")]
            confirmations = int(sum(last.get(c, 0) for c in check_cols))

            # Determine signal
            regime = str(last.get("regime_cat", "UNKNOWN"))
            confidence = float(last.get("confidence", 0.0))

            # Compute persistence: consecutive days in current regime
            persistence = self._compute_persistence(df, regime)

            # Determine signal from raw_long_signal + regime
            raw_long = bool(last.get("raw_long_signal", False))
            if raw_long:
                signal_str = "BUY"
            elif regime == "BEAR":
                signal_str = "SELL"
            else:
                signal_str = "HOLD"

            model_desc = "fallback-4st/diag" if used_fallback else f"ensemble-{n_states_list}"
            config_used = f"{model_desc}/{feature_set}/{cov_type}"

            return {
                "signal": signal_str,
                "regime": regime,
                "confidence": confidence,
                "confirmations": confirmations,
                "min_confirmations": min_confirms,
                "current_price": current_price,
                "ticker": ticker,
                "config_used": config_used,
                "persistence": persistence,
            }

        except Exception as e:
            log.error(f"{ticker}: Signal generation failed: {e}")
            return None

    @staticmethod
    def _compute_persistence(df: pd.DataFrame, current_regime: str) -> int:
        """Count consecutive days in `current_regime` from the end of df."""
        if "regime_cat" not in df.columns:
            return 0
        regimes = df["regime_cat"].values
        count = 0
        for i in range(len(regimes) - 1, -1, -1):
            if regimes[i] == current_regime:
                count += 1
            else:
                break
        return count

    def _scan_all_tickers(self) -> list[dict]:
        """
        Scan all tickers and return list of signals.
        Rate-limited: 12s between Polygon calls to avoid rate limits.
        82 tickers x 12s ~ 16 minutes per daily scan.
        """
        signals = []
        converged = 0
        for i, ticker in enumerate(self.tickers):
            if i > 0:
                time.sleep(12)  # Polygon rate limit
                gc.collect()  # Free HMM/DataFrame memory from previous ticker
            log.info(f"Scanning {ticker} ({i+1}/{len(self.tickers)})...")
            sig = self._generate_signal_for_ticker(ticker)
            if sig:
                signals.append(sig)
                converged += 1
                log.info(f"  {ticker}: {sig['signal']} | {sig['regime']} (conf {sig['confidence']:.2f}, {sig.get('persistence', 0)}d) "
                         f"| {sig['confirmations']}/{sig['min_confirmations']} | {sig['config_used']}")
        log.info(f"Scan complete: {converged}/{len(self.tickers)} converged, "
                 f"{sum(1 for s in signals if s['signal'] == 'BUY')} BUY, "
                 f"{sum(1 for s in signals if s['regime'] == 'BULL')} BULL, "
                 f"{sum(1 for s in signals if s['regime'] == 'BEAR')} BEAR")
        return signals

    def _pick_best_buys(self, signals: list[dict], n_slots: int) -> list[dict]:
        """
        Pick the strongest BUY signals from scanned tickers.
        Score = confidence x (confirmations / min_confirmations) x alt_data_boost.
        Uses per-ticker confirmation threshold in denominator.
        Alt-data boost (insider trading) used as tiebreaker/multiplier.
        Returns up to n_slots best BUY signals (excluding tickers already held or in cooldown).
        """
        # Filter out tickers we already hold and tickers in cooldown
        now = datetime.now(tz=timezone.utc)
        buy_signals = [
            s for s in signals
            if s["signal"] == "BUY"
            and s["ticker"] not in self.positions
            and not (self.cooldowns.get(s["ticker"]) and now < self.cooldowns[s["ticker"]])
        ]
        if not buy_signals:
            return []

        # Fetch alt-data boosts for BUY candidates only
        alt_boosts = {}
        try:
            from src.alternative_data import AlternativeDataScore
            scorer = AlternativeDataScore()
            buy_tickers = [s["ticker"] for s in buy_signals]
            alt_boosts = scorer.scan_tickers(buy_tickers, days=90)
            notable = {t: b for t, b in alt_boosts.items() if b != 1.0}
            if notable:
                for t, b in notable.items():
                    indicator = "+" if b > 1.0 else "-"
                    log.info(f"  [{indicator}] {t}: insider boost {b:.2f}x")
        except Exception as e:
            log.warning(f"Alt-data fetch failed: {e}")

        # Score: confidence x (confirmations / threshold) x alt_data_boost
        def score(s):
            try:
                base = s["confidence"] * (s["confirmations"] / s["min_confirmations"])
                boost = alt_boosts.get(s["ticker"], 1.0)
                # Persistence bonus: established regimes (15+ days) score 1.5x vs fresh flips (1.0x)
                persist = s.get("persistence", 0)
                persistence_bonus = 1.0 + min(max(persist - 3, 0) * 0.10, 0.50)
                return base * boost * persistence_bonus
            except Exception as e:
                log.warning(f"[BERYL] ERROR: Failed to score {s.get('ticker', '?')}: {e}")
                return 0.0

        buy_signals.sort(key=score, reverse=True)
        top_n = buy_signals[:n_slots]

        for sig in top_n:
            try:
                boost = alt_boosts.get(sig["ticker"], 1.0)
                boost_str = f", insider={boost:.2f}x" if boost != 1.0 else ""
                log.info(f"Selected BUY: {sig['ticker']} (score {score(sig):.3f}, conf {sig['confidence']:.2f}, "
                         f"{sig['confirmations']}/{sig['min_confirmations']}, {sig['config_used']}{boost_str})")
            except Exception as e:
                log.warning(f"[BERYL] ERROR: Failed to log BUY selection for {sig.get('ticker', '?')}: {e}")

        return top_n

    def _simulate_fill(self, price: float, side: str) -> float:
        """Add simulated slippage to price."""
        slippage = price * (SLIPPAGE_BPS / 10000)
        if side == "BUY":
            return price + slippage  # Worse fill for buys
        return price - slippage  # Worse fill for sells

    def _close_position(self, ticker: str, exit_price: float, confirmations: int) -> float:
        """Close a specific position by ticker. Returns P&L."""
        pos = self.positions.get(ticker)
        if pos is None:
            return 0.0

        try:
            fill_price = self._simulate_fill(exit_price, "SELL")
            pnl = pos.size * (fill_price - pos.entry_price)
            pnl -= pos.notional * TAKER_FEE * 2  # Both sides
            pnl_pct = (pnl / pos.notional) * 100

            log.info(f"SELL {pos.ticker}: {pos.size:.2f} shares @ ${fill_price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")

            self._log_trade(pos, fill_price, pnl, pnl_pct, confirmations)
        except Exception as e:
            log.error(f"[BERYL] ERROR: Failed to execute SELL order for {ticker}: {e}")
            return 0.0

        try:
            notify_trade("SELL", pos.size, fill_price, pnl=pnl, ticker=pos.ticker, project="BERYL")
        except Exception as e:
            log.warning(f"[BERYL] WARNING: SELL notification failed for {ticker}: {e}")

        # Per-ticker cooldown — only update state after successful order
        self.cooldowns[ticker] = datetime.now(tz=timezone.utc) + timedelta(hours=BERYL_COOLDOWN_HOURS)
        del self.positions[ticker]
        return pnl

    def _open_position(self, signal: dict) -> bool:
        """Open a new position from a BUY signal."""
        ticker = signal["ticker"]
        price = signal["current_price"]
        confidence = signal["confidence"]

        if confidence < 0.70:
            log.info(f"BUY {ticker} signal but confidence {confidence:.2f} < 0.70. Skipping.")
            return False

        if ticker in self.positions:
            log.info(f"BUY {ticker} signal but already holding. Skipping.")
            return False

        if len(self.positions) >= MAX_POSITIONS:
            log.info(f"BUY {ticker} signal but already at max positions ({MAX_POSITIONS}). Skipping.")
            return False

        try:
            fill_price = self._simulate_fill(price, "BUY")
            size = MAX_NOTIONAL_PER_POS / fill_price

            pos = BerylPosition(
                ticker=ticker,
                side="BUY",
                size=size,
                entry_price=fill_price,
            )
            self.positions[ticker] = pos

            log.info(f"BUY {ticker}: {size:.2f} shares @ ${fill_price:.2f} (notional ${size * fill_price:.2f})")
        except Exception as e:
            log.error(f"[BERYL] ERROR: Failed to execute BUY order for {ticker}: {e}")
            # Remove position if it was partially added
            self.positions.pop(ticker, None)
            return False

        try:
            notify_trade("BUY", size, fill_price, ticker=ticker, project="BERYL")
        except Exception as e:
            log.warning(f"[BERYL] WARNING: BUY notification failed for {ticker}: {e}")

        return True

    def process_signals(self, signals: list[dict]) -> bool:
        """
        Process scanned signals with multi-ticker rotation logic.

        Rules:
        1. If holding positions and their tickers are BEAR -> sell those positions
        2. If available slots + BUY signals exist -> enter best BUYs up to MAX_POSITIONS
        """
        traded = False

        # ── Check if any held positions' tickers went BEAR ──────
        tickers_to_close = []
        for ticker, pos in list(self.positions.items()):
            pos_signal = next((s for s in signals if s["ticker"] == ticker), None)

            if pos_signal and pos_signal["regime"] == "BEAR":
                log.info(f"{ticker} regime flipped to BEAR -> closing position")
                tickers_to_close.append((ticker, pos_signal["current_price"], pos_signal["confirmations"]))
            elif pos_signal and pos_signal["signal"] == "SELL":
                log.info(f"{ticker} SELL signal -> closing position")
                tickers_to_close.append((ticker, pos_signal["current_price"], pos_signal["confirmations"]))

        for ticker, price, confirms in tickers_to_close:
            try:
                self._close_position(ticker, price, confirms)
                traded = True
            except Exception as e:
                log.error(f"[BERYL] ERROR: Failed to close position {ticker}: {e}")

        # ── Log cooldown status for any tickers in cooldown ────
        now = datetime.now(tz=timezone.utc)
        active_cooldowns = {t: cd for t, cd in self.cooldowns.items() if now < cd}
        if active_cooldowns:
            for t, cd in active_cooldowns.items():
                remaining = (cd - now).total_seconds() / 3600
                log.info(f"COOLDOWN {t}: {remaining:.1f}h remaining")

        # ── Enter new positions if slots available ────────────
        available_slots = MAX_POSITIONS - len(self.positions)
        if available_slots > 0:
            best_buys = self._pick_best_buys(signals, available_slots)
            for buy_signal in best_buys:
                try:
                    notify_signal(buy_signal["signal"], buy_signal["regime"], buy_signal["confirmations"], buy_signal["current_price"])
                    if self._open_position(buy_signal):
                        traded = True
                except Exception as e:
                    log.error(f"[BERYL] ERROR: Failed to enter position {buy_signal.get('ticker', '?')}: {e}")

        return traded

    def _log_trade(self, pos: BerylPosition, exit_price: float, pnl: float, pnl_pct: float, confirmations: int):
        """Log completed trade to SQLite."""
        now = datetime.now(tz=timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO trades
                (timestamp, ticker, entry_time, exit_time, entry_price, exit_price, side, size, pnl, pnl_pct, signal_strength)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now,
                    pos.ticker,
                    pos.entry_time,
                    now,
                    pos.entry_price,
                    exit_price,
                    pos.side,
                    pos.size,
                    round(pnl, 2),
                    round(pnl_pct, 2),
                    confirmations,
                ),
            )
            conn.commit()

    def check_kill_switch(self) -> bool:
        """Check BERYL kill-switch conditions (across all tickers).

        Rules:
          1. Total P&L loss > 2% of equity -> exit
          2. 0 wins in last 10 trades -> stop
          3. Rolling 5-trade Sharpe < 0.3 -> exit (parity with AGATE/CITRINE)
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT pnl FROM trades ORDER BY timestamp DESC LIMIT 10",
            ).fetchall()

        if not rows:
            return False

        pnls = [r[0] for r in rows]

        # Rule 1: Total P&L loss > 2% of equity
        total_pnl = sum(pnls)
        if total_pnl < -(INITIAL_EQUITY * 0.02):
            reason = f"BERYL: Total loss ${total_pnl:.2f} exceeds 2% of equity"
            log.error(f"KILL-SWITCH: {reason}")
            notify_kill_switch(reason)
            return True

        # Rule 2: 0 wins in last 10 trades
        if len(pnls) >= 10 and all(p <= 0 for p in pnls[:10]):
            reason = f"BERYL: 0/10 winning trades"
            log.error(f"KILL-SWITCH: {reason}")
            notify_kill_switch(reason)
            return True

        # Rule 3: Rolling 5-trade Sharpe < 0.3
        if len(pnls) >= 5:
            recent = np.array(pnls[:5])
            mean_pnl = recent.mean()
            std_pnl = recent.std(ddof=1)
            rolling_sharpe = (mean_pnl / std_pnl) if std_pnl > 0 else 0.0
            if rolling_sharpe < 0.3:
                reason = (f"BERYL: Rolling 5-trade Sharpe {rolling_sharpe:.3f} < 0.3 "
                          f"(trades: {recent.tolist()})")
                log.error(f"KILL-SWITCH: {reason}")
                notify_kill_switch(reason)
                return True

        return False

    # ── Intra-day risk monitoring ───────────────────────────────────────────
    def _sleep_with_risk_checks(self, total_seconds: int) -> None:
        """Sleep for total_seconds but wake every 4 hours to check intraday risk."""
        INTRADAY_INTERVAL = 4 * 3600  # 4 hours
        remaining = total_seconds

        while remaining > 0:
            sleep_time = min(remaining, INTRADAY_INTERVAL)
            time.sleep(sleep_time)
            remaining -= sleep_time

            if remaining > 0:
                try:
                    self._check_intraday_risk()
                except Exception as e:
                    log.error(f"[BERYL] ERROR: Intra-day risk check failed: {e}")

    def _check_intraday_risk(self) -> None:
        """
        Intra-day risk check — called every 4 hours during sleep.
        Fetches latest price for all held positions and checks unrealized P&L.

        Warning: position loss > 5%
        Kill-switch: position loss > 10% (emergency exit for that position)
        Kill-switch: total unrealized loss > 10% of equity (emergency exit all)
        """
        if not self.positions:
            return

        log.info("  [Intraday] Checking unrealized P&L for %d positions...", len(self.positions))

        total_unrealized = 0.0
        total_notional = 0.0
        positions_to_emergency_sell = []

        for ticker, pos in list(self.positions.items()):
            try:
                from src.data_fetcher import fetch_latest_price
                current_price = fetch_latest_price(ticker)
                if current_price is None or current_price <= 0:
                    current_price = pos.entry_price

                # Compute unrealized P&L (BERYL is always LONG)
                unrealized = pos.size * (current_price - pos.entry_price)
                unrealized_pct = unrealized / pos.notional if pos.notional > 0 else 0

                total_unrealized += unrealized
                total_notional += pos.notional

                # Warn if position loss > 5%
                if unrealized_pct < -0.05:
                    msg = (f"  [Intraday] WARNING: {ticker} unrealized "
                           f"{unrealized_pct:.1%} (${unrealized:.2f})")
                    log.warning(msg)
                    _macos_notify("BERYL Risk Warning", msg)

                # Kill-switch: position loss > 10% — emergency exit that position
                if unrealized_pct < -0.10:
                    reason = (f"BERYL INTRADAY: {ticker} unrealized "
                              f"{unrealized_pct:.1%} (${unrealized:.2f}) exceeds 10% loss")
                    log.error(f"KILL-SWITCH: {reason}")
                    notify_kill_switch(reason)
                    positions_to_emergency_sell.append((ticker, current_price, reason))
                else:
                    log.info(f"  [Intraday] {ticker} within limits "
                             f"(unrealized: {unrealized_pct:+.1%}, ${unrealized:+.2f})")
            except Exception as e:
                log.error(f"[BERYL] ERROR: Failed to check intraday risk for {ticker}: {e}")

        # Emergency sell individual positions that hit -10%
        for ticker, price, reason in positions_to_emergency_sell:
            try:
                self._emergency_sell_single(ticker, price, reason)
            except Exception as e:
                log.error(f"[BERYL] ERROR: Failed emergency sell for {ticker}: {e}")

        # Check total portfolio unrealized loss > 10% of equity
        if total_notional > 0 and total_unrealized < -(INITIAL_EQUITY * 0.10):
            reason = (f"BERYL INTRADAY: Total unrealized ${total_unrealized:.2f} "
                      f"exceeds 10% of equity (${INITIAL_EQUITY * 0.10:.2f})")
            log.error(f"KILL-SWITCH: {reason}")
            notify_kill_switch(reason)
            self._emergency_sell_all(reason)

    def _emergency_sell_single(self, ticker: str, current_price: float, reason: str) -> None:
        """Emergency exit a single position when intra-day kill-switch triggers."""
        pos = self.positions.get(ticker)
        if pos is None:
            return

        log.error("EMERGENCY SELL: %s @ $%.2f (reason: %s)",
                  pos.ticker, current_price, reason)

        slippage = current_price * SLIPPAGE_BPS / 10_000
        exit_price = current_price - slippage  # selling

        pnl = pos.size * (exit_price - pos.entry_price)
        fee = pos.size * exit_price * TAKER_FEE
        pnl -= fee
        pnl_pct = (pnl / pos.notional) * 100 if pos.notional > 0 else 0.0

        # Log trade to DB
        now = datetime.now(tz=timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO trades
                (timestamp, ticker, entry_time, exit_time, entry_price, exit_price, side, size, pnl, pnl_pct, signal_strength)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now,
                    pos.ticker,
                    pos.entry_time,
                    now,
                    pos.entry_price,
                    exit_price,
                    pos.side,
                    pos.size,
                    round(pnl, 2),
                    round(pnl_pct, 2),
                    0,
                ),
            )
            conn.commit()

        notify_trade("SELL", pos.size, exit_price, pnl=pnl, ticker=ticker, project="BERYL")

        self.cooldowns[ticker] = datetime.now(tz=timezone.utc) + timedelta(hours=BERYL_COOLDOWN_HOURS)
        del self.positions[ticker]

    def _emergency_sell_all(self, reason: str) -> None:
        """Emergency exit all positions when portfolio-level kill-switch triggers."""
        log.error("EMERGENCY SELL ALL: %d positions (reason: %s)", len(self.positions), reason)

        for ticker in list(self.positions.keys()):
            pos = self.positions[ticker]
            # Use entry price as fallback (we may not have current price)
            from src.data_fetcher import fetch_latest_price
            current_price = fetch_latest_price(ticker)
            if current_price is None or current_price <= 0:
                current_price = pos.entry_price

            self._emergency_sell_single(ticker, current_price, reason)

    def _write_status(self, signals: list[dict]) -> None:
        """Write latest signal status to JSON for dashboard consumption."""
        try:
            # Build positions info for all held positions
            positions_info = {}
            for ticker, pos in self.positions.items():
                positions_info[ticker] = {
                    "ticker": pos.ticker,
                    "side": pos.side,
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                    "notional": pos.notional,
                }

            # For backward compat: single "position" field uses first held position
            position_info = None
            if self.positions:
                first_pos = next(iter(self.positions.values()))
                position_info = {
                    "ticker": first_pos.ticker,
                    "side": first_pos.side,
                    "size": first_pos.size,
                    "entry_price": first_pos.entry_price,
                }

            # Use first held ticker's signal, or first ticker's signal for display
            held_tickers = list(self.positions.keys())
            display_ticker = held_tickers[0] if held_tickers else self.tickers[0]
            display_signal = next((s for s in signals if s["ticker"] == display_ticker), signals[0] if signals else {})

            # Summary of all scanned tickers (with per-ticker config info)
            scan_summary = []
            for s in signals:
                scan_summary.append({
                    "ticker": s["ticker"],
                    "regime": s["regime"],
                    "confidence": round(s["confidence"], 3),
                    "signal": s["signal"],
                    "confirmations": s["confirmations"],
                    "min_confirmations": s.get("min_confirmations", 5),
                    "config_used": s.get("config_used", "unknown"),
                })

            # Regime counts
            n_bull = sum(1 for s in signals if s["regime"] == "BULL")
            n_bear = sum(1 for s in signals if s["regime"] == "BEAR")
            n_chop = sum(1 for s in signals if s["regime"] == "CHOP")

            status = {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "ticker": display_ticker,
                "regime": display_signal.get("regime", "UNKNOWN"),
                "confidence": display_signal.get("confidence", 0.0),
                "confirmations": display_signal.get("confirmations", 0),
                "signal": display_signal.get("signal", "HOLD"),
                "current_price": display_signal.get("current_price", 0.0),
                "position": position_info,
                "positions": positions_info,
                "positions_count": len(self.positions),
                "max_positions": MAX_POSITIONS,
                "config": {"mode": "per-ticker", "default_fallback": BERYL_DEFAULT_CONFIG,
                           "total_tickers": len(self.tickers),
                           "per_ticker_configs_loaded": len(self._per_ticker_params),
                           "model": "ensemble",
                           "max_positions": MAX_POSITIONS,
                           "notional_per_pos": MAX_NOTIONAL_PER_POS},
                "tickers_scanned": len(signals),
                "tickers_total": len(self.tickers),
                "tickers_converged": len(signals),
                "tickers_bull": n_bull,
                "tickers_bear": n_bear,
                "tickers_chop": n_chop,
                "scan_summary": scan_summary,
                "error": None,
            }

            status_path = ROOT / "beryl_status.json"
            with open(status_path, "w") as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            log.warning(f"Could not write status file: {e}")

    def run_forever(self):
        """Main trading loop — runs once per day after market close."""
        log.info(f"\n{'='*60}")
        log.info(f"  BERYL LIVE TRADING — {len(self.tickers)}-TICKER ROTATION")
        log.info(f"  Mode: {'TEST (simulated)' if self.test_mode else 'LIVE'}")
        log.info(f"  Model: EnsembleHMM (3 models, majority voting) + fallback")
        log.info(f"  Max positions: {MAX_POSITIONS} (${MAX_NOTIONAL_PER_POS:,} each)")
        log.info(f"  Per-ticker configs: {len(self._per_ticker_params)} loaded")
        log.info(f"  Default fallback: {BERYL_DEFAULT_CONFIG}")
        log.info(f"  Lookback: {LOOKBACK_DAYS} days | Cooldown: {BERYL_COOLDOWN_HOURS}h")
        log.info(f"  Max notional (total): ${MAX_NOTIONAL:,}")
        log.info(f"  Signal check: Once per day (daily bars)")
        log.info(f"  Est. scan time: ~{len(self.tickers) * 12 // 60} min ({len(self.tickers)} tickers x 12s)")
        if _GLOBAL_MIN_CONFIRMATIONS is not None:
            log.info(f"  Confirmation override: {_GLOBAL_MIN_CONFIRMATIONS} (from --min-confirmations)")
        log.info(f"{'='*60}\n")

        _macos_notify("BERYL Started", f"{len(self.tickers)}-ticker rotation test mode")

        while True:
            try:
                # Check kill-switch (with grace period after restart)
                if self._cycles_since_restart >= self._KILL_SWITCH_GRACE_CYCLES:
                    if self.check_kill_switch():
                        log.error("KILL-SWITCH TRIGGERED — stopping BERYL")
                        break
                else:
                    log.info(f"Kill-switch grace period: cycle {self._cycles_since_restart + 1}"
                             f"/{self._KILL_SWITCH_GRACE_CYCLES} (skipping check)")

                # Scan all tickers
                log.info(f"Scanning {len(self.tickers)} ticker(s)...")
                signals = self._scan_all_tickers()

                if signals and isinstance(signals, list) and len(signals) > 0:
                    self._write_status(signals)
                    self._log_scan_journal(signals)
                    self.process_signals(signals)
                else:
                    log.warning("No signals generated (signals=%s) — retrying next cycle",
                                type(signals).__name__ if signals is not None else "None")

                self._cycles_since_restart += 1

                # Sleep until next check (with 4h intra-day risk monitoring)
                log.info(f"Next signal check in {SIGNAL_INTERVAL_S // 3600}h. Sleeping (risk checks every 4h)...")
                self._sleep_with_risk_checks(SIGNAL_INTERVAL_S)

            except KeyboardInterrupt:
                log.info("Keyboard interrupt — shutting down BERYL")
                break
            except Exception as e:
                log.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait 1 min then retry


def main():
    global _GLOBAL_MIN_CONFIRMATIONS

    parser = argparse.ArgumentParser(description="BERYL: NDX100 live trading")
    parser.add_argument("--test", action="store_true", default=True,
                        help="Run in test mode (simulated fills)")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Single equity ticker (legacy, default: NVDA)")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated tickers for multi-ticker rotation (e.g., NVDA,TSLA,GOOGL,MSFT,AAPL)")
    parser.add_argument("--min-confirmations", type=int, default=None,
                        help="Override min confirmations threshold for all tickers")
    args = parser.parse_args()

    # Handle graceful shutdown
    def _sigterm_handler(signum, frame):
        log.info("SIGTERM received — shutting down BERYL")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Apply global confirmation override if set
    if args.min_confirmations is not None:
        _GLOBAL_MIN_CONFIRMATIONS = args.min_confirmations
        log.info(f"Global min-confirmations override: {_GLOBAL_MIN_CONFIRMATIONS}")

    # Determine ticker list
    # Default (no args): auto-populate from per-ticker JSON (82 tickers)
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    elif args.ticker:
        tickers = [args.ticker]
    else:
        tickers = None  # Let engine auto-populate from per-ticker configs

    engine = BerylLiveEngine(tickers=tickers, test_mode=args.test)
    engine.run_forever()


if __name__ == "__main__":
    main()
