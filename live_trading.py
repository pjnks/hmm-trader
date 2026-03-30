"""
live_trading.py
───────────────
Live trading orchestrator for AGATE multi-ticker crypto rotation.

Scans up to 16 crypto tickers every 4h, picks the strongest BULL signal,
and holds one position at a time (rotation model, same as BERYL equities).

CRITICAL: Runs with 0.25× effective leverage:
  - Max notional: $2,500 (0.25 × $10k account)
  - Max risk per trade: $50 (2% loss limit)
  - Auto-closes positions if kill-switch triggered

Workflow:
1. Every 4h: scan all tickers (ensemble HMM + 8 indicators per ticker)
2. If holding + held ticker turns BEAR → exit
3. If flat + best BUY signal exists → enter that ticker
4. Track P&L, Sharpe, kill-switch conditions
5. Daily report at 9am UTC

Usage
─────
  python live_trading.py --test                                    # all 16 tickers
  python live_trading.py --test --tickers X:BTCUSD,X:ETHUSD,X:SOLUSD  # subset
  python live_trading.py --live                                    # REAL MONEY
"""

from __future__ import annotations

import argparse
import json
import logging
import signal as signal_mod
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import sqlite3

import config
from src.live_broker import LiveBroker, Position
from src.signal_generator import SignalGenerator
from src.live_monitor import LiveMonitor, TradeRecord
from src.notifier import notify_kill_switch, notify_trade, notify_signal, notify_daily
from src.momentum_scanner import scan_momentum
from src.diamond_bridge import get_diamond_boost

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("live_trading")

# ── Configuration ──────────────────────────────────────────────────────────
EFFECTIVE_LEVERAGE = 0.25  # Max notional = $2,500 on $10k account
MAX_NOTIONAL = config.INITIAL_CAPITAL * EFFECTIVE_LEVERAGE
SIGNAL_CHECK_INTERVAL_MINUTES = 240  # Check every 4 hours (same as timeframe)
RATE_LIMIT_SECONDS = 12  # Polygon API rate limit between ticker fetches

# Kill-switch grace period: skip checks for first N cycles after restart
# (prevents restart loops when old DB trades have poor Sharpe)
_KILL_SWITCH_GRACE_CYCLES = 3

# Adaptive confirmations: lower cf by 1 when regime confidence exceeds this
ADAPTIVE_CF_CONFIDENCE = 0.90
ADAPTIVE_CF_REDUCTION = 1  # lower cf by this much at high confidence

# Per-ticker config file (hot-loaded, produced by optimize_agate_multi.py)
_PER_TICKER_CONFIG_PATH = Path(__file__).parent / "agate_per_ticker_configs.json"


def _load_per_ticker_configs() -> dict:
    """Load per-ticker optimized HMM configs if available."""
    if _PER_TICKER_CONFIG_PATH.exists():
        try:
            with open(_PER_TICKER_CONFIG_PATH) as f:
                configs = json.load(f)
            log.info(f"Loaded per-ticker configs for {len(configs)} tickers")
            return configs
        except Exception as e:
            log.warning(f"Failed to load per-ticker configs: {e}")
    return {}


class LiveTradingEngine:
    """
    Multi-ticker crypto rotation engine.

    Scans all configured tickers, picks the single best BULL signal,
    holds one position at a time. Same pattern as BERYL equities.

    Supports per-ticker optimized configs (from optimize_agate_multi.py)
    and adaptive confirmations (lower cf threshold at high regime confidence).
    """

    def __init__(self, tickers: list[str], test_mode: bool = True):
        self.tickers = tickers
        self.test_mode = test_mode

        # Broker starts with a default product; we switch it per trade
        self.broker = LiveBroker(
            product_id=config.CRYPTO_PRODUCT_MAP.get(tickers[0], "SOL-USD"),
            test_mode=test_mode,
        )
        self.monitor = LiveMonitor()

        # Unified vs per-ticker config mode (Sprint 9)
        self.use_unified = getattr(config, "AGATE_USE_UNIFIED_CONFIG", False)
        if self.use_unified:
            self.unified_cfg = config.AGATE_UNIFIED_CONFIG
            self.per_ticker_configs: dict = {}
        else:
            self.unified_cfg = None
            self.per_ticker_configs: dict = _load_per_ticker_configs()

        # Multi-ticker state
        self.held_ticker: Optional[str] = None  # Polygon ticker of current position
        self.cooldown_until: dict[str, datetime] = {}  # per-ticker cooldown

        self.last_signal_check: Optional[datetime] = None
        self.last_email_sent: Optional[datetime] = None
        self._cycles_since_restart = 0

        # Signal journal DB
        self._journal_db = Path(__file__).parent / "agate_journal.db"
        self._init_journal_db()

        mode = "TEST" if test_mode else "REAL"
        log.info(f"Live Trading Engine initialized ({mode} mode)")
        log.info(f"   Tickers: {len(tickers)} crypto ({', '.join(t.replace('X:','') for t in tickers[:5])}{'...' if len(tickers) > 5 else ''})")
        log.info(f"   Timeframe: {config.TIMEFRAME}")
        log.info(f"   Max notional: ${MAX_NOTIONAL:,.0f} (0.25x leverage)")
        if self.use_unified:
            ucfg = self.unified_cfg
            log.info(f"   UNIFIED CONFIG: {ucfg['feature_set']}/{ucfg['n_states']}st/{ucfg['confirmations']}cf/{ucfg['cov_type']}/{ucfg['timeframe']}")
        else:
            log.info(f"   Feature set: {config.FEATURE_SET} (per-ticker overrides: {len(self.per_ticker_configs)})")
        log.info(f"   Min confirmations: {config.MIN_CONFIRMATIONS} (adaptive: cf-{ADAPTIVE_CF_REDUCTION} when conf>{ADAPTIVE_CF_CONFIDENCE})")
        log.info(f"   Cooldown: {config.COOLDOWN_HOURS}h per-ticker")

    # ─────────────────────────────────────────────────────────────────────
    # Multi-ticker scanning (modeled on BERYL's _scan_all_tickers)
    # ─────────────────────────────────────────────────────────────────────

    def _init_journal_db(self):
        """Initialize signal journal database for audit trail."""
        with sqlite3.connect(self._journal_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    regime TEXT,
                    confidence REAL,
                    confirmations INTEGER,
                    signal TEXT,
                    current_price REAL,
                    momentum_score REAL,
                    diamond_boost REAL,
                    final_score REAL,
                    decision TEXT,
                    config_used TEXT,
                    rsi REAL,
                    adx REAL,
                    adaptive_cf TEXT
                )
            """)
            conn.commit()

    def _log_journal(self, signals: list[dict], decision: str):
        """Log full scan state to journal DB for audit trail."""
        try:
            now = datetime.now(tz=timezone.utc).isoformat()
            with sqlite3.connect(self._journal_db) as conn:
                for s in signals:
                    tcfg = s.get("ticker_config", {})
                    conn.execute(
                        """INSERT INTO signal_journal
                        (timestamp, ticker, regime, confidence, confirmations, signal,
                         current_price, momentum_score, diamond_boost, final_score,
                         decision, config_used, rsi, adx, adaptive_cf)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            now,
                            s.get("ticker", ""),
                            s.get("regime", ""),
                            s.get("regime_confidence", 0),
                            s.get("confirmations", 0),
                            s.get("signal", ""),
                            s.get("current_price", 0),
                            s.get("momentum_score", 0),
                            s.get("diamond_boost", 1.0),
                            s.get("final_score", 0),
                            decision,
                            json.dumps(tcfg),
                            s.get("rsi", 0),
                            s.get("adx", 0),
                            s.get("adaptive_cf", ""),
                        ),
                    )
                conn.commit()
        except Exception as e:
            log.warning(f"Journal logging failed: {e}")

    def _check_near_misses(self, signals: list[dict]):
        """Alert on tickers that almost triggered a BUY (within 1 confirmation)."""
        from src.notifier import _pushover_notify
        near_misses = []
        for s in signals:
            if s.get("regime") != "BULL":
                continue
            tcfg = s.get("ticker_config", {})
            required_cf = tcfg.get("confirmations", config.MIN_CONFIRMATIONS)
            conf = s.get("regime_confidence", 0)
            if conf >= ADAPTIVE_CF_CONFIDENCE:
                required_cf = max(required_cf - ADAPTIVE_CF_REDUCTION, 4)
            gap = required_cf - s.get("confirmations", 0)
            if gap == 1:
                near_misses.append(s["ticker"].replace("X:", ""))

        if near_misses:
            msg = f"Near-miss BUY: {', '.join(near_misses)} (1 indicator away)"
            log.info(f"  {msg}")
            try:
                _pushover_notify("AGATE Near-Miss", msg, priority=-1)
            except Exception:
                pass

    def _scan_all_tickers(self) -> list[dict]:
        """
        Scan all configured tickers with ensemble HMM.

        Uses per-ticker optimized configs when available (from agate_per_ticker_configs.json).
        Falls back to global config defaults for tickers without optimization results.

        Returns list of signal dicts, one per ticker:
        {ticker, signal, regime, confidence, confirmations, current_price, error,
         ticker_config: {feature_set, confirmations, timeframe, ...}}
        """
        # Hot-reload per-ticker configs each scan (allows optimizer to update without restart)
        if not self.use_unified:
            self.per_ticker_configs = _load_per_ticker_configs()

        signals = []
        for i, ticker in enumerate(self.tickers):
            if i > 0:
                time.sleep(RATE_LIMIT_SECONDS)

            # Sprint 9: unified config mode — same settings for all tickers
            if self.use_unified:
                ucfg = self.unified_cfg
                ticker_timeframe = ucfg["timeframe"]
                ticker_feature_set = ucfg["feature_set"]
                ticker_confirmations = ucfg["confirmations"]
                ticker_cov_type = ucfg["cov_type"]
                ticker_n_states = ucfg["n_states"]
            else:
                # Legacy per-ticker config (or global defaults)
                tcfg = self.per_ticker_configs.get(ticker, {})
                ticker_timeframe = tcfg.get("timeframe", config.TIMEFRAME)
                ticker_feature_set = tcfg.get("feature_set", config.FEATURE_SET)
                ticker_confirmations = tcfg.get("confirmations", config.MIN_CONFIRMATIONS)
                ticker_cov_type = tcfg.get("cov_type", config.COV_TYPE)
                ticker_n_states = tcfg.get("n_states", config.N_STATES)

            try:
                # Temporarily patch config for this ticker's scan
                saved = {
                    "FEATURE_SET": config.FEATURE_SET,
                    "MIN_CONFIRMATIONS": config.MIN_CONFIRMATIONS,
                    "COV_TYPE": config.COV_TYPE,
                    "N_STATES": config.N_STATES,
                }
                config.FEATURE_SET = ticker_feature_set
                config.MIN_CONFIRMATIONS = ticker_confirmations
                config.COV_TYPE = ticker_cov_type
                config.N_STATES = ticker_n_states

                try:
                    gen = SignalGenerator(
                        ticker=ticker,
                        timeframe=ticker_timeframe,
                        lookback_days=90,
                        use_ensemble=True,
                    )
                    sig = gen.generate_signal()
                finally:
                    # Restore global config
                    for k, v in saved.items():
                        setattr(config, k, v)

                sig["ticker"] = ticker
                # Attach the config used for this ticker (for logging/debugging)
                sig["ticker_config"] = {
                    "feature_set": ticker_feature_set,
                    "confirmations": ticker_confirmations,
                    "timeframe": ticker_timeframe,
                    "cov_type": ticker_cov_type,
                    "n_states": ticker_n_states,
                    "unified": self.use_unified,
                    "optimized": (not self.use_unified) and ticker in self.per_ticker_configs,
                }
                signals.append(sig)

                regime = sig.get("regime", "?")
                conf = sig.get("regime_confidence", 0.0)
                confirms = sig.get("confirmations", 0)
                signal_type = sig.get("signal", "HOLD")
                cfg_marker = "U" if self.use_unified else ("*" if ticker in self.per_ticker_configs else " ")
                log.info(f" {cfg_marker}{ticker.replace('X:',''): <10} {regime: <5} conf={conf:.2f} "
                         f"{confirms}/8 → {signal_type}  [{ticker_feature_set}/{ticker_timeframe}]")

            except Exception as e:
                log.warning(f"  {ticker.replace('X:',''): <10} FAILED: {e}")
                signals.append({
                    "ticker": ticker,
                    "signal": "HOLD",
                    "regime": "UNKNOWN",
                    "regime_confidence": 0.0,
                    "confirmations": 0,
                    "current_price": 0.0,
                    "error": str(e),
                    "ticker_config": {},
                })

        return signals

    def _pick_best_buy(self, signals: list[dict]) -> Optional[dict]:
        """
        From all scanned signals, pick the best BUY candidate.

        Filters: BUY signal + not in cooldown + confidence >= gate.
        Adaptive confirmations: lowers cf threshold by 1 when confidence > 0.90.
        Scores: confidence × (confirmations / 8).
        Returns best scorer or None.
        """
        now = datetime.now(tz=timezone.utc)

        buy_candidates = []
        for s in signals:
            if s.get("error"):
                continue

            ticker = s["ticker"]
            regime = s.get("regime", "UNKNOWN")
            confidence = s.get("regime_confidence", 0.0)
            confirmations = s.get("confirmations", 0)

            # Must be BULL regime (BUY signal comes from build_signal_series,
            # but we also accept BULL+HOLD that pass adaptive cf)
            if regime != "BULL":
                continue

            # Per-ticker cooldown check
            cooldown_end = self.cooldown_until.get(ticker)
            if cooldown_end and now < cooldown_end:
                remaining_h = (cooldown_end - now).total_seconds() / 3600
                log.info(f"  {ticker.replace('X:','')} BULL skipped — cooldown {remaining_h:.1f}h remaining")
                continue

            # Confidence gate
            if confidence < config.REGIME_CONFIDENCE_MIN:
                continue

            # Confirmations gate (adaptive: lower by 1 when high confidence)
            tcfg = s.get("ticker_config", {})
            required_cf = tcfg.get("confirmations", config.MIN_CONFIRMATIONS)

            if confidence >= ADAPTIVE_CF_CONFIDENCE:
                required_cf = max(required_cf - ADAPTIVE_CF_REDUCTION, 4)  # floor at 4

            if confirmations < required_cf:
                log.info(f"  {ticker.replace('X:','')} BULL skipped — {confirmations}/{required_cf} cf "
                         f"(adaptive={'yes' if confidence >= ADAPTIVE_CF_CONFIDENCE else 'no'})")
                continue

            buy_candidates.append(s)

        if not buy_candidates:
            return None

        # ── Enrich candidates with momentum + DIAMOND ────────────────
        try:
            for s in buy_candidates:
                ticker = s["ticker"]

                # Momentum scan (1h bars, 72h lookback)
                try:
                    mom = scan_momentum(ticker)
                    s["momentum_score"] = mom.get("momentum_score", 0.0)
                    s["momentum_components"] = mom.get("components", {})
                except Exception as e:
                    log.warning(f"[AGATE] Failed to scan momentum for {ticker}: {e}")
                    s["momentum_score"] = 0.0

                # DIAMOND anomaly boost
                try:
                    s["diamond_boost"] = get_diamond_boost(ticker)
                except Exception as e:
                    log.warning(f"[AGATE] Failed to get DIAMOND boost for {ticker}: {e}")
                    s["diamond_boost"] = 1.0
        except Exception as e:
            log.error(f"[AGATE] Failed to enrich buy candidates: {e}")
            for s in buy_candidates:
                s.setdefault("momentum_score", 0.0)
                s.setdefault("diamond_boost", 1.0)

        # Score: confidence × (confirmations/8) × momentum_boost × diamond_boost
        def score(s):
            base = s.get("regime_confidence", 0.0) * (s.get("confirmations", 0) / 8)
            # Momentum boost: 1.0 + clip(momentum_score, 0, 0.5) → up to 50% boost
            mom_boost = 1.0 + max(0, min(s.get("momentum_score", 0), 0.5))
            diamond = s.get("diamond_boost", 1.0)
            final = base * mom_boost * diamond
            s["final_score"] = round(final, 4)
            return final

        buy_candidates.sort(key=score, reverse=True)
        best = buy_candidates[0]
        best_cf = best.get("ticker_config", {}).get("confirmations", config.MIN_CONFIRMATIONS)
        adaptive = best.get("regime_confidence", 0) >= ADAPTIVE_CF_CONFIDENCE
        log.info(f"  Best BUY: {best['ticker'].replace('X:','')} score={score(best):.3f} "
                 f"(conf={best.get('regime_confidence', 0):.2f}, {best.get('confirmations', 0)}/8, "
                 f"mom={best.get('momentum_score', 0):.2f}, diamond={best.get('diamond_boost', 1.0):.1f}x, "
                 f"adaptive={'yes' if adaptive else 'no'})")
        return best

    # ─────────────────────────────────────────────────────────────────────
    # Trade execution
    # ─────────────────────────────────────────────────────────────────────

    def process_signals(self, signals: list[dict]) -> bool:
        """
        Process multi-ticker scan results. Exit BEAR positions, enter best BUY.

        Returns True if any trade was executed.
        """
        executed = False

        # ── Check if held position needs exit ─────────────────────────
        try:
            if self.held_ticker and self.broker.position:
                held_signal = next(
                    (s for s in signals if s["ticker"] == self.held_ticker),
                    None,
                )
                if held_signal:
                    regime = held_signal.get("regime", "UNKNOWN")
                    signal_type = held_signal.get("signal", "HOLD")

                    if signal_type == "SELL" or regime == "BEAR":
                        current_price = held_signal.get("current_price", 0.0)
                        log.info(f"CLOSING {self.held_ticker.replace('X:','')} — {regime} regime")
                        executed = self._exit_position(current_price, held_signal)
        except Exception as e:
            log.error(f"[AGATE] Failed to process exit signals: {e}", exc_info=True)

        # ── If flat, look for best BUY ────────────────────────────────
        try:
            if self.broker.position is None:
                best_buy = self._pick_best_buy(signals)
                if best_buy:
                    executed = self._enter_position(best_buy)
        except Exception as e:
            log.error(f"[AGATE] Failed to process entry signals: {e}", exc_info=True)

        return executed

    def _enter_position(self, signal: dict) -> bool:
        """Enter a position on the given ticker."""
        ticker = signal["ticker"]
        current_price = signal.get("current_price", 0.0)
        confirmations = signal.get("confirmations", 0)

        if current_price <= 0:
            log.warning(f"Invalid price for {ticker}: {current_price}")
            return False

        # Switch broker to this ticker's Coinbase product
        product_id = config.CRYPTO_PRODUCT_MAP.get(ticker, "SOL-USD")
        self.broker.product_id = product_id

        # Calculate position size
        size = MAX_NOTIONAL / current_price

        log.info(f"OPENING BUY: {size:.6f} {ticker.replace('X:','')} @ ${current_price:.2f} "
                 f"(notional ${MAX_NOTIONAL:.0f})")

        # Execute order + update state
        try:
            order = self.broker.place_market_order(side="BUY", size=size, current_price=current_price)

            # Open position with ticker tracking
            self.broker.position = Position(
                entry_price=order["execution_price"],
                size=size,
                side="BUY",
                entry_time=datetime.now(tz=timezone.utc).isoformat(),
                notional=size * order["execution_price"],
                ticker=ticker,
            )
            self.held_ticker = ticker

            log.info(f"Position opened: {ticker.replace('X:','')} @ ${order['execution_price']:.2f}")

            # Log entry for paper P&L tracking + state recovery
            self.monitor.log_entry(
                ticker=ticker,
                entry_price=order["execution_price"],
                size=size,
                side="BUY",
                signal_strength=confirmations,
            )
        except Exception as e:
            log.error(f"[AGATE] Failed to place BUY order for {ticker}: {e}", exc_info=True)
            return False

        try:
            notify_trade("BUY", size, current_price, ticker=ticker.replace("X:", ""))
        except Exception as e:
            log.warning(f"[AGATE] Failed to send BUY notification for {ticker}: {e}")

        return True

    def _exit_position(self, current_price: float, signal: dict) -> bool:
        """Exit current position and apply per-ticker cooldown."""
        if not self.broker.position:
            return False

        ticker = self.held_ticker or self.broker.position.ticker
        confirmations = signal.get("confirmations", 0)

        # Switch broker to correct product for exit
        product_id = config.CRYPTO_PRODUCT_MAP.get(ticker, "SOL-USD")
        self.broker.product_id = product_id

        # Execute exit order
        try:
            order = self.broker.place_market_order(side="SELL", size=self.broker.position.size, current_price=current_price)
            trade_dict = self.broker.close_position(exit_price=order["execution_price"])
        except Exception as e:
            log.error(f"[AGATE] Failed to place SELL order for {ticker}: {e}", exc_info=True)
            return False

        if trade_dict:
            # Log to database with ticker
            try:
                trade = TradeRecord(
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                    entry_time=trade_dict.get("entry_time", ""),
                    exit_time=trade_dict.get("exit_time", ""),
                    entry_price=trade_dict["entry_price"],
                    exit_price=trade_dict["exit_price"],
                    side=trade_dict.get("side", "BUY"),
                    size=trade_dict["size"],
                    pnl=trade_dict["pnl"],
                    pnl_pct=trade_dict["pnl_pct"],
                    signal_strength=confirmations,
                    ticker=ticker,
                )
                self.monitor.log_trade(trade)
                self.monitor.clear_entry(ticker)
            except Exception as e:
                log.error(f"[AGATE] Failed to log trade for {ticker}: {e}")

            log.info(f"Position closed: {ticker.replace('X:','')} PnL ${trade_dict['pnl']:.2f} ({trade_dict['pnl_pct']:.2f}%)")
            try:
                notify_trade("SELL", trade_dict["size"], current_price, pnl=trade_dict["pnl"], ticker=ticker.replace("X:", ""))
            except Exception as e:
                log.warning(f"[AGATE] Failed to send SELL notification for {ticker}: {e}")

        # Per-ticker cooldown (only after successful order execution)
        self.cooldown_until[ticker] = datetime.now(tz=timezone.utc) + timedelta(hours=config.COOLDOWN_HOURS)
        log.info(f"Cooldown set for {ticker.replace('X:','')}: {config.COOLDOWN_HOURS}h")

        self.held_ticker = None
        return True

    # ─────────────────────────────────────────────────────────────────────
    # Status + monitoring
    # ─────────────────────────────────────────────────────────────────────

    def _write_status(self, signals: list[dict]) -> None:
        """Write multi-ticker status JSON for dashboard consumption."""
        try:
            position_info = None
            if self.broker.position is not None:
                pos = self.broker.position
                position_info = {
                    "side": pos.side,
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                    "ticker": pos.ticker,
                }

            # Build scan summary (all tickers)
            scan_summary = []
            for s in signals:
                scan_summary.append({
                    "ticker": s.get("ticker", "?"),
                    "regime": s.get("regime", "UNKNOWN"),
                    "confidence": round(float(s.get("regime_confidence", 0.0)), 3),
                    "signal": s.get("signal", "HOLD"),
                    "confirmations": int(s.get("confirmations", 0)),
                    "current_price": round(float(s.get("current_price", 0.0)), 2),
                })

            # Primary display: held ticker or best signal
            held_signal = None
            if self.held_ticker:
                held_signal = next((s for s in signals if s["ticker"] == self.held_ticker), None)
            if not held_signal and signals:
                held_signal = signals[0]

            status = {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "ticker": self.held_ticker or (signals[0]["ticker"] if signals else config.TICKER),
                "regime": held_signal.get("regime", "UNKNOWN") if held_signal else "UNKNOWN",
                "confidence": float(held_signal.get("regime_confidence", 0.0)) if held_signal else 0.0,
                "confirmations": int(held_signal.get("confirmations", 0)) if held_signal else 0,
                "signal": held_signal.get("signal", "HOLD") if held_signal else "HOLD",
                "current_price": float(held_signal.get("current_price", 0.0)) if held_signal else 0.0,
                "rsi": float(held_signal.get("rsi", 0.0)) if held_signal else 0.0,
                "adx": float(held_signal.get("adx", 0.0)) if held_signal else 0.0,
                "position": position_info,
                "use_ensemble": True,
                "feature_set": config.FEATURE_SET,
                "scan_summary": scan_summary,
                "tickers_scanned": len(signals),
                "tickers_bull": sum(1 for s in signals if s.get("regime") == "BULL"),
                "tickers_bear": sum(1 for s in signals if s.get("regime") == "BEAR"),
                "tickers_chop": sum(1 for s in signals if s.get("regime") == "CHOP"),
                "error": held_signal.get("error") if held_signal else None,
            }

            status_path = Path(__file__).parent / "agate_status.json"
            with open(status_path, "w") as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            log.warning(f"Could not write status file: {e}")

    def check_daily_email_and_alerts(self) -> None:
        """Check if it's 9am UTC and send daily report."""
        now = datetime.now(tz=timezone.utc)

        if (
            self.last_email_sent is None
            or (now - self.last_email_sent).total_seconds() > 86400
        ):
            if now.hour == 9:
                log.info("Sending daily report...")
                try:
                    metrics = self.monitor.get_daily_metrics()
                except Exception as e:
                    log.error(f"[AGATE] Failed to get daily metrics: {e}")
                    metrics = {"sharpe_20": 0.0, "total_pnl": 0.0, "num_trades": 0, "win_rate": 0.0}
                notify_daily(
                    sharpe=metrics["sharpe_20"],
                    pnl=metrics["total_pnl"],
                    trades=metrics["num_trades"],
                    win_rate=metrics["win_rate"],
                )
                self.last_email_sent = now

    def check_kill_switch(self) -> bool:
        """Check kill-switch conditions. Returns True if trading should stop."""
        account_equity = config.INITIAL_CAPITAL
        if self.monitor.check_kill_switch(account_equity=account_equity):
            log.error(f"KILL-SWITCH ACTIVATED: {self.monitor.kill_switch_reason}")
            notify_kill_switch(self.monitor.kill_switch_reason)
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────

    def run_forever(self) -> None:
        """Main trading loop (runs until kill-switch or manual stop)."""
        log.info(f"Starting multi-ticker scan loop ({len(self.tickers)} tickers, {SIGNAL_CHECK_INTERVAL_MINUTES}m interval)...")

        try:
            while True:
                try:
                    self._cycles_since_restart += 1

                    # ── Kill-switch (with grace period) ──────────────
                    if self._cycles_since_restart > _KILL_SWITCH_GRACE_CYCLES:
                        if self.check_kill_switch():
                            log.error("Trading stopped due to kill-switch.")
                            break
                    else:
                        log.info(f"Kill-switch grace: cycle {self._cycles_since_restart}/{_KILL_SWITCH_GRACE_CYCLES}")

                    # ── Scan all tickers ─────────────────────────────
                    log.info(f"Scanning {len(self.tickers)} tickers...")
                    signals = self._scan_all_tickers()

                    if signals:
                        # Write status for dashboard
                        self._write_status(signals)

                        # Summary
                        bull_count = sum(1 for s in signals if s.get("regime") == "BULL")
                        bear_count = sum(1 for s in signals if s.get("regime") == "BEAR")
                        chop_count = sum(1 for s in signals if s.get("regime") == "CHOP")
                        buy_count = sum(1 for s in signals if s.get("signal") == "BUY")
                        log.info(f"Scan complete: {bull_count} BULL, {bear_count} BEAR, {chop_count} CHOP | {buy_count} BUY signals")

                        # Process: exit BEAR, enter best BUY
                        executed = self.process_signals(signals)
                        if executed:
                            log.info("Trade executed")

                        # Near-miss alerts (1 indicator away from BUY)
                        self._check_near_misses(signals)

                        # Signal journal (full audit trail)
                        decision = "TRADE" if executed else "HOLD"
                        self._log_journal(signals, decision)

                    # ── Daily email ──────────────────────────────────
                    self.check_daily_email_and_alerts()

                    # ── Sleep ─────────────────────────────────────────
                    log.info(f"Sleeping {SIGNAL_CHECK_INTERVAL_MINUTES}m until next scan...")
                    time.sleep(SIGNAL_CHECK_INTERVAL_MINUTES * 60)

                except Exception as e:
                    log.error(f"Error in main loop: {e}", exc_info=True)
                    time.sleep(60)

        except KeyboardInterrupt:
            log.info("Trading stopped by user")


def main():
    parser = argparse.ArgumentParser(description="AGATE multi-ticker crypto rotation")
    parser.add_argument("--test", action="store_true", help="Run in test mode (no real money)")
    parser.add_argument("--live", action="store_true", help="RUN IN LIVE MODE (REAL MONEY)")
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated Polygon tickers (default: all 16 from config.AGATE_TICKERS)",
    )

    args = parser.parse_args()

    if not args.test and not args.live:
        parser.print_help()
        print("\nYou must specify --test or --live")
        sys.exit(1)

    if args.live:
        response = input(
            "\nWARNING: You are about to trade with REAL MONEY on Coinbase.\n"
            "Max loss per trade: $50 (2% of account)\n"
            "Type 'YES I APPROVE' to continue: "
        )
        if response.strip() != "YES I APPROVE":
            print("Aborted.")
            sys.exit(1)

    # Parse tickers
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    else:
        tickers = config.AGATE_TICKERS

    # Handle graceful shutdown (systemd sends SIGTERM on restart/stop)
    def _sigterm_handler(signum, frame):
        log.info("SIGTERM received — shutting down AGATE")
        sys.exit(0)
    signal_mod.signal(signal_mod.SIGTERM, _sigterm_handler)

    engine = LiveTradingEngine(tickers=tickers, test_mode=args.test)
    engine.run_forever()


if __name__ == "__main__":
    main()
