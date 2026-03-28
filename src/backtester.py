"""
backtester.py
─────────────
Event-driven bar-by-bar backtester with leverage.

Position model
──────────────
  • One position at a time (LONG or SHORT, no pyramiding).
  • notional = equity × LEVERAGE
  • btc_held  = notional / entry_price
  • LONG exit:  pnl = btc_held × (exit_price − entry_price) − fees
  • SHORT exit: pnl = btc_held × (entry_price − exit_price) − fees
  • fee model: TAKER_FEE applied to notional on both entry and exit.

Exit triggers (in priority order)
──────────────────────────────────
  Legacy mode (use_regime_mapper=False):
    1. Regime flip to BEAR → immediate market exit at bar's Close

  Multi-direction mode (use_regime_mapper=True):
    1. Direction flip away from current position side → exit

After every exit a cooldown is enforced (COOLDOWN_HOURS).

Returned objects
────────────────
  BacktestResult
    .equity_curve   – pd.Series of portfolio equity indexed by timestamp
    .trades         – pd.DataFrame with one row per completed trade
    .metrics        – dict with summary statistics
    .df             – the enriched DataFrame that was tested on
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.strategy import SignalEngine, SignalResult

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_time:    datetime
    entry_price:   float
    entry_regime:  str
    entry_conf:    float
    confirmations: int
    direction:     str                  = "LONG"    # "LONG" or "SHORT"
    exit_time:     Optional[datetime]   = None
    exit_price:    Optional[float]      = None
    exit_reason:   str                  = ""
    pnl:           float                = 0.0
    pnl_pct:       float                = 0.0
    notional:      float                = 0.0
    duration_h:    float                = 0.0


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades:       pd.DataFrame
    metrics:      dict
    df:           pd.DataFrame        # enriched df with all signals/regimes


# ─────────────────────────────────────────────────────────────────────────────

class Backtester:

    def __init__(self, use_regime_mapper: bool = False) -> None:
        self._engine = SignalEngine(use_regime_mapper=use_regime_mapper)
        self._use_regime_mapper = use_regime_mapper

    # ── Main entry point ─────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        Run the full backtest on enriched *df* (must have regime, regime_cat,
        confidence, and all indicator columns attached).

        Returns a BacktestResult.
        """
        # Warm-up: skip the first N rows where indicators may be NaN
        warmup = max(
            config.TREND_MA_PERIOD,
            config.MACD_SLOW + config.MACD_SIGNAL,
            config.ADX_PERIOD * 3,
            config.VOLATILITY_PERIOD * 5,   # vol_median needs 5× window
        )
        df_test = df.iloc[warmup:].copy()

        equity        = config.INITIAL_CAPITAL
        equity_series = {}

        # Position state
        position_side:  str               = "FLAT"    # "FLAT" | "LONG" | "SHORT"
        current_trade:  Optional[Trade]   = None
        last_exit_ts:   Optional[datetime] = None
        last_exit_side: str               = "FLAT"    # direction of last closed position
        btc_held:       float             = 0.0

        completed_trades: list[Trade] = []

        for ts, row in df_test.iterrows():
            ts_aware = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)

            # Mark-to-market equity while in position
            if position_side != "FLAT" and current_trade is not None:
                if position_side == "LONG":
                    unrealised = btc_held * (row["Close"] - current_trade.entry_price)
                else:  # SHORT
                    unrealised = btc_held * (current_trade.entry_price - row["Close"])
                equity_series[ts] = equity + unrealised
            else:
                equity_series[ts] = equity

            regime_cat = row.get("regime_cat", "CHOP")
            confidence = row.get("confidence", 0.0)

            signal: SignalResult = self._engine.evaluate(
                row            = row,
                regime_cat     = regime_cat,
                confidence     = confidence,
                in_position    = (position_side != "FLAT"),
                last_exit_ts   = last_exit_ts,
                position_side  = position_side,
                last_exit_side = last_exit_side,
            )

            # ── EXIT (SELL or COVER) ─────────────────────────────────────
            if signal.is_exit and position_side != "FLAT" and current_trade is not None:
                exit_price = row["Close"]
                if position_side == "LONG":
                    pnl_gross = btc_held * (exit_price - current_trade.entry_price)
                else:  # SHORT
                    pnl_gross = btc_held * (current_trade.entry_price - exit_price)

                fee        = current_trade.notional * config.TAKER_FEE * 2
                pnl_net    = pnl_gross - fee
                duration_h = (ts_aware - current_trade.entry_time).total_seconds() / 3600

                equity += pnl_net

                current_trade.exit_time   = ts_aware
                current_trade.exit_price  = exit_price
                current_trade.exit_reason = signal.reason
                current_trade.pnl         = pnl_net
                current_trade.pnl_pct     = pnl_net / (equity - pnl_net) * 100
                current_trade.duration_h  = duration_h

                completed_trades.append(current_trade)
                log.debug("EXIT %s @ %.2f  pnl=%.2f  reason=%s",
                          position_side, exit_price, pnl_net, signal.reason)

                last_exit_side = position_side   # save direction before resetting
                position_side  = "FLAT"
                current_trade  = None
                btc_held       = 0.0
                last_exit_ts   = ts_aware

                # Update equity series with realised value
                equity_series[ts] = equity

            # ── ENTRY (BUY or SHORT) ─────────────────────────────────────
            elif signal.is_entry and position_side == "FLAT":
                notional    = equity * config.LEVERAGE
                entry_price = row["Close"]
                fee_entry   = notional * config.TAKER_FEE
                equity     -= fee_entry
                btc_held    = notional / entry_price

                trade_direction = "LONG" if signal.is_long_entry else "SHORT"

                current_trade = Trade(
                    entry_time    = ts_aware,
                    entry_price   = entry_price,
                    entry_regime  = row.get("regime", "unknown"),
                    entry_conf    = confidence,
                    confirmations = signal.confirmations,
                    direction     = trade_direction,
                    notional      = notional,
                )
                position_side = trade_direction
                log.debug("ENTRY %s @ %.2f  notional=%.2f  confirms=%d",
                          trade_direction, entry_price, notional, signal.confirmations)

        # ── Force-close any open trade at last bar ───────────────────────
        if position_side != "FLAT" and current_trade is not None:
            last_row   = df_test.iloc[-1]
            last_ts    = df_test.index[-1]
            last_ts_aw = last_ts if last_ts.tzinfo else last_ts.replace(tzinfo=timezone.utc)
            exit_price = last_row["Close"]

            if position_side == "LONG":
                pnl_gross = btc_held * (exit_price - current_trade.entry_price)
            else:  # SHORT
                pnl_gross = btc_held * (current_trade.entry_price - exit_price)

            fee        = current_trade.notional * config.TAKER_FEE
            pnl_net    = pnl_gross - fee
            equity    += pnl_net

            current_trade.exit_time   = last_ts_aw
            current_trade.exit_price  = exit_price
            current_trade.exit_reason = "end_of_backtest"
            current_trade.pnl         = pnl_net
            current_trade.pnl_pct     = pnl_net / max(equity - pnl_net, 1) * 100
            current_trade.duration_h  = (
                last_ts_aw - current_trade.entry_time
            ).total_seconds() / 3600
            completed_trades.append(current_trade)
            equity_series[last_ts] = equity

        # ── Build outputs ────────────────────────────────────────────────
        eq_series = pd.Series(equity_series, name="equity")
        trades_df = self._trades_to_df(completed_trades)
        metrics   = self._compute_metrics(eq_series, trades_df, df_test)

        return BacktestResult(
            equity_curve = eq_series,
            trades       = trades_df,
            metrics      = metrics,
            df           = df_test,
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _trades_to_df(trades: list[Trade]) -> pd.DataFrame:
        if not trades:
            return pd.DataFrame(columns=[
                "entry_time", "exit_time", "entry_price", "exit_price",
                "notional", "pnl", "pnl_pct", "duration_h",
                "exit_reason", "entry_regime", "entry_conf", "confirmations",
                "direction",
            ])
        rows = []
        for t in trades:
            rows.append({
                "entry_time":    t.entry_time,
                "exit_time":     t.exit_time,
                "entry_price":   round(t.entry_price, 2),
                "exit_price":    round(t.exit_price, 2) if t.exit_price else None,
                "notional":      round(t.notional, 2),
                "pnl":           round(t.pnl, 2),
                "pnl_pct":       round(t.pnl_pct, 3),
                "duration_h":    round(t.duration_h, 1),
                "exit_reason":   t.exit_reason,
                "entry_regime":  t.entry_regime,
                "entry_conf":    round(t.entry_conf, 4),
                "confirmations": t.confirmations,
                "direction":     t.direction,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _compute_metrics(
        equity:    pd.Series,
        trades:    pd.DataFrame,
        df_test:   pd.DataFrame,
    ) -> dict:
        if equity.empty:
            return {}

        initial = config.INITIAL_CAPITAL
        final   = equity.iloc[-1]

        # ── Returns
        total_return_pct = (final / initial - 1) * 100

        # ── Buy-and-hold benchmark
        bh_start  = df_test["Close"].iloc[0]
        bh_end    = df_test["Close"].iloc[-1]
        bh_return = (bh_end / bh_start - 1) * 100

        # Buy-and-hold equity curve (same starting capital)
        bh_equity = (df_test["Close"] / bh_start) * initial

        # ── Alpha (simple excess return vs B&H)
        alpha_pct = total_return_pct - bh_return

        # ── Drawdown
        rolling_max = equity.cummax()
        drawdown    = (equity - rolling_max) / rolling_max * 100
        max_dd      = drawdown.min()

        # ── Sharpe (annualised, hourly returns)
        hourly_ret = equity.pct_change().dropna()
        if hourly_ret.std() > 0:
            sharpe = (hourly_ret.mean() / hourly_ret.std()) * np.sqrt(24 * 365)
        else:
            sharpe = 0.0

        # ── Trade stats
        n_trades = len(trades)
        if n_trades > 0:
            wins      = (trades["pnl"] > 0).sum()
            win_rate  = wins / n_trades * 100
            avg_win   = trades.loc[trades["pnl"] > 0, "pnl"].mean() if wins > 0 else 0
            avg_loss  = trades.loc[trades["pnl"] < 0, "pnl"].mean() if (n_trades - wins) > 0 else 0
            avg_dur   = trades["duration_h"].mean()
            profit_factor = (
                trades.loc[trades["pnl"] > 0, "pnl"].sum() /
                abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
                if abs(trades.loc[trades["pnl"] < 0, "pnl"].sum()) > 0
                else float("inf")
            )
        else:
            wins = win_rate = avg_win = avg_loss = avg_dur = profit_factor = 0

        # ── Direction breakdown
        long_trades  = 0
        short_trades = 0
        long_win_rate  = 0.0
        short_win_rate = 0.0

        if n_trades > 0 and "direction" in trades.columns:
            long_mask  = trades["direction"] == "LONG"
            short_mask = trades["direction"] == "SHORT"
            long_trades  = long_mask.sum()
            short_trades = short_mask.sum()

            if long_trades > 0:
                long_wins = (trades.loc[long_mask, "pnl"] > 0).sum()
                long_win_rate = long_wins / long_trades * 100
            if short_trades > 0:
                short_wins = (trades.loc[short_mask, "pnl"] > 0).sum()
                short_win_rate = short_wins / short_trades * 100

        return {
            "initial_capital":    initial,
            "final_equity":       round(final, 2),
            "total_return_pct":   round(total_return_pct, 2),
            "bh_return_pct":      round(bh_return, 2),
            "alpha_pct":          round(alpha_pct, 2),
            "max_drawdown_pct":   round(max_dd, 2),
            "sharpe_ratio":       round(sharpe, 3),
            "n_trades":           n_trades,
            "win_rate_pct":       round(win_rate, 1),
            "avg_win_usd":        round(avg_win, 2),
            "avg_loss_usd":       round(avg_loss, 2),
            "avg_duration_h":     round(avg_dur, 1),
            "profit_factor":      round(profit_factor, 3),
            "leverage":           config.LEVERAGE,
            "bh_equity":          bh_equity,
            "long_trades":        long_trades,
            "short_trades":       short_trades,
            "long_win_rate_pct":  round(long_win_rate, 1),
            "short_win_rate_pct": round(short_win_rate, 1),
        }
