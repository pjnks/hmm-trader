"""
citrine_backtest.py
───────────────────
Walk-forward portfolio backtester for the CITRINE rotation strategy.

Pre-caches all tickers' daily data, then runs rolling 6m-train / 3m-test
windows with daily rebalancing. Compares against QQQ buy-and-hold benchmark.

Usage
─────
  python citrine_backtest.py                                # full NDX100
  python citrine_backtest.py --tickers AAPL,TSLA,NVDA       # subset test
  python citrine_backtest.py --train-months 6 --test-months 3
  python citrine_backtest.py --long-only                     # no SHORT positions
  python citrine_backtest.py --cooldown time                 # test cooldown modes
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Path bootstrap ───────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.data_fetcher import build_hmm_features
from src.hmm_model import HMMRegimeModel
from src.indicators import (
    attach_all,
    compute_realized_vol_ratio,
    compute_return_autocorr,
    compute_vol_price_diverge,
    compute_candle_body_ratio,
    compute_bb_width,
)
from src.strategy import build_signal_series
from src.citrine_scanner import CitrineScanner, TickerScan
from src.citrine_allocator import CitrineAllocator, PortfolioWeight

# Data fetching
from walk_forward_ndx import fetch_equity_daily, _attach_hmm_features

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("citrine_backtest")

# ── Output paths ─────────────────────────────────────────────────────────────
RESULTS_CSV  = ROOT / "citrine_wf_results.csv"
EQUITY_HTML  = ROOT / "citrine_wf_equity.html"

# Warmup bars for indicators
_WARMUP = max(
    config.TREND_MA_PERIOD,
    config.MACD_SLOW + config.MACD_SIGNAL,
    config.ADX_PERIOD * 3,
    config.VOLATILITY_PERIOD * 5,
)

# Fee model
_TAKER_FEE   = config.CITRINE_TAKER_FEE       # 0.04% per side
_SLIPPAGE    = config.CITRINE_SLIPPAGE_BPS     # 10 bps


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PortfolioWindowResult:
    """Metrics for one walk-forward test window."""
    window:              int
    train_start:         str
    train_end:           str
    test_start:          str
    test_end:            str
    return_pct:          float
    bh_return_pct:       float       # QQQ buy-and-hold
    alpha_pct:           float
    sharpe_ratio:        float
    max_drawdown_pct:    float
    avg_positions:       float       # mean positions held
    avg_cash_pct:        float
    total_trades:        int         # entry + exit events
    turnover_pct:        float       # total notional traded / avg equity
    start_equity:        float
    end_equity:          float
    test_days:           int


# ─────────────────────────────────────────────────────────────────────────────
class CitrineBacktester:
    """
    Walk-forward portfolio backtester with daily rebalancing.

    For each rolling window:
      1. Fit one HMM per ticker on the training slice
      2. Each day in the test slice: predict regimes, allocate, rebalance
      3. Chain equity into the next window
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        capital: float = config.CITRINE_INITIAL_CAPITAL,
        long_only: bool = config.CITRINE_LONG_ONLY,
        cooldown_mode: str = config.CITRINE_COOLDOWN_MODE,
        quiet: bool = False,
    ):
        self.tickers = tickers or config.CITRINE_UNIVERSE
        self.initial_capital = capital
        self.long_only = long_only
        self.cooldown_mode = cooldown_mode
        self.quiet = quiet

        # HMM params
        self.hmm_params = config.CITRINE_HMM_DEFAULTS.copy()

        # Per-ticker optimized HMM configs (loaded from optimizer results)
        self._per_ticker_params: dict[str, dict] = {}
        self._load_per_ticker_configs()

    def _load_per_ticker_configs(self) -> None:
        """Load per-ticker optimized HMM configs if available."""
        import json
        config_path = Path(__file__).parent / "citrine_per_ticker_configs.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    self._per_ticker_params = json.load(f)
                if not self.quiet:
                    log.info(f"Loaded per-ticker configs for {len(self._per_ticker_params)} tickers")
            except Exception as e:
                log.warning(f"Failed to load per-ticker configs: {e}")

    def run_walk_forward(
        self,
        train_months: int = 6,
        test_months: int = 3,
    ) -> tuple[list[PortfolioWindowResult], pd.Series]:
        """
        Main entry point. Returns per-window results and equity time series.
        """
        # Step 1: Pre-cache all ticker data
        all_data = self._precache_data()
        if not all_data:
            print("ERROR: No data available for any ticker")
            return [], pd.Series(dtype=float)

        # Also fetch QQQ for benchmark
        print("  Fetching QQQ benchmark ...", end=" ", flush=True)
        qqq_data = fetch_equity_daily("QQQ", years=20)
        print(f"{len(qqq_data)} bars")

        # Step 2: Determine window boundaries from available data
        # Use the latest common start date across all tickers
        start_dates = []
        end_dates = []
        for ticker, df in all_data.items():
            if len(df) > _WARMUP + 50:
                start_dates.append(df.index[0])
                end_dates.append(df.index[-1])

        if not start_dates:
            print("ERROR: Insufficient data across tickers")
            return [], pd.Series(dtype=float)

        data_start = max(start_dates)
        data_end = min(end_dates)
        print(f"\n  Data range: {data_start.date()} → {data_end.date()}")

        # Build window boundaries
        windows = self._build_windows(data_start, data_end, train_months, test_months)
        if not windows:
            print("ERROR: Not enough data for even one walk-forward window")
            return [], pd.Series(dtype=float)

        print(f"  Windows: {len(windows)} ({train_months}m train / {test_months}m test)")

        # Step 3: Run each window
        results: list[PortfolioWindowResult] = []
        equity_series = pd.Series(dtype=float)
        chain_equity = self.initial_capital

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            if not self.quiet:
                print(f"\n  ── Window {i+1}/{len(windows)} ──")
                print(f"    Train: {train_start.date()} → {train_end.date()}")
                print(f"    Test:  {test_start.date()} → {test_end.date()}")

            result, window_equity = self._run_window(
                all_data, qqq_data,
                train_start, train_end, test_start, test_end,
                chain_equity, i + 1,
            )

            if result is not None:
                results.append(result)
                if not window_equity.empty:
                    equity_series = pd.concat([equity_series, window_equity])
                chain_equity = result.end_equity

                if not self.quiet:
                    print(f"    Return: {result.return_pct:+.2f}%  "
                          f"BH: {result.bh_return_pct:+.2f}%  "
                          f"Alpha: {result.alpha_pct:+.2f}%  "
                          f"Sharpe: {result.sharpe_ratio:.3f}  "
                          f"DD: {result.max_drawdown_pct:.1f}%  "
                          f"Trades: {result.total_trades}  "
                          f"Avg pos: {result.avg_positions:.1f}")

        # Step 4: Compute aggregate metrics
        if results:
            self._print_summary(results, chain_equity)
            self._save_results(results)
            self._generate_chart(equity_series, qqq_data, results)

        return results, equity_series

    # ── Data Pre-caching ─────────────────────────────────────────────────────

    def _precache_data(self) -> dict[str, pd.DataFrame]:
        """Fetch all tickers' daily data from Polygon. Rate-limit aware."""
        all_data: dict[str, pd.DataFrame] = {}
        total = len(self.tickers)
        print(f"\n  Pre-caching {total} tickers (est. {total * 12 // 60}min) ...")

        for i, ticker in enumerate(self.tickers):
            print(f"    [{i+1}/{total}] {ticker} ...", end=" ", flush=True)
            try:
                df = fetch_equity_daily(ticker, years=20)
                if not df.empty:
                    all_data[ticker] = df
                    print(f"{len(df)} bars")
                else:
                    print("EMPTY")
            except Exception as e:
                print(f"ERROR: {e}")

            if i < total - 1:
                time.sleep(12)  # Polygon rate limit

        print(f"\n  Cached: {len(all_data)}/{total} tickers successfully")
        return all_data

    # ── Window Construction ──────────────────────────────────────────────────

    @staticmethod
    def _build_windows(
        data_start: pd.Timestamp,
        data_end: pd.Timestamp,
        train_months: int,
        test_months: int,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Build rolling walk-forward window boundaries."""
        windows = []
        cursor = data_start

        while True:
            train_start = cursor
            train_end = train_start + pd.DateOffset(months=train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_months)

            if test_end > data_end:
                break

            windows.append((train_start, train_end, test_start, test_end))
            cursor = test_start  # non-overlapping windows

        return windows

    # ── Per-Window Execution ─────────────────────────────────────────────────

    def _run_window(
        self,
        all_data: dict[str, pd.DataFrame],
        qqq_data: pd.DataFrame,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
        chain_equity: float,
        window_idx: int,
    ) -> tuple[Optional[PortfolioWindowResult], pd.Series]:
        """Run one walk-forward window with daily rebalancing."""

        # 1. Fit HMMs on training data
        models = self._fit_models(all_data, train_start, train_end)
        if not models:
            return None, pd.Series(dtype=float)

        # 2. Prepare test data slices with indicators
        test_data = self._prepare_test_data(
            all_data, models, train_start, test_start, test_end
        )

        # 3. Get test trading days (union of all tickers' dates in test window)
        all_dates: set[pd.Timestamp] = set()
        for ticker, df in test_data.items():
            mask = (df.index >= test_start) & (df.index < test_end)
            all_dates.update(df.index[mask])
        trading_days = sorted(all_dates)

        if not trading_days:
            return None, pd.Series(dtype=float)

        # 4. Run daily rebalancing loop
        allocator = CitrineAllocator(
            capital=chain_equity,
            long_only=self.long_only,
            cooldown_mode=self.cooldown_mode,
        )

        equity = chain_equity
        cash = chain_equity
        positions: dict[str, dict] = {}  # ticker → {shares, entry_price, direction, notional}
        equity_history: dict[pd.Timestamp, float] = {}
        total_trades = 0
        total_notional_traded = 0.0
        daily_returns: list[float] = []
        position_counts: list[int] = []
        cash_pcts: list[float] = []
        prev_equity = equity

        for day in trading_days:
            # a. Mark-to-market existing positions
            mtm_value = cash
            for ticker, pos in positions.items():
                if ticker in test_data and day in test_data[ticker].index:
                    current_price = float(test_data[ticker].loc[day, "Close"])
                    if pos["direction"] == "LONG":
                        pos_value = pos["shares"] * current_price
                    else:  # SHORT
                        pos_value = pos["notional"] + (pos["entry_price"] - current_price) * pos["shares"]
                    mtm_value += pos_value

            equity = mtm_value

            # b. Generate scans from pre-computed data
            scans = self._generate_scans_for_day(test_data, models, day)

            # c. Get target allocation
            weights, cash_pct = allocator.allocate(scans)

            # d. Execute rebalancing trades
            new_positions, new_cash, trades = self._execute_rebalance(
                weights, positions, cash, equity, test_data, day
            )

            total_trades += trades
            total_notional_traded += sum(
                abs(new_positions.get(t, {}).get("notional", 0) - positions.get(t, {}).get("notional", 0))
                for t in set(list(new_positions.keys()) + list(positions.keys()))
            )

            positions = new_positions
            cash = new_cash

            # e. Final MTM for the day (after rebalancing)
            final_equity = cash
            for ticker, pos in positions.items():
                if ticker in test_data and day in test_data[ticker].index:
                    current_price = float(test_data[ticker].loc[day, "Close"])
                    if pos["direction"] == "LONG":
                        final_equity += pos["shares"] * current_price
                    else:
                        final_equity += pos["notional"] + (pos["entry_price"] - current_price) * pos["shares"]

            equity = final_equity
            equity_history[day] = equity

            # Track daily return
            if prev_equity > 0:
                daily_returns.append((equity - prev_equity) / prev_equity)
            prev_equity = equity

            position_counts.append(len(positions))
            cash_pcts.append(cash / equity if equity > 0 else 1.0)

            # Update allocator holdings
            allocator.update_holdings(weights)

        # 5. Compute window metrics
        equity_series = pd.Series(equity_history)

        # QQQ benchmark
        qqq_mask = (qqq_data.index >= test_start) & (qqq_data.index < test_end)
        qqq_test = qqq_data[qqq_mask]
        if len(qqq_test) >= 2:
            bh_return = (float(qqq_test["Close"].iloc[-1]) / float(qqq_test["Close"].iloc[0]) - 1) * 100
        else:
            bh_return = 0.0

        return_pct = (equity / chain_equity - 1) * 100
        alpha_pct = return_pct - bh_return

        # Sharpe
        if daily_returns and len(daily_returns) > 1:
            mean_r = np.mean(daily_returns)
            std_r = np.std(daily_returns, ddof=1)
            sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0.0
        else:
            sharpe = 0.0

        # Max drawdown
        if not equity_series.empty:
            running_max = equity_series.expanding().max()
            drawdowns = (equity_series - running_max) / running_max * 100
            max_dd = float(drawdowns.min())
        else:
            max_dd = 0.0

        # Turnover
        avg_equity = (chain_equity + equity) / 2
        turnover = (total_notional_traded / avg_equity * 100) if avg_equity > 0 else 0.0

        result = PortfolioWindowResult(
            window=window_idx,
            train_start=str(train_start.date()),
            train_end=str(train_end.date()),
            test_start=str(test_start.date()),
            test_end=str(test_end.date()),
            return_pct=round(return_pct, 2),
            bh_return_pct=round(bh_return, 2),
            alpha_pct=round(alpha_pct, 2),
            sharpe_ratio=round(sharpe, 3),
            max_drawdown_pct=round(max_dd, 2),
            avg_positions=round(np.mean(position_counts), 1) if position_counts else 0.0,
            avg_cash_pct=round(np.mean(cash_pcts) * 100, 1) if cash_pcts else 100.0,
            total_trades=total_trades,
            turnover_pct=round(turnover, 1),
            start_equity=round(chain_equity, 2),
            end_equity=round(equity, 2),
            test_days=len(trading_days),
        )

        return result, equity_series

    # ── HMM Fitting ──────────────────────────────────────────────────────────

    def _fit_models(
        self,
        all_data: dict[str, pd.DataFrame],
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
    ) -> dict[str, HMMRegimeModel]:
        """Fit one HMM per ticker on the training slice."""
        models: dict[str, HMMRegimeModel] = {}

        for ticker, df_full in all_data.items():
            try:
                # Get per-ticker params (fall back to defaults)
                params = self._per_ticker_params.get(ticker, self.hmm_params)
                feature_set = params.get("feature_set", "base")
                n_states = params.get("n_states", 6)
                cov_type = params.get("cov_type", "diag")
                feature_cols = config.FEATURE_SETS.get(feature_set, config.FEATURE_SETS["base"])

                # Slice training data
                mask = (df_full.index >= train_start) & (df_full.index < train_end)
                df_train = df_full[mask].copy()

                if len(df_train) < _WARMUP + 20:
                    continue

                # Build features
                df_feat = _attach_hmm_features(df_train, feature_set)
                df_feat = df_feat.dropna(subset=feature_cols)

                if len(df_feat) < 50:
                    continue

                # Fit HMM
                model = HMMRegimeModel(n_states=n_states, feature_cols=feature_cols)
                model.fit(df_feat)

                if model.converged:
                    models[ticker] = model

            except Exception as e:
                log.debug(f"{ticker}: HMM fit failed: {e}")

        if not self.quiet:
            print(f"    HMMs fitted: {len(models)}/{len(all_data)} converged")

        return models

    # ── Test Data Preparation ────────────────────────────────────────────────

    def _prepare_test_data(
        self,
        all_data: dict[str, pd.DataFrame],
        models: dict[str, HMMRegimeModel],
        train_start: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        """
        Prepare test data with regime predictions and indicators.
        Uses warmup bars from training period for indicator computation.
        """
        test_data: dict[str, pd.DataFrame] = {}

        for ticker, model in models.items():
            if ticker not in all_data:
                continue

            try:
                # Get per-ticker params (fall back to defaults)
                params = self._per_ticker_params.get(ticker, self.hmm_params)
                feature_set = params.get("feature_set", "base")
                min_confirms = params.get("confirmations", 7)

                df_full = all_data[ticker]
                # Include warmup bars before test window
                warmup_start = test_start - pd.DateOffset(days=_WARMUP * 2)
                mask = (df_full.index >= warmup_start) & (df_full.index < test_end)
                df = df_full[mask].copy()

                if len(df) < _WARMUP:
                    continue

                # Build features and predict
                df = _attach_hmm_features(df, feature_set)
                df = model.predict(df)
                df = attach_all(df)

                # Build signal series for confirmation counts
                saved = config.MIN_CONFIRMATIONS
                config.MIN_CONFIRMATIONS = min_confirms
                try:
                    df = build_signal_series(df, use_regime_mapper=False)
                finally:
                    config.MIN_CONFIRMATIONS = saved

                test_data[ticker] = df

            except Exception as e:
                log.debug(f"{ticker}: test data prep failed: {e}")

        return test_data

    # ── Daily Scan Generation ────────────────────────────────────────────────

    def _generate_scans_for_day(
        self,
        test_data: dict[str, pd.DataFrame],
        models: dict[str, HMMRegimeModel],
        day: pd.Timestamp,
    ) -> list[TickerScan]:
        """Generate TickerScan objects for each ticker on a given day."""
        scans: list[TickerScan] = []

        for ticker, df in test_data.items():
            if day not in df.index:
                continue

            row = df.loc[day]
            regime_cat = str(row.get("regime_cat", "CHOP"))
            confidence = float(row.get("confidence", 0.0))
            confirmations = int(row.get("confirmation_count", 0))
            current_price = float(row.get("Close", 0.0))

            # Count SHORT confirmations
            short_confirms = self._count_short_confirms(row)

            # Persistence: count consecutive regime days up to this day
            mask = df.index <= day
            regime_series = df.loc[mask, "regime_cat"].values
            persistence = 0
            for i in range(len(regime_series) - 1, -1, -1):
                if regime_series[i] == regime_cat:
                    persistence += 1
                else:
                    break

            # Realized vol
            close_mask = df.index <= day
            closes = df.loc[close_mask, "Close"]
            if len(closes) > 20:
                log_rets = np.log(closes / closes.shift(1)).dropna()
                vol = float(log_rets.iloc[-20:].std() * np.sqrt(252))
            else:
                vol = 0.30

            scans.append(TickerScan(
                ticker=ticker,
                regime_cat=regime_cat,
                confidence=confidence,
                persistence=persistence,
                realized_vol=vol if np.isfinite(vol) else 0.30,
                confirmations=confirmations,
                confirmations_short=short_confirms,
                current_price=current_price,
                sector=config.CITRINE_SECTORS.get(ticker, "Unknown"),
                hmm_converged=True,
            ))

        return scans

    @staticmethod
    def _count_short_confirms(row: pd.Series) -> int:
        """Count SHORT indicator confirmations from a row."""
        checks = 0
        rsi = row.get("rsi", 50)
        if config.RSI_LOWER < rsi < config.RSI_UPPER:
            checks += 1
        if row.get("momentum", 0) < 0:
            checks += 1
        vol = row.get("volatility", 0)
        vol_med = row.get("vol_median", vol)
        if vol_med > 0 and vol < config.VOLATILITY_MULT * vol_med:
            checks += 1
        if row.get("volume_ratio", 0) > config.VOLUME_MULT:
            checks += 1
        if row.get("adx", 0) > config.ADX_MIN_SHORT:
            checks += 1
        sma50 = row.get("sma_50", 0)
        close = row.get("Close", 0)
        if sma50 > 0 and close < sma50:
            checks += 1
        macd = row.get("macd", 0)
        macd_sig = row.get("macd_signal", 0)
        if macd < macd_sig:
            checks += 1
        stoch_k = row.get("stoch_k", 50)
        if stoch_k > (100 - config.STOCH_UPPER):
            checks += 1
        return checks

    # ── Rebalancing ──────────────────────────────────────────────────────────

    def _execute_rebalance(
        self,
        weights: list[PortfolioWeight],
        current_positions: dict[str, dict],
        cash: float,
        equity: float,
        test_data: dict[str, pd.DataFrame],
        day: pd.Timestamp,
    ) -> tuple[dict[str, dict], float, int]:
        """
        Execute rebalancing trades based on target weights.
        Returns (new_positions, new_cash, trade_count).
        """
        new_positions: dict[str, dict] = {}
        new_cash = cash
        trades = 0

        # Process exits first (free up cash)
        for w in weights:
            if w.action == "EXIT" and w.ticker in current_positions:
                pos = current_positions[w.ticker]
                if w.ticker in test_data and day in test_data[w.ticker].index:
                    exit_price = float(test_data[w.ticker].loc[day, "Close"])

                    # Apply slippage
                    if pos["direction"] == "LONG":
                        exit_price *= (1 - _SLIPPAGE / 10000)  # worse fill for sells
                    else:
                        exit_price *= (1 + _SLIPPAGE / 10000)  # worse fill for covers

                    # Compute P&L
                    if pos["direction"] == "LONG":
                        pnl = pos["shares"] * (exit_price - pos["entry_price"])
                    else:
                        pnl = pos["shares"] * (pos["entry_price"] - exit_price)

                    # Fee on exit
                    fee = pos["notional"] * _TAKER_FEE
                    pnl -= fee

                    new_cash += pos["notional"] + pnl
                    trades += 1

        # Process entries and holds
        for w in weights:
            if w.action in ("ENTER", "SCALE_UP"):
                if w.ticker in test_data and day in test_data[w.ticker].index:
                    entry_price = float(test_data[w.ticker].loc[day, "Close"])

                    # Apply slippage
                    if w.direction == "LONG":
                        entry_price *= (1 + _SLIPPAGE / 10000)
                    else:
                        entry_price *= (1 - _SLIPPAGE / 10000)

                    notional = w.notional_usd

                    # If already held (SCALE_UP), close old and re-enter at new size
                    if w.ticker in current_positions:
                        old = current_positions[w.ticker]
                        old_price = float(test_data[w.ticker].loc[day, "Close"])
                        if old["direction"] == "LONG":
                            old_value = old["shares"] * old_price
                        else:
                            old_value = old["notional"] + (old["entry_price"] - old_price) * old["shares"]
                        new_cash += old_value
                        # No trade count for rebalance within position

                    # Check we have enough cash
                    fee = notional * _TAKER_FEE
                    if new_cash >= notional + fee:
                        shares = notional / entry_price
                        new_cash -= (notional + fee)
                        new_positions[w.ticker] = {
                            "shares": shares,
                            "entry_price": entry_price,
                            "direction": w.direction,
                            "notional": notional,
                        }
                        if w.action == "ENTER":
                            trades += 1

            elif w.action == "HOLD":
                # Keep existing position unchanged
                if w.ticker in current_positions:
                    new_positions[w.ticker] = current_positions[w.ticker]

        return new_positions, new_cash, trades

    # ── Output ───────────────────────────────────────────────────────────────

    def _print_summary(
        self, results: list[PortfolioWindowResult], final_equity: float
    ) -> None:
        """Print aggregate walk-forward summary."""
        print(f"\n{'='*70}")
        print(f"  CITRINE WALK-FORWARD SUMMARY")
        print(f"{'='*70}")

        sharpes = [r.sharpe_ratio for r in results]
        returns = [r.return_pct for r in results]
        alphas = [r.alpha_pct for r in results]
        pos_windows = sum(1 for r in results if r.return_pct > 0)

        total_return = (final_equity / self.initial_capital - 1) * 100
        avg_sharpe = np.mean(sharpes) if sharpes else 0
        std_sharpe = np.std(sharpes) if len(sharpes) > 1 else 0

        print(f"\n  Total return:    {total_return:+.2f}%")
        print(f"  Final equity:    ${final_equity:,.2f}")
        print(f"  Mean Sharpe:     {avg_sharpe:.3f} (±{std_sharpe:.3f})")
        print(f"  Positive windows: {pos_windows}/{len(results)}")
        print(f"  Mean alpha:      {np.mean(alphas):+.2f}%")
        print(f"  Mean positions:  {np.mean([r.avg_positions for r in results]):.1f}")
        print(f"  Mean cash:       {np.mean([r.avg_cash_pct for r in results]):.0f}%")
        print(f"  Total trades:    {sum(r.total_trades for r in results)}")
        print(f"  Mode:            {'Long-only' if self.long_only else 'Long + Short'}")
        print(f"  Cooldown:        {self.cooldown_mode}")

    def _save_results(self, results: list[PortfolioWindowResult]) -> None:
        """Save per-window results to CSV."""
        rows = [asdict(r) for r in results]
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_CSV, index=False)
        print(f"\n  Saved: {RESULTS_CSV}")

    def _generate_chart(
        self,
        equity_series: pd.Series,
        qqq_data: pd.DataFrame,
        results: list[PortfolioWindowResult],
    ) -> None:
        """Generate Plotly equity curve chart."""
        if equity_series.empty:
            return

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            subplot_titles=["CITRINE Portfolio Equity", "Drawdown"],
            vertical_spacing=0.08,
        )

        # Portfolio equity
        fig.add_trace(
            go.Scatter(
                x=equity_series.index,
                y=equity_series.values,
                name="CITRINE",
                line=dict(color="#00ff88", width=2),
            ),
            row=1, col=1,
        )

        # QQQ benchmark (normalized to same starting equity)
        qqq_mask = (qqq_data.index >= equity_series.index[0]) & \
                   (qqq_data.index <= equity_series.index[-1])
        qqq_test = qqq_data[qqq_mask]
        if len(qqq_test) > 0:
            qqq_normalized = (qqq_test["Close"] / float(qqq_test["Close"].iloc[0])) * self.initial_capital
            fig.add_trace(
                go.Scatter(
                    x=qqq_normalized.index,
                    y=qqq_normalized.values,
                    name="QQQ (B&H)",
                    line=dict(color="#888888", width=1, dash="dash"),
                ),
                row=1, col=1,
            )

        # Drawdown
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                name="Drawdown %",
                fill="tozeroy",
                line=dict(color="#ff2244", width=1),
            ),
            row=2, col=1,
        )

        fig.update_layout(
            template="plotly_dark",
            height=700,
            title=f"CITRINE Walk-Forward: {len(results)} windows | "
                  f"{'Long-only' if self.long_only else 'Long+Short'} | "
                  f"Cooldown: {self.cooldown_mode}",
            showlegend=True,
        )
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

        fig.write_html(str(EQUITY_HTML))
        print(f"  Chart:  {EQUITY_HTML}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="CITRINE Walk-Forward Portfolio Backtester"
    )
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated ticker list (default: full NDX100)")
    parser.add_argument("--train-months", type=int, default=6,
                        help="Training window months (default: 6)")
    parser.add_argument("--test-months", type=int, default=3,
                        help="Test window months (default: 3)")
    parser.add_argument("--capital", type=float, default=config.CITRINE_INITIAL_CAPITAL,
                        help="Starting capital (default: $25,000)")
    parser.add_argument("--long-only", action="store_true",
                        help="Long-only mode (no SHORT positions)")
    parser.add_argument("--long-short", action="store_true",
                        help="Long + Short mode (default)")
    parser.add_argument("--cooldown", type=str, default=config.CITRINE_COOLDOWN_MODE,
                        choices=["none", "time", "threshold"],
                        help="Cooldown mode (default: none)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-window output")
    args = parser.parse_args()

    tickers = args.tickers.split(",") if args.tickers else None
    long_only = args.long_only or (not args.long_short and config.CITRINE_LONG_ONLY)

    print("\n" + "="*70)
    print("  CITRINE — Walk-Forward Portfolio Backtester")
    print("="*70)
    print(f"  Tickers:   {len(tickers) if tickers else len(config.CITRINE_UNIVERSE)}")
    print(f"  Windows:   {args.train_months}m train / {args.test_months}m test")
    print(f"  Capital:   ${args.capital:,.0f}")
    print(f"  Mode:      {'Long-only' if long_only else 'Long + Short'}")
    print(f"  Cooldown:  {args.cooldown}")

    bt = CitrineBacktester(
        tickers=tickers,
        capital=args.capital,
        long_only=long_only,
        cooldown_mode=args.cooldown,
    )

    results, equity = bt.run_walk_forward(
        train_months=args.train_months,
        test_months=args.test_months,
    )

    if not results:
        print("\n  No results — check data availability.")
    else:
        print(f"\n  Done! {len(results)} windows complete.")


if __name__ == "__main__":
    main()
