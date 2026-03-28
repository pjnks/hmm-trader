"""
citrine_scanner.py
──────────────────
Scans NDX100 tickers with HMM regime detection and produces per-ticker
metadata (regime, confidence, persistence, realized volatility, indicator
quality) for the CITRINE portfolio rotation engine.

Usage
─────
  from src.citrine_scanner import CitrineScanner
  scanner = CitrineScanner()                           # full NDX100
  scanner = CitrineScanner(tickers=["AAPL", "TSLA"])   # subset
  results = scanner.scan_all()                         # list[TickerScan]
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Path bootstrap ───────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.data_fetcher import build_hmm_features
from src.hmm_model import HMMRegimeModel
from src.indicators import (
    attach_all,
    compute_realized_vol_ratio,
    compute_return_autocorr,
    compute_candle_body_ratio,
    compute_bb_width,
    compute_vol_price_diverge,
)
from src.strategy import build_signal_series

log = logging.getLogger(__name__)

# Import Polygon fetcher from walk_forward_ndx
try:
    from walk_forward_ndx import fetch_equity_daily, _attach_hmm_features
except ImportError:
    # Fallback: inline minimal fetch if walk_forward_ndx unavailable
    fetch_equity_daily = None
    _attach_hmm_features = None

# Rate limit: 12s between Polygon calls (5 calls/min free tier)
_POLYGON_DELAY_S = 12.0

# Warmup bars required by indicators (SMA-50, ADX, MACD, etc.)
_WARMUP_BARS = max(
    config.TREND_MA_PERIOD,
    config.MACD_SLOW + config.MACD_SIGNAL,
    config.ADX_PERIOD * 3,
    config.VOLATILITY_PERIOD * 5,
)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TickerScan:
    """Result of scanning one ticker with HMM + indicators."""
    ticker:          str
    regime_cat:      str            # BULL / BEAR / CHOP
    confidence:      float          # HMM posterior probability (0-1)
    persistence:     int            # consecutive days in current regime (from end)
    realized_vol:    float          # 20-day annualized volatility
    confirmations:   int            # number of 8 indicator checks passing (LONG)
    confirmations_short: int        # number of 8 indicator checks passing (SHORT)
    current_price:   float
    sector:          str
    hmm_converged:   bool
    regime_half_life: float = 30.0   # expected regime half-life in bars (from transition matrix)
    scan_time:       str = ""
    error:           Optional[str] = None

    @property
    def is_bull(self) -> bool:
        return self.regime_cat == "BULL"

    @property
    def is_bear(self) -> bool:
        return self.regime_cat == "BEAR"


# ─────────────────────────────────────────────────────────────────────────────
class CitrineScanner:
    """
    Scans NDX100 tickers with HMM regime detection.

    For each ticker:
      1. Fetch daily OHLCV from Polygon.io (lookback_days history)
      2. Build HMM features (log_return, price_range, volume_change + extended)
      3. Fit HMMRegimeModel → regime_cat + confidence
      4. Compute all 8 strategy indicators
      5. Count confirmations (LONG and SHORT)
      6. Calculate persistence (consecutive BULL/BEAR days from end)
      7. Calculate 20-day annualized realized volatility

    Rate-limit aware: 12s between Polygon API calls.
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        lookback_days: int = 365,
        hmm_params: dict | None = None,
        quiet: bool = False,
    ):
        self.tickers = tickers or config.CITRINE_UNIVERSE
        self.lookback_days = lookback_days
        self.hmm_params = hmm_params or config.CITRINE_HMM_DEFAULTS
        self.quiet = quiet

        # Per-ticker HMM param overrides (loaded from optimizer results)
        self._per_ticker_params: dict[str, dict] = {}
        self._load_per_ticker_configs()

    def _load_per_ticker_configs(self) -> None:
        """Load per-ticker optimized HMM configs if available."""
        import json
        config_path = ROOT / "citrine_per_ticker_configs.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    self._per_ticker_params = json.load(f)
                if not self.quiet:
                    log.info(f"Loaded per-ticker configs for {len(self._per_ticker_params)} tickers")
            except Exception as e:
                log.warning(f"Failed to load per-ticker configs: {e}")

    def scan_all(self) -> list[TickerScan]:
        """
        Scan all tickers sequentially with rate-limiting.

        Returns list of TickerScan objects (one per ticker). Tickers that fail
        to fetch or fit are included with error field set.
        """
        results: list[TickerScan] = []
        total = len(self.tickers)

        for i, ticker in enumerate(self.tickers):
            try:
                if not self.quiet:
                    print(f"  [{i+1}/{total}] Scanning {ticker} ...", end=" ", flush=True)

                scan = self._scan_ticker(ticker)
                results.append(scan)

                if not self.quiet:
                    if scan.error:
                        print(f"ERROR: {scan.error}")
                    else:
                        print(f"{scan.regime_cat} conf={scan.confidence:.2f} "
                              f"persist={scan.persistence}d confirms={scan.confirmations}/8")

                # Rate limit: wait between Polygon calls (skip on last ticker)
                if i < total - 1:
                    time.sleep(_POLYGON_DELAY_S)

            except Exception as e:
                log.error(f"[CITRINE] ERROR: Failed to scan {ticker}: {e}")
                results.append(self._error_scan(ticker, f"Outer loop error: {e}"))

        return results

    def scan_from_data(self, all_data: dict[str, pd.DataFrame]) -> list[TickerScan]:
        """
        Scan tickers from pre-fetched data (no Polygon calls).
        Used by the backtester to avoid re-fetching during walk-forward.

        Parameters
        ----------
        all_data : dict mapping ticker -> DataFrame with OHLCV columns
        """
        results: list[TickerScan] = []
        for ticker in self.tickers:
            if ticker not in all_data or all_data[ticker].empty:
                results.append(self._error_scan(ticker, "No data available"))
                continue
            scan = self._process_dataframe(ticker, all_data[ticker])
            results.append(scan)
        return results

    def _scan_ticker(self, ticker: str) -> TickerScan:
        """Fetch data and run HMM + indicators for one ticker."""
        try:
            # Fetch daily data from Polygon
            if fetch_equity_daily is None:
                return self._error_scan(ticker, "walk_forward_ndx not available")

            df_raw = fetch_equity_daily(ticker, years=max(2, self.lookback_days // 365 + 1))
            if df_raw.empty or len(df_raw) < _WARMUP_BARS + 50:
                return self._error_scan(ticker, f"Insufficient data ({len(df_raw)} bars)")

            # Use only the last lookback_days of data
            if len(df_raw) > self.lookback_days:
                df_raw = df_raw.iloc[-self.lookback_days:]

            return self._process_dataframe(ticker, df_raw)

        except Exception as e:
            return self._error_scan(ticker, str(e))

    def _process_dataframe(self, ticker: str, df_raw: pd.DataFrame) -> TickerScan:
        """Process a pre-fetched DataFrame through HMM + indicators."""
        try:
            # Get HMM params for this ticker
            params = self._per_ticker_params.get(ticker, self.hmm_params)
            feature_set = params.get("feature_set", "base")
            n_states = params.get("n_states", 6)
            cov_type = params.get("cov_type", "diag")
            min_confirms = params.get("confirmations", 7)

            # Build HMM features
            if _attach_hmm_features is not None:
                df = _attach_hmm_features(df_raw.copy(), feature_set)
            else:
                df = build_hmm_features(df_raw.copy())

            # Get feature columns for HMM
            feature_cols = config.FEATURE_SETS.get(feature_set, config.FEATURE_SETS["base"])

            # Fit HMM
            saved_n = config.N_STATES
            saved_cov = config.COV_TYPE
            config.N_STATES = n_states
            config.COV_TYPE = cov_type
            try:
                model = HMMRegimeModel(n_states=n_states, feature_cols=feature_cols)
                model.fit(df)
                df = model.predict(df)
            finally:
                config.N_STATES = saved_n
                config.COV_TYPE = saved_cov

            # Attach all indicators
            df = attach_all(df)

            # Build signal series to get confirmation counts
            saved_confirms = config.MIN_CONFIRMATIONS
            config.MIN_CONFIRMATIONS = min_confirms
            try:
                df = build_signal_series(df, use_regime_mapper=False)
            finally:
                config.MIN_CONFIRMATIONS = saved_confirms

            # Extract latest bar metrics
            last = df.iloc[-1]
            regime_cat = str(last.get("regime_cat", "CHOP"))
            confidence = float(last.get("confidence", 0.0))
            current_price = float(last.get("Close", 0.0))

            # Count LONG confirmations from latest bar
            long_confirms = int(last.get("confirmation_count", 0))

            # Count SHORT confirmations manually from latest bar
            short_confirms = self._count_short_confirmations(last)

            # Persistence: consecutive days in current regime from end
            persistence = self._compute_persistence(df, regime_cat)

            # Realized volatility (20-day annualized)
            vol = self._compute_realized_vol(df)

            # Regime half-life from HMM transition matrix (for sojourn decay)
            try:
                half_life = model.get_regime_halflife(regime_cat)
            except Exception:
                half_life = 30.0  # conservative default

            # Sector
            sector = config.CITRINE_SECTORS.get(ticker, "Unknown")

            return TickerScan(
                ticker=ticker,
                regime_cat=regime_cat,
                confidence=confidence,
                persistence=persistence,
                realized_vol=vol,
                confirmations=long_confirms,
                confirmations_short=short_confirms,
                current_price=current_price,
                sector=sector,
                hmm_converged=model.converged,
                regime_half_life=half_life,
                scan_time=datetime.now(tz=timezone.utc).isoformat(),
            )

        except Exception as e:
            return self._error_scan(ticker, str(e))

    def _count_short_confirmations(self, row: pd.Series) -> int:
        """Count how many of the 8 SHORT confirmation checks pass."""
        checks = 0
        # 1. RSI in range (same as LONG)
        rsi = row.get("rsi", 50)
        if config.RSI_LOWER < rsi < config.RSI_UPPER:
            checks += 1
        # 2. Momentum < 0 (opposite of LONG)
        if row.get("momentum", 0) < 0:
            checks += 1
        # 3. Volatility < 2× median (same as LONG)
        vol = row.get("volatility", 0)
        vol_med = row.get("vol_median", vol)
        if vol_med > 0 and vol < config.VOLATILITY_MULT * vol_med:
            checks += 1
        # 4. Volume ratio > 1.1 (same as LONG)
        if row.get("volume_ratio", 0) > config.VOLUME_MULT:
            checks += 1
        # 5. ADX > threshold (same threshold for now)
        if row.get("adx", 0) > config.ADX_MIN_SHORT:
            checks += 1
        # 6. Close < SMA-50 (opposite of LONG)
        sma50 = row.get("sma_50", 0)
        close = row.get("Close", 0)
        if sma50 > 0 and close < sma50:
            checks += 1
        # 7. MACD < signal (opposite of LONG)
        macd = row.get("macd", 0)
        macd_sig = row.get("macd_signal", 0)
        if macd < macd_sig:
            checks += 1
        # 8. Stochastic %K > 20 (opposite of LONG: room to fall)
        stoch_k = row.get("stoch_k", 50)
        if stoch_k > (100 - config.STOCH_UPPER):
            checks += 1
        return checks

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

    @staticmethod
    def _compute_realized_vol(df: pd.DataFrame, window: int = 20) -> float:
        """Compute 20-day annualized realized volatility."""
        if "Close" not in df.columns or len(df) < window + 1:
            return 0.0
        log_returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        if len(log_returns) < window:
            return 0.0
        daily_vol = log_returns.iloc[-window:].std()
        annualized = daily_vol * np.sqrt(252)
        return float(annualized) if np.isfinite(annualized) else 0.0

    @staticmethod
    def _error_scan(ticker: str, error_msg: str) -> TickerScan:
        """Create a TickerScan with error state."""
        return TickerScan(
            ticker=ticker,
            regime_cat="CHOP",
            confidence=0.0,
            persistence=0,
            realized_vol=0.0,
            confirmations=0,
            confirmations_short=0,
            current_price=0.0,
            sector=config.CITRINE_SECTORS.get(ticker, "Unknown"),
            hmm_converged=False,
            scan_time=datetime.now(tz=timezone.utc).isoformat(),
            error=error_msg,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CITRINE Scanner — HMM regime scan for NDX100")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated ticker list (default: full NDX100)")
    parser.add_argument("--lookback", type=int, default=365,
                        help="Lookback days for HMM fitting (default: 365)")
    args = parser.parse_args()

    tickers = args.tickers.split(",") if args.tickers else None

    print("\n" + "="*70)
    print("  CITRINE SCANNER — NDX100 Regime Detection")
    print("="*70 + "\n")

    scanner = CitrineScanner(tickers=tickers, lookback_days=args.lookback)
    results = scanner.scan_all()

    # Summary
    bull = [s for s in results if s.is_bull and s.error is None]
    bear = [s for s in results if s.is_bear and s.error is None]
    chop = [s for s in results if s.regime_cat == "CHOP" and s.error is None]
    errors = [s for s in results if s.error is not None]

    print(f"\n{'='*70}")
    print(f"  RESULTS: {len(bull)} BULL | {len(bear)} BEAR | {len(chop)} CHOP | {len(errors)} errors")
    print(f"{'='*70}")

    if bull:
        print(f"\n  🟢 BULL tickers (sorted by confidence):")
        for s in sorted(bull, key=lambda x: x.confidence, reverse=True):
            print(f"    {s.ticker:6s} conf={s.confidence:.2f}  persist={s.persistence:2d}d  "
                  f"confirms={s.confirmations}/8  vol={s.realized_vol:.1%}  ${s.current_price:.2f}")

    if bear:
        print(f"\n  🔴 BEAR tickers (sorted by confidence):")
        for s in sorted(bear, key=lambda x: x.confidence, reverse=True):
            print(f"    {s.ticker:6s} conf={s.confidence:.2f}  persist={s.persistence:2d}d  "
                  f"confirms_short={s.confirmations_short}/8  vol={s.realized_vol:.1%}  ${s.current_price:.2f}")

    if errors:
        print(f"\n  ⚠️  Errors:")
        for s in errors:
            print(f"    {s.ticker:6s} — {s.error}")
