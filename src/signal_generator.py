"""
signal_generator.py
───────────────────
Real-time signal generator for live trading.

Runs every timeframe (4h for SOL) and:
1. Fetches latest OHLCV from Polygon
2. Runs HMM regime detection + indicator checks
3. Generates BUY/SELL/HOLD signals
4. Logs signal strength (# confirmations met)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.data_fetcher import fetch_btc_hourly, resample_ohlcv, build_hmm_features
from src.hmm_model import HMMRegimeModel
from src.ensemble import EnsembleHMM
from src.indicators import attach_all
from src.strategy import build_signal_series

log = logging.getLogger(__name__)


class SignalGenerator:
    """
    Generate live trading signals from real-time market data.

    Workflow:
    1. Fetch latest bars from Polygon
    2. Fit HMM model on recent history
    3. Check 8 indicators
    4. Return signal (BUY, SELL, HOLD) + metadata
    """

    def __init__(
        self,
        ticker: str = config.TICKER,
        timeframe: str = config.TIMEFRAME,
        lookback_days: int = 90,
        use_ensemble: bool = True,
    ):
        self.ticker = ticker
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.use_ensemble = use_ensemble
        self.model = None
        self.last_signal_time = None

    def generate_signal(self) -> dict:
        """
        Generate a trading signal based on current market conditions.

        Returns
        -------
        {
            "timestamp": str,
            "signal": "BUY" | "SELL" | "HOLD",
            "regime": str,  # BULL, BEAR, CHOP
            "regime_confidence": float,
            "confirmations": int,  # how many of 8 indicators confirm
            "current_price": float,
            "rsi": float,
            "momentum": float,
            "adx": float,
            "error": str | None,
        }
        """
        try:
            # ── Fetch latest data ──────────────────────────────────
            log.info(f"Fetching {self.ticker} {self.timeframe} data...")
            df_raw = fetch_btc_hourly(days=self.lookback_days, ticker=self.ticker)

            if df_raw.empty:
                return {
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    "signal": "HOLD",
                    "error": f"No data fetched for {self.ticker}",
                }

            # ── Resample to target timeframe ──────────────────────
            if self.timeframe != "1h":
                df_raw = resample_ohlcv(df_raw, self.timeframe)

            # ── Build HMM features ─────────────────────────────────
            df = build_hmm_features(df_raw)

            # Add extended features if needed (config-driven, Sprint 3.1)
            # Computes only the features required by the active feature set.
            needed = set(config.FEATURE_SETS.get(config.FEATURE_SET, []))
            if "realized_vol_ratio" in needed:
                from src.indicators import compute_realized_vol_ratio
                df["realized_vol_ratio"] = compute_realized_vol_ratio(df)
            if "return_autocorr" in needed:
                from src.indicators import compute_return_autocorr
                df["return_autocorr"] = compute_return_autocorr(df)
            if "candle_body_ratio" in needed:
                from src.indicators import compute_candle_body_ratio
                df["candle_body_ratio"] = compute_candle_body_ratio(df)
            if "bb_width" in needed:
                from src.indicators import compute_bb_width
                df["bb_width"] = compute_bb_width(df)
            if "realized_kurtosis" in needed:
                from src.indicators import compute_realized_kurtosis
                df["realized_kurtosis"] = compute_realized_kurtosis(df)
            if "volume_return_intensity" in needed:
                from src.indicators import compute_volume_return_intensity
                df["volume_return_intensity"] = compute_volume_return_intensity(df)
            if "return_momentum_ratio" in needed:
                from src.indicators import compute_return_momentum_ratio
                df["return_momentum_ratio"] = compute_return_momentum_ratio(df)

            # ── Drop NaN rows (extended features need warmup) ──────
            initial_rows = len(df)
            df = df.dropna()
            if len(df) < initial_rows:
                log.info(f"Dropped {initial_rows - len(df)} NaN rows during feature computation")

            # ── Fit HMM on last 90 days ────────────────────────────
            feature_cols = config.FEATURE_SETS[config.FEATURE_SET]

            if self.use_ensemble:
                self.model = EnsembleHMM(
                    cov_type=config.COV_TYPE,
                    feature_cols=feature_cols,
                )
                log.info("Using EnsembleHMM (3 models, n_states=%s)",
                         config.ENSEMBLE_N_STATES)
            else:
                self.model = HMMRegimeModel(
                    n_states=config.N_STATES,
                    cov_type=config.COV_TYPE,
                    feature_cols=feature_cols,
                )
                log.info("Using single HMMRegimeModel (n_states=%d)", config.N_STATES)

            self.model.fit(df)

            if not self.model.converged:
                log.warning("HMM model did not converge")
                return {
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    "signal": "HOLD",
                    "error": "HMM did not converge",
                }

            # ── Predict regimes ────────────────────────────────────
            df = self.model.predict(df)

            # ── Attach indicators ──────────────────────────────────
            df = attach_all(df)

            # ── Build signal series ────────────────────────────────
            df = build_signal_series(df, use_regime_mapper=False)

            # ── Extract latest bar ─────────────────────────────────
            latest = df.iloc[-1]

            current_price = float(latest["Close"])
            regime = latest.get("regime_cat", "UNKNOWN")
            confidence = latest.get("confidence", 0.0)
            confirmations = latest.get("confirmation_count", 0)

            # Determine signal from raw_long_signal + regime
            # build_signal_series() produces raw_long_signal (bool), NOT a "signal" column
            raw_long = bool(latest.get("raw_long_signal", False))
            if raw_long:
                signal = "BUY"
            elif regime == "BEAR":
                signal = "SELL"
            else:
                signal = "HOLD"

            # ── Log signal ─────────────────────────────────────────
            log.info(
                f"Signal generated: {signal} | {regime} (conf {confidence:.2f}) | "
                f"${current_price:.2f} | {confirmations}/8 confirmations"
            )

            return {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "signal": signal,
                "regime": regime,
                "regime_confidence": round(confidence, 3),
                "confirmations": confirmations,
                "current_price": round(current_price, 2),
                "rsi": round(float(latest.get("rsi", 0.0)), 2),
                "momentum": round(float(latest.get("momentum", 0.0)), 4),
                "adx": round(float(latest.get("adx", 0.0)), 2),
                "error": None,
            }

        except Exception as e:
            log.error(f"Error generating signal: {e}")
            return {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "signal": "HOLD",
                "error": str(e),
            }
