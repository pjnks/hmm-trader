"""
momentum_scanner.py
───────────────────
Short-term momentum confirmation layer for AGATE multi-ticker rotation.

Uses 1h bars with 72h (3-day) lookback to compute intra-regime momentum.
NOT a standalone model — it boosts the HMM regime signal's ranking score
when short-term momentum aligns with the regime direction.

Features:
  - Short-term return momentum (5-bar vs 20-bar)
  - Volume surge ratio (current vs 24h average)
  - RSI(7) — fast RSI for short-term overbought/oversold
  - VWAP deviation (price vs volume-weighted average)
  - Momentum acceleration (rate of change of momentum)

Output: momentum_score in [-1, +1]
  - Positive → upward momentum (confirms BULL regime)
  - Negative → downward momentum (confirms BEAR regime)
  - Near zero → no clear momentum direction
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Lookback and feature parameters
LOOKBACK_HOURS = 72        # 3 days of 1h bars
RSI_FAST_PERIOD = 7        # fast RSI for short-term signals
MOMENTUM_SHORT = 5         # 5-bar momentum (short-term)
MOMENTUM_LONG = 20         # 20-bar momentum (medium-term)
VOLUME_MA_PERIOD = 24      # 24h volume average
VWAP_PERIOD = 24           # 24h VWAP


def compute_momentum_score(df_1h: pd.DataFrame) -> dict:
    """
    Compute short-term momentum score from 1h OHLCV data.

    Parameters
    ----------
    df_1h : pd.DataFrame
        1-hour OHLCV data with at least LOOKBACK_HOURS rows.
        Required columns: Open, High, Low, Close, Volume

    Returns
    -------
    dict with keys:
        momentum_score: float in [-1, +1]
        components: dict of individual feature scores
        current_price: float
        bars_used: int
    """
    if len(df_1h) < LOOKBACK_HOURS:
        return {
            "momentum_score": 0.0,
            "components": {},
            "current_price": 0.0,
            "bars_used": len(df_1h),
            "error": f"Insufficient data: {len(df_1h)} bars < {LOOKBACK_HOURS}",
        }

    # Use last LOOKBACK_HOURS bars
    df = df_1h.tail(LOOKBACK_HOURS).copy()

    scores = {}

    # ── 1. Return momentum ratio (short vs long) ─────────────────────
    # Positive when short-term momentum exceeds long-term → acceleration
    returns = df["Close"].pct_change()
    mom_short = returns.rolling(MOMENTUM_SHORT).mean().iloc[-1]
    mom_long = returns.rolling(MOMENTUM_LONG).mean().iloc[-1]

    if abs(mom_long) > 1e-8:
        mom_ratio = np.clip(mom_short / abs(mom_long), -3, 3) / 3  # normalize to [-1, 1]
    else:
        mom_ratio = np.clip(mom_short * 1000, -1, 1)  # scale small values
    scores["momentum_ratio"] = float(mom_ratio)

    # ── 2. Volume surge ──────────────────────────────────────────────
    # High volume on up-moves = bullish confirmation
    vol_ma = df["Volume"].rolling(VOLUME_MA_PERIOD).mean()
    current_vol = df["Volume"].iloc[-1]
    vol_avg = vol_ma.iloc[-1]

    if vol_avg > 0:
        vol_surge = (current_vol / vol_avg) - 1.0  # 0 = average, >0 = above average
        # Weight by direction: volume surge on up-bar = positive, on down-bar = negative
        last_return = returns.iloc[-1] if not np.isnan(returns.iloc[-1]) else 0
        directional_vol = np.clip(vol_surge * np.sign(last_return), -1, 1)
    else:
        directional_vol = 0.0
    scores["volume_surge"] = float(directional_vol)

    # ── 3. Fast RSI (7-period) ───────────────────────────────────────
    # Maps RSI to [-1, +1]: RSI=70→+0.8, RSI=30→-0.8, RSI=50→0
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_FAST_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(RSI_FAST_PERIOD).mean()
    rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-10)
    rsi_fast = 100 - (100 / (1 + rs))
    rsi_score = (rsi_fast - 50) / 50  # normalize: 50→0, 70→+0.4, 30→-0.4
    rsi_score = np.clip(rsi_score, -1, 1)
    scores["rsi_fast"] = float(rsi_score)

    # ── 4. VWAP deviation ────────────────────────────────────────────
    # Price above VWAP = bullish momentum, below = bearish
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap_window = df.tail(VWAP_PERIOD)
    tp_window = typical_price.tail(VWAP_PERIOD)
    vol_window = vwap_window["Volume"]

    cum_vol = vol_window.sum()
    if cum_vol > 0:
        vwap = (tp_window * vol_window).sum() / cum_vol
        current_price = float(df["Close"].iloc[-1])
        vwap_dev = (current_price - vwap) / (vwap + 1e-10)
        vwap_score = np.clip(vwap_dev * 20, -1, 1)  # scale: 5% deviation → ±1
    else:
        vwap_score = 0.0
    scores["vwap_deviation"] = float(vwap_score)

    # ── 5. Momentum acceleration ─────────────────────────────────────
    # Rate of change of momentum — is momentum speeding up or slowing down?
    mom_series = returns.rolling(MOMENTUM_SHORT).mean()
    mom_diff = mom_series.diff()
    accel = mom_diff.iloc[-1] if not np.isnan(mom_diff.iloc[-1]) else 0
    accel_score = np.clip(accel * 500, -1, 1)  # scale
    scores["acceleration"] = float(accel_score)

    # ── Composite score ──────────────────────────────────────────────
    # Weighted average of components
    weights = {
        "momentum_ratio": 0.30,
        "volume_surge": 0.20,
        "rsi_fast": 0.20,
        "vwap_deviation": 0.20,
        "acceleration": 0.10,
    }

    composite = sum(scores[k] * weights[k] for k in weights)
    composite = float(np.clip(composite, -1, 1))

    return {
        "momentum_score": round(composite, 4),
        "components": {k: round(v, 4) for k, v in scores.items()},
        "current_price": float(df["Close"].iloc[-1]),
        "bars_used": len(df),
    }


def scan_momentum(ticker: str, lookback_hours: int = LOOKBACK_HOURS) -> dict:
    """
    Fetch 1h data and compute momentum score for a crypto ticker.

    Parameters
    ----------
    ticker : str
        Polygon ticker (e.g. "X:BTCUSD")
    lookback_hours : int
        Hours of 1h data to use (default 72)

    Returns
    -------
    dict with momentum_score, components, current_price, ticker
    """
    try:
        from src.data_fetcher import fetch_btc_hourly

        # Fetch 1h bars (no resampling — we want raw hourly)
        # Request extra days to ensure we have enough after market gaps
        days_needed = max(7, (lookback_hours // 24) + 3)
        df = fetch_btc_hourly(days=days_needed, ticker=ticker)

        if df.empty:
            return {
                "ticker": ticker,
                "momentum_score": 0.0,
                "error": f"No data for {ticker}",
            }

        result = compute_momentum_score(df)
        result["ticker"] = ticker
        return result

    except Exception as e:
        return {
            "ticker": ticker,
            "momentum_score": 0.0,
            "error": str(e),
        }
