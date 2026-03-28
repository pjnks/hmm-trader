"""
indicators.py
─────────────
Computes all 8 technical indicators used by the strategy confirmation engine.
Each indicator function returns a pandas Series aligned to *df*.

Indicator list
──────────────
  1. RSI           – Relative Strength Index (14)
  2. Momentum      – Rate-of-Change / price momentum (10)
  3. Volatility    – Rolling historical volatility (20-period std of log returns)
  4. Volume        – Volume vs its 20-period MA
  5. ADX           – Average Directional Index (14)
  6. Price Trend   – Close vs 50-period SMA
  7. MACD          – MACD line vs Signal line (12/26/9)
  8. Stochastic    – %K vs upper overbought level (14/3)

All parameters are loaded from config.py so they can be tuned centrally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _smma(series: pd.Series, n: int) -> pd.Series:
    """Smoothed moving average (Wilder's)."""
    return series.ewm(alpha=1.0 / n, adjust=False).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 1. RSI
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsi(df: pd.DataFrame, period: int = config.RSI_PERIOD) -> pd.Series:
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = _smma(gain, period)
    avg_loss = _smma(loss, period)
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.rename("rsi")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Momentum (Rate of Change)
# ─────────────────────────────────────────────────────────────────────────────

def compute_momentum(df: pd.DataFrame,
                     period: int = config.MOMENTUM_PERIOD) -> pd.Series:
    roc = (df["Close"] / df["Close"].shift(period) - 1) * 100
    return roc.rename("momentum")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Volatility (annualised rolling std of log returns)
# ─────────────────────────────────────────────────────────────────────────────

def compute_volatility(df: pd.DataFrame,
                       period: int = config.VOLATILITY_PERIOD) -> pd.Series:
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol = log_ret.rolling(period).std() * np.sqrt(24 * 365)  # annualised hourly
    return vol.rename("volatility")


def compute_volatility_median(df: pd.DataFrame,
                               period: int = config.VOLATILITY_PERIOD) -> pd.Series:
    """Rolling median of volatility – used to define 'normal' vol level."""
    vol = compute_volatility(df, period)
    return vol.rolling(period * 5).median().rename("vol_median")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Volume Ratio
# ─────────────────────────────────────────────────────────────────────────────

def compute_volume_ratio(df: pd.DataFrame,
                         period: int = config.VOLUME_MA_PERIOD) -> pd.Series:
    vol_ma = df["Volume"].rolling(period).mean()
    ratio  = df["Volume"] / vol_ma.replace(0, np.nan)
    return ratio.rename("volume_ratio")


# ─────────────────────────────────────────────────────────────────────────────
# 5. ADX (Average Directional Index – Wilder)
# ─────────────────────────────────────────────────────────────────────────────

def compute_adx(df: pd.DataFrame, period: int = config.ADX_PERIOD) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]

    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    plus_dm  = np.where((high - prev_high) > (prev_low - low),
                        np.maximum(high - prev_high, 0), 0.0)
    minus_dm = np.where((prev_low - low) > (high - prev_high),
                        np.maximum(prev_low - low, 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr     = pd.Series(_smma(tr, period), index=df.index)
    pdm_s   = pd.Series(_smma(pd.Series(plus_dm,  index=df.index), period), index=df.index)
    mdm_s   = pd.Series(_smma(pd.Series(minus_dm, index=df.index), period), index=df.index)

    pdi = 100 * pdm_s / atr.replace(0, np.nan)
    mdi = 100 * mdm_s / atr.replace(0, np.nan)
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = _smma(dx, period)
    return adx.rename("adx")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Price Trend (Close vs SMA)
# ─────────────────────────────────────────────────────────────────────────────

def compute_price_trend(df: pd.DataFrame,
                        period: int = config.TREND_MA_PERIOD) -> pd.Series:
    sma   = df["Close"].rolling(period).mean()
    trend = (df["Close"] - sma) / sma * 100   # % above / below MA
    return trend.rename("price_trend_pct")


def compute_sma(df: pd.DataFrame,
                period: int = config.TREND_MA_PERIOD) -> pd.Series:
    return df["Close"].rolling(period).mean().rename(f"sma_{period}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MACD
# ─────────────────────────────────────────────────────────────────────────────

def compute_macd(
    df: pd.DataFrame,
    fast:   int = config.MACD_FAST,
    slow:   int = config.MACD_SLOW,
    signal: int = config.MACD_SIGNAL,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast   = _ema(df["Close"], fast)
    ema_slow   = _ema(df["Close"], slow)
    macd_line  = (ema_fast - ema_slow).rename("macd")
    sig_line   = _ema(macd_line, signal).rename("macd_signal")
    histogram  = (macd_line - sig_line).rename("macd_hist")
    return macd_line, sig_line, histogram


# ─────────────────────────────────────────────────────────────────────────────
# 8. Stochastic %K / %D
# ─────────────────────────────────────────────────────────────────────────────

def compute_stochastic(
    df: pd.DataFrame,
    k_period: int = config.STOCH_K,
    d_period: int = config.STOCH_D,
) -> tuple[pd.Series, pd.Series]:
    """Returns (%K, %D)."""
    low_k  = df["Low"].rolling(k_period).min()
    high_k = df["High"].rolling(k_period).max()
    pct_k  = 100 * (df["Close"] - low_k) / (high_k - low_k).replace(0, np.nan)
    pct_d  = pct_k.rolling(d_period).mean()
    return pct_k.rename("stoch_k"), pct_d.rename("stoch_d")


# ─────────────────────────────────────────────────────────────────────────────
# Extended HMM features (used by feature_set = "extended" / "full")
# ─────────────────────────────────────────────────────────────────────────────

def compute_realized_vol_ratio(df: pd.DataFrame,
                                short: int = 2,
                                long:  int = 24) -> pd.Series:
    """Short-window vol / long-window vol — regime volatility regime indicator."""
    log_ret   = np.log(df["Close"] / df["Close"].shift(1))
    short_vol = log_ret.rolling(short).std()
    long_vol  = log_ret.rolling(long).std()
    ratio     = short_vol / long_vol.replace(0, np.nan)
    lo, hi    = ratio.quantile(0.01), ratio.quantile(0.99)
    return ratio.clip(lo, hi).rename("realized_vol_ratio")


def compute_return_autocorr(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """Lag-1 autocorrelation of log returns over a rolling window."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))

    def _autocorr(x: np.ndarray) -> float:
        if len(x) < 4:
            return 0.0
        a, b = x[:-1], x[1:]
        sa, sb = a.std(), b.std()
        if sa < 1e-10 or sb < 1e-10:
            return 0.0
        c = np.corrcoef(a, b)[0, 1]
        return float(c) if not np.isnan(c) else 0.0

    return log_ret.rolling(window).apply(_autocorr, raw=True).rename("return_autocorr")


def compute_vol_price_diverge(df: pd.DataFrame, window: int = 6) -> pd.Series:
    """1 when 6-bar price and volume directions disagree, 0 otherwise.
    WARNING: Binary {0,1} — do NOT use in HMM feature sets. Kept for legacy compat."""
    price_dir = np.sign(df["Close"].diff(window))
    vol_dir   = np.sign(df["Volume"].diff(window))
    return (price_dir != vol_dir).astype(float).rename("vol_price_diverge")


# ─────────────────────────────────────────────────────────────────────────────
# NEW continuous replacements for vol_price_diverge (Sprint 3.1)
# ─────────────────────────────────────────────────────────────────────────────

def compute_realized_kurtosis(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling kurtosis of log returns — tail-risk / fat-tail signal.
    Gaussian = 3.0; fat tails > 3; thin tails < 3.
    Winsorised 1-99% for stability."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    kurt = log_ret.rolling(window).kurt()  # excess kurtosis (0 for Gaussian)
    lo, hi = kurt.quantile(0.01), kurt.quantile(0.99)
    return kurt.clip(lo, hi).rename("realized_kurtosis")


def compute_volume_return_intensity(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Rolling mean of |log_return × volume_change| — captures conviction.
    High values = large moves with volume confirmation.
    Winsorised 1-99% for stability."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    vol_chg = df["Volume"] / df["Volume"].shift(1) - 1
    intensity = (log_ret * vol_chg).abs()
    rolling_mean = intensity.rolling(window).mean()
    lo, hi = rolling_mean.quantile(0.01), rolling_mean.quantile(0.99)
    return rolling_mean.clip(lo, hi).rename("volume_return_intensity")


def compute_return_momentum_ratio(df: pd.DataFrame,
                                   short: int = 5,
                                   long: int = 20) -> pd.Series:
    """Ratio of short-term to long-term cumulative return — momentum regime signal.
    >1 = accelerating, <1 = decelerating, ~1 = steady.
    Winsorised 1-99% for stability."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    short_ret = log_ret.rolling(short).sum()
    long_ret = log_ret.rolling(long).sum()
    # Use ratio of absolute magnitudes with sign preserved
    # Avoid division by zero
    ratio = short_ret / long_ret.replace(0, np.nan)
    lo, hi = ratio.quantile(0.01), ratio.quantile(0.99)
    return ratio.clip(lo, hi).rename("return_momentum_ratio")


def compute_candle_body_ratio(df: pd.DataFrame) -> pd.Series:
    """abs(close-open) / (high-low); zero-range bars filled with 0.5."""
    body  = (df["Close"] - df["Open"]).abs()
    range_ = (df["High"] - df["Low"]).replace(0, np.nan)
    return (body / range_).fillna(0.5).rename("candle_body_ratio")


def compute_bb_width(df: pd.DataFrame,
                     period: int = 20,
                     n_std:  int = 2) -> pd.Series:
    """Bollinger Band width (upper-lower) / mid-band, normalised by price."""
    sma   = df["Close"].rolling(period).mean()
    std   = df["Close"].rolling(period).std()
    width = (2 * n_std * std) / sma.replace(0, np.nan)
    return width.rename("bb_width")


# ─────────────────────────────────────────────────────────────────────────────
# Composite: attach all indicators to the DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def attach_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute every indicator and add as new columns.  Returns an augmented copy.
    """
    df = df.copy()

    df["rsi"]              = compute_rsi(df)
    df["momentum"]         = compute_momentum(df)
    df["volatility"]       = compute_volatility(df)
    df["vol_median"]       = compute_volatility_median(df)
    df["volume_ratio"]     = compute_volume_ratio(df)
    df["adx"]              = compute_adx(df)
    df["price_trend_pct"]  = compute_price_trend(df)
    df[f"sma_{config.TREND_MA_PERIOD}"] = compute_sma(df)

    macd, sig, hist        = compute_macd(df)
    df["macd"]             = macd
    df["macd_signal"]      = sig
    df["macd_hist"]        = hist

    stoch_k, stoch_d       = compute_stochastic(df)
    df["stoch_k"]          = stoch_k
    df["stoch_d"]          = stoch_d

    # Extended features (always computed; HMM uses only those in FEATURE_SETS)
    df["realized_vol_ratio"] = compute_realized_vol_ratio(df)
    df["return_autocorr"]    = compute_return_autocorr(df)
    df["vol_price_diverge"]  = compute_vol_price_diverge(df)  # legacy, NOT used in HMM
    df["candle_body_ratio"]  = compute_candle_body_ratio(df)
    df["bb_width"]           = compute_bb_width(df)

    # Sprint 3.1: new continuous features (replacements for vol_price_diverge)
    df["realized_kurtosis"]       = compute_realized_kurtosis(df)
    df["volume_return_intensity"] = compute_volume_return_intensity(df)
    df["return_momentum_ratio"]   = compute_return_momentum_ratio(df)

    return df
