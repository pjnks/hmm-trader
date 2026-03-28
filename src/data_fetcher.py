"""
data_fetcher.py
───────────────
Pulls hourly BTC/USD OHLCV data from the Polygon.io REST API (crypto endpoint)
and builds the three HMM features:
  • log_return      – 1-period log return
  • price_range     – (High − Low) / Close   (normalised candle range)
  • volume_change   – 1-period log change in volume

API key is loaded from the POLYGON_API_KEY environment variable (via .env).

Polygon endpoint used
─────────────────────
  GET /v2/aggs/ticker/{ticker}/range/1/hour/{from_ms}/{to_ms}
      ?adjusted=true&sort=asc&limit=5000&apiKey={key}

  IMPORTANT: from/to must be Unix millisecond timestamps, NOT date strings.
  Date strings (YYYY-MM-DD) are unreliable for crypto tickers on this endpoint
  — Polygon snaps them to exchange-local day boundaries and often returns only
  the final bar of the requested range.

Response fields mapped to DataFrame columns
───────────────────────────────────────────
  t → UTC datetime index (millisecond Unix timestamp)
  o → Open
  h → High
  l → Low
  c → Close
  v → Volume
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger(__name__)

# Load .env from the project root (one level above src/)
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

POLYGON_BASE = "https://api.polygon.io"

# ── Local data cache ──────────────────────────────────────────────────────────
_ROOT      = Path(__file__).parent.parent
_CACHE_DIR = _ROOT / "data_cache"


def _cache_file(ticker: str) -> Path:
    """
    Return the cache file path for *ticker*.

    Examples
    --------
    "X:BTCUSD"  →  data_cache/btcusd_hourly.csv
    "X:ETHUSD"  →  data_cache/ethusd_hourly.csv
    """
    name = ticker.replace("X:", "").replace(":", "").lower()
    return _CACHE_DIR / f"{name}_hourly.csv"


# ─────────────────────────────────────────────────────────────────────────────

def _load_cache(ticker: str) -> pd.DataFrame | None:
    """
    Load the local CSV cache for *ticker* if it exists.

    Returns a UTC-indexed OHLCV DataFrame, or None if the cache is missing.
    """
    path = _cache_file(ticker)
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "Datetime"
    df.sort_index(inplace=True)
    log.info(
        "Cache loaded [%s]: %d bars  (%s → %s)",
        ticker, len(df),
        df.index[0].strftime("%Y-%m-%d %H:%M UTC"),
        df.index[-1].strftime("%Y-%m-%d %H:%M UTC"),
    )
    return df


def _save_cache(df: pd.DataFrame, ticker: str) -> None:
    """Persist *df* to the local CSV cache for *ticker*."""
    path = _cache_file(ticker)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    log.info("Cache saved → %s  (%d bars)", path, len(df))


def _get_api_key() -> str:
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key or key == "your_polygon_api_key_here":
        raise RuntimeError(
            "POLYGON_API_KEY is not set. "
            "Add it to the .env file in the project root:\n"
            "  POLYGON_API_KEY=your_actual_key_here"
        )
    return key


_RETRY_WAIT_S  = 60   # seconds to wait after a 429
_MAX_RETRIES   = 5    # maximum number of retry attempts per page
_PAGE_SLEEP_S  = 2.0  # mandatory sleep between every page request


def _fetch_page(url: str, params: dict, api_key: str) -> dict:
    """
    GET a single Polygon API page with retry logic for 429 rate-limit errors.

    On a 429 response the request is retried up to _MAX_RETRIES times,
    waiting _RETRY_WAIT_S seconds between each attempt.  All other HTTP
    errors are raised immediately.
    """
    params = {**params, "apiKey": api_key}

    for attempt in range(1, _MAX_RETRIES + 2):   # +2 so last attempt == _MAX_RETRIES+1
        resp = requests.get(url, params=params, timeout=30)

        if resp.status_code == 429:
            if attempt > _MAX_RETRIES:
                raise RuntimeError(
                    f"Polygon rate-limit (429) persisted after {_MAX_RETRIES} "
                    f"retries.  Consider reducing PAGE_LIMIT or adding a longer "
                    f"delay between requests."
                )
            log.warning(
                "Polygon 429 Too Many Requests — waiting %ds before retry %d/%d …",
                _RETRY_WAIT_S, attempt, _MAX_RETRIES,
            )
            time.sleep(_RETRY_WAIT_S)
            continue   # retry

        resp.raise_for_status()   # surface any other 4xx/5xx immediately
        break

    data = resp.json()
    status = data.get("status", "")
    if status not in ("OK", "DELAYED"):
        raise RuntimeError(
            f"Polygon API returned status={status!r}. "
            f"message={data.get('message') or data.get('error', '(none)')}"
        )
    return data


# ─────────────────────────────────────────────────────────────────────────────

def fetch_btc_hourly(
    days:   int = config.DAYS_HISTORY,
    ticker: str = config.TICKER,
) -> pd.DataFrame:
    """
    Download hourly OHLCV bars for *ticker* (Polygon crypto format, e.g.
    ``"X:BTCUSD"``) covering the last *days* calendar days.

    Cache behaviour
    ───────────────
    • On first run the full history is fetched and saved to
      ``data_cache/btcusd_hourly.csv``.
    • On subsequent runs the cache is loaded and only the bars that
      are missing (from the last cached timestamp to now) are fetched
      from Polygon.  The two datasets are merged and the cache is
      updated after every page so an interrupted fetch can be resumed.

    Returns a DataFrame indexed by UTC-aware datetime with columns:
        Open, High, Low, Close, Volume
    """
    api_key = _get_api_key()

    end = datetime.now(tz=timezone.utc)
    target_start = end - timedelta(days=days)

    # ── Load cache and determine fetch window ─────────────────────────────
    cached_df = _load_cache(ticker)

    if cached_df is not None and not cached_df.empty:
        last_cached_ts = cached_df.index[-1]   # tz-aware UTC

        if last_cached_ts >= end - timedelta(hours=2):
            # Cache is fresh enough — no API call needed
            log.info("Cache is up to date (last bar: %s). Skipping fetch.",
                     last_cached_ts.strftime("%Y-%m-%d %H:%M UTC"))
            # Still trim to the requested window before returning
            return cached_df[cached_df.index >= target_start].copy()

        # Resume from one millisecond after the last cached bar
        fetch_start = last_cached_ts + timedelta(milliseconds=1)
        log.info(
            "Resuming fetch from last cached bar: %s",
            last_cached_ts.strftime("%Y-%m-%d %H:%M UTC"),
        )
    else:
        cached_df   = None
        fetch_start = target_start

    from_ms = int(fetch_start.timestamp() * 1000)
    to_ms   = int(end.timestamp()         * 1000)

    log.info(
        "Fetching %s  %s → %s  (interval=1h, source=Polygon)",
        ticker,
        fetch_start.strftime("%Y-%m-%d %H:%M UTC"),
        end.strftime("%Y-%m-%d %H:%M UTC"),
    )
    log.info("Timestamp range: from_ms=%d  to_ms=%d", from_ms, to_ms)

    url = (
        f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}"
        f"/range/1/hour/{from_ms}/{to_ms}"
    )
    params: dict = {
        "adjusted": "true",
        "sort":     "asc",
        "limit":    5000,    # conservative page size to stay within rate limits
    }

    new_bars: list[dict] = []
    page = 0

    while True:
        page += 1
        data    = _fetch_page(url, params, api_key)
        results = data.get("results") or []
        new_bars.extend(results)

        log.info(
            "  page %d: received %d bars  (new bars so far: %d)",
            page, len(results), len(new_bars),
        )

        # Merge and save after every page so progress is never lost
        if new_bars:
            page_df   = _results_to_dataframe(new_bars)
            merged_df = (
                pd.concat([cached_df, page_df])
                .sort_index()
                # Drop any duplicate timestamps that span the cache boundary
                [lambda d: ~d.index.duplicated(keep="last")]
                if cached_df is not None
                else page_df
            )
            _save_cache(merged_df, ticker)

        # Polygon paginates via next_url when result count exceeds limit
        next_url = data.get("next_url")
        if not next_url:
            log.info("  pagination complete — no next_url returned")
            break

        # next_url already contains all query params except apiKey
        url    = next_url
        params = {}   # clear — next_url carries its own params

        # Mandatory sleep between every page to avoid hitting rate limits
        log.info("  sleeping %.1fs before next page …", _PAGE_SLEEP_S)
        time.sleep(_PAGE_SLEEP_S)

    # ── Build final DataFrame from cache (includes both old and new bars) ──
    final_df = _load_cache(ticker)
    if final_df is None or final_df.empty:
        raise RuntimeError(
            f"Polygon returned no data for {ticker} "
            f"(from_ms={from_ms}, to_ms={to_ms}). "
            "Check your API key permissions and the ticker symbol."
        )

    # Trim to the originally requested window
    final_df = final_df[final_df.index >= target_start].copy()

    log.info(
        "Done. %d hourly bars  (%s → %s)",
        len(final_df),
        final_df.index[0].strftime("%Y-%m-%d %H:%M UTC"),
        final_df.index[-1].strftime("%Y-%m-%d %H:%M UTC"),
    )
    return final_df


# ─────────────────────────────────────────────────────────────────────────────

def fetch_test_7d(ticker: str = config.TICKER) -> pd.DataFrame:
    """
    Smoke-test fetch: pull the last 7 days of hourly bars and print a summary.
    Run this first to confirm the API key, ticker symbol, and timestamp format
    are all working before attempting the full 730-day pull.

    Usage:
        python -c "from src.data_fetcher import fetch_test_7d; fetch_test_7d()"
    """
    log.info("=== 7-day smoke test ===")
    df = fetch_btc_hourly(days=7, ticker=ticker)
    expected_min = 7 * 20          # at least 20 h/day accounting for gaps
    expected_max = 7 * 24 + 5      # at most 24 h/day + small buffer

    print(f"\n  Ticker  : {ticker}")
    print(f"  Bars    : {len(df)}  (expected ~{7*24})")
    print(f"  From    : {df.index[0]}")
    print(f"  To      : {df.index[-1]}")
    print(f"  Columns : {list(df.columns)}")
    print(f"\n{df.tail(3).to_string()}\n")

    if len(df) < expected_min:
        print(f"  WARNING: only {len(df)} bars returned — expected ≥ {expected_min}.")
        print("  The timestamp format or ticker symbol may still be wrong.")
    elif len(df) > expected_max:
        print(f"  WARNING: {len(df)} bars returned — more than expected ({expected_max}).")
    else:
        print(f"  OK — bar count looks correct. Safe to run full 730-day fetch.")

    return df


# ─────────────────────────────────────────────────────────────────────────────

def _results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert the Polygon ``results`` list to a clean OHLCV DataFrame."""
    df = pd.DataFrame(results)

    # Polygon timestamp is Unix milliseconds → UTC datetime
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    df.index.name = "Datetime"

    df = df.rename(columns={
        "o": "Open",
        "h": "High",
        "l": "Low",
        "c": "Close",
        "v": "Volume",
    })

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
    df.sort_index(inplace=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────

def build_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the three features the HMM is trained on and attach them to *df*.

    Features are winsorised at the 1st / 99th percentile to avoid extreme
    outliers dominating the Gaussian fit.

    Returns df with new columns:
        log_return, price_range, volume_change
    """
    df = df.copy()

    # 1. 1-period log return
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # 2. Normalised high-low range
    df["price_range"] = (df["High"] - df["Low"]) / df["Close"]

    # 3. 1-period log volume change (guard against zero volume)
    safe_vol = df["Volume"].replace(0, np.nan).ffill()
    df["volume_change"] = np.log(safe_vol / safe_vol.shift(1))

    df.dropna(subset=["log_return", "price_range", "volume_change"], inplace=True)

    # Winsorise
    for col in ["log_return", "price_range", "volume_change"]:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    return df


# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Convenience function: fetch raw OHLCV + build HMM features in one call.
    Returns the combined DataFrame ready for model training and backtesting.
    """
    df = fetch_btc_hourly()
    df = build_hmm_features(df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Multi-timeframe support (used by optimizer)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_timeframe(tf: str) -> tuple[int, str]:
    """
    Convert a timeframe string to Polygon API (multiplier, timespan) pair.

    Examples
    --------
    "1h"  → (1, "hour")
    "4h"  → (4, "hour")
    "1d"  → (1, "day")
    """
    import re
    m = re.fullmatch(r"(\d+)([hdwm])", tf.strip().lower())
    if not m:
        raise ValueError(
            f"Unrecognised timeframe {tf!r}. Expected format: '1h', '4h', '1d'."
        )
    n    = int(m.group(1))
    unit = m.group(2)
    span = {"h": "hour", "d": "day", "w": "week", "m": "month"}[unit]
    return n, span


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample a 1h OHLCV DataFrame to a lower frequency.

    Parameters
    ----------
    df        : hourly OHLCV DataFrame with UTC-aware DatetimeIndex
    timeframe : target timeframe string, e.g. "4h", "1d"
    """
    multiplier, timespan = _parse_timeframe(timeframe)
    if timespan == "hour":
        rule = f"{multiplier}h"
    elif timespan == "day":
        rule = f"{multiplier}D"
    else:
        rule = f"{multiplier}W"

    resampled = df.resample(rule).agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna()
    return resampled


def load_data_for_optimizer(
    timeframe:     str = "1h",
    training_days: int = 730,
    feature_set:   str = "base",
    ticker:        str = config.TICKER,
) -> pd.DataFrame:
    """
    Load, optionally resample, slice to *training_days*, and attach all HMM
    features ready for the optimizer's train/test split.

    Always uses the cached 1h data as the source; resamples on the fly for
    non-hourly timeframes so no extra API calls are needed.  Separate cache
    files are used per ticker (e.g. btcusd_hourly.csv, ethusd_hourly.csv).
    """
    # Load 1h base for the requested ticker
    df_1h = fetch_btc_hourly(days=max(training_days + 30, 730), ticker=ticker)

    # Resample if needed
    if timeframe != "1h":
        df_raw = resample_ohlcv(df_1h, timeframe)
    else:
        df_raw = df_1h.copy()

    # Slice to training window
    from datetime import datetime, timezone, timedelta
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=training_days)
    df_raw = df_raw[df_raw.index >= cutoff].copy()

    # Base HMM features (log_return, price_range, volume_change)
    df = build_hmm_features(df_raw)

    # Extended features (volume_focused only needs realized_vol_ratio)
    if feature_set in ("extended", "full", "volume_focused"):
        import importlib
        ind = importlib.import_module("src.indicators")
        df["realized_vol_ratio"] = ind.compute_realized_vol_ratio(df)
    if feature_set in ("extended", "full"):
        df["return_autocorr"]   = ind.compute_return_autocorr(df)
        df["vol_price_diverge"] = ind.compute_vol_price_diverge(df)
        if feature_set == "full":
            df["candle_body_ratio"] = ind.compute_candle_body_ratio(df)
            df["bb_width"]          = ind.compute_bb_width(df)

    feature_cols = config.FEATURE_SETS[feature_set]
    df.dropna(subset=feature_cols, inplace=True)
    return df


# ── Real-time price snapshot ───────────────────────────────────────
def fetch_latest_price(ticker: str) -> float | None:
    """Fetch latest price for an equity ticker via Polygon Snapshot API.

    Uses /v2/snapshot/locale/us/markets/stocks/tickers/{ticker} — single
    API call, returns last trade price or previous day close.
    Much faster than fetching daily bars for intraday risk checks.

    Returns None on any error (caller should fall back to entry_price).
    """
    api_key = _get_api_key()
    url = (f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks"
           f"/tickers/{ticker.upper()}")
    for attempt in range(2):
        try:
            resp = requests.get(url, params={"apiKey": api_key}, timeout=10)
            if resp.status_code == 429:
                time.sleep(12)
                continue
            if resp.status_code == 403:
                # Snapshot API requires paid plan — fall back to prev close
                return _fetch_prev_close(ticker)
            resp.raise_for_status()
            data = resp.json()
            snap = data.get("ticker", {})
            # Prefer last trade price
            last_trade = snap.get("lastTrade", {})
            if last_trade and last_trade.get("p", 0) > 0:
                return float(last_trade["p"])
            # Fall back to day close
            day = snap.get("day", {})
            if day and day.get("c", 0) > 0:
                return float(day["c"])
            # Fall back to previous day close
            prev = snap.get("prevDay", {})
            if prev and prev.get("c", 0) > 0:
                return float(prev["c"])
            return None
        except Exception as e:
            log.warning(f"fetch_latest_price({ticker}) attempt {attempt+1}: {e}")
    return None


def _fetch_prev_close(ticker: str) -> float | None:
    """Fetch previous day close via /v2/aggs/ticker/{ticker}/prev endpoint."""
    api_key = _get_api_key()
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/prev"
    try:
        resp = requests.get(url, params={"adjusted": "true", "apiKey": api_key},
                            timeout=10)
        if resp.status_code == 429:
            return None
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if results and results[0].get("c", 0) > 0:
            return float(results[0]["c"])
        return None
    except Exception:
        return None
