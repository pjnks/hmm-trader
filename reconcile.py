"""
reconcile.py
────────────
Backtest-vs-Live signal reconciliation.

Compares live trading signals (from status JSON files and trade DBs)
against what the backtester would produce for the same dates and config.
Detects signal drift from code differences.

Usage:
    python reconcile.py                      # reconcile all projects
    python reconcile.py --project agate      # AGATE only
    python reconcile.py --project beryl      # BERYL only
    python reconcile.py --project citrine    # CITRINE only
    python reconcile.py --verbose            # show per-bar details
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.data_fetcher import fetch_btc_hourly, resample_ohlcv, build_hmm_features
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reconcile")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_status_file(filename: str) -> dict | None:
    """Load a status JSON file."""
    path = ROOT / filename
    if not path.exists():
        log.warning(f"Status file not found: {path}")
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        log.error(f"Error reading {path}: {e}")
        return None


def load_trades_db(db_name: str) -> pd.DataFrame:
    """Load all trades from a SQLite database."""
    db_path = ROOT / db_name
    if not db_path.exists():
        log.warning(f"Database not found: {db_path}")
        return pd.DataFrame()
    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp", conn)
    except Exception as e:
        log.error(f"Error reading {db_path}: {e}")
        return pd.DataFrame()


def compute_features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Build HMM features + extended features for a given feature set.

    Supports all feature sets: base, extended, extended_v2, full, full_v2.
    Uses config.FEATURE_SETS to determine which columns are needed.
    """
    df = build_hmm_features(df)

    needed = set(config.FEATURE_SETS.get(feature_set, []))
    base_cols = {"log_return", "price_range", "volume_change"}
    extra = needed - base_cols

    if "realized_vol_ratio" in extra:
        df["realized_vol_ratio"] = compute_realized_vol_ratio(df)
    if "return_autocorr" in extra:
        df["return_autocorr"] = compute_return_autocorr(df)
    if "realized_kurtosis" in extra:
        df["realized_kurtosis"] = compute_realized_kurtosis(df)
    if "candle_body_ratio" in extra:
        df["candle_body_ratio"] = compute_candle_body_ratio(df)
    if "bb_width" in extra:
        df["bb_width"] = compute_bb_width(df)
    if "volume_return_intensity" in extra:
        df["volume_return_intensity"] = compute_volume_return_intensity(df)
    if "return_momentum_ratio" in extra:
        df["return_momentum_ratio"] = compute_return_momentum_ratio(df)

    df = df.dropna()
    return df


def run_backtester_signal(
    df: pd.DataFrame,
    feature_set: str,
    use_ensemble: bool = True,
) -> pd.DataFrame:
    """
    Run full HMM + indicator + signal pipeline on prepared data.
    Returns DataFrame with regime_cat, confidence, confirmation_count, raw_long_signal.
    """
    feature_cols = config.FEATURE_SETS[feature_set]

    if use_ensemble:
        model = EnsembleHMM(
            cov_type=config.COV_TYPE,
            feature_cols=feature_cols,
        )
    else:
        model = HMMRegimeModel(
            n_states=config.N_STATES,
            cov_type=config.COV_TYPE,
            feature_cols=feature_cols,
        )

    model.fit(df)
    if not model.converged:
        log.warning("HMM did not converge during reconciliation")
        return pd.DataFrame()

    df = model.predict(df)
    df = attach_all(df)
    df = build_signal_series(df, use_regime_mapper=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# AGATE RECONCILIATION
# ═══════════════════════════════════════════════════════════════════════════════

def reconcile_agate(verbose: bool = False) -> dict:
    """
    Reconcile AGATE live status vs backtester output.

    Compares:
    1. Current regime (live status vs fresh HMM fit)
    2. Current signal (live vs backtester)
    3. Confidence level
    4. Confirmation count
    """
    log.info("=" * 60)
    log.info("AGATE RECONCILIATION (SOL/4h Ensemble)")
    log.info("=" * 60)

    results = {
        "project": "AGATE",
        "status": "OK",
        "discrepancies": [],
        "live_signal": None,
        "backtest_signal": None,
    }

    # ── Load live status ──────────────────────────────────────────
    status = load_status_file("agate_status.json")
    if status is None:
        results["status"] = "NO_STATUS_FILE"
        log.error("No agate_status.json found — is AGATE running?")
        return results

    live_regime = status.get("regime", "UNKNOWN")
    live_confidence = status.get("confidence", 0.0)
    live_signal = status.get("signal", "HOLD")
    live_confirmations = status.get("confirmations", 0)
    live_price = status.get("current_price", 0.0)
    live_ensemble = status.get("use_ensemble", False)

    log.info(f"Live status:")
    log.info(f"  Regime: {live_regime} (conf {live_confidence:.3f})")
    log.info(f"  Signal: {live_signal} | {live_confirmations}/8 confirmations")
    log.info(f"  Price: ${live_price:.2f}")
    log.info(f"  Ensemble: {live_ensemble}")

    results["live_signal"] = {
        "regime": live_regime,
        "confidence": live_confidence,
        "signal": live_signal,
        "confirmations": live_confirmations,
        "price": live_price,
    }

    # ── Fetch same data and run backtester ────────────────────────
    log.info("Fetching SOL data for backtester comparison...")
    try:
        df_raw = fetch_btc_hourly(days=90, ticker=config.TICKER)
        if df_raw.empty:
            results["status"] = "NO_DATA"
            log.error("No data fetched from Polygon")
            return results

        if config.TIMEFRAME != "1h":
            df_raw = resample_ohlcv(df_raw, config.TIMEFRAME)

        df = compute_features(df_raw, config.FEATURE_SET)

        if len(df) < 50:
            results["status"] = "INSUFFICIENT_DATA"
            log.error(f"Only {len(df)} bars after feature computation")
            return results

        df = run_backtester_signal(df, config.FEATURE_SET, use_ensemble=live_ensemble)
        if df.empty:
            results["status"] = "HMM_FAILED"
            log.error("HMM did not converge in backtester")
            return results

    except Exception as e:
        results["status"] = f"ERROR: {e}"
        log.error(f"Backtester error: {e}")
        return results

    # ── Compare latest bar ────────────────────────────────────────
    latest = df.iloc[-1]
    bt_regime = latest.get("regime_cat", "UNKNOWN")
    bt_confidence = float(latest.get("confidence", 0.0))
    bt_confirmations = int(latest.get("confirmation_count", 0))
    bt_price = float(latest["Close"])

    # Determine backtest signal
    bt_signal = "HOLD"
    if bt_regime == "BULL" and bt_confidence >= config.REGIME_CONFIDENCE_MIN and bt_confirmations >= config.MIN_CONFIRMATIONS:
        bt_signal = "BUY"
    elif bt_regime == "BEAR":
        bt_signal = "SELL"

    log.info(f"\nBacktester output:")
    log.info(f"  Regime: {bt_regime} (conf {bt_confidence:.3f})")
    log.info(f"  Signal: {bt_signal} | {bt_confirmations}/8 confirmations")
    log.info(f"  Price: ${bt_price:.2f}")

    results["backtest_signal"] = {
        "regime": bt_regime,
        "confidence": bt_confidence,
        "signal": bt_signal,
        "confirmations": bt_confirmations,
        "price": bt_price,
    }

    # ── Check for discrepancies ───────────────────────────────────
    if live_regime != bt_regime:
        disc = f"REGIME MISMATCH: live={live_regime} vs backtest={bt_regime}"
        results["discrepancies"].append(disc)
        log.warning(f"  {disc}")

    if live_signal != bt_signal:
        disc = f"SIGNAL MISMATCH: live={live_signal} vs backtest={bt_signal}"
        results["discrepancies"].append(disc)
        log.warning(f"  {disc}")

    conf_diff = abs(live_confidence - bt_confidence)
    if conf_diff > 0.05:
        disc = f"CONFIDENCE DRIFT: live={live_confidence:.3f} vs backtest={bt_confidence:.3f} (diff={conf_diff:.3f})"
        results["discrepancies"].append(disc)
        log.warning(f"  {disc}")

    conf_diff_count = abs(live_confirmations - bt_confirmations)
    if conf_diff_count > 1:
        disc = f"CONFIRMATION MISMATCH: live={live_confirmations} vs backtest={bt_confirmations}"
        results["discrepancies"].append(disc)
        log.warning(f"  {disc}")

    price_diff_pct = abs(live_price - bt_price) / bt_price * 100 if bt_price > 0 else 0
    if price_diff_pct > 1.0:
        disc = f"PRICE DRIFT: live=${live_price:.2f} vs backtest=${bt_price:.2f} ({price_diff_pct:.1f}%)"
        results["discrepancies"].append(disc)
        log.warning(f"  {disc}")

    if not results["discrepancies"]:
        log.info("\n  All checks passed")
    else:
        results["status"] = "DISCREPANCIES_FOUND"
        log.warning(f"\n  {len(results['discrepancies'])} discrepancy(ies) found")

    # ── Load trade history ────────────────────────────────────────
    trades = load_trades_db("paper_trades.db")
    log.info(f"\nTrade history: {len(trades)} trades in paper_trades.db")
    if len(trades) > 0:
        total_pnl = trades["pnl"].sum()
        win_rate = (trades["pnl"] > 0).sum() / len(trades) * 100
        log.info(f"  Total P&L: ${total_pnl:.2f}")
        log.info(f"  Win rate: {win_rate:.1f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# BERYL RECONCILIATION
# ═══════════════════════════════════════════════════════════════════════════════

def reconcile_beryl(verbose: bool = False) -> dict:
    """Reconcile BERYL live status vs backtester output."""
    log.info("\n" + "=" * 60)
    log.info("BERYL RECONCILIATION (NVDA Daily Single-HMM)")
    log.info("=" * 60)

    results = {
        "project": "BERYL",
        "status": "OK",
        "discrepancies": [],
        "live_signal": None,
        "backtest_signal": None,
    }

    # ── Load live status ──────────────────────────────────────────
    status = load_status_file("beryl_status.json")
    if status is None:
        results["status"] = "NO_STATUS_FILE"
        log.error("No beryl_status.json found — is BERYL running?")
        return results

    live_regime = status.get("regime", "UNKNOWN")
    live_confidence = status.get("confidence", 0.0)
    live_signal = status.get("signal", "HOLD")
    live_confirmations = status.get("confirmations", 0)
    live_price = status.get("current_price", 0.0)
    beryl_config = status.get("config", {})

    log.info(f"Live status:")
    log.info(f"  Ticker: {status.get('ticker', 'UNKNOWN')}")
    log.info(f"  Regime: {live_regime} (conf {live_confidence:.3f})")
    log.info(f"  Signal: {live_signal} | {live_confirmations}/{beryl_config.get('confirmations', '?')} confirmations")
    log.info(f"  Price: ${live_price:.2f}")

    results["live_signal"] = {
        "regime": live_regime,
        "confidence": live_confidence,
        "signal": live_signal,
        "confirmations": live_confirmations,
        "price": live_price,
    }

    # ── Fetch NVDA data and run single-HMM backtester ────────────
    ticker = beryl_config.get("ticker", "NVDA")
    feature_set = beryl_config.get("feature_set", "extended")
    n_states = beryl_config.get("n_states", 4)
    cov_type = beryl_config.get("cov_type", "full")

    log.info(f"\nFetching {ticker} daily data for backtester comparison...")
    try:
        from walk_forward_ndx import fetch_equity_daily
        df_raw = fetch_equity_daily(ticker, years=1)
        if df_raw.empty:
            results["status"] = "NO_DATA"
            log.error(f"No data fetched for {ticker}")
            return results

        # Keep last 365 days (120 was too short — 4-state HMM needs more observations)
        cutoff = df_raw.index[-1] - pd.Timedelta(days=365)
        df_raw = df_raw[df_raw.index >= cutoff]

        df = compute_features(df_raw, feature_set)

        if len(df) < 30:
            results["status"] = "INSUFFICIENT_DATA"
            log.error(f"Only {len(df)} bars")
            return results

        # Use single HMM (BERYL doesn't use ensemble)
        feature_cols = config.FEATURE_SETS[feature_set]
        model = HMMRegimeModel(
            n_states=n_states,
            cov_type=cov_type,
            feature_cols=feature_cols,
        )
        model.fit(df)
        if not model.converged:
            results["status"] = "HMM_FAILED"
            log.error("HMM did not converge")
            return results

        df = model.predict(df)
        df = attach_all(df)
        df = build_signal_series(df, use_regime_mapper=False)

    except Exception as e:
        results["status"] = f"ERROR: {e}"
        log.error(f"Backtester error: {e}")
        return results

    # ── Compare latest bar ────────────────────────────────────────
    latest = df.iloc[-1]
    bt_regime = latest.get("regime_cat", "UNKNOWN")
    bt_confidence = float(latest.get("confidence", 0.0))
    bt_confirmations = int(latest.get("confirmation_count", 0))
    bt_price = float(latest["Close"])

    log.info(f"\nBacktester output:")
    log.info(f"  Regime: {bt_regime} (conf {bt_confidence:.3f})")
    log.info(f"  Confirmations: {bt_confirmations}/8")
    log.info(f"  Price: ${bt_price:.2f}")

    results["backtest_signal"] = {
        "regime": bt_regime,
        "confidence": bt_confidence,
        "confirmations": bt_confirmations,
        "price": bt_price,
    }

    # ── Check for discrepancies ───────────────────────────────────
    if live_regime != bt_regime:
        disc = f"REGIME MISMATCH: live={live_regime} vs backtest={bt_regime}"
        results["discrepancies"].append(disc)
        log.warning(f"  {disc}")

    conf_diff = abs(live_confidence - bt_confidence)
    if conf_diff > 0.1:
        disc = f"CONFIDENCE DRIFT: live={live_confidence:.3f} vs backtest={bt_confidence:.3f}"
        results["discrepancies"].append(disc)
        log.warning(f"  {disc}")

    if not results["discrepancies"]:
        log.info("\n  All checks passed")
    else:
        results["status"] = "DISCREPANCIES_FOUND"
        log.warning(f"\n  {len(results['discrepancies'])} discrepancy(ies) found")

    # ── Trade history ─────────────────────────────────────────────
    trades = load_trades_db("beryl_trades.db")
    log.info(f"\nTrade history: {len(trades)} trades in beryl_trades.db")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CITRINE RECONCILIATION
# ═══════════════════════════════════════════════════════════════════════════════

def reconcile_citrine(verbose: bool = False) -> dict:
    """Reconcile CITRINE live portfolio status vs expectations."""
    log.info("\n" + "=" * 60)
    log.info("CITRINE RECONCILIATION (NDX100 Portfolio Rotation)")
    log.info("=" * 60)

    results = {
        "project": "CITRINE",
        "status": "OK",
        "discrepancies": [],
        "trade_count": 0,
        "position_count": 0,
        "duplicate_trades": 0,
    }

    # ── Load trades ───────────────────────────────────────────────
    db_path = ROOT / "citrine_trades.db"
    if not db_path.exists():
        results["status"] = "NO_DATABASE"
        log.error("No citrine_trades.db found")
        return results

    with sqlite3.connect(db_path) as conn:
        trades = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp", conn)
        snapshots = pd.read_sql_query(
            "SELECT * FROM portfolio_snapshots ORDER BY id DESC LIMIT 5", conn,
        )

    results["trade_count"] = len(trades)
    log.info(f"Trade history: {len(trades)} trades")

    # ── Check for duplicate trades ────────────────────────────────
    if len(trades) > 0:
        # Group by ticker + action + timestamp (within 1 minute)
        entry_trades = trades[trades["action"] == "ENTRY"]
        if len(entry_trades) > 0:
            # Check for same ticker entered multiple times within 1 minute
            entry_trades = entry_trades.copy()
            entry_trades["ts"] = pd.to_datetime(entry_trades["timestamp"])
            entry_trades["ts_minute"] = entry_trades["ts"].dt.floor("min")
            dupes = entry_trades.groupby(["ticker", "ts_minute"]).size()
            dupe_count = (dupes > 1).sum()
            results["duplicate_trades"] = int(dupe_count)
            if dupe_count > 0:
                disc = f"DUPLICATE ENTRIES: {dupe_count} tickers with duplicate ENTRY trades within same minute"
                results["discrepancies"].append(disc)
                log.warning(f"  {disc}")

        # Trade P&L analysis
        exit_trades = trades[trades["action"] == "EXIT"]
        if len(exit_trades) > 0 and "pnl" in exit_trades.columns:
            total_pnl = exit_trades["pnl"].sum()
            win_count = (exit_trades["pnl"] > 0).sum()
            log.info(f"  Exits: {len(exit_trades)}, P&L: ${total_pnl:.2f}")
            log.info(f"  Win rate: {win_count}/{len(exit_trades)}")

    # ── Check latest snapshot ─────────────────────────────────────
    if len(snapshots) > 0:
        latest = snapshots.iloc[0]
        equity = latest.get("total_equity", 0)
        cash = latest.get("cash", 0)
        n_positions = latest.get("num_positions", 0)
        positions_json = latest.get("positions_json", "{}")

        results["position_count"] = int(n_positions)

        log.info(f"\nLatest snapshot:")
        log.info(f"  Equity: ${equity:,.2f}")
        log.info(f"  Cash: ${cash:,.2f} ({latest.get('cash_pct', 0):.1f}%)")
        log.info(f"  Positions: {n_positions}")

        # Parse positions
        try:
            positions = json.loads(positions_json) if positions_json else {}
            if isinstance(positions, dict):
                log.info(f"  Tickers: {', '.join(sorted(positions.keys())[:10])}")
                if len(positions) > 10:
                    log.info(f"  ... and {len(positions) - 10} more")
        except json.JSONDecodeError:
            results["discrepancies"].append("INVALID positions_json in snapshot")

        # Equity sanity check
        initial = config.CITRINE_INITIAL_CAPITAL
        equity_change_pct = ((equity - initial) / initial) * 100
        if abs(equity_change_pct) > 20:
            disc = f"LARGE EQUITY CHANGE: ${equity:,.2f} ({equity_change_pct:+.1f}% from ${initial:,})"
            results["discrepancies"].append(disc)
            log.warning(f"  {disc}")

    else:
        log.warning("No portfolio snapshots found")

    if not results["discrepancies"]:
        log.info("\n  All checks passed")
    else:
        results["status"] = "DISCREPANCIES_FOUND"

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(all_results: list[dict]) -> None:
    """Print a summary table of reconciliation results."""
    log.info("\n" + "=" * 60)
    log.info("RECONCILIATION SUMMARY")
    log.info("=" * 60)

    for r in all_results:
        project = r["project"]
        status = r["status"]
        n_disc = len(r.get("discrepancies", []))

        if status == "OK":
            icon = "  "
        elif status == "DISCREPANCIES_FOUND":
            icon = "  "
        else:
            icon = "  "

        log.info(f"{icon} {project:10s} | {status}")

        if n_disc > 0:
            for disc in r["discrepancies"]:
                log.info(f"     -> {disc}")

        # Show signal comparison if available
        live = r.get("live_signal")
        bt = r.get("backtest_signal")
        if live and bt:
            log.info(f"     Live:     {live.get('regime', '?')} (conf {live.get('confidence', 0):.3f}) → {live.get('signal', '?')}")
            log.info(f"     Backtest: {bt.get('regime', '?')} (conf {bt.get('confidence', 0):.3f}) → {bt.get('signal', '?')}")

    log.info("=" * 60)

    # Overall verdict
    all_ok = all(r["status"] == "OK" for r in all_results)
    if all_ok:
        log.info("VERDICT: All projects reconciled successfully")
    else:
        failed = [r["project"] for r in all_results if r["status"] != "OK"]
        log.warning(f"VERDICT: Issues found in {', '.join(failed)}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Backtest-vs-Live Reconciliation")
    parser.add_argument(
        "--project",
        choices=["agate", "beryl", "citrine", "all"],
        default="all",
        help="Which project to reconcile (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-bar details",
    )
    args = parser.parse_args()

    results = []

    if args.project in ("agate", "all"):
        results.append(reconcile_agate(verbose=args.verbose))

    if args.project in ("beryl", "all"):
        results.append(reconcile_beryl(verbose=args.verbose))

    if args.project in ("citrine", "all"):
        results.append(reconcile_citrine(verbose=args.verbose))

    print_summary(results)

    # Save results to JSON
    output_path = ROOT / "reconciliation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
