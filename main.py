"""
main.py
───────
HMM Regime Trading Terminal — entry point.

Usage
─────
  python main.py                      # fetch data, train, backtest (long-only), dashboard
  python main.py --no-dash            # print results only
  python main.py --retrain            # force model re-training (ignore cached model)
  python main.py --regime-mapper      # enable multi-direction trading (LONG/SHORT/FLAT)

Pipeline
────────
  1. Fetch 730 days of hourly crypto data from Polygon.io
  2. Build HMM features
  3. Train N-state Gaussian HMM (or load cached if available)
  4. Predict regimes for every bar
  5. Attach all technical indicators
  6. Build signal confirmation masks
  7. Run bar-by-bar backtest
  8. Print performance summary
  9. Launch Dash dashboard on http://127.0.0.1:8050
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hmm_trader")

# ── Path bootstrap ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.data_fetcher import load_data
from src.hmm_model import HMMRegimeModel
from src.indicators import attach_all
from src.strategy import build_signal_series
from src.backtester import Backtester

MODEL_CACHE    = ROOT / "model_cache.pkl"
ENSEMBLE_CACHE = ROOT / "ensemble_cache.pkl"


# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    retrain: bool = False,
    use_regime_mapper: bool = False,
    use_ensemble: bool = False,
) -> "BacktestResult":  # noqa: F821
    """Execute the full data → model → backtest pipeline."""

    # 1. Data
    log.info("═══ STEP 1/6  Fetching data ════════════════════════════════════")
    df = load_data()
    log.info("Data shape: %s  |  date range: %s → %s",
             df.shape, df.index[0].date(), df.index[-1].date())

    # 2. HMM training / loading
    log.info("═══ STEP 2/6  HMM regime model ════════════════════════════════")
    if use_ensemble:
        from src.ensemble import EnsembleHMM
        cache = ENSEMBLE_CACHE
        if not retrain and cache.exists():
            log.info("Loading cached ensemble from %s", cache)
            model = EnsembleHMM.load(cache)
        else:
            # Use default feature_cols (base 3) — same as single model in main.py.
            # Extended features (full set) are used in walk_forward.py which
            # computes them via _attach_hmm_features() before fitting.
            model = EnsembleHMM()
            model.fit(df)
            model.save(cache)
        log.info("Ensemble: %d models, n_states=%s",
                 len(model.models), model.n_states_list)
    else:
        if not retrain and MODEL_CACHE.exists():
            log.info("Loading cached model from %s", MODEL_CACHE)
            model = HMMRegimeModel.load(MODEL_CACHE)
        else:
            model = HMMRegimeModel()
            model.fit(df)
            model.save(MODEL_CACHE)

    if hasattr(model, "get_state_stats"):
        log.info("State statistics:\n%s", model.get_state_stats().to_string())

    # 3. Regime prediction
    log.info("═══ STEP 3/6  Predicting regimes ══════════════════════════════")
    df = model.predict(df)
    regime_dist = df["regime_cat"].value_counts().to_dict()
    log.info("Regime distribution: %s", regime_dist)

    # 4. Technical indicators
    log.info("═══ STEP 4/6  Computing indicators ════════════════════════════")
    df = attach_all(df)

    # 5. Signal confirmation masks
    log.info("═══ STEP 5/6  Building signal masks ═══════════════════════════")
    df = build_signal_series(df, use_regime_mapper=use_regime_mapper)
    raw_long = df["raw_long_signal"].sum()
    log.info("Raw long signals (before cooldown/position): %d", raw_long)

    if use_regime_mapper and "raw_short_signal" in df.columns:
        raw_short = df["raw_short_signal"].sum()
        log.info("Raw short signals (before cooldown/position): %d", raw_short)

        # Direction distribution summary
        if "allowed_direction" in df.columns:
            dir_dist = df["allowed_direction"].value_counts().to_dict()
            log.info("Direction distribution: %s", dir_dist)

    # 6. Backtest
    log.info("═══ STEP 6/6  Backtesting ══════════════════════════════════════")
    bt  = Backtester(use_regime_mapper=use_regime_mapper)
    res = bt.run(df)

    return res


def print_summary(result, use_regime_mapper: bool = False) -> None:
    m = result.metrics
    sep = "─" * 54

    mode_label = "MULTI-DIRECTION" if use_regime_mapper else "LONG-ONLY"

    print(f"\n{'═'*54}")
    print(f"  HMM REGIME TRADER  ·  BACKTEST RESULTS  [{mode_label}]")
    print(f"{'═'*54}")
    print(f"  Leverage             : {m['leverage']}×")
    print(f"  Initial capital      : ${m['initial_capital']:>12,.2f}")
    print(f"  Final equity         : ${m['final_equity']:>12,.2f}")
    print(sep)
    print(f"  Total return         : {m['total_return_pct']:>+.2f}%")
    print(f"  Buy-and-hold return  : {m['bh_return_pct']:>+.2f}%")
    print(f"  Alpha vs B&H         : {m['alpha_pct']:>+.2f}%")
    print(sep)
    print(f"  Max drawdown         : {m['max_drawdown_pct']:>.2f}%")
    print(f"  Sharpe ratio         : {m['sharpe_ratio']:>.3f}")
    print(sep)
    print(f"  Total trades         : {m['n_trades']}")
    print(f"  Win rate             : {m['win_rate_pct']:.1f}%")
    print(f"  Avg winning trade    : ${m['avg_win_usd']:>+,.2f}")
    print(f"  Avg losing trade     : ${m['avg_loss_usd']:>+,.2f}")
    print(f"  Profit factor        : {m['profit_factor']:.3f}")
    print(f"  Avg trade duration   : {m['avg_duration_h']:.1f} h")

    # Direction breakdown (multi-direction mode)
    if use_regime_mapper and m.get("long_trades", 0) + m.get("short_trades", 0) > 0:
        print(sep)
        print(f"  Long trades          : {m['long_trades']}  "
              f"(WR: {m['long_win_rate_pct']:.1f}%)")
        print(f"  Short trades         : {m['short_trades']}  "
              f"(WR: {m['short_win_rate_pct']:.1f}%)")

    print(f"{'═'*54}\n")

    if not result.trades.empty:
        print("  Last 5 trades:")
        cols = ["entry_time", "exit_price", "pnl", "pnl_pct", "exit_reason"]
        if "direction" in result.trades.columns:
            cols.insert(1, "direction")
        last5 = result.trades.tail(5)[cols].copy()
        last5["pnl"]     = last5["pnl"].apply(lambda v: f"${v:>+,.0f}")
        last5["pnl_pct"] = last5["pnl_pct"].apply(lambda v: f"{v:>+.2f}%")
        print(last5.to_string(index=False))
        print()


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HMM Regime Trading Terminal"
    )
    parser.add_argument("--no-dash",  action="store_true",
                        help="Skip the dashboard and only print results")
    parser.add_argument("--retrain",  action="store_true",
                        help="Force re-training of the HMM (ignore cache)")
    parser.add_argument("--regime-mapper", action="store_true",
                        help="Enable multi-direction trading (LONG/SHORT/FLAT)")
    parser.add_argument("--host",     default=config.DASH_HOST,
                        help=f"Dashboard host (default: {config.DASH_HOST})")
    parser.add_argument("--port",     default=config.DASH_PORT, type=int,
                        help=f"Dashboard port (default: {config.DASH_PORT})")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use 3-model HMM ensemble (n_states=[5,6,7])")
    parser.add_argument("--debug",    action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    result = run_pipeline(
        retrain=args.retrain,
        use_regime_mapper=args.regime_mapper,
        use_ensemble=args.ensemble,
    )
    print_summary(result, use_regime_mapper=args.regime_mapper)

    if not args.no_dash:
        log.info("Launching dashboard → http://%s:%d", args.host, args.port)
        try:
            from src.dashboard import launch
            launch(result, host=args.host, port=args.port, debug=args.debug,
                   use_regime_mapper=args.regime_mapper)
        except Exception as e:
            import traceback; traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
