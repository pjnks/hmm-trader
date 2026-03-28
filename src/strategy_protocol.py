"""
strategy_protocol.py
────────────────────
Defines the StrategyProtocol interface that robustness tests operate against.
Any trading strategy can implement this protocol to plug into the robustness
testing suite without modifying robustness.py.

Current implementations:
  - HMMCryptoStrategy (this file) — wraps the HMM crypto walk-forward pipeline

Future implementations:
  - NDX100Strategy (planned) — equity-based strategy with different data/fees
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Protocol, NamedTuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config

log = logging.getLogger("robustness")


# ─────────────────────────────────────────────────────────────────────────────
# Shared result container
# ─────────────────────────────────────────────────────────────────────────────

class WFResult(NamedTuple):
    """Minimal walk-forward result container for robustness tests."""
    sharpe:      float          # combined OOS annualised Sharpe
    return_pct:  float          # combined OOS total return %
    max_dd_pct:  float          # combined OOS max drawdown %
    n_trades:    int            # total completed trades across all windows
    win_rate:    float          # overall win rate %
    n_windows:   int            # number of walk-forward windows
    pos_windows: int            # windows with positive return
    trades_df:   pd.DataFrame   # all OOS trades concatenated


# ─────────────────────────────────────────────────────────────────────────────
# Protocol
# ─────────────────────────────────────────────────────────────────────────────

class StrategyProtocol(Protocol):
    """Interface that robustness tests operate against.

    Any strategy class can implement these methods (structural subtyping).
    No inheritance required — just match the signatures.
    """

    def run_walk_forward(self, params: dict) -> WFResult | None:
        """Run a full walk-forward backtest with the given params.
        Returns WFResult or None if the run failed / too few windows.
        """
        ...

    def get_base_params(self) -> dict:
        """Return the best-known config as a flat dict."""
        ...

    def get_param_sensitivity_grid(self) -> dict:
        """Return {param_name: [values_to_test]} for sensitivity sweep.
        The base value should be included in each list.
        """
        ...

    def get_tickers(self) -> list:
        """Return list of ticker strings for transferability test."""
        ...

    def get_ticker_param_name(self) -> str:
        """Return the key name used for 'ticker' in the params dict."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# HMM Crypto Adapter
# ─────────────────────────────────────────────────────────────────────────────

class HMMCryptoStrategy:
    """Concrete implementation of StrategyProtocol for the HMM crypto trader.

    Wraps run_walk_forward() + _patch_config/_restore_config from the
    existing optimizer infrastructure.
    """

    WF_TRAIN_MONTHS = 6
    WF_TEST_MONTHS  = 3
    MIN_WINDOWS     = 2

    def __init__(self, use_ensemble: bool = False) -> None:
        self.use_ensemble = use_ensemble

    def get_base_params(self) -> dict:
        params = {
            "ticker":          "X:XRPUSD",
            "timeframe":       "2h",
            "n_states":        6,
            "feature_set":     "full",
            "confirmations":   7,
            "leverage":        2.0,
            "cooldown_hours":  48,
            "covariance_type": "full",
        }
        if self.use_ensemble:
            params["_use_ensemble"] = True
        return params

    def get_param_sensitivity_grid(self) -> dict:
        if self.use_ensemble:
            # n_states excluded — ensemble already spans [5, 6, 7]
            return {
                "confirmations":   [6, 7, 8],
                "leverage":        [1.5, 2.0, 2.5],
                "cooldown_hours":  [24, 48, 72],
                "covariance_type": ["diag", "full"],
            }
        return {
            "n_states":        [5, 6, 7],
            "confirmations":   [6, 7, 8],
            "leverage":        [1.5, 2.0, 2.5],
            "cooldown_hours":  [24, 48, 72],
            "covariance_type": ["diag", "full"],
        }

    def get_tickers(self) -> list:
        return ["X:BTCUSD", "X:ETHUSD", "X:XRPUSD", "X:SOLUSD"]

    def get_ticker_param_name(self) -> str:
        return "ticker"

    def run_walk_forward(self, params: dict) -> WFResult | None:
        from optimize_wf import _patch_config, _restore_config
        from walk_forward import run_walk_forward

        # Extract meta-params (prefixed with _) before patching config
        train_months = params.get("_train_months", self.WF_TRAIN_MONTHS)
        test_months  = params.get("_test_months", self.WF_TEST_MONTHS)
        use_ensemble = params.get("_use_ensemble", self.use_ensemble)
        config_params = {k: v for k, v in params.items() if not k.startswith("_")}

        saved = _patch_config(config_params)
        try:
            results, combined_eq, combined_bh, all_trades = run_walk_forward(
                train_months  = train_months,
                test_months   = test_months,
                ticker        = config_params["ticker"],
                feature_set   = config_params["feature_set"],
                confirmations = config_params["confirmations"],
                timeframe     = config_params["timeframe"],
                quiet         = True,
                use_ensemble  = use_ensemble,
            )

            if len(results) < self.MIN_WINDOWS:
                return None
            if combined_eq.empty or len(combined_eq) < 10:
                return None

            # ── Combined OOS metrics (same math as optimize_wf._run_trial) ──
            oos_initial = float(combined_eq.iloc[0])
            oos_final   = float(combined_eq.iloc[-1])
            oos_ret     = (oos_final / oos_initial - 1) * 100

            roll_max = combined_eq.cummax()
            oos_dd   = float(((combined_eq - roll_max) / roll_max * 100).min())

            hr = combined_eq.pct_change().dropna()
            oos_sharpe = (
                float(hr.mean() / hr.std() * np.sqrt(24 * 365))
                if len(hr) > 1 and hr.std() > 0 else 0.0
            )

            total_trades = sum(r.n_trades for r in results)
            pos_windows  = sum(1 for r in results if r.return_pct > 0)

            overall_wr = 0.0
            if not all_trades.empty and "pnl" in all_trades.columns:
                overall_wr = float(
                    (all_trades["pnl"] > 0).sum() / len(all_trades) * 100
                )

            return WFResult(
                sharpe      = round(oos_sharpe, 3),
                return_pct  = round(oos_ret, 2),
                max_dd_pct  = round(oos_dd, 2),
                n_trades    = total_trades,
                win_rate    = round(overall_wr, 1),
                n_windows   = len(results),
                pos_windows = pos_windows,
                trades_df   = all_trades,
            )

        except Exception as exc:
            log.warning("Walk-forward failed: %s", exc)
            return None
        finally:
            _restore_config(saved)
