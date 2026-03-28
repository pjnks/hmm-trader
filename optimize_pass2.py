"""
optimize_pass2.py
─────────────────
Focused second-pass grid search across tickers and key strategy parameters.
All other HMM/model settings are fixed at their current best values.

Fixed parameters
────────────────
  n_states        = 8
  timeframe       = "1h"
  training_days   = 365
  covariance_type = "diag"

Variable parameters (grid)
──────────────────────────
  ticker          = [X:BTCUSD, X:ETHUSD, X:XRPUSD, X:SOLUSD]
  feature_set     = [volume_focused, full]
  confirmations   = [6, 7, 8]
  leverage        = [1.25, 1.50, 1.75]
  cooldown_hours  = [48, 72]

Total grid: 4 × 2 × 3 × 3 × 2 = 144 combinations

Usage
─────
  python optimize_pass2.py               # sample 100 from 144
  python optimize_pass2.py --all         # run all 144
  python optimize_pass2.py --seed 7      # reproducible sampling
  python optimize_pass2.py --heatmap-only

Output
──────
  optimization_results_pass2.csv
  optimization_heatmap_pass2.html
  feature_importance_pass2.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Path bootstrap ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from src.data_fetcher import (
    fetch_btc_hourly, build_hmm_features, resample_ohlcv,
)
from src.hmm_model import HMMRegimeModel
from src.indicators import (
    attach_all,
    compute_realized_vol_ratio, compute_return_autocorr,
    compute_vol_price_diverge, compute_candle_body_ratio, compute_bb_width,
)
from src.strategy import build_signal_series
from src.backtester import Backtester

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optimizer_pass2")

# ── Output paths ───────────────────────────────────────────────────────────────
RESULTS_CSV  = ROOT / "optimization_results_pass2.csv"
HEATMAP_HTML = ROOT / "optimization_heatmap_pass2.html"
IMP_JSON     = ROOT / "feature_importance_pass2.json"

# ── Fixed parameters ───────────────────────────────────────────────────────────
FIXED = {
    "n_states":        8,
    "timeframe":       "1h",
    "training_days":   365,
    "covariance_type": "diag",
}

# ── Variable grid ──────────────────────────────────────────────────────────────
PARAM_GRID: dict[str, list] = {
    "ticker":         ["X:BTCUSD", "X:ETHUSD", "X:XRPUSD", "X:SOLUSD"],
    "feature_set":    ["volume_focused", "full"],
    "confirmations":  [6, 7, 8],
    "leverage":       [1.25, 1.50, 1.75],
    "cooldown_hours": [48, 72],
}

TOTAL_COMBOS = 4 * 2 * 3 * 3 * 2   # 144

# ── Robustness thresholds ──────────────────────────────────────────────────────
MIN_TRADES    = 10
MAX_NAN_FRAC  = 0.05
IS_FRAC       = 0.70
SAVE_EVERY    = 10
OVERFIT_DELTA = 0.5

# Config keys that map to config module attributes
_CONFIG_MAP = {
    "confirmations":  "MIN_CONFIRMATIONS",
    "leverage":       "LEVERAGE",
    "cooldown_hours": "COOLDOWN_HOURS",
}

# In-memory OHLCV cache keyed by ticker — fetched once, reused across trials
_RAW_CACHE: dict[str, pd.DataFrame] = {}
# Processed feature cache keyed by (ticker, feature_set)
_DATA_CACHE: dict[tuple, pd.DataFrame] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

def _all_combos() -> list[dict]:
    return [
        dict(zip(PARAM_GRID.keys(), vals))
        for vals in product(*PARAM_GRID.values())
    ]


def _sample_combos(n: int, seed: int = 42) -> list[dict]:
    combos = _all_combos()
    rng    = random.Random(seed)
    return rng.sample(combos, min(n, len(combos)))


def _params_key(params: dict) -> str:
    return str(sorted(params.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Config patching
# ─────────────────────────────────────────────────────────────────────────────

def _patch_config(params: dict) -> dict:
    saved = {}
    for pk, cfg_attr in _CONFIG_MAP.items():
        if pk in params:
            saved[cfg_attr] = getattr(config, cfg_attr)
            setattr(config, cfg_attr, params[pk])
    return saved


def _restore_config(saved: dict) -> None:
    for attr, val in saved.items():
        setattr(config, attr, val)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_raw_1h(ticker: str) -> pd.DataFrame:
    """Fetch and cache 1h raw OHLCV for *ticker*."""
    if ticker not in _RAW_CACHE:
        _RAW_CACHE[ticker] = fetch_btc_hourly(
            days=FIXED["training_days"] + 60,
            ticker=ticker,
        )
    return _RAW_CACHE[ticker]


def _get_trial_data(ticker: str, feature_set: str) -> pd.DataFrame | None:
    """
    Return a feature-enriched DataFrame for the given ticker and feature set.
    Uses FIXED training_days (365 days).  Cached in memory.
    """
    cache_key = (ticker, feature_set)
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    try:
        raw_1h = _load_raw_1h(ticker)

        from datetime import datetime, timezone, timedelta
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=FIXED["training_days"])
        raw = raw_1h[raw_1h.index >= cutoff].copy()

        if len(raw) < 200:
            return None

        # Base features (log_return, price_range, volume_change always present)
        df = build_hmm_features(raw)

        # Extended features (volume_focused only needs realized_vol_ratio)
        if feature_set in ("extended", "full", "volume_focused"):
            df["realized_vol_ratio"] = compute_realized_vol_ratio(df)
        if feature_set in ("extended", "full"):
            df["return_autocorr"]  = compute_return_autocorr(df)
            df["vol_price_diverge"] = compute_vol_price_diverge(df)
        if feature_set == "full":
            df["candle_body_ratio"] = compute_candle_body_ratio(df)
            df["bb_width"]          = compute_bb_width(df)

        feature_cols = config.FEATURE_SETS[feature_set]
        df.dropna(subset=feature_cols, inplace=True)

        _DATA_CACHE[cache_key] = df
        return df

    except Exception as exc:
        log.debug("Data load failed for (%s, %s): %s", ticker, feature_set, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Train / test split
# ─────────────────────────────────────────────────────────────────────────────

def _split_is_oos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n     = len(df)
    split = int(n * IS_FRAC)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def _compute_importance(model: HMMRegimeModel, feature_cols: list[str]) -> dict[str, float]:
    means_orig  = model._scaler.inverse_transform(model._hmm.means_)
    var_per_feat = means_orig.var(axis=0)
    total = var_per_feat.sum()
    norm  = var_per_feat / total if total > 0 else var_per_feat
    return dict(zip(feature_cols, norm.tolist()))


# ─────────────────────────────────────────────────────────────────────────────
# Single trial
# ─────────────────────────────────────────────────────────────────────────────

def _run_trial(params: dict) -> dict | None:
    ticker      = params["ticker"]
    feature_set = params["feature_set"]
    feature_cols = config.FEATURE_SETS[feature_set]
    n_states    = FIXED["n_states"]
    cov_type    = FIXED["covariance_type"]

    df = _get_trial_data(ticker, feature_set)
    if df is None:
        return None

    nan_frac = df[feature_cols].isna().mean().max()
    if nan_frac > MAX_NAN_FRAC:
        return None

    df_is, df_oos = _split_is_oos(df)
    if len(df_is) < 200 or len(df_oos) < 100:
        return None

    saved = _patch_config(params)
    try:
        model = HMMRegimeModel(
            n_states     = n_states,
            cov_type     = cov_type,
            feature_cols = feature_cols,
        )
        model.fit(df_is)

        if not model.converged:
            return None

        results = {}
        for split_name, df_split in [("is", df_is), ("oos", df_oos)]:
            df_p   = model.predict(df_split)
            df_ind = attach_all(df_p)
            df_sig = build_signal_series(df_ind)
            bt     = Backtester()
            res    = bt.run(df_sig)
            m      = res.metrics
            if not m or m.get("n_trades", 0) < MIN_TRADES:
                return None
            results[split_name] = m

        is_m   = results["is"]
        oos_m  = results["oos"]
        overfit = int(oos_m["sharpe_ratio"] < is_m["sharpe_ratio"] - OVERFIT_DELTA)
        imp     = _compute_importance(model, feature_cols)

        row: dict[str, Any] = {**params, **FIXED}
        for prefix, m in [("is", is_m), ("oos", oos_m)]:
            row[f"{prefix}_sharpe"]   = round(m["sharpe_ratio"],    3)
            row[f"{prefix}_return"]   = round(m["total_return_pct"], 2)
            row[f"{prefix}_drawdown"] = round(m["max_drawdown_pct"], 2)
            row[f"{prefix}_win_rate"] = round(m.get("win_rate_pct", 0), 1)
            row[f"{prefix}_trades"]   = m["n_trades"]
        row["overfit"] = overfit

        # Store importance for all full-set features; NaN if not in this set
        for feat in config.FEATURE_SETS["full"]:
            row[f"imp_{feat}"] = round(imp.get(feat, float("nan")), 4)

        return row

    except Exception as exc:
        log.debug("Trial failed %s: %s", params, exc)
        return None
    finally:
        _restore_config(saved)


# ─────────────────────────────────────────────────────────────────────────────
# CSV persistence
# ─────────────────────────────────────────────────────────────────────────────

def _save(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap — ticker × parameter, OOS Sharpe
# ─────────────────────────────────────────────────────────────────────────────

def _build_heatmap(df: pd.DataFrame, output: Path) -> None:
    """
    Standalone HTML heatmap with ticker on Y-axis and a selectable parameter
    on X-axis.  Metric and IS/OOS are also selectable.  Overfit cells are
    marked with ⚠.
    """
    import plotly.graph_objects as go

    variable_params = [k for k in PARAM_GRID if k != "ticker"]
    metrics = [
        ("sharpe",   "Sharpe Ratio"),
        ("return",   "Return %"),
        ("drawdown", "Drawdown %"),
        ("win_rate", "Win Rate %"),
    ]
    splits = [("oos", "Out-of-Sample"), ("is", "In-Sample")]
    tickers = PARAM_GRID["ticker"]

    # Pre-compute pivot tables
    all_pivots: dict[str, dict] = {}
    for x_param in variable_params:
        for m_key, _ in metrics:
            for s_key, _ in splits:
                col = f"{s_key}_{m_key}"
                if col not in df.columns:
                    continue
                try:
                    pivot = (
                        df.groupby(["ticker", x_param])[col]
                        .max()
                        .unstack(x_param)
                        .reindex(tickers)
                    )
                    op = (
                        df.groupby(["ticker", x_param])["overfit"]
                        .mean()
                        .unstack(x_param)
                        .reindex(tickers)
                    )

                    x_labels = [str(v) for v in pivot.columns.tolist()]
                    y_labels = [str(v) for v in pivot.index.tolist()]

                    z = [[None if np.isnan(v) else round(float(v), 3)
                          for v in row]
                         for row in pivot.values]
                    o = [[None if np.isnan(v) else round(float(v), 2)
                          for v in row]
                         for row in op.reindex(
                             index=pivot.index, columns=pivot.columns
                         ).values]

                    key = f"{x_param}|{s_key}|{m_key}"
                    all_pivots[key] = {
                        "z":        z,
                        "x_labels": x_labels,
                        "y_labels": y_labels,
                        "overfit":  o,
                    }
                except Exception:
                    pass

    fig = go.Figure(go.Heatmap(
        z=[[0]], x=[""], y=[""], colorscale="RdYlGn",
        showscale=True, visible=True,
    ))
    fig.update_layout(
        height=480,
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#141820",
        font=dict(family="monospace", color="#e0e4f0"),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor="#1e2330"),
        yaxis=dict(gridcolor="#1e2330"),
    )

    controls_html = """
<style>
  body { background:#0d0f14; color:#e0e4f0; font-family:monospace; margin:0; }
  #controls { display:flex; flex-wrap:wrap; gap:18px; padding:14px 20px;
              background:#141820; border-bottom:1px solid #1e2330; align-items:flex-end; }
  .ctl label { display:block; font-size:0.68rem; color:#6b7394;
               text-transform:uppercase; letter-spacing:.08em; margin-bottom:4px; }
  .ctl select { background:#0d0f14; color:#e0e4f0; border:1px solid #1e2330;
                padding:5px 10px; border-radius:4px; font-family:monospace;
                font-size:0.8rem; cursor:pointer; }
  #overfit-key { font-size:0.7rem; color:#e040fb; border:1px solid #440044;
                 background:#1a0022; padding:4px 10px; border-radius:4px;
                 align-self:flex-end; }
  #stats-bar { padding:8px 20px; background:#0d0f14; font-size:0.72rem;
               color:#6b7394; border-bottom:1px solid #1e2330; }
  #title-bar { padding:10px 20px; background:#0d0f14; font-size:0.9rem;
               color:#a0c8ff; border-bottom:1px solid #1e2330; }
</style>
<div id="title-bar">HMM Trader — Pass 2 Optimisation: OOS Sharpe by Ticker × Parameter</div>
<div id="controls">
  <div class="ctl"><label>X Axis (Parameter)</label>
    <select id="x-param"></select></div>
  <div class="ctl"><label>Metric</label>
    <select id="metric">
      <option value="sharpe">Sharpe Ratio</option>
      <option value="return">Return %</option>
      <option value="drawdown">Drawdown %</option>
      <option value="win_rate">Win Rate %</option>
    </select></div>
  <div class="ctl"><label>Split</label>
    <select id="split">
      <option value="oos">Out-of-Sample</option>
      <option value="is">In-Sample</option>
    </select></div>
  <div id="overfit-key">⚠ = >50% runs overfit</div>
</div>
<div id="stats-bar">Loading…</div>
"""

    js_data       = json.dumps(all_pivots, allow_nan=False, default=lambda x: None)
    param_list_js = json.dumps(variable_params)

    post_script = f"""
var ctrls = document.createElement('div');
ctrls.innerHTML = `{controls_html}`;
document.body.insertBefore(ctrls, document.body.firstChild);

var _allData = {js_data};
var _params  = {param_list_js};

var sel = document.getElementById('x-param');
_params.forEach(function(p, i) {{
  var opt = document.createElement('option');
  opt.value = p; opt.textContent = p;
  if (i === 0) opt.selected = true;
  sel.appendChild(opt);
}});

var _gd = document.querySelector('.js-plotly-plot');

function _getColorscale(metric) {{
  return metric === 'drawdown' ? 'RdYlGn_r' : 'RdYlGn';
}}

function _updateChart() {{
  var xp = document.getElementById('x-param').value;
  var m  = document.getElementById('metric').value;
  var s  = document.getElementById('split').value;
  var key = xp + '|' + s + '|' + m;
  var d   = _allData[key];
  if (!d) {{
    document.getElementById('stats-bar').textContent = 'No data for: ' + key;
    return;
  }}

  var annoText = d.z.map(function(row, ri) {{
    return row.map(function(v, ci) {{
      var ov = d.overfit && d.overfit[ri] && d.overfit[ri][ci];
      var vs = v !== null ? v.toFixed(3) : '—';
      return ov > 0.5 ? '⚠ ' + vs : vs;
    }});
  }});

  Plotly.react(_gd, [{{
    type: 'heatmap',
    z: d.z,
    x: d.x_labels,
    y: d.y_labels,
    colorscale: _getColorscale(m),
    text: annoText,
    texttemplate: '%{{text}}',
    hovertemplate: xp + '=%{{x}}<br>ticker=%{{y}}<br>value=%{{z:.3f}}<extra></extra>',
    showscale: true,
    colorbar: {{ bgcolor: '#141820', tickfont: {{ color: '#e0e4f0', family: 'monospace' }} }},
  }}], {{
    paper_bgcolor: '#0d0f14',
    plot_bgcolor:  '#141820',
    font:    {{ family: 'monospace', color: '#e0e4f0' }},
    xaxis:   {{ title: xp, gridcolor: '#1e2330', tickfont: {{ family: 'monospace' }} }},
    yaxis:   {{ title: 'Ticker', gridcolor: '#1e2330', tickfont: {{ family: 'monospace' }} }},
    height:  440,
    margin:  {{ l:120, r:60, t:20, b:100 }},
  }});

  var vals    = d.z.flat().filter(function(v) {{ return v !== null; }});
  var ovCells = (d.overfit || []).flat().filter(function(v) {{ return v > 0.5; }}).length;
  var best    = vals.length ? (m === 'drawdown' ? Math.min(...vals) : Math.max(...vals)) : 'N/A';
  document.getElementById('stats-bar').textContent =
    'Cells: ' + vals.length +
    ' | Overfit (>50%): ' + ovCells +
    ' | Best ' + m + ': ' + (typeof best === 'number' ? best.toFixed(3) : best) +
    ' | Y=ticker, each cell = max across other parameters';
}}

['x-param','metric','split'].forEach(function(id) {{
  document.getElementById(id).addEventListener('change', _updateChart);
}});
setTimeout(_updateChart, 200);
"""

    fig.write_html(
        str(output),
        include_plotlyjs="cdn",
        post_script=post_script,
        full_html=True,
    )
    print(f"Heatmap written → {output}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def _save_feature_importance(results: list[dict], path: Path) -> None:
    if not results:
        return
    df = pd.DataFrame(results)
    if "oos_sharpe" not in df.columns:
        return
    imp_cols = [c for c in df.columns if c.startswith("imp_")]
    if not imp_cols:
        return
    top10    = df.nlargest(10, "oos_sharpe")
    mean_imp = top10[imp_cols].mean()
    out      = {c.replace("imp_", ""): round(float(v), 4)
                for c, v in mean_imp.items() if not np.isnan(v)}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Feature importance written → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Progress bar
# ─────────────────────────────────────────────────────────────────────────────

try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, total, desc=""):
        return _tqdm(iterable, total=total, desc=desc, ncols=80,
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
except ImportError:
    def _progress(iterable, total, desc=""):
        start = time.time()
        for i, item in enumerate(iterable, 1):
            elapsed = time.time() - start
            eta     = (elapsed / i) * (total - i) if i > 0 else 0
            print(f"\r{desc}: {i}/{total}  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                  end="", flush=True)
            yield item
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HMM Trader — Pass 2 Optimizer (ticker × strategy params)"
    )
    parser.add_argument("--runs",         type=int, default=100,
                        help="Max combinations to evaluate (default 100; full grid = 144)")
    parser.add_argument("--all",          action="store_true",
                        help="Run all 144 grid combinations (overrides --runs)")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--resume",       action="store_true",
                        help="Skip combinations already in the output CSV")
    parser.add_argument("--heatmap-only", action="store_true",
                        help="Regenerate heatmap from existing CSV and exit")
    args = parser.parse_args()

    # ── Heatmap-only mode ────────────────────────────────────────────────────
    if args.heatmap_only:
        if not RESULTS_CSV.exists():
            print(f"ERROR: {RESULTS_CSV} not found.")
            sys.exit(1)
        df = pd.read_csv(RESULTS_CSV)
        print(f"Loaded {len(df)} results from {RESULTS_CSV}")
        _build_heatmap(df, HEATMAP_HTML)
        _save_feature_importance(df.to_dict("records"), IMP_JSON)
        return

    # ── Build combination list ────────────────────────────────────────────────
    if args.all:
        combos = _all_combos()
        print(f"Running full grid: {len(combos)} combinations")
    else:
        combos = _sample_combos(args.runs, seed=args.seed)
        print(f"Sampled {len(combos)} from {TOTAL_COMBOS} total "
              f"({100 * len(combos) / TOTAL_COMBOS:.1f}% coverage)")

    # ── Resume ───────────────────────────────────────────────────────────────
    existing  = _load_existing(RESULTS_CSV) if args.resume else []
    done_keys = {_params_key({k: r[k] for k in PARAM_GRID if k in r})
                 for r in existing}
    combos = [c for c in combos if _params_key(c) not in done_keys]
    if args.resume:
        print(f"Resuming: {len(done_keys)} done, {len(combos)} remaining")

    all_results = list(existing)
    skipped     = 0
    completed   = 0

    print(f"\nFixed params: n_states={FIXED['n_states']}, "
          f"timeframe={FIXED['timeframe']}, "
          f"training_days={FIXED['training_days']}, "
          f"cov_type={FIXED['covariance_type']}")
    print(f"Optimizing {len(combos)} trials  "
          f"(min_trades={MIN_TRADES}, IS={IS_FRAC:.0%}/OOS={1-IS_FRAC:.0%})\n")

    # Pre-fetch data for all tickers
    print("Pre-fetching OHLCV data for all tickers …")
    for ticker in PARAM_GRID["ticker"]:
        try:
            _load_raw_1h(ticker)
            print(f"  {ticker} ✓")
        except Exception as exc:
            print(f"  {ticker} FAILED: {exc}")
    print()

    for params in _progress(combos, total=len(combos), desc="Optimizing"):
        result = _run_trial(params)
        if result:
            all_results.append(result)
            completed += 1
        else:
            skipped += 1

        if (completed + skipped) % SAVE_EVERY == 0 and all_results:
            _save(all_results, RESULTS_CSV)

    if all_results:
        _save(all_results, RESULTS_CSV)

    print(f"\nDone — {completed} successful, {skipped} skipped")
    print(f"Results saved → {RESULTS_CSV}")

    if not all_results:
        print("No successful results to report.")
        return

    # ── Top 10 by OOS Sharpe ─────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    if "oos_sharpe" in df.columns:
        cols = ["ticker", "feature_set", "confirmations", "leverage",
                "cooldown_hours", "is_sharpe", "oos_sharpe",
                "oos_return", "oos_drawdown", "oos_trades", "overfit"]
        top = df.nlargest(10, "oos_sharpe")[cols]
        print("\n── TOP 10 by OOS Sharpe ─────────────────────────────────────────")
        print(top.to_string(index=False))
        print()

    # ── Best config per ticker ───────────────────────────────────────────────
    if "oos_sharpe" in df.columns and "ticker" in df.columns:
        print("── Best OOS Sharpe per Ticker ───────────────────────────────────")
        for ticker in PARAM_GRID["ticker"]:
            sub = df[df["ticker"] == ticker]
            if sub.empty:
                print(f"  {ticker}: no data")
                continue
            best = sub.loc[sub["oos_sharpe"].idxmax()]
            print(f"  {ticker:12s}  sharpe={best['oos_sharpe']:.3f}  "
                  f"return={best['oos_return']:.1f}%  "
                  f"feat={best['feature_set']}  "
                  f"conf={int(best['confirmations'])}  "
                  f"lev={best['leverage']:.2f}  "
                  f"cool={int(best['cooldown_hours'])}h")
        print()

    _build_heatmap(df, HEATMAP_HTML)
    _save_feature_importance(all_results, IMP_JSON)


if __name__ == "__main__":
    main()
