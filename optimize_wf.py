"""
optimize_wf.py
──────────────
Walk-forward hyperparameter optimizer for the HMM Regime Trader.

Unlike optimize.py (which uses a simple 70/30 IS/OOS split), this optimizer
uses rolling walk-forward analysis as the fitness function: 6-month train /
3-month test windows, chained equity across windows.  This provides a much
more realistic and robust estimate of out-of-sample performance.

Supports two modes:
  Long-only (default):  optimises base trading parameters
  Multi-direction (--regime-mapper):  adds direction mapping parameters
    (conf_high_threshold, bull_med_action, bear_med_action)

Usage
─────
  python optimize_wf.py                      # 100-combination random search
  python optimize_wf.py --runs 50            # fewer combinations (fast test)
  python optimize_wf.py --resume             # continue from existing CSV
  python optimize_wf.py --heatmap-only       # regenerate heatmap from CSV
  python optimize_wf.py --seed 0             # reproducible sampling
  python optimize_wf.py --regime-mapper      # multi-direction optimizer

Output
──────
  Long-only mode:
    optimization_wf_results.csv   — all trial results
    optimization_wf_heatmap.html  — interactive heatmap

  Multi-direction mode:
    optimization_wf_multidir_results.csv
    optimization_wf_multidir_heatmap.html
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
from src.data_fetcher import fetch_btc_hourly
from walk_forward import run_walk_forward

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optimizer_wf")

# ── Output paths (set dynamically based on --regime-mapper) ──────────────────
RESULTS_CSV_LONG   = ROOT / "optimization_wf_results.csv"
HEATMAP_HTML_LONG  = ROOT / "optimization_wf_heatmap.html"
RESULTS_CSV_MULTI  = ROOT / "optimization_wf_multidir_results.csv"
HEATMAP_HTML_MULTI = ROOT / "optimization_wf_multidir_heatmap.html"
RESULTS_CSV_ENSEMBLE   = ROOT / "optimization_wf_ensemble_results.csv"
HEATMAP_HTML_ENSEMBLE  = ROOT / "optimization_wf_ensemble_heatmap.html"

# ── Walk-forward fixed parameters ────────────────────────────────────────────
# 6m/3m fits all tickers (ETH/XRP/SOL have only ~14 months of Polygon data).
# BTC (24 months) gets ~6 windows; alts get 2-3 windows.
WF_TRAIN_MONTHS = 6
WF_TEST_MONTHS  = 3
MIN_WINDOWS     = 2     # require at least 2 completed WF windows
SAVE_EVERY      = 5     # checkpoint every N trials (WF is slow, save often)

# ── Parameter grids ──────────────────────────────────────────────────────────

# Base grid (long-only mode)
PARAM_GRID_LONG: dict[str, list] = {
    "ticker":          ["X:BTCUSD", "X:ETHUSD", "X:XRPUSD", "X:SOLUSD"],
    "timeframe":       ["1h", "2h", "3h", "4h"],
    "n_states":        [4, 5, 6, 7, 8],
    "feature_set":     ["base", "extended", "full", "extended_v2"],
    "confirmations":   [6, 7, 8],
    "leverage":        [1.0, 1.5, 2.0],
    "cooldown_hours":  [48, 72],
    "covariance_type": ["full", "diag"],
}

# Multi-direction grid: base params + direction mapping + SHORT-specific overrides
PARAM_GRID_MULTI: dict[str, list] = {
    "ticker":               ["X:BTCUSD", "X:ETHUSD", "X:XRPUSD", "X:SOLUSD"],
    "timeframe":            ["1h", "2h", "3h", "4h"],
    "n_states":             [4, 5, 6, 7, 8],
    "feature_set":          ["base", "extended", "full", "extended_v2"],
    "confirmations":        [5, 6, 7, 8],
    "leverage":             [1.0, 1.5, 2.0],
    "cooldown_hours":       [24, 48, 72],
    "covariance_type":      ["full", "diag"],
    # Direction-mapping parameters
    "conf_high_threshold":  [0.70, 0.75, 0.80, 0.85, 0.90],
    "bull_med_action":      ["LONG_OR_FLAT", "LONG", "FLAT"],
    "bear_med_action":      ["SHORT_OR_FLAT", "SHORT", "FLAT"],
    # SHORT-specific overrides (asymmetric)
    "confirmations_short":  [7, 8],
    "cooldown_hours_short": [48, 72],
    "adx_min_short":        [25, 30],
}

# Config attributes that the optimizer monkey-patches per trial.
# confirmations is handled by run_walk_forward() directly.
_CONFIG_MAP = {
    "n_states":             "N_STATES",
    "covariance_type":      "COV_TYPE",
    "leverage":             "LEVERAGE",
    "cooldown_hours":       "COOLDOWN_HOURS",
    # SHORT-specific overrides (multi-direction only)
    "cooldown_hours_short": "COOLDOWN_HOURS_SHORT",
    "adx_min_short":        "ADX_MIN_SHORT",
}


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

def _sample_combos(grid: dict, n: int, seed: int = 42) -> list[dict]:
    """Draw *n* random combinations from the grid (without replacement if possible)."""
    all_combos = [
        dict(zip(grid.keys(), vals))
        for vals in product(*grid.values())
    ]
    rng = random.Random(seed)
    k = min(n, len(all_combos))
    return rng.sample(all_combos, k)


def _params_key(params: dict, grid: dict) -> str:
    return str(sorted((k, params[k]) for k in grid if k in params))


# ─────────────────────────────────────────────────────────────────────────────
# Config patching (single-threaded — safe to mutate global)
# ─────────────────────────────────────────────────────────────────────────────

def _patch_config(params: dict, use_regime_mapper: bool = False) -> dict:
    saved = {}
    for param_key, cfg_attr in _CONFIG_MAP.items():
        if param_key in params:
            saved[cfg_attr] = getattr(config, cfg_attr)
            setattr(config, cfg_attr, params[param_key])

    # Multi-direction: patch regime-direction mapping parameters
    if use_regime_mapper:
        saved["CONFIDENCE_HIGH_THRESHOLD"] = config.CONFIDENCE_HIGH_THRESHOLD
        saved["REGIME_DIRECTION_MAP"]      = config.REGIME_DIRECTION_MAP.copy()

        if "conf_high_threshold" in params:
            config.CONFIDENCE_HIGH_THRESHOLD = params["conf_high_threshold"]

        # Rebuild direction map from params
        bull_med = params.get("bull_med_action", "LONG_OR_FLAT")
        bear_med = params.get("bear_med_action", "SHORT_OR_FLAT")
        config.REGIME_DIRECTION_MAP = {
            ("BULL", "high"):  "LONG",
            ("BULL", "med"):   bull_med,
            ("BEAR", "high"):  "SHORT",
            ("BEAR", "med"):   bear_med,
            ("CHOP", "any"):   "FLAT",
        }

    return saved


def _restore_config(saved: dict) -> None:
    for attr, val in saved.items():
        setattr(config, attr, val)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-cache data
# ─────────────────────────────────────────────────────────────────────────────

def _pre_cache_tickers(tickers: list[str]) -> None:
    """Pre-fetch 1h OHLCV for each unique ticker so walk-forward doesn't re-download."""
    for ticker in tickers:
        print(f"  Pre-caching {ticker} …", end=" ", flush=True)
        try:
            df = fetch_btc_hourly(days=760, ticker=ticker)
            print(f"{len(df)} bars")
        except Exception as e:
            print(f"FAILED: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Single trial execution
# ─────────────────────────────────────────────────────────────────────────────

def _run_trial(params: dict, use_regime_mapper: bool = False,
               use_ensemble: bool = False) -> dict | None:
    """
    Run a single walk-forward trial.  Returns a result dict or None if
    the trial produced fewer than MIN_WINDOWS valid windows.
    """
    saved = _patch_config(params, use_regime_mapper=use_regime_mapper)
    try:
        results, combined_eq, combined_bh, all_trades = run_walk_forward(
            train_months        = WF_TRAIN_MONTHS,
            test_months         = WF_TEST_MONTHS,
            ticker              = params["ticker"],
            feature_set         = params["feature_set"],
            confirmations       = params["confirmations"],
            confirmations_short = params.get("confirmations_short"),
            timeframe           = params["timeframe"],
            quiet               = True,
            use_regime_mapper   = use_regime_mapper,
            use_ensemble        = use_ensemble,
        )

        if len(results) < MIN_WINDOWS:
            return None

        if combined_eq.empty or len(combined_eq) < 10:
            return None

        # ── Combined OOS metrics ─────────────────────────────────────────
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

        n_windows    = len(results)
        pos_windows  = sum(1 for r in results if r.return_pct > 0)
        total_trades = sum(r.n_trades for r in results)

        # ── Per-window consistency ───────────────────────────────────────
        window_sharpes = [r.sharpe_ratio for r in results]
        mean_sharpe    = float(np.mean(window_sharpes))
        std_sharpe     = float(np.std(window_sharpes))

        window_returns = [r.return_pct for r in results]
        mean_return    = float(np.mean(window_returns))
        std_return     = float(np.std(window_returns))

        # Overall win rate across all trades
        if not all_trades.empty and "pnl" in all_trades.columns and len(all_trades) > 0:
            overall_wr = float((all_trades["pnl"] > 0).sum() / len(all_trades) * 100)
        else:
            overall_wr = 0.0

        # Direction breakdown (multi-direction mode)
        long_trades  = 0
        short_trades = 0
        long_wr      = 0.0
        short_wr     = 0.0
        if (use_regime_mapper and not all_trades.empty
                and "direction" in all_trades.columns):
            long_mask  = all_trades["direction"] == "LONG"
            short_mask = all_trades["direction"] == "SHORT"
            long_trades  = int(long_mask.sum())
            short_trades = int(short_mask.sum())
            if long_trades > 0:
                long_wr = float((all_trades.loc[long_mask, "pnl"] > 0).sum()
                                / long_trades * 100)
            if short_trades > 0:
                short_wr = float((all_trades.loc[short_mask, "pnl"] > 0).sum()
                                 / short_trades * 100)

        # ── Build result row ─────────────────────────────────────────────
        row: dict[str, Any] = {**params}
        row["wf_sharpe"]         = round(oos_sharpe, 3)
        row["wf_return"]         = round(oos_ret, 2)
        row["wf_drawdown"]       = round(oos_dd, 2)
        row["wf_n_windows"]      = n_windows
        row["wf_pos_windows"]    = pos_windows
        row["wf_trades"]         = total_trades
        row["wf_win_rate"]       = round(overall_wr, 1)
        row["wf_mean_sharpe"]    = round(mean_sharpe, 3)
        row["wf_std_sharpe"]     = round(std_sharpe, 3)
        row["wf_mean_return"]    = round(mean_return, 2)
        row["wf_std_return"]     = round(std_return, 2)
        row["wf_initial_equity"] = round(oos_initial, 2)
        row["wf_final_equity"]   = round(oos_final, 2)

        if use_regime_mapper:
            row["wf_long_trades"]  = long_trades
            row["wf_short_trades"] = short_trades
            row["wf_long_wr"]      = round(long_wr, 1)
            row["wf_short_wr"]     = round(short_wr, 1)

        return row

    except Exception as exc:
        log.warning("Trial failed: %s — %s", params, exc)
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
        df = pd.read_csv(path)
        return df.to_dict("records")
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap generation
# ─────────────────────────────────────────────────────────────────────────────

def _build_heatmap(df: pd.DataFrame, output: Path, param_grid: dict) -> None:
    """
    Generate a standalone HTML heatmap with 4 dropdowns:
    X-axis param, Y-axis param, metric.
    """
    import plotly.graph_objects as go

    param_names = list(param_grid.keys())
    metrics = [
        ("wf_sharpe",      "WF Sharpe"),
        ("wf_return",      "WF Return %"),
        ("wf_drawdown",    "WF Drawdown %"),
        ("wf_win_rate",    "WF Win Rate %"),
        ("wf_mean_sharpe", "Mean Window Sharpe"),
        ("wf_std_sharpe",  "Std Window Sharpe"),
        ("wf_trades",      "Total Trades"),
    ]

    # Pre-compute all pivot tables as dicts for JS embedding
    all_pivots: dict[str, dict] = {}
    for x_param in param_names:
        for y_param in param_names:
            if x_param == y_param:
                continue
            for m_key, _ in metrics:
                if m_key not in df.columns:
                    continue
                try:
                    agg_fn = "min" if m_key == "wf_drawdown" else "max"
                    pivot = (df.groupby([x_param, y_param])[m_key]
                               .agg(agg_fn).unstack(y_param))

                    x_labels = [str(v) for v in pivot.index.tolist()]
                    y_labels = [str(v) for v in pivot.columns.tolist()]

                    z = [[None if pd.isna(v) else round(float(v), 3)
                          for v in row]
                         for row in pivot.values]

                    key = f"{x_param}|{y_param}|{m_key}"
                    all_pivots[key] = {
                        "z":        z,
                        "x_labels": x_labels,
                        "y_labels": y_labels,
                    }
                except Exception:
                    pass

    # Build initial empty figure (JS will populate on load)
    fig = go.Figure(go.Heatmap(
        z=[[0]], x=[""], y=[""], colorscale="RdYlGn",
        showscale=True, visible=True,
    ))
    fig.update_layout(
        height=560,
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#141820",
        font=dict(family="monospace", color="#e0e4f0"),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor="#1e2330"),
        yaxis=dict(gridcolor="#1e2330"),
    )

    # Build metric option HTML
    metric_opts = "\n".join(
        f'      <option value="{k}">{label}</option>'
        for k, label in metrics
    )

    controls_html = f"""
<style>
  body {{ background:#0d0f14; color:#e0e4f0; font-family:monospace; margin:0; }}
  #controls {{ display:flex; flex-wrap:wrap; gap:18px; padding:14px 20px;
              background:#141820; border-bottom:1px solid #1e2330; align-items:flex-end; }}
  .ctl label {{ display:block; font-size:0.68rem; color:#6b7394;
               text-transform:uppercase; letter-spacing:.08em; margin-bottom:4px; }}
  .ctl select {{ background:#0d0f14; color:#e0e4f0; border:1px solid #1e2330;
                padding:5px 10px; border-radius:4px; font-family:monospace;
                font-size:0.8rem; cursor:pointer; }}
  #stats-bar {{ padding:8px 20px; background:#0d0f14; font-size:0.72rem;
               color:#6b7394; border-bottom:1px solid #1e2330; }}
</style>
<div id="controls">
  <div class="ctl"><label>X Axis</label>
    <select id="x-param"></select></div>
  <div class="ctl"><label>Y Axis</label>
    <select id="y-param"></select></div>
  <div class="ctl"><label>Metric</label>
    <select id="metric">
{metric_opts}
    </select></div>
</div>
<div id="stats-bar">Loading…</div>
"""

    js_data      = json.dumps(all_pivots, allow_nan=False, default=lambda x: None)
    param_list_js = json.dumps(param_names)

    post_script = f"""
// Inject controls before the Plotly div
var ctrls = document.createElement('div');
ctrls.innerHTML = `{controls_html}`;
document.body.insertBefore(ctrls, document.body.firstChild);

var _allData = {js_data};
var _params  = {param_list_js};

// Populate x/y dropdowns
['x-param','y-param'].forEach(function(id, idx) {{
  var sel = document.getElementById(id);
  _params.forEach(function(p, i) {{
    var opt = document.createElement('option');
    opt.value = p; opt.textContent = p;
    if (i === idx) opt.selected = true;
    sel.appendChild(opt);
  }});
}});

var _gd = document.querySelector('.js-plotly-plot');

function _getColorscale(metric) {{
  return (metric === 'wf_drawdown' || metric === 'wf_std_sharpe')
    ? 'RdYlGn_r' : 'RdYlGn';
}}

function _updateChart() {{
  var xp = document.getElementById('x-param').value;
  var yp = document.getElementById('y-param').value;
  var m  = document.getElementById('metric').value;

  if (xp === yp) {{
    var alt = _params.find(function(p) {{ return p !== xp; }});
    document.getElementById('y-param').value = alt;
    yp = alt;
  }}

  var key = xp + '|' + yp + '|' + m;
  var d   = _allData[key];
  if (!d) {{
    document.getElementById('stats-bar').textContent = 'No data for: ' + key;
    return;
  }}

  var textArr = d.z.map(function(row) {{
    return row.map(function(v) {{ return v !== null ? v.toFixed(3) : '—'; }});
  }});

  Plotly.react(_gd, [{{
    type: 'heatmap',
    z: d.z,
    x: d.y_labels,
    y: d.x_labels,
    colorscale: _getColorscale(m),
    text: textArr,
    texttemplate: '%{{text}}',
    hovertemplate: yp + '=%{{x}}<br>' + xp + '=%{{y}}<br>value=%{{z:.3f}}<extra></extra>',
    showscale: true,
    colorbar: {{ bgcolor: '#141820', tickfont: {{ color: '#e0e4f0', family: 'monospace' }} }},
  }}], {{
    paper_bgcolor: '#0d0f14',
    plot_bgcolor:  '#141820',
    font:    {{ family: 'monospace', color: '#e0e4f0' }},
    xaxis:   {{ title: yp, gridcolor: '#1e2330', tickfont: {{ family: 'monospace' }} }},
    yaxis:   {{ title: xp, gridcolor: '#1e2330', tickfont: {{ family: 'monospace' }} }},
    height:  520,
    margin:  {{ l:100, r:60, t:20, b:100 }},
  }});

  var vals = d.z.flat().filter(function(v) {{ return v !== null; }});
  var best = vals.length
    ? ((m === 'wf_drawdown' || m === 'wf_std_sharpe') ? Math.min(...vals) : Math.max(...vals))
    : 'N/A';
  document.getElementById('stats-bar').textContent =
    'Cells with data: ' + vals.length +
    ' | Best ' + m + ': ' + (typeof best === 'number' ? best.toFixed(3) : best) +
    ' | Each cell = best across all other parameters';
}}

['x-param','y-param','metric'].forEach(function(id) {{
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
# Progress bar
# ─────────────────────────────────────────────────────────────────────────────

try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, total, desc=""):
        return _tqdm(iterable, total=total, desc=desc, ncols=88,
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
        description="HMM Regime Trader — Walk-Forward Optimizer"
    )
    parser.add_argument("--runs",         type=int, default=100,
                        help="Number of random combinations to evaluate (default 100)")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed for combination sampling (default 42)")
    parser.add_argument("--resume",       action="store_true",
                        help="Skip combinations already in results CSV")
    parser.add_argument("--heatmap-only", action="store_true",
                        help="Regenerate heatmap from existing CSV and exit")
    parser.add_argument("--regime-mapper", action="store_true",
                        help="Enable multi-direction optimization (LONG/SHORT/FLAT)")
    parser.add_argument("--ensemble",     action="store_true",
                        help="Use 3-model HMM ensemble (n_states=[5,6,7])")
    args = parser.parse_args()

    use_regime_mapper = args.regime_mapper
    use_ensemble      = args.ensemble

    # Select grid and output paths based on mode
    if use_regime_mapper:
        param_grid   = PARAM_GRID_MULTI
        results_csv  = RESULTS_CSV_MULTI
        heatmap_html = HEATMAP_HTML_MULTI
        mode_label   = "MULTI-DIRECTION"
    else:
        param_grid   = PARAM_GRID_LONG
        results_csv  = RESULTS_CSV_LONG
        heatmap_html = HEATMAP_HTML_LONG
        mode_label   = "LONG-ONLY"

    # Ensemble mode: remove n_states from grid (ensemble spans [5,6,7])
    # and write to separate output files to avoid overwriting single-model results
    if use_ensemble:
        param_grid = {k: v for k, v in param_grid.items() if k != "n_states"}
        results_csv  = RESULTS_CSV_ENSEMBLE
        heatmap_html = HEATMAP_HTML_ENSEMBLE
        mode_label  += " (ENSEMBLE)"

    total_grid = 1
    for v in param_grid.values():
        total_grid *= len(v)

    # ── Heatmap-only mode ────────────────────────────────────────────────
    if args.heatmap_only:
        if not results_csv.exists():
            print(f"ERROR: {results_csv} not found. Run the optimizer first.")
            sys.exit(1)
        df = pd.read_csv(results_csv)
        print(f"Loaded {len(df)} results from {results_csv}")
        _build_heatmap(df, heatmap_html, param_grid)
        return

    # ── Sample combinations ──────────────────────────────────────────────
    combos = _sample_combos(param_grid, args.runs, seed=args.seed)
    print(f"Sampled {len(combos)} combinations from "
          f"{total_grid:,} total ({100*len(combos)/total_grid:.1f}% coverage)")

    # ── Resume: skip already-done combinations ───────────────────────────
    existing = _load_existing(results_csv) if args.resume else []
    done_keys = {_params_key({k: r[k] for k in param_grid if k in r}, param_grid)
                 for r in existing}
    combos = [c for c in combos if _params_key(c, param_grid) not in done_keys]
    if args.resume:
        print(f"Resuming: {len(done_keys)} done, {len(combos)} remaining")

    all_results = list(existing)
    skipped     = 0
    completed   = 0

    print(f"\nWalk-Forward Optimizer [{mode_label}]")
    print(f"  Trials to run  : {len(combos)}")
    print(f"  Train window   : {WF_TRAIN_MONTHS} months")
    print(f"  Test window    : {WF_TEST_MONTHS} months")
    print(f"  Min windows    : {MIN_WINDOWS}")
    if use_regime_mapper:
        print(f"  Direction grid : conf_threshold × bull_med × bear_med = "
              f"{len(PARAM_GRID_MULTI['conf_high_threshold'])} × "
              f"{len(PARAM_GRID_MULTI['bull_med_action'])} × "
              f"{len(PARAM_GRID_MULTI['bear_med_action'])}")
    print()

    # ── Pre-cache 1h data for each ticker in the grid ────────────────────
    tickers_in_use = sorted(set(c["ticker"] for c in combos))
    print("Pre-caching ticker data:")
    _pre_cache_tickers(tickers_in_use)
    print()

    # ── Run trials ───────────────────────────────────────────────────────
    for params in _progress(combos, total=len(combos), desc="WF-Optimizing"):
        result = _run_trial(params, use_regime_mapper=use_regime_mapper,
                            use_ensemble=use_ensemble)
        if result:
            all_results.append(result)
            completed += 1
        else:
            skipped += 1

        # Checkpoint every SAVE_EVERY trials
        if (completed + skipped) % SAVE_EVERY == 0 and all_results:
            _save(all_results, results_csv)

    # Final save
    if all_results:
        _save(all_results, results_csv)

    print(f"\nDone — {completed} successful, {skipped} skipped")
    print(f"Results saved → {results_csv}")

    if not all_results:
        print("No successful results to report.")
        return

    # ── Print top 10 by WF Sharpe ────────────────────────────────────────
    df = pd.DataFrame(all_results)
    if "wf_sharpe" in df.columns:
        display_cols = [
            "ticker", "timeframe", "n_states", "feature_set", "confirmations",
            "leverage", "cooldown_hours", "covariance_type",
        ]
        if use_regime_mapper:
            display_cols += [
                "conf_high_threshold", "bull_med_action", "bear_med_action",
                "confirmations_short", "cooldown_hours_short", "adx_min_short",
            ]
        display_cols += [
            "wf_sharpe", "wf_return", "wf_drawdown",
            "wf_trades", "wf_pos_windows", "wf_n_windows",
            "wf_mean_sharpe", "wf_std_sharpe",
        ]
        if use_regime_mapper:
            display_cols += ["wf_long_trades", "wf_short_trades",
                             "wf_long_wr", "wf_short_wr"]

        display_cols = [c for c in display_cols if c in df.columns]
        top = df.nlargest(10, "wf_sharpe")[display_cols]

        print(f"\n── TOP 10 by Walk-Forward OOS Sharpe [{mode_label}] ────────")
        print(top.to_string(index=False))
        print()

    # ── Heatmap ──────────────────────────────────────────────────────────
    _build_heatmap(df, heatmap_html, param_grid)


if __name__ == "__main__":
    main()
