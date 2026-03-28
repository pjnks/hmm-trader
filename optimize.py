"""
optimize.py
───────────
Random-search hyperparameter optimizer for the HMM Regime Trader.

Usage
─────
  python optimize.py                      # 300-combination random search
  python optimize.py --runs 50            # fewer combinations (fast test)
  python optimize.py --resume             # continue from existing CSV
  python optimize.py --heatmap-only       # regenerate heatmap from CSV
  python optimize.py --seed 0             # reproducible sampling

Output
──────
  optimization_results.csv   — all trial results  (saved every 10 runs)
  optimization_heatmap.html  — interactive parameter heatmap
  feature_importance.json    — per-feature importance from top-10 OOS runs
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
from src.data_fetcher import fetch_btc_hourly, build_hmm_features, resample_ohlcv
from src.hmm_model import HMMRegimeModel
from src.indicators import attach_all
from src.strategy import build_signal_series
from src.backtester import Backtester

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,           # quiet during optimization
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optimizer")

# ── Output paths ───────────────────────────────────────────────────────────────
RESULTS_CSV  = ROOT / "optimization_results.csv"
HEATMAP_HTML = ROOT / "optimization_heatmap.html"
IMP_JSON     = ROOT / "feature_importance.json"

# ── Robustness thresholds ──────────────────────────────────────────────────────
MIN_TRADES    = 10
MAX_NAN_FRAC  = 0.05
IS_FRAC       = 0.70
SAVE_EVERY    = 10
OVERFIT_DELTA = 0.5   # flag if OOS Sharpe < IS Sharpe - this value

# ── Parameter grid ─────────────────────────────────────────────────────────────
PARAM_GRID: dict[str, list] = {
    "timeframe":       ["1h", "4h"],
    "confirmations":   [5, 6, 7, 8],
    "leverage":        [1.0, 1.5, 2.0, 2.5],
    "cooldown_hours":  [24, 48, 72],
    "n_states":        [4, 5, 6, 7, 8],
    "feature_set":     ["base", "extended", "full"],
    "training_days":   [180, 365, 730],
    "covariance_type": ["full", "diag"],
}

# Config attributes that map directly to PARAM_GRID keys
_CONFIG_MAP = {
    "confirmations":  "MIN_CONFIRMATIONS",
    "leverage":       "LEVERAGE",
    "cooldown_hours": "COOLDOWN_HOURS",
}

# In-memory OHLCV cache keyed by (timeframe, training_days) to avoid re-fetching
_DATA_CACHE: dict[tuple, pd.DataFrame] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

def _sample_combos(n: int, seed: int = 42) -> list[dict]:
    """Draw *n* random combinations from PARAM_GRID (without replacement if possible)."""
    all_combos = [
        dict(zip(PARAM_GRID.keys(), vals))
        for vals in product(*PARAM_GRID.values())
    ]
    rng = random.Random(seed)
    k = min(n, len(all_combos))
    return rng.sample(all_combos, k)


def _params_key(params: dict) -> str:
    return str(sorted(params.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Config patching (single-threaded — safe to mutate global)
# ─────────────────────────────────────────────────────────────────────────────

def _patch_config(params: dict) -> dict:
    saved = {}
    for param_key, cfg_attr in _CONFIG_MAP.items():
        if param_key in params:
            saved[cfg_attr] = getattr(config, cfg_attr)
            setattr(config, cfg_attr, params[param_key])
    return saved


def _restore_config(saved: dict) -> None:
    for attr, val in saved.items():
        setattr(config, attr, val)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (with in-memory cache)
# ─────────────────────────────────────────────────────────────────────────────

def _load_base_1h() -> pd.DataFrame:
    """Load 1h raw OHLCV once and reuse across all trials."""
    if "1h_raw" not in _DATA_CACHE:
        _DATA_CACHE["1h_raw"] = fetch_btc_hourly(days=760)
    return _DATA_CACHE["1h_raw"]


def _get_trial_data(timeframe: str, training_days: int, feature_set: str) -> pd.DataFrame | None:
    """
    Return a DataFrame with HMM features for the given (timeframe, training_days,
    feature_set) combination.  Uses in-memory cache for the OHLCV base layer.
    """
    cache_key = (timeframe, training_days, feature_set)
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    try:
        raw_1h = _load_base_1h()

        # Resample if needed
        if timeframe != "1h":
            raw = resample_ohlcv(raw_1h, timeframe)
        else:
            raw = raw_1h.copy()

        # Slice to training window
        from datetime import datetime, timezone, timedelta
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=training_days)
        raw = raw[raw.index >= cutoff].copy()

        if len(raw) < 200:
            return None   # too little data

        # Base HMM features
        df = build_hmm_features(raw)

        # Extended features
        if feature_set in ("extended", "full"):
            from src.indicators import (
                compute_realized_vol_ratio, compute_return_autocorr,
                compute_vol_price_diverge, compute_candle_body_ratio,
                compute_bb_width,
            )
            df["realized_vol_ratio"] = compute_realized_vol_ratio(df)
            df["return_autocorr"]    = compute_return_autocorr(df)
            df["vol_price_diverge"]  = compute_vol_price_diverge(df)
            if feature_set == "full":
                df["candle_body_ratio"] = compute_candle_body_ratio(df)
                df["bb_width"]          = compute_bb_width(df)

        feature_cols = config.FEATURE_SETS[feature_set]
        df.dropna(subset=feature_cols, inplace=True)

        _DATA_CACHE[cache_key] = df
        return df

    except Exception as exc:
        log.debug("Data load failed for %s: %s", cache_key, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Train / test split
# ─────────────────────────────────────────────────────────────────────────────

def _split_is_oos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split = int(n * IS_FRAC)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance from HMM emissions
# ─────────────────────────────────────────────────────────────────────────────

def _compute_importance(model: HMMRegimeModel, feature_cols: list[str]) -> dict[str, float]:
    """
    Importance = variance of per-state emission means across states (original scale).
    Higher variance → feature more useful for regime discrimination.
    Normalised to sum to 1.
    """
    means_orig = model._scaler.inverse_transform(model._hmm.means_)
    var_per_feat = means_orig.var(axis=0)
    total = var_per_feat.sum()
    if total > 0:
        norm = var_per_feat / total
    else:
        norm = var_per_feat
    return dict(zip(feature_cols, norm.tolist()))


# ─────────────────────────────────────────────────────────────────────────────
# Single trial execution
# ─────────────────────────────────────────────────────────────────────────────

def _run_trial(params: dict) -> dict | None:
    """
    Run a single parameter combination.  Returns a result dict or None if
    the trial was skipped (no convergence, too few trades, too many NaNs).
    """
    feature_cols  = config.FEATURE_SETS[params["feature_set"]]
    timeframe     = params["timeframe"]
    training_days = params["training_days"]
    n_states      = params["n_states"]
    cov_type      = params["covariance_type"]

    df = _get_trial_data(timeframe, training_days, params["feature_set"])
    if df is None:
        return None

    # NaN check
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

        # Predict on IS and OOS independently (IS scaler applied to both)
        df_is_p  = model.predict(df_is)
        df_oos_p = model.predict(df_oos)

        # Attach indicators and signals to both splits
        results = {}
        for split_name, df_split in [("is", df_is_p), ("oos", df_oos_p)]:
            df_ind = attach_all(df_split)
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

        row: dict[str, Any] = {**params}
        for prefix, m in [("is", is_m), ("oos", oos_m)]:
            row[f"{prefix}_sharpe"]   = round(m["sharpe_ratio"],    3)
            row[f"{prefix}_return"]   = round(m["total_return_pct"], 2)
            row[f"{prefix}_drawdown"] = round(m["max_drawdown_pct"], 2)
            row[f"{prefix}_win_rate"] = round(m.get("win_rate_pct", 0), 1)
            row[f"{prefix}_trades"]   = m["n_trades"]
        row["overfit"] = overfit

        for feat in config.FEATURE_SETS["full"]:
            row[f"imp_{feat}"] = round(imp.get(feat, float("nan")), 4)

        return row

    except Exception as exc:
        log.debug("Trial failed: %s — %s", params, exc)
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

def _build_heatmap(df: pd.DataFrame, output: Path) -> None:
    """
    Generate a standalone HTML heatmap with 4 dropdowns:
    X-axis param, Y-axis param, metric, IS/OOS.
    Uses Plotly.react() via injected JS — no server required.
    """
    import plotly.graph_objects as go

    param_names = list(PARAM_GRID.keys())
    metrics = [
        ("sharpe",   "Sharpe Ratio"),
        ("return",   "Return %"),
        ("drawdown", "Drawdown %"),
        ("win_rate", "Win Rate %"),
    ]
    splits = [("is", "In-Sample"), ("oos", "Out-of-Sample")]

    # Pre-compute all pivot tables as dicts for JS embedding
    all_pivots: dict[str, dict] = {}
    for x_param in param_names:
        for y_param in param_names:
            if x_param == y_param:
                continue
            for m_key, _ in metrics:
                for s_key, _ in splits:
                    col = f"{s_key}_{m_key}"
                    if col not in df.columns:
                        continue
                    try:
                        pivot = (df.groupby([x_param, y_param])[col]
                                   .max().unstack(y_param))
                        op    = (df.groupby([x_param, y_param])["overfit"]
                                   .mean().unstack(y_param))

                        x_labels = [str(v) for v in pivot.index.tolist()]
                        y_labels = [str(v) for v in pivot.columns.tolist()]

                        z = [[None if np.isnan(v) else round(float(v), 3)
                              for v in row]
                             for row in pivot.values]
                        o = [[None if np.isnan(v) else round(float(v), 2)
                              for v in row]
                             for row in op.reindex(index=pivot.index,
                                                   columns=pivot.columns).values]

                        key = f"{x_param}|{y_param}|{s_key}|{m_key}"
                        all_pivots[key] = {
                            "z":        z,
                            "x_labels": x_labels,
                            "y_labels": y_labels,
                            "overfit":  o,
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

    # Inline CSS + controls HTML injected via post_script
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
</style>
<div id="controls">
  <div class="ctl"><label>X Axis</label>
    <select id="x-param"></select></div>
  <div class="ctl"><label>Y Axis</label>
    <select id="y-param"></select></div>
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
  <div id="overfit-key">■ Cell highlighted = >50% runs overfit</div>
</div>
<div id="stats-bar">Loading…</div>
"""

    js_data = json.dumps(all_pivots, allow_nan=False,
                         default=lambda x: None)

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
  return metric === 'drawdown' ? 'RdYlGn_r' : 'RdYlGn';
}}

function _updateChart() {{
  var xp = document.getElementById('x-param').value;
  var yp = document.getElementById('y-param').value;
  var m  = document.getElementById('metric').value;
  var s  = document.getElementById('split').value;

  if (xp === yp) {{
    // swap y to first different param
    var alt = _params.find(function(p) {{ return p !== xp; }});
    document.getElementById('y-param').value = alt;
    yp = alt;
  }}

  var key = xp + '|' + yp + '|' + s + '|' + m;
  var d   = _allData[key];
  if (!d) {{
    document.getElementById('stats-bar').textContent =
      'No data for: ' + key;
    return;
  }}

  // Build text matrix and overfit color overlay
  var textArr = d.z.map(function(row) {{
    return row.map(function(v) {{ return v !== null ? v.toFixed(3) : '—'; }});
  }});

  // Overfit mask: make cells with overfit_rate > 0.5 stand out via text marker
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
    x: d.y_labels,
    y: d.x_labels,
    colorscale: _getColorscale(m),
    text: annoText,
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

  // Count cells, overfit cells, best value
  var vals = d.z.flat().filter(function(v) {{ return v !== null; }});
  var ovCells = (d.overfit || []).flat().filter(function(v) {{ return v > 0.5; }}).length;
  var best = vals.length ? (m === 'drawdown' ? Math.min(...vals) : Math.max(...vals)) : 'N/A';
  document.getElementById('stats-bar').textContent =
    'Cells with data: ' + vals.length +
    ' | Overfit cells (>50%): ' + ovCells +
    ' | Best ' + m + ': ' + (typeof best === 'number' ? best.toFixed(3) : best) +
    ' | Each cell = max across all other parameters';
}}

['x-param','y-param','metric','split'].forEach(function(id) {{
  document.getElementById(id).addEventListener('change', _updateChart);
}});

// Initial render after Plotly loads
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
# Feature importance summary
# ─────────────────────────────────────────────────────────────────────────────

def _save_feature_importance(results: list[dict], path: Path) -> None:
    """Average importance across the top-10 OOS Sharpe runs."""
    if not results:
        return
    df = pd.DataFrame(results)
    if "oos_sharpe" not in df.columns:
        return

    imp_cols = [c for c in df.columns if c.startswith("imp_")]
    if not imp_cols:
        return

    top10 = df.nlargest(10, "oos_sharpe")
    mean_imp = top10[imp_cols].mean()
    out = {c.replace("imp_", ""): round(float(v), 4)
           for c, v in mean_imp.items() if not np.isnan(v)}

    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Feature importance written → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Progress bar (tqdm with plain fallback)
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
    parser = argparse.ArgumentParser(description="HMM Regime Trader — Optimizer")
    parser.add_argument("--runs",         type=int, default=300,
                        help="Number of random combinations to evaluate (default 300)")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed for combination sampling (default 42)")
    parser.add_argument("--resume",       action="store_true",
                        help="Skip combinations already in optimization_results.csv")
    parser.add_argument("--heatmap-only", action="store_true",
                        help="Regenerate heatmap/importance from existing CSV and exit")
    args = parser.parse_args()

    # ── Heatmap-only mode ────────────────────────────────────────────────────
    if args.heatmap_only:
        if not RESULTS_CSV.exists():
            print(f"ERROR: {RESULTS_CSV} not found. Run the optimizer first.")
            sys.exit(1)
        df = pd.read_csv(RESULTS_CSV)
        print(f"Loaded {len(df)} results from {RESULTS_CSV}")
        _build_heatmap(df, HEATMAP_HTML)
        _save_feature_importance(df.to_dict("records"), IMP_JSON)
        return

    # ── Sample combinations ──────────────────────────────────────────────────
    combos = _sample_combos(args.runs, seed=args.seed)
    print(f"Sampled {len(combos)} combinations from "
          f"{2*4*4*3*5*3*3*2:,} total ({100*len(combos)/(2*4*4*3*5*3*3*2):.1f}% coverage)")

    # ── Resume: skip already-done combinations ───────────────────────────────
    existing = _load_existing(RESULTS_CSV) if args.resume else []
    done_keys = {_params_key({k: r[k] for k in PARAM_GRID if k in r})
                 for r in existing}
    combos = [c for c in combos if _params_key(c) not in done_keys]
    if args.resume:
        print(f"Resuming: {len(done_keys)} done, {len(combos)} remaining")

    all_results = list(existing)
    skipped     = 0
    completed   = 0

    print(f"\nOptimizing {len(combos)} trials  "
          f"(min_trades={MIN_TRADES}, max_nan={MAX_NAN_FRAC:.0%}, "
          f"IS={IS_FRAC:.0%}/OOS={1-IS_FRAC:.0%})\n")

    # Pre-load 1h base data once
    print("Pre-loading 1h base data …")
    _load_base_1h()
    print("Data ready.\n")

    for params in _progress(combos, total=len(combos), desc="Optimizing"):
        result = _run_trial(params)
        if result:
            all_results.append(result)
            completed += 1
        else:
            skipped += 1

        # Save checkpoint every SAVE_EVERY trials
        if (completed + skipped) % SAVE_EVERY == 0 and all_results:
            _save(all_results, RESULTS_CSV)

    # Final save
    if all_results:
        _save(all_results, RESULTS_CSV)

    print(f"\nDone — {completed} successful, {skipped} skipped")
    print(f"Results saved → {RESULTS_CSV}")

    if not all_results:
        print("No successful results to report.")
        return

    # ── Print top 10 by OOS Sharpe ───────────────────────────────────────────
    df = pd.DataFrame(all_results)
    if "oos_sharpe" in df.columns:
        top = df.nlargest(10, "oos_sharpe")[
            ["timeframe", "n_states", "feature_set", "confirmations",
             "leverage", "cooldown_hours", "covariance_type", "training_days",
             "is_sharpe", "oos_sharpe", "oos_return", "oos_drawdown",
             "oos_trades", "overfit"]
        ]
        print("\n── TOP 10 by OOS Sharpe ──────────────────────────────────────")
        print(top.to_string(index=False))
        print()

    # ── Heatmap & feature importance ─────────────────────────────────────────
    _build_heatmap(df, HEATMAP_HTML)
    _save_feature_importance(all_results, IMP_JSON)


if __name__ == "__main__":
    main()
