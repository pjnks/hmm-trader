#!/usr/bin/env python3
"""
evaluate_index_ic.py — Index Regime IC Gate (Kalshi range-pricer pre-flight)

PURPOSE
───────
Falsifiable, zero-cost gate answering ONE question before anyone builds a
Kalshi financial-markets range/direction pricer:

    "Does the ensemble HMM regime carry tradeable DIRECTIONAL and/or
     VOLATILITY information on a broad index (SPY / QQQ)?"

This is the §5 test from the DIAMOND-fork strategy memo. It is deliberately
adversarial: it decodes regimes CAUSALLY (no smoothing lookahead) and judges
the result against thresholds PRE-REGISTERED below, before any data is seen —
the same discipline that gave DIAMOND a clean "no" instead of years of doubt.

WHY THIS IS NOT evaluate_model.py
─────────────────────────────────
evaluate_model.py reads BERYL's scan_journal (98 equities, never SPY/QQQ) and
correlates RAW confidence with forward returns. Raw confidence is regime-blind:
a 0.95 BEAR and a 0.95 BULL both score "high" but mean opposite things, so the
Spearman is near-meaningless across mixed regimes. This harness:
  1. Fits the HMM on the INDEX itself (SPY/QQQ), walk-forward, no lookahead.
  2. Uses a SIGNED predictor: +conf (BULL) / -conf (BEAR) / 0 (CHOP).
  3. Adds a VOLATILITY-regime separation test — the signal that actually
     matters for a range-pricer (the memo's "winning cell").

METHOD (production-faithful, mirrors backfill_scan_journal.py)
─────────────────────────────────────────────────────────────
  • Quarterly expanding-window refit: train on data strictly < quarter_start,
    capped to a 365-day lookback (matches BERYL/CITRINE production).
  • CAUSAL daily decode: for each day t in the quarter, predict on features
    UP TO t only and take the last row. At the series endpoint the smoothed
    posterior equals the filtered posterior, so this is exactly what LIVE
    trading sees — no forward-backward leakage from days after t.
  • Forward returns / forward realized vol joined by date; tail rows with no
    t+h data are dropped (standard).

Usage:
  python evaluate_index_ic.py                       # SPY + QQQ, 5y, primary spec
  python evaluate_index_ic.py --tickers SPY         # single index
  python evaluate_index_ic.py --years 6
  python evaluate_index_ic.py --feature-set base    # SECONDARY robustness only
  python evaluate_index_ic.py --self-check          # validate the math, no network
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ════════════════════════════════════════════════════════════════════════════
#  PRE-REGISTERED GATES  — committed BEFORE any data is observed.
#  Do not edit these after seeing results. (DIAMOND-culture: thresholds first.)
# ════════════════════════════════════════════════════════════════════════════
PRIMARY_HORIZON = 1                 # T+1 is the decision horizon for direction
DIR_HORIZONS = [1, 3, 5]           # forward-return horizons (trading days)
VOL_HORIZONS = [5, 10]            # forward realized-vol windows (trading days)

# Directional gate (signed-IC at PRIMARY_HORIZON):
DIR_IC_SIGNAL = 0.05              # |IC| >= 0.05 & p<0.05  -> build a direction layer
DIR_IC_WEAK = 0.03               # 0.03<=|IC|<0.05 & p<0.10 -> marginal, revisit
DIR_P_SIGNAL = 0.05
DIR_P_WEAK = 0.10

# Volatility gate (regime -> forward realized vol, at VOL_HORIZONS[0]):
VOL_KW_P = 0.05                  # Kruskal-Wallis across BULL/BEAR/CHOP
VOL_RATIO_SIGNAL = 1.30          # max/min median fwd-vol ratio -> range-pricer viable
VOL_RATIO_WEAK = 1.15

# Pre-registered model spec (NOT swept — committed). extended_v2 is the
# A/B-validated production winner; n_states=4 suits an index (fewer regimes,
# fewer obs/window than a single volatile stock).
PRIMARY_SPEC = {"n_states": 4, "feature_set": "extended_v2", "cov_type": "diag"}

# Walk-forward params (match production)
HISTORY_YEARS_DEFAULT = 5
TRAINING_LOOKBACK_DAYS = 365
MIN_TRAINING_OBS = 200
DECODE_TRAILING_ROWS = 320       # causal-decode window per day (>=252 + warmup)

# Lazy imports of the trading stack (kept out of --self-check path)
def _load_stack():
    import config
    from src.data_fetcher import build_hmm_features
    from src.ensemble import EnsembleHMM
    from src.hmm_model import HMMRegimeModel
    from src.indicators import attach_all
    from walk_forward_ndx import fetch_equity_daily
    return config, build_hmm_features, EnsembleHMM, HMMRegimeModel, attach_all, fetch_equity_daily


# ── Data prep ───────────────────────────────────────────────────────────────
def prepare_prices(fetch_equity_daily, ticker: str, years: int) -> pd.DataFrame:
    """Fetch daily OHLCV (capitalized cols), dedupe, tz-strip, sort."""
    df = fetch_equity_daily(ticker, years=years)
    if df is None or len(df) == 0:
        raise RuntimeError(f"No data for {ticker}")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def build_quarterly_windows(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """All (q_start, q_end) decode windows between start and end, chronological."""
    windows = []
    q_month = ((start.month - 1) // 3) * 3 + 1   # snap to 1/4/7/10
    cur = datetime(start.year, q_month, 1)
    while cur <= end:
        nm = cur.month + 3                        # advance one quarter
        ny = cur.year + (nm - 1) // 12
        nm = ((nm - 1) % 12) + 1
        nxt = datetime(ny, nm, 1)
        windows.append((cur, min(nxt - timedelta(days=1), end)))
        cur = nxt
    return windows


# ── Walk-forward causal decode ──────────────────────────────────────────────
def decode_index(ticker: str, years: int, spec: dict, stack) -> pd.DataFrame:
    """
    Returns per-day DataFrame indexed by date with columns:
        regime_cat, confidence, Close   (causal — no lookahead)
    """
    config, build_hmm_features, EnsembleHMM, HMMRegimeModel, attach_all, fetch_equity_daily = stack

    feature_set = spec["feature_set"]
    feature_cols = config.FEATURE_SETS.get(feature_set, config.FEATURE_SETS["base"])
    n_states = int(spec["n_states"])
    cov_type = spec["cov_type"]

    px = prepare_prices(fetch_equity_daily, ticker, years)
    # Causal features: every rolling stat at row t uses only data <= t.
    feats_full = attach_all(build_hmm_features(px))

    windows = build_quarterly_windows(px.index.min().to_pydatetime(),
                                      px.index.max().to_pydatetime())
    out_rows = []
    n_fallback = 0
    for q_start, q_end in windows:
        train = feats_full[feats_full.index < q_start]
        if len(train) < MIN_TRAINING_OBS:
            continue
        train = train[train.index >= (q_start - timedelta(days=TRAINING_LOOKBACK_DAYS))]
        train = train.dropna(subset=feature_cols)
        if len(train) < MIN_TRAINING_OBS:
            continue

        # Fit ensemble; fall back to single 4-state diag HMM (production pattern)
        model, used_fb = _fit_with_fallback(
            train, feature_cols, n_states, cov_type, EnsembleHMM, HMMRegimeModel
        )
        if model is None:
            continue
        n_fallback += int(used_fb)

        decode_days = feats_full[(feats_full.index >= q_start) &
                                 (feats_full.index <= q_end)].index
        for t in decode_days:
            # CAUSAL: features strictly up to t, trailing window, predict, take last row
            sl = feats_full[feats_full.index <= t].dropna(subset=feature_cols)
            if len(sl) < 30:
                continue
            sl = sl.iloc[-DECODE_TRAILING_ROWS:]
            try:
                pred = model.predict(sl)
            except Exception:
                continue
            last = pred.iloc[-1]
            out_rows.append({
                "date": t,
                "regime_cat": str(last.get("regime_cat", "UNKNOWN")),
                "confidence": float(last.get("confidence", np.nan)),
                "Close": float(px.loc[t, "Close"]) if t in px.index else np.nan,
            })

    res = pd.DataFrame(out_rows).dropna(subset=["confidence", "Close"])
    res = res.drop_duplicates(subset="date").set_index("date").sort_index()
    res.attrs["n_fallback"] = n_fallback
    res.attrs["n_windows"] = len(windows)
    return res


def _fit_with_fallback(train, feature_cols, n_states, cov_type, EnsembleHMM, HMMRegimeModel):
    try:
        n_list = [max(2, n_states - 1), n_states, n_states + 1]
        ens = EnsembleHMM(n_states_list=n_list, cov_type=cov_type, feature_cols=feature_cols)
        ens.fit(train)
        if getattr(ens, "converged", False):
            return ens, False
    except Exception:
        pass
    try:
        fb = HMMRegimeModel(n_states=4, cov_type="diag",
                            feature_cols=["log_return", "price_range", "volume_change"])
        fb.fit(train)
        return fb, True
    except Exception:
        return None, False


# ── Signal construction ─────────────────────────────────────────────────────
def attach_signals(decoded: pd.DataFrame) -> pd.DataFrame:
    """Signed directional predictor + forward returns + forward realized vol."""
    df = decoded.copy()
    # Signed predictor: regime-aware. This is the corrected IC input.
    sign = df["regime_cat"].map({"BULL": 1.0, "BEAR": -1.0, "CHOP": 0.0}).fillna(0.0)
    df["signed_signal"] = sign * df["confidence"]

    close = df["Close"]
    logret = np.log(close / close.shift(1))
    for h in DIR_HORIZONS:
        df[f"fwd_ret_T{h}"] = close.shift(-h) / close - 1.0
    for k in VOL_HORIZONS:
        # forward realized vol = std of daily log returns over (t, t+k], annualized
        vals = []
        lr = logret.values
        n = len(df)
        for i in range(n):
            seg = lr[i + 1:i + 1 + k]
            vals.append(np.std(seg, ddof=1) * np.sqrt(252) if len(seg) == k else np.nan)
        df[f"fwd_vol_{k}"] = vals
    return df


# ── Metrics ─────────────────────────────────────────────────────────────────
def directional_ic(df: pd.DataFrame, horizon: int) -> dict:
    col = f"fwd_ret_T{horizon}"
    v = df.dropna(subset=[col, "signed_signal"])
    if len(v) < 20 or v["signed_signal"].nunique() < 3:
        return {"ic": np.nan, "p": np.nan, "n": len(v)}
    ic, p = stats.spearmanr(v["signed_signal"], v[col])
    return {"ic": float(ic), "p": float(p), "n": int(len(v))}


def regime_return_table(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    col = f"fwd_ret_T{horizon}"
    rows = []
    for r in ["BULL", "CHOP", "BEAR"]:
        s = df[(df["regime_cat"] == r)][col].dropna()
        if len(s):
            rows.append({"regime": r, "n": len(s), "mean_%": s.mean() * 100,
                         "median_%": s.median() * 100, "hit_%": (s > 0).mean() * 100})
    return pd.DataFrame(rows)


def vol_separation(df: pd.DataFrame, k: int) -> dict:
    col = f"fwd_vol_{k}"
    groups, medians = {}, {}
    for r in ["BULL", "CHOP", "BEAR"]:
        s = df[(df["regime_cat"] == r)][col].dropna()
        if len(s) >= 10:
            groups[r] = s.values
            medians[r] = float(np.median(s))
    if len(groups) < 2:
        return {"kw_p": np.nan, "ratio": np.nan, "medians": medians, "n": 0,
                "bear_bull_p": np.nan}
    kw_stat, kw_p = stats.kruskal(*groups.values())
    ratio = max(medians.values()) / min(medians.values()) if min(medians.values()) > 0 else np.nan
    bear_bull_p = np.nan
    if "BEAR" in groups and "BULL" in groups:
        _, bear_bull_p = stats.mannwhitneyu(groups["BEAR"], groups["BULL"], alternative="two-sided")
    return {"kw_p": float(kw_p), "ratio": float(ratio), "medians": medians,
            "n": int(sum(len(g) for g in groups.values())), "bear_bull_p": float(bear_bull_p)}


# ── Verdicts (mechanical comparison to PRE-REGISTERED gates) ─────────────────
def dir_verdict(ic: float, p: float) -> str:
    if np.isnan(ic):
        return "N/A"
    a = abs(ic)
    if a >= DIR_IC_SIGNAL and p < DIR_P_SIGNAL:
        return "SIGNAL"
    if a >= DIR_IC_WEAK and p < DIR_P_WEAK:
        return "WEAK"
    return "DEAD"


def vol_verdict(kw_p: float, ratio: float) -> str:
    if np.isnan(kw_p) or np.isnan(ratio):
        return "N/A"
    if kw_p < VOL_KW_P and ratio >= VOL_RATIO_SIGNAL:
        return "SIGNAL"
    if kw_p < VOL_KW_P and ratio >= VOL_RATIO_WEAK:
        return "WEAK"
    return "DEAD"


# ── Reporting ───────────────────────────────────────────────────────────────
def run_ticker(ticker: str, years: int, spec: dict, stack) -> dict:
    print(f"\n{'─'*72}\n  {ticker}  —  decoding regimes (walk-forward, causal)\n{'─'*72}")
    t0 = time.time()
    decoded = decode_index(ticker, years, spec, stack)
    if decoded.empty:
        print(f"  {ticker}: no decoded rows — skipped.")
        return {}
    df = attach_signals(decoded)
    dist = df["regime_cat"].value_counts().to_dict()
    print(f"  decoded {len(df)} days  ({df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}) "
          f"in {time.time()-t0:.0f}s")
    print(f"  regime mix: " + ", ".join(f"{k}={v}" for k, v in dist.items())
          + f"  | fallback windows: {decoded.attrs.get('n_fallback',0)}/{decoded.attrs.get('n_windows','?')}")

    # 1. Directional IC (signed)
    print("\n  DIRECTIONAL — signed-IC  Spearman(+conf BULL/-conf BEAR/0 CHOP, fwd ret)")
    dir_res = {}
    for h in DIR_HORIZONS:
        r = directional_ic(df, h)
        dir_res[h] = r
        star = "***" if r["p"] < 0.01 else "**" if r["p"] < 0.05 else "*" if r["p"] < 0.10 else ""
        flag = f"  [{dir_verdict(r['ic'], r['p'])}]" if h == PRIMARY_HORIZON else ""
        print(f"    T+{h}: IC={r['ic']:+.4f}  p={r['p']:.4f}{star}  N={r['n']}{flag}")

    # Regime forward-return table (interpretability) at primary horizon
    print(f"\n  regime → T+{PRIMARY_HORIZON} forward return:")
    rt = regime_return_table(df, PRIMARY_HORIZON)
    for _, row in rt.iterrows():
        print(f"    {row['regime']:4s}  N={int(row['n']):4d}  mean={row['mean_%']:+.3f}%  "
              f"median={row['median_%']:+.3f}%  hit={row['hit_%']:.1f}%")

    # 2. Volatility-regime separation
    print("\n  VOLATILITY — regime → forward realized vol (annualized)")
    vol_res = {}
    for k in VOL_HORIZONS:
        v = vol_separation(df, k)
        vol_res[k] = v
        med_str = ", ".join(f"{r}={v['medians'].get(r, float('nan')):.3f}"
                            for r in ["BULL", "CHOP", "BEAR"] if r in v["medians"])
        flag = f"  [{vol_verdict(v['kw_p'], v['ratio'])}]" if k == VOL_HORIZONS[0] else ""
        print(f"    fwd_{k}d: KW p={v['kw_p']:.4f}  max/min ratio={v['ratio']:.2f}  "
              f"(BEARvBULL p={v['bear_bull_p']:.4f})  medians[{med_str}]{flag}")

    return {"ticker": ticker, "n_days": len(df), "regime_mix": dist,
            "directional": dir_res, "volatility": vol_res,
            "dir_verdict": dir_verdict(dir_res[PRIMARY_HORIZON]["ic"], dir_res[PRIMARY_HORIZON]["p"]),
            "vol_verdict": vol_verdict(vol_res[VOL_HORIZONS[0]]["kw_p"], vol_res[VOL_HORIZONS[0]]["ratio"])}


def print_preregistered():
    print("═" * 72)
    print("  INDEX REGIME IC GATE  —  PRE-REGISTERED THRESHOLDS (committed)")
    print("═" * 72)
    print(f"  Spec (not swept): {PRIMARY_SPEC}")
    print(f"  DIRECTIONAL (signed-IC @ T+{PRIMARY_HORIZON}):")
    print(f"     SIGNAL  |IC|>={DIR_IC_SIGNAL:.2f} & p<{DIR_P_SIGNAL}   → build a direction calibration layer")
    print(f"     WEAK    |IC|>={DIR_IC_WEAK:.2f} & p<{DIR_P_WEAK}    → marginal, revisit")
    print(f"     DEAD    otherwise                  → no directional edge")
    print(f"  VOLATILITY (regime→fwd vol @ {VOL_HORIZONS[0]}d):")
    print(f"     SIGNAL  KW p<{VOL_KW_P} & ratio>={VOL_RATIO_SIGNAL}  → range-pricer fork viable")
    print(f"     WEAK    KW p<{VOL_KW_P} & ratio>={VOL_RATIO_WEAK}  → modest separation")
    print(f"     DEAD    otherwise                  → regime doesn't separate vol")


# ── Self-check (validates the math; no network) ─────────────────────────────
def self_check() -> bool:
    print("Running self-check (synthetic data, no network)...")
    ok = True
    rng = np.random.RandomState(0)

    # 1. Perfect signed predictor → IC ≈ +1
    n = 400
    sig = rng.uniform(-1, 1, n)
    df = pd.DataFrame({"signed_signal": sig, "fwd_ret_T1": sig * 0.01 + rng.normal(0, 1e-4, n)})
    r = directional_ic(df, 1)
    print(f"  [1] perfect predictor: IC={r['ic']:+.3f} (expect ~+1)")
    ok &= r["ic"] > 0.9

    # 2. Pure noise → |IC| small
    df2 = pd.DataFrame({"signed_signal": rng.uniform(-1, 1, n),
                        "fwd_ret_T1": rng.normal(0, 0.01, n)})
    r2 = directional_ic(df2, 1)
    print(f"  [2] noise predictor:   IC={r2['ic']:+.3f} (expect ~0)")
    ok &= abs(r2["ic"]) < 0.15

    # 3. Vol separation: BEAR high-vol, BULL low-vol → KW sig, ratio>1.3
    bull = pd.DataFrame({"regime_cat": "BULL", "fwd_vol_5": rng.normal(0.10, 0.01, 150).clip(0.01)})
    bear = pd.DataFrame({"regime_cat": "BEAR", "fwd_vol_5": rng.normal(0.25, 0.03, 150).clip(0.01)})
    chop = pd.DataFrame({"regime_cat": "CHOP", "fwd_vol_5": rng.normal(0.15, 0.02, 150).clip(0.01)})
    dvol = pd.concat([bull, bear, chop], ignore_index=True)
    v = vol_separation(dvol, 5)
    print(f"  [3] vol-separated:     KW p={v['kw_p']:.2e}  ratio={v['ratio']:.2f} (expect p<.05, ratio>1.3)")
    ok &= (v["kw_p"] < 0.05 and v["ratio"] > 1.3)

    # 4. Vol NOT separated → DEAD
    flat = pd.concat([
        pd.DataFrame({"regime_cat": r, "fwd_vol_5": rng.normal(0.15, 0.02, 150).clip(0.01)})
        for r in ["BULL", "BEAR", "CHOP"]], ignore_index=True)
    vf = vol_separation(flat, 5)
    print(f"  [4] vol-flat:          ratio={vf['ratio']:.2f}  verdict={vol_verdict(vf['kw_p'], vf['ratio'])} (expect DEAD/WEAK)")
    ok &= vol_verdict(vf["kw_p"], vf["ratio"]) in ("DEAD", "WEAK")

    print(f"\n  SELF-CHECK: {'PASS ✓' if ok else 'FAIL ✗'}")
    return ok


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Index Regime IC Gate")
    ap.add_argument("--tickers", default="SPY,QQQ", help="comma-separated index proxies")
    ap.add_argument("--years", type=int, default=HISTORY_YEARS_DEFAULT)
    ap.add_argument("--feature-set", default=None, help="SECONDARY robustness override (marks non-primary)")
    ap.add_argument("--n-states", type=int, default=None, help="SECONDARY robustness override")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--out", default="index_ic_results.json")
    args = ap.parse_args()

    if args.self_check:
        sys.exit(0 if self_check() else 1)

    # quick self-check on the math before spending compute
    if not self_check():
        print("Self-check failed — aborting before network/compute.")
        sys.exit(1)
    print()

    spec = dict(PRIMARY_SPEC)
    secondary = False
    if args.feature_set:
        spec["feature_set"] = args.feature_set; secondary = True
    if args.n_states:
        spec["n_states"] = args.n_states; secondary = True

    print_preregistered()
    if secondary:
        print(f"\n  ⚠ SECONDARY RUN (spec overridden: {spec}) — NOT the pre-registered primary. "
              "Robustness only; do not use to overturn the primary verdict.")

    stack = _load_stack()
    results = []
    for tk in [t.strip().upper() for t in args.tickers.split(",") if t.strip()]:
        try:
            r = run_ticker(tk, args.years, spec, stack)
            if r:
                results.append(r)
        except Exception as e:
            import traceback
            print(f"  {tk}: FAILED — {type(e).__name__}: {e}")
            traceback.print_exc()

    # Summary verdict
    print("\n" + "═" * 72)
    print("  VERDICT (vs pre-registered gates)")
    print("═" * 72)
    for r in results:
        print(f"  {r['ticker']:4s}  DIRECTION: {r['dir_verdict']:6s}   VOLATILITY: {r['vol_verdict']:6s}   (N={r['n_days']} days)")
    print()
    any_dir = any(r["dir_verdict"] == "SIGNAL" for r in results)
    any_vol = any(r["vol_verdict"] == "SIGNAL" for r in results)
    if any_vol:
        print("  → VOLATILITY signal present: a regime-conditioned range-pricer for Kalshi")
        print("    financial markets is worth scoping (the memo's 'winning cell').")
    if any_dir:
        print("  → DIRECTIONAL signal present: a direction calibration layer is justified.")
    if not (any_dir or any_vol):
        print("  → No signal on either axis at the pre-registered bar. The HMM does not")
        print("    carry index-level edge; do NOT build the range-pricer on it. Clean no.")

    Path(args.out).write_text(json.dumps(
        {"spec": spec, "secondary": secondary, "years": args.years,
         "gates": {"DIR_IC_SIGNAL": DIR_IC_SIGNAL, "VOL_RATIO_SIGNAL": VOL_RATIO_SIGNAL,
                   "VOL_KW_P": VOL_KW_P, "primary_horizon": PRIMARY_HORIZON},
         "results": results}, indent=2, default=str))
    print(f"\n  Results written → {args.out}")


if __name__ == "__main__":
    main()
