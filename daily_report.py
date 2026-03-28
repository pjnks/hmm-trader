"""
daily_report.py
───────────────
Automated daily report for AGATE + BERYL + CITRINE.

Run daily (or anytime) to get instant status on all projects:
  python daily_report.py

Flags:
  --snapshot      Save maturity scores to history (runs via midnight cron)
  --checkin       Interactive 5-min accountability check-in
  --report        Standard report output (default)

Pulls data from:
  - paper_trades.db (AGATE live trades)
  - beryl_optimization_results.csv (BERYL optimization)
  - citrine_trades.db (CITRINE portfolio trades)
  - citrine_wf_results.csv (CITRINE backtest results)
  - maturity_history.csv (score snapshots over time)
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent


# ── Project Maturity Scoring ──────────────────────────────────────────────
# Each milestone adds points. Max = 100%.
# Stages: Concept(10) → Backtest(15) → Optimization(15) → Infrastructure(15)
#       → Live Micro(15) → Validated(15) → Scaled(10) → Production(5)

def _agate_score() -> tuple[int, str]:
    """Calculate AGATE maturity score based on milestones achieved."""
    score = 0
    stage = "Concept"

    # Concept (0-10): Strategy designed
    score += 10; stage = "Concept ✅"

    # Backtest (10-25): Walk-forward validated
    score += 15; stage = "Backtest ✅"

    # Optimization (25-40): Ensemble winner found (Sharpe 6.240)
    score += 15; stage = "Optimization ✅"

    # Infrastructure (40-55): Live harness built + tested
    harness_exists = (ROOT / "live_trading.py").exists()
    broker_exists = (ROOT / "src" / "live_broker.py").exists()
    notifier_exists = (ROOT / "src" / "notifier.py").exists()
    if harness_exists and broker_exists and notifier_exists:
        score += 15; stage = "Infrastructure ✅"

    # Live Micro (55-70): Running live (test or real)
    # Check if live_trading process has been launched (db exists = went live)
    db_path = ROOT / "paper_trades.db"
    if db_path.exists():
        score += 7; stage = "Live (Micro) 🔄"

        # Extra points for actual trades
        try:
            with sqlite3.connect(db_path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
                if count > 0:
                    score += 4; stage = "Live (Micro) 🔄 — trading"
                if count >= 10:
                    score += 4; stage = "Live (Micro) 🔄 — 10+ trades"
        except Exception:
            pass

    # Validated (70-85): 4+ weeks profitable, Sharpe > 0.5
    # (checked dynamically from trade history when enough data exists)

    # Scaled (85-95): Increased leverage
    # Production (95-100): Fully autonomous

    return score, stage


def _beryl_score() -> tuple[int, str]:
    """Calculate BERYL maturity score based on milestones achieved."""
    score = 0
    stage = "Concept"

    # Concept (0-10)
    score += 10; stage = "Concept ✅"

    # Backtest (10-25): Walk-forward framework exists
    if (ROOT / "walk_forward_ndx.py").exists():
        score += 5; stage = "Backtest 🔄"

    # Optimization (25-40): Running or completed
    csv_path = ROOT / "beryl_optimization_results.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            trials = len(df)
            if trials > 0:
                score += 3; stage = "Optimization 🔄"
            if trials >= 100:
                score += 5; stage = "Optimization 🔄"
            if trials >= 200:
                score += 7; stage = "Optimization ✅"

            # Bonus: any Sharpe > 1.0?
            if (df["wf_sharpe"] > 1.0).any():
                score += 5; stage = "Optimization ✅ (winner found!)"
        except Exception:
            pass

    return score, stage


def _citrine_score() -> tuple[int, str]:
    """Calculate CITRINE maturity score based on milestones achieved."""
    score = 0
    stage = "Concept"

    # Concept (0-10): Scanner + Allocator designed
    scanner_exists = (ROOT / "src" / "citrine_scanner.py").exists()
    allocator_exists = (ROOT / "src" / "citrine_allocator.py").exists()
    if scanner_exists and allocator_exists:
        score += 10; stage = "Concept ✅"

    # Backtest (10-25): Walk-forward backtester exists + has results
    backtest_exists = (ROOT / "citrine_backtest.py").exists()
    if backtest_exists:
        score += 5; stage = "Backtest 🔄"

    wf_csv = ROOT / "citrine_wf_results.csv"
    if wf_csv.exists():
        try:
            df = pd.read_csv(wf_csv)
            if len(df) > 0:
                score += 5; stage = "Backtest 🔄 — results exist"
            # Bonus: any positive Sharpe window?
            sharpe_col = "sharpe_ratio" if "sharpe_ratio" in df.columns else "sharpe"
            if sharpe_col in df.columns and (df[sharpe_col] > 0).any():
                score += 5; stage = "Backtest ✅"
        except Exception:
            pass

    # Optimization (25-40): Optimizer exists + per-ticker configs
    optimizer_exists = (ROOT / "optimize_citrine.py").exists()
    if optimizer_exists:
        score += 5; stage = "Optimization 🔄"

    per_ticker_configs = ROOT / "citrine_per_ticker_configs.json"
    if per_ticker_configs.exists():
        try:
            import json as _json
            with open(per_ticker_configs) as f:
                configs = _json.load(f)
            if len(configs) >= 10:
                score += 5; stage = "Optimization ✅"
            elif len(configs) > 0:
                score += 3; stage = "Optimization 🔄"
        except Exception:
            pass

    # Infrastructure (40-55): Live engine + dashboard exist
    live_exists = (ROOT / "live_trading_citrine.py").exists()
    dash_exists = (ROOT / "citrine_dashboard.py").exists()
    if live_exists:
        score += 8; stage = "Infrastructure 🔄"
    if live_exists and dash_exists:
        score += 7; stage = "Infrastructure ✅"

    # Live Micro (55-70): Running live (test or real)
    db_path = ROOT / "citrine_trades.db"
    if db_path.exists():
        score += 5; stage = "Live (Micro) 🔄"
        try:
            with sqlite3.connect(db_path) as conn:
                trade_count = conn.execute(
                    "SELECT COUNT(*) FROM trades WHERE action = 'EXIT'"
                ).fetchone()[0]
                if trade_count > 0:
                    score += 5; stage = "Live (Micro) 🔄 — trading"
                if trade_count >= 20:
                    score += 5; stage = "Live (Micro) 🔄 — 20+ trades"
        except Exception:
            pass

    # Validated (70-85), Scaled (85-95), Production (95-100): future

    return min(score, 100), stage


def _score_bar(score: int, width: int = 20) -> str:
    """Generate visual progress bar."""
    filled = int(score / 100 * width)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    return f"[{bar}] {score}%"


# ── Maturity Snapshot ─────────────────────────────────────────────────────

def save_snapshot() -> None:
    """Save daily maturity scores to maturity_history.csv for tracking over time."""
    csv_path = ROOT / "maturity_history.csv"
    today = datetime.now().strftime("%Y-%m-%d")

    agate_score, agate_stage = _agate_score()
    beryl_score, beryl_stage = _beryl_score()
    citrine_score, citrine_stage = _citrine_score()

    # Check if today already has an entry
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if today in df["date"].values:
            # Update today's row
            df.loc[df["date"] == today, "agate_score"] = agate_score
            df.loc[df["date"] == today, "agate_stage"] = agate_stage
            df.loc[df["date"] == today, "beryl_score"] = beryl_score
            df.loc[df["date"] == today, "beryl_stage"] = beryl_stage
            df.loc[df["date"] == today, "citrine_score"] = citrine_score
            df.loc[df["date"] == today, "citrine_stage"] = citrine_stage
            df.to_csv(csv_path, index=False)
            print(f"Updated snapshot for {today}: AGATE {agate_score}%, BERYL {beryl_score}%, CITRINE {citrine_score}%")
            return

    # Append new row
    row = {
        "date": today,
        "agate_score": agate_score,
        "agate_stage": agate_stage,
        "beryl_score": beryl_score,
        "beryl_stage": beryl_stage,
        "citrine_score": citrine_score,
        "citrine_stage": citrine_stage,
    }

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Snapshot saved for {today}: AGATE {agate_score}%, BERYL {beryl_score}%, CITRINE {citrine_score}%")


# ── Accountability Check-In ───────────────────────────────────────────────

def run_checkin() -> None:
    """
    Interactive 5-minute accountability check-in.

    Forces you to actively acknowledge each step.
    Logs completion to checkin_log.csv so you can see your streak.
    """
    import subprocess

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M")

    print(f"\n{'='*60}")
    print(f"  5-MIN DAILY CHECK-IN — {date_str}")
    print(f"{'='*60}\n")

    # Show the report first
    agate_score, agate_stage = _agate_score()
    beryl_score, beryl_stage = _beryl_score()
    citrine_score, citrine_stage = _citrine_score()

    print("┌─ PROJECT MATURITY")
    print(f"│  AGATE   (SOL)   {_score_bar(agate_score)}")
    print(f"│                  Stage: {agate_stage}")
    print(f"│  BERYL   (NDX)   {_score_bar(beryl_score)}")
    print(f"│                  Stage: {beryl_stage}")
    print(f"│  CITRINE (Port)  {_score_bar(citrine_score)}")
    print(f"│                  Stage: {citrine_stage}")
    print("│")
    print(_agate_report())
    print(_beryl_report())
    print(_citrine_report())
    print(f"\n{'='*60}\n")

    # ── Build dynamic checklist with live data ─────────────────
    # Step 1: Maturity trend
    history_path = ROOT / "maturity_history.csv"
    maturity_detail = (f"AGATE: {agate_score}% ({agate_stage}) | BERYL: {beryl_score}% ({beryl_stage})"
                       f" | CITRINE: {citrine_score}% ({citrine_stage})")
    if history_path.exists():
        try:
            hist = pd.read_csv(history_path)
            if len(hist) >= 2:
                prev = hist.iloc[-1]
                a_delta = agate_score - int(prev["agate_score"])
                b_delta = beryl_score - int(prev["beryl_score"])
                a_arrow = f"+{a_delta}" if a_delta > 0 else (str(a_delta) if a_delta < 0 else "=")
                b_arrow = f"+{b_delta}" if b_delta > 0 else (str(b_delta) if b_delta < 0 else "=")
                maturity_detail += f"\\nTrend: AGATE {a_arrow}% | BERYL {b_arrow}% vs yesterday"
        except Exception:
            pass

    # Step 2: AGATE trade activity
    agate_detail = "No trades yet"
    db_path = ROOT / "paper_trades.db"
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
                if count > 0:
                    row_data = conn.execute(
                        "SELECT SUM(pnl), AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) * 100 FROM trades"
                    ).fetchone()
                    total_pnl, win_rate = row_data
                    agate_detail = f"{count} trades | P&L: ${total_pnl:.2f} | Win rate: {win_rate:.0f}%"
                else:
                    agate_detail = "0 trades — waiting for 8/8 confirmations"
        except Exception:
            pass

    # Step 3: BERYL optimization (live data, not a question)
    beryl_detail = "No results yet"
    csv_path = ROOT / "beryl_optimization_results.csv"
    if csv_path.exists():
        try:
            bdf = pd.read_csv(csv_path)
            trials = len(bdf)
            above_1 = (bdf["wf_sharpe"] > 1.0).sum()
            best_sharpe = bdf["wf_sharpe"].max()
            best_ticker = bdf.loc[bdf["wf_sharpe"].idxmax(), "ticker"]
            if above_1 > 0:
                beryl_detail = f"🏆 {above_1} configs with Sharpe > 1.0! Best: {best_ticker} ({best_sharpe:.3f})"
            else:
                beryl_detail = f"{trials} trials | 0 configs > 1.0 | Best: {best_ticker} ({best_sharpe:.3f})"
        except Exception:
            pass

    # Step 4: Kill-switch status
    kill_status = "✅ All clear — no kill-switch triggers"
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as conn:
                pnl_sum = conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades").fetchone()[0]
                if pnl_sum < -200:
                    kill_status = "🚨 TRIGGERED — cumulative loss > 2%"
        except Exception:
            pass

    # Step for CITRINE portfolio
    citrine_detail = "Not started yet"
    citrine_db = ROOT / "citrine_trades.db"
    if citrine_db.exists():
        try:
            with sqlite3.connect(citrine_db) as conn:
                snap = conn.execute(
                    "SELECT total_equity, num_positions, bull_count "
                    "FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if snap:
                    eq, npos, bulls = snap
                    pnl = eq - 25000
                    citrine_detail = (f"Equity: ${eq:,.0f} ({pnl:+,.0f}) | "
                                     f"{npos} positions | {bulls} BULL tickers")
                exit_count = conn.execute(
                    "SELECT COUNT(*) FROM trades WHERE action='EXIT'"
                ).fetchone()[0]
                if exit_count > 0:
                    citrine_detail += f" | {exit_count} closed trades"
        except Exception:
            pass

    steps = [
        ("1/6", "Maturity scores", maturity_detail),
        ("2/6", "AGATE trade activity", agate_detail),
        ("3/6", "BERYL optimization", beryl_detail),
        ("4/6", "CITRINE portfolio", citrine_detail),
        ("5/6", "Kill-switch status", kill_status),
        ("6/6", "Decision: Continue / Investigate / Pause", "What's your gut say?"),
    ]

    responses = {}
    timestamps = {}
    all_acknowledged = True

    for step_num, step_title, step_question in steps:
        print(f"\n  [{step_num}] {step_title}")
        print(f"        {step_question}")

        # macOS dialog for each step
        try:
            # Step 5 gets special decision buttons
            if step_num == "6/6":
                result = subprocess.run(
                    [
                        "osascript", "-e",
                        f'display dialog "[{step_num}] {step_title}\\n\\n{step_question}" '
                        f'buttons {{"⏸ Pause", "🔍 Investigate", "✅ Continue"}} default button "✅ Continue" '
                        f'with title "AGATE/BERYL/CITRINE Check-In" with icon note',
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                action_time = datetime.now().strftime("%H:%M:%S")
                output = result.stdout.strip()
                if "Continue" in output:
                    responses[step_num] = "continue"
                    timestamps[step_num] = action_time
                    print(f"        ✅ Continue at {action_time}")
                elif "Investigate" in output:
                    responses[step_num] = "investigate"
                    timestamps[step_num] = action_time
                    print(f"        🔍 Investigate at {action_time}")
                else:
                    responses[step_num] = "pause"
                    timestamps[step_num] = action_time
                    print(f"        ⏸ Pause at {action_time}")
            else:
                result = subprocess.run(
                    [
                        "osascript", "-e",
                        f'display dialog "[{step_num}] {step_title}\\n\\n{step_question}" '
                        f'buttons {{"Skip", "✅ Acknowledged"}} default button "✅ Acknowledged" '
                        f'with title "AGATE/BERYL/CITRINE Check-In" with icon note',
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                action_time = datetime.now().strftime("%H:%M:%S")
                output = result.stdout.strip()
                if "Acknowledged" in output:
                    responses[step_num] = "acknowledged"
                    timestamps[step_num] = action_time
                    print(f"        ✅ Acknowledged at {action_time}")
                else:
                    responses[step_num] = "skipped"
                    timestamps[step_num] = action_time
                    print(f"        ⏭️  Skipped at {action_time}")
                    all_acknowledged = False
        except subprocess.TimeoutExpired:
            responses[step_num] = "timeout"
            timestamps[step_num] = datetime.now().strftime("%H:%M:%S")
            print(f"        ⏰ Timed out")
            all_acknowledged = False
        except Exception as e:
            # Fallback to terminal input if osascript fails
            resp = input(f"        Press Enter to acknowledge (or 's' to skip): ").strip().lower()
            action_time = datetime.now().strftime("%H:%M:%S")
            if resp == "s":
                responses[step_num] = "skipped"
                timestamps[step_num] = action_time
                print(f"        ⏭️  Skipped at {action_time}")
                all_acknowledged = False
            else:
                responses[step_num] = "acknowledged"
                timestamps[step_num] = action_time
                print(f"        ✅ Acknowledged at {action_time}")

    # ── Log the check-in ──────────────────────────────────────
    log_path = ROOT / "checkin_log.csv"
    write_header = not log_path.exists()

    row = {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "step_1": responses.get("1/6", "missed"),
        "step_1_at": timestamps.get("1/6", ""),
        "step_2": responses.get("2/6", "missed"),
        "step_2_at": timestamps.get("2/6", ""),
        "step_3": responses.get("3/6", "missed"),
        "step_3_at": timestamps.get("3/6", ""),
        "step_4": responses.get("4/6", "missed"),
        "step_4_at": timestamps.get("4/6", ""),
        "step_5": responses.get("5/6", "missed"),
        "step_5_at": timestamps.get("5/6", ""),
        "step_6": responses.get("6/6", "missed"),
        "step_6_at": timestamps.get("6/6", ""),
        "all_complete": all_acknowledged,
    }

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    # ── Streak calculation ────────────────────────────────────
    if log_path.exists():
        df = pd.read_csv(log_path)
        streak = 0
        for _, r in df.iloc[::-1].iterrows():
            if r.get("all_complete", False):
                streak += 1
            else:
                break

        print(f"\n{'='*60}")
        if all_acknowledged:
            print(f"  ✅ CHECK-IN COMPLETE — Streak: {streak} day{'s' if streak != 1 else ''} 🔥")
        else:
            print(f"  ⚠️  CHECK-IN INCOMPLETE — Some steps skipped")
            print(f"       Previous streak: {streak} day{'s' if streak != 1 else ''}")
        print(f"{'='*60}\n")

    # Send completion notification (macOS + phone)
    try:
        from src.notifier import _macos_notify, _pushover_notify
        decision = responses.get("6/6", "missed")
        if all_acknowledged:
            msg = f"All 5 steps done. Decision: {decision.upper()}. Streak: {streak} days"
            _macos_notify("Check-In Complete", msg)
            _pushover_notify("Check-In Complete", msg, priority=-1)
        else:
            msg = f"Some steps skipped. Decision: {decision.upper()}. Streak: {streak} days"
            _macos_notify("Check-In Incomplete", msg)
            _pushover_notify("Check-In Incomplete", msg, priority=0)
    except Exception:
        pass


# ── Reports ───────────────────────────────────────────────────────────────

def _agate_report() -> str:
    """Generate AGATE (SOL live trading) report."""
    db_path = ROOT / "paper_trades.db"

    lines = []
    lines.append("├─ AGATE (SOL Live)")

    if not db_path.exists():
        lines.append("│  └─ No trades yet (database not created)")
        return "\n".join(lines)

    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC", conn)

        if df.empty:
            lines.append("│  ├─ Trades executed: 0")
            lines.append("│  ├─ Status: ✅ Running (no trades yet)")
            lines.append("│  └─ Next signal: Waiting for 8/8 confirmations")
            return "\n".join(lines)

        # All-time metrics
        total_trades = len(df)
        winning = (df["pnl"] > 0).sum()
        win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
        total_pnl = df["pnl"].sum()

        # Rolling 20-trade Sharpe
        sharpe_20 = 0.0
        if len(df) >= 20:
            last_20 = df.head(20)["pnl"].values
            sharpe_20 = np.mean(last_20) / (np.std(last_20) + 1e-6)

        # Today's metrics
        today = datetime.now(tz=timezone.utc).date().isoformat()
        today_trades = df[df["timestamp"].str.startswith(today)]
        today_pnl = today_trades["pnl"].sum() if not today_trades.empty else 0

        # Kill-switch check
        kill_switch = "No"
        if total_pnl < -200:  # 2% of $10k
            kill_switch = "🚨 YES (cumulative loss > 2%)"
        elif sharpe_20 < 0.3 and len(df) >= 20:
            kill_switch = "🚨 YES (rolling Sharpe < 0.3)"

        lines.append(f"│  ├─ Trades executed: {total_trades}")
        lines.append(f"│  ├─ Win rate: {win_rate:.1f}%")
        lines.append(f"│  ├─ Total P&L: ${total_pnl:.2f}")
        lines.append(f"│  ├─ Today P&L: ${today_pnl:.2f}")
        lines.append(f"│  ├─ Rolling Sharpe (20): {sharpe_20:.3f}")
        lines.append(f"│  ├─ Kill-switch: {kill_switch}")
        lines.append(f"│  └─ Status: {'✅ Running' if kill_switch == 'No' else '🚨 STOPPED'}")

    except Exception as e:
        lines.append(f"│  └─ Error reading DB: {e}")

    return "\n".join(lines)


def _beryl_report() -> str:
    """Generate BERYL (NDX100 optimization) report."""
    csv_path = ROOT / "beryl_optimization_results.csv"

    lines = []
    lines.append("│")
    lines.append("├─ BERYL (NDX100 Optimization)")

    if not csv_path.exists():
        lines.append("│  └─ No results yet (optimization not started or in progress)")
        return "\n".join(lines)

    try:
        df = pd.read_csv(csv_path)

        if df.empty:
            lines.append("│  └─ No valid results yet")
            return "\n".join(lines)

        total_trials = len(df)
        positive_sharpe = (df["wf_sharpe"] > 0).sum()
        high_sharpe = (df["wf_sharpe"] > 1.0).sum()
        best_sharpe = df["wf_sharpe"].max()
        best_row = df.loc[df["wf_sharpe"].idxmax()]
        best_ticker = best_row.get("ticker", "?")
        best_return = best_row.get("wf_return", 0)

        # Per-ticker breakdown
        ticker_best = df.groupby("ticker")["wf_sharpe"].max()

        lines.append(f"│  ├─ Trials completed: {total_trials}")
        lines.append(f"│  ├─ Positive Sharpe: {positive_sharpe} ({positive_sharpe/total_trials*100:.0f}%)")
        lines.append(f"│  ├─ Sharpe > 1.0: {high_sharpe}")
        lines.append(f"│  ├─ Best config: {best_ticker} (Sharpe {best_sharpe:.3f}, Return {best_return:.1f}%)")
        lines.append(f"│  ├─ Best per ticker:")
        for ticker, sharpe in ticker_best.items():
            status = "✅" if sharpe > 0.5 else ("⚠️" if sharpe > 0 else "❌")
            lines.append(f"│  │  {status} {ticker}: {sharpe:.3f}")
        lines.append(f"│  └─ Status: ✅ On track" if high_sharpe > 0 else f"│  └─ Status: ⚠️ No winners yet")

    except Exception as e:
        lines.append(f"│  └─ Error reading CSV: {e}")

    return "\n".join(lines)


def _citrine_report() -> str:
    """Generate CITRINE (NDX100 portfolio rotation) report."""
    db_path = ROOT / "citrine_trades.db"

    lines = []
    lines.append("│")
    lines.append("├─ CITRINE (NDX100 Portfolio Rotation)")

    # Show backtest results even if live trading hasn't started
    wf_csv = ROOT / "citrine_wf_results.csv"
    if not db_path.exists():
        if wf_csv.exists():
            try:
                wf = pd.read_csv(wf_csv)
                sharpe_col = "sharpe_ratio" if "sharpe_ratio" in wf.columns else "sharpe"
                if not wf.empty and sharpe_col in wf.columns:
                    pos_windows = int((wf[sharpe_col] > 0).sum())
                    best_sharpe = wf[sharpe_col].max()
                    total_ret = wf["return_pct"].sum() if "return_pct" in wf.columns else 0
                    total_trades = int(wf["total_trades"].sum()) if "total_trades" in wf.columns else 0
                    lines.append(f"│  ├─ Backtest: {pos_windows}/{len(wf)} positive windows, "
                                 f"best Sharpe {best_sharpe:.3f}")
                    lines.append(f"│  ├─ Total return: {total_ret:+.2f}% | {total_trades} trades")
                    lines.append(f"│  └─ Live trading: Not started yet")
                    return "\n".join(lines)
            except Exception:
                pass
        lines.append("│  └─ Not started yet")
        return "\n".join(lines)

    try:
        with sqlite3.connect(db_path) as conn:
            # Trade stats
            trades_df = pd.read_sql_query(
                "SELECT * FROM trades WHERE action = 'EXIT' AND pnl IS NOT NULL "
                "ORDER BY timestamp DESC", conn,
            )
            all_trades = pd.read_sql_query(
                "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 1", conn,
            )

            # Latest snapshot
            snap_df = pd.read_sql_query(
                "SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1",
                conn,
            )

        if trades_df.empty and snap_df.empty:
            lines.append("│  ├─ Status: Initialized (no trades yet)")
            lines.append("│  └─ Waiting for first daily scan cycle")
            return "\n".join(lines)

        # Portfolio snapshot
        if not snap_df.empty:
            snap = snap_df.iloc[0]
            equity = snap["total_equity"]
            cash = snap["cash"]
            num_pos = snap["num_positions"]
            num_long = snap["num_long"]
            num_short = snap["num_short"]
            bull_count = snap["bull_count"]
            bear_count = snap["bear_count"]

            pnl_from_start = equity - 25000  # CITRINE_INITIAL_CAPITAL
            pnl_pct = (pnl_from_start / 25000) * 100

            lines.append(f"│  ├─ Equity: ${equity:,.2f} ({pnl_pct:+.2f}%)")
            lines.append(f"│  ├─ Cash: ${cash:,.0f} | Positions: {num_pos} "
                         f"({num_long}L / {num_short}S)")
            lines.append(f"│  ├─ Market: {bull_count} BULL / {bear_count} BEAR")

        # Closed trade stats
        if not trades_df.empty:
            total_closed = len(trades_df)
            winning = (trades_df["pnl"] > 0).sum()
            win_rate = (winning / total_closed * 100) if total_closed > 0 else 0
            total_pnl = trades_df["pnl"].sum()
            avg_pnl = trades_df["pnl"].mean()

            # Rolling Sharpe
            sharpe = 0.0
            if total_closed >= 5:
                pnls = trades_df["pnl"].values[:20]
                std = np.std(pnls)
                sharpe = float(np.mean(pnls) / std) if std > 1e-9 else 0.0

            lines.append(f"│  ├─ Closed trades: {total_closed}")
            lines.append(f"│  ├─ Win rate: {win_rate:.1f}%")
            lines.append(f"│  ├─ Total P&L: ${total_pnl:,.2f} (avg ${avg_pnl:.2f})")
            lines.append(f"│  ├─ Rolling Sharpe: {sharpe:.3f}")

            # Kill-switch check
            kill = "No"
            if total_pnl < -(25000 * 0.05):
                kill = "🚨 YES (loss > 5%)"
            elif sharpe < 0.3 and total_closed >= 20:
                kill = "🚨 YES (Sharpe < 0.3)"
            lines.append(f"│  ├─ Kill-switch: {kill}")

        # Backtest results
        wf_csv = ROOT / "citrine_wf_results.csv"
        if wf_csv.exists():
            try:
                wf = pd.read_csv(wf_csv)
                sharpe_col = "sharpe_ratio" if "sharpe_ratio" in wf.columns else "sharpe"
                if not wf.empty and sharpe_col in wf.columns:
                    pos_windows = (wf[sharpe_col] > 0).sum()
                    best_sharpe = wf[sharpe_col].max()
                    avg_return = wf["return_pct"].mean() if "return_pct" in wf.columns else 0
                    lines.append(f"│  ├─ Backtest: {pos_windows}/{len(wf)} positive windows, "
                                 f"best Sharpe {best_sharpe:.3f}")
            except Exception:
                pass

        lines.append(f"│  └─ Status: ✅ Active")

    except Exception as e:
        lines.append(f"│  └─ Error reading DB: {e}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="AGATE/BERYL/CITRINE Daily Report")
    parser.add_argument("--snapshot", action="store_true", help="Save maturity snapshot to history")
    parser.add_argument("--checkin", action="store_true", help="Interactive 5-min accountability check-in")
    args = parser.parse_args()

    if args.snapshot:
        save_snapshot()
        return

    if args.checkin:
        run_checkin()
        return

    # Default: standard report
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M")

    # Calculate maturity scores
    agate_score, agate_stage = _agate_score()
    beryl_score, beryl_stage = _beryl_score()
    citrine_score, citrine_stage = _citrine_score()

    print(f"\n{'='*60}")
    print(f"  DAILY REPORT — {date_str}")
    print(f"{'='*60}\n")

    # ── Project Maturity Overview ─────────────────────────────
    print("┌─ PROJECT MATURITY")
    print(f"│  AGATE   (SOL)   {_score_bar(agate_score)}")
    print(f"│                  Stage: {agate_stage}")
    print(f"│  BERYL   (NDX)   {_score_bar(beryl_score)}")
    print(f"│                  Stage: {beryl_stage}")
    print(f"│  CITRINE (Port)  {_score_bar(citrine_score)}")
    print(f"│                  Stage: {citrine_stage}")

    # Show trend if history exists
    history_path = ROOT / "maturity_history.csv"
    if history_path.exists():
        try:
            hist = pd.read_csv(history_path)
            if len(hist) >= 2:
                prev = hist.iloc[-2]
                agate_delta = agate_score - int(prev["agate_score"])
                beryl_delta = beryl_score - int(prev["beryl_score"])
                agate_arrow = f"↑{agate_delta}" if agate_delta > 0 else (f"↓{abs(agate_delta)}" if agate_delta < 0 else "→")
                beryl_arrow = f"↑{beryl_delta}" if beryl_delta > 0 else (f"↓{abs(beryl_delta)}" if beryl_delta < 0 else "→")
                print(f"│  Trend: AGATE {agate_arrow} | BERYL {beryl_arrow} (vs yesterday)")
        except Exception:
            pass

    # Show check-in streak
    checkin_path = ROOT / "checkin_log.csv"
    if checkin_path.exists():
        try:
            cl = pd.read_csv(checkin_path)
            streak = 0
            for _, r in cl.iloc[::-1].iterrows():
                if r.get("all_complete", False):
                    streak += 1
                else:
                    break
            if streak > 0:
                print(f"│  Check-in streak: {streak} day{'s' if streak != 1 else ''} 🔥")
        except Exception:
            pass

    print("│")

    # ── Detailed Reports ──────────────────────────────────────
    print(_agate_report())
    print(_beryl_report())
    print(_citrine_report())

    print("│")
    print("└─ Decision: [Review above and decide]")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
