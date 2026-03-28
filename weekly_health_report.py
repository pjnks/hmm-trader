"""
weekly_health_report.py
-----------------------
Comprehensive weekly health report for all HMM-Trader sub-projects.

Run manually or via launchd on Sunday nights:
  python weekly_health_report.py

Sections:
  1. Service Health — check 7 systemd services on VM (optional SSH)
  2. BERYL Status — from beryl_status.json
  3. AGATE Status — from agate_status.json
  4. CITRINE Status — from citrine_trades.db
  5. Trade Summary — all 3 DBs, trades this week
  6. Optimizer Results — CSV freshness + top results
  7. Kill-Switch Status — DB checks for triggers
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict | None:
    """Read a JSON file, return None if missing or invalid."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _query_db(db_path: Path, query: str, params: tuple = ()) -> list:
    """Run a SQLite query, return rows. Returns [] on any error."""
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(query, params).fetchall()]
    except Exception:
        return []


def _query_scalar(db_path: Path, query: str, params: tuple = ()):
    """Run a SQLite query returning a single scalar value."""
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(query, params).fetchone()
            return row[0] if row else None
    except Exception:
        return None


def _file_modified_within(path: Path, days: int) -> bool:
    """Check if a file was modified within the last N days."""
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).days < days


def _week_ago_iso() -> str:
    """ISO timestamp for 7 days ago."""
    return (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()


# ── Section 1: Service Health ────────────────────────────────────────────

SERVICES = [
    "agate-trader", "beryl-trader", "citrine-trader", "diamond-monitor",
    "live-dashboard", "citrine-dashboard", "diamond-dashboard",
]

def _check_services() -> list[str]:
    """
    Check systemd services via SSH to VM.
    Returns list of report lines.
    """
    lines = ["=" * 60, "1. SERVICE HEALTH (VM)", "=" * 60]

    ssh_key = Path.home() / ".ssh" / "hmm-trader.key"
    if not ssh_key.exists():
        lines.append("  SSH key not found — skipping VM service check")
        lines.append("  (run on VM directly for service status)")
        return lines

    try:
        cmd = (
            f'ssh -i {ssh_key} -o ConnectTimeout=10 -o StrictHostKeyChecking=no '
            f'ubuntu@129.158.40.51 '
            f'"systemctl is-active {" ".join(SERVICES)}"'
        )
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=20
        )
        statuses = result.stdout.strip().split("\n")

        all_ok = True
        for svc, status in zip(SERVICES, statuses):
            icon = "OK" if status.strip() == "active" else "FAIL"
            if icon == "FAIL":
                all_ok = False
            lines.append(f"  {icon:4s} {svc:24s} [{status.strip()}]")

        if all_ok:
            lines.append("  --- All 7 services active ---")
        else:
            lines.append("  !!! Some services NOT active !!!")

    except subprocess.TimeoutExpired:
        lines.append("  SSH timed out — VM may be unreachable")
    except Exception as e:
        lines.append(f"  SSH error: {e}")

    return lines


# ── Section 2: BERYL Status ─────────────────────────────────────────────

def _beryl_status() -> list[str]:
    lines = ["", "=" * 60, "2. BERYL STATUS", "=" * 60]

    data = _read_json(ROOT / "beryl_status.json")
    if not data:
        lines.append("  beryl_status.json not found or unreadable")
        return lines

    ts = data.get("timestamp", "unknown")
    lines.append(f"  Last update: {ts}")

    # Convergence
    scan = data.get("scan_summary", [])
    tickers_scanned = data.get("tickers_scanned", len(scan))
    converged = sum(1 for s in scan if s.get("regime") is not None)
    lines.append(f"  Convergence: {converged}/{tickers_scanned}")

    # Regime split
    regimes = {"BULL": 0, "BEAR": 0, "CHOP": 0}
    buy_count = 0
    for s in scan:
        r = s.get("regime", "CHOP")
        regimes[r] = regimes.get(r, 0) + 1
        if s.get("signal") == "BUY":
            buy_count += 1
    lines.append(f"  Regimes: {regimes['BULL']} BULL / {regimes['BEAR']} BEAR / {regimes['CHOP']} CHOP")
    lines.append(f"  BUY signals: {buy_count}")

    # Positions — check positions list or single position
    positions = data.get("positions", [])
    if positions:
        lines.append(f"  Open positions ({len(positions)}):")
        for p in positions:
            if isinstance(p, dict):
                t = p.get("ticker", "?")
                ep = p.get("entry_price", 0)
                lines.append(f"    {t} @ ${ep:.2f}")
            else:
                lines.append(f"    {p}")
    elif data.get("position"):
        pos = data["position"]
        lines.append(f"  Position: {pos}")
    else:
        lines.append("  No open positions")

    return lines


# ── Section 3: AGATE Status ─────────────────────────────────────────────

def _agate_status() -> list[str]:
    lines = ["", "=" * 60, "3. AGATE STATUS", "=" * 60]

    data = _read_json(ROOT / "agate_status.json")
    if not data:
        lines.append("  agate_status.json not found or unreadable")
        return lines

    ts = data.get("timestamp", "unknown")
    regime = data.get("regime", "?")
    conf = data.get("confidence", 0)
    signal = data.get("signal", "?")
    price = data.get("current_price", 0)
    ensemble = data.get("use_ensemble", False)
    feat = data.get("feature_set", "?")

    lines.append(f"  Last update: {ts}")
    lines.append(f"  Regime: {regime} (confidence: {conf:.3f})")
    lines.append(f"  Signal: {signal} | Price: ${price:,.2f}")
    lines.append(f"  Ensemble: {ensemble} | Features: {feat}")

    pos = data.get("position")
    if pos:
        lines.append(f"  Position: {pos}")
    else:
        lines.append("  No open position")

    # Multi-ticker scan summary
    scan = data.get("scan_summary", [])
    if len(scan) > 1:
        lines.append(f"  Scan tickers: {len(scan)}")
        for s in scan:
            t = s.get("ticker", "?")
            r = s.get("regime", "?")
            c = s.get("confidence", 0)
            sig = s.get("signal", "?")
            lines.append(f"    {t:12s} {r:5s} conf={c:.3f} sig={sig}")

    return lines


# ── Section 4: CITRINE Status ───────────────────────────────────────────

def _citrine_status() -> list[str]:
    lines = ["", "=" * 60, "4. CITRINE STATUS", "=" * 60]

    db = ROOT / "citrine_trades.db"
    if not db.exists():
        lines.append("  citrine_trades.db not found")
        return lines

    # Open positions from latest snapshot
    snap = _query_db(db, "SELECT * FROM portfolio_snapshots ORDER BY id DESC LIMIT 1")
    if snap:
        s = snap[0]
        equity = s.get("total_equity", 0)
        cash = s.get("cash", 0)
        n_pos = s.get("num_positions", 0)
        n_long = s.get("num_long", 0)
        n_short = s.get("num_short", 0)
        bull = s.get("bull_count", 0)
        bear = s.get("bear_count", 0)
        chop = s.get("chop_count", 0)
        ts = s.get("timestamp", "?")

        lines.append(f"  Last snapshot: {ts}")
        lines.append(f"  Equity: ${equity:,.2f} | Cash: ${cash:,.2f} ({cash/equity*100:.0f}%)" if equity > 0 else f"  Equity: ${equity:,.2f} | Cash: ${cash:,.2f}")
        lines.append(f"  Positions: {n_pos} ({n_long}L / {n_short}S)")
        lines.append(f"  Regime counts: {bull} BULL / {bear} BEAR / {chop} CHOP")

        # Parse positions_json for ticker details
        pj = s.get("positions_json")
        if pj:
            try:
                positions = json.loads(pj)
                if isinstance(positions, dict):
                    tickers = sorted(positions.keys())
                    lines.append(f"  Holdings: {', '.join(tickers[:15])}")
                    if len(tickers) > 15:
                        lines.append(f"           ... and {len(tickers) - 15} more")
                elif isinstance(positions, list):
                    lines.append(f"  Holdings: {len(positions)} tickers")
            except (json.JSONDecodeError, TypeError):
                pass
    else:
        lines.append("  No portfolio snapshots found")

    # Closed trades
    total_trades = _query_scalar(db, "SELECT COUNT(*) FROM trades WHERE pnl IS NOT NULL")
    total_pnl = _query_scalar(db, "SELECT SUM(pnl) FROM trades WHERE pnl IS NOT NULL")
    lines.append(f"  Closed trades: {total_trades or 0}")
    if total_pnl is not None:
        lines.append(f"  Total P&L: ${total_pnl:+,.2f}")

    # Rolling Sharpe (last 20 trades)
    recent = _query_db(db, "SELECT pnl_pct FROM trades WHERE pnl_pct IS NOT NULL ORDER BY id DESC LIMIT 20")
    if len(recent) >= 5:
        pnls = [r["pnl_pct"] for r in recent]
        import numpy as np
        arr = np.array(pnls)
        if arr.std() > 0:
            sharpe = arr.mean() / arr.std() * (252 ** 0.5)
            lines.append(f"  Rolling Sharpe (last {len(recent)}): {sharpe:.3f}")

    return lines


# ── Section 5: Trade Summary (This Week) ────────────────────────────────

def _trade_summary() -> list[str]:
    lines = ["", "=" * 60, "5. TRADE SUMMARY (Last 7 Days)", "=" * 60]

    week_ago = _week_ago_iso()
    total_trades = 0
    total_wins = 0
    total_pnl = 0.0

    # AGATE (paper_trades.db)
    agate_db = ROOT / "paper_trades.db"
    agate_trades = _query_db(
        agate_db,
        "SELECT pnl FROM trades WHERE timestamp >= ?",
        (week_ago,)
    )
    a_count = len(agate_trades)
    a_wins = sum(1 for t in agate_trades if t["pnl"] > 0)
    a_pnl = sum(t["pnl"] for t in agate_trades)
    total_trades += a_count
    total_wins += a_wins
    total_pnl += a_pnl
    lines.append(f"  AGATE:   {a_count:3d} trades | {a_wins} wins | P&L ${a_pnl:+,.2f}")

    # BERYL (beryl_trades.db)
    beryl_db = ROOT / "beryl_trades.db"
    beryl_trades = _query_db(
        beryl_db,
        "SELECT pnl FROM trades WHERE timestamp >= ?",
        (week_ago,)
    )
    b_count = len(beryl_trades)
    b_wins = sum(1 for t in beryl_trades if t["pnl"] > 0)
    b_pnl = sum(t["pnl"] for t in beryl_trades)
    total_trades += b_count
    total_wins += b_wins
    total_pnl += b_pnl
    lines.append(f"  BERYL:   {b_count:3d} trades | {b_wins} wins | P&L ${b_pnl:+,.2f}")

    # CITRINE (citrine_trades.db)
    citrine_db = ROOT / "citrine_trades.db"
    citrine_trades = _query_db(
        citrine_db,
        "SELECT pnl FROM trades WHERE pnl IS NOT NULL AND timestamp >= ?",
        (week_ago,)
    )
    c_count = len(citrine_trades)
    c_wins = sum(1 for t in citrine_trades if t["pnl"] > 0)
    c_pnl = sum(t["pnl"] for t in citrine_trades)
    total_trades += c_count
    total_wins += c_wins
    total_pnl += c_pnl
    lines.append(f"  CITRINE: {c_count:3d} trades | {c_wins} wins | P&L ${c_pnl:+,.2f}")

    lines.append(f"  {'─' * 50}")
    wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    lines.append(f"  TOTAL:   {total_trades:3d} trades | WR {wr:.0f}% | P&L ${total_pnl:+,.2f}")

    return lines


# ── Section 6: Optimizer Results ─────────────────────────────────────────

def _optimizer_results() -> list[str]:
    lines = ["", "=" * 60, "6. OPTIMIZER RESULTS (Updated This Week?)", "=" * 60]

    optimizers = [
        ("BERYL daily", ROOT / "beryl_daily_results.csv", "wf_sharpe"),
        ("CITRINE",     ROOT / "citrine_optimization_results.csv", "wf_sharpe"),
        ("AGATE multi", ROOT / "agate_multi_optimization_results.csv", "wf_sharpe"),
    ]

    for name, csv_path, sharpe_col in optimizers:
        if not csv_path.exists():
            lines.append(f"  {name:15s}: CSV not found")
            continue

        fresh = _file_modified_within(csv_path, 7)
        mtime = datetime.fromtimestamp(csv_path.stat().st_mtime, tz=timezone.utc)
        age_days = (datetime.now(timezone.utc) - mtime).days

        try:
            import csv as csvmod
            with open(csv_path) as f:
                reader = csvmod.DictReader(f)
                rows = list(reader)
            trial_count = len(rows)

            # Find best Sharpe
            best_sharpe = max(
                (float(r[sharpe_col]) for r in rows if r.get(sharpe_col)),
                default=0
            )

            status = "UPDATED" if fresh else f"{age_days}d ago"
            lines.append(
                f"  {name:15s}: {trial_count:4d} trials | "
                f"best Sharpe {best_sharpe:+.3f} | {status}"
            )
        except Exception as e:
            lines.append(f"  {name:15s}: error reading CSV ({e})")

    return lines


# ── Section 7: Kill-Switch Status ────────────────────────────────────────

def _kill_switch_status() -> list[str]:
    lines = ["", "=" * 60, "7. KILL-SWITCH STATUS", "=" * 60]

    checks = [
        ("AGATE",   ROOT / "paper_trades.db"),
        ("BERYL",   ROOT / "beryl_trades.db"),
        ("CITRINE", ROOT / "citrine_trades.db"),
    ]

    for name, db_path in checks:
        if not db_path.exists():
            lines.append(f"  {name:8s}: DB not found — no kill-switch data")
            continue

        # Check total trades and win rate
        total = _query_scalar(db_path, "SELECT COUNT(*) FROM trades WHERE pnl IS NOT NULL") or 0

        if total == 0:
            lines.append(f"  {name:8s}: No closed trades — kill-switch N/A")
            continue

        wins = _query_scalar(db_path, "SELECT COUNT(*) FROM trades WHERE pnl > 0") or 0
        wr = wins / total * 100 if total > 0 else 0

        # Check last 10 trades for 0/10 wins rule
        last_10 = _query_db(
            db_path,
            "SELECT pnl FROM trades WHERE pnl IS NOT NULL ORDER BY id DESC LIMIT 10"
        )
        last_10_wins = sum(1 for t in last_10 if t["pnl"] > 0)
        zero_wins_flag = len(last_10) >= 10 and last_10_wins == 0

        # Rolling 5-trade Sharpe (rough check)
        last_5 = _query_db(
            db_path,
            "SELECT pnl_pct FROM trades WHERE pnl_pct IS NOT NULL ORDER BY id DESC LIMIT 5"
        )
        low_sharpe_flag = False
        if len(last_5) >= 5:
            import numpy as np
            pnls = np.array([t["pnl_pct"] for t in last_5])
            if pnls.std() > 0:
                rs = pnls.mean() / pnls.std()
                low_sharpe_flag = rs < 0.3

        # Daily loss check (today)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_pnl = _query_scalar(
            db_path,
            "SELECT SUM(pnl) FROM trades WHERE pnl IS NOT NULL AND timestamp LIKE ?",
            (f"{today}%",)
        )
        daily_loss_flag = (today_pnl or 0) < -200  # rough 2% of $10k

        flags = []
        if zero_wins_flag:
            flags.append("0/10 WINS")
        if low_sharpe_flag:
            flags.append("LOW SHARPE")
        if daily_loss_flag:
            flags.append("DAILY LOSS")

        status = "TRIGGERED: " + ", ".join(flags) if flags else "OK"
        lines.append(f"  {name:8s}: {total} trades | WR {wr:.0f}% | last10 {last_10_wins}W | {status}")

    return lines


# ── Main Report ──────────────────────────────────────────────────────────

def generate_report() -> str:
    """Generate the full weekly health report as a string."""
    sections = []

    header = [
        "",
        "#" * 60,
        "#  WEEKLY HEALTH REPORT",
        f"#  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "#" * 60,
    ]
    sections.append("\n".join(header))

    sections.append("\n".join(_check_services()))
    sections.append("\n".join(_beryl_status()))
    sections.append("\n".join(_agate_status()))
    sections.append("\n".join(_citrine_status()))
    sections.append("\n".join(_trade_summary()))
    sections.append("\n".join(_optimizer_results()))
    sections.append("\n".join(_kill_switch_status()))

    footer = [
        "",
        "=" * 60,
        "END OF WEEKLY HEALTH REPORT",
        "=" * 60,
    ]
    sections.append("\n".join(footer))

    return "\n".join(sections)


def _build_pushover_summary(report: str) -> str:
    """
    Condense the full report into a Pushover-friendly summary (<1024 chars).
    """
    lines = []

    # Extract key metrics by re-running lightweight checks
    # AGATE
    data = _read_json(ROOT / "agate_status.json")
    if data:
        lines.append(f"AGATE: {data.get('regime','?')} conf={data.get('confidence',0):.2f} sig={data.get('signal','?')}")

    # BERYL
    data = _read_json(ROOT / "beryl_status.json")
    if data:
        scan = data.get("scan_summary", [])
        buys = sum(1 for s in scan if s.get("signal") == "BUY")
        n_pos = len(data.get("positions", []))
        lines.append(f"BERYL: {len(scan)} scanned, {buys} BUY, {n_pos} pos")

    # CITRINE
    db = ROOT / "citrine_trades.db"
    snap = _query_db(db, "SELECT total_equity, num_positions, cash FROM portfolio_snapshots ORDER BY id DESC LIMIT 1")
    if snap:
        s = snap[0]
        lines.append(f"CITRINE: ${s['total_equity']:,.0f} equity, {s['num_positions']} pos")

    # Weekly trades
    week_ago = _week_ago_iso()
    total = 0
    pnl = 0.0
    for db_name in ["paper_trades.db", "beryl_trades.db"]:
        rows = _query_db(ROOT / db_name, "SELECT pnl FROM trades WHERE timestamp >= ?", (week_ago,))
        total += len(rows)
        pnl += sum(r["pnl"] for r in rows)
    citrine_rows = _query_db(
        ROOT / "citrine_trades.db",
        "SELECT pnl FROM trades WHERE pnl IS NOT NULL AND timestamp >= ?",
        (week_ago,)
    )
    total += len(citrine_rows)
    pnl += sum(r["pnl"] for r in citrine_rows)

    lines.append(f"Week: {total} trades, P&L ${pnl:+,.2f}")

    # Kill-switch
    ks_ok = True
    for db_name in ["paper_trades.db", "beryl_trades.db", "citrine_trades.db"]:
        last_10 = _query_db(ROOT / db_name, "SELECT pnl FROM trades WHERE pnl IS NOT NULL ORDER BY id DESC LIMIT 10")
        if len(last_10) >= 10 and sum(1 for t in last_10 if t["pnl"] > 0) == 0:
            ks_ok = False
    lines.append(f"Kill-switch: {'OK' if ks_ok else 'CHECK NEEDED'}")

    # Optimizers
    opt_updated = []
    for name, csv_name in [("BERYL", "beryl_daily_results.csv"), ("CITRINE", "citrine_optimization_results.csv"), ("AGATE", "agate_multi_optimization_results.csv")]:
        if _file_modified_within(ROOT / csv_name, 7):
            opt_updated.append(name)
    if opt_updated:
        lines.append(f"Optimizers updated: {', '.join(opt_updated)}")
    else:
        lines.append("Optimizers: no updates this week")

    return "\n".join(lines)


def main():
    # Generate and print full report
    report = generate_report()
    print(report)

    # Send Pushover summary
    try:
        sys.path.insert(0, str(ROOT))
        from src.notifier import _pushover_notify, _macos_notify
        summary = _build_pushover_summary(report)
        _pushover_notify("Weekly Health Report", summary, priority=0)
        _macos_notify("HMM-Trader", "Weekly health report generated", sound="default")
        print("\n[Pushover notification sent]")
    except Exception as e:
        print(f"\n[Notification failed: {e}]")


if __name__ == "__main__":
    main()
