"""
model_health.py
───────────────
Model degradation detection and auto-retrain triggers.

Monitors rolling Sharpe vs backtest expectations for each project.
Alerts when drift exceeds 2 sigma. Triggers re-optimization when
degradation is confirmed over 5+ consecutive trading days.

Usage:
    # As a module (called by dashboard or cron)
    from src.model_health import ModelHealthMonitor
    monitor = ModelHealthMonitor()
    report = monitor.check_all()

    # Standalone check with notifications
    python -m src.model_health
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


@dataclass
class HealthReport:
    """Health assessment for a single project."""
    project: str
    status: str           # "healthy", "warning", "critical", "too_early", "no_data"
    rolling_sharpe: float
    expected_sharpe: float
    sigma_drift: float    # how many sigma below expected
    total_trades: int
    consecutive_degraded_days: int
    retrain_recommended: bool
    detail: str


# Expected backtest Sharpe per project (from optimization results)
# Live trading typically achieves 10-20% of backtest Sharpe
BACKTEST_SHARPE = {
    "AGATE":   0.837,
    "BERYL":   0.825,
    "CITRINE": 1.665,
}

# Live degradation factor: expect this fraction of backtest Sharpe
LIVE_DEGRADATION_FACTOR = 0.15

# Alert thresholds
WARNING_SIGMA = -1.0
CRITICAL_SIGMA = -2.0
RETRAIN_CONSECUTIVE_DAYS = 5
MIN_TRADES_FOR_ASSESSMENT = 10

# DB paths
PROJECT_DBS = {
    "AGATE":   ROOT / "paper_trades.db",
    "BERYL":   ROOT / "beryl_trades.db",
    "CITRINE": ROOT / "citrine_trades.db",
}

# Health history file (tracks consecutive degraded days)
HEALTH_HISTORY = ROOT / "model_health_history.json"


class ModelHealthMonitor:
    """
    Monitors model performance and triggers retraining when degradation is detected.

    Detection logic:
    1. Compute rolling 20-trade Sharpe for each project
    2. Compare to expected live Sharpe (backtest × degradation factor)
    3. Compute sigma drift: (actual - expected) / (expected × 0.5)
    4. If sigma < -2.0 for 5+ consecutive days → recommend retrain
    """

    def __init__(self):
        self.history = self._load_history()

    def _load_history(self) -> dict:
        """Load consecutive degradation day counts."""
        if HEALTH_HISTORY.exists():
            try:
                with open(HEALTH_HISTORY) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_history(self):
        """Save consecutive degradation day counts."""
        with open(HEALTH_HISTORY, "w") as f:
            json.dump(self.history, f, indent=2)

    def _load_trades(self, project: str) -> pd.DataFrame:
        """Load closed trades (with P&L) for a project."""
        db_path = PROJECT_DBS.get(project)
        if not db_path or not db_path.exists():
            return pd.DataFrame()
        try:
            with sqlite3.connect(str(db_path)) as conn:
                df = pd.read_sql_query(
                    "SELECT * FROM trades ORDER BY timestamp ASC", conn,
                )
            # Filter to closed trades only (ENTER/SCALE_UP have NULL pnl)
            if "pnl" in df.columns:
                df = df.dropna(subset=["pnl"])
            return df
        except Exception:
            return pd.DataFrame()

    def check_project(self, project: str) -> HealthReport:
        """Assess health of a single project."""
        df = self._load_trades(project)

        if df.empty or "pnl" not in df.columns:
            return HealthReport(
                project=project, status="no_data", rolling_sharpe=0.0,
                expected_sharpe=0.0, sigma_drift=0.0, total_trades=0,
                consecutive_degraded_days=0, retrain_recommended=False,
                detail="No trade data available",
            )

        pnls = df["pnl"].values.astype(float)
        total_trades = len(pnls)

        if total_trades < MIN_TRADES_FOR_ASSESSMENT:
            return HealthReport(
                project=project, status="too_early", rolling_sharpe=0.0,
                expected_sharpe=0.0, sigma_drift=0.0, total_trades=total_trades,
                consecutive_degraded_days=0, retrain_recommended=False,
                detail=f"Need {MIN_TRADES_FOR_ASSESSMENT} trades, have {total_trades}",
            )

        # Rolling Sharpe (last 20 trades)
        window = min(20, total_trades)
        recent_pnls = pnls[-window:]
        std = np.std(recent_pnls)
        rolling_sharpe = float(np.mean(recent_pnls) / std) if std > 1e-9 else 0.0

        # Expected live Sharpe
        backtest = BACKTEST_SHARPE.get(project, 1.0)
        expected_live = backtest * LIVE_DEGRADATION_FACTOR

        # Sigma drift
        scale = max(abs(expected_live) * 0.5, 0.1)
        sigma_drift = (rolling_sharpe - expected_live) / scale

        # Determine status
        if sigma_drift < CRITICAL_SIGMA:
            status = "critical"
        elif sigma_drift < WARNING_SIGMA:
            status = "warning"
        else:
            status = "healthy"

        # Track consecutive degraded days
        history_key = f"{project}_degraded_days"
        if status == "critical":
            current_days = self.history.get(history_key, 0) + 1
            self.history[history_key] = current_days
        else:
            self.history[history_key] = 0
            current_days = 0

        # Retrain recommendation
        retrain = (status == "critical" and current_days >= RETRAIN_CONSECUTIVE_DAYS)

        detail = (
            f"Rolling Sharpe: {rolling_sharpe:.3f} "
            f"(expected: {expected_live:.3f}, backtest: {backtest:.3f}). "
            f"Drift: {sigma_drift:.1f} sigma. "
            f"Consecutive degraded days: {current_days}."
        )
        if retrain:
            detail += " AUTO-RETRAIN RECOMMENDED."

        return HealthReport(
            project=project, status=status, rolling_sharpe=rolling_sharpe,
            expected_sharpe=expected_live, sigma_drift=sigma_drift,
            total_trades=total_trades, consecutive_degraded_days=current_days,
            retrain_recommended=retrain, detail=detail,
        )

    def check_all(self) -> list[HealthReport]:
        """Assess health of all projects. Saves history."""
        reports = []
        for project in PROJECT_DBS:
            report = self.check_project(project)
            reports.append(report)
            log.info(f"[{project}] {report.status.upper()}: {report.detail}")

        self._save_history()
        return reports

    def trigger_retrain(self, project: str) -> bool:
        """
        Trigger re-optimization for a degraded project.

        Returns True if retrain was triggered, False otherwise.
        Currently logs the intent and sends notification — actual re-optimization
        is handled by the existing Sunday cron (this just forces an immediate run).
        """
        report = self.check_project(project)
        if not report.retrain_recommended:
            log.info(f"[{project}] Retrain not recommended (status: {report.status})")
            return False

        log.warning(f"[{project}] AUTO-RETRAIN TRIGGERED: {report.detail}")

        # Send notification
        try:
            from src.notifier import _pushover_notify
            _pushover_notify(
                f"{project} Model Degradation — Retrain Triggered",
                report.detail,
                priority=1,  # high priority
            )
        except Exception as e:
            log.error(f"Failed to send retrain notification: {e}")

        # Reset degradation counter after triggering
        self.history[f"{project}_degraded_days"] = 0
        self._save_history()

        return True


def main():
    """Standalone health check with console output and notifications."""
    import sys
    sys.path.insert(0, str(ROOT))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(message)s",
        datefmt="%H:%M:%S",
    )

    monitor = ModelHealthMonitor()
    reports = monitor.check_all()

    print("\n" + "=" * 60)
    print("  MODEL HEALTH REPORT")
    print("=" * 60)

    any_retrain = False
    for r in reports:
        status_icon = {"healthy": "+", "warning": "~", "critical": "!", "too_early": "?", "no_data": "-"}
        icon = status_icon.get(r.status, "?")
        print(f"\n  [{icon}] {r.project}: {r.status.upper()}")
        print(f"      Sharpe(20): {r.rolling_sharpe:.3f} (expected: {r.expected_sharpe:.3f})")
        print(f"      Drift: {r.sigma_drift:.1f} sigma | Trades: {r.total_trades}")
        if r.retrain_recommended:
            print(f"      >>> RETRAIN RECOMMENDED ({r.consecutive_degraded_days} degraded days)")
            any_retrain = True

    print("\n" + "=" * 60)

    # Auto-trigger retrains
    for r in reports:
        if r.retrain_recommended:
            monitor.trigger_retrain(r.project)

    # Send summary notification if any issues
    critical_count = sum(1 for r in reports if r.status == "critical")
    warning_count = sum(1 for r in reports if r.status == "warning")

    if critical_count > 0 or warning_count > 0:
        try:
            from src.notifier import _pushover_notify
            summary = f"{critical_count} critical, {warning_count} warning"
            for r in reports:
                if r.status in ("critical", "warning"):
                    summary += f"\n{r.project}: Sharpe {r.rolling_sharpe:.3f} ({r.sigma_drift:.1f}σ)"
            _pushover_notify("Model Health Alert", summary, priority=0)
        except Exception:
            pass


if __name__ == "__main__":
    main()
