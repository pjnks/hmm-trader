#!/bin/bash
# Weekly signal reconciliation (Sunday 6:30am)
# Compares live trading signals vs backtester output for all 3 projects.
# Runs before optimizers finish so results are ready for morning review.
set -euo pipefail

cd /Users/perryjenkins/Documents/trdng/HMM-Trader
PYTHON=/Users/perryjenkins/opt/anaconda3/bin/python
LOGFILE=reconciliation.log

echo "" >> "$LOGFILE"
echo "=== Reconciliation Start: $(date) ===" >> "$LOGFILE"

$PYTHON reconcile.py >> "$LOGFILE" 2>&1
EXIT_CODE=$?

echo "=== Reconciliation End: $(date) (exit=$EXIT_CODE) ===" >> "$LOGFILE"

# Run model health check (degradation detection + auto-retrain triggers)
echo "=== Model Health Check: $(date) ===" >> "$LOGFILE"
$PYTHON -m src.model_health >> "$LOGFILE" 2>&1
echo "=== Model Health Check Done: $(date) ===" >> "$LOGFILE"

# Extract summary and notify
$PYTHON -c "
import sys
sys.path.insert(0, '.')
from src.notifier import _macos_notify, _pushover_notify

if $EXIT_CODE == 0:
    title = 'Weekly Reconciliation Done'
    msg = 'Signal reconciliation complete. Check reconciliation.log for details.'
else:
    title = 'Reconciliation FAILED'
    msg = f'Exit code: $EXIT_CODE. Check reconciliation.log.'

_macos_notify(title, msg)
_pushover_notify(title, msg, priority=-1)
print(f'{title}: {msg}')
" >> "$LOGFILE" 2>&1
