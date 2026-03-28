#!/bin/bash
# BERYL weekly re-optimization (Sunday 4am)
# Auto-applies: uploads beryl_per_ticker_configs.json to VM and restarts service
set -euo pipefail

cd /Users/perryjenkins/Documents/trdng/HMM-Trader
PYTHON=/Users/perryjenkins/opt/anaconda3/bin/python
LOGFILE=optimization_beryl.log

echo "" >> "$LOGFILE"
echo "=== BERYL Optimizer Start: $(date) ===" >> "$LOGFILE"

$PYTHON optimize_beryl_daily.py --runs 50 --resume --workers 6 >> "$LOGFILE" 2>&1
EXIT_CODE=$?

echo "=== BERYL Optimizer End: $(date) (exit=$EXIT_CODE) ===" >> "$LOGFILE"

# Extract summary and notify
$PYTHON -c "
import pandas as pd, sys
sys.path.insert(0, '.')
from src.notifier import _macos_notify, _pushover_notify

try:
    df = pd.read_csv('beryl_daily_results.csv', on_bad_lines='skip')
    total = len(df)
    positive = len(df[df['wf_sharpe'] > 0])

    # Best per ticker
    best = df.loc[df.groupby('ticker')['wf_sharpe'].idxmax()]
    per_ticker = ', '.join(f'{r.ticker}({r.wf_sharpe:.3f})' for _, r in best.sort_values('wf_sharpe', ascending=False).iterrows())

    msg = f'{total} trials, {positive} positive\nBest per ticker: {per_ticker}'
    title = 'BERYL Daily Optimizer Done'
except Exception as e:
    msg = f'Error parsing results: {e}'
    title = 'BERYL Optimizer'

_macos_notify(title, msg)
_pushover_notify(title, msg, priority=-1)
print(f'{title}: {msg}')
" >> "$LOGFILE" 2>&1

# Auto-deploy: upload updated per-ticker configs to VM and restart beryl-trader
SSH_KEY="$HOME/.ssh/hmm-trader.key"
VM="ubuntu@129.158.40.51"
CONFIG_FILE="beryl_per_ticker_configs.json"

if [ -f "$CONFIG_FILE" ]; then
    echo "=== Auto-deploying $CONFIG_FILE to VM ===" >> "$LOGFILE"
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no "$CONFIG_FILE" "$VM:/home/ubuntu/HMM-Trader/$CONFIG_FILE" >> "$LOGFILE" 2>&1
    if [ $? -eq 0 ]; then
        ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$VM" "sudo systemctl restart beryl-trader" >> "$LOGFILE" 2>&1
        echo "BERYL configs deployed and service restarted." >> "$LOGFILE"
    else
        echo "WARNING: Failed to upload $CONFIG_FILE to VM." >> "$LOGFILE"
    fi
fi
