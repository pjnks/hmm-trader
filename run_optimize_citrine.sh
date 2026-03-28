#!/bin/bash
# CITRINE weekly re-optimization (Sunday 2am)
# Hot-swappable: auto-updates citrine_per_ticker_configs.json on completion
set -euo pipefail

cd /Users/perryjenkins/Documents/trdng/HMM-Trader
PYTHON=/Users/perryjenkins/opt/anaconda3/bin/python
LOGFILE=optimization_citrine.log

echo "" >> "$LOGFILE"
echo "=== CITRINE Optimizer Start: $(date) ===" >> "$LOGFILE"

$PYTHON optimize_citrine.py --runs 100 --resume --workers 6 >> "$LOGFILE" 2>&1
EXIT_CODE=$?

echo "=== CITRINE Optimizer End: $(date) (exit=$EXIT_CODE) ===" >> "$LOGFILE"

# Extract summary and notify
$PYTHON -c "
import pandas as pd, json, sys
sys.path.insert(0, '.')
from src.notifier import _macos_notify, _pushover_notify

try:
    df = pd.read_csv('citrine_optimization_results.csv', on_bad_lines='skip')
    total = len(df)
    positive = len(df[df['wf_sharpe'] > 0])

    with open('citrine_per_ticker_configs.json') as f:
        configs = json.load(f)
    n_tickers = len(configs)

    # Top 5 by Sharpe
    best = df.sort_values('wf_sharpe', ascending=False).head(5)
    top5 = ', '.join(f'{r.ticker}({r.wf_sharpe:.2f})' for _, r in best.iterrows())

    msg = f'{total} trials, {positive} positive Sharpe\n{n_tickers} tickers configured\nTop: {top5}'
    title = 'CITRINE Optimizer Done'
except Exception as e:
    msg = f'Error parsing results: {e}'
    title = 'CITRINE Optimizer'

_macos_notify(title, msg)
_pushover_notify(title, msg, priority=-1)
print(f'{title}: {msg}')
" >> "$LOGFILE" 2>&1

# Auto-deploy: upload updated per-ticker configs to VM (no restart needed — hot-loaded)
SSH_KEY="$HOME/.ssh/hmm-trader.key"
VM="ubuntu@129.158.40.51"
CONFIG_FILE="citrine_per_ticker_configs.json"

if [ -f "$CONFIG_FILE" ]; then
    echo "=== Auto-deploying $CONFIG_FILE to VM ===" >> "$LOGFILE"
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no "$CONFIG_FILE" "$VM:/home/ubuntu/HMM-Trader/$CONFIG_FILE" >> "$LOGFILE" 2>&1
    if [ $? -eq 0 ]; then
        echo "CITRINE configs deployed (hot-loaded, no restart needed)." >> "$LOGFILE"
    else
        echo "WARNING: Failed to upload $CONFIG_FILE to VM." >> "$LOGFILE"
    fi
fi
