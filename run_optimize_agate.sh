#!/bin/bash
# AGATE weekly multi-ticker crypto re-optimization (Sunday 8am)
# Per-ticker configs auto-saved to agate_per_ticker_configs.json (hot-loaded by live_trading.py)
set -euo pipefail

cd /Users/perryjenkins/Documents/trdng/HMM-Trader
PYTHON=/Users/perryjenkins/opt/anaconda3/bin/python
LOGFILE=optimization_agate.log

echo "" >> "$LOGFILE"
echo "=== AGATE Multi-Ticker Optimizer Start: $(date) ===" >> "$LOGFILE"

# Run multi-ticker optimizer (25 trials across 16 tickers × 144 configs)
# Per-ticker configs are auto-saved to agate_per_ticker_configs.json
$PYTHON optimize_agate_multi.py --runs 25 --resume --workers 6 >> "$LOGFILE" 2>&1
EXIT_CODE=$?

echo "=== AGATE Multi-Ticker Optimizer End: $(date) (exit=$EXIT_CODE) ===" >> "$LOGFILE"

# Extract summary and notify
$PYTHON -c "
import pandas as pd, json, sys
sys.path.insert(0, '.')
from src.notifier import _macos_notify, _pushover_notify

try:
    df = pd.read_csv('agate_multi_optimization_results.csv', on_bad_lines='skip')
    total = len(df)
    positive = len(df[df['wf_sharpe'] > 0])
    tickers_tested = df['ticker'].nunique()

    # Top 5 overall
    best = df.sort_values('wf_sharpe', ascending=False).head(5)
    top5 = ', '.join(f'{r.ticker.replace(\"X:\",\"\")}/{r.timeframe}/{r.feature_set}({r.wf_sharpe:.2f})' for _, r in best.iterrows())

    # Per-ticker config status
    try:
        with open('agate_per_ticker_configs.json') as f:
            configs = json.load(f)
        n_optimized = len(configs)
        n_positive = sum(1 for c in configs.values() if c.get('wf_sharpe', 0) > 0)
    except:
        n_optimized = 0
        n_positive = 0

    msg = f'{total} trials, {positive} positive, {tickers_tested} tickers\nTop: {top5}\nConfigs: {n_optimized} tickers ({n_positive} positive)'
    title = 'AGATE Multi-Ticker Optimizer Done'
except Exception as e:
    msg = f'Error parsing results: {e}'
    title = 'AGATE Optimizer'

_macos_notify(title, msg)
_pushover_notify(title, msg, priority=-1)
print(f'{title}: {msg}')
" >> "$LOGFILE" 2>&1
