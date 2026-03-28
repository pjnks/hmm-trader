#!/bin/bash
# Daily report runner — triggered by cron at 8:45am and 5:30pm
# Runs daily_report.py and sends macOS + Pushover notification

cd /Users/perryjenkins/Documents/trdng/HMM-Trader

# Run the report and capture output
OUTPUT=$(/Users/perryjenkins/opt/anaconda3/bin/python daily_report.py 2>&1)

# Save to log file
echo "--- $(date) ---" >> /Users/perryjenkins/Documents/trdng/HMM-Trader/daily_report.log
echo "$OUTPUT" >> /Users/perryjenkins/Documents/trdng/HMM-Trader/daily_report.log

# Extract key metrics for notification
AGATE_SCORE=$(echo "$OUTPUT" | grep "AGATE (SOL)" | head -1 | grep -o '[0-9]*%')
BERYL_SCORE=$(echo "$OUTPUT" | grep "BERYL (NDX)" | head -1 | grep -o '[0-9]*%')
TRADES=$(echo "$OUTPUT" | grep "Trades executed" | grep -o '[0-9]*')
BEST_BERYL=$(echo "$OUTPUT" | grep "Best config" | sed 's/.*Best config: //' | head -1)

# macOS notification
osascript -e "display notification \"AGATE: ${AGATE_SCORE:-?} | BERYL: ${BERYL_SCORE:-?} | Trades: ${TRADES:-0}\nBERYL best: ${BEST_BERYL:-pending}\" with title \"Daily Report\" sound name \"default\""

# Pushover phone notification (quiet — no sound, just appears)
/Users/perryjenkins/opt/anaconda3/bin/python -c "
from src.notifier import notify_daily
notify_daily(sharpe=0.0, pnl=0.0, trades=${TRADES:-0})
" 2>/dev/null
