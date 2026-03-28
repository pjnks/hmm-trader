#!/bin/bash
# Equity Check — runs on Mac, queries VM databases
# Reports AGATE + BERYL + CITRINE portfolio status

SSH_KEY="$HOME/.ssh/hmm-trader.key"
VM="ubuntu@129.158.40.51"

RESULT=$(ssh -i "$SSH_KEY" "$VM" 'cd /home/ubuntu/HMM-Trader && /home/ubuntu/miniconda3/bin/python -c "
import sqlite3, json
from datetime import datetime

now = datetime.utcnow().strftime(\"%H:%M UTC\")

# AGATE
agate_pos = \"FLAT\"
agate_pnl = 0
try:
    with open(\"agate_status.json\") as f:
        a = json.load(f)
    pos = a.get(\"position\")
    if pos and pos.get(\"side\"):
        entry = pos[\"entry_price\"]
        curr = a[\"current_price\"]
        pnl_pct = (curr - entry) / entry * 100
        pnl_usd = (curr - entry) * pos[\"size\"]
        agate_pos = f\"{pos[\"ticker\"]} {pnl_pct:+.1f}% (\${pnl_usd:+.0f})\"
        agate_pnl = pnl_usd
except: pass

# BERYL
beryl_pos = \"FLAT\"
beryl_pnl = 0
try:
    with open(\"beryl_status.json\") as f:
        b = json.load(f)
    positions = b.get(\"positions\", {})
    if positions:
        parts = []
        for t, p in positions.items():
            parts.append(t)
        beryl_pos = \", \".join(parts)
except: pass

# CITRINE
citrine_equity = 25000
citrine_cash = 0
citrine_positions = 0
citrine_invested = 0
try:
    db = sqlite3.connect(\"citrine_trades.db\")
    c = db.cursor()
    c.execute(\"SELECT total_equity, cash, invested, num_positions, positions_json FROM portfolio_snapshots ORDER BY rowid DESC LIMIT 1\")
    snap = c.fetchone()
    if snap:
        citrine_equity = snap[0]
        citrine_cash = snap[1]
        citrine_invested = snap[2]
        citrine_positions = snap[3]
        positions = json.loads(snap[4])
        total_unreal = sum((p[\"current\"] - p[\"entry\"]) * p[\"shares\"] for p in positions.values())
    db.close()
except: pass

total_equity = citrine_equity
pnl_from_start = total_equity - 25000
pnl_pct = pnl_from_start / 25000 * 100

print(f\"=== Portfolio Check ({now}) ===\")
print(f\"CITRINE: \${citrine_equity:,.0f} ({pnl_pct:+.2f}%) | {citrine_positions} positions | Cash: \${citrine_cash:,.0f}\")
print(f\"AGATE: {agate_pos}\")
print(f\"BERYL: {beryl_pos}\")
print(f\"Total from start: {pnl_pct:+.2f}% (\${pnl_from_start:+,.0f})\")
"' 2>&1)

echo "$RESULT"

# Send via Pushover
if [ -f "$HOME/Documents/quant/trading-core/.env" ]; then
    source <(grep PUSHOVER "$HOME/Documents/quant/trading-core/.env" | sed 's/^/export /')
    if [ -n "$PUSHOVER_USER_KEY" ] && [ -n "$PUSHOVER_APP_TOKEN" ]; then
        curl -s \
            --form-string "token=$PUSHOVER_APP_TOKEN" \
            --form-string "user=$PUSHOVER_USER_KEY" \
            --form-string "message=$RESULT" \
            --form-string "title=Portfolio Check" \
            https://api.pushover.net/1/messages.json > /dev/null 2>&1
    fi
fi
