#!/bin/bash
# HMM-Trader Comprehensive Health Check
# Run: bash health_check.sh
# Covers all 3 projects: AGATE, BERYL, CITRINE
# -------------------------------------------------
set -uo pipefail

cd /Users/perryjenkins/Documents/trdng/HMM-Trader
PYTHON=/Users/perryjenkins/opt/anaconda3/bin/python

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

PASS=0
FAIL=0
WARN=0

pass_check() { echo -e "  ${GREEN}PASS${NC} $1"; PASS=$((PASS+1)); }
fail_check() { echo -e "  ${RED}FAIL${NC} $1"; FAIL=$((FAIL+1)); }
warn_check() { echo -e "  ${YELLOW}WARN${NC} $1"; WARN=$((WARN+1)); }

echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}  HMM-Trader Comprehensive Health Check${NC}"
echo -e "${BOLD}${CYAN}  $(date)${NC}"
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════${NC}"
echo ""

# ─────────────────────────────────────────────
# 1. PROCESS HEALTH
# ─────────────────────────────────────────────
echo -e "${BOLD}[1/8] PROCESS HEALTH${NC}"

if pgrep -f "live_trading.py --test" > /dev/null; then
    PID=$(pgrep -f "live_trading.py --test" | head -1)
    pass_check "AGATE running (PID $PID)"
else
    fail_check "AGATE not running!"
fi

if pgrep -f "live_trading_beryl.py" > /dev/null; then
    PID=$(pgrep -f "live_trading_beryl.py" | head -1)
    pass_check "BERYL running (PID $PID)"
else
    fail_check "BERYL not running!"
fi

if pgrep -f "live_trading_citrine.py" > /dev/null; then
    PID=$(pgrep -f "live_trading_citrine.py" | head -1)
    pass_check "CITRINE running (PID $PID)"
else
    fail_check "CITRINE not running!"
fi

if pgrep -f "live_dashboard.py" > /dev/null; then
    pass_check "Dashboard :8060 running"
else
    warn_check "Dashboard :8060 not running"
fi

if pgrep -f "citrine_dashboard.py" > /dev/null; then
    pass_check "Dashboard :8070 running"
else
    warn_check "Dashboard :8070 not running"
fi

echo ""

# ─────────────────────────────────────────────
# 2. LOG FRESHNESS (detect laptop sleep gaps)
# ─────────────────────────────────────────────
echo -e "${BOLD}[2/8] LOG FRESHNESS${NC}"

NOW=$(date +%s)

check_log_age() {
    local file=$1
    local name=$2
    local max_hours=$3
    if [ -f "$file" ]; then
        MOD=$(stat -f %m "$file" 2>/dev/null || stat -c %Y "$file" 2>/dev/null)
        AGE_H=$(( (NOW - MOD) / 3600 ))
        if [ "$AGE_H" -le "$max_hours" ]; then
            pass_check "$name last updated ${AGE_H}h ago"
        else
            fail_check "$name stale — ${AGE_H}h since last update (max ${max_hours}h)"
        fi
    else
        fail_check "$name log file missing!"
    fi
}

check_log_age "agate_test.log" "AGATE log" 6
check_log_age "beryl_test.log" "BERYL log" 26    # daily scan, allow 26h
check_log_age "citrine_test.log" "CITRINE log" 26
check_log_age "agate_status.json" "AGATE status" 6
check_log_age "beryl_status.json" "BERYL status" 26

echo ""

# ─────────────────────────────────────────────
# 3. NETWORK CONNECTIVITY
# ─────────────────────────────────────────────
echo -e "${BOLD}[3/8] NETWORK CONNECTIVITY${NC}"

if curl -sf --max-time 5 "https://api.polygon.io/v2/aggs/ticker/X:SOLUSD/range/1/hour/1700000000000/1700003600000?apiKey=test" > /dev/null 2>&1 || [ $? -ne 7 ]; then
    pass_check "Polygon API reachable"
else
    fail_check "Polygon API unreachable (DNS/network issue)"
fi

# Check for DNS errors in recent logs
if tail -50 agate_test.log 2>/dev/null | grep -q "Failed to resolve"; then
    warn_check "Recent DNS failure in AGATE log"
else
    pass_check "No recent DNS failures in AGATE log"
fi

echo ""

# ─────────────────────────────────────────────
# 4. CONFIG INTEGRITY
# ─────────────────────────────────────────────
echo -e "${BOLD}[4/8] CONFIG INTEGRITY${NC}"

$PYTHON -c "
import config
issues = []

# Feature set check
if config.FEATURE_SET != 'extended_v2':
    issues.append(f'FEATURE_SET={config.FEATURE_SET} (expected extended_v2)')

# Confirmations
if config.MIN_CONFIRMATIONS != 7:
    issues.append(f'MIN_CONFIRMATIONS={config.MIN_CONFIRMATIONS} (expected 7)')

# vol_price_diverge must NOT be in any feature set
for name, feats in config.FEATURE_SETS.items():
    if 'vol_price_diverge' in feats:
        issues.append(f'vol_price_diverge in {name} feature set!')

# CITRINE thresholds
if config.CITRINE_ENTRY_CONFIDENCE != 0.9:
    issues.append(f'CITRINE_ENTRY={config.CITRINE_ENTRY_CONFIDENCE} (expected 0.9)')
if config.CITRINE_EXIT_CONFIDENCE != 0.5:
    issues.append(f'CITRINE_EXIT={config.CITRINE_EXIT_CONFIDENCE} (expected 0.5)')
if config.CITRINE_MAX_PER_SECTOR != 4:
    issues.append(f'CITRINE_MAX_PER_SECTOR={config.CITRINE_MAX_PER_SECTOR} (expected 4)')

if issues:
    for i in issues:
        print(f'FAIL:{i}')
else:
    print('PASS:All config values correct')
" 2>&1 | while IFS=: read -r status msg; do
    if [ "$status" = "PASS" ]; then
        pass_check "$msg"
    else
        fail_check "$msg"
    fi
done

echo ""

# ─────────────────────────────────────────────
# 5. DATABASE HEALTH
# ─────────────────────────────────────────────
echo -e "${BOLD}[5/8] DATABASE HEALTH${NC}"

for db in paper_trades.db beryl_trades.db citrine_trades.db; do
    if [ -f "$db" ]; then
        SIZE=$(du -h "$db" | cut -f1)
        pass_check "$db exists ($SIZE)"
    else
        fail_check "$db missing!"
    fi
done

# CITRINE trade count and open positions
$PYTHON -c "
import sqlite3
conn = sqlite3.connect('citrine_trades.db')
c = conn.cursor()

# Total trades
c.execute('SELECT COUNT(*) FROM trades')
total = c.fetchone()[0]
print(f'INFO:CITRINE: {total} total trade events')

# Open positions from latest snapshot
c.execute('SELECT portfolio_json FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1')
row = c.fetchone()
if row:
    import json
    positions = json.loads(row[0])
    print(f'INFO:CITRINE: {len(positions)} open positions')

# Closed P&L
c.execute(\"SELECT SUM(pnl) FROM trades WHERE side='SELL'\")
total_pnl = c.fetchone()[0] or 0
print(f'INFO:CITRINE realized P&L: \${total_pnl:.2f}')

# Kill-switch check
c.execute(\"SELECT COUNT(*) FROM trades WHERE side='SELL' AND pnl IS NOT NULL\")
closed = c.fetchone()[0]
if closed >= 10:
    c.execute(\"SELECT pnl FROM trades WHERE side='SELL' AND pnl IS NOT NULL ORDER BY timestamp DESC LIMIT 10\")
    recent = [r[0] for r in c.fetchall()]
    wins = sum(1 for p in recent if p > 0)
    if wins == 0:
        print('FAIL:CITRINE kill-switch: 0/10 recent wins!')
    else:
        print(f'PASS:CITRINE recent win rate: {wins}/10')
else:
    print(f'INFO:Only {closed} closed trades — kill-switch monitoring needs 10+')

conn.close()
" 2>&1 | while IFS=: read -r status msg; do
    case "$status" in
        PASS) pass_check "$msg" ;;
        FAIL) fail_check "$msg" ;;
        INFO) echo -e "  ${CYAN}INFO${NC} $msg" ;;
    esac
done

# AGATE trade count
$PYTHON -c "
import sqlite3
conn = sqlite3.connect('paper_trades.db')
c = conn.cursor()
c.execute('SELECT COUNT(*) FROM trades')
total = c.fetchone()[0]
c.execute(\"SELECT COUNT(*) FROM trades WHERE pnl IS NOT NULL\")
closed = c.fetchone()[0]
print(f'INFO:AGATE: {total} trades ({closed} closed)')
conn.close()
" 2>&1 | while IFS=: read -r status msg; do
    echo -e "  ${CYAN}INFO${NC} $msg"
done

echo ""

# ─────────────────────────────────────────────
# 6. TEST SUITE
# ─────────────────────────────────────────────
echo -e "${BOLD}[6/8] TEST SUITE${NC}"

TEST_OUTPUT=$($PYTHON -m pytest tests/test_strategy_consistency.py -q 2>&1)
TEST_EXIT=$?

if [ $TEST_EXIT -eq 0 ]; then
    PASSED=$(echo "$TEST_OUTPUT" | grep -oE '[0-9]+ passed' | head -1)
    pass_check "Test suite: $PASSED"
else
    FAILED=$(echo "$TEST_OUTPUT" | tail -5)
    fail_check "Test suite FAILED: $FAILED"
fi

echo ""

# ─────────────────────────────────────────────
# 7. HMM CONVERGENCE CHECK
# ─────────────────────────────────────────────
echo -e "${BOLD}[7/8] HMM CONVERGENCE (from recent logs)${NC}"

# AGATE — check last ensemble fit
AGATE_CONVERGED=$(grep -c "Ensemble fitted: 3/3 models converged" agate_test.log 2>/dev/null || echo 0)
AGATE_PARTIAL=$(grep -c "Ensemble fitted: 2/3 models converged" agate_test.log 2>/dev/null || echo 0)
if [ "$AGATE_CONVERGED" -gt 0 ]; then
    pass_check "AGATE ensemble: 3/3 converged (seen ${AGATE_CONVERGED}x)"
elif [ "$AGATE_PARTIAL" -gt 0 ]; then
    warn_check "AGATE ensemble: only 2/3 converged (seen ${AGATE_PARTIAL}x)"
else
    warn_check "AGATE: no convergence info in log"
fi

# BERYL — check for non-convergence warnings
BERYL_FAIL=$(grep -c "HMM did not converge" beryl_test.log 2>/dev/null || echo 0)
if [ "$BERYL_FAIL" -gt 0 ]; then
    TICKERS=$(grep -B1 "HMM did not converge" beryl_test.log 2>/dev/null | grep -oE '(NVDA|TSLA|GOOGL|MSFT|AAPL)' | sort -u | tr '\n' ',' | sed 's/,$//')
    warn_check "BERYL: $BERYL_FAIL convergence failures ($TICKERS)"
else
    pass_check "BERYL: all HMMs converged"
fi

# CITRINE — check for convergence in last scan
CITRINE_CONVERGE=$(tail -200 citrine_test.log 2>/dev/null | grep -c "HMM did not converge" 2>/dev/null || echo 0)
CITRINE_CONVERGE=$(echo "$CITRINE_CONVERGE" | tr -d '[:space:]')
if [ "$CITRINE_CONVERGE" -gt 0 ]; then
    warn_check "CITRINE: $CITRINE_CONVERGE convergence failures in last scan"
else
    pass_check "CITRINE: all HMMs converged in last scan"
fi

echo ""

# ─────────────────────────────────────────────
# 8. DATA CACHE FRESHNESS
# ─────────────────────────────────────────────
echo -e "${BOLD}[8/8] DATA CACHE${NC}"

for cache in data_cache/solusd_hourly.csv; do
    if [ -f "$cache" ]; then
        MOD=$(stat -f %m "$cache" 2>/dev/null || stat -c %Y "$cache" 2>/dev/null)
        AGE_H=$(( (NOW - MOD) / 3600 ))
        if [ "$AGE_H" -le 8 ]; then
            pass_check "SOL cache fresh (${AGE_H}h old)"
        else
            warn_check "SOL cache stale (${AGE_H}h old)"
        fi
    else
        warn_check "SOL cache missing"
    fi
done

# Per-ticker configs exist
if [ -f "citrine_per_ticker_configs.json" ]; then
    N_TICKERS=$($PYTHON -c "import json; print(len(json.load(open('citrine_per_ticker_configs.json'))))")
    pass_check "CITRINE per-ticker configs: $N_TICKERS tickers"
else
    fail_check "citrine_per_ticker_configs.json missing!"
fi

# Crontab check
CRON_COUNT=$(crontab -l 2>/dev/null | grep -c "run_optimize" || echo 0)
if [ "$CRON_COUNT" -eq 3 ]; then
    pass_check "All 3 optimizer cron jobs installed"
else
    warn_check "Only $CRON_COUNT/3 optimizer cron jobs found"
fi

echo ""

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════${NC}"
TOTAL=$((PASS + FAIL + WARN))
echo -e "  ${GREEN}$PASS passed${NC}  ${RED}$FAIL failed${NC}  ${YELLOW}$WARN warnings${NC}  (${TOTAL} total checks)"

if [ "$FAIL" -eq 0 ]; then
    echo -e "  ${GREEN}${BOLD}ALL CLEAR${NC}"
elif [ "$FAIL" -le 2 ]; then
    echo -e "  ${YELLOW}${BOLD}MINOR ISSUES — review failures above${NC}"
else
    echo -e "  ${RED}${BOLD}ACTION REQUIRED — $FAIL failures need attention${NC}"
fi
echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════${NC}"
