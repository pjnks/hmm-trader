#!/usr/bin/env python3
"""Pull latest performance stats from all project DBs."""
import sqlite3, json, pandas as pd

def q(db, sql):
    con = sqlite3.connect(db)
    df = pd.read_sql(sql, con)
    con.close()
    return df

# --- BERYL ---
beryl = q('/home/ubuntu/HMM-Trader/beryl_trades.db', 'SELECT * FROM trades ORDER BY timestamp')
beryl_snap = q('/home/ubuntu/HMM-Trader/beryl_trades.db', 'SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1')
# BERYL uses side column, not action; each row is a closed round-trip
closed = beryl[beryl['pnl'].notna() & (beryl['exit_price'].notna())]
print('=== BERYL ===')
print(f'Total trades: {len(beryl)}')
print(f'Closed: {len(closed)}, Wins: {(closed["pnl"] > 0).sum()}, Losses: {(closed["pnl"] <= 0).sum()}')
if len(closed) > 0:
    print(f'Win rate: {100*(closed["pnl"] > 0).mean():.1f}%')
    print(f'Realized PnL: ${closed["pnl"].sum():.2f}')
if len(beryl_snap) > 0:
    snap = beryl_snap.iloc[0]
    print(f'Latest snapshot: {snap["timestamp"]}')
    eq_col = "total_equity" if "total_equity" in snap.index else "equity"
    pos_col = "num_positions" if "num_positions" in snap.index else "position_count"
    print(f'Equity: ${snap[eq_col]:.2f}, Positions: {snap[pos_col]}')
    try:
        pos = json.loads(snap['positions_json'])
        for k, v in pos.items():
            if isinstance(v, dict):
                print(f'  {k}: entry=${v.get("entry_price","?")}, conf={v.get("confidence","?")}')
    except: pass
print()

# Backfill stats
bf = q('/home/ubuntu/HMM-Trader/beryl_trades.db', 'SELECT * FROM scan_journal_backfill')
print('=== BACKFILL (scan_journal_backfill) ===')
print(f'Rows: {len(bf)}, Tickers: {bf["ticker"].nunique()}, Days: {bf["scan_date"].nunique()}')
print(f'Date range: {bf["scan_date"].min()} to {bf["scan_date"].max()}')
print(f'Fallback: {bf["used_fallback"].sum()} ({100*bf["used_fallback"].mean():.1f}%)')
print(f'Regimes: BULL={len(bf[bf["regime"]=="BULL"])}, BEAR={len(bf[bf["regime"]=="BEAR"])}, CHOP={len(bf[bf["regime"]=="CHOP"])}')
print()

# --- CITRINE ---
citrine = q('/home/ubuntu/HMM-Trader/citrine_trades.db', 'SELECT * FROM trades ORDER BY timestamp')
cit_snap = q('/home/ubuntu/HMM-Trader/citrine_trades.db', 'SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1')
shadow = q('/home/ubuntu/HMM-Trader/citrine_trades.db', 'SELECT COUNT(*) as n FROM shadow_trades')
cit_closed = citrine[citrine['action'].isin(['EXIT','STOP_EXIT'])]
print('=== CITRINE ===')
print(f'Total trades: {len(citrine)}')
print(f'Closed: {len(cit_closed)}, Wins: {(cit_closed["pnl"] > 0).sum()}, Losses: {(cit_closed["pnl"] <= 0).sum()}')
if len(cit_closed) > 0:
    print(f'Win rate: {100*(cit_closed["pnl"] > 0).mean():.1f}%')
    print(f'Realized PnL: ${cit_closed["pnl"].sum():.2f}')
print(f'Shadow trades: {shadow.iloc[0]["n"]}')
if len(cit_snap) > 0:
    snap = cit_snap.iloc[0]
    eq_col = "total_equity" if "total_equity" in snap.index else "equity"
    pos_col = "num_positions" if "num_positions" in snap.index else "position_count"
    print(f'Equity: ${snap[eq_col]:.2f}, Positions: {snap[pos_col]}')
print()

# --- AGATE ---
agate = q('/home/ubuntu/HMM-Trader/paper_trades.db', 'SELECT * FROM trades ORDER BY timestamp')
ag_closed = agate[agate['pnl'].notna() & (agate['exit_price'].notna())]
print('=== AGATE ===')
print(f'Total trades: {len(agate)}')
print(f'Closed: {len(ag_closed)}')
if len(ag_closed) > 0:
    print(f'Wins: {(ag_closed["pnl"] > 0).sum()}, Losses: {(ag_closed["pnl"] <= 0).sum()}')
    print(f'Realized PnL: ${ag_closed["pnl"].sum():.2f}')
print()

# --- DIAMOND ---
try:
    dia = q('/home/ubuntu/kalshi-diamond/diamond.db', 'SELECT COUNT(*) as n, SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins FROM paper_trades WHERE settled=1')
    dia_total = q('/home/ubuntu/kalshi-diamond/diamond.db', 'SELECT COUNT(*) as n, SUM(pnl) as total_pnl FROM paper_trades WHERE settled=1')
    print('=== DIAMOND ===')
    r = dia_total.iloc[0]
    w = dia.iloc[0]
    print(f'Settled: {int(r["n"])}, Wins: {int(w["wins"])}, WR: {100*w["wins"]/r["n"]:.1f}%')
    print(f'Total PnL: ${r["total_pnl"]:.2f}')
except Exception as e:
    print(f'=== DIAMOND === (error: {e})')
