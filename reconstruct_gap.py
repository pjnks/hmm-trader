#!/usr/bin/env python3
"""
Reconstruct what CITRINE would have done on Mar 25 2026 (missed cycle).
Uses the same scanner, allocator, and configs as live trading.
"""
import json
import sys
import sqlite3

sys.path.insert(0, '/home/ubuntu/HMM-Trader')

from src.citrine_scanner import CitrineScanner
from src.citrine_allocator import CitrineAllocator


def main():
    print('=' * 70)
    print('  CITRINE GAP RECONSTRUCTION — Mar 25, 2026')
    print('  What would have happened if the daily cycle ran?')
    print('=' * 70)

    # Load the last known portfolio state
    conn = sqlite3.connect('/home/ubuntu/HMM-Trader/citrine_trades.db')
    conn.row_factory = sqlite3.Row
    snap = conn.execute(
        'SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1'
    ).fetchone()

    if not snap:
        print('ERROR: No portfolio snapshots found')
        return

    positions = json.loads(snap['positions_json']) if snap['positions_json'] else {}
    ts = dict(snap)['timestamp'][:16]
    equity = dict(snap)['total_equity']
    print(f'\n  Last snapshot: {ts}')
    print(f'  Equity: ${equity:.2f}')
    print(f'  Positions: {len(positions)}')
    print(f'  Held tickers: {", ".join(sorted(positions.keys()))}')

    # Reconstruct allocator state
    allocator = CitrineAllocator()
    for ticker in positions:
        allocator._holdings[ticker] = 3  # Assume day 3+ (fully scaled)

    tickers = (
        'FER,MELI,STX,CEG,WBD,AXON,TSLA,WDC,MU,GEHC,REGN,ZS,PCAR,GOOGL,'
        'PANW,AMAT,TXN,TMUS,NVDA,MAR,LIN,MDLZ,WMT,CSCO,FAST,NXPI,CTAS,'
        'MCHP,NFLX,INTU,ROP,ADSK,CRWD,SNPS,EA,CTSH,ROST,MPWR,CDNS,MSTR,'
        'CPRT,LRCX,AMZN,BKNG,KLAC,COST,XEL,PYPL,KDP,MNST,CSGP,INTC,AAPL,'
        'AEP,PLTR,ODFL,SHOP,AMD,EXC,ADI,TEAM,VRSK,CCEP,DASH,ASML,INSM,'
        'FANG,BKR,QCOM,APP,CSX,AVGO,PDD,TRI,KHC,AMGN,FTNT,ABNB,ADBE,'
        'HON,MSFT,WDAY'
    ).split(',')

    print(f'\n  Scanning {len(tickers)} tickers...')
    scanner = CitrineScanner(tickers=tickers)
    scans = scanner.scan_all()

    valid = [s for s in scans if s.error is None]
    bull = [s for s in valid if s.regime_cat == 'BULL']
    bear = [s for s in valid if s.regime_cat == 'BEAR']
    chop = [s for s in valid if s.regime_cat == 'CHOP']

    print(f'  Scan complete: {len(valid)}/{len(tickers)} valid')
    print(f'  Regimes: {len(bull)} BULL | {len(bear)} BEAR | {len(chop)} CHOP')

    # Run allocator
    print(f'\n  Running allocator with reconstructed state...')
    weights, cash_pct = allocator.allocate(scans)

    enters = [w for w in weights if w.action == 'ENTER']
    exits = [w for w in weights if w.action == 'EXIT']
    holds = [w for w in weights if w.action in ('HOLD', 'SCALE_UP')]

    print(f'\n  Allocator result: {len(enters)} ENTER | {len(exits)} EXIT | '
          f'{len(holds)} HOLD/SCALE | Cash target: {cash_pct:.0%}')

    # Show exits
    total_exit_pnl = 0.0
    if exits:
        print(f'\n  -- WOULD HAVE EXITED ({len(exits)} positions) --')
        for w in sorted(exits, key=lambda x: x.ticker):
            pos = positions.get(w.ticker, {})
            entry_price = pos.get('entry', 0) if isinstance(pos, dict) else 0
            current_price = pos.get('current', entry_price) if isinstance(pos, dict) else entry_price
            shares = pos.get('shares', 0) if isinstance(pos, dict) else 0

            scan = next((s for s in scans if s.ticker == w.ticker), None)
            regime = scan.regime_cat if scan else '?'
            conf = scan.confidence if scan else 0

            unrealized = shares * (current_price - entry_price)
            total_exit_pnl += unrealized

            print(f'    EXIT {w.ticker:>5s}  {shares:>6.1f}sh  '
                  f'entry=${entry_price:.2f}  close=${current_price:.2f}  '
                  f'P&L=${unrealized:>+8.2f}  '
                  f'reason: {regime} conf={conf:.2f}')
        print(f'    Total missed exit P&L: ${total_exit_pnl:>+.2f}')

    # Show enters
    if enters:
        print(f'\n  -- WOULD HAVE ENTERED ({len(enters)} positions) --')
        for w in sorted(enters, key=lambda x: x.raw_score, reverse=True):
            scan = next((s for s in scans if s.ticker == w.ticker), None)
            price = scan.current_price if scan else 0
            conf = scan.confidence if scan else 0
            persist = scan.persistence if scan else 0
            confirms = scan.confirmations if scan else 0
            print(f'    ENTER {w.ticker:>5s}  score={w.raw_score:.3f}  '
                  f'weight={w.target_weight:.1%}  notional=${w.notional_usd:.0f}  '
                  f'@ ${price:.2f}  conf={conf:.2f}  persist={persist}d  '
                  f'confirms={confirms}/8')

    # Show holds
    if holds:
        print(f'\n  -- WOULD HAVE HELD ({len(holds)} positions) --')
        for w in sorted(holds, key=lambda x: x.ticker):
            pos = positions.get(w.ticker, {})
            entry_price = pos.get('entry', 0) if isinstance(pos, dict) else 0
            current_price = pos.get('current', entry_price) if isinstance(pos, dict) else entry_price
            shares = pos.get('shares', 0) if isinstance(pos, dict) else 0
            unrealized = shares * (current_price - entry_price)
            print(f'    HOLD {w.ticker:>5s}  {shares:>6.1f}sh  '
                  f'P&L=${unrealized:>+8.2f}  action={w.action}')

    # Summary
    print(f'\n  -- IMPACT SUMMARY --')
    if exits:
        print(f'  Missed exits would have realized: ${total_exit_pnl:>+.2f}')
        exited_tickers = [w.ticker for w in exits]
        print(f'  Tickers that should have been exited: {", ".join(exited_tickers)}')
        print(f'  (These positions were held an extra day due to the outage)')

    conn.close()
    print()


if __name__ == '__main__':
    main()
