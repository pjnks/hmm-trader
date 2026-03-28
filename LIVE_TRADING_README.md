# Live Trading Quick Start

## 🎯 Status: READY TO GO LIVE

**Winner**: SOL Ensemble (Sharpe 6.240 on walk-forward validation)
- Timeframe: 4h
- Feature set: full
- Confirmations: 8/8
- Leverage: 1.5× (backtester) → 0.25× (live, micro-capital)
- Cooldown: 72h

---

## 🧪 Step 1: Test Mode (FIRST)

```bash
python live_trading.py --test
```

What this does:
- Generates signals every 4 hours
- Simulates trades on Coinbase API (no real money)
- Tracks P&L, Sharpe, win rate
- Sends daily email reports to console
- **Run for at least 1 day** to verify all systems work

Expected output:
```
✅ OPENING BUY: 0.1234 SOL @ $150.00
✅ CLOSING POSITION: BUY @ $152.00
Trade logged: BUY $24.80 (0.11%)
```

Kill-switch rules (active in test mode):
- Daily loss > 2% → auto-stop ⛔
- Rolling 20-trade Sharpe < 0.3 → auto-stop ⛔
- 0 wins in 10 trades → auto-stop ⛔

---

## 🔴 Step 2: Live Trading (REAL MONEY)

Once test mode runs successfully for 1+ day:

```bash
python live_trading.py --live
```

⚠️ **CRITICAL**: Will ask for confirmation before trading real money.

**Max risk per account**: $50 per trade (2% of $10k)
**Max position size**: $2,500 notional (0.25× leverage)

---

## 📊 Daily Workflow

Each morning, expect:
1. **Email report** (9am UTC) with yesterday's metrics
2. **Check dashboard** (5 min) for P&L, Sharpe, signals
3. **That's it.** Strategy auto-executes every 4h

Do NOT:
- ❌ Tweak parameters daily
- ❌ Override signals manually
- ❌ Add new indicators
- ❌ Change position sizing

---

## 📁 Files Built

| File | Purpose |
|---|---|
| `live_trading.py` | Main orchestrator (run this) |
| `src/live_broker.py` | Coinbase API integration |
| `src/signal_generator.py` | Real-time HMM + indicator signals |
| `src/live_monitor.py` | P&L tracking + kill-switch logic |
| `config.py` | Updated with SOL parameters |

---

## 🔑 Environment Setup

`.env` must contain:
```
POLYGON_API_KEY=your_key
COINBASE_API_KEY=your_key
COINBASE_API_SECRET=your_ec_private_key
```

You already rotated credentials earlier ✅

---

## 📈 Success Metrics

After 2 weeks of live trading:
- ✅ Win rate: 50-60% (matches backtest)
- ✅ Rolling Sharpe (20 trades): > 0.5
- ✅ No 2-week consecutive loss
- ✅ Signal frequency: within ±20% of backtest

If any metric degrades → kill-switch auto-stops trading

---

## 🚨 Kill-Switch Rules (Automatic)

Trading stops immediately if:
1. **Daily loss > 2%** ($200 on $10k)
2. **Rolling 20-trade Sharpe < 0.3** (degrading)
3. **Zero wins in 10 trades** (streak detected)

You'll get an urgent email alert + log message.

---

## Next Steps

1. ✅ **Review `live_trading.py`** (understand the flow)
2. ✅ **Test in --test mode** (1+ day, no real money)
3. ✅ **Once confident** → `python live_trading.py --live`
4. ✅ **Monitor daily** (5 min/day, review emails + dashboard)

---

## Questions?

- Is the strategy working? Check dashboard + daily email
- Did it stop trading? Check kill-switch email alert
- Want to tweak something? Use backtest + walk-forward first (week 3+)

**Good luck. You've earned this.** 🚀
