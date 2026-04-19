
# HMM Regime Trader — Claude Code Instructions

## Project Purpose
A Hidden Markov Model (HMM) based trading system with three active sub-projects:

- **AGATE** (Crypto Multi-Ticker) — Grounded, stable (like the gemstone). Ensemble HMM crypto trading on 14 Coinbase derivatives (BTC, ETH, SOL, XRP, LTC, SUI, DOGE, ADA, BCH, LINK, HBAR, XLM, AVAX, ENA). **Sprint 7 COMPLETE (2026-03-26)**: paper P&L tracking via `open_positions` table, entry/exit logging, deduplication guard. First live BUY: ENAUSD (8/8 confirmations, 0.98 confidence). **Sprint 8 (2026-03-28)**: discovered 4 tickers (LTC, SUI, DOGE, ADA) had only ~95d cached data (added Sprint 6, 2025-12-22) — too short for 9-month WF optimizer (need 270d+). Backfilling to 730d, then re-optimizing. 12/14 tickers have per-ticker configs; LTC/SUI/DOGE/ADA will be added post-backfill. Currently in test mode scanning every 4h.
- **BERYL** (NDX100 Single-Stock) — Rare, transparent (like the gemstone). Ensemble HMM regime trading on NDX100 equities. Phase 3 COMPLETE (242 trials). **Sprint 5 COMPLETE**: 98-ticker ensemble rotation, multi-position (up to 3), 365-day lookback (100% convergence), smart insider scoring (10b5-1 filtering). 1 closed trade: ARM +$23.11 (+2.77%). Live test running (98 tickers, ensemble HMM). **Intraday risk checks fixed (Sprint 7)**: now uses `fetch_latest_price()` Polygon Snapshot API (was silently failing with TypeError).
- **CITRINE** (NDX100 Portfolio Rotation) — Pale yellow, luminous (like the gemstone). Confidence-weighted portfolio rotation across 100 NDX100 tickers. Daily HMM scan → adaptive allocation → risk parity rebalancing. Walk-forward validated (39 tickers, long-only, Sharpe +1.341). **Sprint 7 COMPLETE (2026-03-26)**: (1) cash band enforced as hard floor, (2) persistence bonus replaced with sojourn decay from HMM transition matrix, (3) continuous cash scaling (replaces rigid bands), (4) intraday price fix. Live test: $24,892 equity, 10 positions, 38% cash, 108 trades. Re-optimizer: 585 trials, 82/99 positive Sharpe.
- **DIAMOND** (Kalshi Unusual Volume) — Clear, brilliant (like the gemstone). Real-time anomaly detection on Kalshi prediction markets. Standalone repo (`/Users/perryjenkins/Documents/quant/diamond/`). BUILT and DEPLOYED on Oracle Cloud VM. WebSocket streaming + 6-feature anomaly detection + paper trading engine + Dash dashboard at :8080.
- **EMERALD** (NBA Sports Predictions) — Green, vibrant (like the gemstone). XGBoost ensemble + Kelly criterion for NBA spreads, totals, and player props. Standalone repo (`/Users/perryjenkins/Documents/quant/emerald/`). Daily ingest + pre-game odds capture via cron.
- Future sub-projects follow gemstone alphabetical naming: FLUORITE, GARNET, etc.

Core capabilities:
- Labels market regimes (BULL / BEAR / CHOP) from OHLCV features using GaussianHMM (single or ensemble)
- Generates entry signals (LONG and SHORT) gated by regime confidence + 8-indicator confirmation
- Supports multi-direction trading via RegimeMapper: (regime, confidence) → LONG/SHORT/FLAT
- Backtests on Polygon.io historical data (crypto + equities) with realistic fees and leverage
- Provides interactive Dash dashboards (backtest at :8050, live monitoring at :8060, CITRINE at :8070, DIAMOND at :8080)
- Includes ensemble HMM (3 models voting, n_states=[5,6,7]) for production trading
- Live trading infrastructure with kill-switch automation and multi-channel notifications

## Philosophy & North Star

> "The market has hidden states, model them, trade the transitions — this is the same insight Jim Simons had. You're just doing it at indie scale. And honestly, at small capital ($25k), you have advantages they don't: zero market impact, no regulatory burden, and you can enter/exit tiny-cap positions they can't touch."

### Indie Edge vs Institutional
- Renaissance (Medallion): 66% avg annual return since 1988, 300 PhDs, $130B AUM, 10,000+ GPUs
- HMM Trader: same core math (HMM regime detection), indie scale, Claude instead of PhDs
- **Small capital advantages**: zero market impact, no SEC reporting, no investor LPs to explain to, can trade micro-caps institutions can't touch, no compliance overhead
- **The 18%**: only 18% of hedge funds rely on ML for majority signal generation — you're in that camp
- **Most edge comes from data** (alternative data costs $500K-5M/year at institutional scale) **and speed** (co-located servers at exchanges), not just better algorithms. At indie scale, our edge is free/cheap alternative data + the structural advantages of small capital.

### Alternative Data Upgrade Roadmap

**Phase 1 (Free, $0)**: Congressional trades (Quiver Quantitative, Capitol Trades) + SEC insider filings (Form 4) as CITRINE filter. Only enter BULL positions where insiders also bought in last 90 days.

**Phase 2 ($50/mo)**: Unusual Whales — options flow + dark pool prints as confirmation layer. HMM detects BULL regime + unusual call buying confirms smart money agrees. Single highest-ROI data purchase for a retail trader.

**Phase 3 ($100-150/mo)**: Ortex (real-time short data) + AltIndex (sentiment). Five independent signal sources:
1. HMM regime (statistical)
2. Insider/Congress (informed money)
3. Options flow (institutional positioning)
4. Short interest (crowding risk)
5. Sentiment (crowd psychology)

**Key insight**: The highest-ROI data for retail is data about what *other market participants* are doing — they have information you don't. Congress members have non-public legislative knowledge. Corporate insiders know their own business. Options market makers see institutional order flow. Short sellers have done deep fundamental research. Your HMM detects the *effect* (price regimes). This data reveals the *cause* (who's moving and why). Combining both is the edge.

### DIAMOND ↔ CITRINE Cross-System Roadmap

DIAMOND currently detects *that* something unusual is happening. Alternative data tells you *why*. CITRINE tells you the broader regime context. Connecting all three transforms them from independent experiments into a cohesive trading system.

| Phase | What | Effort | Cost |
|-------|------|--------|------|
| **Phase 1** | Map Kalshi market tickers to equity tickers (e.g., "NVDA earnings" → NVDA) | 1-2 sessions | Free |
| **Phase 2** | Feed congressional/insider data into DIAMOND scoring | 2-3 sessions | Free |
| **Phase 3** | Cross-reference CITRINE regime signals with DIAMOND markets | 1-2 sessions | Free |
| **Phase 4** | Add Unusual Whales options flow as DIAMOND feature #7 | 1-2 sessions | $50/mo |

Phase 1 is the foundation — once Kalshi markets are linked to equity tickers, everything else flows naturally.

---

## 3-Month Aggressive Validation Plan (CURRENT PHASE)

**Goal**: Validate strategy on live market data and go live with small capital by week 13 (if performance supports it).

**Current Status (Weeks 1-2 — ALL 3 SPRINTS COMPLETE)**:
- **AGATE**: Running 14-ticker multi-crypto rotation in `--test` mode. **Sprint 6 COMPLETE (2026-03-22)**: expanded from SOL-only to 14 Coinbase derivatives, adaptive confirmations (cf-1 at conf>0.90), per-ticker optimized configs from 179-trial optimizer, near-miss alerts. Best ticker: BCH (WF Sharpe +3.52, all 5 windows positive). Zero trades yet — max 5/7 confirmations.
- **BERYL**: Phase 1 COMPLETE (195/200, TSLA Sharpe 0.921). Phase 2 COMPLETE (NVDA Sharpe 1.094). Phase 3 COMPLETE (196/200 — AAPL Sharpe 0.733, NVDA 0.652, TSLA 0.634; daily timeframe dominates equities 53% positive vs 10-19% intraday). Multi-ticker rotation live test running (NVDA, TSLA, GOOGL, MSFT, AAPL).
- **CITRINE**: Allocator optimization COMPLETE (entry=0.90, exit=0.50, Sharpe +1.665). Per-ticker re-optimization COMPLETE (585 trials, 82/99 positive Sharpe — was 68). State persistence bug fixed. Live test restarted 2026-03-17 with 82 optimized tickers, long-only mode, $25k capital.
- **Sprint 1 critical bugs fixed**: (1) AGATE signal_generator always returned HOLD — fixed to use `raw_long_signal`, (2) AGATE used single HMM in live trading — fixed to use EnsembleHMM, (3) CITRINE lost positions on restart — fixed with `_restore_state_from_db()`, (4) CITRINE mark-to-market used stale entry prices — fixed to use scan prices.
- **Sprint 2 infrastructure**: Test harness built (69 tests, all passing). Reconciliation script (`reconcile.py`) validated live vs backtest signals.
- **Sprint 3 model fine-tuning**: (1) New `extended_v2` feature set with `realized_kurtosis` (+2.747 Sharpe A/B test), (2) HMM `n_init=3` covariance regularization, (3) CITRINE sector cap max 4/sector, (4) Stability test suite (`stability_test.py`), (5) CITRINE intra-day kill-switch every 4h, (6) Dashboard position health + signal frequency. Stability test confirmed extended_v2/7cf as optimal (+0.837 vs -4.668).
- **Pivot**: AGATE skipped paper trading → moved directly to micro-capital live ($2.5k max notional, 0.25x effective leverage). Rationale: ensemble walk-forward Sharpe strong enough to justify real-money validation.
- **Single-model XRP**: FAILED validation (walk-forward Sharpe -1.893 vs backtest 3.512 — massive gap). Replaced by ensemble approach.

**Revised Phases**:
1. **Week 1**: AGATE ensemble optimization + BERYL NDX100 optimization (DONE)
2. **Weeks 2-4**: AGATE live test mode (simulated fills, real data, monitoring active)
3. **Weeks 5-8**: AGATE micro-capital live ($2.5k max notional) IF test mode validates
4. **Weeks 9-12+**: Scale to 0.5x leverage IF Sharpe > 0.5 over 4 weeks
5. **Parallel**: BERYL optimization continues; enters test mode when ready

**Success Criteria**:
- Live/test Sharpe > 0.5 (at least 1/12th of ensemble backtest)
- Win rate 50-60% (matches backtest prediction)
- Signal frequency ±20% of backtest
- Max drawdown < 20%
- No 2-week consecutive loss

**Honest Probability Estimate**: 30-40% chance of live trading with positive Sharpe by end of 3 months (assuming test mode validates backtest edge).

*See `/Users/perryjenkins/.claude/plans/delegated-tumbling-feather.md` for detailed implementation plan.*

---

## Risk Disclaimers & Backtest-to-Live Degradation

**Critical Reality Check**: Ensemble backtest Sharpe 6.240 → expect live Sharpe 0.5-1.5 (75-90% degradation). Single-model XRP already failed (backtest 3.512, live -1.893).

Why the gap?
- **Slippage**: Backtest assumes perfect fills. Real fills are 5-20 bps worse per side. Test mode simulates 10bps.
- **Latency**: Signal computed on bar-close. By order placement, price has moved 10-50 bps against you.
- **Regime shift**: Model trained on 2024-2025 SOL data. Market conditions change constantly.
- **Crowding**: If others trade the same pattern, alpha compresses or disappears.
- **Ensemble overhead**: 3 HMMs must agree — majority voting reduces signal frequency vs single model.

**Kill-Switch Rules** (automatic, non-negotiable):
1. **Daily loss > 2% of account** → exit all positions immediately
2. **Rolling 5-trade Sharpe < 0.3** → all positions auto-exit, reassess strategy
3. **Consecutive 2-week flat/negative** → stop trading, debug root cause
4. **Slippage > 2× expected** → liquidate, check market conditions

**No Live Trading Unless**:
- [ ] Paper trading completed (4+ weeks with Sharpe > 0.5)
- [ ] Multi-asset validation passed (≥2 of 4 tickers positive)
- [ ] Coinbase CDP credentials in `.env` (verified working on testnet)
- [ ] Position sizing limits configured (max 2% loss per trade)
- [ ] Monitoring infrastructure deployed (daily email alerts, Sharpe dashboard)
- [ ] User explicitly approves going live

---

## Monitoring Infrastructure

### Notification Channels
- **Pushover phone push** (primary on VM) — kill-switch alerts sent as emergency priority (repeated until acknowledged)
- **Terminal bell** — fallback for all events when running in terminal
- **macOS native notifications** (`src/notifier.py` via `osascript`) — Mac-only; auto-detected, silently skipped on Linux VM
- **Kill-switch modal dialog** — macOS-only (`osascript` display dialog) + emergency Pushover push
- `src/notifier.py` auto-detects OS via `platform.system()` — no more `[Errno 2] No such file or directory: 'osascript'` warnings on Linux

### AGATE Live Monitoring
- **`src/live_monitor.py`** — SQLite-backed trade logging (`paper_trades.db`), rolling Sharpe calculation
- **3 kill-switch rules** (automatic, non-negotiable):
  1. Daily loss > 2% of account → exit all positions immediately
  2. Rolling 5-trade Sharpe < 0.3 → all positions auto-exit, reassess strategy
  3. Win rate 0/10 (zero wins in last 10 trades) → stop trading, debug root cause
- **Live dashboard** (`live_dashboard.py`) at port 8060 — equity curve, kill-switch status, trade table, BERYL progress, regime status panels (BULL/BEAR/CHOP badges for AGATE and BERYL)
- **Daily report** (`daily_report.py`) — AGATE + BERYL status, maturity scoring (0-100%), visual progress bars
- **Cron job** — `run_daily_report.sh` runs at 6pm daily with macOS notification

### BERYL Monitoring (When Active)
- Progress tracked in `beryl_log.md` (updated daily during optimization)
- Daily report includes BERYL optimization status and best Sharpe

### Visual Interfaces
- **Live Dashboard** (`live_dashboard.py`) at `http://129.158.40.51:8060` — equity curve, kill-switch status, trade table, regime panels (BULL/BEAR/CHOP badges)
- **CITRINE Dashboard** (`citrine_dashboard.py`) at `http://129.158.40.51:8070` — portfolio equity, allocation, position health, signal frequency, kill-switch
- **DIAMOND Dashboard** (`diamond_dashboard.py`) at `http://129.158.40.51:8080` — anomaly feed, paper trading portfolio
- **Consolidated Dashboard** (`consolidated_dashboard.py`) at `http://129.158.40.51:8090` — retro-futuristic bento-grid overview of all projects (Swiss Design palette, glassmorphism cards, 12-column CSS Grid, JetBrains Mono font; redesigned Sprint 8)
- **Daily Report** (`python daily_report.py`) — terminal report with maturity scores, project status (Mac-only)
- **Check-In Dialogs** (`python daily_report.py --checkin`) — 5 macOS dialogs with live data, streak tracking (Mac-only)
- All dashboards include HTTP `Cache-Control: no-store` headers to prevent stale browser caching
- **BERYL P1 Heatmap** (`beryl_optimization_heatmap.html`) — interactive Plotly heatmap of Phase 1 results
- **BERYL P2 Heatmap** (`beryl_p2_heatmap.html`) — Phase 2 results (ticker × timeframe)
- **AGATE WF Heatmap** (`optimization_wf_heatmap.html`) — walk-forward optimization results

### Automated Schedule (cron)
- **8:45 AM** — Daily report → macOS + phone notification
- **5:30 PM** — Daily report → macOS + phone notification
- **9:30 PM** — Accountability check-in → 5 macOS dialogs (user at MacBook)
- **12:00 AM** — Maturity snapshot → saves scores to `maturity_history.csv`

---

## Oracle Cloud Deployment (2026-03-20)

All 4 sub-projects run 24/7 on an Oracle Cloud Free Tier VM as systemd services.

### VM Details
| Property | Value |
|----------|-------|
| Shape | VM.Standard.E2.1.Micro (Always Free) |
| CPU | 1 OCPU (AMD EPYC) |
| RAM | 1 GB + 4 GB swap |
| Disk | 100 GB boot volume (~89 GB free) |
| OS | Ubuntu 22.04 |
| IP | `129.158.40.51` |
| Python | `/home/ubuntu/miniconda3/bin/python` (Python 3.13) |

### SSH Access
```bash
ssh -i ~/.ssh/hmm-trader.key ubuntu@129.158.40.51
```

### 8 systemd Services
| Service | Command | RAM | Port |
|---------|---------|-----|------|
| `agate-trader` | `python live_trading.py --test` | ~110 MB | — |
| `beryl-trader` | `python live_trading_beryl.py --test` | ~183 MB | — |
| `citrine-trader` | `python live_trading_citrine.py --test --long-only --tickers ...` | ~28 MB | — |
| `diamond-monitor` | `python diamond_monitor.py` | ~30 MB | — |
| `live-dashboard` | `python live_dashboard.py --host 0.0.0.0 --port 8060` | ~19 MB | 8060 |
| `citrine-dashboard` | `python citrine_dashboard.py` | ~48 MB | 8070 |
| `diamond-dashboard` | `python diamond_dashboard.py` | ~83 MB | 8080 |
| `consolidated-dashboard` | `python consolidated_dashboard.py` | ~93 MB | 8090 |

All services: `Restart=always`, `RestartSec=30`, `WorkingDirectory=/home/ubuntu/HMM-Trader` (or `/home/ubuntu/kalshi-diamond` for DIAMOND).

### Service Management
```bash
# Check all services
sudo systemctl status agate-trader beryl-trader citrine-trader diamond-monitor live-dashboard citrine-dashboard diamond-dashboard

# Restart a service
sudo systemctl restart agate-trader

# View logs
sudo journalctl -u agate-trader -f --no-pager

# Enable/disable
sudo systemctl enable agate-trader   # start on boot
sudo systemctl disable agate-trader  # don't start on boot
```

### Project Folder Structure (Mac — updated 2026-03-24)
```
~/Documents/quant/                    # Master quant folder
├── trading-core/                     # Shared AGATE+BERYL+CITRINE codebase (was HMM-Trader)
├── agate/                            # Crypto project folder (symlinks to trading-core scripts)
├── beryl/                            # NDX single-stock folder (symlinks to trading-core scripts)
├── citrine/                          # NDX portfolio folder (symlinks to trading-core scripts)
├── diamond/                          # Kalshi anomaly detection (standalone, was kalshi-diamond)
├── emerald/                          # NBA sports predictions (standalone, was sports-predictions)
└── hub/                              # Cross-project dashboards + orchestration
```
Backward-compat symlinks preserve old paths:
- `~/Documents/trdng/HMM-Trader` → `~/Documents/quant/trading-core/`
- `~/Documents/trdng/kalshi-diamond` → `~/Documents/quant/diamond/`
- `~/Documents/sports-predictions` → `~/Documents/quant/emerald/`

### VM File Locations
| Mac Path | VM Path |
|----------|---------|
| `~/Documents/quant/trading-core/` | `/home/ubuntu/HMM-Trader/` |
| `~/Documents/quant/diamond/` | `/home/ubuntu/kalshi-diamond/` |
| `.env` (API keys) | `/home/ubuntu/HMM-Trader/.env`, `/home/ubuntu/kalshi-diamond/.env` |

### Firewall (Oracle Cloud Security List)
Ports 8060, 8070, 8080, and 8090 open for dashboard access. AGATE/BERYL/CITRINE trading services don't expose ports. **Important**: Security List rules must be on the correct Security List — the one attached to the instance's subnet. Navigate: Compute → Instances → instance → Primary VNIC → Subnet → Security Lists.

### Version Control (GitHub)
**Repo**: `github.com/pjnks/hmm-trader` (private), **Branch**: `main`
**Auto-push**: launchd plist `com.quant.autopush-trading-core` commits + pushes every 4h.
`.gitignore` excludes: `.env*`, `*.db*`, `data_cache/`, `*.pkl`, `*.html`, `*.log`, `*_results.csv`, `*_status.json`, `*.pptx`, `node_modules/`, `__pycache__/`, `.claude/`
Auto-push requires Full Disk Access for `/bin/bash` (System Settings > Privacy & Security).

### Deploying Code Updates
```bash
scp -i ~/.ssh/hmm-trader.key <file> ubuntu@129.158.40.51:/home/ubuntu/HMM-Trader/
ssh -i ~/.ssh/hmm-trader.key ubuntu@129.158.40.51 "sudo systemctl restart <service-name>"
```

### RAM Constraints
Total ~587 MB used of 956 MB available + 4 GB swap. Tight but stable. If more services needed, consider:
- Oracle A1.Flex (free, 4 OCPU/24GB ARM) in another region
- Hetzner CCX23 (~$25/mo, 4 dedicated CPU/16GB RAM) — could consolidate everything including optimizers

### Log Rotation
Service logs (`*_test.log`, `*_dashboard.log`) use logrotate at `/etc/logrotate.d/hmm-trader`:
- Rotate daily, keep 7 days
- Compress rotated logs
- Max size 50MB per log

### VM Python Environment
**On VM, always use miniconda Python** — NOT the system `python3`:
```bash
/home/ubuntu/miniconda3/bin/python   # Python 3.13 — has all deps
python3                               # System Python — MISSING deps, do NOT use
```

---

## Auto-Optimization System (2026-03-22)

Three optimizer scripts run every Sunday on the Mac (not VM — optimizers are CPU-intensive and need M1 Air).

**Uses launchd** (migrated from crontab 2026-03-22 — crontab silently skips jobs when Mac sleeps; launchd catches up on wake):

| Day | Time | Project | Script | Duration |
|-----|------|---------|--------|----------|
| Sunday | 2:00 AM | CITRINE | `run_optimize_citrine.sh` | ~60 min |
| Sunday | 4:00 AM | BERYL | `run_optimize_beryl.sh` (→ `optimize_beryl_daily.py`) | ~2-3 hours |
| Sunday | 8:00 AM | AGATE | `run_optimize_agate.sh` | ~3-5 hours |

Each script:
- Runs the optimizer with `--resume` (incremental — adds trials to existing CSV)
- Uses `--workers 6` for multiprocessing (default: `cpu_count()-2` = 6 on M1 Air). Expected 5-6x speedup vs single-threaded.
- Logs output to `optimization_<project>.log`
- Extracts top results from CSV
- Sends Pushover + macOS notification with summary
- CITRINE auto-updates `citrine_per_ticker_configs.json` (hot-swappable, no restart needed)
- BERYL auto-updates `beryl_per_ticker_configs.json` (auto-generated by `optimize_beryl_daily.py`)
- AGATE results require manual review before applying

launchd plist agents (Mac):
```
~/Library/LaunchAgents/com.hmm-trader.optimize-citrine.plist  (Sunday 2am)
~/Library/LaunchAgents/com.hmm-trader.optimize-beryl.plist    (Sunday 4am)
~/Library/LaunchAgents/com.hmm-trader.optimize-agate.plist    (Sunday 8am)
```

Management:
```bash
# List agents
launchctl list | grep hmm-trader

# Manually trigger (test)
launchctl start com.hmm-trader.optimize-citrine

# Unload/reload
launchctl unload ~/Library/LaunchAgents/com.hmm-trader.optimize-citrine.plist
launchctl load ~/Library/LaunchAgents/com.hmm-trader.optimize-citrine.plist
```

---

## Python Environment (Mac — for development and optimization)
**Always use the base conda environment.** It contains all dependencies.

```bash
/Users/perryjenkins/opt/anaconda3/bin/python   # Python 3.9.12
```

Never use `mynewenv` — it is missing hmmlearn, dash, plotly, and other core deps.

To confirm the right interpreter is active:
```bash
which python   # should show /Users/perryjenkins/opt/anaconda3/bin/python
```

## Live Trading Checklist (Before Week 11 Go-Live)

### Pre-Live Requirements
- [ ] Paper trading shows Sharpe > 0.5 (required)
- [ ] Win rate 50-60% (matches backtest prediction)
- [ ] Max drawdown < 20% during paper trading
- [ ] Signal frequency within ±20% of backtest
- [ ] NO 2-week consecutive loss
- [ ] Coinbase CDP credentials verified on testnet
- [ ] Position sizing formula implemented (max 2% loss per trade)
- [ ] Kill-switch automation deployed (auto-exit at -2% or Sharpe < 0.3)
- [ ] Monitoring system live (email alerts, dashboard, Sharpe tracking)
- [ ] User explicitly approves going live

### Position Sizing Formula
```python
position_size_usd = min(
    account_equity * 0.02 * leverage,  # e.g., $10K * 0.02 * 0.25 = $50 risk
    account_equity * 0.10  # Never risk more than 10% per trade
)
```

### Scaling Schedule (If Live Sharpe > 0.5)
- **Weeks 11-12**: 0.25× leverage ($2.5K notional on $10K account)
- **Week 13+ (1 month positive)**: Scale to 0.5× leverage ($5K notional)
- **Month 2 (2 months positive)**: Scale to 1.0× leverage ($10K notional)
- **Never exceed 1.0× leverage** without 3+ months of positive live returns

---

## How to Run

```bash
# Full pipeline + Dash dashboard at http://127.0.0.1:8050
python main.py

# Run backtester only, print results to terminal (no browser)
python main.py --no-dash

# Force HMM re-training (delete model_cache.pkl and retrain)
python main.py --retrain

# Multi-direction mode (LONG/SHORT/FLAT via regime mapper)
python main.py --no-dash --regime-mapper

# Custom host/port
python main.py --host 0.0.0.0 --port 9000

# Random-search hyperparameter optimizer (300 trials by default)
python optimize.py

# Fewer trials (fast test)
python optimize.py --runs 5

# Resume from existing CSV checkpoint
python optimize.py --runs 300 --resume

# Regenerate heatmap HTML only (no re-running trials)
python optimize.py --heatmap-only

# Walk-forward optimizer (multi-ticker, multi-timeframe, 6m train / 3m test)
python optimize_wf.py                   # 100-combination random search
python optimize_wf.py --runs 50         # fewer combinations
python optimize_wf.py --resume          # continue from CSV checkpoint
python optimize_wf.py --heatmap-only    # regenerate heatmap only

# Standalone walk-forward analysis (single config, verbose output)
python walk_forward.py                          # BTC 1h defaults
python walk_forward.py --ticker X:ETHUSD --timeframe 4h
python walk_forward.py --train-months 6 --test-months 3
python walk_forward.py --regime-mapper          # multi-direction walk-forward

# Walk-forward report from optimizer results
python -m src.wf_report
python -m src.wf_report --top 20

# Stability testing suite (replaces robustness.py — Sprint 3)
python stability_test.py --project agate           # Run all 3 stability tests for AGATE
python stability_test.py --project beryl            # Run all 3 stability tests for BERYL
python stability_test.py --project citrine          # Run all 3 stability tests for CITRINE
python stability_test.py --project agate --quick    # Quick mode (fewer perturbations/seeds)

# AGATE: Live trading (test mode — simulated fills, real data)
python live_trading.py --test

# AGATE: Live trading (real money — requires explicit user approval)
python live_trading.py --live

# BERYL: NDX100 optimizer
python optimize_beryl.py --runs 200
python optimize_beryl.py --runs 200 --resume

# BERYL: Phase 2 optimizer (TSLA+NVDA, intraday timeframes)
python optimize_beryl_p2.py --runs 200
python optimize_beryl_p2.py --runs 200 --resume

# BERYL: Live trading test mode (multi-ticker rotation)
python live_trading_beryl.py --test                                       # Single ticker (default config)
python live_trading_beryl.py --test --tickers NVDA,TSLA,GOOGL,MSFT,AAPL # Multi-ticker rotation (production)

# CITRINE: Portfolio rotation optimizer (per-ticker HMM param optimization)
python optimize_citrine.py --runs 200          # 200 random trials from 36 combos per ticker
python optimize_citrine.py --runs 200 --resume # Resume from checkpoint
python optimize_citrine.py --ticker AAPL       # Single ticker test
python optimize_citrine.py --heatmap-only      # Regenerate heatmap from results

# CITRINE: Walk-forward backtest (portfolio level, multiple scenarios)
python citrine_backtest.py --tickers NVDA,AAPL,TSLA,MSFT,AMZN,GOOGL,META,AMGN,PEP,COST # default (10 tickers)
python citrine_backtest.py --long-only          # Long-only vs long+short
python citrine_backtest.py --cooldown time      # Time-based cooldown (vs default none)
python citrine_backtest.py --cooldown threshold # Confidence-threshold cooldown
python citrine_backtest.py --quiet              # Suppress verbose output

# CITRINE: Live trading test mode (daily scan + rebalance)
python live_trading_citrine.py --test                                    # Full NDX100
python live_trading_citrine.py --test --tickers NVDA,AAPL,TSLA,MSFT    # Subset
python live_trading_citrine.py --test --long-only                       # Long-only mode
python live_trading_citrine.py --test --cooldown time                   # Time-based cooldown

# CITRINE: Portfolio dashboard at http://127.0.0.1:8070 (auto-refresh 60s)
python citrine_dashboard.py

# AGATE: Indicator threshold optimizer (81 combos: adx_min, stoch_upper, volume_mult, volatility_mult)
python optimize_indicators.py --ensemble          # Ensemble mode (production)
python optimize_indicators.py --runs 81 --resume  # Resume from checkpoint

# BERYL: Phase 3 optimizer (5 tickers: TSLA/NVDA/AAPL/MSFT/GOOGL, expanded grid)
python optimize_beryl_p3.py --runs 200
python optimize_beryl_p3.py --runs 200 --resume

# CITRINE: Allocator parameter optimizer (pre-compute HMMs once, replay 3,456 allocation combos)
python optimize_allocator.py                      # Full 3,456-combo grid (~38 min)
python optimize_allocator.py --resume             # Resume from checkpoint

# Tests: Strategy consistency test suite (69 tests, no external API calls)
/Users/perryjenkins/opt/anaconda3/bin/python -m pytest tests/test_strategy_consistency.py -v

# Reconciliation: Compare live trading signals vs backtester output
python reconcile.py                               # Runs AGATE + BERYL + CITRINE reconciliation

# Daily report (runs automatically at 6pm via cron, now includes AGATE+BERYL+CITRINE)
python daily_report.py
python daily_report.py --snapshot  # Save maturity scores to history (cron at midnight)
python daily_report.py --checkin   # Interactive 5-step accountability check-in

# Live monitoring dashboards (run on VM as systemd services, not locally)
# http://129.158.40.51:8060  — AGATE + BERYL
# http://129.158.40.51:8070  — CITRINE portfolio
# http://129.158.40.51:8080  — DIAMOND anomaly feed

# Test notification channels (macOS popup + Pushover)
python -m src.notifier
```

## Current Best Configuration (config.py) — AGATE Multi-Ticker (Sprint 6)

Multi-ticker rotation across 14 Coinbase derivatives. Per-ticker configs in `agate_per_ticker_configs.json`. Default fallback config below.

| Parameter | Value | Notes |
|---|---|---|
| `AGATE_TICKERS` | 14 tickers | BTC, ETH, SOL, XRP, LTC, SUI, DOGE, ADA, BCH, LINK, HBAR, XLM, AVAX, ENA |
| `TICKER` | `"X:SOLUSD"` | Legacy default (multi-ticker overrides this) |
| `N_STATES` | 6 | HMM hidden states (ensemble uses [5,6,7]) |
| `COV_TYPE` | `"diag"` | Diagonal covariance |
| `FEATURE_SET` | `"extended_v2"` | A/B tested: +5.0 Sharpe vs extended on current data |
| `TIMEFRAME` | `"4h"` | 4-hour bars |
| `DAYS_HISTORY` | 730 | ~2 years of data |
| `MIN_CONFIRMATIONS` | 7 | Stability-tested optimum (was 6, was 8) |
| `MIN_CONFIRMATIONS_SHORT` | 7 | SHORT entry gate (independent from LONG) |
| `REGIME_CONFIDENCE_MIN` | 0.70 | HMM posterior probability gate |
| `LEVERAGE` | 1.5 | Backtest multiplier (live uses 0.25x effective) |
| `COOLDOWN_HOURS` | 72 | Mandatory wait after LONG exit |
| `COOLDOWN_HOURS_SHORT` | 48 | Mandatory wait after SHORT exit |
| `ADX_MIN` | 20 | Min trend strength for LONG (updated Sprint 1 — was 25, +17% Sharpe improvement) |
| `ADX_MIN_SHORT` | 25 | Min trend strength for SHORT |
| `TAKER_FEE` | 0.0006 | 0.06% per side |
| `INITIAL_CAPITAL` | 10,000 | USD |

**Walk-Forward Results (ensemble, long-only):** WF Sharpe 6.240 (SOL/4h/6st/diag/full/8cf/1.5x/72h)
**Stability Test (2026-03-17):** extended_v2/7cf → WF Sharpe +0.837 on current data (extended/6cf → -4.668, regime degradation)
**Previous best (single-model XRP, now superseded):** WF Sharpe 3.512, +56.8% return (XRP/2h/6st/full/full/7cf/2.0x/48h) — FAILED live validation (Sharpe -1.893)
**Walk-Forward Results (multi-dir, best consistent):** WF Sharpe 1.409, +10.6% return, both OOS windows positive (XRP/4h/6st/ext/8cf, FLAT/SHORT, cs=7, cds=72, adx_s=25)

**Regime-Direction Mapping** (for `--regime-mapper` mode):
| Regime | Confidence | Direction |
|--------|-----------|-----------|
| BULL | ≥ 0.85 | LONG |
| BULL | 0.70–0.85 | LONG_OR_FLAT |
| BEAR | ≥ 0.85 | SHORT |
| BEAR | 0.70–0.85 | SHORT_OR_FLAT |
| CHOP | any | FLAT |

**FEATURE_SETS available:**
- `"base"`: `log_return`, `price_range`, `volume_change` (3 features)
- `"extended"`: base + `realized_vol_ratio`, `return_autocorr` (5 features — `vol_price_diverge` removed Sprint 1, it is binary and causes degenerate HMM covariance)
- `"extended_v2"`: base + `realized_vol_ratio`, `return_autocorr`, `realized_kurtosis`
- `"full"`: extended + `candle_body_ratio`, `bb_width` (7 features — was 8, `vol_price_diverge` removed Sprint 1)
- `"full_v2"`: extended_v2 + `candle_body_ratio`, `bb_width`, `volume_return_intensity`, `return_momentum_ratio`
- `"atr_normalized"`: `atr_norm_return`, `atr_norm_range`, `atr_norm_volume`, `realized_vol_ratio`, `return_autocorr`, `realized_kurtosis` (6 features — ATR-normalized inputs for universal cross-asset HMM config; Sprint 8)

## CITRINE Configuration (config.py) — NDX100 Portfolio Rotation

Walk-forward validated (6m train / 3m test, 10 tickers, daily bars).

| Parameter | Value | Notes |
|---|---|---|
| `CITRINE_UNIVERSE` | 100 NDX tickers | Full NDX100 coverage (100 stocks, 22 sectors) |
| `CITRINE_ENTRY_CONFIDENCE` | 0.90 | Entry threshold (new positions — updated Sprint 1, was 0.80) |
| `CITRINE_EXIT_CONFIDENCE` | 0.50 | Exit threshold (existing positions — updated Sprint 1, was 0.65) |
| `CITRINE_PERSISTENCE_DAYS` | 3 | Min consecutive days in regime (new entries) |
| `CITRINE_MAX_POSITIONS` | 15 | Max open positions at any time (updated Sprint 1, was 10) |
| `CITRINE_MAX_NOTIONAL` | $5,000 | Max per-position notional (0.2% account risk max) |
| `CITRINE_LONG_ONLY` | False | Support both LONG and SHORT positions |
| `CITRINE_COOLDOWN_MODE` | "none" | Hysteresis only; testable modes: "time" (5 days), "threshold" |
| `CITRINE_INITIAL_CAPITAL` | $25,000 | Portfolio capital |
| `CITRINE_TAKER_FEE` | 0.0004 | 0.04% per side (equities) |
| `CITRINE_SLIPPAGE_BPS` | 10 | 10bps simulated slippage |
| `CITRINE_CASH_BANDS` | Dynamic | Adaptive cash % by BULL count (80% cash if 0-2 BULL) |

**Walk-Forward Results (10-ticker baseline, default params):** WF Sharpe -1.091, -9.09% total return, 2/8 positive windows, best window Sharpe 1.761
- **Window 7 (2024-12-30 to 2025-03-30)**: +2.13% vs BH -9.05%, **alpha +11.18%** (defensive in downturn)
- **302 total trades**, avg 1.2 open positions, 89% cash (conservative with 10 tickers)
- Backtest optimizes: daily bars > intraday, per-ticker HMM params matter, confidence-weighted allocation works

**CITRINE Scoring Formula:**
```
citrine_score = confidence_weight × inverse_vol × indicator_quality × persistence_bonus

Where:
  confidence_weight = clip((confidence - 0.65) / 0.25, 0, 1)
  inverse_vol       = clip(median_vol / realized_vol, 0.5, 2.0)  [risk parity]
  indicator_quality = confirmations / 8
  persistence_bonus = 1.0 + min(max(persistence - 3, 0) × 0.10, 0.50)
```

**Hysteresis Bands (updated Sprint 1 — allocator optimization winner):**
- Entry: confidence ≥ 0.90 + persistence ≥ 3 days (was 0.80)
- Exit: confidence ≥ 0.50 (existing positions — was 0.65)
- Dead zone: 0.50–0.90 (hold if in, don't enter if out — wider band is the key improvement: 0.40 gap vs 0.15)

**Adaptive Cash Management:**
- 15+ BULL tickers → 20% cash (very bullish)
- 8-14 BULL → 30% cash
- 3-7 BULL → 50% cash
- 0-2 BULL → 80% cash (very cautious)

**Gradual Position Scaling (updated Sprint 1 — fast scale profile):**
- Day 1: 50% of target weight (was 25%)
- Day 2+: 100% of target weight (was Day 5+)
- Note: slow scale (25%/50%/100% at day 1/3/5) is more robust (85% positive vs 75% for fast), but fast scale wins on mean Sharpe

## Project Status & Roadmap

### Completed Phases
1. **Phase 1 — Long-only HMM strategy** (core pipeline, backtester, dashboard, indicators)
2. **Phase 2 — Multi-direction trading** (RegimeMapper, SHORT signals, LONG/SHORT/FLAT positions)
3. **Phase 2.5 — Asymmetric SHORT params** (`MIN_CONFIRMATIONS_SHORT`, `COOLDOWN_HOURS_SHORT`, `ADX_MIN_SHORT`)
4. **Optimization** — 5 passes total: 2 simple IS/OOS, 1 WF long-only, 2 WF multi-dir (see Optimization History)
5. **Robustness Testing** — Replaced planned single-model robustness suite with ensemble approach. Single-model XRP FAILED live validation (backtest Sharpe 3.512, walk-forward Sharpe -1.893). Ensemble (3 HMMs, majority voting) solved this: SOL ensemble WF Sharpe 6.240.
6. **Live Trading Infrastructure** — `live_trading.py`, `src/live_broker.py`, `src/signal_generator.py`, `src/live_monitor.py`, `src/notifier.py`, `live_dashboard.py`, `daily_report.py`
7. **Ensemble Models** — `src/ensemble.py` implements 3-model ensemble with n_states=[5,6,7], majority regime voting, confidence averaging
8. **Sprint 1 — Critical Bug Fixes** (2026-03-17):
   - AGATE `signal_generator.py`: Fixed critical bug where `signal = latest.get("signal", "HOLD")` always returned HOLD — `build_signal_series()` never creates a "signal" column; fixed to use `raw_long_signal` + `regime_cat`
   - AGATE `signal_generator.py`: Now instantiates `EnsembleHMM` by default (`use_ensemble=True`) instead of single `HMMRegimeModel`
   - `config.py` + all sources: Removed `vol_price_diverge` from "extended" (6→5 features) and "full" (8→7 features) — binary feature causes degenerate HMM covariance
   - CITRINE `live_trading_citrine.py`: Added `_restore_state_from_db()` — state now persists across restarts; added deduplication guard in `_enter_position()`
   - CITRINE `live_trading_citrine.py`: `_log_snapshot()` now uses scan prices for mark-to-market (was using stale entry prices)
9. **Sprint 2 — Testing Infrastructure** (2026-03-17):
   - `tests/test_strategy_consistency.py`: 69 tests covering feature validation, ensemble voting, HMM model, signal series, kill-switch, config integrity, indicators, CITRINE config, position persistence — all passing
   - `reconcile.py`: Reconciliation script comparing live trading signals vs backtester output for all 3 projects
   - BERYL `live_trading_beryl.py`: Rewritten for multi-ticker rotation — `_scan_all_tickers()`, `_pick_best_buy()`, `process_signals()`; status JSON includes `scan_summary`
10. **Sprint 3 — Model Fine-Tuning & Advanced Validation** (2026-03-17):
    - **3.1 New continuous features**: Added `realized_kurtosis`, `volume_return_intensity`, `return_momentum_ratio` to `src/indicators.py`. New feature sets `extended_v2` (6 features) and `full_v2` (10 features) in `config.py`. A/B test: `extended_v2` beats `extended` by +2.747 Sharpe on SOL/4h/ensemble. `full_v2` overfits. Updated `_attach_hmm_features()` in `walk_forward.py` and `walk_forward_ndx.py` to config-driven feature computation (no hardcoded feature set names). Tests: 69/69 pass (6 new tests for v2 features).
    - **3.2 HMM covariance regularization**: `HMMRegimeModel.fit()` now uses `n_init=3` (3 random starts, picks best log-likelihood), `_regularize_covariance()` (adds ridge 1e-4 when any eigenvalue < 1e-6), and `_check_min_obs_per_state()` (flags non-converged if any state has < 5 observations). 3x training cost but significantly more robust convergence.
    - **3.3 CITRINE sector concentration limits**: Added `CITRINE_MAX_PER_SECTOR = 4` to `config.py`. New `_apply_sector_cap()` step in `CitrineAllocator` pipeline (step 5b, between position-cap and cash determination). Prevents semiconductor-heavy portfolios.
    - **3.4 Walk-forward stability tests**: New `stability_test.py` — 3 test types: parameter sensitivity (+/-10%), data window sensitivity (train=[5,6,7]m, test=[2,3,4]m), seed sensitivity (5 seeds). Flags configs as "fragile" if Sharpe drops >50% (sensitivity), CV >50% (window), or CV >30% (seed). Replaces old `robustness.py` concept.
    - **3.5 CITRINE intra-day kill-switch**: Added `_sleep_with_risk_checks()` to `live_trading_citrine.py` — wakes every 4 hours during daily sleep for unrealized P&L checks. Added `_check_intraday_risk()` (warnings: position >5% loss, portfolio >3% loss; kill-switch: total unrealized >5% of capital) and `_emergency_exit_all()` for forced liquidation.
    - **3.6 Dashboard enhancements**: Added `_position_health_table()` (ticker, direction, entry/current price, unrealized P&L, sector) and `_signal_frequency_card()` (7-day entry/exit counts vs expected, flags >20% deviation) to `citrine_dashboard.py`.

### Current Phase: All Four Sub-Projects Running 24/7 on Oracle Cloud (2026-03-22)
- **8 systemd services + watchdog timer deployed to Oracle Cloud VM** (VM.Standard.E2.1.Micro, 1 OCPU, 1GB RAM) with auto-restart. Watchdog monitors all 8 services (4 traders + 4 dashboards) every 15 min.
- **AGATE** running `--test` with ensemble HMM on 14 crypto tickers, scanning every 4h. 4 open positions (2026-04-07): DOGE, ETH, LINK, XRP. 6 closed trades (1 win), realized P&L: -$378.11. **2026-04-07**: Zero BULL regimes (0 BULL / 3 BEAR / 3 CHOP). Only 6/14 tickers in scan — possible data fetch failures. Config: `extended_v2`/7cf.
- **BERYL** running 98-ticker ensemble rotation with up to 3 positions (58 with BERYL-optimized configs, 40 use defaults). **4/4 wins (100%)**: ARM +$23.11, ARM +$18.50, MCHP +$10.41, ROP +$7.76. Realized P&L: +$59.78. 3 open positions (2026-04-07): MELI, ADI, CSX (new — insider 1.15x, score 2.056). **2026-04-07**: Scan 93/98, 27 BULL / 43 BEAR / 23 CHOP — BEAR tilt continues.
- **CITRINE** running 82-ticker long-only live test. **Sprint 12 (2026-04-04, DEPLOYED)**: Chandelier day-0 immunity + execution-observation fix + dashboard P&L decomposition. **2026-04-07**: 4 regime_flip exits + 6 entries morning, then 7 intraday -2% hard stops in afternoon selloff (-$135 combined). Current: $24,377 equity, 7 positions, 60% cash, 313 trades, 98 closed, 28.6% WR. Realized P&L: -$643. Shadow tracker: 78 trades logged.
- **DIAMOND** running WebSocket anomaly detection on all Kalshi markets with paper trading engine + ML scorer (Lasso, AUC 0.823, A/B logging mode). Dashboard at :8080.
- **All 4 dashboards accessible**: :8060 (AGATE+BERYL, with regime panels + realized/unrealized P&L), :8070 (CITRINE, with position health + signal frequency + realized/unrealized split), :8080 (DIAMOND, anomaly feed + portfolio status), :8090 (Consolidated, all projects + realized/unrealized split)
- **Watchdog timer** runs every 15 minutes — checks journald log freshness + memory footprint, auto-restarts zombie services. Deployed 2026-03-26 after all 3 traders went zombie simultaneously.
- Kill-switch automation active across all four projects (with grace period for CITRINE and BERYL)
- All optimizers COMPLETE: BERYL P3 (242), AGATE WF ensemble (200), CITRINE re-opt (585)
- **Auto-optimization uses launchd** (migrated from crontab 2026-03-22 — crontab missed runs when Mac sleeps). `~/Library/LaunchAgents/com.hmm-trader.optimize-{citrine,beryl,agate}.plist`. Fires on wake if missed. Sundays: CITRINE 2am, BERYL 4am, AGATE 8am.
- **Notifier** auto-detects OS — skips `osascript` on Linux VM, uses Pushover + terminal bell only
- **No local Mac processes** — all traders/dashboards migrated to VM; stale Mac processes cleaned up 2026-03-21
- **Sprint 4 fixes (2026-03-21)**: (1) CITRINE kill-switch grace period, (2) notifier OS auto-detection, (3) live-dashboard systemd service, (4) dashboard cache-busting headers, (5) citrine_dashboard position health fix, (6) cash % shows actual ratio, (7) logrotate for all service logs, (8) CITRINE DB reset on VM
- **Sprint 5 — BERYL Model Excellence (2026-03-22)**: (1) 98-ticker expansion from citrine_per_ticker_configs.json, (2) ensemble HMM with majority voting, (3) fallback retry (ensemble fail → single 4-state HMM), (4) 365-day lookback (100% convergence), (5) multi-position mode (up to 3, $833 each), (6) lower default confirmations (7→5), (7) smart insider scoring (10b5-1 filtering, extreme-only penalty), (8) dedicated daily optimizer (`optimize_beryl_daily.py`), (9) launchd migration (crontab→plist for sleep resilience)
- **KEY MILESTONE (2026-03-22)**: BERYL convergence went from 71% (70/98) to **100% (98/98)** after Sprint 5 rewrite. The three changes that achieved this: (1) 365-day lookback (was 180 — doubled training observations from ~90 to ~252 trading days), (2) Ensemble HMM with fallback retry (3 models voting + single 4-state fallback if ensemble fails), (3) forced diag covariance on VM. This 100% convergence rate validates the entire Sprint 5 approach.

### BERYL Optimization & Live Test
- **Phase 1**: COMPLETE (195/200 trials, best TSLA Sharpe 0.921)
  - Optimizer: `optimize_beryl.py` with 5,400-combination grid
  - Walk-forward: `walk_forward_ndx.py` using Polygon equities API
- **Phase 2**: COMPLETE (TSLA+NVDA, intraday timeframes — best NVDA Sharpe 1.094: extended/4st/7cf/1.5x/full)
  - Optimizer: `optimize_beryl_p2.py`
- **Phase 3**: COMPLETE (242 trials, 5 tickers: TSLA/NVDA/AAPL/MSFT/GOOGL + extended_v2)
  - Best: NVDA Sharpe +0.825 (6st/extended/7cf/1.5x/72h/diag/1d), TSLA +0.807, AAPL +0.733, NVDA/extended_v2 +0.762
  - Key finding: 1d timeframe dominates equities (48% positive vs 11-25% for 2h/4h)
  - Optimizer: `optimize_beryl_p3.py` — expanded grid, 5 n_states, 3 timeframes
- **Sprint 5 — BERYL Model Excellence** (2026-03-22, 7 upgrades):
  1. **98-ticker expansion**: Loads all 98 NDX100 tickers (was 5 hardcoded). **Note**: Sprint 8 (2026-03-28) separated ticker universe (from citrine config) from HMM params (from `beryl_per_ticker_configs.json`) — see `_load_ticker_universe()`
  2. **Ensemble HMM**: `EnsembleHMM(n_states_list=[N-1, N, N+1])` with majority voting (was single-model)
  3. **Fallback retry**: If ensemble doesn't converge, retries with single `HMMRegimeModel(n_states=4, cov_type="diag")`
  4. **365-day lookback**: `LOOKBACK_DAYS = 365` (was 180), ~252 trading days → **100% convergence** (was 71%)
  5. **Multi-position**: Holds up to 3 positions ($833 each), per-ticker cooldowns (was single position)
  6. **Lower confirmation default**: `BERYL_DEFAULT_CONFIG["confirmations"] = 5` (was 7), `--min-confirmations` CLI override
  7. **Smart insider scoring**: 10b5-1 pre-planned sales ignored, only extreme discretionary selling penalized (0.85x vs old 0.70x)
  - First scan result: 98/98 converged, 4 BUY signals, entered ROP (insider 1.15x), ARM, MPWR
- **Dedicated daily optimizer**: `optimize_beryl_daily.py` — 98 tickers × [3,4,5,6] n_states × 3 feature sets × [4,5,6,7] confirms = 4,704 combos, ensemble mode, daily bars only
- **Multi-ticker live test**: RUNNING (`live_trading_beryl.py --test` with 98 tickers, ensemble HMM, 3 positions)
  - Daily signal loop (Mon-Fri), scans all 98 tickers, buys top 3 BUY signals by score
  - Score = confidence × (confirmations / threshold) × alt_data_boost
  - Position sizing: MAX_NOTIONAL = $2,500 total ($833 per position), trades logged to `beryl_trades.db`
  - Intra-day risk checks every 4h on all positions
  - **First profitable trade (2026-03-23)**: SELL ARM +$23.11 (+2.77%), BUY INSM @ $144.11
- **Sprint 6 — AGATE Multi-Ticker & Folder Restructure** (2026-03-22/23, 8 changes):
  1. **14-ticker expansion**: `AGATE_TICKERS` in config.py — BTC, ETH, SOL, XRP, LTC, SUI, DOGE, ADA, BCH, LINK, HBAR, XLM, AVAX, ENA (dropped SHIB/DOT — 0% positive in optimizer)
  2. **Multi-ticker rotation**: `live_trading.py` rewritten with BERYL-style `_scan_all_tickers()` + `_pick_best_buy()` — scans all tickers every 4h, buys top BUY signal
  3. **Adaptive confirmations**: At conf > 0.90, confirmation threshold drops by 1 (e.g., 7→6). Increases signal frequency without lowering quality.
  4. **Per-ticker configs**: `agate_per_ticker_configs.json` — best n_states, feature_set, confirmations, cov_type, timeframe per crypto ticker (from 179-trial optimizer). NOTE: only 12/14 tickers optimized — LTC, SUI, DOGE, ADA had insufficient data (added 2025-12-22 with ~95d cache; WF needs 270d+). Backfilling to 730d in Sprint 8 (2026-03-28).
  5. **Near-miss alerts**: Pushover notification when tickers are 1 indicator away from BUY
  6. **Momentum tracking**: Logs confirmation count changes between scans (gaining/losing)
  7. **CSV schema validation**: `_validate_result()` prevents corrupt optimizer CSV lines
  8. **Folder restructure**: `~/Documents/trdng/` → `~/Documents/quant/` with gemstone project folders, backward-compat symlinks
  - Optimizer results (179 trials): BCH Sharpe +3.52 (5/5 windows ★), ETH +3.59, XLM +2.15, SOL +1.23
  - SHIB and DOT dropped (0% positive across all trials)
- **Sprint 7 — PhD Quant Review Fixes** (2026-03-26, 8 changes across all projects):
  1. **Cash band enforcement** (CITRINE): `_execute_rebalance()` now enforces `min_cash = equity * cash_pct` as hard floor. Entries skipped if they'd breach. Was 0.1% cash, now 38%.
  2. **Sojourn decay** (CITRINE): Replaced persistence_bonus (1.0→1.5x as regime ages) with sojourn_factor (1.0→0.5x). Added `get_regime_halflife()` to `HMMRegimeModel` and `EnsembleHMM` using transition matrix `A[k,k]`. Half-life: `-ln(2)/ln(A[k,k])`. New `regime_half_life` field in `TickerScan`.
  3. **Continuous cash scaling** (CITRINE): Replaced rigid bucketed bands with `cash_pct = max(0.10, 1.0 - bull_ratio^0.6)`. Eliminates cliff-edge transitions.
  4. **Intraday price fix** (BERYL+CITRINE): Added `fetch_latest_price()` using Polygon Snapshot API. Fixed silent TypeError where `fetch_equity_daily(days=5)` failed because `days` kwarg doesn't exist.
  5. **Paper P&L tracking** (AGATE): Added `open_positions` table to `LiveMonitor` with `log_entry()`, `clear_entry()`, `get_open_positions()`. Deduplication guard prevents phantom entries.
  6. **Signal decay analysis** (AGATE): New `signal_decay_analysis.py` — measures forward returns at t+1/t+2/t+4 bars after BUY signals.
  7. **Semi-Markov research**: New `research/semi_markov_analysis.py` — KS test comparing geometric vs empirical sojourn distributions.
  8. **DIAMOND→CITRINE mapping**: New `research/kalshi_equity_mapping.json` — maps Kalshi prediction markets to NDX100 tickers.
  - **Zombie process incident (2026-03-25/26)**: All 3 traders went zombie after file deploy (stale `__pycache__`). CITRINE missed full trading day. Fixed with `ExecStartPre` pycache clear + watchdog timer (15min).
  - **DIAMOND ML scorer fixes**: OOS-only comparison (was in-sample), Platt scaling removed at N<500, realized edge tracking, Wilson CI on win rate.
- **Sprint 8 — ATR Features, Optimizer Penalty & Dashboard Redesign** (2026-03-30, 6 changes):
  1. **ATR-normalized features**: 4 new functions in `src/indicators.py` — `compute_atr()`, `compute_atr_normalized_return()`, `compute_atr_normalized_range()`, `compute_atr_normalized_volume()`. New `"atr_normalized"` feature set (6 features) in `config.py`. Divides HMM inputs by ATR percentage so the model sees vol-adjusted moves — enables a single universal config across assets with wildly different volatility profiles (BTC $2000 ATR vs HBAR $0.001 ATR).
  2. **Trade frequency penalty**: `optimize_agate_multi.py` rank_score changed from `Sharpe × consistency` to `Sharpe × consistency × log(trades)`. Penalizes low-N configs that inflate Sharpe by chance (Lo 2002 intuition: `SE(Sharpe) ≈ 1/√N`). Applied in both per-trial scoring and `_save_per_ticker_configs` config selection.
  3. **Consolidated dashboard redesign**: Complete rewrite of `consolidated_dashboard.py` (~1000 lines). Removed `dash_bootstrap_components` dependency. Swiss Design palette (obsidian #08090c, neon-cyan #00e8ff), JetBrains Mono font, glassmorphism cards (`backdrop-filter: blur(12px)`), bento-box 12-column CSS Grid layout, staggered entrance animations, spring-physics hover effects, SVG fractalNoise texture overlay. Custom `app.index_string` HTML template.
  4. **Dashboard CSS bug fix**: Dash 4.0 dynamically injects elements via React callbacks — CSS `animation-fill-mode: both` kept elements at `opacity: 0` (the `from` keyframe) because animations don't re-trigger on DOM insertion. Fixed by changing to `forwards`. Also fixed `body::before` z-index overlay blocking clicks, and moved Google Fonts from `@import` to `<link>` tag.
  5. **Quant audit results** (carried from previous session): Universal param test FAILED (6/15 tickers positive with global config), beta test FAILED (3/13 tickers had alpha after market-beta adjustment), frequency test PASSED (8/10 adequate-N configs had >1.0 Sharpe). ETH and LINK are the only two tickers surviving ALL statistical tests (DSR, frequency, alpha). Per-ticker configs confirmed as necessary — universal config doesn't work across crypto assets with different microstructures.
  6. **DIAMOND↔CITRINE bridge**: `src/diamond_bridge.py` rewritten with `CitrineDiamondBridge` class. Maps Kalshi anomalies to NDX100 equity alt-data boosts via indirect correlations (BTC→MSTR/COIN, WTI→energy, macro→all tickers). CITRINE `_fetch_alt_data_boosts()` now combines SEC insider × DIAMOND anomaly boosts multiplicatively, clamped [0.7, 1.5].

- **Sprint 9 — CITRINE Risk Engine & MAE Calibration** (2026-03-30/31, DEPLOYED):
  1. **MAE/MFE calibration** (`mae_calibration.py`): Offline walk-forward analysis of 562 trades across 30 NDX100 tickers over 3 years. HMM-only exits (no trailing stops). Results: 95th percentile non-zero winner MAE = **1.793 ATR**, 99th = 2.235 ATR. Recommended Chandelier multiplier: **2.0 ATR** (95th × 1.1 buffer). Prior expectation of 3.0 was too conservative. At 2.0 ATR: stops 3% of winners, 11% of losers. Average hold duration ~2.9 days (winners), ~2.4 days (losers). 97% of winner MAE occurs in first 2 days. Expectancy: +0.39%/trade.
  2. **CITRINE turnaround diagnosis**: Forensic analysis of 108 live trades revealed three compounding failures: (a) loss asymmetry 3:1 (avg loser ~3× avg winner), (b) HMM exit latency — 87% of exits occur in BEAR/CHOP (regime already flipped), (c) concentration in serial losers. Root cause: HMM is a probabilistic low-pass filter — requires multiple adverse observations to overcome transition probability A[i,i].
  3. **Risk engine DEPLOYED** to `live_trading_citrine.py`: 4-part system separating alpha model (HMM) from risk model: (a) ATR-based position sizing (inverse vol parity, 1% risk budget per trade, 15% max notional cap), (b) Chandelier exit at entry_atr × 2.0 (CRITICAL: must use entry_atr, not current_atr — ATR autocorrelation causes stop to widen during crashes), (c) confidence velocity exit (delta_conf < -0.25 AND close < prev_close), (d) MAE/MFE tracking per trade (8 new DB columns). `_check_risk_exits()` runs before allocator in daily cycle — Chandelier + conf velocity checked on every held position.
  4. **`CitrinePosition` expanded** with risk engine fields: `entry_atr`, `entry_confidence`, `highest_price`/`lowest_price` watermarks, `mae_pct`/`mfe_pct`/`mae_atr`/`mfe_atr` tracking, `chandelier_stop()` method. `TickerScan` now includes `current_atr` field computed during scan.
  5. **DB schema expanded**: 8 new columns on trades table — `entry_atr`, `exit_reason`, `mae_pct`, `mfe_pct`, `mae_atr`, `mfe_atr`, `entry_confidence`, `hold_days`. Exit reasons categorized: `chandelier_stop`, `conf_velocity`, `regime_flip`, `kill_switch`, `manual`.
  6. **Metrics gap addressed**: 17 metrics across trade/position/portfolio levels previously not tracked. Key gaps now filled: MAE/MFE, entry confidence, exit reason categorization. Remaining gaps: rolling Sharpe for BERYL/CITRINE, profit factor, portfolio heat, win/loss streak.
  7. **Live deployment confirmed** (2026-03-31): Deployed to VM, 2 full scan cycles completed. Cycle 1: 4 exits (MDLZ -1.04%, INTC +5.20%, KDP -1.07%, CSX +1.94% — all with MAE/MFE logged), 5 ATR-sized entries (HON $586, GOOGL $506, NVDA $443, ABNB $262, CEG $190). Cycle 2: 5/13 positions have Chandelier stops (new entries only; old positions safely fall back to -2% pct stop). Cash floor enforced correctly (skipped VRSK and MSFT entries that would breach 41% target). ATR sizing produced position sizes 88-96% smaller than old flat $5k cap — volatile stocks like CEG ($15.36 ATR) get appropriately tiny positions ($190).

- **Sprint 10 — BERYL Position Persistence & Dashboard Fixes** (2026-04-01, DEPLOYED):
  1. **Position persistence (CRITICAL)**: BERYL had no state persistence — positions silently lost on every service restart. March 30 positions (ARM, MELI, TMUS — $2,499 notional) abandoned when service restarted (PID 270432→309683). No exits logged, P&L unrecoverable. **Fix**: Added `portfolio_snapshots` DB table, `_log_snapshot()` (writes after every scan cycle), `_restore_state_from_db()` (rebuilds positions on startup). Mirrors CITRINE Sprint 1 pattern.
  2. **Status file timing (CRITICAL)**: `_write_status()` called BEFORE `process_signals()` — dashboard always showed pre-trade state. Fixed by moving after `process_signals()`.
  3. **Dashboard dict normalization**: `_beryl_regime_panel()` in `live_dashboard.py` did `positions[:3]` which fails on dicts. Fixed with `isinstance` check + `list(dict.values())`.
  4. **Current price in status**: Added `current_price` to scan_summary entries for dashboard unrealized P&L.
  5. **Snapshot logging**: `_log_snapshot()` writes position JSON + equity + count after every scan. First verified snapshot: 3 positions (MCHP, ARM, MELI), $833 each.

- **Sprint 11 — CITRINE Shadow Tracker & Low-N Analysis** (2026-04-02, DEPLOYED):
  1. **Shadow/harvesting tracker**: `ShadowTracker` class (~230 lines) in `live_trading_citrine.py`. Parallel allocator with relaxed thresholds (conf≥0.70, persist≥1d, max 30 positions) runs on same scan data as live engine. Logs to `shadow_trades` (24 columns) and `shadow_snapshots` tables. Uses config monkey-patching (temporarily overrides config globals, safe single-threaded). Independent `CitrineAllocator` instance with separate state. Shadow positions persist across restarts.
  2. **TickerScan per-indicator breakdown**: Added `indicator_details: dict` to `TickerScan` in `src/citrine_scanner.py`. Extracts 8 boolean check columns (rsi, momentum, volatility, volume, adx, price_trend, macd, stochastic) from `build_signal_series()`. Enables analysis of which indicators block profitable entries.
  3. **Shadow schema expansion**: 5 new columns — `confirmations_short`, `target_weight`, `scaled_weight`, `live_would_enter` (flag: conf≥0.90 AND persist≥3 AND BULL), `indicator_json` (serialized per-indicator booleans). Safe ALTER TABLE migration.
  4. **Grinold's Law motivation**: IR ≈ IC × √Breadth — low trade frequency caps information ratio. Shadow at 0.70/1d captures ~27 entries vs ~7 live = 3.8x breadth multiplier. Data will determine whether relaxation maintains or dilutes IC.
  5. **First results**: 50 shadow trades logged, 22 held vs 9 live (~2.4x breadth). 23 with full indicator JSON.
  - **Chandelier calibration concern** (2026-04-02): 6 same-day Chandelier stops — **RESOLVED in Sprint 12** (see below).

- **Sprint 12 — Chandelier Execution Fix + Dashboard P&L Decomposition** (2026-04-04, DEPLOYED):
  1. **Chandelier day-0 immunity (P0)**: `_check_risk_exits()` skips Chandelier when `entry_date == today`. Root cause of Apr 2 stop-outs: execution-observation misalignment — intraday spot prices evaluated against daily-ATR-derived stops before the stop had time to establish a meaningful trailing level. The -2% hard stop in `_check_intraday_risk()` still protects day-0 entries. Validated: zero false stops Apr 3-4, 10 positions granted immunity correctly.
  2. **Chandelier removed from intraday risk checks (P0)**: `_check_intraday_risk()` no longer evaluates Chandelier stops. Chandelier is a daily-frequency tool — evaluating it against intraday noise caused 6 premature exits. Intraday checks now use only the hard -2% pct stop as catastrophic safety net.
  3. **entry_time restoration from DB**: `_restore_allocator_holdings()` now writes `self.positions[ticker].entry_time = ts`. Previously, restarted positions got `entry_time = datetime.now()`, incorrectly granting day-0 immunity to all positions post-restart.
  4. **STOP_EXIT trades missing from CITRINE dashboard**: `_load_trades()` had `WHERE action = 'EXIT'`, missing 9 stop-loss exits worth -$173. Fixed to `WHERE action IN ('EXIT', 'STOP_EXIT')`.
  5. **Realized/unrealized P&L split**: CITRINE dashboard (:8070) top row now shows Realized P&L (closed trades) and Unrealized P&L (open positions) separately. Consolidated dashboard (:8090) same. Service re-enabled.
  6. **Blank :8060 dashboard**: AGATE+BERYL dashboard rendered blank (HTTP 200 but no content). `signal_strength` column in `paper_trades.db` contained binary blobs — Dash JSON serializer crash. Fixed: coerce to string in `_trades_table()`.

- **Sprint 15 — BERYL Cross-Sectional Ranker Kill & Time-Series Pivot** (2026-04-17/18, COMPLETE):
  1. **IC measurement**: T+3 IC = -0.069 (p=0.031) — statistically significant negative. Built bimodal scoring function, market-neutralization, feature importance gate with 3 audit trap patches (collinearity protection, structural drift diagnostic, long-only illusion check).
  2. **6-month backfill** (`backfill_scan_journal.py`): Option C quarterly expanding-window + Ensemble HMM. 13,230 rows, 98 tickers, 135 trading days (Oct 2025 → Apr 2026), 4.5% fallback. Data retained for time-series analysis.
  3. **Gate 2 PASSED**: All 4 features survived (confidence 0.441, velocity 0.325, persistence 0.151, confirmations 0.083). Velocity ρ=+0.49 with confidence — NOT collinear, captures independent momentum info. Permutation importance confirmed all 4 features >6x their noise floor.
  4. **Gate 3 FAILED**: Bimodal IC = -0.010 (t=-1.04), Top-Bottom neutralized spread = -0.04% (flat, 3 inversions). Structural drift trap fired: ≥95% confidence zone = -0.05% neutralized return (N=1,819). Phase 1 (0.60-0.70) had +4bps gross alpha but execution friction >5bps = negative EV.
  5. **KILL ACCEPTED**: HMM has zero cross-sectional predictive power. Cannot rank assets against each other. `cross_sectional_ranker.py` and `feature_importance.py` DELETED from Mac and VM.
  6. **Pivot to time-series eviction**: BERYL is a pure absolute-momentum regime filter. New direction: entry at 0.60-0.80 confidence (capturing velocity of transition), exit on self-degradation below 0.60 or 1.5x ATR catastrophe stop. No relative comparison — each asset evaluated independently.
  - **Key insight**: HMM IS a time-series regime filter (11/11 wins from detecting WHEN). ISN'T a cross-sectional ranker (cannot tell WHICH asset is better).
  - Remaining file: `backfill_scan_journal.py` (data useful for time-series analysis)

### CITRINE Optimization & Portfolio Rotation
- **Per-ticker HMM optimization**: COMPLETE + RE-OPTIMIZED (585 trials across 100 tickers, 82 positive Sharpe — 83%)
  - Top tickers: FER (1.301), MELI (1.170), STX (1.169), CEG (1.122), WBD (0.944), AXON (0.836), TSLA (0.828)
  - Re-optimizer improved 47 tickers, added 14 new positive-Sharpe tickers (68→82)
  - Configs saved to `citrine_per_ticker_configs.json` (auto-loaded by scanner)
- **Allocator optimization**: COMPLETE (3,456/3,456 trials, ~38 min)
  - Winner: entry=0.90, exit=0.50, persist=3, aggressive cash, fast scale → **Sharpe +1.665**
  - Previous default: entry=0.80, exit=0.65, slow scale → Sharpe +1.341
  - Key insight: wider hysteresis band (0.40 gap) is the single biggest improvement
- **Walk-forward backtest results** (tested 6 configs):
  | Config | Return | Sharpe | Notes |
  |--------|--------|--------|-------|
  | Baseline (10 tickers, default) | -9.09% | -1.091 | 2/8 positive windows |
  | Per-ticker optimized (10 tickers) | -6.77% | -0.965 | Improved but underwater |
  | Top-10 Sharpe tickers | -3.75% | -0.931 | Better ticker selection |
  | 39 tickers, long+short | -6.27% | -1.386 | SHORT hurts |
  | 10 tickers, long-only | -0.03% | -0.536 | Nearly flat |
  | **39 tickers, long-only** | **+6.78%** | **+1.341** | **Winner — 2/3 windows positive** |
- **Live test mode**: RUNNING (started 2026-03-15, restarted 2026-03-17 with 82 tickers)
  - Command: `python live_trading_citrine.py --test --long-only --tickers FER,MELI,STX,CEG,WBD,AXON,TSLA,WDC,MU,GEHC,REGN,ZS,PCAR,GOOGL,PANW,AMAT,TXN,TMUS,NVDA,MAR,LIN,MDLZ,WMT,CSCO,FAST,NXPI,CTAS,MCHP,NFLX,INTU,ROP,ADSK,CRWD,SNPS,EA,CTSH,ROST,MPWR,CDNS,MSTR,CPRT,LRCX,AMZN,BKNG,KLAC,COST,XEL,PYPL,KDP,MNST,CSGP,INTC,AAPL,AEP,PLTR,ODFL,SHOP,AMD,EXC,ADI,TEAM,VRSK,CCEP,DASH,ASML,INSM,FANG,BKR,QCOM,APP,CSX,AVGO,PDD,TRI,KHC,AMGN,FTNT,ABNB,ADBE,HON,MSFT,WDAY`
  - 82 positive-Sharpe tickers (expanded from 74), long-only mode, $25k capital
  - Optimized allocator params: entry=0.90, exit=0.50, fast scale
  - Kill-switch rules: loss > 5%, rolling 20-trade Sharpe < 0.3, 0/10 wins
  - Trades logged to `citrine_trades.db` with portfolio snapshots
- **Portfolio dashboard**: LIVE (`citrine_dashboard.py` at :8070)
  - Equity curve, allocation pie, regime distribution, P&L histogram, kill-switch status
- **All bugs fixed (2026-03-15 to 2026-03-17)**:
  1. `optimize_citrine.py`: `r.sharpe` → `r.sharpe_ratio`, `r.converged` → `r.hmm_converged`
  2. `citrine_backtest.py`: `_fit_models()` and `_prepare_test_data()` ignoring per-ticker configs — fixed to load `citrine_per_ticker_configs.json`
  3. `walk_forward_ndx.py`: `_attach_hmm_features()` missing `dropna()` after extended features — rolling windows produce NaN, caused 20/39 tickers to fail with "Input contains NaN"
  4. `live_trading_citrine.py`: State persistence added (`_restore_state_from_db()`), mark-to-market fixed (now uses scan prices), deduplication guard added

### Key Strategic Decisions

**Single-Model XRP → Ensemble SOL**:
- Single-model XRP (backtest Sharpe 3.512) failed catastrophically in walk-forward validation (Sharpe -1.893)
- Ensemble approach (3 HMMs voting) dramatically improved robustness: SOL ensemble WF Sharpe 6.240
- Ensemble is now the production model; single-model code remains for research

**Long-Only is Production Mode**:
After 2 rounds of multi-direction optimization (193 total trials), the conclusion is clear:
- Long-only significantly outperforms multi-dir in both single-model and ensemble modes
- SHORT signals in crypto face structural headwinds (upward bias, higher volatility)
- Multi-dir code remains in the codebase for research but long-only is the production strategy
- `--regime-mapper` flag enables multi-dir for experimentation; default mode is long-only

**Paper Trading → Micro-Capital Live**:
- Pivoted from 4-week paper trading to direct micro-capital live validation
- Rationale: ensemble WF Sharpe strong enough; real slippage/fills more informative than simulated
- Risk bounded by MAX_NOTIONAL = $2,500 (0.25x leverage on $10k account)

## Architectural Decisions

### Why walk-forward over simple IS/OOS split?
Simple 70/30 splits overfit heavily — 74/106 trials were flagged as overfit in Pass 1. Walk-forward (6m train / 3m test, rolling) forces fresh HMM fits per window, preventing look-ahead bias. The walk-forward Sharpe of 3.512 is more trustworthy than the simple split's 2.745.

### Why asymmetric LONG/SHORT parameters?
SHORT and LONG trades have fundamentally different risk profiles in crypto. Shared thresholds penalise both sides. Adding `MIN_CONFIRMATIONS_SHORT`, `COOLDOWN_HOURS_SHORT`, and `ADX_MIN_SHORT` doubled the number of all-windows-positive multi-dir configs (2→4) and improved SHORT win rate from 41% to 54%. The key finding: `confirmations_short=8` (requiring ALL 8 indicators) is critical for SHORT quality.

### Why config monkey-patching instead of parameter objects?
The optimizer (`optimize.py`, `optimize_wf.py`) mutates `config` module globals directly via `_patch_config()` / `_restore_config()`. This is simple and works because the optimizer is single-threaded. If parallelization is ever needed, replace with a per-trial config dataclass.

### Why no `ta` library?
All 8 indicators and 5 extended HMM features are computed manually in `src/indicators.py`. This avoids the `ta` library's dependency issues and gives full control over edge cases (NaN handling, warmup periods).

### Why Plotly Scatter instead of add_vrect?
Plotly 6.x broke `fig.add_vrect()` with thousands of shape objects (performance collapse). Regime bands use `go.Scatter` with `fill="toself"` on an overlay `yaxis2` — see `build_price_chart()` in `src/dashboard.py`.

## Polygon.io Ticker Format

Crypto tickers use Polygon's `X:` prefix format:

| Asset | Ticker |
|---|---|
| Bitcoin | `X:BTCUSD` |
| Ethereum | `X:ETHUSD` |
| XRP | `X:XRPUSD` |
| Solana | `X:SOLUSD` |

Set via `TICKER` in `config.py` or passed as an argument to `fetch_btc_hourly()`.

API key must be set in `.env` at the project root:
```
POLYGON_API_KEY=your_actual_key_here
```

Cache files are stored in `data_cache/` as `btcusd_hourly.csv`, etc.

## Source File Reference

### `config.py`
All tunable constants. Single source of truth for parameters. Monkey-patched per-trial during optimization. Edit this file to change strategy parameters.

### `src/data_fetcher.py`
- `fetch_btc_hourly(days, ticker)` — downloads hourly OHLCV from Polygon.io with cache-and-resume logic (saves `data_cache/{ticker}_hourly.csv`)
- `build_hmm_features(df)` — computes base 3 features (log_return, price_range, volume_change), winsorised 1–99%
- `load_data()` — convenience: fetch + build features in one call
- `resample_ohlcv(df, timeframe)` — resamples 1h base data to 4h, 1d, etc.
- `load_data_for_optimizer(timeframe, training_days, feature_set)` — full data prep for optimizer trials

### `src/hmm_model.py`
- `HMMRegimeModel` — wraps hmmlearn's `GaussianHMM`
- `fit(df)` — fits on `feature_cols` columns; sets `self.converged`. **Sprint 3 enhancements**: `n_init=3` (tries 3 random starts, picks best log-likelihood — 3x training cost but more robust); `_regularize_covariance()` adds ridge (1e-4) when any eigenvalue < 1e-6; `_check_min_obs_per_state()` flags non-converged if any state has < 5 observations
- `predict(df)` — Viterbi decoding + posterior probabilities; adds columns: `regime`, `regime_cat` (BULL/BEAR/CHOP), `confidence`
- Auto-labels states by mean return rank: top 2 → BULL, bottom 2 → BEAR, middle → CHOP
- Supports `n_states=4–8`; extra states beyond 7 get `chop_extra_N` labels
- Cached to `model_cache.pkl`; reload with `--retrain`

### `src/indicators.py`
Computes all 8 strategy indicators, 8 extended HMM features, and 4 ATR-normalized features — no `ta` library dependency.

**Strategy indicators** (attached by `attach_all(df)`):
`rsi`, `momentum`, `volatility`, `vol_median`, `volume_ratio`, `adx`, `price_trend_pct`, `sma_50`, `macd`, `macd_signal`, `macd_hist`, `stoch_k`, `stoch_d`

**Extended HMM features** (computed by `attach_all`):
`realized_vol_ratio`, `return_autocorr`, `candle_body_ratio`, `bb_width`, `realized_kurtosis`, `volume_return_intensity`, `return_momentum_ratio`
Note: `vol_price_diverge` is still computed by `attach_all` for reference but must NOT be included in any HMM feature set — it is binary (0/1) and causes degenerate covariance matrices.

**ATR-normalized features** (Sprint 8, computed by `attach_all`):
`atr`, `atr_norm_return`, `atr_norm_range`, `atr_norm_volume`
- `compute_atr(df, period=14)` — Average True Range via EWM (standard volatility measure)
- `compute_atr_normalized_return(df)` — `log_return / (ATR / price)` — how many ATR units the bar moved; dimensionless across assets
- `compute_atr_normalized_range(df)` — `(high-low)/close / (ATR/close)` — expansion vs contraction bars relative to ATR
- `compute_atr_normalized_volume(df)` — `volume_ratio × atr_pct` — volume spike in context of volatility regime
These features enable a universal HMM config across assets with wildly different volatility profiles (BTC $2000 ATR vs HBAR $0.001 ATR). All three are dimensionless in "ATR units" — a 2-ATR move in BTC is comparable to a 2-ATR move in HBAR.

**Sprint 3 new features** (continuous, HMM-safe):
- `compute_realized_kurtosis()` — rolling kurtosis of log returns; captures tail-risk regime differences
- `compute_volume_return_intensity()` — volume-weighted absolute return; captures conviction behind moves
- `compute_return_momentum_ratio()` — short-term vs long-term return momentum ratio; captures trend acceleration

### `src/types.py`
- `StrategyDirection` enum: `LONG`, `SHORT`, `FLAT`, `LONG_OR_FLAT`, `SHORT_OR_FLAT`
- Helper properties: `allows_long`, `allows_short`, `is_flat`

### `src/regime_mapper.py`
- `RegimeMapper.get_direction(regime_cat, confidence) → StrategyDirection`
- Maps (regime_cat, confidence) → allowed trading direction using `config.REGIME_DIRECTION_MAP`
- Confidence split: `>= CONFIDENCE_HIGH_THRESHOLD` → "high", else → "med"
- Fallback: FLAT for any unmatched combo

### `src/strategy.py`
- `SignalEngine(use_regime_mapper=False)` — bar-by-bar signal evaluator
  - Legacy mode (`use_regime_mapper=False`): BULL→BUY, BEAR→SELL (identical to Phase 1)
  - Multi-direction mode (`use_regime_mapper=True`): uses RegimeMapper for direction-aware entry/exit
- `SignalResult` — actions: `"BUY"`, `"SELL"`, `"SHORT"`, `"COVER"`, `"HOLD"`, `"COOLDOWN"`; `direction` field: `"LONG"` | `"SHORT"` | `None`
- `build_signal_series(df, use_regime_mapper=False)` — vectorised signal masks
  - Legacy: `check_*`, `confirmation_count`, `raw_long_signal`
  - Multi-dir: `check_*_long`, `check_*_short`, `long_confirmation_count`, `short_confirmation_count`, `allowed_direction`, `raw_long_signal`, `raw_short_signal`

**8 Indicator checks (LONG / SHORT):**

| # | Indicator | LONG confirm | SHORT confirm |
|---|-----------|-------------|---------------|
| 1 | RSI | 30 < rsi < 70 | 30 < rsi < 70 (same) |
| 2 | Momentum | momentum > 0 | momentum < 0 |
| 3 | Volatility | vol < 2× median | vol < 2× median (same) |
| 4 | Volume | volume > 1.1× MA | volume > 1.1× MA (same) |
| 5 | ADX | adx > ADX_MIN (25) | adx > ADX_MIN_SHORT (25, tunable) |
| 6 | Price Trend | Close > SMA-50 | Close < SMA-50 |
| 7 | MACD | macd > signal | macd < signal |
| 8 | Stochastic | %K < 80 | %K > 20 |

### `src/backtester.py`
- `Backtester(use_regime_mapper=False)` — bar-by-bar event loop
- Supports LONG and SHORT positions (one at a time, no pyramiding)
- `Trade` dataclass includes `direction: str = "LONG"` field
- LONG PnL: `btc_held × (exit − entry) − fees`; SHORT PnL: `btc_held × (entry − exit) − fees`
- Exit triggers: direction flip from RegimeMapper (multi-dir) or BEAR regime flip (legacy)
- `metrics` dict includes: `long_trades`, `short_trades`, `long_win_rate_pct`, `short_win_rate_pct`

### `src/dashboard.py`
- `launch(result, host, port, debug, use_regime_mapper=False)` — creates Dash app with `use_reloader=False`
- Dark terminal UI using Dash + dash-bootstrap-components
- Multi-direction support: separate LONG ▲ (green) and SHORT ▼ (purple) trade markers
- Trade table includes Direction column when available
- Signal panel shows LONG SIGNAL / SHORT SIGNAL / FLAT in multi-direction mode
- Metric cards include long/short trade breakdown
- Price chart uses Plotly 6.x-compatible Scatter traces for regime bands (NOT `add_vrect`)

### `optimize.py`
Random-search hyperparameter optimizer (standalone script).
- Samples from 8,640-combination grid; default 300 trials
- 70/30 IS/OOS train-test split; flags overfit when OOS Sharpe < IS Sharpe − 0.5
- Skips trials with: HMM non-convergence, <10 trades, >5% NaN features
- Checkpoints to `optimization_results.csv` every 10 runs (resumable)
- Generates `optimization_heatmap.html` — interactive Plotly heatmap with 4 dropdowns
- Generates `feature_importance.json` from top-10 OOS Sharpe runs

### `optimize_wf.py`
Walk-forward hyperparameter optimizer — uses rolling 6m-train/3m-test windows instead of simple IS/OOS split.
- Two modes:
  - **Long-only** (default): 8,640-combination grid (ticker × timeframe × n_states × feature_set × confirmations × leverage × cooldown × cov_type)
  - **Multi-direction** (`--regime-mapper`): 6,220,800-combination grid (base × conf_high_threshold × bull_med_action × bear_med_action × confirmations_short × cooldown_hours_short × adx_min_short)
- Uses `run_walk_forward()` from `walk_forward.py` as fitness function
- Checkpoints every 5 trials (resumable with `--resume`)
- Long-only outputs: `optimization_wf_results.csv` / `optimization_wf_heatmap.html`
- Multi-dir outputs: `optimization_wf_multidir_results.csv` / `optimization_wf_multidir_heatmap.html`
- Monkey-patches config per trial including `REGIME_DIRECTION_MAP` rebuild for multi-dir mode
- Multi-dir grid adds: `conf_high_threshold` [0.70–0.90], `bull_med_action` [LONG_OR_FLAT/LONG/FLAT], `bear_med_action` [SHORT_OR_FLAT/SHORT/FLAT], `confirmations_short` [7, 8], `cooldown_hours_short` [48, 72], `adx_min_short` [25, 30]

### `walk_forward.py`
Rolling walk-forward analysis — trains a fresh HMM per window, chains equity across test periods.
- `run_walk_forward(train_months, test_months, ticker, feature_set, confirmations, timeframe, quiet, use_regime_mapper)` — main entry point
- `WindowResult` NamedTuple: per-window metrics (return, sharpe, drawdown, trades, etc.)
- Supports any timeframe via `resample_ohlcv()` — 1h, 2h, 3h, 4h all tested
- Supports multi-direction via `use_regime_mapper=True` (LONG/SHORT/FLAT)
- `_attach_hmm_features()` — Sprint 3 update: now config-driven feature computation (reads `config.FEATURE_SETS[feature_set]` instead of hardcoded feature set names); supports `extended_v2` and `full_v2`
- Generates `walk_forward_results.csv` and `walk_forward_chart.html`

### `src/wf_report.py`
Summary report generator for walk-forward optimizer results.
- Ranks configs by combined OOS Sharpe
- Reports per-window consistency (std of Sharpe across windows)
- Best config per ticker breakdown

### `robustness.py` *(superseded by `stability_test.py` in Sprint 3)*
Originally planned as a 4-test robustness suite (Monte Carlo bootstrap, window variation, parameter sensitivity, ticker transferability). The ensemble approach (`src/ensemble.py`) replaced this for model robustness. Now fully superseded by `stability_test.py` which provides systematic fragility detection. The file may still exist for reference but is no longer the primary validation mechanism.

### `stability_test.py`
Walk-forward stability testing suite — replaces `robustness.py` concept (Sprint 3).
- 3 test types:
  - **Parameter sensitivity**: perturbs each config param by +/-10%, re-runs walk-forward; flags "fragile" if Sharpe drops >50%
  - **Data window sensitivity**: tests train_months=[5,6,7] × test_months=[2,3,4] (9 combos); flags "fragile" if CV of Sharpe across windows >50%
  - **Seed sensitivity**: runs 5 different random seeds; flags "fragile" if CV of Sharpe >30%
- Supports all 3 projects: `--project agate`, `--project beryl`, `--project citrine`
- Quick mode: `--quick` reduces parameter perturbation count and seed count
- Outputs: terminal report with pass/fail per test type and fragility flags
- CLI: `python stability_test.py --project agate`, `python stability_test.py --project agate --quick`

### `live_trading.py`
Main orchestrator for AGATE live trading.
- CLI: `--test` (simulated fills, real data) or `--live` (real money via Coinbase CDP)
- 4h signal loop: fetches live Polygon data, runs ensemble HMM, generates signals, executes trades
- Position sizing: MAX_NOTIONAL = $2,500, effective 0.25x leverage on $10k account
- Integrates `SignalGenerator`, `LiveBroker`, `LiveMonitor`, and `Notifier`
- `_write_status(signal)` — writes `agate_status.json` with regime/confidence/signal/position info for dashboard regime panels

### `src/live_broker.py`
Coinbase CDP integration for order execution.
- `LiveBroker` class wraps Coinbase Advanced Trade API
- `Position` dataclass tracks open positions (direction, size, entry price)
- Test mode: simulates fills with 10bps slippage added to market price
- Live mode: places real market orders via Coinbase CDP (requires JWT signing)

### `src/signal_generator.py`
Real-time signal generation for live trading.
- `SignalGenerator(use_ensemble=True)` — `use_ensemble=True` by default (Sprint 1 fix); instantiates `EnsembleHMM(cov_type, feature_cols)` instead of single `HMMRegimeModel`
- Fetches live OHLCV data from Polygon.io (90-day lookback window)
- Fits fresh ensemble HMM on current data (no stale model cache)
- Runs all 8 indicator checks against current bar
- **Critical fix (Sprint 1)**: Signal determination now correctly uses `raw_long_signal` boolean column + `regime_cat` from `build_signal_series()`. Previously read `latest.get("signal", "HOLD")` which always returned HOLD because `build_signal_series()` never creates a "signal" column.
- Config-driven feature computation (Sprint 3.1) — computes only features needed by active feature set

### `src/live_monitor.py`
P&L tracking and kill-switch automation.
- SQLite-backed trade logging (`paper_trades.db`): timestamp, entry, exit, PnL, signal strength
- Rolling Sharpe calculation (5-trade and 20-trade windows)
- 3 kill-switch rules:
  1. Daily loss > 2% of account → exit all positions
  2. Rolling 5-trade Sharpe < 0.3 → auto-exit, reassess
  3. Win rate 0/10 → stop trading, debug root cause
- Kill-switch triggers `Notifier` modal dialog + emergency push

### `src/notifier.py`
Multi-channel alert system.
- macOS native notifications via `osascript` — trade signals, regime changes, summaries
- Pushover phone push — kill-switch alerts as emergency priority (repeated until acknowledged)
- Terminal bell — fallback for all events
- Kill-switch uses modal dialog (`osascript` display dialog, must click OK) + emergency Pushover push
- Standalone test: `python -m src.notifier` sends test notification on all channels

### `src/ensemble.py`
Ensemble HMM model for production trading.
- 3 `GaussianHMM` models with `n_states=[5, 6, 7]`
- Majority voting on regime classification (BULL/BEAR/CHOP)
- Confidence averaging across all 3 models
- Implements `StrategyProtocol` for polymorphic usage with `SignalEngine` and `Backtester`
- Significantly more robust than single-model: SOL ensemble WF Sharpe 6.240 vs single-model XRP -1.893

### `src/strategy_protocol.py`
Strategy protocol for polymorphic HMM usage.
- Defines `StrategyProtocol` interface that both `HMMRegimeModel` and ensemble implement
- Allows `SignalEngine` and `Backtester` to work with either single or ensemble models without code changes

### `src/alternative_data.py`
SEC Form 4 insider trading data fetcher and scorer for BERYL alt-data boost.
- Fetches recent SEC Form 4 filings via EDGAR for a given ticker
- Parses `aff10b5One` XML field to identify pre-planned 10b5-1 sales (routine compensation disposals)
- Separates discretionary vs routine compensation sales
- Scoring: only flags BEARISH for extreme discretionary selling (>10x buy value, 3+ unique sellers → 0.85x penalty). Routine/10b5-1 sales → 1.0x neutral. Net buying → boost (up to 1.15x)
- Used by `live_trading_beryl.py` to compute `alt_data_boost` multiplier in `_pick_best_buys()` scoring
- Data cached in `alternative_data.db` (SQLite) to avoid redundant EDGAR requests

### `daily_report.py`
Autonomous daily reporting script.
- AGATE status: current position, P&L, Sharpe, kill-switch state
- BERYL status: optimization progress, best Sharpe, trial count
- Maturity scoring (0-100%) for each sub-project
- Visual progress bars in terminal output
- Triggered by `run_daily_report.sh` cron job at 6pm

### `live_dashboard.py`
Live monitoring dashboard (Dash app at port 8060).
- Equity curve with real-time updates
- Kill-switch status indicators (green/red)
- Trade table with all executed trades
- BERYL optimization progress panel
- Regime status panels: BULL/BEAR/CHOP badges for AGATE and BERYL (reads `agate_status.json` and `beryl_status.json`)
- **Sprint 10 fix**: `_beryl_regime_panel()` normalizes dict positions to list (`isinstance` check + `list(dict.values())`) — BERYL stores positions as `{"MCHP": {...}}` dict, not a list like AGATE
- Separate from backtest dashboard (`src/dashboard.py` at port 8050)

### `walk_forward_ndx.py`
NDX100 walk-forward analysis for BERYL sub-project.
- Uses Polygon.io equities API (not crypto `X:` prefix)
- Supports AAPL, TSLA, MSFT, GOOGL, AMZN, NVDA and other NDX100 components
- Same walk-forward methodology as `walk_forward.py` (6m train / 3m test)
- `_attach_hmm_features()` — Sprint 3 update: config-driven feature computation (same change as `walk_forward.py`); supports `extended_v2` and `full_v2`

### `optimize_beryl.py`
BERYL NDX100 optimizer.
- 5,400-combination grid across NDX100 tickers and HMM parameters
- Uses `walk_forward_ndx.py` as fitness function
- Checkpoints to CSV (resumable with `--resume`)
- Best result so far: TSLA Sharpe 0.921

### `beryl_log.md`
BERYL optimization log, updated daily during optimization runs.

### `run_daily_report.sh`
Cron script for automated 6pm daily report.
- Runs `python daily_report.py`
- Sends macOS notification on completion
- Add to crontab: `0 18 * * * /path/to/run_daily_report.sh`

### `optimize_beryl_p2.py`
BERYL Phase 2 optimizer — focused on TSLA + NVDA only with intraday timeframes.
- Grid: ticker(2) × n_states(3) × feature_set(2) × confirmations(2) × leverage(2) × cooldown(2) × cov_type(2) × ensemble(2) × timeframe(3) = 1,152 combinations
- Timeframes: 1d (daily), 2h, 4h — tests whether intraday bars improve Sharpe for volatile equities
- Sends Pushover notification on completion
- Outputs: `beryl_p2_results.csv`, `beryl_p2_heatmap.html`

### `live_trading_beryl.py`
BERYL live trading engine for NDX100 equities — 98-ticker ensemble rotation (Sprint 5 rewrite, Sprint 10 persistence).
- CLI: `--test`, `--tickers NVDA,TSLA,...` (default: all 98 NDX100 tickers), `--min-confirmations N` (override per-ticker thresholds)
- **Ensemble HMM** (Sprint 5): uses `EnsembleHMM(n_states_list=[N-1, N, N+1])` per ticker with fallback to single `HMMRegimeModel(n_states=4)` if ensemble fails
- **Multi-position** (Sprint 5): holds up to `MAX_POSITIONS = 3` ($833 each), per-ticker cooldowns (`self.cooldowns: dict[str, datetime]`)
- **365-day lookback** (Sprint 5): `LOOKBACK_DAYS = 365`, fetches 2 years of data → 100% convergence (was 71% at 180 days)
- **Position persistence** (Sprint 10): `_restore_state_from_db()` rebuilds `self.positions` from `portfolio_snapshots` table on startup. `_log_snapshot()` writes position JSON after every scan cycle. Mirrors CITRINE's Sprint 1 pattern.
- Daily signal loop (Mon-Fri only), scans all 98 tickers with 12s rate-limit between Polygon calls
- `_scan_all_tickers()` — sequential scan with `gc.collect()` between tickers
- `_pick_best_buys(signals, n_slots)` — scores BUY signals by confidence × (confirmations / threshold) × alt_data_boost; returns top N for available position slots, filtering out held tickers and cooldown tickers
- `process_signals()` — exits BEAR positions, enters top BUY signals up to MAX_POSITIONS cap
- `_close_position(ticker, exit_price, confirmations)` — closes specific position by ticker
- `_check_intraday_risk()` — loops through all positions every 4h, emergency exit if single position >10% loss or total >5%
- `_write_status(signals)` — writes `beryl_status.json` with scan_summary (including `current_price`), positions list, convergence stats. **Sprint 10 fix**: called AFTER `process_signals()` (was before — showed pre-trade state)
- `_log_snapshot(signals)` — Sprint 10: persists current positions to `portfolio_snapshots` table after every scan cycle for restart recovery
- `_restore_state_from_db()` — Sprint 10: reads last `portfolio_snapshots` row, rebuilds `self.positions` dict with `BerylPosition` objects
- Position sizing: MAX_NOTIONAL = $2,500 total ($833 per position), 10bps slippage + 0.04% commission
- Logs trades to `beryl_trades.db` (separate from AGATE's `paper_trades.db`). DB includes `portfolio_snapshots` table (Sprint 10).
- Kill-switch rules: total loss > 2%, 0/10 wins, rolling 5-trade Sharpe < 0.3 (with 3-cycle grace period)
- Default fallback config: `n_states=4, feature_set="base", confirmations=5, cov_type="diag"`

### `optimize_beryl_daily.py`
Dedicated BERYL daily-only optimizer — focused grid for ensemble HMM on daily bars (Sprint 5).
- Grid: 98 tickers × n_states[3,4,5,6] × feature_set["base","extended","extended_v2"] × confirmations[4,5,6,7] × cov_type["diag"] = 4,704 combinations
- Uses ensemble HMM mode in walk-forward trials
- Default 200 trials per run, `--resume` support, `--workers N` for multiprocessing
- Auto-generates `beryl_per_ticker_configs.json` with best config per ticker
- Outputs: `beryl_daily_results.csv`, `beryl_daily_heatmap.html`
- Pushover notification on completion
- Replaces `optimize_beryl_p3.py` in Sunday cron (`run_optimize_beryl.sh`)

### `src/citrine_scanner.py`
Daily OHLCV scanner for 100 NDX100 tickers with per-ticker HMM regime detection.
- `TickerScan` dataclass: ticker, regime_cat, confidence, persistence, realized_vol, confirmations, confirmations_short, current_price, sector, hmm_converged, regime_half_life, current_atr, scan_time, error, **indicator_details** (Sprint 11: dict of per-indicator booleans — `{"rsi": true, "momentum": false, ...}` — extracted from `build_signal_series()` check columns)
- `CitrineScanner` class:
  - `scan_all()` — rate-limited scan (12s between Polygon calls) of all 100 NDX tickers
  - `scan_from_data()` — alternative entry point using pre-fetched data
  - `_scan_ticker()` — single-ticker HMM fit + indicators + persistence calculation
  - `_compute_persistence()` — counts consecutive BULL days from end of series
  - `_compute_realized_vol()` — 20-day annualized volatility
- Loads per-ticker optimized HMM configs from `citrine_per_ticker_configs.json` if available (fallback to defaults)
- Reuses: `walk_forward_ndx.fetch_equity_daily()`, `build_hmm_features()`, `HMMRegimeModel`, `attach_all()`, `build_signal_series()`

### `src/citrine_allocator.py`
Portfolio weight allocation engine with hysteresis bands, cooldown modes, sector caps, and gradual scaling.
- `PortfolioWeight` dataclass: ticker, direction, raw_score, target_weight, scaled_weight, notional_usd, days_held, action, sector, confidence, persistence
- `CitrineAllocator` class:
  - Allocation pipeline: regime filter → hysteresis (entry ≥90%, exit ≥50% — updated Sprint 1) → cooldown check → CITRINE score → position cap → **sector cap** (Sprint 3: `_apply_sector_cap()`, max `CITRINE_MAX_PER_SECTOR` positions per sector, keeps highest-scoring) → adaptive cash → normalize → gradual scale
  - Tracks state across calls: `_holdings` (days held), `_exit_day` (cooldown dates), `_day_counter`
  - 3 cooldown modes (configurable): `"none"` (hysteresis only), `"time"` (5-day block after exit), `"threshold"` (85% re-entry)
  - Adaptive cash: 15+ BULL→20% cash, 8-14→30%, 3-7→50%, 0-2→80%
  - Gradual scaling: day 1→25%, day 3→50%, day 5+→100%
- CITRINE score formula: `confidence_weight × inverse_vol × indicator_quality × persistence_bonus`
  - confidence_weight: clip((confidence − 0.65) / 0.25, 0, 1)
  - inverse_vol: clip(median_vol / realized_vol, 0.5, 2.0)
  - indicator_quality: confirmations / 8
  - persistence_bonus: 1.0 + min(max(persistence − 3, 0) × 0.10, 0.50)

### `citrine_backtest.py`
Walk-forward portfolio backtester with daily rebalancing.
- `PortfolioWindowResult` dataclass: per-window metrics (return, benchmark, alpha, sharpe, drawdown, avg_positions, cash%, trades, turnover)
- `CitrineBacktester` class:
  - `run_walk_forward()` — main entry point, returns window results + equity curve
  - `_precache_data()` — batch fetches all 100 NDX tickers once (with Polygon rate limiting)
  - `_build_windows()` — creates rolling 6m train / 3m test windows
  - `_run_window()` — per-window: fit HMMs, daily rebalance loop, mark-to-market
  - `_daily_rebalance()` — single day: scan tickers, allocate weights, rebalance positions
- Fee model: 0.04% taker per side + 10bps slippage on changed notional only
- Outputs: `citrine_wf_results.csv` (window metrics), `citrine_wf_equity.html` (Plotly equity curve vs benchmark)
- CLI args: `--tickers`, `--train-months`, `--test-months`, `--long-only`, `--long-short`, `--cooldown [none/time/threshold]`, `--quiet`
- Reuses: `CitrineScanner`, `CitrineAllocator`, HMM infrastructure, indicator stack

### `live_trading_citrine.py`
Daily portfolio rotation engine for CITRINE (test/live mode).
- `ShadowTracker` class (Sprint 11): parallel allocator with relaxed thresholds that logs theoretical trades alongside live engine. Zero impact on live trading.
  - Thresholds: `SHADOW_ENTRY_CONFIDENCE=0.70`, `SHADOW_EXIT_CONFIDENCE=0.35`, `SHADOW_PERSISTENCE_DAYS=1`, `SHADOW_MAX_POSITIONS=30`
  - `_init_shadow_table()` — creates `shadow_trades` (24 columns) and `shadow_snapshots` tables with safe ALTER TABLE migration for new columns
  - `_restore_shadow_state()` — rebuilds shadow positions from latest `shadow_snapshots` row on restart
  - `run_shadow_cycle(scans, alt_data_boosts)` — runs relaxed-threshold allocation against same scan data; temporarily monkey-patches config globals, then restores
  - `_log_shadow_trade()` — logs to `shadow_trades` with full metadata: 24 fields including `indicator_json` (per-indicator boolean breakdown), `live_would_enter` (flag: conf≥0.90 AND persist≥3 AND BULL), `target_weight`, `scaled_weight`
  - `_log_shadow_snapshot()` — writes position JSON + theoretical equity to `shadow_snapshots`
- `CitrinePosition` dataclass: ticker, direction, entry_price, entry_date, size, sector, signal_strength, entry_atr, entry_confidence, highest_price, lowest_price, mae_pct, mfe_pct, mae_atr, mfe_atr. `chandelier_stop()` method returns stop price = `highest_price - entry_atr × 2.0`. Watermarks updated on every daily cycle via `update_watermarks(high, low)`.
- `CitrineLiveEngine` class:
  - `run_forever()` — infinite loop: daily 4pm ET scan→allocate→rebalance (Mon-Fri only for equities)
  - `_run_daily_cycle()` — orchestrator: scan, **risk exits**, allocate, rebalance, **shadow cycle** (Step 4b), log, check kill-switch. Risk exits run BEFORE allocator so stopped-out positions free capital for new entries. Shadow cycle runs after allocator update with same scan data.
  - `_check_risk_exits()` — Sprint 9: iterates all held positions, checks (a) Chandelier stop: `current_price < position.chandelier_stop()`, (b) confidence velocity: `delta_conf < -0.25 AND close < prev_close`. Exits with categorized `exit_reason`.
  - `_execute_rebalance()` — compare current vs target positions, exit/enter/scale as needed
  - `_enter_position()` — market order entry with 10bps slippage (test) or live CDP (live); deduplication guard skips if ticker already held. Sprint 9: now records `entry_atr` and `entry_confidence` from scan data.
  - `_exit_position()` — market order exit, calculate P&L, log trade. Sprint 9: logs `exit_reason`, `mae_pct`, `mfe_pct`, `mae_atr`, `mfe_atr`, `hold_days` to DB.
  - `_scale_position()` — increase size on Day 2 per fast scaling schedule (Sprint 1 update)
  - `_restore_state_from_db()` — Sprint 1 fix: reads latest portfolio_snapshots from `citrine_trades.db` and rebuilds `self.positions` on restart; calls `_restore_allocator_holdings()` to rebuild `allocator._holdings`
  - `_log_snapshot()` — Sprint 1 fix: uses scan_prices dict for mark-to-market (was using stale entry prices)
  - `_sleep_with_risk_checks()` — Sprint 3: replaces simple sleep between daily cycles; wakes every 4 hours to run `_check_intraday_risk()`
  - `_check_intraday_risk()` — Sprint 3: fetches current prices, computes unrealized P&L; warns on position loss >5% or portfolio loss >3%; triggers `_emergency_exit_all()` if total unrealized loss >5% of capital
  - `_emergency_exit_all()` — Sprint 3: forced liquidation of all positions when intra-day kill-switch triggers
  - Kill-switch automation: total loss > 5%, rolling 20-trade Sharpe < 0.3, 0/10 wins, **intra-day unrealized loss > 5%** (Sprint 3)
  - **Kill-switch grace period** (Sprint 4): `_KILL_SWITCH_GRACE_CYCLES = 3` — skips kill-switch checks for first 3 daily cycles after restart, preventing restart loops when old DB trades have poor Sharpe
- SQLite logging: `citrine_trades.db` (trades: timestamp, ticker, side, size, price, P&L, signal_strength, entry_atr, exit_reason, mae_pct, mfe_pct, mae_atr, mfe_atr, entry_confidence, hold_days) + portfolio_snapshots (equity, cash, positions JSON)
- Position sizing: $25k capital, ATR-based inverse vol parity (1% risk budget per trade, 15% max notional cap), adaptive cash management per allocator
- CLI args: `--test`, `--live`, `--tickers`, `--long-only`, `--cooldown`
- Reuses: `CitrineScanner`, `CitrineAllocator`, `src/notifier.py` (all notification channels)

### `citrine_dashboard.py`
Real-time portfolio visualization dashboard (Dash + Plotly, port 8070).
- Dash app with dark terminal UI + auto-refresh every 60 seconds
- Key charts:
  - Equity curve: cumulative total equity vs QQQ benchmark (if live trades exist, else backtest baseline)
  - Cash % over time: shows adaptive cash band behavior
  - Current allocation: pie chart of tickers × direction × weights
  - Regime distribution: bar chart showing BULL/BEAR/CHOP count daily
  - P&L histogram: closed trades binned by return % (for live trading view)
  - Kill-switch panel: 3 rule indicators (green/red) + status explanations
  - **Position health table** (Sprint 3): `_position_health_table()` — ticker, direction, entry/current price, unrealized P&L, sector for all open positions
  - **Signal frequency card** (Sprint 3): `_signal_frequency_card()` — 7-day entry/exit counts vs expected frequency, flags if >20% deviation from backtest baseline
- Data sources: `citrine_trades.db` (live trades), `citrine_wf_results.csv` (backtest baseline if no live trades)
- Falls back gracefully if live trading hasn't started (shows backtest equity curve)
- Reuses: Dash + dash-bootstrap-components framework from `live_dashboard.py` (AGATE)

### `optimize_citrine.py`
Per-ticker HMM parameter optimizer for CITRINE.
- Grid definition: n_states(3: 4,5,6) × feature_set(3: "base"/"extended"/"extended_v2") × confirmations(3: 6,7,8) × cov_type(2: "diag"/"full") = **54 combos per ticker**
- 100 NDX tickers × 36 combos = 3,600 total possible trials
- Default sampling: 200 random trials per optimizer run (covers ~6% of space)
- Main function: `_run_single_trial(trial)` — runs walk-forward (6m train / 3m test) for one config, returns per-window Sharpes
- Aggregates: per-ticker best config, global best config, outputs heatmap
- Outputs:
  - `citrine_optimization_results.csv` — all trial results (trial_id, ticker, n_states, feature_set, confirmations, cov_type, mean_sharpe, std_sharpe, windows_positive)
  - `citrine_per_ticker_configs.json` — best config per ticker (maps ticker → {n_states, feature_set, confirmations, cov_type, mean_sharpe})
  - `citrine_optimization_heatmap.html` — interactive Plotly heatmap (4 dropdowns: ticker, feature_set, confirmations, cov_type) showing Sharpe per n_states choice
- CLI args: `--runs 200` (default), `--resume`, `--ticker AAPL` (single ticker), `--heatmap-only` (regenerate plot only)
- Checkpoints every 10 runs to CSV (resumable with `--resume`)
- Per-ticker configs feed back into `CitrineScanner` via `load_per_ticker_configs()` for optimized scanning
- Reuses: `walk_forward_ndx.run_walk_forward()` as fitness function, same HMM infrastructure as BERYL optimizer

### `optimize_beryl_p3.py`
BERYL Phase 3 optimizer — 5 tickers with expanded grid.
- Grid: ticker(5: TSLA/NVDA/AAPL/MSFT/GOOGL) × n_states(5: 3-7) × feature_set(3) × confirmations(3) × leverage(2) × cooldown(2) × cov_type(2) × timeframe(3) = 5,400 combinations
- Uses `walk_forward_ndx.py` as fitness function (same as P1/P2)
- Checkpoints to `beryl_p3_results.csv` (resumable with `--resume`)
- Outputs: `beryl_p3_results.csv`, `beryl_p3_heatmap.html`

### `optimize_allocator.py`
CITRINE allocator parameter optimizer — pre-compute + replay approach.
- Pre-computes HMMs and scan results once for all walk-forward windows (expensive step done once)
- Replays allocation logic for 3,456 parameter combinations at ~0.7 trials/second (~38 min total)
- Grid: entry_confidence(6) × exit_confidence(6) × persistence_days(4) × max_positions(4) × cash_profile(3) × scale_profile(2) = 3,456 combos
- **Winner (Sprint 1)**: entry=0.90, exit=0.50, persist=3, max_positions=15, aggressive cash, fast scale → Sharpe +1.665
- Outputs: `allocator_optimization_results.csv`, sensitivity analysis by parameter

### `optimize_indicators.py`
AGATE indicator threshold optimizer — fixes HMM config, varies indicator thresholds.
- Grid: adx_min(3: 15/20/25/30) × stoch_upper(3: 70/75/80) × volume_mult(3: 1.0/1.1/1.2) × volatility_mult(3: 1.5/2.0/2.5) = 81 combinations
- Uses full ensemble HMM (`--ensemble` flag); fixes feature set, timeframe, n_states
- **Winner (Sprint 1)**: ADX_MIN=20 → Sharpe 2.879 (was 2.455 at ADX=25); volatility_mult has ZERO impact
- Outputs: `indicator_optimization_results.csv`, sensitivity breakdown by parameter
- CLI: `--ensemble` (use ensemble HMM), `--runs N`, `--resume`

### `optimize_agate_multi.py`
AGATE multi-ticker walk-forward optimizer (Sprint 6, updated Sprint 8).
- Grid: 14 tickers × n_states[4,5,6] × feature_set["base","extended","extended_v2"] × confirmations[5,6,7,8] × cov_type["diag","full"] × timeframe["3h","4h"] = 2,016 combinations
- Uses ensemble HMM in walk-forward trials (6m train / 3m test)
- CSV schema validation (`_validate_result()`) prevents corrupt output lines
- **Trade frequency penalty (Sprint 8)**: `rank_score = Sharpe × consistency × log(trades)` — penalizes low-N configs that may have high Sharpe by chance. Same intuition as Lo (2002) Sharpe standard error: `SE(Sharpe) ≈ 1/√N`. The `log(N)` form barely penalizes 30→50 trade difference but crushes 3→1 trade configs.
- Auto-generates `agate_per_ticker_configs.json` with best config per crypto ticker
- Outputs: `agate_multi_optimization_results.csv`, per-ticker best configs
- CLI: `--runs 200` (default), `--resume`, `--heatmap-only`, `--workers N`
- Results (179 trials): BCH Sharpe +3.52 (5/5 windows ★), ETH +3.59, XLM +2.15

### `reconcile.py`
Backtest-vs-live reconciliation script for validating signal consistency.
- Runs all 3 sub-projects (AGATE, BERYL, CITRINE) through backtester and compares vs live trading signals
- Detects signal mismatches, feature calculation discrepancies, and regime assignment drift
- Outputs: `reconciliation_results.json` — per-project match rates and any detected discrepancies
- **Key finding (Sprint 2)**: Identified critical AGATE signal bug (always HOLD) via reconciliation; after fix, first BUY signal confirmed

### `tests/__init__.py`
Empty init file for test package.

### `tests/test_strategy_consistency.py`
Comprehensive strategy consistency test suite — 69 tests, all passing.
- Tests cover: feature set validation (3/5/7 features after vol_price_diverge removal), ensemble voting correctness, HMM model fit/predict, signal series column names, kill-switch threshold enforcement, signal consistency across restarts, config integrity checks, indicator computation, CITRINE allocator config, position state persistence, v2 feature continuity (realized_kurtosis, volume_return_intensity, return_momentum_ratio)
- Uses synthetic OHLCV data (no external API calls — fully offline)
- Run with: `/Users/perryjenkins/opt/anaconda3/bin/python -m pytest tests/test_strategy_consistency.py -v`

### `health_check.sh`
Comprehensive automated health check script (8 sections).
- Process health (verify all services running)
- Log freshness (detect sleep gaps — AGATE: max 6h, BERYL/CITRINE: max 26h)
- Network connectivity (Polygon API reachable)
- Config integrity (FEATURE_SET=extended_v2, confirmations=7, no vol_price_diverge)
- Database health (all .db files exist, trade counts, kill-switch win rate)
- Test suite (69/69 pytest tests pass)
- HMM convergence (ensemble convergence checks)
- Data cache (SOL cache freshness, per-ticker configs count, cron jobs)
- Run with: `bash health_check.sh`

### `run_optimize_citrine.sh` / `run_optimize_beryl.sh` / `run_optimize_agate.sh`
Weekly auto-optimization cron scripts (Sunday 2am/4am/8am).
- Each runs optimizer with `--resume`, logs output, extracts top results
- Sends Pushover + macOS notification with summary
- CITRINE auto-updates `citrine_per_ticker_configs.json` (hot-swappable)
- AGATE/BERYL results require manual review

### `mae_calibration.py`
Offline MAE/MFE calibration for CITRINE's Chandelier exit multiplier.
- Walk-forward HMM backtests on NDX100 tickers using HMM-only exits (no trailing stops)
- Tracks per-trade MAE/MFE in both % and ATR units using intraday high/low
- `ExcursionTrade` dataclass: ticker, entry/exit dates/prices, entry_atr, pnl_pct, mae_pct/mfe_pct, mae_atr/mfe_atr, mae_day/mfe_day, hold_days, is_winner
- Loads per-ticker configs from `citrine_per_ticker_configs.json`
- Addresses three statistical traps: (1) zero-MAE mass skewing percentiles, (2) time-in-trade normalization, (3) bull/bear regime coverage
- Outputs recommended Chandelier multiplier: `abs(95th_pctile_nonzero_winner_MAE) × 1.1`
- Includes "would-have" analysis showing impact at 2.0, 2.5, 3.0, 3.5 ATR stops
- **Calibration results (2026-03-30)**: 562 trades across 30 tickers, 3 years. 95th pctile winner MAE = 1.793 ATR → recommended multiplier = **2.0 ATR**
- CLI: `python mae_calibration.py --tickers 20`

### `signal_decay_analysis.py`
AGATE signal decay analysis — measures forward returns at t+1, t+2, t+4 bars (4h intervals) after BUY signals.
- Reads AGATE journal DB for historical BUY signals
- Reports mean forward return by ticker and overall
- Determines if edge is immediate or requires regime settling
- Run with: `python signal_decay_analysis.py`

### `research/semi_markov_analysis.py`
Semi-Markov sojourn distribution analysis — KS test comparing geometric vs empirical.
- Fits HMM on SOL/ETH, extracts transition matrix
- Computes theoretical geometric sojourn CDF vs empirical from Viterbi decode
- If empirical significantly deviates (p < 0.05), recommends Semi-Markov investigation
- Run with: `python research/semi_markov_analysis.py`

### `research/kalshi_equity_mapping.json`
Manual mapping of Kalshi prediction markets to NDX100 equity tickers for cross-system hedging.
- Covers earnings events (NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA)
- Macro indicators (Fed rate, CPI, GDP, unemployment)
- Sector events (chip export restrictions → semiconductor tickers)
- Integration plan: DIAMOND anomaly scores as CITRINE signal multiplier

### `citrine_meta_model.py`
Meta-learning analysis of CITRINE optimizer results — identifies under-explored configs and generates recommendations.
- Analyzes 734 optimizer trials across 100 tickers
- Identifies extended_v2 as under-deployed (7% usage, highest mean Sharpe)
- Generates v2 config recommendations per ticker
- **CRO warning**: Recommendations are unvalidated hypotheses — each must be walk-forward tested before deployment

### Polygon.io Data Availability
- BTC: ~24 months of hourly data (full 730-day window)
- ETH/XRP/SOL: ~14 months of hourly data (cache starts Jan 2025)
- Walk-forward optimizer uses 6m/3m windows to accommodate shorter alt data histories

---

## DIAMOND — Kalshi Unusual Volume Tracker (BUILT & DEPLOYED)

**Status**: Built and running 24/7 on Oracle Cloud VM. WebSocket streaming active, paper trading engine operational.

**Project home**: `/Users/perryjenkins/Documents/trdng/kalshi-diamond/` (standalone repo, NOT in HMM-Trader)
**VM location**: `/home/ubuntu/kalshi-diamond/` on Oracle Cloud VM

### What DIAMOND Does
Real-time anomaly detection on Kalshi prediction markets. Streams trades via WebSocket, computes 6 detection features per trade, flags unusual activity, and executes paper trades on high-confidence signals. Dashboard at port 8080 shows live anomaly feed + portfolio status.

### Architecture (Deployed)
```
Kalshi WebSocket ──► Stream Processor ──► Feature Engine ──► Alert Engine
     (trade channel)    (asyncio)         (6 detectors)     (Pushover/SQLite)
         │                                                        │
         └──► REST Poller ──► Market Metadata Cache               ▼
              (order book,     (refresh 5min)              Dashboard (:8080)
               market info)                                Paper Trading Engine
                                                           (auto-fill, P&L tracking)
```

### Detection Features (6 Implemented)
1. **Trade Size Z-Score**: `z = (trade_size - mean) / std` per market, rolling 24h window
2. **Volume Spike Ratio**: `current_1h_volume / rolling_24h_avg_hourly_volume`
3. **Order Book Imbalance**: `|yes_depth - no_depth| / (yes_depth + no_depth)`
4. **Taker Side Skew**: `yes_taker_volume / total_volume` in last 1h (deviation from 0.5)
5. **Price Impact**: price change within 30s after large trade
6. **Cross-Market Correlation**: simultaneous anomalies across related markets (same event series)

### Alert Levels
- **LOG** (score > 0.3): SQLite only
- **NOTABLE** (score > 0.5): macOS/terminal notification
- **ALERT** (score > 0.7): Pushover push
- **CRITICAL** (score > 0.9 or multiple features > 0.7): Emergency Pushover

### Paper Trading Engine
- Automatically enters trades on high-confidence anomalies
- Tracks fill price, P&L, settlement status
- Kill-switch: max 10 concurrent positions, daily loss limit
- Dashboard shows: total P&L, return %, win rate, trades 1h/24h, best/worst trade, deployed capital

### Database Schema
- `trades` — raw trade log (trade_id, ticker, size, price, side, timestamp)
- `anomalies` — flagged events (trade_id, ticker, z_score, volume_ratio, features, alert_level)
- `market_profiles` — per-market stats (ticker, mean_size, std_size, mean_volume, updated_at)
- `book_snapshots` — periodic order book state
- `paper_trades` — paper trading positions and P&L

### Key Files (in kalshi-diamond repo)
| File | Purpose |
|------|---------|
| `config.py` | API keys, thresholds, feature weights |
| `src/kalshi_client.py` | WebSocket + REST API client |
| `src/diamond_store.py` | SQLite storage + rolling aggregates |
| `src/diamond_features.py` | 6 detection features + composite score |
| `src/diamond_alerts.py` | Alert engine with cooldowns |
| `src/paper_trader.py` | Paper trading engine (auto-fill, P&L) |
| `diamond_monitor.py` | Main async loop (entry point) |
| `diamond_dashboard.py` | Dash dashboard at :8080 (anomaly feed + portfolio status) |

### Kalshi API
- **WebSocket**: `wss://api.kalshi.co/trade-api/ws/v2` — `trade` channel (public), `orderbook_delta` (auth), `ticker` (public)
- **REST**: `/trade-api/v2/markets/{ticker}/trades`, `/orderbook`, market metadata
- **Rate limits**: Basic tier (20 read/sec, 10 write/sec) — sufficient for monitoring

---

## Known Issues to Avoid

### Do NOT use `add_vrect()` in Plotly
Plotly 6.x broke `fig.add_vrect()` with thousands of shape objects. Use `go.Scatter` with `fill="toself"` on an overlay `yaxis2` instead. See `build_price_chart()` in `src/dashboard.py` for the correct pattern.

### Do NOT use `use_reloader=True` in Dash
The Werkzeug reloader spawns a subprocess that inherits the parent's state incorrectly and causes silent exits. Always run with `use_reloader=False`.

### Do NOT run with `mynewenv`
The `mynewenv` conda environment is missing critical dependencies (hmmlearn, dash, plotly). Always use the base environment at `/Users/perryjenkins/opt/anaconda3/bin/python`.

### Stale `__pycache__` after edits (FIXED on VM — ExecStartPre auto-clears)
All systemd services now have `ExecStartPre=/usr/bin/find /home/ubuntu/HMM-Trader -name "*.pyc" -delete` which auto-clears bytecode on every restart. On Mac, manually clear if needed:
```bash
find . -name "*.pyc" -delete
```

### `build_signal_series` column name
The confidence column in the HMM output is named `confidence`, not `regime_confidence`. Any code checking for this column must use the correct name.

### `Backtester` API
`Backtester(use_regime_mapper=False)` — optional bool. The DataFrame is passed to `.run(df)`, not the constructor. `BacktestResult` exposes `.metrics` (dict) not direct attributes like `.total_return_pct`. `SignalEngine(use_regime_mapper=False)` follows the same pattern.

### Config monkey-patching in optimizer
`_patch_config` / `_restore_config` in `optimize.py` mutate the `config` module globals directly. This is safe only because the optimizer is single-threaded. Do not parallelize trials without replacing this mechanism.

### Polygon timestamp format
The Polygon `/v2/aggs/` endpoint requires Unix **millisecond** timestamps for `from`/`to` parameters, not date strings. Date strings silently return wrong data.

### `vol_price_diverge` is binary — REMOVED from all feature sets (Sprint 1)
`vol_price_diverge` computes `(price_dir != vol_dir).astype(float)` — values are only 0 or 1. The GaussianHMM assumes continuous Gaussian emissions; a binary feature produces a degenerate covariance matrix (non-positive-definite for `full` covariance, and NaN `startprob_` for `diag` with ≥5 states). **Fixed in Sprint 1**: removed from `config.py` "extended" (6→5 features) and "full" (8→7 features) definitions. Also removed from `src/signal_generator.py` and `live_trading_beryl.py` imports. `vol_price_diverge` is still computed by `attach_all()` for reference but must NOT be added back to any feature set.

### Prefer `extended_v2` over `extended` for AGATE (Sprint 3)
The `extended_v2` feature set (6 features: base + `realized_vol_ratio` + `return_autocorr` + `realized_kurtosis`) outperforms `extended` (5 features) by +2.747 Sharpe on SOL/4h/ensemble A/B test. Use `extended_v2` for any new AGATE optimization or live trading configuration. Do NOT use `full_v2` (10 features) — it overfits due to too many dimensions for the HMM to reliably estimate.

### HMM training is 3x slower after Sprint 3 regularization
`HMMRegimeModel.fit()` now uses `n_init=3` (3 random initialization attempts). This makes each HMM fit ~3x slower but significantly reduces convergence failures and degenerate covariance issues. The ensemble (3 HMMs) therefore takes ~9x the time of a single pre-Sprint-3 fit. Optimizer runtimes are proportionally affected.

### CITRINE sector cap may silently drop high-scoring candidates
`_apply_sector_cap()` (Sprint 3) enforces `CITRINE_MAX_PER_SECTOR = 4`. If a sector (e.g., semiconductors) has >4 candidates passing all filters, only the top 4 by CITRINE score are kept. The dropped tickers are not logged at INFO level — check DEBUG logs if allocation seems unexpectedly low.

### HMM label ordering with n_states > 7 (old bug, now fixed)
Prior to the fix in `_build_label_map`, states were labelled by concatenating `_LABEL_TEMPLATE + _LABEL_EXTRA`. With 8 states this put the worst-return state at rank 7 into `chop_extra_0` (CHOP) instead of `bear_crash` (BEAR). The fix assigns bull/bear labels from top/bottom of the return ranking, and assigns chop labels to the middle states by HMM covariance magnitude. No hardcoded length assumptions remain.

### `build_signal_series` output columns (critical — do not regress)
`build_signal_series()` in `src/strategy.py` produces `raw_long_signal` (bool), `regime_cat` (str), `confidence` (float), and indicator check columns — it does NOT create a column called `"signal"`. Any code using `latest.get("signal", "HOLD")` will always return HOLD. **Fixed in Sprint 1** in `src/signal_generator.py`: now uses `raw_long_signal` and `regime_cat` to determine the signal string (BUY/SELL/HOLD). Do not revert this change.

### Coinbase CDP authentication is placeholder
The `src/live_broker.py` Coinbase CDP integration uses placeholder JWT signing. For `--live` mode to work with real money, proper JWT signing with the CDP API key must be implemented. `--test` mode works without real credentials (simulates fills locally with 10bps slippage).

### Ensemble optimization output filename
When running `optimize_wf.py --ensemble`, the output still writes to `optimization_wf_results.csv` (same as single-model). This means ensemble results overwrite single-model results if both are run. Consider renaming to `optimization_wf_ensemble_results.csv` if both need to coexist.

### Polygon intraday equity data requires pagination
The Polygon `/v2/aggs/` endpoint returns max ~900 bars per request for hourly equity data. `walk_forward_ndx.py` now uses cursor-based pagination to fetch all bars (e.g., 7,986 hourly bars for TSLA over 2 years). The `fetch_equity_hourly()` function handles this automatically.

### Original WF Sharpe degrades on current data
The ensemble WF Sharpe of 6.240 (from optimization) has degraded to -4.668 on current SOL data (2026-03-17 stability test). This is expected — market regimes shift constantly. The `extended_v2` feature set with `confirmations=7` produces Sharpe +0.837 on current data, confirming the Sprint 3.1 feature improvements were critical.

### BERYL HMM convergence with few daily observations
BERYL uses daily bars for equities. With ~58 daily observations per 3-month test window, HMMs with 4+ states may not converge (especially for volatile tickers like TSLA). This is structural — not a bug. The optimizer skips non-converged trials.

### `osascript` notifications don't work on Linux VM
`src/notifier.py` uses `osascript` for macOS native notifications. On the Oracle Cloud Ubuntu VM, these silently fail. Only Pushover push and terminal bell work on the VM. This is expected — macOS-specific features are no-ops on Linux.

### VM RAM is tight (568MB/956MB used)
The 8 systemd services consume ~680 MB total on 956 MB available RAM + 4 GB swap. Adding more services may cause swap thrashing. Monitor with `free -h` on the VM. If RAM becomes an issue, consider migrating to Hetzner CCX23 (~$25/mo, 4 dedicated CPU/16GB RAM).

### `reconcile.py` may have import issues on VM
The reconciliation script was designed for Mac development environment. Some imports may fail on the VM's miniconda Python 3.13 (vs Mac's Python 3.9). Run reconciliation on Mac, not VM.

### VM Python version mismatch (3.13 vs 3.9)
Mac uses Python 3.9.12 (base conda), VM uses Python 3.13 (miniconda). Most code works on both, but watch for deprecated stdlib usage or `match/case` syntax that only works on 3.10+. Test on Mac first, then deploy to VM.

### CITRINE kill-switch restart loop (FIXED 2026-03-21)
Old trades with poor Sharpe (-0.382) caused kill-switch to trigger immediately on restart → systemd restarted → kill-switch again → infinite loop. CITRINE missed a full trading day (2026-03-20) because of this. Fixed by: (1) DB reset on VM (backed up old DB), (2) added `_KILL_SWITCH_GRACE_CYCLES = 3` — skips kill-switch checks for first 3 daily cycles after restart.

### Local Mac processes conflict with VM dashboards
After migrating to Oracle Cloud, stale Mac processes (traders + dashboards) may still be running locally on the same ports, serving old data. This causes confusion (e.g., seeing 03/19 trades when VM has 03/21 data). Kill all local processes after migration:
```bash
ps aux | grep -E '(live_dashboard|citrine_dashboard|diamond_dashboard|live_trading)\.py' | grep -v grep
# Kill any found PIDs — all trading/dashboards now run on VM only
```

### Oracle Cloud Security List — correct subnet
Port ingress rules must be added to the Security List attached to the instance's actual subnet, not the default VCN security list (which may be different). Navigate: **Compute → Instances → instance → Primary VNIC → Subnet → Security Lists**. This is the #1 gotcha when ports appear open but connections time out.

### Dashboard browser caching
Dash dashboards can serve stale pages from browser cache after service restarts or DB resets. Fixed by adding `Cache-Control: no-store` headers to all dashboard apps (live_dashboard.py, citrine_dashboard.py, diamond_dashboard.py). If still stale: `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R` (Windows).

### CITRINE `_position_health_table()` dict iteration (FIXED 2026-03-21)
Portfolio snapshots store positions as a JSON dict. The `_position_health_table()` function was iterating dict keys (strings like "NVDA") but calling `.get()` on them, causing `AttributeError: 'str' object has no attribute 'get'`. Fixed to iterate `positions.items()` and handle both dict-of-dicts and list-of-dicts formats.

### CITRINE Cash % metric showed allocator target, not actual
The top-row "CASH" metric showed `cash_pct` from the allocator (target %), not actual cash/equity ratio. This caused a mismatch with the pie chart (which showed actual). Fixed to compute `cash / equity` directly from portfolio snapshot.

### macOS crontab doesn't fire during sleep (FIXED 2026-03-22)
Sunday optimizer cron jobs at 2am/4am/8am were silently skipped when the MacBook lid was closed. macOS cron does NOT catch up on missed jobs. **Fixed**: migrated all 3 optimizer schedules to `launchd` plist agents in `~/Library/LaunchAgents/com.hmm-trader.optimize-*.plist`. launchd fires missed jobs on wake. Old crontab entries removed.

### BERYL ensemble scan takes ~33 min (vs ~16 min single-model)
Ensemble HMM trains 3 models per ticker (Sprint 5). With 98 tickers × 12s rate limit × 3 HMM fits, a full scan takes ~33 minutes. This is acceptable for daily-bar trading but means signals are generated ~17 min later than before. If scan duration becomes an issue, consider reducing `gc.collect()` frequency or parallelizing non-Polygon-dependent work.

### BERYL position persistence added (Sprint 10 — FIXED 2026-04-01)
Prior to Sprint 10, BERYL had NO state persistence — positions were silently lost on every service restart. March 30 positions (ARM, MELI, TMUS — $2,499 total notional) were abandoned when the service restarted between March 30 and 31 (PID changed 270432 → 309683). No exits logged, P&L never recorded — unrecoverable. **Fix**: Added `portfolio_snapshots` table to `beryl_trades.db`, `_log_snapshot()` after every scan cycle, and `_restore_state_from_db()` on startup. Mirrors CITRINE's Sprint 1 pattern.

### BERYL status file showed stale data (Sprint 10 — FIXED 2026-04-01)
`_write_status()` was called BEFORE `process_signals()` in the main loop — the dashboard JSON always showed pre-trade state (empty positions after restart, no price movement). Fixed by moving `_write_status()` after `process_signals()`.

### BERYL dashboard positions dict vs list (Sprint 10 — FIXED 2026-04-01)
`_beryl_regime_panel()` in `live_dashboard.py` assumed positions were a list (`positions[:3]`), but BERYL stores them as a dict (`{"MCHP": {...}}`). Fixed with `isinstance(raw_positions, dict)` → `list(dict.values())`.

### Insider 10b5-1 field may be absent in older filings
The `aff10b5One` XML field in SEC Form 4 was introduced relatively recently. Older filings may not have this field, in which case `is_10b5_1` defaults to `False` (conservative — treats as discretionary). This means the smart scoring is most accurate for recent filings (last 1-2 years).

### AGATE multi-ticker scan competes with Mac optimizer for Polygon API
When both the VM's 4h AGATE scan and the Mac optimizer are running simultaneously, both hit Polygon 429 rate limits. The VM scan may fail on 2-3 tickers per cycle (they succeed on the next cycle). Avoid running optimizers during active scan windows if possible. Polygon free tier returns only 83 bars/page (not the 5000 requested) — large backfills require ~210 pages per ticker with frequent 429 retries, taking ~20 min/ticker.

### AGATE data cache gaps (4 tickers, Sprint 8)
LTC, SUI, DOGE, and ADA were added to `AGATE_TICKERS` on 2025-12-22 (Sprint 6) with only ~95 days of cached data. Walk-forward optimization requires ≥270 days (6m train + 3m test = 9 months). These tickers silently returned `None` from the optimizer (all 144 combos failed due to "Need 9 months of data; only 3 available"). No error was visible in the optimizer CSV or summary — trials returned `None` and were dropped. **Fix (2026-03-28)**: backfilling all 4 to 730 days via Polygon API, then re-running `optimize_agate_multi.py --ticker X:LTCUSD` etc. until added to `agate_per_ticker_configs.json`.

### Folder restructure backward-compat symlinks
`~/Documents/trdng/HMM-Trader` is now a symlink to `~/Documents/quant/trading-core/`. All cron jobs, launchd plists, and scripts using the old path continue working. If a symlink breaks, recreate: `ln -s ~/Documents/quant/trading-core ~/Documents/trdng/HMM-Trader`

### CITRINE `alt_data_boosts` crash (FIXED 2026-03-23)
`citrine_allocator.py` `_compute_scores()` method was missing the `alt_data_boosts` parameter — `allocate()` accepted it but didn't pass it through. Caused `NameError` on every scan cycle, preventing all new trades since 2026-03-21. Fixed by adding the parameter to `_compute_scores()` signature and threading it through the pipeline.

### Zombie process incident (2026-03-25/26 — FIXED)
All 3 traders went zombie after file deploy — systemd showed "active (running)" but memory was 1-3MB (should be 28-183MB) and zero log output. CITRINE missed a full trading day. Root cause: stale `__pycache__` after SCP file updates. Fixed with: (1) `ExecStartPre` auto-clears pycache on every restart, (2) watchdog timer checks journald freshness + memory every 15 minutes, auto-restarts zombies.

### `fetch_equity_daily(ticker, days=5)` TypeError (FIXED 2026-03-26)
Three call sites in BERYL and CITRINE intraday risk checks called `fetch_equity_daily(ticker, days=5)`, but the function signature is `fetch_equity_daily(ticker, years=20)` — `days` kwarg doesn't exist. Silent TypeError → except block → fallback to entry_price → all intraday P&L showed +0.0%. Fixed by adding `fetch_latest_price()` using Polygon Snapshot API.

### CITRINE cash band not enforced (FIXED 2026-03-26)
The allocator computed `cash_pct` (e.g., 30%) and multiplied weights by `invested_pct`, but `_execute_rebalance()` ignored the floor — `_enter_position()` used `notional = min(w.notional_usd, self.cash)` which drained cash to $0. Portfolio was at 0.1% cash when it should have been 30%. Fixed by computing `min_cash = equity * cash_pct` and guarding entries.

### Persistence bonus was inverted (FIXED 2026-03-26)
CITRINE `_compute_scores()` gave a 1.0-1.5x persistence_bonus as regimes aged — scaling positions UP at the worst time. Regimes are metastable; expected remaining lifetime decreases with age. ROP and INTU were at max size when regimes flipped → -$257 combined loss. Fixed by replacing with sojourn decay using HMM transition matrix half-life.

### CITRINE exit at $0.00 (FIXED 2026-03-26)
When CITRINE exited positions for tickers not in the latest scan (e.g., dropped below min observations), `scan_map.get(ticker)` returned None → exit price was $0.00 → phantom -100% loss logged. Fixed by using scan prices from the scan and guarding against None.

### AGATE `log_entry()` duplicate insertions (FIXED 2026-03-26)
`log_entry()` was called on every scan cycle, not just the first entry, creating 13 duplicate ENAUSD entries in `open_positions`. Fixed by adding deduplication guard: checks if ticker already exists before inserting.

### DIAMOND ML scorer in-sample fraud (FIXED 2026-03-25)
`compare_with_handtuned()` used `predict()` on the same data the model was trained on — the +$6.26 ML P&L was entirely in-sample. Honest OOS numbers: 55.6% win rate on 36 trades, +27 cents. Fixed: comparison now uses only held-out CV fold predictions. Platt scaling disabled at N<500. Wilson CI added.

### Chandelier stops may fire on same-day entries (2026-04-02 — FIXED 2026-04-04, Sprint 12)
On 2026-04-02, 6 positions were stopped out intraday by Chandelier stops, several entered *that same morning*. Root cause: execution-observation misalignment — intraday spot prices evaluated against daily-ATR-derived stops. Two fixes deployed: (1) Day-0 Chandelier immunity in `_check_risk_exits()` — skips Chandelier when `entry_date == today`, (2) Chandelier removed entirely from `_check_intraday_risk()` — intraday checks use only -2% hard stop. Additionally, `entry_time` restoration from DB was fixed to prevent false immunity after restarts. Validated Apr 3-4: zero false stops, 10 positions correctly handled.

### macOS Full Disk Access blocks ALL launchd agents (2026-03-28)
All launchd plists accessing `~/Documents/` fail with "Operation not permitted" — including the Sunday optimizer plists (`com.hmm-trader.optimize-*`) and auto-push plists (`com.quant.autopush-*`). Root cause: macOS `com.apple.provenance` attribute on `/bin/bash` prevents filesystem access from launchd context. **Fix**: System Settings → Privacy & Security → Full Disk Access → add `/bin/bash` (click `+`, `Cmd+Shift+G`, type `/bin/bash`). This is a one-time system setting change that fixes all current and future launchd agents. Also discovered the Sunday optimizer plists have been silently failing since the macOS update — they were NOT running.

## Optimization History

### Pass 1 — 106 trials (300 sampled, 106 valid)
- 74/106 runs flagged as overfit (OOS Sharpe < IS Sharpe − 0.5)
- Best non-overfit result: OOS Sharpe 2.567, +27.9% return, −10.3% DD, 60% WR
- Key findings:
  - 1h timeframe dominated (19/20 top runs)
  - n_states=8 consistently outperformed 4–7
  - volume_change (65.6%), realized_vol_ratio (44.5%), vol_price_diverge (38.9%) are dominant features
  - log_return and price_range contribute near zero — consider dropping
  - 72h cooldown and 365 training days were most robust
- Results saved in optimization_results.csv

### Pass 2 — 31 trials (100 sampled, 31 valid)
Fixed: n_states=8, timeframe=1h, training_days=365, cov_type=diag
Variable: ticker [BTC/ETH/XRP/SOL], feature_set [volume_focused, full], confirmations [6/7/8], leverage [1.25/1.5/1.75], cooldown [48/72]
- 69/100 skipped (mostly volume_focused non-convergence or <10 trades)
- **BTC only** made the top 10 — ETH and XRP had negative best OOS Sharpe
- SOL: modest positive (OOS Sharpe 1.022, +10.2%)
- Best overall: BTC + full + 6 confirmations + 72h cooldown → OOS Sharpe 2.745, +47% return, −23% DD
- `full` feature set dominated `volume_focused` in all tickers
- 72h cooldown consistently outperformed 48h
- Results saved in optimization_results_pass2.csv, heatmap in optimization_heatmap_pass2.html

### Walk-Forward Pass — 98 trials (100 sampled, 2 skipped)
Grid: ticker (BTC/ETH/XRP/SOL) × timeframe (1h/2h/3h/4h) × n_states × feature_set × confirmations × leverage × cooldown × cov_type
Walk-forward: 6m train / 3m test rolling windows
- 98/100 completed (2 skipped due to diag+8-state NaN startprob)
- **XRP dominated** — both top spots, best per-ticker Sharpe across all tickers
- Best overall: XRP + 2h + 6-state + full + 7-confirms + 2.0× + 48h + full → WF Sharpe 3.512, +56.8% return, −13.8% DD
- Key findings:
  - 7 confirmations (not 8) dominated top 10 — slightly looser entry filter is better
  - 2h and 3h timeframes outperformed 1h and 4h — sweet spot between noise and signal
  - 6 states appeared most often in top results — 5 also strong
  - Both `full` and `diag` covariance worked well (no clear winner)
  - BTC had 5 WF windows (more robust) but lower returns; alts had 2 windows
  - Most consistent config: SOL/2h (std_sharpe=0.003)
- Best per ticker: XRP WF Sharpe 3.512, SOL 2.463, ETH 2.358, BTC 2.264
- Results saved in optimization_wf_results.csv, heatmap in optimization_wf_heatmap.html

### Walk-Forward Multi-Direction Pass — 97 trials (100 sampled, 3 skipped)
Grid: base params + conf_high_threshold (5) × bull_med_action (3) × bear_med_action (3) = 777,600 combinations
Walk-forward: 6m train / 3m test rolling windows, multi-direction mode
- 97/100 completed (3 skipped due to diag+high-state NaN startprob)
- Only **20/97 (21%) had positive Sharpe** — vs 43% for long-only → multi-dir is significantly harder
- Mean Sharpe: -1.552 (vs -0.383 for long-only)
- Only **2/97 configs had ALL windows positive**
- Best overall: SOL/4h/4-state/base/8-confirms/2.0×/24h/diag, ct=0.90, LONG_OR_FLAT, SHORT → WF Sharpe 3.700, +36.6%
  - BUT inconsistent: std_sharpe=7.506, only 1/2 windows positive, SHORT WR=0%
- **Most consistent config**: ETH/1h/8st/extended/8cf, ct=0.80, LONG_OR_FLAT, SHORT_OR_FLAT → WF Sharpe 1.137, +5.06%, DD=-4.87%, std_sharpe=0.332, both windows positive
- Key findings:
  - **4h timeframe dominated** multi-dir: 41% positive vs 8–18% for other TFs
  - **LONG + SHORT** combo (bull_med=LONG, bear_med=SHORT): 50% positive (3/6), best mean Sharpe (-0.62)
  - **LONG_OR_FLAT + SHORT_OR_FLAT**: most conservative, appeared in both all-windows-positive configs
  - **conf_high_threshold=0.80**: most positive configs (7/25), good balance of signal frequency vs quality
  - **48h or 72h cooldown**: better than 24h (same as long-only finding)
  - SHORT trades average ~40.7% WR — comparable to LONG (40.6%) but worse consistency
  - ETH appeared 4× in top 10 consistent configs — best ticker for multi-dir
- Best per ticker: SOL WF Sharpe 3.700 (inconsistent), ETH 1.782, XRP 1.648, BTC 0.534
- **Conclusion**: Multi-direction adds risk without proportional reward at this stage. Long-only (WF Sharpe 3.512, +56.8%) significantly outperforms the best consistent multi-dir config (Sharpe 1.137, +5.06%). SHORT signals need further refinement before production use.
- Results saved in optimization_wf_multidir_results.csv (overwritten by Pass 2), heatmap in optimization_wf_multidir_heatmap.html

### Walk-Forward Multi-Direction Pass 2 (Asymmetric SHORT) — 96 trials (100 sampled, 4 skipped)
Grid: Pass 1 grid × confirmations_short [7, 8] × cooldown_hours_short [48, 72] × adx_min_short [25, 30] = 6,220,800 combinations
Walk-forward: 6m train / 3m test rolling windows, multi-direction mode with asymmetric SHORT params
- 96/100 completed (4 skipped due to diag+high-state NaN startprob)
- **21/96 (22%) had positive Sharpe** — similar to Pass 1 (21%), but **4 all-windows-positive configs** (up from 2)
- Mean Sharpe: -1.268 (improved from -1.552)
- Best overall: XRP/2h/7st/full/7cf/2.0×/72h/diag, ct=0.75, LOF/SHORT, cs=7, cds=48, adx_s=30 → WF Sharpe 1.680, +31.8%
  - BUT inconsistent: std_sharpe=1.742, only 1/2 windows positive
- **Most consistent**: SOL/4h/8st/full/8cf/2.0×/72h/diag, ct=0.75, FLAT/FLAT, cs=7, cds=48, adx_s=25 → WF Sharpe 1.594, +7.9%, DD=-12.3%, std_sharpe=0.025, BOTH windows positive
  - Note: FLAT/FLAT mapping is effectively long-only within multi-dir framework
- **Best genuine multi-dir consistent**: XRP/4h/6st/ext/8cf/1.5×/48h/full, ct=0.75, FLAT/SHORT, cs=7, cds=72, adx_s=25 → WF Sharpe 1.409, +10.6%, BOTH windows positive, std=0.493
- Key findings (asymmetric SHORT params):
  - **`confirmations_short=8` is the clearest winner**: 26% positive vs 17% for cs=7; mean Sharpe -1.090 vs -1.496; dominated positive configs 14:7
  - **`adx_min_short=30`**: marginal improvement in mean Sharpe (-1.185 vs -1.366) but fewer positive configs (19% vs 25%) — mixed signal
  - **`cooldown_hours_short=72`**: marginal improvement (24% vs 20% positive)
  - **SHORT win rate improved to 54.2%** among positive configs (vs ~40.7% in Pass 1) — asymmetric params filtering out bad SHORT entries
  - SHORT trades less frequent but higher quality: 11.0 mean SHORT trades vs 33.2 LONG trades
- **72h cooldown** appeared in top 3 configs (consistent with long-only findings)
- **conf_high_threshold=0.75** dominated top 10 — lower than Pass 1's 0.80, more signal frequency
- Best per ticker: XRP Sharpe 1.680, SOL 1.594, BTC 1.422, ETH 1.193
- **Conclusion**: Asymmetric SHORT params modestly improved multi-dir: doubled all-windows-positive configs (2→4), improved SHORT WR (41%→54%), improved mean Sharpe (-1.55→-1.27). `confirmations_short=8` is the single most impactful new param. However, long-only (Sharpe 3.512) still significantly outperforms the best consistent genuine multi-dir config (Sharpe 1.409).
- Results saved in optimization_wf_multidir_results.csv, heatmap in optimization_wf_multidir_heatmap.html

### Ensemble Walk-Forward Pass — 200 trials (2026-03-19)
Grid: ticker (BTC/ETH/XRP/SOL) × timeframe (2h/3h/4h) × n_states × feature_set (base/extended/extended_v2/full) × confirmations × cov_type
Walk-forward: 6m train / 3m test rolling windows, ensemble HMM (3 models, n_states=[N-1, N, N+1])
- 200/200 completed, 58 positive Sharpe (29%), **10 all-windows-positive configs**
- **Best overall**: ETH/3h/extended_v2/7cf/1.0x/72h/full → Sharpe +3.703 (but only 1/2 windows positive)
- **Best consistent (all windows positive)**:
  - SOL/3h/extended/7cf/diag → Sharpe +3.324, return +33.4%, std 1.559
  - ETH/4h/full/6cf/diag → Sharpe +3.320, return +48.7%, std 2.364
  - ETH/3h/extended_v2/6cf/full → Sharpe +2.645, return +14.0%, std 1.407
  - XRP/3h/base/6cf/diag → Sharpe +2.526, return +32.2%, std 0.344 (most consistent!)
  - ETH/4h/extended_v2/7cf/diag → Sharpe +2.139, return +22.9%, std 0.655
- Key findings:
  - **ETH dominates ensemble**: 6 of 10 all-windows-positive configs are ETH
  - **3h timeframe strong**: appeared in 4 of top 5 consistent configs
  - **extended_v2 validated**: appears in 3 of top 10 configs, confirming Sprint 3 feature improvements
  - **SOL current config** (4h/extended_v2/7cf): Sharpe +1.718 — decent but SOL/3h/extended/7cf outperforms at +3.324
  - **Actionable**: Consider switching AGATE to ETH/4h/extended_v2/7cf/diag (Sharpe 2.139, both windows positive, low std 0.655)
- Results saved in optimization_wf_ensemble_results.csv