# ML Roadmap — Accelerated Timeline

*Generated 2026-03-25 based on tonight's DIAMOND ML scorer + CITRINE meta-model results.*

## What We Proved Tonight

1. **Lasso beats GBM at N=192** — simpler models win on small data (DIAMOND: AUC 0.827, Brier 0.173)
2. **Entry price is the most predictive feature** — market-implied probability matters more than any anomaly signal
3. **Null importance testing killed 11 of 15 features** — most hand-tuned features are noise
4. **Sector-based config transfer works** — CITRINE meta-model found +0.461 avg Sharpe improvement across 37 tickers
5. **extended_v2 is massively under-deployed** — only 7% of CITRINE configs use it despite 3x better mean Sharpe in sectors where tested

---

## Accelerated 6-Week Plan

### Week 1 (Now): DIAMOND A/B + CITRINE Config Upgrade

| Task | Status | Impact |
|------|--------|--------|
| DIAMOND ML scorer deployed (A/B logging) | DONE | Baseline for edge-based trading |
| CITRINE meta-model analysis | DONE | 37 config changes identified |
| Apply CITRINE v2 configs to live | TODO | +0.461 avg Sharpe per changed ticker |
| Run extended_v2 optimizer on 20 priority tickers | TODO | Fill 53-ticker gap |

**How to apply v2 configs:**
```bash
# Review changes
python citrine_meta_model.py --recommend
# Swap configs (hot-swappable, no restart needed)
cp citrine_per_ticker_configs.json citrine_per_ticker_configs_backup.json
cp citrine_per_ticker_configs_v2.json citrine_per_ticker_configs.json
scp -i ~/.ssh/hmm-trader.key citrine_per_ticker_configs.json ubuntu@129.158.40.51:/home/ubuntu/HMM-Trader/
```

### Week 2: DIAMOND ML Activation + CITRINE Optimizer Integration

| Task | Impact |
|------|--------|
| Review DIAMOND A/B data (2 weeks) | Validate ML edge on out-of-sample |
| If ML wins: activate `ML_SCORER_ACTIVE=true` | +$10/week projected (from -$3.73 to +$6.26) |
| Integrate meta-model into Sunday CITRINE optimizer | Auto-recommend configs after each run |
| Add extended_v2 to CITRINE optimizer grid | Currently only 56/734 trials use it |

### Week 3-4: Cross-Project Intelligence

| Task | Impact |
|------|--------|
| **BERYL meta-model** (same approach as CITRINE) | 100 trials, 58 tickers — identify winning configs |
| **AGATE extended_v2 rollout** (14 crypto tickers) | BCH already at +3.52, test extended_v2 on rest |
| **DIAMOND → CITRINE cross-signal** | Map Kalshi event tickers to equity tickers (Phase 1 of cross-system roadmap) |
| **Conviction half-life optimization** | Use 7,154 conviction signals to find optimal decay rate |

### Week 5-6: Advanced ML (If Data Supports)

| Task | Prereq | Impact |
|------|--------|--------|
| **DIAMOND: GBM tier unlock** | N > 300 settled trades | Beat Lasso if nonlinear patterns emerge |
| **CITRINE: Volatility-aware confirmations** | Meta-model + 20 extended_v2 trials | 6cf for high-vol, 7-8cf for low-vol |
| **CITRINE: Config ensemble (top-3 voting)** | v2 configs validated | Lower drawdown via multi-config agreement |
| **All: Automated feature importance dashboard** | All models deployed | Track which features matter over time |

### Month 2+: Neural Network Tier (When Data Allows)

| Task | Prereq | Architecture |
|------|--------|-------------|
| **DIAMOND: LSTM anomaly sequence model** | N > 500 trades, 6+ weeks of ml_edge logs | Sequence of anomaly features → trade outcome |
| **CITRINE: Attention-based multi-ticker model** | 3+ months daily regime data | Cross-ticker regime correlations |
| **CITRINE: RL allocation** | 6+ months of daily snapshots | State = regime vector, action = position weights |
| **AGATE: LSTM regime detector** | 12+ months of 4h bars per ticker | Replace HMM with learned regime boundaries |

---

## Data Accumulation Tracker

Track when each project hits ML-viable sample sizes:

| Project | Current Labeled Data | Next Milestone | ETA |
|---------|---------------------|---------------|-----|
| **DIAMOND** | 192 settled trades | 300 (GBM tier) | ~2-3 weeks |
| **DIAMOND** | 137K anomalies | Already sufficient for unsupervised | Now |
| **CITRINE** | 47 trades, 6 snapshots | 200 trades | ~3-4 months |
| **BERYL** | 1 trade | 50 trades | ~2-3 months |
| **AGATE** | 0 trades (test mode) | 50 trades | ~3-4 months |

**DIAMOND is the only project with enough labeled data for supervised ML right now.** CITRINE/BERYL/AGATE need months of live trading before ML becomes viable for those systems.

---

## Key Principle: Earn Your Complexity

From tonight's results, the hierarchy is clear:

1. **Hand-tuned heuristics** → baseline (current state for all projects)
2. **Lasso/linear models** → first ML tier (DIAMOND proven: AUC 0.827)
3. **Gradient boosting** → only if it beats Lasso by >0.005 Brier (DIAMOND: GBM failed this test)
4. **Neural networks** → only when N > 500 and GBM is the incumbent
5. **Reinforcement learning** → only for allocation/sizing decisions with 6+ months of daily data

Never skip a tier. If Lasso can't beat hand-tuned, no point trying GBM. If GBM can't beat Lasso, no point trying NN. Each tier must earn its complexity.

---

## Files Created/Modified Tonight

| File | Project | Purpose |
|------|---------|---------|
| `src/diamond_ml.py` | DIAMOND | Tiered ML scorer (Lasso/GBM) |
| `diamond_ml_train.py` | DIAMOND | Training CLI |
| `diamond_config.py` | DIAMOND | ML config toggles |
| `src/diamond_features.py` | DIAMOND | ML edge injection |
| `diamond_monitor.py` | DIAMOND | Daily retrain loop |
| `citrine_meta_model.py` | CITRINE | Meta-learning analysis |
| `citrine_per_ticker_configs_v2.json` | CITRINE | 37 improved configs |
| `ML_ROADMAP.md` | All | This file |
