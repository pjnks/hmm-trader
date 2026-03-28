# BERYL Optimization Log

## Day 1 (3/14)
**Hypothesis Tested:** Baseline NDX100 (200 trials, 5 tickers, daily bars)
**Results:**
- Trials: 195/200 (97% complete)
- Positive Sharpe: 101 (52%)
- Sharpe > 0.5: 19 configs
- Sharpe > 1.0: 0 configs (none yet)
- Best: TSLA 4st/base/7cf/1.0x/48h (Sharpe 0.921, +47.4%)
- Runner-up: NVDA 6st/extended/6cf/1.5x/72h (Sharpe 0.899, +159.3%)
- Pattern: Higher volatility tickers (TSLA, NVDA) fit HMM regimes better than stable ones (AAPL, MSFT)
**Action:** Consider adding 2h/4h intraday timeframes; test more volatile tickers
**Status:** Approaching target (need Sharpe > 1.0 for deeper analysis)
