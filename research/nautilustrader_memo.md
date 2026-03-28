# NautilusTrader Evaluation Memo

## What It Is
NautilusTrader is an open-source, high-performance algorithmic trading platform written in Python/Cython. It provides event-driven backtesting with realistic orderbook simulation, multi-venue support, and latency-accurate execution modeling.

## Capabilities vs Current System

| Capability | Current (Backtester) | NautilusTrader |
|---|---|---|
| Execution model | Bar-by-bar, fixed 10bps slippage | Tick-by-tick, orderbook-realistic |
| Order types | Market only | Market, Limit, Stop, Trailing, Bracket |
| Slippage model | Fixed percentage | Modeled from L2 orderbook depth |
| Latency simulation | None | Configurable per-venue latency |
| Multi-venue | No | Yes (Coinbase, Binance, Interactive Brokers) |
| Live trading | Custom (live_broker.py) | Built-in adapters for major exchanges |
| Backtesting speed | ~1K bars/sec (Python) | ~100K+ bars/sec (Cython) |

## Integration Effort

**High.** Would require:
1. Rewriting `src/backtester.py` and `src/strategy.py` to conform to NautilusTrader's `Strategy` and `Actor` base classes
2. Converting OHLCV data to NautilusTrader's `Bar` or `QuoteTick` format
3. Implementing a custom `DataClient` for Polygon.io
4. Rewriting the HMM regime detection as a NautilusTrader `Indicator`

Estimated: 2-3 weeks of development.

## Recommendation

**Not yet.** NautilusTrader is overkill for the current testing phase:
- CITRINE and BERYL trade daily bars — orderbook simulation adds no value at daily granularity
- AGATE trades 4h bars — marginal benefit from tick-level simulation
- The current 10bps fixed slippage is a reasonable approximation for test mode

**When to revisit:**
- When moving to LIVE trading with real capital on Coinbase (AGATE)
- When testing intraday strategies (sub-1h timeframes)
- When orderbook data is available (Coinbase WebSocket provides L2 data)

**Alternative:** For AGATE crypto, the simpler fix is to track the actual bid/ask spread at signal time using Coinbase WebSocket data already available via the CDP integration. This gives realistic slippage estimation without the full NautilusTrader migration.

## References
- GitHub: https://github.com/nautechsystems/nautilus_trader
- Docs: https://nautilustrader.io/docs/
- Key dependency: Requires Rust toolchain for Cython compilation on ARM (M1 Mac)
