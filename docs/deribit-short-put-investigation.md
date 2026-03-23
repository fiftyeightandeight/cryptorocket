# Deribit Short BTC Put Strategy — Investigation

*Date: 2026-03-15*

## Objective

Evaluate feasibility of implementing a short BTC put strategy on Deribit, focusing on data availability, backtest infrastructure requirements, and integration path into the existing cryptorocket framework.

## Strategy Overview

Sell BTC put options on Deribit to collect premium. The seller profits when BTC stays above the strike price at expiry (option expires worthless, premium kept). Risk is BTC falling below the strike, where the loss is `strike - settlement_price + premium_collected`.

Typical parameterization:
- **Strike selection**: sell X-delta puts (e.g., 20-delta ≈ 80% probability of expiring worthless)
- **Expiry selection**: 7–30 DTE (days to expiry), balancing theta decay vs. gamma risk
- **Position sizing**: based on max acceptable loss at strike

## Deribit API Capabilities

Deribit provides a JSON-RPC API over HTTP and WebSocket. No authentication is required for market data endpoints.

### Endpoints Relevant to Options Backtesting

| Endpoint | Description | Auth | Historical Depth |
|----------|-------------|------|-----------------|
| `public/get_last_settlements_by_currency` | All settlement/delivery events for BTC options. Includes instrument name, index price, mark price, delivery price. Paginated (max 1000 per page). | Public | Back to exchange launch (~2019) |
| `public/get_delivery_prices` | Settlement prices at each expiry date for a given index. | Public | Back to exchange launch |
| `public/get_instruments` | List active instruments. `expired=true` returns recently expired (weeks, not years). | Public | Recent weeks only |
| `public/get_tradingview_chart_data` | OHLCV candles for a specific instrument by name. | Public | Unclear retention for long-expired options |
| `public/get_historical_volatility` | Realized BTC volatility. | Public | Full history |
| `public/get_volatility_index_data` | DVOL (Deribit Volatility Index, analogous to VIX). | Public | Since ~2021 |
| `public/ticker` | Current bid/ask, mark price, Greeks (delta, gamma, theta, vega, rho), mark IV. | Public | Current snapshot only |

### Critical Gap

The Deribit API does **not** provide historical snapshots of option pricing or Greeks. There is no way to query "what was the bid/ask and delta of BTC-28MAR25-80000-P on January 15?" — only current-state and settlement records are available.

This means:
- We **can** know the outcome of every historical option expiry (from settlements)
- We **cannot** know the exact price at which we would have entered a trade historically

### Comparison to Hyperliquid

| Capability | Hyperliquid | Deribit |
|------------|-------------|---------|
| OHLCV candles | `candleSnapshot` | `public/get_tradingview_chart_data` |
| Resolutions | 1m, 5m, 15m, 1h, 4h, 1d | 1, 3, 5, 10, 15, 30, 60, 120, 180, 360, 720 min, 1D |
| Historical depth | ~2 years | Since 2019 (for settlements) |
| Instrument types | Perps, spot | **Options**, futures, perps |
| Funding rates | Yes | Yes (perps only) |
| Greeks | N/A | Yes (real-time via ticker) |
| Implied volatility | N/A | Bid IV, ask IV, mark IV |
| Historical volatility | N/A | `public/get_historical_volatility` |
| DVOL index | N/A | `public/get_volatility_index_data` |
| Auth model | Ethereum wallet signing | OAuth 2.0 / API key + secret |
| Protocols | REST, WebSocket | JSON-RPC (HTTP, WS), FIX |
| Testnet | Yes | Yes (`test.deribit.com`) |
| Python SDK | `hyperliquid-python-sdk` (official) | `deribit-trading-sdk` (community, async) |

## Third-Party Data: Tardis.dev

[Tardis.dev](https://tardis.dev/) captures tick-level Deribit data since **2019-03-30**, filling the historical pricing gap.

### Relevant Channels

| Channel | Available Since | Content |
|---------|----------------|---------|
| `options_chain` | 2020-03-01 | Full options chain snapshots |
| `markprice.options` | 2019-10-01 | Mark price + IV for all options |
| `ticker` | 2019-10-01 | Full ticker including all Greeks |
| `trades` | 2019-03-30 | Every option trade execution |
| `deribit_volatility_index` | 2021-04-01 | DVOL updates |

### Access

- **Free**: CSV downloads for the 1st day of each month (no API key required)
- **Paid**: Full historical access via subscription (Solo, Pro, Business tiers)
- **Format**: Exchange-native WebSocket message format with local timestamps

## Proposed Approach

### Phase 1: Settlement-Based Backtest (Free)

Use Deribit's settlement API to reconstruct every historical option expiry, then model entry premiums with Black-Scholes using historical volatility.

**Data collection**:
1. Paginate through `public/get_last_settlements_by_currency` for all BTC option deliveries
2. Parse instrument names (e.g., `BTC-28MAR25-80000-P`) into structured fields (expiry, strike, type)
3. Pull `public/get_delivery_prices` for BTC settlement prices
4. Pull `public/get_historical_volatility` for realized vol

**Backtest logic**:
1. For each expiry cycle, identify the put at the desired delta (approximate via Black-Scholes using realized vol as IV proxy)
2. Model entry premium at listing date using Black-Scholes
3. Calculate P&L at settlement: `max(premium - max(strike - delivery_price, 0), -max_loss)`

**Limitations**: Entry premium is modeled, not actual market price. Ignores bid/ask spread, liquidity, and the difference between realized and implied volatility.

### Phase 2: Validate with Tardis Free Samples

Download free Tardis CSV samples (1st of each month) containing actual `options_chain` or `ticker` data. Compare model-derived premiums from Phase 1 against real bid/ask prices to quantify the model's accuracy.

### Phase 3: Full Tick-Data Backtest (If Promising)

If Phase 1+2 show the strategy concept is viable, either:
- Subscribe to Tardis for full historical options chain data, or
- Build a forward-looking live data collector using Deribit WebSocket to accumulate our own dataset

## Proposed DuckDB Schema

```sql
-- Parsed metadata for each option instrument
CREATE TABLE options_instruments (
    instrument_name VARCHAR PRIMARY KEY,
    underlying VARCHAR,         -- 'BTC'
    expiry_date DATE,
    strike DOUBLE,
    option_type VARCHAR         -- 'P' or 'C'
);

-- Settlement outcomes (from Deribit settlement API)
CREATE TABLE options_settlements (
    instrument_name VARCHAR,
    timestamp BIGINT,
    settlement_type VARCHAR,    -- 'delivery' | 'settlement'
    index_price DOUBLE,         -- BTC price at settlement
    mark_price DOUBLE,
    delivery_price DOUBLE,
    PRIMARY KEY (instrument_name, timestamp)
);

-- BTC delivery prices by expiry date
CREATE TABLE delivery_prices (
    timestamp BIGINT PRIMARY KEY,
    index_name VARCHAR,         -- 'btc_usd'
    delivery_price DOUBLE
);

-- Realized volatility (from Deribit API)
CREATE TABLE historical_volatility (
    timestamp BIGINT PRIMARY KEY,
    currency VARCHAR,
    volatility DOUBLE
);

-- Tick-level snapshots (from Tardis or live collector, Phase 3)
CREATE TABLE options_snapshots (
    timestamp BIGINT,
    instrument_name VARCHAR,
    bid DOUBLE,
    ask DOUBLE,
    mark_price DOUBLE,
    mark_iv DOUBLE,
    underlying_price DOUBLE,
    delta DOUBLE,
    gamma DOUBLE,
    theta DOUBLE,
    vega DOUBLE,
    PRIMARY KEY (timestamp, instrument_name)
);
```

## Architecture Considerations

### Options vs. Perpetuals Pipeline

The existing Moonshot-style pipeline (`prices_to_signals → weights → positions → returns`) assumes continuous position holding with daily rebalancing. Options strategies are fundamentally different:

1. **Instrument selection** — pick expiry and strike (e.g., 30-delta put, 30 DTE)
2. **Entry** — sell at bid (collect premium)
3. **Hold or manage** — optionally roll or close before expiry
4. **Settlement** — at expiry, worthless (profit = premium) or assigned (loss = strike - settlement + premium)

This is an **event-driven cycle**, not a continuous weight-based rebalance. A separate `OptionsStrategy` base class is recommended rather than forcing options into `CryptoMoonshot`.

### Implemented Module Structure

```
data/
  deribit_client.py           Deribit JSON-RPC API client (rate-limited, with retry)
  deribit_ingest.py           Snapshot, DVOL, realized vol, settlement ingestion
  schema.py                   DuckDB schema (extended with 6 options tables)
scripts/
  collect_deribit.py          CLI for hourly cron and one-time backfills
backtest/
  options_estimators.py       SpreadEstimator + IVPremiumEstimator (fit/estimate/save/load)
.github/workflows/
  collect-deribit-data.yml    Hourly GitHub Actions workflow
tests/
  test_options_estimators.py  Unit tests for client, estimators, serialization
```

### Future Modules (Not Yet Implemented)

```
strategies/
  options_base.py             OptionsStrategy base class (event-driven)
  options/
    short_put.py              Short put strategy implementation
backtest/
  options_engine.py           Options-specific backtest engine
```

## Additional Endpoints Investigated

| Endpoint | Description | Historical Retention |
|----------|-------------|---------------------|
| `public/get_last_trades_by_currency_and_time` | Tick-level trade executions filtered by time range. | Available for active instruments; expired instruments are purged after a few weeks. |
| `public/get_mark_price_history` | Mark price history for a specific instrument. | Available for active instruments only; historical data for expired options is not retained. |
| `public/get_book_summary_by_currency` | Snapshot of all live instruments: bid, ask, mark_price, mark_iv, open_interest. | Current state only — no historical lookback. |
| `public/ticker` | Full ticker with Greeks (delta, gamma, theta, vega, rho), bid_iv, ask_iv. | Current state only. |

**Key finding**: While these endpoints provide rich real-time data for *live* instruments, Deribit does not retain historical pricing, IV, or Greek data for expired options. This confirms the need for estimator models calibrated from accumulated live snapshots.

## Estimator Design

### Problem

The Phase 1 settlement-based backtest uses Black-Scholes with realized volatility as the IV proxy. This has two systematic biases:

1. **No bid/ask spread** — real entry occurs at the bid (for selling puts), not at mid or theoretical price
2. **IV ≠ realized vol** — implied volatility consistently trades at a premium over realized, varying by moneyness, DTE, and vol regime

### Solution: Calibrated Estimators

We accumulate hourly live options chain snapshots via the Deribit API (automated by GitHub Actions). From these snapshots, two estimators are fitted:

**SpreadEstimator** — learns `spread_pct = (ask - bid) / mark_price` bucketed by:
- |delta|: [0, 0.10), [0.10, 0.25), [0.25, 0.40), [0.40, 0.50]
- DTE: [1, 7), [7, 14), [14, 30), [30, 90), [90, ∞)
- DVOL regime: [0, 40), [40, 60), [60, 80), [80, ∞)

Each bucket stores the median spread_pct. Unknown buckets fall back to the global median.

**IVPremiumEstimator** — learns `iv_premium = mark_iv - realized_vol` with the same bucket structure. At estimation time: `sigma_implied = realized_vol + iv_premium(bucket)`.

### Integration with Phase 1 Backtest

```python
sigma = iv_est.estimate(delta=0.20, dte=30, dvol=dvol_at_entry, realized_vol=rv_at_entry)
bs_price = black_scholes_put(S=btc_price, K=strike, T=dte/365, r=0, sigma=sigma)
entry_premium = bs_price - spread_est.estimate(delta=0.20, dte=30, dvol=dvol_at_entry) / 2
```

This replaces the naive `sigma = realized_vol` assumption and accounts for the bid-side execution discount.

### Assumption

The *structure* of spreads and IV premiums (how they vary by delta, DTE, and vol regime) is more stable over time than their absolute levels. Calibrating from recent live data and applying retroactively is a reasonable first-order approximation.

### Data Collection

- **Hourly snapshots**: `scripts/collect_deribit.py` runs every hour via `.github/workflows/collect-deribit-data.yml`
- **One-time backfills**: DVOL, realized vol, settlements, and delivery prices
- **Storage**: Same DuckDB file (`data/crypto.duckdb`) with new tables: `options_snapshots`, `dvol`, `realized_volatility`, `options_settlements`, `options_instruments`, `delivery_prices`

## Open Questions

1. **How far back does `get_tradingview_chart_data` work for expired options?** Needs empirical testing — if it returns OHLCV for options expired years ago, we get free historical premium data.
2. **Tardis.dev pricing for Derivatives tier?** Exact cost not publicly listed; need to request a quote.
3. **Roll/management rules**: Should the strategy hold to expiry only, or include early exit rules (e.g., close at 50% profit, roll if delta exceeds threshold)?
4. **Margin requirements**: Deribit uses a portfolio margin system. Backtesting needs to model margin to be realistic about capital efficiency.

## References

- [Deribit API Documentation](https://docs.deribit.com/)
- [Deribit Settlement Endpoint](https://docs.deribit.com/api-reference/market-data/public-get_last_settlements_by_currency)
- [Tardis.dev Deribit Data](https://docs.tardis.dev/historical-data-details/deribit)
- [Tardis.dev Free Samples](https://docs.tardis.dev/downloadable-csv-files)
- [deribit-trading-sdk (PyPI)](https://pypi.org/project/deribit-trading-sdk/)
- [schepal/deribit_data_collector (GitHub)](https://github.com/schepal/deribit_data_collector)
