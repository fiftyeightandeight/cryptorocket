# Cryptorocket

Crypto trading system targeting Hyperliquid DEX. Mirrors the Moonshot signal pipeline for backtesting and live execution on perpetual and spot markets.

## Architecture

```
config.py                  Environment / wallet / DB config
data/
  client.py                Hyperliquid Info API client (rate-limited)
  schema.py                DuckDB schema and connection helpers
  ingest.py                Historical + incremental candle ingestion
  store.py                 get_prices() — Moonshot-compatible data access
backtest/
  engine.py                Backtest runner
  commission.py            Hyperliquid fee models (perps / spot)
  tearsheet.py             Performance metrics and equity curve
execution/
  order_manager.py         Order stub → Hyperliquid order conversion
  blotter.py               Local position / fill tracking
  executor.py              Live order submission via SDK
strategies/
  base.py                  CryptoMoonshot base class
  examples/
    momentum.py            Cross-sectional momentum + mean reversion
    carry.py               Funding rate carry
    intraday.py            Session-based intraday (US / Asia / EU)
scripts/
  backfill_data.py         One-time historical data backfill
  collect_data.py          Incremental collection (cron)
  gh_collect.py            GitHub Actions data collector
  run_backtest.py          CLI backtest runner
  run_live.py              Live trading loop
```

## Strategy Pipeline

Same 5-step chain as Moonshot:

```
get_prices() → prices_to_signals() → signals_to_target_weights()
→ target_weights_to_positions() → positions_to_gross_returns()
```

Subclass `CryptoMoonshot` and override `prices_to_signals()` at minimum. See `strategies/examples/` for working examples.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in wallet credentials

# Backfill 2 years of candle data
python -m scripts.backfill_data

# Run a backtest
python -m scripts.run_backtest crypto-momentum --plot

# Live trading (testnet)
HYPERLIQUID_TESTNET=true python -m scripts.run_live crypto-momentum
```

## Available Strategies

| Code | Description |
|------|-------------|
| `crypto-momentum` | Long top-N by 20d trailing return |
| `crypto-mean-reversion` | Z-score mean reversion (long oversold, short overbought) |
| `funding-carry` | Long negative funding, short positive funding |
| `intraday-momentum-us` | Hourly momentum during US session (14-22 UTC) |
| `intraday-mr-us` | Hourly mean reversion during US session |
| `intraday-momentum-asia` | Hourly momentum during Asia session (00-08 UTC) |
| `intraday-momentum-eu` | Hourly momentum during Europe session (08-16 UTC) |

## CI / Automation

- **Data collection** (`.github/workflows/collect-crypto-data.yml`): Runs hourly, ingests latest candles for top perps, commits DuckDB via LFS.
- **Backtest** (`.github/workflows/run-backtest.yml`): Manual dispatch, runs any registered strategy, uploads equity curve artifact.

## Dependencies

- `hyperliquid-python-sdk` — API client and exchange SDK
- `eth_account` — Ethereum wallet signing
- `duckdb` — Embedded columnar database
- `pandas`, `numpy`, `matplotlib` — Data analysis and plotting
