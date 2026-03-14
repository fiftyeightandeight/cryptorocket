#!/usr/bin/env python3
"""CLI backtest runner for crypto strategies."""

import argparse
import importlib
import logging
import sys

from backtest.engine import run_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Registry of built-in strategies (single source of truth)
STRATEGIES = {
    "crypto-momentum": "strategies.examples.momentum:CryptoMomentum",
    "crypto-mean-reversion": "strategies.examples.momentum:CryptoMeanReversion",
    "intraday-momentum-us": "strategies.examples.intraday:IntradayMomentumUS",
    "intraday-mr-us": "strategies.examples.intraday:IntradayMeanReversionUS",
    "intraday-momentum-asia": "strategies.examples.intraday:IntradayMomentumAsia",
    "intraday-momentum-eu": "strategies.examples.intraday:IntradayMomentumEurope",
    "liquidation-cascade": "strategies.examples.liquidation_cascade:LiquidationCascadeMomentum",
}


def load_strategy(name: str):
    """Load a strategy class by name or module:ClassName path."""
    if name in STRATEGIES:
        path = STRATEGIES[name]
    else:
        path = name

    if ":" not in path:
        print(f"Unknown strategy: {name}")
        print(f"Available: {', '.join(STRATEGIES.keys())}")
        print("Or specify as module.path:ClassName")
        sys.exit(1)

    module_path, class_name = path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def main():
    parser = argparse.ArgumentParser(description="Run crypto strategy backtest")
    parser.add_argument("strategy", help="Strategy name or module:ClassName")
    parser.add_argument("-s", "--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("-e", "--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--plot", action="store_true", help="Generate equity curve plot")
    parser.add_argument("-o", "--output", help="Save plot to file (PNG/PDF)")
    parser.add_argument("--db-path", help="Custom database path")
    args = parser.parse_args()

    strategy = load_strategy(args.strategy)
    if args.db_path:
        strategy.DB_PATH = args.db_path

    results = run_backtest(
        strategy,
        start_date=args.start_date,
        end_date=args.end_date,
        plot=args.plot,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
