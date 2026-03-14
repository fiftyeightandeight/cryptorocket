#!/usr/bin/env python3
"""Live trading loop for crypto strategies."""

import argparse
import importlib
import logging
import sys
import time
from datetime import datetime, timedelta, timezone

from data.ingest import ingest_candles, ingest_funding
from execution.blotter import Blotter
from execution.executor import HyperliquidExecutor
from execution.order_manager import (
    compute_position_deltas,
    create_carry_orders,
    stubs_to_hyperliquid_orders,
)
from scripts.run_backtest import STRATEGIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_strategy(name: str):
    """Load a strategy class by name or module:ClassName path."""
    if name in STRATEGIES:
        path = STRATEGIES[name]
    else:
        path = name

    if ":" not in path:
        print(f"Unknown strategy: {name}")
        sys.exit(1)

    module_path, class_name = path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def run_once(strategy, executor, blotter, dry_run=False):
    """Execute one trading cycle."""
    logger.info(f"=== Trading cycle: {strategy.CODE} at {datetime.now(timezone.utc)} ===")

    # 1. Refresh data
    logger.info("Collecting latest candle data...")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=3)
    symbols = strategy._resolve_symbols()
    try:
        ingest_candles(
            symbols=symbols,
            intervals=[strategy.INTERVAL],
            start_date=start,
            end_date=end,
            db_path=strategy.DB_PATH,
        )
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        return

    # 1b. Refresh funding data if strategy uses it
    is_carry = hasattr(strategy, "FUNDING_ENTRY_THRESHOLD")
    if is_carry:
        logger.info("Collecting funding rate data...")
        try:
            ingest_funding(
                symbols=symbols,
                start_date=start,
                end_date=end,
                db_path=strategy.DB_PATH,
            )
        except Exception as e:
            logger.error(f"Funding data collection failed: {e}")

    # 2. Generate target weights from strategy
    logger.info("Running strategy signal pipeline...")
    order_stubs = strategy.trade()

    if order_stubs.empty:
        logger.info("No trades to execute")
        return

    logger.info(f"Generated {len(order_stubs)} order stubs")
    logger.info(f"\n{order_stubs.to_string()}")

    if dry_run:
        logger.info("DRY RUN — not submitting orders")
        return

    # 3. Get account state
    account_value = executor.get_account_value()
    current_prices = executor.get_current_prices()
    sz_decimals = executor.get_sz_decimals(include_spot=is_carry)

    logger.info(f"Account value: ${account_value:,.2f}")

    # 4. Convert to exchange orders
    if is_carry:
        # Carry strategy: create paired spot+perp orders
        orders = []
        for _, stub in order_stubs.iterrows():
            symbol = stub["Symbol"]
            action = stub["Action"]
            weight = stub["Weight"]
            price = current_prices.get(symbol, 0)
            if price <= 0:
                continue

            notional = account_value * weight
            size = notional / price
            spot_name = executor.get_spot_name(symbol)
            spot_px = current_prices.get(spot_name, price)

            is_open = action == "BUY"  # BUY = opening long perp carry
            pair = create_carry_orders(
                base_symbol=symbol,
                spot_symbol=spot_name,
                size=size,
                spot_price=spot_px,
                perp_price=price,
                sz_decimals=sz_decimals.get(symbol, 2),
                is_open=is_open,
            )
            orders.extend(pair)
    else:
        orders = stubs_to_hyperliquid_orders(
            order_stubs, account_value, current_prices, sz_decimals
        )

    if not orders:
        logger.info("No valid orders after conversion")
        return

    # 5. Record orders in blotter
    order_ids = []
    for order in orders:
        oid = blotter.record_order(
            strategy=strategy.CODE,
            symbol=order["coin"],
            side="buy" if order["is_buy"] else "sell",
            size=order["sz"],
            order_type="market",
            limit_price=order["limit_px"],
        )
        order_ids.append(oid)

    # 6. Submit to exchange
    logger.info(f"Submitting {len(orders)} orders to Hyperliquid...")
    results = executor.execute_orders(orders)

    # 7. Process results
    for oid, order, result in zip(order_ids, orders, results):
        if isinstance(result, dict) and result.get("status") == "ok":
            # Record fill
            fills = result.get("response", {}).get("data", {}).get("statuses", [])
            for fill in fills:
                if "filled" in fill:
                    blotter.record_fill(
                        order_id=oid,
                        fill_price=float(fill["filled"]["avgPx"]),
                        fill_size=float(fill["filled"]["totalSz"]),
                    )
        else:
            logger.warning(f"Order {oid} may have failed: {result}")

    # 8. Log positions
    positions = executor.get_positions()
    if positions:
        logger.info("Current positions:")
        for p in positions:
            logger.info(
                f"  {p['symbol']}: size={p['size']} entry={p['entry_price']:.2f} "
                f"uPnL={p['unrealized_pnl']:.2f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Live crypto trading")
    parser.add_argument("strategy", help="Strategy name or module:ClassName")
    parser.add_argument("--dry-run", action="store_true", help="Generate orders without submitting")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval-minutes", type=int, default=60, help="Loop interval in minutes")
    parser.add_argument("--testnet", action="store_true", help="Use testnet")
    parser.add_argument("--db-path", help="Custom database path")
    args = parser.parse_args()

    strategy = load_strategy(args.strategy)
    if args.db_path:
        strategy.DB_PATH = args.db_path

    executor = HyperliquidExecutor(testnet=args.testnet)
    blotter = Blotter(db_path=args.db_path)

    if args.loop:
        logger.info(f"Starting trading loop (interval: {args.interval_minutes}m)")
        while True:
            try:
                run_once(strategy, executor, blotter, dry_run=args.dry_run)
            except Exception as e:
                logger.error(f"Trading cycle error: {e}", exc_info=True)

            logger.info(f"Sleeping {args.interval_minutes} minutes...")
            time.sleep(args.interval_minutes * 60)
    else:
        run_once(strategy, executor, blotter, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
