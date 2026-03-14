#!/usr/bin/env python3
"""Live trading loop for crypto strategies."""

import argparse
import importlib
import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone

from data.ingest import ingest_candles
from execution.blotter import Blotter
from execution.executor import HyperliquidExecutor
from execution.order_manager import (
    compute_position_deltas,
    stubs_to_hyperliquid_orders,
)
from scripts.run_backtest import STRATEGIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_ERRORS = 5
BACKOFF_BASE_SECONDS = 30

_shutdown_requested = False


def _handle_signal(signum, frame):
    global _shutdown_requested
    logger.info(f"Received signal {signum}, shutting down after current cycle...")
    _shutdown_requested = True


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
    sz_decimals = executor.get_sz_decimals()

    logger.info(f"Account value: ${account_value:,.2f}")

    # 4. Convert to exchange orders
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
        try:
            if not isinstance(result, dict):
                logger.warning(f"Order {oid}: unexpected result type {type(result)}: {result}")
                continue

            if result.get("status") != "ok":
                logger.warning(f"Order {oid} failed: {result}")
                continue

            statuses = (
                result.get("response", {})
                .get("data", {})
                .get("statuses", [])
            )
            if not statuses:
                logger.warning(f"Order {oid}: status ok but no fill statuses in response")
                continue

            for status_entry in statuses:
                filled = status_entry.get("filled")
                if filled and "avgPx" in filled and "totalSz" in filled:
                    blotter.record_fill(
                        order_id=oid,
                        fill_price=float(filled["avgPx"]),
                        fill_size=float(filled["totalSz"]),
                    )
                elif "error" in status_entry:
                    logger.warning(f"Order {oid}: exchange error: {status_entry['error']}")
                elif "resting" in status_entry:
                    logger.info(f"Order {oid}: resting on book (oid={status_entry['resting'].get('oid')})")
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Order {oid}: failed to parse result: {e} — raw: {result}")

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
    parser.add_argument("--max-errors", type=int, default=MAX_CONSECUTIVE_ERRORS,
                        help="Max consecutive errors before stopping")
    parser.add_argument("--testnet", action="store_true", help="Use testnet")
    parser.add_argument("--db-path", help="Custom database path")
    args = parser.parse_args()

    strategy = load_strategy(args.strategy)
    if args.db_path:
        strategy.DB_PATH = args.db_path

    executor = HyperliquidExecutor(testnet=args.testnet)
    blotter = Blotter(db_path=args.db_path)

    if args.loop:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        logger.info(f"Starting trading loop (interval: {args.interval_minutes}m, "
                     f"max_errors: {args.max_errors})")
        consecutive_errors = 0

        while not _shutdown_requested:
            try:
                run_once(strategy, executor, blotter, dry_run=args.dry_run)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Trading cycle error ({consecutive_errors}/{args.max_errors}): {e}",
                             exc_info=True)

                if consecutive_errors >= args.max_errors:
                    logger.critical(
                        f"Hit {consecutive_errors} consecutive errors — stopping to prevent damage"
                    )
                    sys.exit(1)

                backoff = min(BACKOFF_BASE_SECONDS * (2 ** (consecutive_errors - 1)), 600)
                logger.info(f"Backing off {backoff}s before retry...")
                time.sleep(backoff)
                continue

            if _shutdown_requested:
                break

            logger.info(f"Sleeping {args.interval_minutes} minutes...")
            time.sleep(args.interval_minutes * 60)

        logger.info("Graceful shutdown complete")
    else:
        run_once(strategy, executor, blotter, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
