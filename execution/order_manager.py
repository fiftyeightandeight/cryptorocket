import logging
import math
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def round_size(size: float, sz_decimals: int) -> float:
    """Round order size to valid step size for a symbol."""
    factor = 10 ** sz_decimals
    return math.floor(abs(size) * factor) / factor


def round_price(price: float, is_spot: bool = False) -> float:
    """Round price to 5 significant figures (Hyperliquid requirement).

    Perps: 6 decimal places max. Spot: 8 decimal places max.
    """
    if price <= 0:
        return 0.0
    max_decimals = 8 if is_spot else 6
    rounded = float(f"{price:.5g}")
    return round(rounded, max_decimals)


def stubs_to_hyperliquid_orders(
    order_stubs: pd.DataFrame,
    account_value: float,
    current_prices: dict[str, float],
    sz_decimals: dict[str, int],
) -> list[dict]:
    """Convert strategy order stubs to Hyperliquid SDK order format.

    Args:
        order_stubs: DataFrame with columns [Symbol, Weight, Action, Strategy, order_type, tif]
        account_value: Current account value in USD
        current_prices: Dict of symbol -> current price
        sz_decimals: Dict of symbol -> size decimal places

    Returns:
        List of dicts ready for Exchange.order() or Exchange.bulk_orders()
    """
    orders = []

    for _, stub in order_stubs.iterrows():
        symbol = stub["Symbol"]
        action = stub["Action"]
        weight = stub["Weight"]
        order_type = stub.get("order_type", "market")

        price = current_prices.get(symbol)
        if price is None or price <= 0:
            logger.warning(f"No price for {symbol}, skipping")
            continue

        # Convert weight to notional, then to size
        notional = account_value * weight
        raw_size = notional / price
        decimals = sz_decimals.get(symbol, 2)
        size = round_size(raw_size, decimals)

        if size <= 0:
            logger.debug(f"{symbol}: size rounds to 0, skipping")
            continue

        is_buy = action == "BUY"

        if order_type == "market":
            # Market order = aggressive limit IoC with slippage
            slippage = 0.05  # 5% max slippage
            limit_px = price * (1 + slippage) if is_buy else price * (1 - slippage)
            limit_px = round_price(limit_px)
            order = {
                "coin": symbol,
                "is_buy": is_buy,
                "sz": size,
                "limit_px": limit_px,
                "order_type": {"limit": {"tif": "Ioc"}},
                "reduce_only": False,
            }
        elif order_type == "limit":
            limit_px = round_price(stub.get("limit_price", price))
            tif = stub.get("tif", "Gtc")
            order = {
                "coin": symbol,
                "is_buy": is_buy,
                "sz": size,
                "limit_px": limit_px,
                "order_type": {"limit": {"tif": tif}},
                "reduce_only": False,
            }
        else:
            logger.warning(f"Unknown order type: {order_type}")
            continue

        orders.append(order)
        logger.info(
            f"Order: {'BUY' if is_buy else 'SELL'} {size} {symbol} "
            f"@ {order['limit_px']} ({order_type})"
        )

    return orders


def compute_position_deltas(
    target_weights: dict[str, float],
    current_positions: dict[str, float],
    account_value: float,
    current_prices: dict[str, float],
    sz_decimals: dict[str, int],
    min_notional: float = 10.0,
) -> pd.DataFrame:
    """Compute the delta between target and current positions.

    Args:
        target_weights: Dict of symbol -> target weight
        current_positions: Dict of symbol -> current size (signed)
        account_value: Current account value
        current_prices: Dict of symbol -> price
        sz_decimals: Dict of symbol -> size decimals
        min_notional: Minimum order notional to bother with

    Returns:
        DataFrame with order stubs (Symbol, Weight, Action, Strategy)
    """
    all_symbols = set(target_weights.keys()) | set(current_positions.keys())
    stubs = []

    for symbol in all_symbols:
        target_w = target_weights.get(symbol, 0)
        price = current_prices.get(symbol)
        if price is None or price <= 0:
            continue

        # Target size from weight
        target_notional = account_value * target_w
        target_size = target_notional / price

        # Current size
        current_size = current_positions.get(symbol, 0)

        # Delta
        delta_size = target_size - current_size
        delta_notional = abs(delta_size * price)

        if delta_notional < min_notional:
            continue

        decimals = sz_decimals.get(symbol, 2)
        rounded_delta = round_size(delta_size, decimals)
        if rounded_delta <= 0:
            continue

        stubs.append({
            "Symbol": symbol,
            "Weight": abs(target_w - (current_size * price / account_value)),
            "Action": "BUY" if delta_size > 0 else "SELL",
        })

    return pd.DataFrame(stubs) if stubs else pd.DataFrame(columns=["Symbol", "Weight", "Action"])


def create_carry_orders(
    base_symbol: str,
    spot_symbol: str,
    size: float,
    spot_price: float,
    perp_price: float,
    sz_decimals: int,
    is_open: bool = True,
    slippage: float = 0.05,
) -> list[dict]:
    """Create paired spot+perp orders for a carry trade.

    Opening: buy spot + sell perp (delta-neutral, collect funding)
    Closing: sell spot + buy perp (unwind)

    Args:
        base_symbol: Perp symbol (e.g., 'BTC')
        spot_symbol: Spot pair (e.g., 'BTC/USDC')
        size: Position size in base units
        spot_price: Current spot mid price
        perp_price: Current perp mid price
        sz_decimals: Size decimal places
        is_open: True to open carry, False to unwind
        slippage: Max slippage for market orders

    Returns:
        List of two order dicts (spot leg, perp leg) for bulk_orders()
    """
    rounded_size = round_size(size, sz_decimals)
    if rounded_size <= 0:
        return []

    if is_open:
        # Open: buy spot, sell perp
        spot_buy = True
        perp_buy = False
    else:
        # Close: sell spot, buy perp
        spot_buy = False
        perp_buy = True

    spot_limit = spot_price * (1 + slippage) if spot_buy else spot_price * (1 - slippage)
    perp_limit = perp_price * (1 + slippage) if perp_buy else perp_price * (1 - slippage)

    return [
        {
            "coin": spot_symbol,
            "is_buy": spot_buy,
            "sz": rounded_size,
            "limit_px": round_price(spot_limit, is_spot=True),
            "order_type": {"limit": {"tif": "Ioc"}},
            "reduce_only": False,
        },
        {
            "coin": base_symbol,
            "is_buy": perp_buy,
            "sz": rounded_size,
            "limit_px": round_price(perp_limit, is_spot=False),
            "order_type": {"limit": {"tif": "Ioc"}},
            "reduce_only": not is_open,
        },
    ]
