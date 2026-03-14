import logging
from typing import Optional

import eth_account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils.constants import MAINNET_API_URL, TESTNET_API_URL

from config import (
    API_WALLET_KEY,
    BASE_URL,
    USE_TESTNET,
    WALLET_ADDRESS,
    WALLET_PRIVATE_KEY,
)

logger = logging.getLogger(__name__)


class HyperliquidExecutor:
    """Live order execution via Hyperliquid SDK."""

    def __init__(
        self,
        private_key: Optional[str] = None,
        wallet_address: Optional[str] = None,
        testnet: Optional[bool] = None,
    ):
        key = private_key or API_WALLET_KEY or WALLET_PRIVATE_KEY
        if not key:
            raise ValueError(
                "No private key provided. Set HYPERLIQUID_PRIVATE_KEY or "
                "HYPERLIQUID_API_WALLET_KEY environment variable."
            )

        wallet = eth_account.Account.from_key(key)
        address = wallet_address or WALLET_ADDRESS

        use_testnet = testnet if testnet is not None else USE_TESTNET
        url = TESTNET_API_URL if use_testnet else MAINNET_API_URL

        self.exchange = Exchange(wallet, url, account_address=address)
        self.info = Info(url, skip_ws=True)
        self.address = address or wallet.address
        self.is_testnet = use_testnet

        logger.info(
            f"Executor initialized: {'TESTNET' if use_testnet else 'MAINNET'} "
            f"address={self.address[:10]}..."
        )

    def execute_orders(self, orders: list[dict]) -> list[dict]:
        """Submit orders to Hyperliquid.

        Args:
            orders: List of order dicts (from stubs_to_hyperliquid_orders)

        Returns:
            List of response dicts from the exchange
        """
        if not orders:
            logger.info("No orders to execute")
            return []

        logger.info(f"Submitting {len(orders)} orders...")
        result = self.exchange.bulk_orders(orders)
        logger.info(f"Order result: {result}")
        return result if isinstance(result, list) else [result]

    def execute_single_order(self, order: dict) -> dict:
        """Submit a single order."""
        result = self.exchange.order(
            name=order["coin"],
            is_buy=order["is_buy"],
            sz=order["sz"],
            limit_px=order["limit_px"],
            order_type=order["order_type"],
            reduce_only=order.get("reduce_only", False),
        )
        return result

    def market_open(
        self, symbol: str, is_buy: bool, size: float, slippage: float = 0.05
    ) -> dict:
        """Open a position with a market order."""
        return self.exchange.market_open(symbol, is_buy, size, slippage=slippage)

    def market_close(self, symbol: str, size: Optional[float] = None) -> dict:
        """Close a position with a market order."""
        return self.exchange.market_close(symbol, sz=size)

    def cancel_all_orders(self, symbol: Optional[str] = None):
        """Cancel all open orders, optionally filtered by symbol."""
        open_orders = self.info.open_orders(self.address)
        cancels = []
        for order in open_orders:
            if symbol and order["coin"] != symbol:
                continue
            cancels.append({"coin": order["coin"], "oid": order["oid"]})

        if cancels:
            result = self.exchange.bulk_cancel(cancels)
            logger.info(f"Cancelled {len(cancels)} orders: {result}")
            return result
        logger.info("No open orders to cancel")
        return None

    def update_leverage(self, symbol: str, leverage: int, is_cross: bool = True):
        """Set leverage for a symbol."""
        return self.exchange.update_leverage(leverage, symbol, is_cross)

    def get_positions(self) -> list[dict]:
        """Get current positions from the exchange."""
        state = self.info.user_state(self.address)
        positions = []
        for pos in state["assetPositions"]:
            p = pos["position"]
            positions.append({
                "symbol": p["coin"],
                "size": float(p["szi"]),
                "entry_price": float(p["entryPx"]) if p.get("entryPx") else 0,
                "unrealized_pnl": float(p["unrealizedPnl"]),
                "leverage": p["leverage"],
                "liquidation_px": float(p["liquidationPx"]) if p.get("liquidationPx") else None,
            })
        return positions

    def get_account_value(self) -> float:
        """Get total account value."""
        state = self.info.user_state(self.address)
        return float(state["marginSummary"]["accountValue"])

    def get_balances(self) -> dict:
        """Get margin summary."""
        state = self.info.user_state(self.address)
        summary = state["marginSummary"]
        return {
            "account_value": float(summary["accountValue"]),
            "total_margin_used": float(summary["totalMarginUsed"]),
            "total_ntl_pos": float(summary["totalNtlPos"]),
            "withdrawable": float(state["withdrawable"]),
        }

    def get_current_prices(self) -> dict[str, float]:
        """Get all current mid prices."""
        raw = self.info.all_mids()
        return {k: float(v) for k, v in raw.items()}

    def get_spot_balances(self) -> dict[str, float]:
        """Get spot token balances. Returns dict of token_name -> hold amount."""
        state = self.info.spot_user_state(self.address)
        balances = {}
        for b in state.get("balances", []):
            balances[b["coin"]] = float(b["hold"])
        return balances

    def usd_class_transfer(self, amount: float, to_perp: bool = True):
        """Move USDC between spot and perp margin accounts."""
        result = self.exchange.usd_class_transfer(amount, to_perp=to_perp)
        direction = "spot→perp" if to_perp else "perp→spot"
        logger.info(f"Transferred ${amount:,.2f} {direction}: {result}")
        return result

    def get_sz_decimals(self, include_spot: bool = False) -> dict[str, int]:
        """Get size decimals for all assets."""
        meta = self.info.meta()
        result = {a["name"]: a["szDecimals"] for a in meta["universe"]}
        if include_spot:
            spot_meta = self.info.spot_meta_and_asset_ctxs()
            tokens = {t["index"]: t for t in spot_meta[0]["tokens"]}
            for pair in spot_meta[0]["universe"]:
                base = tokens[pair["tokens"][0]]
                quote = tokens[pair["tokens"][1]]
                spot_name = f'{base["name"]}/{quote["name"]}'
                result[spot_name] = base["szDecimals"]
                result[pair["name"]] = base["szDecimals"]
        return result

    def get_spot_name(self, base: str, quote: str = "USDC") -> str:
        """Get the spot pair name for ordering (e.g., 'BTC/USDC' or '@1').

        The SDK's name_to_coin mapping accepts both 'BASE/QUOTE' and the
        canonical '@N' format. We try 'BASE/QUOTE' first.
        """
        return f"{base}/{quote}"
