import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from data.schema import get_connection

logger = logging.getLogger(__name__)


class Blotter:
    """Local position and fill tracking (replaces QuantRocket blotter)."""

    def __init__(self, db_path=None):
        self.db_path = db_path

    def _conn(self):
        return get_connection(self.db_path)

    def record_order(
        self,
        strategy: str,
        symbol: str,
        side: str,
        size: float,
        order_type: str,
        limit_price: Optional[float] = None,
    ) -> str:
        """Record a new order, returning the order_id."""
        order_id = str(uuid.uuid4())[:12]
        conn = self._conn()
        conn.execute(
            """INSERT INTO orders (order_id, strategy, symbol, side, size, order_type, limit_price, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
            [order_id, strategy, symbol, side, size, order_type, limit_price,
             datetime.now(timezone.utc)],
        )
        return order_id

    def record_fill(
        self,
        order_id: str,
        fill_price: float,
        fill_size: float,
        fee: float = 0.0,
    ):
        """Record a fill for an existing order and update position."""
        now = datetime.now(timezone.utc)
        conn = self._conn()

        # Update order
        conn.execute(
            """UPDATE orders SET status = 'filled', fill_price = ?, fill_size = ?, fee = ?, filled_at = ?
               WHERE order_id = ?""",
            [fill_price, fill_size, fee, now, order_id],
        )

        # Get order details
        row = conn.execute(
            "SELECT strategy, symbol, side, size FROM orders WHERE order_id = ?",
            [order_id],
        ).fetchone()

        if row:
            strategy, symbol, side, _ = row
            signed_size = fill_size if side == "buy" else -fill_size

            # Update position
            existing = conn.execute(
                "SELECT size, entry_price FROM positions WHERE strategy = ? AND symbol = ?",
                [strategy, symbol],
            ).fetchone()

            if existing:
                old_size, old_entry = existing
                new_size = old_size + signed_size
                # Weighted average entry for increasing positions
                if abs(new_size) > abs(old_size) and old_size * signed_size > 0:
                    new_entry = (old_entry * abs(old_size) + fill_price * fill_size) / abs(new_size)
                elif abs(new_size) < 1e-10:
                    new_entry = 0
                else:
                    new_entry = old_entry if abs(new_size) > 0 else 0

                conn.execute(
                    """UPDATE positions SET size = ?, entry_price = ?, updated_at = ?
                       WHERE strategy = ? AND symbol = ?""",
                    [new_size, new_entry, now, strategy, symbol],
                )
            else:
                conn.execute(
                    """INSERT INTO positions (strategy, symbol, size, entry_price, updated_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    [strategy, symbol, signed_size, fill_price, now],
                )

        logger.info(f"Fill: order={order_id} price={fill_price} size={fill_size} fee={fee}")

    def cancel_order(self, order_id: str):
        """Mark an order as cancelled."""
        conn = self._conn()
        conn.execute(
            "UPDATE orders SET status = 'cancelled' WHERE order_id = ?",
            [order_id],
        )

    def get_positions(self, strategy: Optional[str] = None) -> dict[str, float]:
        """Get current positions as dict of symbol -> size (signed)."""
        conn = self._conn()
        if strategy:
            rows = conn.execute(
                "SELECT symbol, size FROM positions WHERE strategy = ? AND ABS(size) > 1e-10",
                [strategy],
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT symbol, size FROM positions WHERE ABS(size) > 1e-10"
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def get_position_details(self, strategy: Optional[str] = None) -> pd.DataFrame:
        """Get detailed position information."""
        conn = self._conn()
        if strategy:
            df = conn.execute(
                "SELECT * FROM positions WHERE strategy = ? AND ABS(size) > 1e-10",
                [strategy],
            ).fetchdf()
        else:
            df = conn.execute(
                "SELECT * FROM positions WHERE ABS(size) > 1e-10"
            ).fetchdf()
        return df

    def get_order_history(
        self,
        strategy: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Get order history."""
        conn = self._conn()
        query = "SELECT * FROM orders WHERE 1=1"
        params = []
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        df = conn.execute(query, params).fetchdf()
        return df

    def get_pnl(self, strategy: str, current_prices: dict[str, float]) -> dict:
        """Calculate realized + unrealized PnL for a strategy."""
        conn = self._conn()

        # Realized PnL from filled orders
        fills = conn.execute(
            """SELECT symbol, side, fill_price, fill_size, fee
               FROM orders WHERE strategy = ? AND status = 'filled'""",
            [strategy],
        ).fetchall()

        realized_pnl = -sum(f[4] for f in fills)  # Start with negative fees

        # Unrealized PnL from open positions
        positions = conn.execute(
            "SELECT symbol, size, entry_price FROM positions WHERE strategy = ? AND ABS(size) > 1e-10",
            [strategy],
        ).fetchall()

        unrealized_pnl = 0.0
        for symbol, size, entry_price in positions:
            current_price = current_prices.get(symbol, entry_price)
            unrealized_pnl += size * (current_price - entry_price)

        return {
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": realized_pnl + unrealized_pnl,
        }
