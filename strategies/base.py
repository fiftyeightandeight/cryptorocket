from typing import Optional

import numpy as np
import pandas as pd

from backtest.commission import CommissionModel, HyperliquidPerpsCommission
from data.store import get_available_symbols, get_prices


class CryptoMoonshot:
    """Base class for crypto trading strategies, mirroring Moonshot's pipeline.

    Subclasses must define:
        CODE: str - strategy identifier
        SYMBOLS: list[str] - tradable universe
        prices_to_signals() - at minimum

    Pipeline (same as Moonshot):
        get_prices() → prices_to_signals() → signals_to_target_weights()
        → target_weights_to_positions() → positions_to_gross_returns()

    Session support:
        Set SESSION_START_HOUR and SESSION_END_HOUR (UTC) to restrict
        trading to a time window within each day. Positions are
        automatically flattened at session end.
    """

    CODE: str = ""
    SYMBOLS: list[str] = []
    INTERVAL: str = "1d"
    FIELDS: list[str] = ["Open", "High", "Low", "Close", "Volume"]
    SLIPPAGE_BPS: float = 5.0
    COMMISSION_CLASS: type[CommissionModel] = HyperliquidPerpsCommission
    BENCHMARK: Optional[str] = None
    LEVERAGE: float = 1.0
    DB_PATH: Optional[str] = None

    # Session window (UTC hours, None = no session filtering)
    SESSION_START_HOUR: Optional[int] = None  # e.g. 14 for 14:00 UTC
    SESSION_END_HOUR: Optional[int] = None    # e.g. 22 for 22:00 UTC

    # --- Pipeline methods (override in subclass) ---

    def prices_to_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from prices.

        Args:
            prices: MultiIndex DataFrame (Field, Symbol) indexed by Date.
                    Access via prices.loc['Close'], prices.loc['Open'], etc.

        Returns:
            DataFrame indexed by Date with symbols as columns.
            Values represent signal strength (e.g., 1 = long, -1 = short, 0 = flat).
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement prices_to_signals()")

    def signals_to_target_weights(
        self, signals: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Convert signals to portfolio target weights.

        Default: equal-weight all non-zero signals, scaled by LEVERAGE.
        """
        # Count non-zero signals per row
        n_active = (signals != 0).sum(axis=1).replace(0, np.nan)
        weights = signals.div(n_active, axis=0).fillna(0)
        return weights * self.LEVERAGE

    def target_weights_to_positions(
        self, target_weights: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply execution constraints to target weights.

        Default: assume instant fills (positions = target_weights).
        If session is configured, flattens positions on the last bar
        of each session so there is no overnight carry.
        Override for limit order modeling, position size limits, etc.
        """
        positions = target_weights.copy()

        if self._has_session:
            positions = self._flatten_at_session_end(positions)

        return positions

    def _flatten_at_session_end(self, positions: pd.DataFrame) -> pd.DataFrame:
        """Zero out positions on the last bar of each session day."""
        idx = positions.index
        # Group by date; last bar in each day's session is the exit bar
        dates = idx.date
        last_bar_mask = pd.Series(dates, index=idx).groupby(dates).transform(
            lambda g: g.index == g.index[-1]
        )
        positions.loc[last_bar_mask.values] = 0.0
        return positions

    def positions_to_gross_returns(
        self, positions: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate gross returns from positions and prices.

        Default: close-to-close returns weighted by prior day's position.
        """
        closes = prices["Close"]
        pct_returns = closes.pct_change()
        gross_returns = positions.shift(1) * pct_returns
        return gross_returns.fillna(0)

    def order_stubs_to_orders(
        self, orders: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Convert position-change order stubs to exchange-ready orders.

        Default: market orders.
        Override for limit orders, stop losses, etc.
        """
        orders["order_type"] = "market"
        orders["tif"] = "Ioc"
        return orders

    # --- Framework methods ---

    @property
    def _has_session(self) -> bool:
        return self.SESSION_START_HOUR is not None and self.SESSION_END_HOUR is not None

    @property
    def _periods_per_year(self) -> float:
        """Annualization factor based on interval and session config."""
        interval = self.INTERVAL
        # Bars per 24h
        bars_map = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}
        bars_per_day = bars_map.get(interval, 1)

        if self._has_session and bars_per_day > 1:
            # Session hours
            start_h = self.SESSION_START_HOUR
            end_h = self.SESSION_END_HOUR
            session_hours = (end_h - start_h) % 24
            bars_per_session = int(session_hours * (bars_per_day / 24))
            # ~365 sessions per year
            return bars_per_session * 365
        elif bars_per_day == 1:
            return 365
        else:
            return bars_per_day * 365

    def _resolve_symbols(self) -> list[str]:
        """Resolve the tradable universe. If SYMBOLS is empty, use all available."""
        if self.SYMBOLS:
            return self.SYMBOLS
        return get_available_symbols(interval=self.INTERVAL, db_path=self.DB_PATH)

    def _get_prices(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch price data, filtered to session window if configured."""
        symbols = self._resolve_symbols()
        prices = get_prices(
            symbols=symbols,
            interval=self.INTERVAL,
            fields=self.FIELDS,
            start_date=start_date,
            end_date=end_date,
            db_path=self.DB_PATH,
        )

        if self._has_session and not prices.empty:
            prices = self._filter_session(prices)

        return prices

    def _filter_session(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Keep only bars within the session window."""
        start_h = self.SESSION_START_HOUR
        end_h = self.SESSION_END_HOUR

        idx = prices.index
        hours = idx.hour

        if start_h < end_h:
            # Normal window e.g. 14:00-22:00
            mask = (hours >= start_h) & (hours < end_h)
        else:
            # Overnight window e.g. 22:00-06:00
            mask = (hours >= start_h) | (hours < end_h)

        return prices.loc[mask]

    def backtest(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict:
        """Run the full backtest pipeline.

        Returns dict with keys:
            - returns: pd.Series of net portfolio returns
            - gross_returns: pd.DataFrame of gross returns per symbol
            - positions: pd.DataFrame of positions over time
            - signals: pd.DataFrame of raw signals
            - target_weights: pd.DataFrame of target weights
            - commissions: pd.DataFrame of commission costs
            - periods_per_year: float annualization factor
        """
        # 1. Get prices
        prices = self._get_prices(start_date, end_date)

        # 2. Signal pipeline
        signals = self.prices_to_signals(prices)
        target_weights = self.signals_to_target_weights(signals, prices)
        positions = self.target_weights_to_positions(target_weights, prices)
        gross_returns = self.positions_to_gross_returns(positions, prices)

        # 3. Apply slippage
        position_changes = positions.diff().fillna(0)
        slippage_cost = position_changes.abs() * (self.SLIPPAGE_BPS / 10000)

        # 4. Apply commissions
        commission_model = self.COMMISSION_CLASS()
        closes = prices["Close"]
        commissions = commission_model.get_commissions(closes, position_changes)

        # 5. Net returns
        net_returns = gross_returns - slippage_cost - commissions
        portfolio_returns = net_returns.sum(axis=1)

        return {
            "returns": portfolio_returns,
            "gross_returns": gross_returns,
            "net_returns": net_returns,
            "positions": positions,
            "signals": signals,
            "target_weights": target_weights,
            "commissions": commissions,
            "slippage": slippage_cost,
            "periods_per_year": self._periods_per_year,
        }

    def trade(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate current order stubs for live trading.

        Returns DataFrame with columns: Symbol, Qty, Action, Strategy
        """
        prices = self._get_prices(start_date, end_date)
        signals = self.prices_to_signals(prices)
        target_weights = self.signals_to_target_weights(signals, prices)
        positions = self.target_weights_to_positions(target_weights, prices)

        if positions.empty:
            return pd.DataFrame(columns=["Symbol", "Action", "Weight", "Strategy"])

        # Get latest target position
        latest = positions.iloc[-1]
        # Previous position (for delta calculation)
        prev = positions.iloc[-2] if len(positions) >= 2 else pd.Series(0, index=latest.index)

        delta = latest - prev
        delta = delta[delta.abs() > 1e-8]  # Filter noise

        if delta.empty:
            return pd.DataFrame(columns=["Symbol", "Action", "Weight", "Strategy"])

        orders = pd.DataFrame({
            "Symbol": delta.index,
            "Weight": delta.abs().values,
            "Action": ["BUY" if d > 0 else "SELL" for d in delta.values],
            "Strategy": self.CODE,
        })

        return self.order_stubs_to_orders(orders, prices)
