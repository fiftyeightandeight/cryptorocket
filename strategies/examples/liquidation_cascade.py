import numpy as np
import pandas as pd

from backtest.commission import HyperliquidPerpsCommission
from data.store import get_funding_rates
from strategies.base import CryptoMoonshot


class LiquidationCascadeMomentum(CryptoMoonshot):
    """Detect and ride liquidation cascades using funding rate + breakout + volume.

    v1 proxy approach — uses funding rates as a proxy for crowded leverage
    (standing in for historical OI which isn't yet in the pipeline).

    Signal logic:
        1. Funding rate extreme: rolling cumulative funding exceeds a
           z-score threshold, indicating crowded positioning on one side.
        2. Price breakout: price breaks beyond the recent ATR-scaled range
           in the direction AGAINST the crowded side (triggering liquidations).
        3. Volume confirmation: volume spikes above its rolling mean,
           consistent with forced-liquidation flow.

    When all three conditions align, the strategy enters in the cascade
    direction (contra to the crowded side) and holds for HOLD_BARS.

    Direction mapping:
        - Crowded longs  (high +funding) + price breaks DOWN → SHORT
        - Crowded shorts (high -funding) + price breaks UP   → LONG
    """

    CODE = "liquidation-cascade"
    SYMBOLS = []
    INTERVAL = "1h"
    LEVERAGE = 1.0
    SLIPPAGE_BPS = 5
    COMMISSION_CLASS = HyperliquidPerpsCommission

    # --- Tunable parameters ---

    # Funding rate analysis (8h rates accumulated over a rolling window)
    FUNDING_WINDOW = 72        # hours (~3 days) for rolling cumulative funding
    FUNDING_Z_THRESHOLD = 1.5  # z-score of cumulative funding to flag "extreme"

    # Price breakout detection
    BREAKOUT_LOOKBACK = 24     # hours for ATR and range calculation
    BREAKOUT_ATR_MULT = 2.0    # ATR multiples beyond rolling mid for breakout

    # Volume confirmation
    VOLUME_LOOKBACK = 48       # hours for rolling volume mean
    VOLUME_SPIKE_MULT = 2.5    # multiple of rolling mean to qualify as spike

    # Position holding
    HOLD_BARS = 6              # bars to hold after a cascade signal fires

    def prices_to_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        closes = prices["Close"]
        highs = prices["High"]
        lows = prices["Low"]
        volumes = prices["Volume"]

        symbols = closes.columns.tolist()
        signals = pd.DataFrame(0.0, index=closes.index, columns=symbols)

        # --- 1. Funding rate extreme (proxy for crowded leverage) ---
        funding_extreme = self._funding_extreme(closes)

        # --- 2. Price breakout detection ---
        atr = (highs - lows).rolling(self.BREAKOUT_LOOKBACK, min_periods=1).mean()
        rolling_mid = closes.rolling(self.BREAKOUT_LOOKBACK, min_periods=1).mean()

        breakout_up = closes > (rolling_mid + self.BREAKOUT_ATR_MULT * atr)
        breakout_down = closes < (rolling_mid - self.BREAKOUT_ATR_MULT * atr)

        # --- 3. Volume spike confirmation ---
        vol_ma = volumes.rolling(self.VOLUME_LOOKBACK, min_periods=1).mean()
        vol_spike = volumes > (vol_ma * self.VOLUME_SPIKE_MULT)

        # --- 4. Combine: cascade fires when all three align ---
        # Crowded longs (funding_extreme > 0) + price breaks DOWN + volume spike → SHORT
        short_cascade = (funding_extreme > 0) & breakout_down & vol_spike
        # Crowded shorts (funding_extreme < 0) + price breaks UP + volume spike → LONG
        long_cascade = (funding_extreme < 0) & breakout_up & vol_spike

        signals[long_cascade] = 1.0
        signals[short_cascade] = -1.0

        # --- 5. Hold signals for HOLD_BARS after trigger ---
        signals = self._hold_signals(signals)

        return signals

    def _funding_extreme(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Compute a funding-extreme indicator aligned to the price index.

        Returns a DataFrame of same shape as closes:
            +1 where cumulative funding is extremely positive (crowded longs)
            -1 where cumulative funding is extremely negative (crowded shorts)
             0 otherwise
        """
        symbols = closes.columns.tolist()
        result = pd.DataFrame(0.0, index=closes.index, columns=symbols)

        try:
            start_str = str(closes.index[0])
            end_str = str(closes.index[-1])
            funding = get_funding_rates(
                symbols=symbols,
                start_date=start_str,
                end_date=end_str,
                db_path=self.DB_PATH,
            )
        except Exception:
            return result

        if funding.empty:
            return result

        # Keep only symbols present in both datasets
        common = [s for s in symbols if s in funding.columns]
        if not common:
            return result

        # Reindex funding to hourly, forward-filling the 8h snapshots
        funding = funding[common].reindex(closes.index, method="ffill")

        # Rolling cumulative funding over the window
        cum_funding = funding.rolling(self.FUNDING_WINDOW, min_periods=1).sum()

        # Z-score the cumulative funding
        cum_mean = cum_funding.rolling(self.FUNDING_WINDOW, min_periods=1).mean()
        cum_std = cum_funding.rolling(self.FUNDING_WINDOW, min_periods=1).std()
        cum_std = cum_std.replace(0, np.nan)
        z = (cum_funding - cum_mean) / cum_std

        result[common] = 0.0
        result.loc[:, common] = result.loc[:, common].where(
            ~(z > self.FUNDING_Z_THRESHOLD), 1.0
        )
        result.loc[:, common] = result.loc[:, common].where(
            ~(z < -self.FUNDING_Z_THRESHOLD), -1.0
        )

        return result

    def _hold_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill each non-zero signal for HOLD_BARS periods.

        Newer signals override earlier ones: if a new trigger fires during
        an active hold window, the new signal's direction and hold timer
        take over.
        """
        raw = signals.copy()
        held = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        for col in raw.columns:
            col_idx = held.columns.get_loc(col)
            active_val = 0.0
            bars_left = 0

            for i in range(len(raw)):
                raw_val = raw.iloc[i, col_idx]
                if raw_val != 0:
                    # New trigger — start (or restart) the hold window
                    active_val = raw_val
                    bars_left = self.HOLD_BARS
                if bars_left > 0:
                    held.iloc[i, col_idx] = active_val
                    bars_left -= 1

        return held

    def signals_to_target_weights(
        self, signals: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Equal-weight active signals, scaled by leverage."""
        n_active = (signals != 0).sum(axis=1).replace(0, np.nan)
        weights = signals.div(n_active, axis=0).fillna(0)
        return weights * self.LEVERAGE
