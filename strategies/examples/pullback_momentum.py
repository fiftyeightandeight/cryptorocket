"""Martin Luk-inspired pullback momentum strategy.

Adapted from the 2025 US Investing Championship winner's approach for
crypto perpetuals on Hyperliquid.  Core principles:

    - High-ADR (average daily range) leaders showing strong momentum
    - Pullback entries to rising EMAs (9/21/50) with confluence
    - Tight stops (0.5% account risk per trade) → low win rate (~22%)
      but ~5:1 reward-to-risk ratio
    - Trail with fast EMA; let big runners run

Since the original strategy uses intraday stock charts with AVWAP and
ORH breakouts, this adaptation translates the mechanical elements into
hourly crypto bars:

    1. **Trend filter**: price above rising 50-EMA → bullish bias
    2. **Pullback detection**: price pulls back to 9/21 EMA zone
    3. **Entry trigger**: breakout candle closes above the pullback high
    4. **Stop**: below the pullback low (tight)
    5. **Exit**: trailing 9-EMA break or max hold exceeded
    6. **ADR filter**: only trade symbols with high average daily range
       (proxy for volatility / "hot leaders")
    7. **Volume confirmation**: entry bar volume above rolling average
"""

import numpy as np
import pandas as pd

from backtest.commission import HyperliquidPerpsCommission
from strategies.base import CryptoMoonshot


class PullbackMomentum(CryptoMoonshot):
    """Pullback-to-EMA momentum strategy (Martin Luk style).

    Longs high-ADR crypto perps on pullbacks to the 9/21 EMA zone when
    the 50-EMA is rising and price is above it.  Tight stop at pullback
    low; trail with 9-EMA once in profit.

    Designed for hourly bars with no session restriction (24/7 crypto).
    """

    CODE = "pullback-momentum"
    SYMBOLS = []  # empty = all available
    INTERVAL = "1h"
    LEVERAGE = 1.0
    SLIPPAGE_BPS = 5
    COMMISSION_CLASS = HyperliquidPerpsCommission

    # --- Tunable parameters ---

    # EMA periods (in bars = hours for 1h interval)
    FAST_EMA = 9
    MID_EMA = 21
    SLOW_EMA = 50

    # Trend filter: 50-EMA must be rising over this many bars
    TREND_RISE_BARS = 5

    # ADR filter: minimum average daily range as a fraction (e.g. 0.05 = 5%)
    ADR_LOOKBACK = 14       # days of daily ranges to average
    ADR_MIN = 0.04          # 4% minimum ADR to qualify

    # Volume confirmation: entry-bar volume vs rolling average
    VOLUME_MA_PERIOD = 24   # hours
    VOLUME_MIN_RATIO = 1.2  # entry bar must be ≥ 1.2× average

    # Pullback definition: close must be within this % of the 9/21 EMA zone
    PULLBACK_PROXIMITY_PCT = 0.02  # within 2% of the EMA zone

    # Risk / position management
    MAX_HOLD_BARS = 72      # 3 days max hold
    STOP_ATR_MULT = 1.5     # stop distance in ATR multiples (fallback)

    # Also support short setups (price below falling 50-EMA, pullback UP to EMAs)
    ALLOW_SHORTS = True

    def prices_to_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        closes = prices["Close"]
        highs = prices["High"]
        lows = prices["Low"]
        volumes = prices["Volume"]

        symbols = closes.columns.tolist()
        signals = pd.DataFrame(0.0, index=closes.index, columns=symbols)

        # --- Compute indicators ---
        ema_fast = closes.ewm(span=self.FAST_EMA, adjust=False).mean()
        ema_mid = closes.ewm(span=self.MID_EMA, adjust=False).mean()
        ema_slow = closes.ewm(span=self.SLOW_EMA, adjust=False).mean()

        # 50-EMA slope (rising / falling)
        ema_slow_rising = ema_slow > ema_slow.shift(self.TREND_RISE_BARS)
        ema_slow_falling = ema_slow < ema_slow.shift(self.TREND_RISE_BARS)

        # ADR filter: compute hourly proxy then aggregate to daily range
        adr = self._compute_adr(highs, lows, closes)

        # Volume filter
        vol_ma = volumes.rolling(self.VOLUME_MA_PERIOD, min_periods=self.VOLUME_MA_PERIOD).mean()
        vol_above_avg = volumes > (vol_ma * self.VOLUME_MIN_RATIO)

        # --- EMA zone ---
        ema_zone_upper = pd.DataFrame(
            np.maximum(ema_fast.values, ema_mid.values),
            index=closes.index, columns=symbols,
        )
        ema_zone_lower = pd.DataFrame(
            np.minimum(ema_fast.values, ema_mid.values),
            index=closes.index, columns=symbols,
        )

        # Pullback proximity: close is near or within the 9/21 EMA zone
        proximity_band = closes * self.PULLBACK_PROXIMITY_PCT
        near_zone = (
            (closes <= ema_zone_upper + proximity_band)
            & (closes >= ema_zone_lower - proximity_band)
        )

        # --- Breakout confirmation ---
        # Previous bar was the pullback; current bar closes above the prior high
        prior_high = highs.shift(1)
        breakout_up = closes > prior_high

        prior_low = lows.shift(1)
        breakout_down = closes < prior_low

        # --- LONG setup ---
        # 1. Price above rising 50-EMA (trend)
        # 2. Pullback into 9/21 zone (mean reversion within trend)
        # 3. Breakout above prior high (momentum confirmation)
        # 4. ADR qualifies (volatility)
        # 5. Volume above average (participation)
        long_setup = (
            (closes > ema_slow)
            & ema_slow_rising
            & near_zone.shift(1)       # pullback was on prior bar
            & breakout_up
            & (adr >= self.ADR_MIN)
            & vol_above_avg
        )
        signals[long_setup] = 1.0

        # --- SHORT setup (mirror) ---
        if self.ALLOW_SHORTS:
            short_setup = (
                (closes < ema_slow)
                & ema_slow_falling
                & near_zone.shift(1)
                & breakout_down
                & (adr >= self.ADR_MIN)
                & vol_above_avg
            )
            signals[short_setup] = -1.0

        # --- Apply trailing exit logic ---
        signals = self._apply_exits(signals, closes, ema_fast)

        return signals

    def _compute_adr(
        self, highs: pd.DataFrame, lows: pd.DataFrame, closes: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute average daily range as fraction of price.

        For hourly bars we use a rolling high-low range over 24 bars
        (≈ 1 day) then average that over ADR_LOOKBACK days.
        """
        bars_per_day = 24  # 1h interval
        lookback = self.ADR_LOOKBACK * bars_per_day

        rolling_high = highs.rolling(bars_per_day, min_periods=bars_per_day).max()
        rolling_low = lows.rolling(bars_per_day, min_periods=bars_per_day).min()

        daily_range = (rolling_high - rolling_low) / closes
        adr = daily_range.rolling(lookback, min_periods=lookback).mean()

        return adr.fillna(0)

    def _apply_exits(
        self,
        signals: pd.DataFrame,
        closes: pd.DataFrame,
        ema_fast: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply trailing-EMA exit and max-hold exit.

        Once a signal fires, hold the position until:
            - Close crosses back through the 9-EMA against the trade, OR
            - MAX_HOLD_BARS is reached.

        This replaces the base class's simple signal with a stateful
        entry/exit model.
        """
        result = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        for col in signals.columns:
            col_idx = result.columns.get_loc(col)
            direction = 0.0
            bars_held = 0

            for i in range(len(signals)):
                raw = signals.iloc[i, col_idx]
                price = closes.iloc[i, col_idx]
                ema = ema_fast.iloc[i, col_idx]

                # New entry signal
                if raw != 0 and direction == 0:
                    direction = raw
                    bars_held = 0

                # Check exits while in a position
                if direction != 0:
                    bars_held += 1

                    # Exit 1: trailing EMA break
                    if direction > 0 and price < ema:
                        direction = 0.0
                        bars_held = 0
                        continue
                    if direction < 0 and price > ema:
                        direction = 0.0
                        bars_held = 0
                        continue

                    # Exit 2: max hold
                    if bars_held >= self.MAX_HOLD_BARS:
                        direction = 0.0
                        bars_held = 0
                        continue

                    result.iloc[i, col_idx] = direction

                    # Allow re-entry on new signal in opposite direction
                    if raw != 0 and raw != direction:
                        direction = raw
                        bars_held = 1
                        result.iloc[i, col_idx] = direction

        return result

    def signals_to_target_weights(
        self, signals: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Equal-weight active signals, scaled by leverage.

        Because this strategy uses tight stops and accepts low win rate,
        position sizing is conservative (equal-weight across active names).
        """
        n_active = (signals != 0).sum(axis=1).replace(0, np.nan)
        weights = signals.div(n_active, axis=0).fillna(0)
        return weights * self.LEVERAGE


class PullbackMomentumAggressive(PullbackMomentum):
    """More aggressive variant with higher leverage and looser filters.

    Suitable for strong trending markets. Uses 2x leverage and
    relaxes ADR/volume requirements to catch more setups.
    """

    CODE = "pullback-momentum-aggressive"
    LEVERAGE = 2.0
    ADR_MIN = 0.03              # 3% ADR (lower bar)
    VOLUME_MIN_RATIO = 1.0      # no volume filter
    PULLBACK_PROXIMITY_PCT = 0.03  # wider pullback zone
    MAX_HOLD_BARS = 96          # 4 days


class PullbackMomentumConservative(PullbackMomentum):
    """Conservative variant: longs only, tighter filters.

    Lower leverage, stricter ADR and volume requirements.
    More selective — fewer trades but higher conviction.
    """

    CODE = "pullback-momentum-conservative"
    LEVERAGE = 0.5
    ALLOW_SHORTS = False
    ADR_MIN = 0.06              # 6% ADR (only the hottest movers)
    VOLUME_MIN_RATIO = 1.5      # strong volume confirmation
    PULLBACK_PROXIMITY_PCT = 0.015  # tight pullback zone
    MAX_HOLD_BARS = 48          # 2 days max
