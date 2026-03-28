#!/usr/bin/env python3
"""Phase 1 settlement-based short BTC option backtest (put or call, naked or spread).

For each historical expiry cycle (2021-2026):
  1. Identify the ~20-delta OTM option at listing time using Black-Scholes
  2. Optionally select a long protective leg further OTM (vertical spread)
  3. Model entry premium using calibrated IV and spread estimators
  4. Calculate P&L at settlement

Usage:
  python -m scripts.run_short_put_backtest                          # naked put
  python -m scripts.run_short_put_backtest --side call              # naked call
  python -m scripts.run_short_put_backtest --spread-width 0.10      # put spread (10% wide)
  python -m scripts.run_short_put_backtest --side call --spread-width 0.10  # call spread
"""

import argparse
import logging
import math
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from backtest.options_estimators import SpreadEstimator, IVPremiumEstimator
from backtest.tearsheet import compute_metrics, print_tearsheet
from data.schema import get_connection, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Black-Scholes pricing and Greeks
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _d1d2(S: float, K: float, T: float, sigma: float, r: float = 0):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def black_scholes_put(S: float, K: float, T: float, sigma: float, r: float = 0) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0)
    d1, d2 = _d1d2(S, K, T, sigma, r)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def black_scholes_call(S: float, K: float, T: float, sigma: float, r: float = 0) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0)
    d1, d2 = _d1d2(S, K, T, sigma, r)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_delta_put(S: float, K: float, T: float, sigma: float, r: float = 0) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return -1.0 if K > S else 0.0
    d1, _ = _d1d2(S, K, T, sigma, r)
    return _norm_cdf(d1) - 1.0


def bs_delta_call(S: float, K: float, T: float, sigma: float, r: float = 0) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 1.0 if S > K else 0.0
    d1, _ = _d1d2(S, K, T, sigma, r)
    return _norm_cdf(d1)


def find_strike_for_delta(
    S: float, T: float, sigma: float, target_delta: float,
    side: str = "put", r: float = 0,
) -> Optional[float]:
    """Find the strike giving target delta via bisection.

    For puts, target_delta should be negative (e.g. -0.20).
    For calls, target_delta should be positive (e.g. +0.20).
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return None
    lo, hi = S * 0.1, S * 3.0
    delta_fn = bs_delta_put if side == "put" else bs_delta_call
    for _ in range(100):
        mid = (lo + hi) / 2
        d = delta_fn(S, mid, T, sigma, r)
        if abs(d - target_delta) < 1e-6:
            return mid
        if side == "put":
            if d < target_delta:
                hi = mid
            else:
                lo = mid
        else:
            if d > target_delta:
                lo = mid
            else:
                hi = mid
    return (lo + hi) / 2

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_dvol_series(conn) -> pd.Series:
    df = conn.execute("""
        SELECT timestamp, close FROM dvol
        WHERE currency = 'BTC' ORDER BY timestamp
    """).fetchdf()
    if df.empty:
        return pd.Series(dtype=float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")["close"]


def load_delivery_prices(conn) -> dict[date, float]:
    rows = conn.execute("""
        SELECT delivery_date, delivery_price FROM delivery_prices
        WHERE index_name = 'btc_usd' ORDER BY delivery_date
    """).fetchall()
    return {r[0]: r[1] for r in rows}


def load_expiry_cycles(conn, start_date: date, option_type: str = "P") -> pd.DataFrame:
    df = conn.execute("""
        SELECT DISTINCT i.expiry_date, i.strike, i.instrument_name
        FROM options_instruments i
        WHERE i.option_type = ?
          AND i.expiry_date >= ?
        ORDER BY i.expiry_date, i.strike
    """, [option_type, start_date]).fetchdf()
    return df


def compute_rv_series(delivery_prices: dict[date, float], window: int = 30) -> pd.Series:
    """Rolling annualized realized vol (%) from daily delivery prices."""
    prices = pd.Series(delivery_prices).sort_index()
    log_ret = np.log(prices / prices.shift(1))
    rv = log_ret.rolling(window, min_periods=max(7, window // 3)).std() * np.sqrt(365) * 100
    return rv

# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    db_path=None,
    side: str = "put",
    target_delta: float = -0.20,
    target_dte_min: int = 14,
    target_dte_max: int = 45,
    start_date: date = date(2021, 3, 24),
    notional_btc: float = 1.0,
    spread_width: float = 0.0,
) -> dict:
    """Run settlement-based short option backtest.

    side: 'put' or 'call'
    target_delta: negative for puts, positive for calls
    spread_width: 0 = naked, >0 = vertical spread as fraction of short strike
                  (e.g. 0.10 = long leg is 10% further OTM)
    """
    init_db(db_path)
    conn = get_connection(db_path)
    is_spread = spread_width > 0

    logger.info("Fitting estimators from snapshot data...")
    spread_est = SpreadEstimator().fit(db_path)
    iv_est = IVPremiumEstimator().fit(db_path)

    logger.info("Loading market data...")
    dvol_series = load_dvol_series(conn)
    delivery_prices = load_delivery_prices(conn)
    option_type = "P" if side == "put" else "C"
    expiry_data = load_expiry_cycles(conn, start_date, option_type)

    if dvol_series.empty:
        logger.error("No DVOL data available")
        return {}

    rv_series = compute_rv_series(delivery_prices)

    expiry_data["expiry_date_d"] = pd.to_datetime(expiry_data["expiry_date"]).dt.date
    expiry_dates = sorted(expiry_data["expiry_date_d"].unique())
    mode_str = f"{side} spread ({spread_width:.0%} wide)" if is_spread else f"naked {side}"
    logger.info("Found %d expiry dates from %s [%s]", len(expiry_dates), start_date, mode_str)

    bs_price_fn = black_scholes_put if side == "put" else black_scholes_call
    trades = []
    last_expiry = None

    for expiry in expiry_dates:
        delivery_price = delivery_prices.get(expiry)
        if delivery_price is None:
            continue

        entry_date = expiry - timedelta(days=target_dte_max)
        if entry_date < start_date:
            continue
        if last_expiry and entry_date < last_expiry:
            continue

        dte = (expiry - entry_date).days
        if dte < target_dte_min:
            continue

        entry_ts = pd.Timestamp(entry_date, tz="UTC")
        dvol_at_entry = dvol_series.asof(entry_ts)
        if pd.isna(dvol_at_entry):
            idx = dvol_series.index.searchsorted(entry_ts)
            if idx < len(dvol_series):
                dvol_at_entry = dvol_series.iloc[idx]
            else:
                continue

        btc_price = delivery_prices.get(entry_date)
        if btc_price is None:
            nearby_dates = [d for d in delivery_prices if d <= entry_date]
            if nearby_dates:
                btc_price = delivery_prices[max(nearby_dates)]
            else:
                continue

        rv_at_entry = rv_series.get(entry_date, default=np.nan)
        if pd.isna(rv_at_entry) or rv_at_entry <= 5.0:
            sigma_est = dvol_at_entry / 100.0
        else:
            sigma_est = iv_est.estimate(
                delta=target_delta, dte=dte,
                dvol=dvol_at_entry, realized_vol=rv_at_entry,
            ) / 100.0
        if sigma_est <= 0:
            continue

        T = dte / 365.0

        # --- Short leg ---
        short_strike = find_strike_for_delta(btc_price, T, sigma_est, target_delta, side=side)
        if short_strike is None:
            continue

        available_strikes = expiry_data[expiry_data["expiry_date_d"] == expiry]["strike"].values
        if len(available_strikes) == 0:
            continue
        short_strike = float(available_strikes[np.argmin(np.abs(available_strikes - short_strike))])

        if side == "put":
            short_delta = bs_delta_put(btc_price, short_strike, T, sigma_est)
        else:
            short_delta = bs_delta_call(btc_price, short_strike, T, sigma_est)

        spread_iv_pct = spread_est.estimate(delta=short_delta, dte=dte, dvol=dvol_at_entry)
        bid_sigma = sigma_est * (1 - spread_iv_pct / 2)
        if bid_sigma <= 0:
            bid_sigma = sigma_est * 0.95
        ask_sigma = sigma_est * (1 + spread_iv_pct / 2)

        short_premium = bs_price_fn(btc_price, short_strike, T, bid_sigma)

        # --- Long leg (if spread) ---
        long_strike = None
        long_premium = 0.0
        if is_spread:
            if side == "put":
                long_target = short_strike * (1 - spread_width)
            else:
                long_target = short_strike * (1 + spread_width)

            # Snap to nearest available strike (must be further OTM than short)
            if side == "put":
                candidates = available_strikes[available_strikes < short_strike]
            else:
                candidates = available_strikes[available_strikes > short_strike]

            if len(candidates) == 0:
                continue
            long_strike = float(candidates[np.argmin(np.abs(candidates - long_target))])
            long_premium = bs_price_fn(btc_price, long_strike, T, ask_sigma)

        net_premium = short_premium - long_premium

        # --- Settlement P&L ---
        if side == "put":
            short_intrinsic = max(short_strike - delivery_price, 0)
            long_intrinsic = max(long_strike - delivery_price, 0) if long_strike else 0.0
        else:
            short_intrinsic = max(delivery_price - short_strike, 0)
            long_intrinsic = max(delivery_price - long_strike, 0) if long_strike else 0.0

        net_intrinsic = short_intrinsic - long_intrinsic
        pnl_usd = net_premium - net_intrinsic

        if is_spread:
            width = abs(short_strike - long_strike)
            max_loss = width - net_premium
        else:
            width = 0.0
            max_loss = 0.0
        pnl_pct = pnl_usd / short_strike

        trade = {
            "entry_date": entry_date,
            "expiry_date": expiry,
            "dte": dte,
            "btc_price_entry": btc_price,
            "delivery_price": delivery_price,
            "short_strike": short_strike,
            "short_delta": short_delta,
            "sigma": sigma_est,
            "bid_sigma": bid_sigma,
            "dvol": dvol_at_entry,
            "rv": rv_at_entry if not pd.isna(rv_at_entry) else 0.0,
            "spread_iv_pct": spread_iv_pct,
            "short_premium": short_premium,
            "net_premium": net_premium,
            "net_intrinsic": net_intrinsic,
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
            "outcome": "profit" if pnl_usd >= 0 else "loss",
        }
        if is_spread:
            trade["long_strike"] = long_strike
            trade["long_premium"] = long_premium
            trade["width"] = width
            trade["max_loss"] = max_loss
        trades.append(trade)
        last_expiry = expiry

    if not trades:
        logger.warning("No trades generated")
        return {}

    trades_df = pd.DataFrame(trades)
    trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
    trades_df["expiry_date"] = pd.to_datetime(trades_df["expiry_date"])

    returns = trades_df.set_index("expiry_date")["pnl_pct"]
    returns.index = pd.to_datetime(returns.index)

    metrics = compute_metrics(returns, periods_per_year=365 / trades_df["dte"].mean())

    return {
        "trades": trades_df,
        "returns": returns,
        "metrics": metrics,
        "spread_est": spread_est,
        "iv_est": iv_est,
        "side": side,
        "is_spread": is_spread,
        "spread_width": spread_width,
    }

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_trade_summary(trades_df: pd.DataFrame, side: str = "put", is_spread: bool = False,
                        spread_width: float = 0.0):
    if is_spread:
        label = f"SHORT {side.upper()} SPREAD ({spread_width:.0%} wide)"
    else:
        label = f"SHORT {side.upper()} (naked)"
    print(f"\n{'=' * 80}")
    print(f"  {label} — TRADE SUMMARY")
    print(f"{'=' * 80}")

    n = len(trades_df)
    wins = (trades_df["pnl_usd"] >= 0).sum()
    losses = n - wins

    print(f"\n  Total trades:     {n}")
    print(f"  Winners:          {wins} ({wins/n:.1%})")
    print(f"  Losers:           {losses} ({losses/n:.1%})")
    print(f"  Date range:       {trades_df['entry_date'].min().date()} to "
          f"{trades_df['expiry_date'].max().date()}")

    print(f"\n  --- P&L Summary (per trade, 1 BTC notional) ---")
    print(f"  Mean P&L:         ${trades_df['pnl_usd'].mean():,.0f}")
    print(f"  Median P&L:       ${trades_df['pnl_usd'].median():,.0f}")
    print(f"  Total P&L:        ${trades_df['pnl_usd'].sum():,.0f}")
    print(f"  Best trade:       ${trades_df['pnl_usd'].max():,.0f}")
    print(f"  Worst trade:      ${trades_df['pnl_usd'].min():,.0f}")

    print(f"\n  --- Strike / Premium ---")
    print(f"  Avg short strike: ${trades_df['short_strike'].mean():,.0f}")
    if is_spread:
        print(f"  Avg long strike:  ${trades_df['long_strike'].mean():,.0f}")
        print(f"  Avg width:        ${trades_df['width'].mean():,.0f}")
        print(f"  Avg short prem:   ${trades_df['short_premium'].mean():,.0f}")
        print(f"  Avg long prem:    ${trades_df['long_premium'].mean():,.0f}")
    print(f"  Avg net premium:  ${trades_df['net_premium'].mean():,.0f}")
    print(f"  Avg delta:        {trades_df['short_delta'].mean():.3f}")
    print(f"  Avg DTE:          {trades_df['dte'].mean():.0f} days")
    print(f"  Avg sigma:        {trades_df['sigma'].mean():.1%}")
    print(f"  Avg bid sigma:    {trades_df['bid_sigma'].mean():.1%}")
    print(f"  Avg RV:           {trades_df['rv'].mean():.1f}%")
    print(f"  Avg spread(IV):   {trades_df['spread_iv_pct'].mean():.3f}")

    print(f"\n  --- Yearly Breakdown ---")
    yearly = trades_df.copy()
    yearly["year"] = trades_df["expiry_date"].dt.year
    for year, group in yearly.groupby("year"):
        w = (group["pnl_usd"] >= 0).sum()
        total = group["pnl_usd"].sum()
        print(f"  {year}: {len(group):3d} trades, "
              f"{w}/{len(group)} wins ({w/len(group):.0%}), "
              f"P&L ${total:>+10,.0f}")

    print(f"\n  --- Losing Trades ---")
    losers = trades_df[trades_df["pnl_usd"] < 0].sort_values("pnl_usd")
    if losers.empty:
        print("  None!")
    else:
        for _, t in losers.head(10).iterrows():
            if is_spread:
                print(f"  {t['expiry_date'].date()}: "
                      f"K={t['short_strike']:,.0f}/{t['long_strike']:,.0f}, "
                      f"delivery={t['delivery_price']:,.0f}, "
                      f"P&L=${t['pnl_usd']:+,.0f}")
            else:
                print(f"  {t['expiry_date'].date()}: K={t['short_strike']:,.0f}, "
                      f"delivery={t['delivery_price']:,.0f}, "
                      f"P&L=${t['pnl_usd']:+,.0f}")


def main():
    parser = argparse.ArgumentParser(description="Short BTC option backtest (Phase 1)")
    parser.add_argument("--db-path", help="Custom database path")
    parser.add_argument("--side", choices=["put", "call"], default="put",
                        help="Option side to sell (default: put)")
    parser.add_argument("--spread-width", type=float, default=0.0,
                        help="Vertical spread width as fraction of short strike "
                             "(0 = naked, 0.10 = 10%% wide)")
    parser.add_argument("--target-delta", type=float, default=None,
                        help="Target delta magnitude (default: 0.20)")
    parser.add_argument("--target-dte-min", type=int, default=14)
    parser.add_argument("--target-dte-max", type=int, default=45)
    parser.add_argument("--start-date", type=str, default="2021-03-24")
    parser.add_argument("--notional", type=float, default=1.0)
    args = parser.parse_args()

    if args.target_delta is None:
        target_delta = -0.20 if args.side == "put" else 0.20
    else:
        target_delta = -abs(args.target_delta) if args.side == "put" else abs(args.target_delta)

    start = date.fromisoformat(args.start_date)

    results = run_backtest(
        db_path=args.db_path,
        side=args.side,
        target_delta=target_delta,
        target_dte_min=args.target_dte_min,
        target_dte_max=args.target_dte_max,
        start_date=start,
        notional_btc=args.notional,
        spread_width=args.spread_width,
    )

    if not results:
        print("No results — check logs for errors.")
        return

    trades_df = results["trades"]
    metrics = results["metrics"]
    returns = results["returns"]
    side = results["side"]
    is_spread = results["is_spread"]
    sw = results["spread_width"]

    print_trade_summary(trades_df, side=side, is_spread=is_spread, spread_width=sw)
    if is_spread:
        label = f"Short{side.title()}Sprd-{int(abs(target_delta)*100)}d-{int(sw*100)}w"
    else:
        label = f"Short{side.title()}-{int(abs(target_delta)*100)}d"
    print_tearsheet(label, metrics, {"returns": returns})


if __name__ == "__main__":
    main()
