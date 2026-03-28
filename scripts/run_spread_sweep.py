#!/usr/bin/env python3
"""Weekly rolling spread backtest with DTE optimization sweep.

Strategies:
  - Put Spread (90/80):    short 90%·S put, long 80%·S put
  - Call Spread (110/120):  short 110%·S call, long 120%·S call
  - Naked Call (110):       short 110%·S call
Plus long-BTC buy-and-hold overlay for the two call strategies.

Entry: every 7 days, 0.25 BTC notional per trade.
DTE sweep: [7, 14, 21, 30, 45, 60, 90].

Usage:
  python -m scripts.run_spread_sweep
"""

import logging
import math
from bisect import bisect_left
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from backtest.options_estimators import SpreadEstimator, IVPremiumEstimator
from backtest.tearsheet import compute_metrics, print_tearsheet
from data.schema import get_connection, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Black-Scholes (copied from run_short_put_backtest to keep self-contained)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _d1d2(S, K, T, sigma, r=0):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def bs_put(S, K, T, sigma, r=0):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0)
    d1, d2 = _d1d2(S, K, T, sigma, r)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_call(S, K, T, sigma, r=0):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0)
    d1, d2 = _d1d2(S, K, T, sigma, r)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def delta_put(S, K, T, sigma, r=0):
    if T <= 0 or sigma <= 0 or S <= 0:
        return -1.0 if K > S else 0.0
    d1, _ = _d1d2(S, K, T, sigma, r)
    return _norm_cdf(d1) - 1.0


def delta_call(S, K, T, sigma, r=0):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 1.0 if S > K else 0.0
    d1, _ = _d1d2(S, K, T, sigma, r)
    return _norm_cdf(d1)


def find_put_strike(S, T, sigma, target_delta, r=0):
    """Find strike where put delta = target_delta (negative) via bisection."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return None
    lo, hi = S * 0.05, S * 1.5
    for _ in range(100):
        mid = (lo + hi) / 2
        d = delta_put(S, mid, T, sigma, r)
        if abs(d - target_delta) < 1e-6:
            return mid
        if d < target_delta:
            hi = mid
        else:
            lo = mid
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


def load_expiry_strikes(conn, start_date: date, option_type: str) -> tuple[dict, list]:
    """Return (expiry→sorted_strikes_array, sorted_expiry_list)."""
    df = conn.execute("""
        SELECT DISTINCT i.expiry_date, i.strike
        FROM options_instruments i
        WHERE i.option_type = ?
          AND i.expiry_date >= ?
        ORDER BY i.expiry_date, i.strike
    """, [option_type, start_date]).fetchdf()

    df["expiry_d"] = pd.to_datetime(df["expiry_date"]).dt.date
    strikes_map: dict[date, np.ndarray] = {}
    for exp, grp in df.groupby("expiry_d"):
        strikes_map[exp] = np.sort(grp["strike"].values)
    expiry_list = sorted(strikes_map.keys())
    return strikes_map, expiry_list


def compute_rv_series(delivery_prices: dict[date, float], window: int = 30) -> pd.Series:
    prices = pd.Series(delivery_prices).sort_index()
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window, min_periods=max(7, window // 3)).std() * np.sqrt(365) * 100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snap(available: np.ndarray, target: float):
    if len(available) == 0:
        return None
    return float(available[np.argmin(np.abs(available - target))])


def _btc_price_at(entry_date, sorted_dates, delivery_prices):
    if entry_date in delivery_prices:
        return delivery_prices[entry_date]
    idx = bisect_left(sorted_dates, entry_date) - 1
    return delivery_prices[sorted_dates[idx]] if idx >= 0 else None


def _nearest_expiry(target_date, expiry_list, tolerance_days):
    idx = bisect_left(expiry_list, target_date)
    best, best_diff = None, tolerance_days + 1
    for i in range(max(0, idx - 1), min(len(expiry_list), idx + 2)):
        diff = abs((expiry_list[i] - target_date).days)
        if diff < best_diff:
            best, best_diff = expiry_list[i], diff
    return best if best_diff <= tolerance_days else None

# ---------------------------------------------------------------------------
# Weekly-rolling backtest engine
# ---------------------------------------------------------------------------

def run_weekly_rolling(
    dvol_series: pd.Series,
    rv_series: pd.Series,
    delivery_prices: dict,
    sorted_dates: list,
    strikes_map: dict,
    expiry_list: list,
    spread_est: SpreadEstimator,
    iv_est: IVPremiumEstimator,
    *,
    side: str,
    target_dte: int,
    moneyness_short: float | None = None,
    moneyness_long: float | None = None,
    target_delta: float | None = None,
    spread_width_pct: float = 0.0,
    start_date: date = date(2021, 3, 24),
    notional: float = 0.25,
    interval: int = 7,
    dte_tol: int = 5,
) -> list[dict]:
    """Generate trades for one strategy variant across the full date range.

    Strike selection modes (pick one):
      - moneyness_short/moneyness_long: fixed % of spot
      - target_delta + spread_width_pct: delta-based short leg,
        long leg at short_K * (1 - spread_width_pct) for puts
    """
    use_delta = target_delta is not None
    is_spread = (moneyness_long is not None) or (spread_width_pct > 0)
    price_fn = bs_put if side == "put" else bs_call
    dfn = delta_put if side == "put" else delta_call

    last_delivery = sorted_dates[-1]
    last_entry = last_delivery - timedelta(days=max(target_dte - dte_tol, 1))

    trades = []
    entry_date = start_date

    while entry_date <= last_entry:
        target_exp = entry_date + timedelta(days=target_dte)
        expiry = _nearest_expiry(target_exp, expiry_list, dte_tol)
        if expiry is None:
            entry_date += timedelta(days=interval)
            continue

        actual_dte = (expiry - entry_date).days
        if actual_dte <= 0:
            entry_date += timedelta(days=interval)
            continue

        delivery = delivery_prices.get(expiry)
        if delivery is None:
            entry_date += timedelta(days=interval)
            continue

        S = _btc_price_at(entry_date, sorted_dates, delivery_prices)
        if S is None:
            entry_date += timedelta(days=interval)
            continue

        # IV estimation
        ts = pd.Timestamp(entry_date, tz="UTC")
        dvol = dvol_series.asof(ts)
        if pd.isna(dvol):
            ix = dvol_series.index.searchsorted(ts)
            dvol = dvol_series.iloc[ix] if ix < len(dvol_series) else None
        if dvol is None:
            entry_date += timedelta(days=interval)
            continue

        rv = rv_series.get(entry_date, default=np.nan)
        if pd.isna(rv) or rv <= 5.0:
            sigma = dvol / 100.0
        else:
            sigma = iv_est.estimate(delta=-0.20, dte=actual_dte, dvol=dvol, realized_vol=rv) / 100.0
        if sigma <= 0:
            entry_date += timedelta(days=interval)
            continue

        T = actual_dte / 365.0
        avail = strikes_map.get(expiry, np.array([]))
        if len(avail) == 0:
            entry_date += timedelta(days=interval)
            continue

        # --- Short leg ---
        if use_delta:
            theoretical_K = find_put_strike(S, T, sigma, target_delta)
            if theoretical_K is None:
                entry_date += timedelta(days=interval)
                continue
            short_K = _snap(avail, theoretical_K)
        else:
            short_K = _snap(avail, S * moneyness_short)
        if short_K is None:
            entry_date += timedelta(days=interval)
            continue

        sd = dfn(S, short_K, T, sigma)
        siv = spread_est.estimate(delta=sd, dte=actual_dte, dvol=dvol)
        bid_sig = max(sigma * (1 - siv / 2), sigma * 0.90)
        ask_sig = sigma * (1 + siv / 2)
        short_prem = price_fn(S, short_K, T, bid_sig)

        # --- Long leg ---
        long_K = None
        long_prem = 0.0
        if is_spread:
            if use_delta:
                lt = short_K * (1 - spread_width_pct)
            elif moneyness_long is not None:
                lt = S * moneyness_long
            else:
                entry_date += timedelta(days=interval)
                continue

            if side == "put":
                cands = avail[avail < short_K]
            else:
                cands = avail[avail > short_K]
            if len(cands) == 0:
                entry_date += timedelta(days=interval)
                continue
            long_K = _snap(cands, lt)
            if long_K is None:
                entry_date += timedelta(days=interval)
                continue
            long_prem = price_fn(S, long_K, T, ask_sig)

        net_prem = short_prem - long_prem

        # --- Settlement ---
        if side == "put":
            s_intr = max(short_K - delivery, 0)
            l_intr = max(long_K - delivery, 0) if long_K else 0.0
        else:
            s_intr = max(delivery - short_K, 0)
            l_intr = max(delivery - long_K, 0) if long_K else 0.0

        pnl_per_btc = net_prem - (s_intr - l_intr)

        trades.append({
            "entry_date": entry_date,
            "expiry_date": expiry,
            "dte": actual_dte,
            "btc_entry": S,
            "delivery": delivery,
            "short_K": short_K,
            "long_K": long_K,
            "short_delta": sd,
            "net_prem": net_prem,
            "pnl_per_btc": pnl_per_btc,
            "pnl_usd": pnl_per_btc * notional,
        })
        entry_date += timedelta(days=interval)

    return trades

# ---------------------------------------------------------------------------
# Equity curve construction
# ---------------------------------------------------------------------------

def build_daily_pnl(trades: list[dict], date_range: pd.DatetimeIndex) -> pd.Series:
    """Aggregate trade settlements into a daily P&L series (USD)."""
    daily = pd.Series(0.0, index=date_range)
    bucket: dict[date, float] = defaultdict(float)
    for t in trades:
        bucket[t["expiry_date"]] += t["pnl_usd"]
    for d, pnl in bucket.items():
        ts = pd.Timestamp(d)
        if ts in daily.index:
            daily.loc[ts] += pnl
        else:
            idx = daily.index.searchsorted(ts)
            if idx < len(daily):
                daily.iloc[idx] += pnl
    return daily


def daily_returns_from_pnl(daily_pnl: pd.Series, initial_capital: float) -> pd.Series:
    equity = initial_capital + daily_pnl.cumsum()
    equity = equity.clip(lower=1.0)
    ret = equity / equity.shift(1) - 1
    ret.iloc[0] = 0.0
    return ret

# ---------------------------------------------------------------------------
# Sweep + reporting
# ---------------------------------------------------------------------------

DTE_VALUES = [7, 14, 21, 30, 45, 60, 90]

STRATEGIES = [
    ("PutSprd 90/80",     "put",  0.90, 0.80),
    ("CallSprd 110/120",  "call", 1.10, 1.20),
    ("NakedCall 110",     "call", 1.10, None),
]


def run_dte_sweep(db_path=None, start_date: date = date(2021, 3, 24)):
    init_db(db_path)
    conn = get_connection(db_path)

    logger.info("Fitting estimators...")
    spread_est = SpreadEstimator().fit(db_path)
    iv_est = IVPremiumEstimator().fit(db_path)

    logger.info("Loading market data...")
    dvol_series = load_dvol_series(conn)
    delivery_prices = load_delivery_prices(conn)
    rv_series = compute_rv_series(delivery_prices)

    sorted_dates = sorted(delivery_prices.keys())
    end_date = sorted_dates[-1]
    initial_btc = _btc_price_at(start_date, sorted_dates, delivery_prices)
    initial_capital = initial_btc

    put_strikes, put_exp = load_expiry_strikes(conn, start_date, "P")
    call_strikes, call_exp = load_expiry_strikes(conn, start_date, "C")

    date_range = pd.date_range(start_date, end_date, freq="D")

    # BTC daily P&L (for overlay)
    btc_series = pd.Series(delivery_prices).sort_index()
    btc_daily = btc_series.reindex(date_range).ffill().bfill()
    btc_daily_pnl = btc_daily.diff().fillna(0.0)
    btc_end = float(btc_daily.iloc[-1])
    btc_bh_ret = btc_end / initial_btc - 1
    btc_returns = daily_returns_from_pnl(btc_daily_pnl, initial_capital)
    btc_metrics = compute_metrics(btc_returns, periods_per_year=365)

    all_results: dict[tuple[int, str], dict] = {}

    for target_dte in DTE_VALUES:
        logger.info("DTE = %d ...", target_dte)
        for name, side, m_short, m_long in STRATEGIES:
            sm = put_strikes if side == "put" else call_strikes
            el = put_exp if side == "put" else call_exp

            trades = run_weekly_rolling(
                dvol_series, rv_series, delivery_prices, sorted_dates,
                sm, el, spread_est, iv_est,
                side=side, target_dte=target_dte,
                moneyness_short=m_short, moneyness_long=m_long,
                start_date=start_date,
            )

            if not trades:
                all_results[(target_dte, name)] = {"trades": 0}
                if "Call" in name:
                    all_results[(target_dte, name + " +BTC")] = {"trades": 0}
                continue

            dpnl = build_daily_pnl(trades, date_range)
            ret = daily_returns_from_pnl(dpnl, initial_capital)
            m = compute_metrics(ret, periods_per_year=365)
            m["trades"] = len(trades)
            m["total_pnl"] = sum(t["pnl_usd"] for t in trades)
            wins = sum(1 for t in trades if t["pnl_usd"] >= 0)
            m["trade_win_rate"] = wins / len(trades)
            all_results[(target_dte, name)] = m

            # Overlay with long BTC for call strategies
            if "Call" in name:
                overlay_dpnl = dpnl + btc_daily_pnl
                overlay_ret = daily_returns_from_pnl(overlay_dpnl, initial_capital)
                om = compute_metrics(overlay_ret, periods_per_year=365)
                om["trades"] = len(trades)
                om["total_pnl"] = sum(t["pnl_usd"] for t in trades) + (btc_end - initial_btc)
                om["trade_win_rate"] = wins / len(trades)
                all_results[(target_dte, name + " +BTC")] = om

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print(f"  WEEKLY ROLLING SPREAD BACKTEST — DTE OPTIMIZATION")
    print(f"  Entry: every 7 days, 0.25 BTC/trade | {start_date} → {end_date}")
    print(f"{'=' * 100}")

    print(f"\n  BTC Buy & Hold: ${initial_btc:,.0f} → ${btc_end:,.0f} ({btc_bh_ret:+.1%})")
    if btc_metrics:
        print(f"  BTC B&H Sharpe: {btc_metrics.get('sharpe', 0):.2f}  "
              f"MaxDD: {btc_metrics.get('max_drawdown', 0):.1%}")

    col_names = [n for n, *_ in STRATEGIES]
    for n in list(col_names):
        if "Call" in n:
            col_names.append(n + " +BTC")

    for col in col_names:
        print(f"\n  --- {col} ---")
        print(f"  {'DTE':>4} {'#Tr':>5} {'TotP&L':>11} {'CAGR':>8} {'Sharpe':>8} "
              f"{'MaxDD':>8} {'WinRate':>8}")
        for dte in DTE_VALUES:
            m = all_results.get((dte, col), {})
            tr = m.get("trades", 0)
            if tr == 0:
                print(f"  {dte:>4}     —")
                continue
            print(f"  {dte:>4} {tr:>5} ${m.get('total_pnl', 0):>+10,.0f} "
                  f"{m.get('cagr', 0):>7.1%} {m.get('sharpe', 0):>8.2f} "
                  f"{m.get('max_drawdown', 0):>7.1%} {m.get('trade_win_rate', 0):>7.1%}")

    # Best DTE per strategy
    print(f"\n  --- Best DTE by Sharpe ---")
    for col in col_names:
        best_dte = max(
            DTE_VALUES,
            key=lambda d: all_results.get((d, col), {}).get("sharpe", -999),
        )
        m = all_results.get((best_dte, col), {})
        if m.get("trades", 0) > 0:
            print(f"  {col:<26s}  DTE={best_dte:>2}  "
                  f"Sharpe={m['sharpe']:.2f}  CAGR={m['cagr']:.1%}  MaxDD={m['max_drawdown']:.1%}")

    # Full tearsheet for the best put spread DTE
    best_ps_dte = max(
        DTE_VALUES,
        key=lambda d: all_results.get((d, "PutSprd 90/80"), {}).get("sharpe", -999),
    )
    m = all_results.get((best_ps_dte, "PutSprd 90/80"), {})
    if m.get("trades", 0) > 0:
        print_tearsheet(f"PutSprd-90/80-DTE{best_ps_dte}", m, {"returns": pd.Series(dtype=float)})

    return all_results


DELTA_VALUES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


def run_ps_delta_sweep(db_path=None, start_date: date = date(2021, 3, 24)):
    """Put Spread grid: delta × DTE, 10% width, weekly rolling 0.25 BTC."""
    init_db(db_path)
    conn = get_connection(db_path)

    logger.info("Fitting estimators...")
    spread_est = SpreadEstimator().fit(db_path)
    iv_est = IVPremiumEstimator().fit(db_path)

    logger.info("Loading market data...")
    dvol_series = load_dvol_series(conn)
    delivery_prices = load_delivery_prices(conn)
    rv_series = compute_rv_series(delivery_prices)

    sorted_dates = sorted(delivery_prices.keys())
    end_date = sorted_dates[-1]
    initial_btc = _btc_price_at(start_date, sorted_dates, delivery_prices)
    initial_capital = initial_btc

    put_strikes, put_exp = load_expiry_strikes(conn, start_date, "P")
    date_range = pd.date_range(start_date, end_date, freq="D")

    grid: dict[tuple[float, int], dict] = {}

    for td in DELTA_VALUES:
        for dte in DTE_VALUES:
            logger.info("delta=%.2f  DTE=%d ...", td, dte)
            trades = run_weekly_rolling(
                dvol_series, rv_series, delivery_prices, sorted_dates,
                put_strikes, put_exp, spread_est, iv_est,
                side="put", target_dte=dte,
                target_delta=-td, spread_width_pct=0.10,
                start_date=start_date,
            )

            if not trades:
                grid[(td, dte)] = {"trades": 0}
                continue

            dpnl = build_daily_pnl(trades, date_range)
            ret = daily_returns_from_pnl(dpnl, initial_capital)
            m = compute_metrics(ret, periods_per_year=365)
            m["trades"] = len(trades)
            m["total_pnl"] = sum(t["pnl_usd"] for t in trades)
            wins = sum(1 for t in trades if t["pnl_usd"] >= 0)
            m["trade_win_rate"] = wins / len(trades)
            avg_delta = np.mean([t["short_delta"] for t in trades])
            m["avg_delta"] = avg_delta
            grid[(td, dte)] = m

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 110}")
    print(f"  PUT SPREAD (10% width) — Delta × DTE Grid")
    print(f"  Weekly rolling, 0.25 BTC/trade | {start_date} → {end_date}")
    print(f"{'=' * 110}")

    # --- Sharpe table ---
    print(f"\n  SHARPE RATIO")
    hdr = f"  {'Delta':>6}"
    for dte in DTE_VALUES:
        hdr += f"  {'DTE='+str(dte):>8}"
    print(hdr)
    for td in DELTA_VALUES:
        row = f"  {td:>6.2f}"
        for dte in DTE_VALUES:
            m = grid.get((td, dte), {})
            if m.get("trades", 0) == 0:
                row += f"  {'—':>8}"
            else:
                row += f"  {m.get('sharpe', 0):>8.2f}"
        print(row)

    # --- CAGR table ---
    print(f"\n  CAGR")
    print(hdr)
    for td in DELTA_VALUES:
        row = f"  {td:>6.2f}"
        for dte in DTE_VALUES:
            m = grid.get((td, dte), {})
            if m.get("trades", 0) == 0:
                row += f"  {'—':>8}"
            else:
                row += f"  {m.get('cagr', 0):>7.1%} "
        print(row)

    # --- Max Drawdown table ---
    print(f"\n  MAX DRAWDOWN")
    print(hdr)
    for td in DELTA_VALUES:
        row = f"  {td:>6.2f}"
        for dte in DTE_VALUES:
            m = grid.get((td, dte), {})
            if m.get("trades", 0) == 0:
                row += f"  {'—':>8}"
            else:
                row += f"  {m.get('max_drawdown', 0):>7.1%} "
        print(row)

    # --- Calmar table ---
    print(f"\n  CALMAR RATIO (CAGR / MaxDD)")
    print(hdr)
    for td in DELTA_VALUES:
        row = f"  {td:>6.2f}"
        for dte in DTE_VALUES:
            m = grid.get((td, dte), {})
            if m.get("trades", 0) == 0:
                row += f"  {'—':>8}"
            else:
                row += f"  {m.get('calmar', 0):>8.2f}"
        print(row)

    # --- Win Rate table ---
    print(f"\n  WIN RATE")
    print(hdr)
    for td in DELTA_VALUES:
        row = f"  {td:>6.2f}"
        for dte in DTE_VALUES:
            m = grid.get((td, dte), {})
            if m.get("trades", 0) == 0:
                row += f"  {'—':>8}"
            else:
                row += f"  {m.get('trade_win_rate', 0):>7.1%} "
        print(row)

    # --- Total P&L table ---
    print(f"\n  TOTAL P&L ($)")
    hdr_wide = f"  {'Delta':>6}"
    for dte in DTE_VALUES:
        hdr_wide += f"  {'DTE='+str(dte):>10}"
    print(hdr_wide)
    for td in DELTA_VALUES:
        row = f"  {td:>6.2f}"
        for dte in DTE_VALUES:
            m = grid.get((td, dte), {})
            if m.get("trades", 0) == 0:
                row += f"  {'—':>10}"
            else:
                row += f"  ${m.get('total_pnl', 0):>+9,.0f}"
        print(row)

    # --- Best cell ---
    best_key = max(
        ((td, dte) for td in DELTA_VALUES for dte in DTE_VALUES
         if grid.get((td, dte), {}).get("trades", 0) > 0),
        key=lambda k: grid[k].get("sharpe", -999),
        default=None,
    )
    if best_key:
        bt, bd = best_key
        bm = grid[best_key]
        print(f"\n  BEST by Sharpe: delta={bt:.2f} DTE={bd}")
        print(f"    Sharpe={bm['sharpe']:.2f}  CAGR={bm['cagr']:.1%}  "
              f"MaxDD={bm['max_drawdown']:.1%}  WinRate={bm['trade_win_rate']:.1%}  "
              f"P&L=${bm['total_pnl']:+,.0f}  Trades={bm['trades']}  "
              f"AvgDelta={bm['avg_delta']:.3f}")
        print_tearsheet(
            f"PutSprd-{int(bt*100)}d-DTE{bd}",
            bm, {"returns": pd.Series(dtype=float)},
        )

    return grid


def main():
    import sys
    if "--moneyness" in sys.argv:
        run_dte_sweep()
    else:
        run_ps_delta_sweep()


if __name__ == "__main__":
    main()
