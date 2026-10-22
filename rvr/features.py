# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Daily per-contract characteristics (PREREGISTRATION.md §5–7).

Every matrix is indexed by day (epoch ms at 00:00 UTC) × symbol. Row d uses only daily closes up to
d's own 00:00 close — the close of the traded 23:00–00:00 bar — so it is known to a decision at
d 00:05. Only bars with volume count. tests/test_features.py corrupts everything from a cutoff on
and checks no earlier row changes.
"""
import numpy as np
import pandas as pd

from rvr.config import HOUR_MS
from rvr.pit import traded_bars

DAY_MS = 24 * HOUR_MS


def day_range(start, end):
    """Days (epoch ms at 00:00 UTC) from start to end inclusive; start/end are dates or epoch ms."""
    def to_ms(x):
        return int(x) if isinstance(x, (int, np.integer)) else int(pd.Timestamp(x, tz="UTC").value // 10**6)

    return pd.Index(np.arange(to_ms(start), to_ms(end) + DAY_MS, DAY_MS, dtype="int64"), name="day")


def daily_closes(prices, days):
    """Close of the traded bar ending at each day's 00:00 UTC; NaN if that bar didn't trade."""
    bars = traded_bars(prices)
    at_midnight = bars[(bars["close_time"] + 1) % DAY_MS == 0]
    wide = (at_midnight.assign(day=at_midnight["close_time"] + 1)
            .pivot(index="day", columns="symbol", values="close"))
    return wide.reindex(days)


def daily_quote_volume(prices, days):
    """USDT volume of the 24 hourly bars ending at each day's 00:00 UTC."""
    day = (prices["close_time"] // DAY_MS + 1) * DAY_MS
    wide = prices.assign(day=day).pivot_table(index="day", columns="symbol", values="quote_volume",
                                              aggfunc="sum", fill_value=0.0)
    return wide.reindex(days, fill_value=0.0)


def _broadcast(series, like):
    return pd.DataFrame(np.broadcast_to(series.to_numpy()[:, None], like.shape), index=like.index, columns=like.columns)


def _rolling_sum(frame, window):
    return frame.rolling(window, min_periods=1).sum()


def masked_beta(returns, market, mask, window, min_obs):
    """Rolling OLS slope of returns on market, over the days in mask within each window."""
    m = _broadcast(market, returns).where(mask, 0.0)
    x = returns.where(mask, 0.0)
    n = _rolling_sum(mask.astype(float), window)
    sx, sm = _rolling_sum(x, window), _rolling_sum(m, window)
    cov = _rolling_sum(x * m, window) / n - sx * sm / n**2
    var = _rolling_sum(m * m, window) / n - (sm / n) ** 2
    return (cov / var).where(n >= min_obs)


def downside_beta(returns, market, window, min_down_days, min_days):
    """Beta on days the market fell; else ordinary beta; else 1.0 (PREREGISTRATION.md §6)."""
    valid = returns.notna() & _broadcast(market.notna(), returns)
    down = valid & _broadcast(market < 0, returns)
    beta_down = masked_beta(returns, market, down, window, min_down_days)
    beta_all = masked_beta(returns, market, valid, window, min_days)
    return beta_down.fillna(beta_all).fillna(1.0)


def coskewness(returns, market, window, min_obs):
    """E[(r − r̄)(m − m̄)²] / (σ_r σ_m²) over each window (Harvey & Siddique 2000)."""
    valid = returns.notna() & _broadcast(market.notna(), returns)
    x = returns.where(valid, 0.0)
    m = _broadcast(market, returns).where(valid, 0.0)
    n = _rolling_sum(valid.astype(float), window)
    def mean(frame):
        return _rolling_sum(frame, window) / n

    a, b = mean(x), mean(m)
    exm2, exm, em2, ex2 = mean(x * m * m), mean(x * m), mean(m * m), mean(x * x)
    numerator = exm2 - 2 * b * exm - a * em2 + 2 * a * b**2
    denominator = np.sqrt((ex2 - a**2).clip(lower=0)) * (em2 - b**2)
    return (numerator / denominator).where(n >= min_obs)


def characteristics(prices, days, spec):
    """Wide day × symbol frames for every characteristic the strategies and pricing test use."""
    closes = daily_closes(prices, days)
    volume = daily_quote_volume(prices, days)
    symbols = closes.columns.union(volume.columns)
    closes, volume = closes.reindex(columns=symbols), volume.reindex(columns=symbols, fill_value=0.0)
    returns = closes / closes.shift(1) - 1
    market = returns[spec.hedge_symbol]
    adv = _rolling_sum(volume, spec.adv_days) / spec.adv_days
    return {
        "ret_1d": returns,
        "adv": adv,
        "log_adv": np.log(adv.where(adv > 0)),
        "vol": returns.rolling(spec.vol_days, min_periods=spec.vol_min_obs).std(),
        "momentum": closes / closes.shift(spec.momentum_days) - 1,
        "max_ret": returns.rolling(spec.vol_days, min_periods=spec.vol_min_obs).max(),
        "downside_beta": downside_beta(returns, market, spec.beta_days, spec.beta_min_down_days, spec.beta_min_days),
        "coskew": coskewness(returns, market, spec.beta_days, spec.beta_min_days),
    }


def lookup(frame, days, symbols):
    """frame[day, symbol] for each pair; NaN where either is missing."""
    rows = frame.index.get_indexer(np.asarray(days))
    cols = frame.columns.get_indexer(np.asarray(symbols))
    out = np.full(len(rows), np.nan)
    found = (rows >= 0) & (cols >= 0)
    out[found] = frame.to_numpy(dtype=float)[rows[found], cols[found]]
    return out
