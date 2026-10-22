# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Point-in-time lookups — the only way strategy code is allowed to read market data.

Each lookup answers "what could a trader have known at decision_time?" and nothing more:
  - a funding rate is known strictly after its published calc_time (jitter included)
  - a price is known strictly after its bar closes, and only bars with volume count:
    Binance keeps archiving flat, zero-volume bars for months after a contract delists
  - a contract is tradable while it keeps trading and once its current run of trading is
    long enough, judged from past bars alone — a future delisting never removes it early,
    and a contract that stops and later restarts must rebuild its history

Every join is merge_asof(direction="backward", allow_exact_matches=False).
tests/test_leakage.py proves no lookup changes when every row at or after the decision
time is corrupted.
"""
import numpy as np
import pandas as pd

from rvr.config import HOUR_MS


def _require_epoch_ms(frame, column):
    if not pd.api.types.is_integer_dtype(frame[column]):
        raise TypeError(f"{column} must be int64 epoch milliseconds, got {frame[column].dtype}")


def asof_strictly_before(left, right, left_on, right_on, by="symbol", tolerance_ms=None):
    """For each left row (original order kept), attach the latest right row with right_on < left_on."""
    _require_epoch_ms(left, left_on)
    _require_epoch_ms(right, right_on)
    left = left.assign(_row=np.arange(len(left)))
    merged = pd.merge_asof(
        left.sort_values(left_on, kind="mergesort"),
        right.sort_values(right_on, kind="mergesort"),
        left_on=left_on, right_on=right_on, by=by,
        direction="backward", allow_exact_matches=False,
        tolerance=None if tolerance_ms is None else int(tolerance_ms),
    )
    return merged.sort_values("_row").drop(columns="_row").reset_index(drop=True)


def decision_grid(symbols, start, end, every_hours, offset_minutes=0):
    """(symbol, decision_time) for every symbol on a fixed UTC schedule in [start, end)."""
    times = (pd.date_range(start, end, freq=f"{every_hours}h", tz="UTC", inclusive="left")
             + pd.Timedelta(minutes=offset_minutes))
    epoch_ms = ((times - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)).astype("int64")
    index = pd.MultiIndex.from_product([sorted(symbols), epoch_ms], names=["symbol", "decision_time"])
    return index.to_frame(index=False)


def traded_bars(prices):
    """Bars with actual trading. Flat zero-volume bars are not prices anyone could trade at."""
    return prices[prices["quote_volume"] > 0]


def latest_funding(decisions, funding, max_age_hours=None):
    """Most recent funding rate published strictly before each decision_time."""
    right = funding[["symbol", "calc_time", "interval_h", "rate", "rate_per_hour"]]
    tolerance = None if max_age_hours is None else max_age_hours * HOUR_MS
    out = asof_strictly_before(decisions, right, "decision_time", "calc_time", tolerance_ms=tolerance)
    return out.rename(columns={"calc_time": "funding_known_at"})


def latest_close(decisions, prices, max_staleness_hours):
    """Close of the latest traded bar that closed strictly before decision_time, within max_staleness_hours."""
    right = traded_bars(prices)[["symbol", "close_time", "close"]]
    out = asof_strictly_before(decisions, right, "decision_time", "close_time",
                               tolerance_ms=max_staleness_hours * HOUR_MS)
    return out.rename(columns={"close_time": "price_known_at"})


def _trailing_sum(decisions, events, time_col, value_col, window_hours):
    """Sum of value_col over events with time_col in [decision_time − window, decision_time)."""
    events = events[["symbol", time_col, value_col]].sort_values(["symbol", time_col], kind="mergesort")
    cumulative = events.assign(_cum=events.groupby("symbol")[value_col].cumsum()).drop(columns=value_col)
    base = decisions[["symbol", "decision_time"]].assign(
        window_start=decisions["decision_time"].to_numpy() - int(window_hours * HOUR_MS))
    now = asof_strictly_before(base, cumulative, "decision_time", time_col)["_cum"].fillna(0.0)
    then = asof_strictly_before(base, cumulative, "window_start", time_col)["_cum"].fillna(0.0)
    return (now - then).to_numpy()


def trailing_quote_volume(decisions, prices, window_hours):
    """USDT volume of bars that closed in [decision_time − window, decision_time)."""
    total = _trailing_sum(decisions, prices, "close_time", "quote_volume", window_hours)
    return decisions.assign(**{f"quote_volume_{window_hours}h": total})


def trailing_funding(decisions, funding, window_hours):
    """Average hourly funding rate: rates published in [decision_time − window, decision_time), ÷ window.

    Summing per-settlement rates gives the total cost of holding a long over the window whatever the
    contract's schedule, so 8h, 4h and 1h contracts are directly comparable."""
    total = _trailing_sum(decisions, funding, "calc_time", "rate", window_hours)
    return decisions.assign(**{f"carry_{window_hours}h": total / window_hours})


def trading_runs(prices, restart_gap_hours):
    """
    Traded bars, each tagged with run_start: the open_time that began its current run of trading.

    A run restarts after more than restart_gap_hours without a traded bar, so a contract that
    is delisted and relisted (or goes dormant and revives) starts a new run. run_start depends
    only on the bar itself and earlier bars.
    """
    bars = (traded_bars(prices)[["symbol", "open_time", "close_time"]]
            .sort_values(["symbol", "open_time"], kind="mergesort").reset_index(drop=True))
    gap = bars.groupby("symbol")["open_time"].diff()
    new_run = gap.isna() | (gap > restart_gap_hours * HOUR_MS)
    run_start = bars["open_time"].where(new_run).groupby(bars["symbol"]).ffill()
    return bars.assign(run_start=run_start.astype("int64"))


def tradable_universe(decisions, prices, min_history_hours, max_staleness_hours, restart_gap_hours):
    """
    Flag which (symbol, decision_time) pairs were tradable, from traded bars closed before then.

    Eligible when the latest traded bar closed within max_staleness_hours and the current run of
    trading has lasted at least min_history_hours. run_start exists only once a run's first bar
    has closed, so a contract's future listing date never appears early.
    """
    runs = (trading_runs(prices, restart_gap_hours)[["symbol", "close_time", "run_start"]]
            .rename(columns={"close_time": "last_trade_close_time"}))
    out = asof_strictly_before(decisions, runs, "decision_time", "last_trade_close_time")
    run_hours = (out["decision_time"] - out["run_start"]) / HOUR_MS
    staleness_hours = (out["decision_time"] - out["last_trade_close_time"]) / HOUR_MS
    return out.assign(run_age_hours=run_hours,
                      eligible=(run_hours >= min_history_hours) & (staleness_hours <= max_staleness_hours))
