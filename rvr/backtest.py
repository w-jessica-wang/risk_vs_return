# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Execution and P&L (PREREGISTRATION.md §4–5).

contract_days() prices every (day, contract) a strategy might hold — entry, exit, halt, funding —
exactly once. pnl() turns any weight table into daily returns net of costs. Strategies differ only
in their weights, so every variant is costed by identical machinery.

A position opened on day d is held from t0 = d 01:00 to t1 = d+1 01:00 UTC.
"""
import numpy as np
import pandas as pd

from rvr.config import HOUR_MS
from rvr.features import DAY_MS, lookup
from rvr.pit import traded_bars


def execution_prices(traded, queries, spec):
    """
    For each (symbol, t): the latest traded close in (t − window, t] ("on_time"); else the first
    traded close in (t, t + max delay] ("delayed" — the order waits for trading to resume); else
    NaN ("halted"). Also returns when the fill happened (t, or the end of the delayed bar) and the last
    traded close at or before t, for halt exits.
    """
    bars = traded[["symbol", "close_time", "close"]].sort_values("close_time", kind="mergesort")
    q = queries[["symbol", "t"]].assign(_row=np.arange(len(queries))).sort_values("t", kind="mergesort")
    def asof(**kw):
        return pd.merge_asof(q, bars, left_on="t", right_on="close_time", by="symbol", **kw)

    on_time = asof(direction="backward", tolerance=spec.execution_window_hours * HOUR_MS - 1)
    delayed = asof(direction="forward", allow_exact_matches=False, tolerance=spec.max_execution_delay_hours * HOUR_MS)
    last = asof(direction="backward")
    out = pd.DataFrame({
        "_row": q["_row"].to_numpy(),
        "price": on_time["close"].fillna(delayed["close"]).to_numpy(),
        "kind": np.where(on_time["close"].notna(), "on_time", np.where(delayed["close"].notna(), "delayed", "halted")),
        "fill_time": np.where(on_time["close"].notna(), q["t"].to_numpy(), delayed["close_time"].to_numpy() + 1),
        "last_close": last["close"].to_numpy(),
        "last_close_time": last["close_time"].to_numpy(),
    })
    return out.sort_values("_row").drop(columns="_row").reset_index(drop=True)


def contract_days(prices, funding, keys, spec, mark=None):
    """
    One row per (day, symbol) in keys: entry and exit prices, whether the contract halted while held,
    and funding_long, the funding a long of 1.0 would earn (negative when longs pay). A position earns every
    settlement between its actual entry and actual exit (event time): normally (d 01:00, d+1 01:00], longer when
    an exit waits for trading to resume, through the first hour after the last trade when it halts. Its whole
    P&L is booked on day d, the day it was opened. With `mark` (hourly
    mark-price closes, Amendment A7b) each settlement is valued on the mark price instead of the traded close;
    mark_fallbacks counts settlements with no mark bar, valued on the traded close as before.
    """
    traded = traded_bars(prices)
    keys = keys[["day", "symbol"]].drop_duplicates().sort_values(["day", "symbol"]).reset_index(drop=True)
    t0 = keys["day"].to_numpy() + spec.execution_hour * HOUR_MS
    queries = pd.concat([keys.assign(t=t0), keys.assign(t=t0 + DAY_MS)], ignore_index=True)
    prices_at = execution_prices(traded, queries, spec)
    entry, exit_ = prices_at.iloc[:len(keys)].reset_index(drop=True), prices_at.iloc[len(keys):].reset_index(drop=True)

    out = keys.assign(entry=entry["price"], entry_kind=entry["kind"], exit=exit_["price"], exit_kind=exit_["kind"],
                      last_close=exit_["last_close"], last_close_time=exit_["last_close_time"])
    out["halted"] = out["entry"].notna() & out["exit"].isna()
    # funding window (entry_time, exit_time]; a halted position is settled through the first hour after its last trade
    out["entry_time"] = entry["fill_time"].to_numpy(dtype=float)
    out["exit_time"] = np.where(out["halted"], np.ceil((out["last_close_time"] + 1) / HOUR_MS) * HOUR_MS,
                                exit_["fill_time"].to_numpy(dtype=float))
    out["price_factor"] = np.where(out["halted"], out["last_close"], out["exit"]) / out["entry"]
    out["funding_long"], out["mark_fallbacks"] = _funding_long(traded, funding, out, spec, mark)
    return out


def _funding_long(traded, funding, cdays, spec, mark=None):
    held = cdays[cdays["entry"].notna()]
    f = funding.loc[funding["symbol"].isin(held["symbol"].unique()), ["symbol", "settle_hour", "rate"]]
    f = f.astype({"settle_hour": "int64", "rate": "float64"}).sort_values("settle_hour", kind="mergesort")
    # settlement s belongs to the position whose window (entry_time, exit_time] contains it; windows of one
    # contract never overlap, because a delayed exit and the next day's entry fill on the same bar
    windows = held[["day", "symbol", "entry", "entry_time", "exit_time"]].astype({"entry_time": "int64"})
    f = pd.merge_asof(f, windows.sort_values("entry_time", kind="mergesort"), left_on="settle_hour",
                      right_on="entry_time", by="symbol", direction="backward", allow_exact_matches=False)
    f = f[f["day"].notna() & (f["settle_hour"] <= f["exit_time"])].astype({"day": "int64"})
    bars = traded[["symbol", "close_time", "close"]].sort_values("close_time", kind="mergesort")
    f = pd.merge_asof(f, bars, left_on="settle_hour", right_on="close_time", by="symbol", direction="backward")
    price = f["close"].fillna(f["entry"])
    fallback = pd.Series(0, index=f.index)
    if mark is not None:
        # the mark bar ending at the settlement hour: close_time = settle_hour − 1 ms
        on_mark = f[["symbol", "settle_hour"]].merge(
            mark.assign(settle_hour=mark["close_time"] + 1)[["symbol", "settle_hour", "mark"]],
            on=["symbol", "settle_hour"], how="left")["mark"].to_numpy()
        fallback = pd.Series(np.isnan(on_mark).astype(int), index=f.index)
        price = pd.Series(np.where(np.isnan(on_mark), price, on_mark), index=f.index)
    f = f.assign(paid=-f["rate"] * price / f["entry"], fallback=fallback)
    f = f.groupby(["day", "symbol"], as_index=False)[["paid", "fallback"]].sum()
    merged = cdays[["day", "symbol"]].merge(f, on=["day", "symbol"], how="left")
    return merged["paid"].fillna(0.0).to_numpy(), merged["fallback"].fillna(0).astype(int).to_numpy()


def half_spread(adv, spec):
    adv = np.nan_to_num(adv, nan=spec.fallback_adv)
    conditions = [adv >= floor for floor, _ in spec.spread_tiers]
    return np.select(conditions, [rate for _, rate in spec.spread_tiers], default=spec.spread_tiers[-1][1])


def pnl(weights, cdays, chars, days, spec, aum=None, cost_multiple=1.0, cost_model=None):
    """
    Daily returns of a weight table (day, symbol, w) over `days`, net of costs.

    Weights are fractions of capital set at each decision. Costs are charged on the gap between today's
    target and yesterday's positions after they drifted with prices, plus a halt exit at the haircut price.
    cost_model (rvr.costs.CostModel) replaces the §5 costs in spec, e.g. with Amendment A1's calibration.
    """
    aum = spec.aum if aum is None else aum
    w = weights[weights["w"] != 0].merge(cdays, on=["day", "symbol"], how="left")
    w.loc[w["entry"].isna(), "w"] = 0.0  # entry had no execution price: the order never filled
    w["halted"] = w["halted"].fillna(False).astype(bool)

    symbols = pd.Index(sorted(w["symbol"].unique()))
    day_index = pd.Index(days)
    shape = (len(day_index), len(symbols))
    r, c = day_index.get_indexer(w["day"]), symbols.get_indexer(w["symbol"])
    inside = r >= 0
    W, F, FL, HALT = np.zeros(shape), np.ones(shape), np.zeros(shape), np.zeros(shape, dtype=bool)
    factor = np.where(w["halted"], w["price_factor"] * (1 - spec.halt_haircut * np.sign(w["w"])), w["price_factor"])
    W[r[inside], c[inside]] = w["w"].to_numpy()[inside]
    F[r[inside], c[inside]] = np.nan_to_num(factor[inside], nan=1.0)
    FL[r[inside], c[inside]] = w["funding_long"].fillna(0.0).to_numpy()[inside]
    HALT[r[inside], c[inside]] = w["halted"].to_numpy()[inside]

    grid_days, grid_symbols = np.repeat(day_index, len(symbols)), np.tile(symbols, len(day_index))
    adv = np.nan_to_num(lookup(chars["adv"], grid_days, grid_symbols), nan=spec.fallback_adv).reshape(shape)
    adv = np.where(adv > 0, adv, spec.fallback_adv)
    sigma = np.nan_to_num(lookup(chars["vol"], grid_days, grid_symbols), nan=spec.fallback_daily_vol).reshape(shape)
    if cost_model is None:
        static = spec.fee + half_spread(adv, spec)
        def unit_cost(notional_usd, i):
            return static[i] + spec.impact_coefficient * sigma[i] * np.sqrt(notional_usd / adv[i])
    else:
        def unit_cost(notional_usd, i):
            return cost_model.unit_cost(notional_usd, adv[i], sigma[i])

    def trade_cost(notional, i):
        return float(np.sum(notional * unit_cost(notional * aum, i))) * cost_multiple

    carried = np.zeros(len(symbols))
    rows = []
    for i in range(len(day_index)):
        target = W[i]
        traded_notional = np.abs(target - carried)
        price = float(np.sum(target * (F[i] - 1)))
        funding_paid = float(np.sum(target * FL[i]))
        halt_exit = np.abs(target * F[i]) * HALT[i]
        cost = trade_cost(traded_notional, i) + trade_cost(halt_exit, i)
        net = price + funding_paid - cost
        end = target * F[i] / (1 + net)
        end[HALT[i]] = 0.0
        carried = end
        rows.append((day_index[i], price, funding_paid, cost, net, float(traded_notional.sum() + halt_exit.sum()),
                     int((target > 0).sum()), int((target < 0).sum()), float(np.abs(target).sum()), int(HALT[i].sum())))
    return pd.DataFrame(rows, columns=["day", "price", "funding", "cost", "net", "turnover",
                                       "n_long", "n_short", "gross_exposure", "halts"]).set_index("day")
