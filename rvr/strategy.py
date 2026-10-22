# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
The decision panel and every pre-registered strategy's weights (PREREGISTRATION.md §2, §3, §6).

decision_panel() is the only place eligibility and signals are assembled, entirely from rvr.pit
lookups and rvr.features rows known by 00:05 UTC. The weight functions never see a return.
"""
import numpy as np
import pandas as pd

from rvr.config import HOUR_MS
from rvr.features import day_range, lookup
from rvr.pit import latest_funding, tradable_universe, trailing_funding
from rvr.universe import is_crypto

PANEL_CHARACTERISTICS = ["adv", "log_adv", "vol", "momentum", "max_ret", "downside_beta", "coskew"]


def decision_panel(prices, funding, chars, spec, start, end):
    """One row per (day, crypto contract) with eligibility, carry signals and characteristics."""
    days = day_range(start, end)
    first_close = prices.groupby("symbol")["close_time"].min()
    symbols = sorted(s for s in first_close.index[first_close < days[-1]] if is_crypto(s))
    grid = pd.MultiIndex.from_product([days, symbols], names=["day", "symbol"]).to_frame(index=False)
    grid["decision_time"] = grid["day"] + spec.decision_offset_minutes * 60_000
    decisions = grid[["symbol", "decision_time"]]

    trading = tradable_universe(decisions, prices, spec.min_history_hours, spec.max_staleness_hours,
                                spec.restart_gap_hours)
    latest = latest_funding(decisions, funding)
    fresh = (decisions["decision_time"] - latest["funding_known_at"]
             <= spec.fresh_funding_intervals * latest["interval_h"] * HOUR_MS)

    panel = grid.assign(trading=trading["eligible"].to_numpy(), fresh_funding=fresh.to_numpy())
    windows = {spec.signal_hours, *spec.signal_variants.values()}
    for hours in sorted(windows):
        panel[f"carry_{hours}h"] = trailing_funding(decisions, funding, hours)[f"carry_{hours}h"].to_numpy()
    for name in PANEL_CHARACTERISTICS:
        panel[name] = lookup(chars[name], panel["day"], panel["symbol"])
    panel["eligible"] = (panel["trading"] & panel["fresh_funding"] & (panel["adv"] >= spec.min_adv)
                         & panel[f"carry_{spec.signal_hours}h"].notna())
    return panel


def quintile_weights(panel, signal, spec, candidates=None):
    """
    Short the top quintile of `signal`, long the bottom, equal weight, gross 1 (§3).

    `candidates` (default: eligible) are the contracts ranked; the 20-contract minimum always applies to
    the eligible count (§2), so a filter can thin the ranking without changing which days trade.
    """
    candidates = panel["eligible"] if candidates is None else candidates
    breadth = panel[panel["eligible"]].groupby("day").size()
    p = panel.loc[candidates & panel["eligible"], ["day", "symbol", signal]].sort_values(["day", "symbol"])
    p = p[p["day"].map(breadth).fillna(0) >= spec.min_eligible]
    group = p.groupby("day")[signal]
    size = group.transform("size")
    rank = group.rank(method="first")
    n = np.floor(size * spec.quantile)
    w = np.where(rank <= n, 0.5 / n, np.where(rank > size - n, -0.5 / n, 0.0))
    out = p[["day", "symbol"]].assign(w=np.where(n > 0, w, 0.0))
    return out[out["w"] != 0].reset_index(drop=True)


def low_volatility_candidates(panel, spec):
    """Eligible contracts minus the most volatile quintile; contracts without a volatility estimate drop too (§6.1)."""
    p = panel[panel["eligible"] & panel["vol"].notna()]
    rank = p.groupby("day")["vol"].rank(method="first", ascending=False)
    n_drop = np.floor(p.groupby("day")["vol"].transform("size") * spec.vol_filter_quantile)
    keep = pd.Series(False, index=panel.index)
    keep[p.index[rank > n_drop]] = True
    return keep


def add_downside_beta_hedge(weights, panel, spec):
    """Offset the portfolio's downside beta with the hedge contract (§6.2)."""
    beta = weights.merge(panel[["day", "symbol", "downside_beta"]], on=["day", "symbol"], how="left")
    exposure = (beta["w"] * beta["downside_beta"].fillna(1.0)).groupby(beta["day"]).sum()
    hedge = pd.DataFrame({"day": exposure.index, "symbol": spec.hedge_symbol, "w": -exposure.to_numpy()})
    combined = pd.concat([weights, hedge], ignore_index=True)
    return combined.groupby(["day", "symbol"], as_index=False)["w"].sum()


def volatility_scale(unscaled_returns, spec):
    """
    Scale per day so the book targets spec.vol_target (§6.3). Day d's return is only complete at
    d+1 01:00, after the d 00:05 decision, so day d uses returns through day d − 2.
    """
    realised = unscaled_returns.rolling(spec.vol_target_days, min_periods=spec.vol_target_min_obs).std() * np.sqrt(365)
    scale = (spec.vol_target / realised).clip(upper=spec.max_scale).shift(2)
    return scale.fillna(1.0)


def scale_weights(weights, scale):
    return weights.assign(w=weights["w"] * weights["day"].map(scale).fillna(1.0).to_numpy())


def buffered_weights(panel, signal, spec, exit_quantile=0.4):
    """
    B with a no-trade buffer (Amendment A3): enter in the extreme quintiles as B does, but keep a position
    while the contract stays eligible and inside the extreme `exit_quantile` of the signal. Equal weight per
    side, each side gross 0.5. Path-dependent, so built day by day from yesterday's holdings only.
    """
    breadth = panel[panel["eligible"]].groupby("day").size()
    p = panel.loc[panel["eligible"], ["day", "symbol", signal]].dropna(subset=[signal])
    rows, longs, shorts = [], set(), set()
    for day, g in p.groupby("day", sort=True):
        if breadth.get(day, 0) < spec.min_eligible:
            longs, shorts = set(), set()
            continue
        g = g.sort_values([signal, "symbol"], kind="mergesort")
        size = len(g)
        rank = pd.Series(np.arange(1, size + 1), index=g["symbol"].to_numpy())
        n_enter, n_exit = int(np.floor(size * spec.quantile)), int(np.floor(size * exit_quantile))
        if n_enter == 0:
            longs, shorts = set(), set()
            continue
        keep_long = {s for s in longs if s in rank.index and rank[s] <= n_exit}
        keep_short = {s for s in shorts if s in rank.index and rank[s] > size - n_exit}
        longs = keep_long | set(rank.index[rank <= n_enter])
        shorts = keep_short | set(rank.index[rank > size - n_enter])
        rows += [(day, s, 0.5 / len(longs)) for s in sorted(longs)]
        rows += [(day, s, -0.5 / len(shorts)) for s in sorted(shorts)]
    return pd.DataFrame(rows, columns=["day", "symbol", "w"])


def band_weights(panel, signal, spec, inner=0.1, outer=0.3):
    """
    Carry without the extreme tails (Amendment A5c, "B_core"): long contracts ranked between the `inner` and
    `outer` quantiles of the signal from the bottom, short the mirror band at the top; equal weight, gross 1.
    """
    breadth = panel[panel["eligible"]].groupby("day").size()
    p = panel.loc[panel["eligible"], ["day", "symbol", signal]].sort_values(["day", "symbol"])
    p = p[p["day"].map(breadth).fillna(0) >= spec.min_eligible]
    group = p.groupby("day")[signal]
    size, rank = group.transform("size"), group.rank(method="first")
    lo, hi = np.floor(size * inner), np.floor(size * outer)
    n = hi - lo
    long_leg, short_leg = (rank > lo) & (rank <= hi), (rank > size - hi) & (rank <= size - lo)
    w = np.where(long_leg, 0.5 / n, np.where(short_leg, -0.5 / n, 0.0))
    out = p[["day", "symbol"]].assign(w=np.where(n > 0, w, 0.0))
    return out[out["w"] != 0].reset_index(drop=True)
