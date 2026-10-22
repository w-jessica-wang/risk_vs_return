# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Execution and P&L tests — every money-moving rule in PREREGISTRATION.md §4–5 against hand-computed
answers: funding signs, which settlements a position earns, drift, halts, delayed fills, unfilled
orders, spread tiers and square-root impact.

Run:  pytest tests/ -q
"""
import numpy as np
import pandas as pd
import pytest

from rvr.backtest import contract_days, half_spread, pnl
from rvr.config import HOUR_MS as H
from rvr.features import DAY_MS
from rvr.spec import Spec

D = 1_704_067_200_000  # 2024-01-01 00:00 UTC: positions run 01:00 → next day 01:00
SPEC = Spec(fee=0.001, spread_tiers=((0.0, 0.0),), impact_coefficient=0.0)
NO_CHARS = {"adv": pd.DataFrame(), "vol": pd.DataFrame()}


def bars(symbol, closes, traded=lambda hour: True):
    """closes: {hour (from D) at which the bar ENDS: close}."""
    hours = sorted(closes)
    end = D + np.array(hours, dtype="int64") * H
    return pd.DataFrame({"symbol": symbol, "open_time": end - H, "close_time": end - 1,
                         "close": [float(closes[h]) for h in hours],
                         "quote_volume": [1000.0 if traded(h) else 0.0 for h in hours]})


def settlements(symbol, rates):
    return pd.DataFrame({"symbol": symbol, "settle_hour": pd.Series([D + h * H for h in rates], dtype="int64"),
                         "rate": pd.Series(list(rates.values()), dtype="float64")})


def run(prices, funding, weights, days, spec=SPEC, chars=NO_CHARS):
    weights = pd.DataFrame(weights, columns=["day", "symbol", "w"])
    cdays = contract_days(prices, funding, weights[["day", "symbol"]], spec)
    return cdays, pnl(weights, cdays, chars, pd.Index(days), spec)


# AAA: 100 until the bar ending at hour 20, then 110. Settlements at 1h (belongs to the previous
# position), 9h, 17h, 25h (= this position's exit, included) and 26h (the next position's).
AAA = bars("AAAUSDT", {h: 100.0 if h <= 20 else 110.0 for h in range(-2, 61)})
AAA_FUNDING = settlements("AAAUSDT", {1: 0.05, 9: 0.001, 17: 0.002, 25: 0.003, 26: 0.004})


def test_settlements_in_the_holding_window_are_scaled_by_notional():
    cdays, _ = run(AAA, AAA_FUNDING, [(D, "AAAUSDT", 0.5)], [D])
    # −(0.001·100 + 0.002·100 + 0.003·110) / 100
    assert cdays["funding_long"].iloc[0] == pytest.approx(-0.0063)
    assert cdays["price_factor"].iloc[0] == pytest.approx(1.1)


def test_long_earns_price_and_pays_positive_funding():
    _, daily = run(AAA, AAA_FUNDING, [(D, "AAAUSDT", 0.5)], [D])
    row = daily.loc[D]
    assert row["price"] == pytest.approx(0.05)
    assert row["funding"] == pytest.approx(-0.00315)
    assert row["cost"] == pytest.approx(0.0005)
    assert row["net"] == pytest.approx(0.04635)


def test_short_receives_positive_funding():
    _, daily = run(AAA, AAA_FUNDING, [(D, "AAAUSDT", -0.5)], [D])
    row = daily.loc[D]
    assert row["price"] == pytest.approx(-0.05)
    assert row["funding"] == pytest.approx(0.00315)
    assert row["net"] == pytest.approx(-0.05 + 0.00315 - 0.0005)


def test_costs_follow_drift_back_to_target():
    _, daily = run(AAA, AAA_FUNDING, [(D, "AAAUSDT", 0.5), (D + DAY_MS, "AAAUSDT", 0.5)], [D, D + DAY_MS])
    drifted = 0.5 * 1.1 / (1 + 0.04635)
    assert daily.loc[D + DAY_MS, "turnover"] == pytest.approx(abs(0.5 - drifted))
    assert daily.loc[D + DAY_MS, "cost"] == pytest.approx(0.001 * abs(0.5 - drifted))
    assert daily.loc[D + DAY_MS, "funding"] == pytest.approx(-0.5 * 0.004)  # the 26h settlement


# HHH crashes to 50 in the bar ending at 10h, then only zero-volume bars: a halt like LUNA's.
HHH = bars("HHHUSDT", {h: (100.0 if h <= 9 else 50.0) for h in range(-2, 300)}, traded=lambda h: h <= 10)
HHH_FUNDING = settlements("HHHUSDT", {9: 0.001, 17: 0.002})


def test_halted_short_exits_at_the_last_trade_with_an_adverse_haircut():
    cdays, daily = run(HHH, HHH_FUNDING, [(D, "HHHUSDT", -0.5)], [D])
    assert bool(cdays["halted"].iloc[0])
    row = daily.loc[D]
    assert row["price"] == pytest.approx(-0.5 * (50 * 1.2 / 100 - 1))  # +0.2
    assert row["funding"] == pytest.approx(0.0005)  # 9h settlement only: the 17h one came after the halt
    assert row["cost"] == pytest.approx(0.0005 + 0.3 * 0.001)  # entry + exit of the remaining 0.3
    assert row["halts"] == 1


def test_halted_long_takes_the_haircut_the_other_way():
    _, daily = run(HHH, HHH_FUNDING, [(D, "HHHUSDT", 0.5)], [D])
    assert daily.loc[D, "price"] == pytest.approx(0.5 * (50 * 0.8 / 100 - 1))  # −0.3


def test_exit_waits_for_trading_to_resume_after_a_gap():
    gap = bars("GAPUSDT", {h: (100.0 if h <= 20 else 120.0) for h in [*range(-2, 21), *range(28, 60)]})
    cdays, daily = run(gap, settlements("GAPUSDT", {}), [(D, "GAPUSDT", 0.5)], [D])
    assert cdays["exit_kind"].iloc[0] == "delayed"
    assert not bool(cdays["halted"].iloc[0])
    assert daily.loc[D, "price"] == pytest.approx(0.5 * 0.2)


def test_an_order_with_no_tradable_price_never_fills():
    dead = bars("DEADUSDT", {h: 100.0 for h in range(-30, -4)})
    _, daily = run(dead, settlements("DEADUSDT", {}), [(D, "DEADUSDT", 0.5)], [D])
    assert daily.loc[D, ["price", "funding", "cost", "turnover"]].abs().sum() == 0


def test_half_spread_tiers_and_fallback():
    spec = Spec()
    assert half_spread(np.array([2e9, 5e8, 5e7, 1e7, np.nan]), spec).tolist() == pytest.approx(
        [0.5e-4, 2e-4, 5e-4, 10e-4, 10e-4])


def test_square_root_impact():
    spec = Spec(fee=0.0, spread_tiers=((0.0, 0.0),), impact_coefficient=1.0, aum=1e6)
    chars = {"adv": pd.DataFrame([[1e8]], index=[D], columns=["AAAUSDT"]),
             "vol": pd.DataFrame([[0.04]], index=[D], columns=["AAAUSDT"])}
    _, daily = run(AAA, AAA_FUNDING, [(D, "AAAUSDT", 0.5)], [D], spec=spec, chars=chars)
    assert daily.loc[D, "cost"] == pytest.approx(0.5 * 0.04 * np.sqrt(0.5 * 1e6 / 1e8))


def test_funding_is_valued_on_the_mark_price_with_a_traded_close_fallback():
    # settlements at hours 9, 17 and 25 (relative to D); mark bars end at 9 and 25 only
    mark = pd.DataFrame({"symbol": "AAAUSDT", "close_time": pd.Series([D + 9 * H - 1, D + 25 * H - 1], dtype="int64"),
                         "mark": [102.0, 108.0]})
    weights = pd.DataFrame([(D, "AAAUSDT", 0.5)], columns=["day", "symbol", "w"])
    cdays = contract_days(AAA, AAA_FUNDING, weights[["day", "symbol"]], SPEC, mark=mark)
    # −(0.001·102 + 0.002·100 [traded close, no mark bar] + 0.003·108) / 100
    assert cdays["funding_long"].iloc[0] == pytest.approx(-(0.102 + 0.2 + 0.324) / 100)
    assert cdays["mark_fallbacks"].iloc[0] == 1


def test_a_delayed_exit_earns_funding_until_it_actually_exits_and_nothing_is_counted_twice():
    # no trades between the bar ending at hour 20 and the bar ending at hour 28; the day-D exit (hour 25) waits
    # until hour 28, and the day-(D+1) entry (hour 25) fills on the same bar
    gap = bars("GAPUSDT", {h: (100.0 if h <= 20 else 120.0) for h in [*range(-2, 21), *range(28, 60)]})
    rates = settlements("GAPUSDT", {9: 0.001, 25: 0.002, 27: 0.004, 29: 0.008})
    cdays, _ = run(gap, rates, [(D, "GAPUSDT", 0.5), (D + DAY_MS, "GAPUSDT", 0.5)], [D, D + DAY_MS])
    first, second = cdays.iloc[0], cdays.iloc[1]
    assert first["exit_kind"] == "delayed" and second["entry_kind"] == "delayed"
    # day D: settlements at 9, 25 and 27 (during the wait), valued at the last traded close of 100
    assert first["funding_long"] == pytest.approx(-(0.001 + 0.002 + 0.004))
    # day D+1 starts at the delayed fill (hour 28): only the settlement at 29, valued at 120 against entry 120
    assert second["funding_long"] == pytest.approx(-0.008)
