# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Replication tests (Hyperliquid, daily bars) — the decision panel never sees a candle that hasn't closed or
funding not yet published, eligibility follows A3, settlements land on the right position, and a delisted
coin's last day takes the haircut.

Run:  pytest tests/ -q
"""
import numpy as np
import pandas as pd
import pytest

from rvr.config import HOUR_MS
from rvr.features import DAY_MS, day_range
from rvr.replication import contract_days, panel, replication_verdict, run_lengths
from rvr.spec import Spec

SPEC = Spec(min_eligible=2, vol_min_obs=3, vol_days=5, beta_min_days=3, beta_min_down_days=2, momentum_days=5,
            min_history_hours=5 * 24)
START = 1_685_577_600_000  # 2023-06-01
DAYS = day_range(START, START + 59 * DAY_MS)
COINS = ["BTC", "AAA", "BBB", "CCC"]


def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for coin in COINS:
        price = 100.0
        for d in day_range(START - 10 * DAY_MS, DAYS[-1]):
            o = price
            price *= np.exp(rng.normal(0, 0.03))
            rows.append((coin, int(d), int(d) + DAY_MS - 1, o, price, 1e7 * rng.uniform(0.5, 1.5), 1000))
    candles = pd.DataFrame(rows, columns=["symbol", "open_time", "close_time", "open", "close", "quote_volume", "trades"])
    t = np.arange(START - 10 * DAY_MS, DAYS[-1] + DAY_MS, HOUR_MS)
    funding = pd.concat([pd.DataFrame({"symbol": c, "calc_time": t + 37, "rate": rng.normal(0, 1e-4, len(t))})
                         for c in COINS], ignore_index=True)
    funding["settle_hour"] = (funding["calc_time"] + HOUR_MS // 2) // HOUR_MS * HOUR_MS
    return candles, funding


def test_panel_rows_never_change_when_the_future_is_corrupted():
    candles, funding = synthetic()
    clean, _ = panel(candles, funding, DAYS, SPEC, floor=1e6)
    cutoff = int(DAYS[30])
    bad_c, bad_f = candles.copy(), funding.copy()
    future_c, future_f = bad_c["open_time"] >= cutoff, bad_f["calc_time"] >= cutoff
    bad_c.loc[future_c, ["open", "close"]] *= 50.0
    bad_c.loc[future_c, "quote_volume"] = 1.0
    bad_f.loc[future_f, "rate"] = 0.5
    dirty, _ = panel(bad_c, bad_f, DAYS, SPEC, floor=1e6)
    upto = clean["day"] <= cutoff
    pd.testing.assert_frame_equal(clean[upto].reset_index(drop=True), dirty[upto].reset_index(drop=True))
    assert not clean[~upto].equals(dirty[~upto])  # the corruption is visible later, so the test can fail


def test_signal_is_the_average_hourly_rate_over_the_previous_three_days():
    candles, funding = synthetic()
    p, _ = panel(candles, funding, DAYS, SPEC, floor=1e6)
    d = int(DAYS[20])
    window = funding[(funding["symbol"] == "AAA") & (funding["calc_time"] < d) & (funding["calc_time"] >= d - 72 * HOUR_MS)]
    row = p[(p["day"] == d) & (p["symbol"] == "AAA")].iloc[0]
    assert row["carry_72h"] == pytest.approx(window["rate"].sum() / 72)


def test_eligibility_needs_history_liquidity_and_yesterday_trading():
    candles, funding = synthetic()
    candles.loc[(candles["symbol"] == "CCC") & (candles["open_time"] == int(DAYS[25])), "trades"] = 0
    p, _ = panel(candles, funding, DAYS, SPEC, floor=1e6)
    ccc = p[p["symbol"] == "CCC"].set_index("day")["eligible"]
    assert ccc[int(DAYS[24])] and not ccc[int(DAYS[26])] and ccc[int(DAYS[27])]
    assert not panel(candles, funding, DAYS, SPEC, floor=1e9)[0]["eligible"].any()


def test_run_lengths_restart_after_a_long_gap():
    traded = pd.DataFrame({"x": [True] * 3 + [False] * 9 + [True] * 2})
    assert run_lengths(traded, restart_gap_days=7)["x"].tolist() == [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 2]


def test_settlements_go_to_the_position_held_and_delisted_last_day_is_halted():
    candles, funding = synthetic()
    d = int(DAYS[10])
    last = candles[candles["symbol"] == "BBB"]["open_time"].max()
    out = contract_days(candles, funding, pd.DataFrame({"day": [d, last], "symbol": ["AAA", "BBB"]}), delisted={"BBB"})
    aaa = out[out["symbol"] == "AAA"].iloc[0]
    held = funding[(funding["symbol"] == "AAA") & (funding["settle_hour"] > d) & (funding["settle_hour"] <= d + DAY_MS)]
    assert len(held) == 24 and aaa["funding_long"] == pytest.approx(-held["rate"].sum())
    assert out[out["symbol"] == "BBB"]["halted"].iloc[0] and not aaa["halted"]


def test_replication_verdict():
    assert replication_verdict([0.1, 2.0], {"mean": -1.0, "newey_west_t": -2.5}, -1) == "Replicates"
    assert replication_verdict([-0.1, 2.0], {"mean": -1.0, "newey_west_t": -2.5}, -1) == "Partly replicates"
    assert replication_verdict([-0.1, 2.0], {"mean": 1.0, "newey_west_t": 2.5}, -1) == "Does not replicate"
