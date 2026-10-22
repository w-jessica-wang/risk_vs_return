# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Feature tests — daily rows line up with what a 00:05 UTC decision could know, the beta and
coskewness estimators match their definitions, and no row moves when the future is corrupted.

Run:  pytest tests/ -q
"""
import numpy as np
import pandas as pd
import pytest

from rvr.config import HOUR_MS as H
from rvr.features import DAY_MS, characteristics, coskewness, daily_closes, daily_quote_volume, day_range, downside_beta
from rvr.spec import Spec

D0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC
SPEC = Spec(vol_days=10, vol_min_obs=5, beta_days=20, beta_min_down_days=3, beta_min_days=5,
            momentum_days=5, adv_days=3)


def hourly(symbol, closes, volume=1.0):
    open_time = D0 + np.arange(len(closes), dtype="int64") * H
    return pd.DataFrame({"symbol": symbol, "open_time": open_time, "close_time": open_time + H - 1,
                         "close": np.asarray(closes, dtype=float), "quote_volume": volume})


def test_daily_close_is_the_bar_ending_at_midnight():
    prices = hourly("AAAUSDT", np.arange(72))
    closes = daily_closes(prices, day_range(D0, D0 + 2 * DAY_MS))["AAAUSDT"]
    assert np.isnan(closes.iloc[0])  # nothing has closed by 2024-01-01 00:00
    assert closes.iloc[1:].tolist() == [23.0, 47.0]


def test_zero_volume_bar_gives_no_daily_close():
    prices = hourly("AAAUSDT", np.arange(72))
    prices.loc[prices["close"] == 47, "quote_volume"] = 0.0
    assert np.isnan(daily_closes(prices, day_range(D0, D0 + 2 * DAY_MS))["AAAUSDT"].iloc[2])


def test_daily_volume_counts_the_24_bars_ending_at_midnight():
    volume = daily_quote_volume(hourly("AAAUSDT", np.arange(72)), day_range(D0, D0 + 2 * DAY_MS))["AAAUSDT"]
    assert volume.tolist() == [0.0, 24.0, 24.0]


def test_downside_beta_uses_only_down_days():
    rng = np.random.default_rng(0)
    market = pd.Series(rng.normal(0, 0.02, 60))
    returns = pd.DataFrame({"X": np.where(market < 0, 2.0 * market, 0.5 * market)})
    beta = downside_beta(returns, market, window=90, min_down_days=10, min_days=20)
    assert beta["X"].iloc[-1] == pytest.approx(2.0)


def test_coskewness_matches_its_definition():
    rng = np.random.default_rng(1)
    market = pd.Series(rng.normal(0, 0.02, 50))
    x = 0.5 * market + 3 * market**2 + rng.normal(0, 0.01, 50)
    got = coskewness(pd.DataFrame({"X": x}), market, window=50, min_obs=20)["X"].iloc[-1]
    a, b = x - x.mean(), market - market.mean()
    expected = np.mean(a * b**2) / (np.sqrt(np.mean(a**2)) * np.mean(b**2))
    assert got == pytest.approx(expected)


def test_characteristics_ignore_everything_after_the_cutoff():
    rng = np.random.default_rng(2)
    n = 80 * 24
    prices = pd.concat([hourly("BTCUSDT", 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), 5.0),
                        hourly("AAAUSDT", 10 * np.exp(np.cumsum(rng.normal(0, 0.02, n))), 2.0)], ignore_index=True)
    cutoff = D0 + 60 * DAY_MS
    dirty = prices.copy()
    future = dirty["close_time"] >= cutoff
    dirty.loc[future, "close"] *= 3.0
    dirty.loc[future, "quote_volume"] *= 10.0
    dirty = dirty[~(future & (dirty["open_time"] % (7 * H) == 0))]
    days = day_range(D0, D0 + 79 * DAY_MS)
    clean, corrupted = characteristics(prices, days, SPEC), characteristics(dirty, days, SPEC)
    for name in clean:
        pd.testing.assert_frame_equal(clean[name].loc[:cutoff], corrupted[name].loc[:cutoff], obj=name)
    assert not clean["ret_1d"].loc[cutoff + DAY_MS:].equals(corrupted["ret_1d"].loc[cutoff + DAY_MS:])
