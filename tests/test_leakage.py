# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Leakage tests — prove every point-in-time lookup in rvr.pit sees only data that existed
strictly before the decision time.

Synthetic contracts with KNOWN answers, built through the real parsers:
  AAAUSDT  trades hours 0–99; funding every 8h, with millisecond publish jitter at 16h and 40h
  NEWUSDT  lists at hour 50
  DELUSDT  trades hours 0–29, then delists — followed, as in Binance's real archives, by flat
           zero-volume "zombie" bars through hour 99
  REVUSDT  trades 0–29, goes dormant (zombie bars) 30–79, revives and trades 80–119

The final test corrupts every row at or after a cutoff (flips and scales funding, triples
prices, zeroes and drops bars, lists a new contract, relists a dead one) and asserts that no
lookup at or before the cutoff changes.

Run:  pytest tests/ -q
"""
import numpy as np
import pandas as pd
import pytest

from rvr.config import HOUR_MS as H
from rvr.parse import KLINE_COLUMNS, parse_funding, parse_klines
from rvr.pit import (decision_grid, latest_close, latest_funding, tradable_universe, trailing_funding,
                     trailing_quote_volume)

T0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC
AAA_FUNDING_HOURS = list(range(8, 100, 8))
AAA_JITTER_MS = [14 if hour in (16, 40) else 0 for hour in AAA_FUNDING_HOURS]
CUTOFF_HOUR = 56  # an AAAUSDT settlement lands exactly on the cutoff
UNIVERSE = dict(min_history_hours=24, max_staleness_hours=2, restart_gap_hours=24)


def make_bars(symbol, first_hour, n, traded=True):
    """Traded bars close at 100 + hour with quote volume 1000 + hour. Zombie bars (traded=False)
    are flat at the previous hour's close with zero volume, like Binance's post-delisting archive."""
    hours = first_hour + np.arange(n)
    open_time = T0 + hours * H
    close = 100.0 + hours if traded else np.full(n, 100.0 + first_hour - 1)
    wick = 1.0 if traded else 0.0
    raw = pd.DataFrame({
        "open_time": open_time, "open": close, "high": close + wick, "low": close - wick, "close": close,
        "volume": 10.0 if traded else 0.0, "close_time": open_time + H - 1,
        "quote_volume": 1000.0 + hours if traded else 0.0, "count": 5 if traded else 0,
        "taker_buy_volume": 5.0 if traded else 0.0, "taker_buy_quote_volume": 500.0 if traded else 0.0,
        "ignore": 0,
    })[KLINE_COLUMNS]
    return parse_klines(raw, symbol)[0]


def make_funding(symbol, hours, jitter_ms=None):
    """8h settlements at the given hours; the k-th settlement's rate is 0.0001 × k."""
    hours = np.asarray(hours, dtype="int64")
    jitter = np.zeros(len(hours), dtype="int64") if jitter_ms is None else np.asarray(jitter_ms, dtype="int64")
    raw = pd.DataFrame({"calc_time": T0 + hours * H + jitter, "funding_interval_hours": 8,
                        "last_funding_rate": 0.0001 * np.arange(1, len(hours) + 1)})
    return parse_funding(raw, symbol)[0]


def at(symbol, *times):
    return pd.DataFrame({"symbol": symbol, "decision_time": np.asarray(times, dtype="int64")})


@pytest.fixture
def market():
    prices = pd.concat([
        make_bars("AAAUSDT", 0, 100),
        make_bars("NEWUSDT", 50, 50),
        make_bars("DELUSDT", 0, 30), make_bars("DELUSDT", 30, 70, traded=False),
        make_bars("REVUSDT", 0, 30), make_bars("REVUSDT", 30, 50, traded=False), make_bars("REVUSDT", 80, 40),
    ], ignore_index=True)
    funding = pd.concat([make_funding("AAAUSDT", AAA_FUNDING_HOURS, AAA_JITTER_MS),
                         make_funding("DELUSDT", [8, 16, 24])], ignore_index=True)
    return prices, funding


# ── Funding ────────────────────────────────────────────────────────────────────

def test_funding_published_exactly_at_decision_time_is_not_visible(market):
    _, funding = market
    out = latest_funding(at("AAAUSDT", T0 + 24 * H, T0 + 24 * H + 1), funding)
    assert out["rate"].tolist() == pytest.approx([0.0002, 0.0003])  # the 16h rate at 24h; 24h rate 1ms later


def test_jittered_funding_is_not_visible_until_actually_published(market):
    """The 16h settlement was published at 16:00:00.014, so it is unknown at 16:00:00.000 and .014."""
    _, funding = market
    out = latest_funding(at("AAAUSDT", T0 + 16 * H, T0 + 16 * H + 14, T0 + 16 * H + 15), funding)
    assert out["rate"].tolist() == pytest.approx([0.0001, 0.0001, 0.0002])
    assert out["funding_known_at"].iloc[2] == T0 + 16 * H + 14


def test_carry_signal_sums_only_settlements_published_inside_the_window(market):
    """At 24h with a 16h window: the 8h and 16h settlements count (16h was published at 16:00:00.014);
    the 24h one, published exactly at the decision, does not."""
    _, funding = market
    out = trailing_funding(at("AAAUSDT", T0 + 24 * H, T0 + 24 * H + 1), funding, window_hours=16)
    assert out["carry_16h"].tolist() == pytest.approx([(0.0001 + 0.0002) / 16, (0.0002 + 0.0003) / 16])


def test_no_funding_is_known_before_the_first_settlement(market):
    _, funding = market
    assert latest_funding(at("AAAUSDT", T0 + 8 * H), funding)["rate"].isna().all()


# ── Prices ─────────────────────────────────────────────────────────────────────

def test_bar_is_visible_only_after_it_closes(market):
    prices, _ = market
    out = latest_close(at("AAAUSDT", T0 + 10 * H - 1, T0 + 10 * H), prices, max_staleness_hours=2)
    assert out["close"].tolist() == [108.0, 109.0]  # at bar 9's own close_time, only bar 8 is known


def test_prices_go_stale_after_delisting_despite_zombie_bars(market):
    """At 31h the newest bar is a zombie (closed 31h−1ms); the price must come from the last traded bar."""
    prices, _ = market
    out = latest_close(at("DELUSDT", T0 + 31 * H, T0 + 33 * H, T0 + 60 * H), prices, max_staleness_hours=2)
    assert out["close"].iloc[0] == 129.0
    assert out["price_known_at"].iloc[0] == T0 + 30 * H - 1
    assert out["close"].iloc[1:].isna().all()


def test_trailing_volume_counts_only_bars_closed_inside_the_window(market):
    prices, _ = market
    out = trailing_quote_volume(at("AAAUSDT", T0 + 10 * H, T0 + 2 * H), prices, window_hours=3)
    assert out["quote_volume_3h"].tolist() == pytest.approx([1007 + 1008 + 1009, 1000 + 1001])


# ── Universe ───────────────────────────────────────────────────────────────────

def test_contract_not_yet_listed_is_ineligible_and_reveals_no_listing_date(market):
    prices, _ = market
    out = tradable_universe(at("NEWUSDT", T0 + 40 * H, T0 + 50 * H), prices,
                            min_history_hours=0, max_staleness_hours=2, restart_gap_hours=24)
    assert not out["eligible"].any()
    assert out["run_start"].isna().all()  # its first bar hasn't closed yet at either time


def test_minimum_history_is_enforced(market):
    prices, _ = market
    out = tradable_universe(at("NEWUSDT", T0 + 73 * H, T0 + 74 * H), prices, **UNIVERSE)
    assert out["eligible"].tolist() == [False, True]


def test_future_delisting_is_not_anticipated(market):
    """DELUSDT's last traded bar closes just before 30h. It must still be held at 29h and 30h;
    dropping it earlier would use knowledge of the delisting."""
    prices, _ = market
    out = tradable_universe(at("DELUSDT", T0 + 29 * H, T0 + 30 * H, T0 + 33 * H), prices, **UNIVERSE)
    assert out["eligible"].tolist() == [True, True, False]


def test_zombie_bars_never_make_a_dead_contract_tradable(market):
    prices, _ = market
    out = tradable_universe(at("DELUSDT", T0 + 40 * H, T0 + 70 * H, T0 + 100 * H), prices, **UNIVERSE)
    assert not out["eligible"].any()


def test_revived_contract_must_rebuild_its_history(market):
    prices, _ = market
    out = tradable_universe(at("REVUSDT", T0 + 50 * H, T0 + 81 * H, T0 + 103 * H, T0 + 104 * H), prices, **UNIVERSE)
    assert out["eligible"].tolist() == [False, False, False, True]  # dormant; revived 1h; 23h; 24h
    assert out["run_start"].iloc[3] == T0 + 80 * H


def test_decision_grid_is_on_a_fixed_utc_schedule():
    grid = decision_grid(["BUSDT", "AUSDT"], "2024-01-01", "2024-01-02", every_hours=8, offset_minutes=5)
    assert grid["symbol"].tolist() == ["AUSDT"] * 3 + ["BUSDT"] * 3
    five_min = 5 * 60_000
    assert grid["decision_time"].tolist()[:3] == [T0 + five_min, T0 + 8 * H + five_min, T0 + 16 * H + five_min]
    assert grid["decision_time"].dtype == "int64"


# ── The core guarantee ─────────────────────────────────────────────────────────

def _lookups(prices, funding, decisions):
    return {
        "funding": latest_funding(decisions, funding),
        "carry": trailing_funding(decisions, funding, window_hours=24),
        "close": latest_close(decisions, prices, max_staleness_hours=2),
        "volume": trailing_quote_volume(decisions, prices, window_hours=5),
        "universe": tradable_universe(decisions, prices, **UNIVERSE),
    }


def _corrupt_future(prices, funding, cutoff):
    prices, funding = prices.copy(), funding.copy()
    future_bar = prices["close_time"] >= cutoff
    prices.loc[future_bar, ["open", "high", "low", "close"]] *= 3.0
    prices.loc[future_bar, "quote_volume"] *= 10.0
    prices.loc[future_bar & (prices["open_time"] % (7 * H) == 0), "quote_volume"] = 0.0
    prices = prices[~(future_bar & (prices["open_time"] % (5 * H) == 0))]
    future_rate = funding["calc_time"] >= cutoff
    funding.loc[future_rate, ["rate", "rate_per_hour"]] *= -50.0
    late_hour = (cutoff - T0) // H + 5
    prices = pd.concat([prices, make_bars("LATEUSDT", late_hour, 10), make_bars("DELUSDT", late_hour + 50, 10)],
                       ignore_index=True)
    funding = pd.concat([funding, make_funding("LATEUSDT", [late_hour + 3])], ignore_index=True)
    return prices, funding


def test_lookups_ignore_everything_at_or_after_the_decision_time(market):
    prices, funding = market
    cutoff = T0 + CUTOFF_HOUR * H
    times = np.r_[T0 + np.arange(CUTOFF_HOUR + 1) * H, T0 + 16 * H + 14, cutoff - 1]
    symbols = ["AAAUSDT", "NEWUSDT", "DELUSDT", "REVUSDT", "LATEUSDT"]
    decisions = pd.DataFrame([(s, t) for s in symbols for t in times], columns=["symbol", "decision_time"])
    decisions["decision_time"] = decisions["decision_time"].astype("int64")

    dirty_prices, dirty_funding = _corrupt_future(prices, funding, cutoff)
    clean = _lookups(prices, funding, decisions)
    dirty = _lookups(dirty_prices, dirty_funding, decisions)

    later = at("AAAUSDT", cutoff + 10 * H)  # sanity: the corruption is visible to a later decision
    assert latest_close(later, prices, 2)["close"].iloc[0] != latest_close(later, dirty_prices, 2)["close"].iloc[0]
    for name in clean:
        pd.testing.assert_frame_equal(clean[name], dirty[name], obj=name)
