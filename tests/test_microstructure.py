# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Microstructure tests — the execution-window reductions measure what they claim on hand-built books and
trades, the book-walk cost matches closed forms, and no holdout day can be sampled.

Run:  pytest tests/ -q
"""
import numpy as np
import pandas as pd
import pytest

from rvr import microstructure as m

T = 1_700_000_000_000  # execution time, epoch ms
MIN = 60_000


def _csv(frame):
    return frame.to_csv(index=False).encode()


def _quotes(rows):
    return pd.DataFrame(rows, columns=["best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty",
                                       "transaction_time"])


def test_quoted_spread_is_time_weighted_and_includes_the_quote_standing_at_window_start():
    lo, hi = T - 10 * MIN, T + 10 * MIN
    book = _quotes([
        (99.0, 1, 101.0, 1, lo - 5 * MIN),   # stands from before the window until T: half-spread 1%
        (99.5, 1, 100.5, 1, T),              # from T to hi: half-spread 0.5%
        (90.0, 1, 110.0, 1, hi + MIN),       # after the window: ignored
    ])
    out = m.summarize_quotes(m.read_quotes(_csv(book), lo, hi), lo, hi)
    assert out["half_spread"] == pytest.approx((0.01 + 0.005) / 2)
    assert out["touch_notional"] == pytest.approx(100.0)


def test_crossed_or_empty_quotes_are_dropped():
    book = _quotes([(101.0, 1, 100.0, 1, T), (0.0, 1, 100.0, 1, T + 1)])
    assert m.read_quotes(_csv(book), T - MIN, T + MIN).empty


def test_reading_stops_at_the_window_and_spans_chunks():
    n = 450_000  # more than two read chunks
    times = T - 20 * MIN + np.arange(n) * 10
    book = _quotes({"best_bid_price": 99.0, "best_bid_qty": 1.0, "best_ask_price": 101.0, "best_ask_qty": 1.0,
                    "transaction_time": times})
    q = m.read_quotes(_csv(book), T, T + MIN)
    assert q["transaction_time"].iloc[0] < T <= q["transaction_time"].iloc[1]
    assert (q["transaction_time"] < T + MIN).all()


def test_taker_orders_group_fills_and_slippage_is_against_the_prior_mid():
    trades = pd.DataFrame({
        "agg_trade_id": [1, 2, 3], "price": [101.0, 102.0, 99.0], "quantity": [1.0, 1.0, 2.0],
        "first_trade_id": 0, "last_trade_id": 0,
        "transact_time": [T, T, T + 5], "is_buyer_maker": ["false", "false", "true"],
    })
    orders = m.taker_orders(_csv(trades), T - MIN, T + MIN)
    buy = orders[orders["side"] == 1].iloc[0]
    assert buy["notional"] == pytest.approx(203.0) and buy["vwap"] == pytest.approx(101.5)
    quotes = _quotes([(99.0, 1, 101.0, 1, T - 1), (95.0, 1, 105.0, 1, T)])  # second quote is the trade's own update
    filled = m.order_slippage(orders, quotes).set_index("side")
    assert filled.loc[1, "slippage"] == pytest.approx(0.015)          # mid 100, not the later 100 from 95/105
    assert filled.loc[-1, "slippage"] == pytest.approx(-(99.0 / 100.0 - 1))


def test_depth_summary_takes_the_median_over_window_snapshots():
    stamps = pd.to_datetime([T - 20 * MIN, T - MIN, T, T + MIN], unit="ms").strftime("%Y-%m-%d %H:%M:%S")
    rows = [(s, p, 1.0, value * abs(p)) for s, value in zip(stamps, [999.0, 10.0, 20.0, 30.0])
            for p in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)]
    out = m.summarize_book_depth(_csv(pd.DataFrame(rows, columns=["timestamp", "percentage", "depth", "notional"])), T)
    assert out["snapshots"] == 3
    assert out["ask_1"] == pytest.approx(20.0) and out["bid_5"] == pytest.approx(100.0)


def test_walk_cost_on_a_uniform_book_is_half_the_price_reached():
    depth = [[1e6, 2e6, 3e6, 4e6, 5e6]]  # uniform density: 1M per percent
    assert m.walk_cost([5e5], [0.0], depth)[0] == pytest.approx(0.0025)   # reaches 0.5% → averages 0.25%
    assert m.walk_cost([3e6], [1e-4], depth)[0] == pytest.approx(1e-4 + 0.015)
    assert m.walk_cost([1e7], [0.0], depth)[0] == pytest.approx(0.05)     # extends the last band's density


def test_walk_cost_is_increasing_in_size():
    depth = [[2e5, 9e5, 1.4e6, 3e6, 3.1e6]] * 50
    sizes = np.geomspace(1e3, 1e8, 50)
    costs = m.walk_cost(sizes, np.zeros(50), depth)
    assert np.all(np.diff(costs) > 0)


def test_sampling_refuses_holdout_days(tmp_path):
    with pytest.raises(ValueError, match="holdout"):
        m.measure(("bookDepth", "AAAUSDT", "2025-09-01"), out_dir=tmp_path, fetch=lambda key: None)
    with pytest.raises(ValueError, match="holdout"):
        m.measure(("execution", "AAAUSDT", "2025-09-01"), out_dir=tmp_path, fetch=lambda key: None)
    assert m.measure(("bookDepth", "AAAUSDT", "2025-08-31"), out_dir=tmp_path, fetch=lambda key: None) == "missing"


def test_default_sample_stays_in_the_design_period():
    for days in m.default_days().values():
        assert max(days) <= "2025-08-30"
