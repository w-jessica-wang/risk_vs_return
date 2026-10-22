# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Maker-execution tests (Amendment A5b) — the fill simulator respects queue position, trade-throughs and side,
measures the unfilled order's adverse move, and the cost model behaves at its limits.

Run:  pytest tests/ -q
"""
import numpy as np
import pandas as pd
import pytest

from rvr.costs import CostModel
from rvr.execution import MakerCostModel, fit_adverse, fit_fill, simulate_order

T = 1_700_000_000_000
MIN = 60_000


def quotes(rows):
    return pd.DataFrame(rows, columns=["best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty",
                                       "transaction_time"])


def trades(rows):
    """rows: (time, price, quantity, taker_sell)."""
    return pd.DataFrame(rows, columns=["transact_time", "price", "quantity", "taker_sell"])


BOOK = quotes([(99.0, 50.0, 101.0, 50.0, T - MIN), (99.0, 50.0, 101.0, 50.0, T + 31 * MIN)])


def test_order_behind_the_queue_fills_only_after_the_queue_ahead():
    flow = trades([(T + MIN, 99.0, 30.0, True), (T + 2 * MIN, 99.0, 40.0, True)])  # 70 sold at the bid; 50 ahead
    r = simulate_order(BOOK, flow, T, side=1, notional=99.0 * 40)                  # wants 40
    assert r["fill"] == pytest.approx(20 / 40)
    assert r["touch_notional"] == pytest.approx(50 * 99.0)


def test_a_trade_through_the_price_fills_only_by_its_quantity():
    flow = trades([(T + MIN, 98.5, 60.0, True)])                    # 60 sold through the bid; 50 ahead
    assert simulate_order(BOOK, flow, T, side=1, notional=99.0 * 1000)["fill"] == pytest.approx(10 / 1000)
    assert simulate_order(BOOK, flow, T, side=1, notional=99.0 * 5)["fill"] == 1.0


def test_buys_ignore_taker_buys_and_sells_mirror_buys():
    flow = trades([(T + MIN, 101.0, 500.0, False)])          # takers lifting the ask
    assert simulate_order(BOOK, flow, T, side=1, notional=990)["fill"] == 0.0
    assert simulate_order(BOOK, flow, T, side=-1, notional=101.0 * 10)["fill"] == 1.0   # 500 > 50 ahead + 10


def test_trades_after_tau_do_not_count_and_the_adverse_move_is_signed():
    moved = quotes([(99.0, 50.0, 101.0, 50.0, T - MIN), (103.0, 50.0, 105.0, 50.0, T + 20 * MIN)])
    flow = trades([(T + 31 * MIN, 90.0, 1.0, True)])
    r = simulate_order(moved, flow, T, side=1, notional=990)
    assert r["fill"] == 0.0
    assert r["move"] == pytest.approx(104.0 / 100.0 - 1)       # price ran away from an unfilled buy
    assert simulate_order(moved, flow, T, side=-1, notional=1010)["move"] == pytest.approx(-(104.0 / 100.0 - 1))


def test_fill_and_adverse_fits_recover_known_relations():
    rng = np.random.default_rng(0)
    n = 4000
    ratio, sigma = np.exp(rng.uniform(-3, 3, n)), rng.uniform(0.02, 0.08, n)
    p = 1 / (1 + np.exp(-(0.5 - 0.8 * np.log(ratio))))
    sims = pd.DataFrame({"notional": ratio * 1e4, "touch_notional": 1e4, "sigma": sigma,
                         "fill": rng.binomial(1, p).astype(float), "move": 0.0})
    sims.loc[sims["fill"] < 1, "move"] = 0.05 * sims.loc[sims["fill"] < 1, "sigma"]
    coef = fit_fill(sims)["coef"]
    assert coef[1] == pytest.approx(-0.8, abs=0.1)
    assert fit_adverse(sims)["kappa"] == pytest.approx(0.05)


def test_maker_cost_limits():
    taker = CostModel("t", 5e-4, 0.2, spread_coef=(np.log(2e-4), 0.0, 0.0))   # half-spread 2 bp
    always = MakerCostModel("m", taker, fill_coef=(50.0, 0.0, 0.0), touch_coef=(np.log(1e4), 0.0, 0.0), kappa=0.0)
    never = MakerCostModel("m", taker, fill_coef=(-50.0, 0.0, 0.0), touch_coef=(np.log(1e4), 0.0, 0.0), kappa=0.0)
    adv, sigma, q = np.array([1e8]), np.array([0.04]), np.array([1e4])
    assert always.unit_cost(q, adv, sigma)[0] == pytest.approx(2e-4 - 2e-4)          # maker fee − half-spread
    assert never.unit_cost(q, adv, sigma)[0] == pytest.approx(taker.unit_cost(q, adv, sigma)[0])
