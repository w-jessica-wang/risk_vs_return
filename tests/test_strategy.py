# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Strategy tests — quintile portfolios, the breadth rule, the volatility filter, the downside-beta
hedge, and the volatility target's two-day information lag (PREREGISTRATION.md §3, §6).

Run:  pytest tests/ -q
"""
import numpy as np
import pandas as pd
import pytest

from rvr.spec import Spec
from rvr.strategy import add_downside_beta_hedge, low_volatility_candidates, quintile_weights, volatility_scale

SPEC = Spec()
DAY = 1_704_067_200_000


def panel(n, eligible=True, **columns):
    frame = pd.DataFrame({"day": DAY, "symbol": [f"S{i:02d}" for i in range(n)], "eligible": eligible,
                          "carry_72h": np.arange(n, dtype=float)})
    return frame.assign(**columns)


def as_dict(weights):
    return dict(zip(weights["symbol"], weights["w"]))


def test_short_the_highest_carry_quintile_long_the_lowest():
    w = as_dict(quintile_weights(panel(25), "carry_72h", SPEC))
    assert w == pytest.approx({**{f"S{i:02d}": 0.1 for i in range(5)}, **{f"S{i:02d}": -0.1 for i in range(20, 25)}})
    assert sum(w.values()) == pytest.approx(0.0)
    assert sum(abs(v) for v in w.values()) == pytest.approx(1.0)


def test_no_portfolio_below_twenty_eligible_contracts():
    assert quintile_weights(panel(19), "carry_72h", SPEC).empty


def test_volatility_filter_drops_the_most_volatile_quintile_before_ranking():
    p = panel(25, vol=25.0 - np.arange(25))  # S00 is the most volatile
    w = as_dict(quintile_weights(p, "carry_72h", SPEC, candidates=low_volatility_candidates(p, SPEC)))
    assert set(w) == {"S05", "S06", "S07", "S08", "S21", "S22", "S23", "S24"}
    assert w["S05"] == pytest.approx(0.125)


def test_downside_beta_hedge_offsets_portfolio_beta_with_btc():
    weights = pd.DataFrame({"day": DAY, "symbol": ["A", "B", "BTCUSDT"], "w": [0.5, -0.5, 0.1]})
    betas = pd.DataFrame({"day": DAY, "symbol": ["A", "B", "BTCUSDT"], "downside_beta": [2.0, 1.0, 1.0]})
    hedged = as_dict(add_downside_beta_hedge(weights, betas, SPEC))
    # exposure = 0.5·2 − 0.5·1 + 0.1·1 = 0.6, so BTC goes from 0.1 to −0.5
    assert hedged == pytest.approx({"A": 0.5, "B": -0.5, "BTCUSDT": -0.5})


def test_volatility_target_only_uses_returns_completed_before_the_decision():
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0, 0.01, 120))
    shocked = returns.copy()
    shocked.iloc[80] = 0.5  # day 80's return is only known at day 81 01:00
    clean, dirty = volatility_scale(returns, SPEC), volatility_scale(shocked, SPEC)
    pd.testing.assert_series_equal(clean.iloc[:82], dirty.iloc[:82])
    assert dirty.iloc[82] < clean.iloc[82]
    assert (volatility_scale(returns, SPEC) <= SPEC.max_scale).all()


def _days(*signals):
    """One panel per day; signals[k] maps symbol index → signal (all eligible)."""
    frames = [pd.DataFrame({"day": DAY + k * 86_400_000, "symbol": [f"S{i:02d}" for i in range(len(s))],
                            "eligible": True, "carry_72h": np.asarray(s, dtype=float)}) for k, s in enumerate(signals)]
    return pd.concat(frames, ignore_index=True)


def test_buffer_matches_b_when_nothing_is_held_yet():
    from rvr.strategy import buffered_weights
    p = panel(40)
    pd.testing.assert_frame_equal(buffered_weights(p, "carry_72h", SPEC).sort_values("symbol").reset_index(drop=True),
                                  quintile_weights(p, "carry_72h", SPEC).sort_values("symbol").reset_index(drop=True),
                                  check_dtype=False)


def test_buffer_keeps_a_position_inside_the_exit_band_and_drops_it_outside():
    from rvr.strategy import buffered_weights
    base = np.arange(40, dtype=float)
    drift_in, drift_out = base.copy(), base.copy()
    drift_in[0], drift_out[0] = 12.5, 20.5   # S00 starts lowest (long); moves to rank 13 (inside 40%) or 21 (outside)
    kept = buffered_weights(_days(base, drift_in), "carry_72h", SPEC)
    dropped = buffered_weights(_days(base, drift_out), "carry_72h", SPEC)
    day2 = DAY + 86_400_000
    kept2 = as_dict(kept[kept["day"] == day2])
    assert "S00" in kept2 and kept2["S00"] == pytest.approx(0.5 / 9)       # 8 entrants + 1 held
    assert "S00" not in as_dict(dropped[dropped["day"] == day2])
    for frame in (kept, dropped):
        by_day = frame.groupby("day")["w"]
        assert np.allclose(by_day.sum(), 0.0) and np.allclose(frame["w"].abs().groupby(frame["day"]).sum(), 1.0)


def test_buffer_drops_ineligible_holdings_and_resets_after_a_thin_day():
    from rvr.strategy import buffered_weights
    base = np.arange(40, dtype=float)
    p = _days(base, base, base)
    p.loc[(p["day"] == DAY + 86_400_000) & (p["symbol"] != "S00"), "eligible"] = False      # day 2: too thin
    p.loc[(p["day"] == DAY + 2 * 86_400_000) & (p["symbol"] == "S39"), "eligible"] = False  # day 3: a short delists
    w = buffered_weights(p, "carry_72h", SPEC)
    assert DAY + 86_400_000 not in set(w["day"])
    day3 = as_dict(w[w["day"] == DAY + 2 * 86_400_000])
    assert "S39" not in day3 and len(day3) == 2 * 7   # 39 eligible → 7 per side, nothing carried over


def test_core_band_skips_the_extreme_decile_on_each_side():
    from rvr.strategy import band_weights
    w = as_dict(band_weights(panel(50), "carry_72h", SPEC))
    longs, shorts = sorted(s for s, x in w.items() if x > 0), sorted(s for s, x in w.items() if x < 0)
    assert longs == [f"S{i:02d}" for i in range(5, 15)]       # ranks 6–15 of 50: 10th–30th percentile
    assert shorts == [f"S{i:02d}" for i in range(35, 45)]
    assert sum(w.values()) == pytest.approx(0.0) and sum(abs(x) for x in w.values()) == pytest.approx(1.0)
