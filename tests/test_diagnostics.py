# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Diagnostics tests — the pass-through regression recovers a known slope, and its t-statistics test the two
hypotheses they claim to (no pass-through, b = 0; funding parity, b = 1).

Run:  pytest tests/ -q
"""
import numpy as np
import pandas as pd
import pytest

from rvr.diagnostics import annual_performance, daily_pass_through, summarize_slopes


def _frame(b, days=300, n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        f = rng.normal(0, 0.002, n)
        rows.append(pd.DataFrame({"day": d * 86_400_000, "f": f, "r": b * f + rng.normal(0, 0.002, n)}))
    return pd.concat(rows, ignore_index=True)


@pytest.mark.parametrize("b", [0.0, 1.0, -0.8])
def test_pass_through_recovers_the_slope(b):
    slopes = daily_pass_through(_frame(b), "r", "f", min_contracts=20, winsor=0.0)
    result = summarize_slopes(slopes, lags=5)
    assert result["b"] == pytest.approx(b, abs=0.05)


def test_t_statistics_reject_the_right_hypothesis():
    parity = summarize_slopes(daily_pass_through(_frame(1.0), "r", "f", 20, 0.0), lags=5)
    assert abs(parity["t_vs_1"]) < 3 and parity["t_vs_0"] > 10
    none = summarize_slopes(daily_pass_through(_frame(0.0), "r", "f", 20, 0.0), lags=5)
    assert abs(none["t_vs_0"]) < 3 and none["t_vs_1"] < -10


def test_thin_days_are_skipped():
    f = _frame(0.5, days=3, n=10)
    assert daily_pass_through(f, "r", "f", min_contracts=20, winsor=0.0).empty


def test_annual_performance_groups_by_calendar_year():
    idx = pd.Index([pd.Timestamp("2021-12-31", tz="UTC").value // 10**6, pd.Timestamp("2022-01-01", tz="UTC").value // 10**6,
                    pd.Timestamp("2022-01-02", tz="UTC").value // 10**6])
    out = annual_performance(pd.DataFrame({"B": [0.01, 0.02, 0.04]}, index=idx))
    assert out["B"][2022]["annual_return"] == pytest.approx(0.03 * 365)
