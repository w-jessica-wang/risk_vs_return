# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Factor and spanning-test tests — tercile portfolios are dollar-neutral with the right legs, the HAC
regression recovers known coefficients and matches statsmodels, and a return built from factors is spanned.

Run:  pytest tests/ -q
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from rvr.factors import MODELS, newey_west_ols, spanning_test, spanning_verdict, tercile_weights
from rvr.spec import Spec

SPEC = Spec(min_eligible=3, newey_west_lags=5)


def _panel(n_days=3, n_symbols=9):
    rows = [(d, f"S{i}USDT", True, float(i)) for d in range(n_days) for i in range(n_symbols)]
    return pd.DataFrame(rows, columns=["day", "symbol", "eligible", "x"])


def test_tercile_weights_are_dollar_neutral_with_the_right_legs():
    w = tercile_weights(_panel(), "x", long_high=True, min_eligible=3)
    day0 = w[w["day"] == 0].set_index("symbol")["w"]
    assert day0.sum() == pytest.approx(0.0) and day0.abs().sum() == pytest.approx(1.0)
    assert set(day0[day0 > 0].index) == {"S6USDT", "S7USDT", "S8USDT"}
    flipped = tercile_weights(_panel(), "x", long_high=False, min_eligible=3)
    assert set(flipped[(flipped["day"] == 0) & (flipped["w"] > 0)]["symbol"]) == {"S0USDT", "S1USDT", "S2USDT"}


def test_ineligible_and_thin_days_hold_nothing():
    p = _panel()
    p.loc[p["day"] == 1, "eligible"] = p.loc[p["day"] == 1, "symbol"].isin(["S0USDT", "S1USDT"])
    w = tercile_weights(p, "x", long_high=True, min_eligible=3)
    assert 1 not in set(w["day"])


def test_newey_west_ols_matches_statsmodels():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(500), rng.normal(size=(500, 2))])
    y = X @ np.array([0.1, 0.5, -0.2]) + rng.normal(size=500)
    beta, t, _ = newey_west_ols(y, X, lags=5)
    ref = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5, "use_correction": False})
    assert beta == pytest.approx(ref.params)
    assert t == pytest.approx(ref.tvalues, rel=1e-6)


def test_a_return_built_from_factors_is_spanned_and_one_with_alpha_is_not():
    rng = np.random.default_rng(1)
    factors = pd.DataFrame(rng.normal(0, 0.01, size=(2000, 5)), columns=MODELS["c_five_factor"])
    spanned = factors @ np.array([0.2, 0.5, -0.3, 0.8, 0.1]) + rng.normal(0, 0.002, 2000)
    assert spanning_verdict(spanning_test(pd.Series(spanned), factors, SPEC)) == "Spanned by known factors"
    alpha = spanned + 0.002
    result = spanning_test(pd.Series(alpha), factors, SPEC)
    assert spanning_verdict(result) == "Not spanned by known factors"
    assert result["c_five_factor"]["alpha_annual"] == pytest.approx(0.002 * 365, rel=0.1)
    assert result["c_five_factor"]["loadings"]["REV"]["beta"] == pytest.approx(0.8, abs=0.05)
