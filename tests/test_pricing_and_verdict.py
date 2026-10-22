# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Pricing-test and verdict tests — Fama-MacBeth recovers a known structure, and each verdict table
in PREREGISTRATION.md §6–7, §10 maps the inputs it should.

Run:  pytest tests/ -q
"""
import numpy as np
import pandas as pd

from rvr import verdict
from rvr.pricing import pricing_test
from rvr.spec import Spec

SPEC = Spec(newey_west_lags=5)


def synthetic(carry_drives_returns, seed=8, days=80, n=40):
    rng = np.random.default_rng(seed)
    frames = []
    for d in range(days):
        beta = rng.normal(0, 1, n)
        carry = beta + 0.3 * rng.normal(0, 1, n)
        driver = carry if carry_drives_returns else beta
        frames.append(pd.DataFrame({"day": d, "carry": carry, "downside_beta": beta,
                                    "y": 0.01 * driver + 0.002 * rng.normal(0, 1, n)}))
    return pd.concat(frames, ignore_index=True)


def test_carry_that_only_proxies_crash_exposure_is_explained_away():
    result = pricing_test(synthetic(carry_drives_returns=False), "y", "carry", ["downside_beta"], SPEC)
    assert result["model1"]["coefficients"]["carry"]["newey_west_t"] > 10
    assert verdict.pricing_verdict(result, "carry") == "Risk premium: crash exposure explains carry"


def test_carry_that_prices_returns_itself_survives_the_controls():
    result = pricing_test(synthetic(carry_drives_returns=True), "y", "carry", ["downside_beta"], SPEC)
    assert verdict.pricing_verdict(result, "carry") == "Edge survives crash-risk controls"


def stats(sharpe, ci, es_per_vol=-3.0):
    return {"sharpe": sharpe, "sharpe_ci95": list(ci), "es_1pct_per_vol": es_per_vol}


def test_portfolio_verdict_table():
    base = stats(2.0, (1.0, 3.0), es_per_vol=-4.0)
    assert verdict.portfolio_verdict(stats(0.5, (-0.5, 3.0)), stats(0.1, (-1, 1)), SPEC) == verdict.NO_PREMIUM
    assert verdict.portfolio_verdict(base, stats(0.5, (-0.4, 1.4), es_per_vol=-3.5), SPEC) == verdict.CONTROLS_FAILED
    assert verdict.portfolio_verdict(base, stats(0.5, (-0.4, 1.4), es_per_vol=-2.5), SPEC) == verdict.RISK_PREMIUM
    assert verdict.portfolio_verdict(base, stats(1.6, (0.6, 2.6), es_per_vol=-2.5), SPEC) == verdict.EDGE_SURVIVES
    assert verdict.portfolio_verdict(base, stats(1.2, (-0.1, 2.5), es_per_vol=-2.5), SPEC) == verdict.INCONCLUSIVE


def test_holdout_confirmation():
    design = {"B": stats(2.0, (1, 3)), "M": stats(0.4, (-1, 1))}
    assert verdict.holdout_confirms(design, {"B": stats(1.0, (0, 2), -4.0), "M": stats(0.2, (-1, 1), -2.0)})
    assert not verdict.holdout_confirms(design, {"B": stats(-1.0, (-2, 0), -4.0), "M": stats(0.2, (-1, 1), -2.0)})
