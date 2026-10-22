# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Cost-model tests — the pre-registered model reproduces §5 exactly through pnl(), the calibrated model
charges what Amendment A1 says, stress scaling multiplies every component, and the A1 fitting procedure
recovers known parameters from synthetic order-book samples.

Run:  pytest tests/ -q
"""
import numpy as np
import pandas as pd
import pytest

from rvr.backtest import pnl
from rvr.costs import CostModel, fit_half_spread, fit_impact, impact_points, preregistered
from rvr.spec import Spec
from tests.test_backtest import AAA, AAA_FUNDING, D, run

CHARS = {"adv": pd.DataFrame([[1e8]], index=[D], columns=["AAAUSDT"]),
         "vol": pd.DataFrame([[0.04]], index=[D], columns=["AAAUSDT"])}


def test_preregistered_model_reproduces_the_section_5_costs():
    spec = Spec(aum=1e6)
    cdays, default = run(AAA, AAA_FUNDING, [(D, "AAAUSDT", 0.5)], [D], spec=spec, chars=CHARS)
    weights = pd.DataFrame([(D, "AAAUSDT", 0.5)], columns=["day", "symbol", "w"])
    explicit = pnl(weights, cdays, CHARS, pd.Index([D]), spec, cost_model=preregistered(spec))
    assert explicit.loc[D, "cost"] == pytest.approx(default.loc[D, "cost"])
    assert default.loc[D, "cost"] == pytest.approx(0.5 * (5e-4 + 2e-4 + 0.04 * np.sqrt(0.5e6 / 1e8)))


def test_calibrated_model_charges_fee_fitted_spread_and_fitted_impact():
    spec = Spec(aum=1e6)
    a, b, c = -2.0, -0.4, 0.5
    model = CostModel("test", 5e-4, 0.3, spread_coef=(a, b, c))
    cdays, _ = run(AAA, AAA_FUNDING, [(D, "AAAUSDT", 0.5)], [D], spec=spec, chars=CHARS)
    weights = pd.DataFrame([(D, "AAAUSDT", 0.5)], columns=["day", "symbol", "w"])
    daily = pnl(weights, cdays, CHARS, pd.Index([D]), spec, cost_model=model)
    hs = np.exp(a + b * np.log(1e8) + c * np.log(0.04))
    assert daily.loc[D, "cost"] == pytest.approx(0.5 * (5e-4 + hs + 0.3 * 0.04 * np.sqrt(0.5e6 / 1e8)))


def test_fitted_half_spread_is_clipped():
    model = CostModel("test", 0.0, 0.0, spread_coef=(10.0, 0.0, 0.0))
    assert model.half_spread([1e6], [0.05])[0] == pytest.approx(50e-4)


@pytest.mark.parametrize("model", [preregistered(Spec()), CostModel("c", 5e-4, 0.4, spread_coef=(-3.0, -0.3, 0.6))])
def test_stress_scaling_multiplies_every_component(model):
    adv, sigma = np.array([3e7, 5e8]), np.array([0.03, 0.06])
    doubled = model.scaled(2.0)
    assert doubled.static(adv, sigma) == pytest.approx(2 * model.static(adv, sigma))
    assert doubled.impact_coefficient == pytest.approx(2 * model.impact_coefficient)


def _execution_sample(rng, n=400, a=-1.0, b=-0.45, c=0.6, Y=0.35):
    adv = np.exp(rng.uniform(np.log(1e7), np.log(5e9), n))
    sigma = rng.uniform(0.02, 0.08, n)
    hs = np.exp(a + b * np.log(adv) + c * np.log(sigma) + rng.normal(0, 0.1, n))
    sample = pd.DataFrame({"symbol": [f"S{i % 80}" for i in range(n)], "day": np.arange(n) // 80, "available": True,
                           "half_spread": hs, "adv": adv, "sigma": sigma})
    for bucket, notional in enumerate([500.0, 2e3, 6e3, 2e4, 6e4, 2e5]):
        sample[f"orders_b{bucket}"] = 20
        sample[f"notional_b{bucket}"] = notional
        sample[f"slippage_b{bucket}"] = hs + Y * sigma * np.sqrt(notional / adv) + rng.normal(0, 2e-6, n)
    return sample


def test_a1_fitting_recovers_known_spread_and_impact_parameters():
    rng = np.random.default_rng(0)
    sample = _execution_sample(rng)
    spread = fit_half_spread(sample)
    assert spread["coef"] == pytest.approx([-1.0, -0.45, 0.6], abs=0.1)
    points = impact_points(sample)
    assert set(points["bucket"]) == {1, 2, 3, 4, 5}  # orders under $1,000 are excluded
    impact = fit_impact(points, n_boot=200)
    assert impact["Y"] == pytest.approx(0.35, rel=0.05)
    assert impact["ci95"][0] < 0.35 < impact["ci95"][1]


def test_buckets_with_too_few_orders_are_dropped():
    sample = _execution_sample(np.random.default_rng(1), n=80)
    sample["orders_b5"] = 4
    assert 5 not in set(impact_points(sample)["bucket"])
