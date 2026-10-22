# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Statistics tests — Newey-West against statsmodels, a reproducible bootstrap, the deflated Sharpe
ratio against its definition, and the tail measures on known inputs.

Run:  pytest tests/ -q
"""
import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats as sps

from rvr.stats import (bootstrap_sharpe_ci, deflated_sharpe, expected_shortfall, max_drawdown, newey_west_t,
                       sharpe)


def ar1(n, phi, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0.0005, 0.01)
    return x


def test_newey_west_matches_statsmodels():
    x = ar1(600, 0.3, 4)
    fit = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": 10, "use_correction": False})
    assert newey_west_t(x, 10) == pytest.approx(fit.tvalues[0], rel=1e-9)


def test_bootstrap_is_reproducible_and_brackets_the_sample_sharpe():
    x = np.random.default_rng(5).normal(0.001, 0.01, 800)
    ci = bootstrap_sharpe_ci(x, mean_block=20, n_samples=2000, seed=7)
    assert ci == bootstrap_sharpe_ci(x, mean_block=20, n_samples=2000, seed=7)
    assert ci[0] < sharpe(x) < ci[1]


def test_deflated_sharpe_reduces_to_the_probabilistic_sharpe_with_one_trial():
    x = np.random.default_rng(6).normal(0.001, 0.01, 1000)
    sr = x.mean() / x.std(ddof=1)
    psr = sps.norm.cdf(sr * np.sqrt(len(x) - 1) / np.sqrt(
        1 - sps.skew(x) * sr + (sps.kurtosis(x, fisher=False) - 1) / 4 * sr**2))
    assert deflated_sharpe(x, [sr]) == pytest.approx(psr)
    assert deflated_sharpe(x, [sr, 0.0, 0.05, -0.03, 0.08]) < psr  # more trials, higher bar


def test_tail_measures():
    x = np.array([0.1, -0.2, 0.05, -0.1, 0.0, 0.02, -0.05, 0.03, 0.01, -0.01])
    assert expected_shortfall(x, 0.2) == pytest.approx(-0.15)  # mean of the worst 2
    # peak after day 1 (1.10), trough after day 7: the fall is the product of days 2–7
    assert max_drawdown(x) == pytest.approx(0.8 * 1.05 * 0.9 * 1.0 * 1.02 * 0.95 - 1)
    assert max_drawdown([0.1, 0.1]) == 0.0
    assert max_drawdown([-0.20]) == pytest.approx(-0.20)            # a loss on the first day counts
    assert max_drawdown([-0.10, 0.05, -0.10]) == pytest.approx(0.9 * 1.05 * 0.9 - 1)
    assert max_drawdown([0.10, -0.20]) == pytest.approx(-0.20)
