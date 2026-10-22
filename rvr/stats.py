# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Inference for daily strategy returns (PREREGISTRATION.md §8).

Sharpe ratios are annualised with √365 (crypto trades every day). The primary confidence interval is
a stationary block bootstrap, which keeps the autocorrelation and volatility clustering a plain
bootstrap would destroy.
"""
import numpy as np
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329
PERIODS = 365


def sharpe(returns):
    returns = np.asarray(returns, dtype=float)
    sd = returns.std(ddof=1)
    return float(returns.mean() / sd * np.sqrt(PERIODS)) if sd > 0 else np.nan


def stationary_bootstrap_indices(n_obs, n_samples, mean_block, rng):
    """Politis & Romano (1994): blocks start at random points, with geometric lengths of mean `mean_block`."""
    idx = np.empty((n_samples, n_obs), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n_obs, n_samples)
    for t in range(1, n_obs):
        restart = rng.random(n_samples) < 1.0 / mean_block
        idx[:, t] = np.where(restart, rng.integers(0, n_obs, n_samples), (idx[:, t - 1] + 1) % n_obs)
    return idx


def bootstrap_sharpe_ci(returns, mean_block, n_samples, seed, level=0.95, chunk=1000):
    returns = np.asarray(returns, dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, n_samples, chunk):
        sample = returns[stationary_bootstrap_indices(len(returns), min(chunk, n_samples - start), mean_block, rng)]
        draws.append(sample.mean(axis=1) / sample.std(axis=1, ddof=1) * np.sqrt(PERIODS))
    draws = np.concatenate(draws)
    tail = (1 - level) / 2 * 100
    return float(np.percentile(draws, tail)), float(np.percentile(draws, 100 - tail))


def newey_west_t(returns, lags):
    """t-statistic of the mean with a Bartlett-kernel HAC standard error."""
    x = np.asarray(returns, dtype=float)
    e = x - x.mean()
    n = len(x)
    long_run = e @ e / n
    for k in range(1, lags + 1):
        long_run += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(long_run / n))


def deflated_sharpe(returns, trial_sharpes):
    """
    Bailey & López de Prado (2014): probability the true Sharpe exceeds the best one expected from
    len(trial_sharpes) unskilled trials. Sharpes here are per-period (daily), not annualised.
    """
    x = np.asarray(returns, dtype=float)
    n = len(x)
    sr = x.mean() / x.std(ddof=1)
    trials = np.asarray(trial_sharpes, dtype=float)
    if len(trials) > 1:
        sd_trials = trials.std(ddof=1)
        sr0 = sd_trials * ((1 - EULER_GAMMA) * sps.norm.ppf(1 - 1 / len(trials))
                           + EULER_GAMMA * sps.norm.ppf(1 - 1 / (len(trials) * np.e)))
    else:
        sr0 = 0.0
    skew = sps.skew(x)
    kurt = sps.kurtosis(x, fisher=False)
    z = (sr - sr0) * np.sqrt(n - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return float(sps.norm.cdf(z))


def expected_shortfall(returns, level):
    x = np.sort(np.asarray(returns, dtype=float))
    k = max(1, int(np.floor(level * len(x))))
    return float(x[:k].mean())


def max_drawdown(returns):
    """Largest peak-to-trough fall of wealth, starting from 1 before the first return."""
    wealth = np.concatenate([[1.0], np.cumprod(1 + np.asarray(returns, dtype=float))])
    return float((wealth / np.maximum.accumulate(wealth) - 1).min())


def downside_beta(returns, market):
    x, m = np.asarray(returns, dtype=float), np.asarray(market, dtype=float)
    down = (m < 0) & np.isfinite(x) & np.isfinite(m)
    return float(np.cov(x[down], m[down])[0, 1] / np.var(m[down], ddof=1))


def summarize(returns, market, spec):
    """Every §8 statistic for one daily return series."""
    x = np.asarray(returns, dtype=float)
    low, high = bootstrap_sharpe_ci(x, spec.bootstrap_block, spec.bootstrap_samples, spec.bootstrap_seed)
    vol = x.std(ddof=1)
    out = {
        "days": int(len(x)),
        "annual_return": float(x.mean() * PERIODS),
        "annual_vol": float(vol * np.sqrt(PERIODS)),
        "sharpe": sharpe(x),
        "sharpe_ci95": [low, high],
        "newey_west_t": newey_west_t(x, spec.newey_west_lags),
        "skew": float(sps.skew(x)),
        "excess_kurtosis": float(sps.kurtosis(x)),
        "worst_day": float(x.min()),
        "max_drawdown": max_drawdown(x),
        "downside_beta_btc": downside_beta(x, market),
    }
    for level in spec.es_levels:
        es = expected_shortfall(x, level)
        out[f"es_{int(level * 100)}pct"] = es
        out[f"es_{int(level * 100)}pct_per_vol"] = es / vol if vol > 0 else np.nan
    return out
