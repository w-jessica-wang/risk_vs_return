# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Structural-break tests (Amendment A7a) — the fast F path matches a brute-force fit, a clear shift is found at
the right place with a small p-value, and a series with no shift is not called a break.

Run:  pytest tests/ -q
"""
import numpy as np
import pytest

from rvr.breaks import break_path, sup_f, test_break as run_break_test


def test_fast_f_path_matches_brute_force():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    ks, f = break_path(y, trim=0.15)
    n = len(y)
    ssr0 = np.sum((y - y.mean()) ** 2)
    for k, value in zip(ks[::17], f[::17], strict=True):
        ssr1 = np.sum((y[:k] - y[:k].mean()) ** 2) + np.sum((y[k:] - y[k:].mean()) ** 2)
        assert value == pytest.approx((ssr0 - ssr1) / (ssr1 / (n - 2)))


def test_a_clear_shift_is_found_where_it_is():
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.normal(-1.0, 1.0, 300), rng.normal(1.0, 1.0, 500)])
    result = run_break_test(y, n_boot=200, mean_block=5)
    assert abs(result["split"] - 300) <= 10
    assert result["p_value"] < 0.01
    assert result["split_ci95"][0] <= 300 <= result["split_ci95"][1]
    assert result["mean_before"] == pytest.approx(-1.0, abs=0.15)


def test_no_shift_is_not_called_a_break():
    rng = np.random.default_rng(2)
    y = rng.normal(size=600)
    assert run_break_test(y, n_boot=200, mean_block=5)["p_value"] > 0.05


def test_trimming_keeps_the_split_away_from_the_ends():
    y = np.r_[10.0, np.zeros(99)]   # an outlier at the very start
    _, k = sup_f(y, trim=0.15)
    assert 15 <= k <= 85
