# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
When did crypto carry die? A formal test for a shift in funding pass-through (PREREGISTRATION.md A7a).

A single break in the mean of the daily pass-through slopes is fitted by least squares at every candidate
date (Bai 1997), with 15% trimmed from each end. Evidence for a break is the sup-F statistic, its p-value from
a stationary bootstrap of the demeaned series (the null of no break, with the series' autocorrelation kept),
and the break date's 95% interval from a residual block bootstrap around the fitted two-mean model.
Exploratory: the slopes come from data already seen.

Run:  python -m rvr.breaks          # needs results/pass_through_daily.parquet from rvr.diagnostics
"""
import json

import numpy as np
import pandas as pd

from rvr.config import ROOT
from rvr.spec import SPEC
from rvr.stats import stationary_bootstrap_indices

RESULTS_DIR = ROOT / "results"
TRIM = 0.15
N_BOOT = 2000


def break_path(y, trim=TRIM):
    """
    (candidate split indices, F statistic at each). A split at k puts y[:k] before the break and y[k:] after.
    F = (SSR without a break − SSR with it) / (SSR with it / (n − 2)).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    lo, hi = max(1, int(np.floor(trim * n))), min(n - 1, int(np.ceil((1 - trim) * n)))
    ks = np.arange(lo, hi + 1)
    csum, csq = np.cumsum(y), np.cumsum(y * y)
    total, total_sq = csum[-1], csq[-1]
    left_sum, left_sq = csum[ks - 1], csq[ks - 1]
    right_sum, right_sq = total - left_sum, total_sq - left_sq
    ssr_break = (left_sq - left_sum**2 / ks) + (right_sq - right_sum**2 / (n - ks))
    ssr_none = total_sq - total**2 / n
    return ks, (ssr_none - ssr_break) / (ssr_break / (n - 2))


def sup_f(y, trim=TRIM):
    """(sup-F, split index at the maximum)."""
    ks, f = break_path(y, trim)
    i = int(np.argmax(f))
    return float(f[i]), int(ks[i])


def test_break(y, trim=TRIM, n_boot=N_BOOT, mean_block=SPEC.bootstrap_block, seed=SPEC.bootstrap_seed):
    """sup-F, its stationary-bootstrap p-value, the split, the two means, and the split's 95% bootstrap interval."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    stat, k = sup_f(y, trim)
    rng = np.random.default_rng(seed)

    null = (y - y.mean())[stationary_bootstrap_indices(n, n_boot, mean_block, rng)]
    p_value = float(np.mean([sup_f(row, trim)[0] >= stat for row in null]))

    fitted = np.where(np.arange(n) < k, y[:k].mean(), y[k:].mean())
    resid = y - fitted
    draws = (fitted + resid[stationary_bootstrap_indices(n, n_boot, mean_block, rng)])
    splits = np.array([sup_f(row, trim)[1] for row in draws])
    return {"sup_f": stat, "p_value": p_value, "split": k, "n": n,
            "mean_before": float(y[:k].mean()), "mean_after": float(y[k:].mean()),
            "split_ci95": [int(np.percentile(splits, 2.5)), int(np.percentile(splits, 97.5))]}


def dated(result, index):
    """Attach calendar dates (the first day after the break) to a test_break result."""
    day = lambda i: str(pd.to_datetime(int(index[min(i, len(index) - 1)]), unit="ms").date())  # noqa: E731
    return result | {"break_date": day(result["split"]),
                     "break_date_ci95": [day(result["split_ci95"][0]), day(result["split_ci95"][1])]}


def main():
    total = pd.read_parquet(RESULTS_DIR / "total_slope_daily.parquet")["slope"].sort_index()
    slopes = pd.read_parquet(RESULTS_DIR / "pass_through_daily.parquet")["slope"].sort_index()
    y, index = slopes.to_numpy(), slopes.index.to_numpy()
    first = dated(test_break(y), index)
    k = first["split"]
    second = {}
    for name, part, offset in (("before", y[:k], 0), ("after", y[k:], k)):
        r = test_break(part)
        second[name] = dated(r | {"split": r["split"] + offset,
                                  "split_ci95": [c + offset for c in r["split_ci95"]]}, index)
    lo, hi = pd.Timestamp("2021-06-01"), pd.Timestamp("2022-12-31")
    in_window = lo <= pd.Timestamp(first["break_date"]) <= hi
    holds = (first["p_value"] < 0.05 and in_window and first["mean_before"] < 0.5 and first["mean_after"] >= 0.5)
    robustness = dated(test_break(total.to_numpy()), total.index.to_numpy())
    out = {"note": "A7a, exploratory: the slopes come from data already seen.", "break": first,
           "second_break_within_segments": second, "prediction_A7a": "holds" if holds else "falsified",
           "total_return_slope_break": robustness | {
               "note": "Robustness, added after review: the same test on the total-return slope (b − ρ), whose "
                       "no-carry value is 0; not part of the A7a prediction."}}
    (RESULTS_DIR / "breaks.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
