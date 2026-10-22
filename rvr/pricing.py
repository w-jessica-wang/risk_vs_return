# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
The thesis test in pricing form: Fama-MacBeth regressions (PREREGISTRATION.md §7).

Each day, every eligible contract's next-day long return is regressed on cross-sectionally
standardised characteristics; the daily slopes are then averaged and tested with Newey-West errors.
If carry is compensation for crash exposure, its slope should shrink once crash exposure is controlled.
"""
import numpy as np
import pandas as pd

from rvr.stats import newey_west_t


def _winsorize(x, level):
    low, high = np.nanquantile(x, [level, 1 - level])
    return np.clip(x, low, high)


def _zscore(x):
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def daily_slopes(panel, y_col, x_cols, spec):
    """One row of OLS slopes per day with at least spec.fm_min_contracts complete observations."""
    rows = []
    for day, g in panel.groupby("day", sort=True):
        g = g.dropna(subset=[y_col, *x_cols])
        if len(g) < spec.fm_min_contracts:
            continue
        y = _winsorize(g[y_col].to_numpy(dtype=float), spec.fm_winsor)
        X = np.column_stack([np.ones(len(g))] + [_zscore(_winsorize(g[c].to_numpy(dtype=float), spec.fm_winsor))
                                                 for c in x_cols])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rows.append((day, len(g), *coef))
    return pd.DataFrame(rows, columns=["day", "n", "const", *x_cols]).set_index("day")


def fama_macbeth(panel, y_col, x_cols, spec):
    slopes = daily_slopes(panel, y_col, x_cols, spec)
    summary = {c: {"mean": float(slopes[c].mean()), "newey_west_t": newey_west_t(slopes[c], spec.newey_west_lags)}
               for c in x_cols}
    return {"days": int(len(slopes)), "avg_contracts": float(slopes["n"].mean()), "coefficients": summary}, slopes


def pricing_test(panel, y_col, carry_col, controls, spec):
    """Model 1 (carry) and Model 2 (carry + controls) on the identical sample."""
    sample = panel.dropna(subset=[y_col, carry_col, *controls])
    model1, _ = fama_macbeth(sample, y_col, [carry_col], spec)
    model2, _ = fama_macbeth(sample, y_col, [carry_col, *controls], spec)
    return {"model1": model1, "model2": model2}
