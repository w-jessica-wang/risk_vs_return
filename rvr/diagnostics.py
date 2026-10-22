# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Why did carry fail out of sample? Exploratory analysis run AFTER the holdout was unsealed — none of this
was pre-registered, and RESULTS.md labels it so.

The central tool is a funding pass-through regression, the perpetual-futures analogue of the uncovered
interest parity test in currency carry (Fama 1984). Each day, across contracts, regress a long position's
next-day price return on the funding it is expected to pay that day:

    price_return = a + b × expected_funding_cost

  b = 0  prices ignore funding: a long keeps (or pays) all of the funding it actually pays
  b = ρ  prices move by the funding actually paid next day, ρ ≈ 0.7 of the trailing signal (signal_persistence):
         "funding parity", carry earns nothing
  b < 0  prices move against funding: the squeeze that made carry's price leg profitable in 2020–21

The no-profit benchmark is therefore b = ρ, not 1. The total-return slope (price + funding on the signal,
equal to b − ρ) measures carry directly: 0 means none, negative means carry earns.

Run:  python -m rvr.diagnostics       # needs both results/design and results/holdout
"""
import json

import numpy as np
import pandas as pd

from rvr.backtest import contract_days
from rvr.config import ROOT
from rvr.features import characteristics, day_range
from rvr.holdout import load_panels
from rvr.pricing import _winsorize
from rvr.spec import SPEC
from rvr.stats import newey_west_t
from rvr.strategy import decision_panel

HOURS_PER_DAY = 24
RESULTS_DIR = ROOT / "results"


def daily_pass_through(frame, y_col, x_col, min_contracts, winsor):
    """One cross-sectional OLS slope of y on x per day (both winsorised), in raw units."""
    rows = []
    for day, g in frame.dropna(subset=[y_col, x_col]).groupby("day", sort=True):
        if len(g) < min_contracts:
            continue
        x = _winsorize(g[x_col].to_numpy(dtype=float), winsor)
        y = _winsorize(g[y_col].to_numpy(dtype=float), winsor)
        if x.std() == 0:
            continue
        slope = np.cov(x, y, ddof=0)[0, 1] / x.var()
        rows.append((day, slope, len(g)))
    return pd.DataFrame(rows, columns=["day", "slope", "n"]).set_index("day")


def summarize_slopes(slopes, lags):
    """Mean slope with Newey-West t-statistics against b = 0 (no pass-through) and b = 1 (funding parity)."""
    b = slopes["slope"].to_numpy()
    t0 = newey_west_t(b, lags)
    se = abs(b.mean() / t0) if t0 != 0 else np.nan
    return {"days": int(len(b)), "b": float(b.mean()), "se": float(se), "t_vs_0": float(t0),
            "t_vs_1": float((b.mean() - 1) / se) if se and np.isfinite(se) else np.nan}


def by_year(slopes, lags):
    years = pd.to_datetime(slopes.index, unit="ms").year
    return {int(y): summarize_slopes(slopes[years == y], lags) for y in sorted(set(years))}


def quintile_sorts(frame, signal, return_cols, period_of):
    """
    Mean daily return of each signal quintile (1 = lowest funding), equal weight, per period — the
    standard portfolio-sort table. Shows whether the carry relation holds in the tails or only in the middle.
    """
    f = frame.dropna(subset=[signal, *return_cols]).copy()
    f["quintile"] = f.groupby("day")[signal].transform(
        lambda x: np.ceil(x.rank(method="first") / len(x) * 5).clip(1, 5)).astype(int)
    f["period"] = period_of(f["day"])
    daily = f.groupby(["period", "day", "quintile"])[list(return_cols)].mean()
    table = daily.groupby(level=["period", "quintile"]).mean() * 365
    return {str(period): {c: {int(q): float(table.loc[(period, q), c]) for q in range(1, 6)} for c in return_cols}
            for period in table.index.get_level_values(0).unique()}


def pass_through_gate(slopes, returns, window, threshold=0.5, lag_days=2):
    """
    Could the regime have been seen in real time? Hold `returns` only while the trailing `window`-day mean
    pass-through is below `threshold`. Day t's slope uses returns realised by t + 1 01:00 UTC, so it is first
    usable at decision t + 2 (lag_days).
    """
    trailing = slopes["slope"].rolling(window, min_periods=window // 2).mean().shift(lag_days).reindex(returns.index)
    on = trailing < threshold
    return returns.where(on, 0.0), on


def interaction_slopes(frame, y_col, x_col, dummy_col, min_contracts, winsor):
    """
    Each day: OLS of y on [1, x, D, x·D] (x and y winsorised; D a 0/1 dummy). Days where the design matrix is
    rank-deficient (too few contracts on one side of D) are skipped. Returns daily coefficients.
    """
    rows = []
    for day, g in frame.dropna(subset=[y_col, x_col, dummy_col]).groupby("day", sort=True):
        if len(g) < min_contracts:
            continue
        x = _winsorize(g[x_col].to_numpy(dtype=float), winsor)
        y = _winsorize(g[y_col].to_numpy(dtype=float), winsor)
        d = g[dummy_col].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(g)), x, d, x * d])
        if np.linalg.matrix_rank(X) < 4:
            continue
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rows.append((day, *coef, int(d.sum()), len(g)))
    return pd.DataFrame(rows, columns=["day", "const", "slope", "young", "slope_x_young", "n_young", "n"]).set_index("day")


def young_mechanism(frame, prices, spec, young_days=180):
    """
    Amendment A7c: does funding pass into price more for contracts listed in the last `young_days` days?
    Age comes from each contract's first traded bar, which is always in the past at the decision.
    """
    first = prices[prices["quote_volume"] > 0].groupby("symbol")["close_time"].min()
    age_days = (frame["day"] - frame["symbol"].map(first)) / 86_400_000
    f = frame.assign(young=(age_days < young_days).astype(float))
    daily = interaction_slopes(f, "price_return", "expected_funding_cost", "young", spec.fm_min_contracts, spec.fm_winsor)
    split = int(pd.Timestamp("2022-01-01", tz="UTC").value // 10**6)
    out = {}
    for name, part in (("2020-05_2021-12", daily[daily.index < split]), ("2022-01_2026-08", daily[daily.index >= split])):
        out[name] = {"days": int(len(part)), "avg_young_share": float((part["n_young"] / part["n"]).mean())}
        for c in ("slope", "slope_x_young"):
            out[name][c] = {"mean": float(part[c].mean()), "newey_west_t": newey_west_t(part[c], spec.newey_west_lags)}
    inter = out["2022-01_2026-08"]["slope_x_young"]
    verdict = ("holds" if inter["mean"] > 0 and inter["newey_west_t"] >= 1.96
               else "falsified" if inter["mean"] < 0 and abs(inter["newey_west_t"]) >= 1.96 else "inconclusive")
    return out | {"prediction_A7c": verdict, "young_days": young_days}


def annual_performance(daily):
    """Sharpe and mean annual return per calendar year for each column of a daily return frame."""
    years = pd.to_datetime(daily.index, unit="ms").year
    out = {}
    for name in daily.columns:
        g = daily[name].groupby(years)
        out[name] = {int(y): {"sharpe": float(m / s * np.sqrt(365)) if s > 0 else np.nan, "annual_return": float(m * 365)}
                     for y, m, s in zip(g.mean().index, g.mean(), g.std())}
    return out


def pass_through_frame(panel, cdays, spec):
    """Eligible contract-days with next-day price return, expected and realised funding cost, and long return."""
    eligible = panel[panel["eligible"]]
    frame = eligible.merge(cdays[["day", "symbol", "price_factor", "funding_long"]], on=["day", "symbol"])
    frame = frame.assign(price_return=frame["price_factor"] - 1,
                         expected_funding_cost=frame[f"carry_{spec.signal_hours}h"] * HOURS_PER_DAY,
                         realised_funding_cost=-frame["funding_long"])
    frame["long_return"] = frame["price_return"] - frame["realised_funding_cost"]
    return frame


def main():
    spec = SPEC
    prices, funding = load_panels(unseal=True)
    days = day_range(spec.design_start, spec.holdout_end)
    chars = characteristics(prices, days, spec)
    panel = decision_panel(prices, funding, chars, spec, spec.design_start, spec.holdout_end)
    cdays = contract_days(prices, funding, panel.loc[panel["eligible"], ["day", "symbol"]], spec)
    frame = pass_through_frame(panel, cdays, spec)
    slopes = daily_pass_through(frame, "price_return", "expected_funding_cost", spec.fm_min_contracts, spec.fm_winsor)
    realised = daily_pass_through(frame, "realised_funding_cost", "expected_funding_cost", spec.fm_min_contracts,
                                  spec.fm_winsor)
    total = daily_pass_through(frame, "long_return", "expected_funding_cost", spec.fm_min_contracts, spec.fm_winsor)
    split = int(pd.Timestamp(spec.holdout_start, tz="UTC").value // 10**6)
    years = pd.to_datetime(frame["day"], unit="ms").dt.year
    q = frame.groupby([years, "day"])[f"carry_{spec.signal_hours}h"]
    spread = (q.quantile(0.9) - q.quantile(0.1)).groupby(level=0).mean() * 8760

    daily = pd.concat([pd.read_parquet(RESULTS_DIR / p / "daily_net_by_cost_model.parquet") for p in ("design", "holdout")])
    frame["funding_return"] = -frame["realised_funding_cost"]
    cols = ("long_return", "price_return", "funding_return")
    sorts = {"by_period": quintile_sorts(frame, "expected_funding_cost", cols,
                                         lambda d: np.where(d >= split, "holdout", "design")),
             "by_year": quintile_sorts(frame, "expected_funding_cost", cols,
                                       lambda d: pd.to_datetime(d, unit="ms").dt.year.to_numpy())}
    gates = {}
    for window in (90, 180, 365):
        timed, on = pass_through_gate(slopes, daily["B|calibrated"], window)
        gate_years = pd.to_datetime(timed.index, unit="ms").year
        gates[window] = {"days_on_by_year": {int(y): float(v) for y, v in on.groupby(gate_years).mean().items()},
                         "performance": annual_performance(pd.DataFrame({"gated": timed}))["gated"],
                         "sharpe_all": float(timed.mean() / timed.std() * np.sqrt(365))}

    mechanism = young_mechanism(frame, prices, spec)
    out = {
        "note": "Exploratory, run after the holdout was unsealed; not pre-registered.",
        "young_mechanism_A7c": mechanism,
        "pass_through_gate_B_calibrated": gates,
        "pass_through": {
            "design": summarize_slopes(slopes[slopes.index < split], spec.newey_west_lags),
            "holdout": summarize_slopes(slopes[slopes.index >= split], spec.newey_west_lags),
            "by_year": by_year(slopes, spec.newey_west_lags),
        },
        "total_return_slope": {
            "note": "slope of a long's price + funding return on expected funding cost; −1 = full carry, 0 = none",
            "design": summarize_slopes(total[total.index < split], spec.newey_west_lags),
            "holdout": summarize_slopes(total[total.index >= split], spec.newey_west_lags),
            "by_year": by_year(total, spec.newey_west_lags),
        },
        "signal_persistence": {"design": summarize_slopes(realised[realised.index < split], spec.newey_west_lags),
                               "holdout": summarize_slopes(realised[realised.index >= split], spec.newey_west_lags),
                               "by_year": by_year(realised, spec.newey_west_lags)},
        "quintile_sorts": sorts,
        "funding_spread_p90_p10_annualised": {int(y): float(v) for y, v in spread.items()},
        "annual_performance": annual_performance(daily),
    }
    path = RESULTS_DIR / "diagnostics.json"
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    slopes.to_parquet(RESULTS_DIR / "pass_through_daily.parquet")
    total.to_parquet(RESULTS_DIR / "total_slope_daily.parquet")
    print(json.dumps(out["pass_through"], indent=1, default=float))


if __name__ == "__main__":
    main()
