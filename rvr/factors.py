# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Is carry a new premium, or known crypto factors in disguise? (PREREGISTRATION.md Amendment A2.)

Factors are built from the same eligible universe, decision timing and next-day long returns (price +
funding, before costs) as the carry portfolio, so any difference in alpha is not a difference in
plumbing. Sorting characteristics use daily closes up to the decision day's 00:00 UTC close only.
"""
import numpy as np
import pandas as pd

from rvr.features import daily_closes, lookup

FACTOR_SPECS = {
    # name: (characteristic, long the high tercile?)
    "SIZE": ("adv", False),
    "MOM": ("ret_21d", True),
    "REV": ("ret_7d", False),
    "LOWVOL": ("vol", False),
}
MODELS = {
    "a_market": ["MKT"],
    "b_three_factor": ["MKT", "SIZE", "MOM"],
    "c_five_factor": ["MKT", "SIZE", "MOM", "REV", "LOWVOL"],
}


def sort_characteristics(prices, panel, days):
    """7- and 21-day past returns for each panel row, from closes known at the decision."""
    closes = daily_closes(prices, days)
    out = {}
    for k in (7, 21):
        out[f"ret_{k}d"] = lookup(closes / closes.shift(k) - 1, panel["day"], panel["symbol"])
    return panel.assign(**out)


def tercile_weights(panel, column, long_high, min_eligible):
    """Long the top (or bottom) tercile of `column`, short the other end; each leg gross 0.5, equal weight."""
    p = panel.loc[panel["eligible"] & panel[column].notna(), ["day", "symbol", column]]
    breadth = panel[panel["eligible"]].groupby("day").size()
    p = p[p["day"].map(breadth).fillna(0) >= min_eligible]
    group = p.groupby("day")[column]
    size, rank = group.transform("size"), group.rank(method="first")
    n = np.floor(size / 3)
    top, bottom = rank > size - n, rank <= n
    long_leg, short_leg = (top, bottom) if long_high else (bottom, top)
    w = np.where(long_leg, 0.5 / n, np.where(short_leg, -0.5 / n, 0.0))
    out = p[["day", "symbol"]].assign(w=np.where(n > 0, w, 0.0))
    return out[out["w"] != 0]


def market_weights(panel, min_eligible):
    p = panel.loc[panel["eligible"], ["day", "symbol"]]
    size = p.groupby("day")["symbol"].transform("size")
    return p[size >= min_eligible].assign(w=1.0 / size[size >= min_eligible])


def factor_returns(panel, cdays, days, spec):
    """Daily gross return of each factor: Σ w × (price + funding) of the long position, before costs."""
    long_return = cdays.assign(r=cdays["price_factor"] - 1 + cdays["funding_long"])[["day", "symbol", "r"]]
    weights = {"MKT": market_weights(panel, spec.min_eligible)}
    for name, (column, long_high) in FACTOR_SPECS.items():
        weights[name] = tercile_weights(panel, column, long_high, spec.min_eligible)
    out = {}
    for name, w in weights.items():
        merged = w.merge(long_return, on=["day", "symbol"], how="left")
        # an order that never filled holds nothing, as in the backtest
        out[name] = (merged["w"] * merged["r"].fillna(0.0)).groupby(merged["day"]).sum()
    return pd.DataFrame(out).reindex(days)


def newey_west_ols(y, X, lags):
    """OLS coefficients with Bartlett-kernel HAC standard errors. X includes the constant."""
    y, X = np.asarray(y, dtype=float), np.asarray(X, dtype=float)
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = X * (y - X @ beta)[:, None]
    s = u.T @ u
    for k in range(1, lags + 1):
        gamma = u[k:].T @ u[:-k]
        s += (1 - k / (lags + 1)) * (gamma + gamma.T)
    se = np.sqrt(np.diag(xtx_inv @ s @ xtx_inv))
    resid = y - X @ beta
    return beta, beta / se, float(1 - resid.var() / y.var()) if y.var() > 0 else np.nan


def spanning_test(strategy_returns, factors, spec):
    """Regress the strategy's daily return on each factor model (A2); alpha annualised (× 365)."""
    results = {}
    for model, names in MODELS.items():
        frame = pd.concat([strategy_returns.rename("y"), factors[names]], axis=1).dropna()
        X = np.column_stack([np.ones(len(frame))] + [frame[c] for c in names])
        beta, t, r2 = newey_west_ols(frame["y"], X, spec.newey_west_lags)
        results[model] = {"days": int(len(frame)), "alpha_annual": float(beta[0] * 365), "alpha_t": float(t[0]),
                          "r2": r2, "loadings": {c: {"beta": float(b), "t": float(tt)}
                                                 for c, b, tt in zip(names, beta[1:], t[1:])}}
    return results


def spanning_verdict(results, critical=1.96):
    alpha_t = results["c_five_factor"]["alpha_t"]
    return "Not spanned by known factors" if abs(alpha_t) >= critical else "Spanned by known factors"
