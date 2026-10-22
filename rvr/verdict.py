# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""The pre-registered verdict tables (PREREGISTRATION.md §6, §7, §10), translated literally."""

NO_PREMIUM = "No significant carry premium to explain"
RISK_PREMIUM = "Risk premium: managing crash risk removes the return"
EDGE_SURVIVES = "Edge survives: part of carry is not crash compensation"
INCONCLUSIVE = "Inconclusive"
CONTROLS_FAILED = "Inconclusive: controls did not remove crash risk"


def _includes_zero(ci):
    return ci[0] <= 0 <= ci[1]


def portfolio_verdict(baseline, managed, spec):
    """baseline/managed: rvr.stats.summarize() outputs for B and M."""
    if _includes_zero(baseline["sharpe_ci95"]):
        return NO_PREMIUM
    tail_b, tail_m = abs(baseline["es_1pct_per_vol"]), abs(managed["es_1pct_per_vol"])
    if tail_m > (1 - spec.validity_es_reduction) * tail_b:
        return CONTROLS_FAILED
    if managed["sharpe"] <= 0.5 * baseline["sharpe"] and _includes_zero(managed["sharpe_ci95"]):
        return RISK_PREMIUM
    if managed["sharpe_ci95"][0] > 0:
        return EDGE_SURVIVES
    return INCONCLUSIVE


def pricing_verdict(result, carry_col, critical=1.96):
    """result: rvr.pricing.pricing_test() output."""
    one = result["model1"]["coefficients"][carry_col]
    two = result["model2"]["coefficients"][carry_col]
    if abs(one["newey_west_t"]) < critical:
        return "No cross-sectional carry premium"
    if abs(two["mean"]) <= 0.5 * abs(one["mean"]) and abs(two["newey_west_t"]) < critical:
        return "Risk premium: crash exposure explains carry"
    if abs(two["newey_west_t"]) >= critical and (two["mean"] > 0) == (one["mean"] > 0):
        return "Edge survives crash-risk controls"
    return INCONCLUSIVE


def holdout_confirms(design, holdout):
    """§10: B and M keep their design-period signs, and M's tail per unit of volatility stays below B's."""
    same_sign = all((design[k]["sharpe"] > 0) == (holdout[k]["sharpe"] > 0) for k in ("B", "M"))
    thinner_tail = abs(holdout["M"]["es_1pct_per_vol"]) < abs(holdout["B"]["es_1pct_per_vol"])
    return bool(same_sign and thinner_tail)
