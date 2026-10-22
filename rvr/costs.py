# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Trading-cost models (PREREGISTRATION.md §5 and Amendment A1).

A cost model charges, per unit of traded notional q on a contract with average daily volume ADV and daily
volatility σ:

    fee + half_spread(ADV, σ) + Y × σ × √(q / ADV)

preregistered() uses the §5 spread tiers and Y = 1. main() fits the half-spread and Y from the order-book
sample in rvr/microstructure.py, exactly as Amendment A1 specifies, and calibrated() turns the fit into a
CostModel; nothing here sees a strategy return.

Run:  python -m rvr.costs        # fit A1 from data/processed/micro_*.parquet → results/design/costs.json
"""
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from rvr.config import PROCESSED_DIR, ROOT
from rvr.features import characteristics, day_range, lookup
from rvr.holdout import load_panels
from rvr.microstructure import SIZE_BUCKETS, walk_cost
from rvr.spec import SPEC

COSTS_PATH = ROOT / "results" / "design" / "costs.json"
MIN_ORDER_NOTIONAL = 1e3
MIN_ORDERS = 5
SPREAD_CLIP = (0.1e-4, 50e-4)


@dataclass(frozen=True)
class CostModel:
    name: str
    fee: float
    impact_coefficient: float
    spread_tiers: tuple = ()          # (ADV floor, half-spread) — used when spread_coef is empty
    spread_coef: tuple = ()           # (a, b, c): log hs = a + b·log ADV + c·log σ

    def half_spread(self, adv, sigma):
        adv, sigma = np.asarray(adv, dtype=float), np.asarray(sigma, dtype=float)
        if self.spread_coef:
            a, b, c = self.spread_coef
            return np.clip(np.exp(a + b * np.log(adv) + c * np.log(sigma)), *SPREAD_CLIP)
        conditions = [adv >= floor for floor, _ in self.spread_tiers]
        return np.select(conditions, [rate for _, rate in self.spread_tiers], default=self.spread_tiers[-1][1])

    def static(self, adv, sigma):
        """Fee plus half-spread per unit traded."""
        return self.fee + self.half_spread(adv, sigma)

    def unit_cost(self, notional, adv, sigma):
        """Cost per unit of traded notional (USDT): fee + half-spread + square-root impact."""
        return self.static(adv, sigma) + self.impact_coefficient * sigma * np.sqrt(notional / adv)

    def scaled(self, multiple):
        """Every component times `multiple` — for stress tests."""
        a, b, c = self.spread_coef or (None, None, None)
        return CostModel(
            name=f"{self.name} ×{multiple:g}", fee=self.fee * multiple,
            impact_coefficient=self.impact_coefficient * multiple,
            spread_tiers=tuple((floor, rate * multiple) for floor, rate in self.spread_tiers),
            spread_coef=(a + np.log(multiple), b, c) if self.spread_coef else ())


def preregistered(spec=SPEC):
    return CostModel("pre-registered (§5)", spec.fee, spec.impact_coefficient, spread_tiers=tuple(spec.spread_tiers))


def attach_characteristics(sample, chars):
    """Point-in-time ADV and σ_daily for each (day, symbol) — the same values the backtest charges with."""
    return sample.assign(adv=lookup(chars["adv"], sample["day"], sample["symbol"]),
                         sigma=lookup(chars["vol"], sample["day"], sample["symbol"]))


def fit_half_spread(execution):
    """A1 step 1: OLS of log half-spread on log ADV and log σ."""
    s = execution[execution["available"] & (execution["half_spread"] > 0)
                  & (execution["adv"] > 0) & (execution["sigma"] > 0)]
    X = np.column_stack([np.ones(len(s)), np.log(s["adv"]), np.log(s["sigma"])])
    y = np.log(s["half_spread"].to_numpy())
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return {"coef": [float(c) for c in coef], "r2": float(1 - resid.var() / y.var()), "n": int(len(s)),
            "resid_sd": float(resid.std())}


def impact_points(execution):
    """A1 step 2: one point per (contract, day, size bucket ≥ $1K) with at least 5 orders."""
    rows = []
    for b in range(len(SIZE_BUCKETS) - 1):
        if SIZE_BUCKETS[b + 1] <= MIN_ORDER_NOTIONAL:
            continue
        cols = [f"orders_b{b}", f"notional_b{b}", f"slippage_b{b}"]
        if not set(cols) <= set(execution.columns):
            continue
        g = execution[["symbol", "day", "half_spread", "adv", "sigma", *cols]].dropna()
        g = g[(g[cols[0]] >= MIN_ORDERS) & (g[cols[1]] >= MIN_ORDER_NOTIONAL) & (g["adv"] > 0) & (g["sigma"] > 0)]
        rows.append(pd.DataFrame({
            "symbol": g["symbol"], "day": g["day"], "bucket": b, "orders": g[cols[0]], "notional": g[cols[1]],
            "y": g[cols[2]] - g["half_spread"], "x": g["sigma"] * np.sqrt(g[cols[1]] / g["adv"]),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fit_impact(points, n_boot=2000, seed=SPEC.bootstrap_seed):
    """Ŷ by order-weighted least squares through the origin; 95% CI from a bootstrap over contracts."""
    def estimate(p):
        w = p["orders"].to_numpy(dtype=float)
        return float(np.sum(w * p["x"] * p["y"]) / np.sum(w * p["x"] ** 2))

    y_hat = estimate(points)
    rng = np.random.default_rng(seed)
    groups = {s: g for s, g in points.groupby("symbol")}
    names = np.array(list(groups))
    draws = [estimate(pd.concat([groups[s] for s in rng.choice(names, len(names))])) for _ in range(n_boot)]
    by_bucket = points.groupby("bucket").apply(
        lambda g: pd.Series({"points": len(g), "orders": int(g["orders"].sum()), "mean_notional": g["notional"].mean(),
                             "mean_excess_slippage": np.average(g["y"], weights=g["orders"]),
                             "mean_model": y_hat * np.average(g["x"], weights=g["orders"])}), include_groups=False)
    return {"Y": y_hat, "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "points": int(len(points)), "contracts": int(len(names)), "orders": int(points["orders"].sum()),
            "by_bucket": {int(k): {c: float(v) for c, v in row.items()} for k, row in by_bucket.iterrows()}}


def calibrated(fit, spec=SPEC):
    return CostModel("calibrated (A1)", spec.fee, fit["impact"]["Y"], spread_coef=tuple(fit["half_spread"]["coef"]))


def depth_checks(depth, model, trade_notional):
    """
    A1 checks (reported, not used): (i) book-walk cost from measured depth for a trade of `trade_notional`
    against the calibrated half-spread + impact, by year; (ii) year drift in log 1% depth given ADV and σ.
    """
    d = depth[depth["available"] & (depth["adv"] > 0) & (depth["sigma"] > 0)].copy()
    levels = np.column_stack([(d[f"bid_{k}"] + d[f"ask_{k}"]) / 2 for k in range(1, 6)])
    ok = np.isfinite(levels).all(axis=1) & (levels[:, 0] > 0)
    d, levels = d[ok], levels[ok]
    hs = model.half_spread(d["adv"], d["sigma"])
    q = np.full(len(d), trade_notional)
    d["walk"] = walk_cost(q, hs, levels)
    d["model"] = hs + model.impact_coefficient * d["sigma"] * np.sqrt(q / d["adv"])
    d["year"] = pd.to_datetime(d["day"], unit="ms").dt.year
    by_year = d.groupby("year").agg(contract_days=("walk", "size"), walk_bp=("walk", "median"),
                                    model_bp=("model", "median"), depth_1pct=("bid_1", "median")) \
        .assign(walk_bp=lambda f: f["walk_bp"] * 1e4, model_bp=lambda f: f["model_bp"] * 1e4)

    years = sorted(d["year"].unique())
    X = np.column_stack([np.ones(len(d)), np.log(d["adv"]), np.log(d["sigma"])]
                        + [(d["year"] == y).to_numpy(float) for y in years[1:]])
    coef, *_ = np.linalg.lstsq(X, np.log(levels[:, 0]), rcond=None)
    drift = {int(y): float(np.exp(c) - 1) for y, c in zip(years[1:], coef[3:])}
    return {"trade_notional": trade_notional,
            "by_year": {int(y): {k: float(v) for k, v in row.items()} for y, row in by_year.iterrows()},
            "depth_elasticity_adv": float(coef[1]), "depth_elasticity_sigma": float(coef[2]),
            "depth_change_vs_first_year": drift}


def main():
    execution = pd.read_parquet(PROCESSED_DIR / "micro_execution.parquet")
    depth = pd.read_parquet(PROCESSED_DIR / "micro_bookDepth.parquet")
    prices, _ = load_panels()
    days = day_range(SPEC.design_start, SPEC.design_end)
    chars = characteristics(prices, days, SPEC)
    execution, depth = attach_characteristics(execution, chars), attach_characteristics(depth, chars)

    fit = {"half_spread": fit_half_spread(execution), "impact": fit_impact(impact_points(execution))}
    model = calibrated(fit)
    typical_trade = SPEC.aum * 0.55 / 65  # baseline: 0.55 daily turnover over ~65 positions
    fit["depth_checks"] = depth_checks(depth, model, typical_trade)
    fit["model"] = asdict(model)
    fit["sample"] = {"execution_contract_days": int(execution["available"].sum()),
                     "depth_contract_days": int(depth["available"].sum())}
    COSTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    COSTS_PATH.write_text(json.dumps(fit, indent=2, default=float) + "\n")
    print(json.dumps(fit, indent=2, default=float))


def load_calibrated(path=COSTS_PATH, spec=SPEC):
    return calibrated(json.loads(path.read_text()), spec)


if __name__ == "__main__":
    main()
