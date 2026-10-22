# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Maker execution (PREREGISTRATION.md Amendment A5b): post at the touch, cross the spread for whatever is
left after τ.

simulate_day() replays real quotes and fills around 01:00 UTC on A1's execution sample and records, for each
simulated limit order, how much filled and what the unfilled part cost. fit_fill(), fit_touch() and
fit_adverse() turn those rows into the pre-registered model; MakerCostModel charges it in the backtest, with
the taker fee taken from A1's model. Nothing here reads a strategy return.

Run:  python -m rvr.execution            # simulate on A1's execution sample, then fit → results/design/maker.json
"""
import io
import json
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from rvr.config import HOUR_MS, PROCESSED_DIR, ROOT
from rvr.costs import CostModel, attach_characteristics, load_calibrated
from rvr.download import _run_all
from rvr.microstructure import BOOK_TICKER_RANGE, MICRO_DIR, daily_key, fetch_verified, read_quotes
from rvr.spec import SPEC

TAU_MS = 30 * 60_000
POST_OFFSETS_MIN = (-30, -25, -20, -15, -10, -5, 0)
SIZES = (1e3, 3e3, 1e4, 3e4, 1e5)
MAKER_FEE = 2.0e-4
MAKER_PATH = ROOT / "results" / "design" / "maker.json"


def read_trades(csv_bytes, lo, hi):
    trades = pd.read_csv(io.BytesIO(csv_bytes), usecols=["price", "quantity", "transact_time", "is_buyer_maker"])
    trades = trades[(trades["transact_time"] > lo) & (trades["transact_time"] <= hi)]
    sell = trades["is_buyer_maker"].astype(str).str.lower() == "true"  # buyer was the maker: a taker sold
    return trades.assign(taker_sell=sell.to_numpy())


def _quote_at(quotes, t):
    """The quote standing at time t (last update at or before t)."""
    i = np.searchsorted(quotes["transaction_time"].to_numpy(), t, side="right") - 1
    return None if i < 0 else quotes.iloc[i]


def simulate_order(quotes, trades, post, side, notional, tau_ms=TAU_MS):
    """
    One limit order at the touch: side +1 buys at the bid, −1 sells at the ask. Joins the back of the queue and
    fills by the quantity of taker flow traded at or through its price, after the queue ahead; never repriced.
    """
    q0, q1 = _quote_at(quotes, post), _quote_at(quotes, post + tau_ms)
    if q0 is None or q1 is None:
        return None
    bid0, ask0 = q0["best_bid_price"], q0["best_ask_price"]
    mid0, mid1 = (bid0 + ask0) / 2, (q1["best_bid_price"] + q1["best_ask_price"]) / 2
    price = bid0 if side > 0 else ask0
    queue = q0["best_bid_qty"] if side > 0 else q0["best_ask_qty"]
    size = notional / price
    window = trades[(trades["transact_time"] > post) & (trades["transact_time"] <= post + tau_ms)]
    hits = window[window["taker_sell"]] if side > 0 else window[~window["taker_sell"]]
    # A resting order is in the way of every trade at or through its price: those trades fill the queue
    # ahead first, then this order, by their quantity (A5b bug fix: a print through the price is not a
    # full fill — a large order would have absorbed it).
    reached = (hits["price"] <= price) if side > 0 else (hits["price"] >= price)
    filled = float(np.clip(hits.loc[reached, "quantity"].sum() - queue, 0.0, size))
    return {"side": side, "notional": notional, "fill": filled / size, "touch_notional": queue * price,
            "half_spread": (ask0 - bid0) / (2 * mid0),
            "half_spread_tau": (q1["best_ask_price"] - q1["best_bid_price"]) / (2 * mid1),
            "move": side * (mid1 / mid0 - 1)}


def simulate_day(job, out_dir=MICRO_DIR, fetch=fetch_verified):
    symbol, day = job
    shard = out_dir / "maker" / f"{day}_{symbol}.parquet"
    if shard.exists():
        return "cached"
    t_exec = int(pd.Timestamp(day, tz="UTC").value // 10**6) + SPEC.execution_hour * HOUR_MS
    lo, hi = t_exec + min(POST_OFFSETS_MIN) * 60_000, t_exec + max(POST_OFFSETS_MIN) * 60_000 + TAU_MS
    quotes_csv, trades_csv = fetch(daily_key("bookTicker", symbol, day)), fetch(daily_key("aggTrades", symbol, day))
    rows = []
    if quotes_csv is not None and trades_csv is not None:
        quotes, trades = read_quotes(quotes_csv, lo, hi + 1), read_trades(trades_csv, lo, hi)
        for offset in POST_OFFSETS_MIN:
            for side in (1, -1):
                for notional in SIZES:
                    r = simulate_order(quotes, trades, t_exec + offset * 60_000, side, notional)
                    if r is not None:
                        rows.append(r | {"post_offset_min": offset})
    frame = pd.DataFrame(rows).assign(symbol=symbol, day=int(pd.Timestamp(day, tz="UTC").value // 10**6))
    shard.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(shard)
    return "simulated" if rows else "missing"


def fit_fill(sims):
    """Fractional logit: f = logistic(α + β·log(q / touch) + γ·log σ), by maximum quasi-likelihood."""
    s = sims[(sims["touch_notional"] > 0) & (sims["sigma"] > 0)]
    X = np.column_stack([np.ones(len(s)), np.log(s["notional"] / s["touch_notional"]), np.log(s["sigma"])])
    y = s["fill"].to_numpy().clip(0, 1)

    def loss(theta):
        p = 1 / (1 + np.exp(-(X @ theta)))
        p = p.clip(1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    theta = minimize(loss, np.zeros(3), method="BFGS").x
    return {"coef": [float(c) for c in theta], "n": int(len(s)), "mean_fill": float(y.mean())}


def fit_touch(execution):
    """OLS of log touch notional on log ADV and log σ, from A1's quoted touch sizes."""
    s = execution[execution["available"] & (execution["touch_notional"] > 0) & (execution["adv"] > 0)
                  & (execution["sigma"] > 0)]
    X = np.column_stack([np.ones(len(s)), np.log(s["adv"]), np.log(s["sigma"])])
    coef, *_ = np.linalg.lstsq(X, np.log(s["touch_notional"].to_numpy()), rcond=None)
    return {"coef": [float(c) for c in coef], "n": int(len(s))}


def fit_adverse(sims):
    """κ: mean signed mid move over τ of orders not fully filled, in units of daily volatility."""
    s = sims[(sims["fill"] < 1) & (sims["sigma"] > 0)]
    return {"kappa": float((s["move"] / s["sigma"]).mean()), "n": int(len(s)),
            "kappa_filled": float((sims.loc[sims["fill"] >= 1, "move"] / sims.loc[sims["fill"] >= 1, "sigma"]).mean())}


@dataclass(frozen=True)
class MakerCostModel:
    """A5b: expected cost per unit of a post-then-cross order, on top of A1's taker model for the remainder."""
    name: str
    taker: CostModel
    fill_coef: tuple
    touch_coef: tuple
    kappa: float
    maker_fee: float = MAKER_FEE

    def fill(self, notional, adv, sigma):
        touch = np.exp(self.touch_coef[0] + self.touch_coef[1] * np.log(adv) + self.touch_coef[2] * np.log(sigma))
        a, b, c = self.fill_coef
        z = a + b * np.log(np.maximum(notional, 1.0) / touch) + c * np.log(sigma)
        return 1 / (1 + np.exp(-z))

    def unit_cost(self, notional, adv, sigma):
        hs = self.taker.half_spread(adv, sigma)
        f = self.fill(notional, adv, sigma)
        taker = self.taker.fee + hs + self.kappa * sigma + self.taker.impact_coefficient * sigma * np.sqrt(notional / adv)
        return f * (self.maker_fee - hs) + (1 - f) * taker


def load_maker(path=MAKER_PATH, spec=SPEC):
    fit = json.loads(path.read_text())
    return MakerCostModel("maker, then taker (A5b)", load_calibrated(spec=spec), tuple(fit["fill"]["coef"]),
                          tuple(fit["touch"]["coef"]), fit["adverse"]["kappa"])


def main():
    execution = pd.read_parquet(PROCESSED_DIR / "micro_execution.parquet")
    lo, hi = BOOK_TICKER_RANGE
    days = execution.assign(date=pd.to_datetime(execution["day"], unit="ms").dt.strftime("%Y-%m-%d"))
    days = days[days["available"] & days["date"].between(lo, hi)]
    jobs = list(zip(days["symbol"], days["date"]))
    print(f"{len(jobs)} contract-days to simulate", flush=True)
    outcomes, failures = _run_all(jobs, simulate_day, 24, "simulated")
    print(pd.Series([o for o in outcomes if o]).value_counts().to_dict(), flush=True)
    if failures:
        print(failures[:5], file=sys.stderr)
        return 1

    from rvr.features import characteristics, day_range
    from rvr.holdout import load_panels
    prices, _ = load_panels()
    chars = characteristics(prices, day_range(SPEC.design_start, SPEC.design_end), SPEC)
    sims = pd.concat([pd.read_parquet(p) for p in sorted((MICRO_DIR / "maker").glob("*.parquet"))], ignore_index=True)
    sims = attach_characteristics(sims, chars)
    sims.to_parquet(PROCESSED_DIR / "maker_simulations.parquet")
    fit = {"fill": fit_fill(sims), "touch": fit_touch(attach_characteristics(execution, chars)),
           "adverse": fit_adverse(sims), "simulated_orders": int(len(sims)),
           "fill_by_size": {f"{q:,.0f}": float(sims.loc[sims["notional"] == q, "fill"].mean()) for q in SIZES}}
    MAKER_PATH.write_text(json.dumps(fit, indent=2) + "\n")
    print(json.dumps(fit, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
