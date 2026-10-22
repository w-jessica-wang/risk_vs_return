# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Replication on other venues, Hyperliquid and dYdX, on daily bars (PREREGISTRATION.md Amendments A3, A5a, A6).

Only the plumbing differs from Binance: decisions at 00:00 UTC, entry at the open of that day's candle and
exit at its close. The weights (rvr.strategy), P&L and costs (rvr.backtest.pnl), Fama-MacBeth test
(rvr.pricing) and statistics (rvr.stats) are the same functions the Binance results come from.

Point-in-time rules for a decision at day d 00:00:
  - a funding rate is known strictly after its calc_time
  - a daily candle is known only after it closes, so the latest usable candle is day d − 1's
  - eligibility, ADV, volatility and every characteristic use candles up to d − 1 only

Run:  python -m rvr.replication [--venue dydx]              # design period
      python -m rvr.replication --holdout [--venue dydx]    # the holdout year (already unsealed for Binance)
"""
import argparse
import json

import numpy as np
import pandas as pd

from rvr.backtest import pnl
from rvr.config import HOUR_MS, PROCESSED_DIR, ROOT
from rvr.costs import COSTS_PATH, CostModel, load_calibrated
from rvr.features import DAY_MS, coskewness, day_range, downside_beta
from rvr.holdout import HOLDOUT_START_MS
from rvr.pricing import pricing_test
from rvr.spec import SPEC
from rvr.stats import summarize
from rvr.strategy import buffered_weights, quintile_weights
from rvr.verdict import pricing_verdict

RESULTS_ROOT = ROOT / "results"
VENUES = {
    # name: (processed-file prefix, base taker fee, BTC ticker, first design day) — A3 and A5a
    "hyperliquid": ("hl", 4.5e-4, "BTC", "2023-06-01"),
    "dydx": ("dydx", 5.0e-4, "BTC-USD", "2023-10-01"),
}
FLOORS = (5e6, 2e6, 1e6)
FLOOR_COVERAGE = 0.9
CONTROLS = ["downside_beta", "coskew", "log_adv", "momentum", "vol", "max_ret"]


def load(processed_dir=PROCESSED_DIR, unseal=False, holdout_start_ms=HOLDOUT_START_MS, venue="hyperliquid"):
    """(daily candles, funding); nothing at or after the holdout start unless unsealed."""
    prefix = VENUES[venue][0]
    candles = pd.read_parquet(processed_dir / f"{prefix}_daily.parquet").rename(columns={"coin": "symbol"})
    funding = pd.read_parquet(processed_dir / f"{prefix}_funding.parquet").rename(columns={"coin": "symbol"})
    if not unseal:
        candles = candles[candles["close_time"] < holdout_start_ms]
        funding = funding[funding["calc_time"] < holdout_start_ms]
    return candles.reset_index(drop=True), funding.reset_index(drop=True)


def wide(candles, column, days):
    """day × symbol frame of a candle column, indexed by the candle's own open day."""
    return candles.pivot_table(index="open_time", columns="symbol", values=column, aggfunc="last").reindex(days)


def run_lengths(traded, restart_gap_days):
    """Days in each contract's current run of trading, judged from traded days only (a gap > restart_gap ends a run)."""
    out = np.zeros(traded.shape)
    t = traded.to_numpy()
    for j in range(t.shape[1]):
        run, gap = 0, restart_gap_days + 1
        for i in range(t.shape[0]):
            if t[i, j]:
                run = run + 1 if gap <= restart_gap_days else 1
                gap = 0
            else:
                gap += 1
            out[i, j] = run if gap <= restart_gap_days else 0
    return pd.DataFrame(out, index=traded.index, columns=traded.columns)


def panel(candles, funding, days, spec, floor, hedge="BTC"):
    """
    One row per (decision day, contract): eligibility (A3), carry signal and §7 characteristics. Row d uses
    only candles that opened on or before d − 1 (closed by d 00:00) and funding published before d 00:00.
    """
    traded_candles = candles[(candles["trades"] > 0) & (candles["quote_volume"] > 0)]
    close = wide(traded_candles, "close", days)
    volume = wide(candles, "quote_volume", days).fillna(0.0)
    traded = close.notna()
    lag = lambda f: f.shift(1)  # noqa: E731 — row d sees candle d − 1 at most

    adv = lag(volume.rolling(spec.adv_days, min_periods=1).mean())
    runs = lag(run_lengths(traded, spec.restart_gap_hours // 24))
    returns = close / close.shift(1) - 1
    market = returns[hedge] if hedge in returns else pd.Series(np.nan, index=days)
    chars = {
        "adv": adv,
        "log_adv": np.log(adv.where(adv > 0)),
        "vol": lag(returns.rolling(spec.vol_days, min_periods=spec.vol_min_obs).std()),
        "momentum": lag(close / close.shift(spec.momentum_days) - 1),
        "max_ret": lag(returns.rolling(spec.vol_days, min_periods=spec.vol_min_obs).max()),
        "downside_beta": lag(downside_beta(returns, market, spec.beta_days, spec.beta_min_down_days, spec.beta_min_days)),
        "coskew": lag(coskewness(returns, market, spec.beta_days, spec.beta_min_days)),
    }
    grid = pd.MultiIndex.from_product([days, close.columns], names=["day", "symbol"]).to_frame(index=False)
    out = grid.assign(**{name: frame.stack(future_stack=True).reindex(pd.MultiIndex.from_frame(grid)).to_numpy()
                         for name, frame in chars.items()})
    yesterday = traded.shift(1, fill_value=False)
    out["traded_yesterday"] = yesterday.stack(future_stack=True).reindex(pd.MultiIndex.from_frame(grid)).fillna(False).to_numpy(bool)
    out["run_days"] = runs.stack(future_stack=True).reindex(pd.MultiIndex.from_frame(grid)).fillna(0).to_numpy()

    # Funding published in [d − 72 h, d) is the three daily buckets before d; fresh = a settlement in [d − 2 h, d).
    if spec.signal_hours % 24:
        raise ValueError("daily replication needs a signal window in whole days")
    f = funding.assign(bucket=funding["calc_time"] // DAY_MS * DAY_MS)
    by_day = f.pivot_table(index="bucket", columns="symbol", values="rate", aggfunc="sum")
    by_day = by_day.reindex(index=days.union(by_day.index), columns=close.columns)
    carry = by_day.rolling(spec.signal_hours // 24, min_periods=1).sum().shift(1).reindex(days) / spec.signal_hours
    late = f[f["calc_time"] % DAY_MS >= DAY_MS - spec.fresh_funding_intervals * HOUR_MS]
    fresh = late.pivot_table(index="bucket", columns="symbol", values="rate", aggfunc="size")
    fresh = fresh.reindex(index=days.union(fresh.index), columns=close.columns).shift(1).reindex(days).notna()
    out[f"carry_{spec.signal_hours}h"] = carry.stack(future_stack=True).reindex(pd.MultiIndex.from_frame(grid)).to_numpy()
    out["fresh_funding"] = fresh.stack(future_stack=True).reindex(pd.MultiIndex.from_frame(grid)).fillna(False).to_numpy(bool)
    out["eligible"] = (out["traded_yesterday"] & (out["run_days"] >= spec.min_history_hours // 24)
                       & out["fresh_funding"] & (out["adv"] >= floor) & out[f"carry_{spec.signal_hours}h"].notna())
    return out, chars


def choose_floor(candles, funding, days, spec, hedge="BTC"):
    """A3: the largest floor leaving ≥ 20 eligible contracts on ≥ 90% of design days. Volume data only."""
    coverage = {}
    for floor in FLOORS:
        p, _ = panel(candles, funding, days, spec, floor, hedge)
        breadth = p.groupby("day")["eligible"].sum().reindex(days, fill_value=0)
        coverage[floor] = float((breadth >= spec.min_eligible).mean())
    chosen = next((f for f in FLOORS if coverage[f] >= FLOOR_COVERAGE), FLOORS[-1])
    return chosen, coverage


def contract_days(candles, funding, keys, delisted):
    """
    Entry at the open of day d's candle, exit at its close; funding settled in (d, d + 1 day]. A delisted coin's
    last traded candle exits with the §4 haircut. Same columns rvr.backtest.pnl expects.
    """
    traded = candles[(candles["trades"] > 0) & (candles["quote_volume"] > 0)]
    c = keys.merge(traded[["symbol", "open_time", "open", "close"]].rename(columns={"open_time": "day"}),
                   on=["day", "symbol"], how="left")
    last_day = traded.groupby("symbol")["open_time"].max()
    c["halted"] = c["symbol"].isin(delisted) & (c["day"] == c["symbol"].map(last_day)) & c["open"].notna()
    c["entry"], c["exit"] = c["open"], c["close"]
    c["price_factor"] = c["close"] / c["open"]
    f = funding.assign(day=(funding["settle_hour"] - 1) // DAY_MS * DAY_MS)  # settles in (d, d + 1 day]
    paid = (-f["rate"]).groupby([f["day"], f["symbol"]]).sum().rename("funding_long").reset_index()
    c = c.merge(paid, on=["day", "symbol"], how="left")
    c["funding_long"] = c["funding_long"].fillna(0.0)
    return c[["day", "symbol", "entry", "exit", "halted", "price_factor", "funding_long"]]


def cost_models(spec, fee):
    pre = CostModel("pre-registered §5, venue fee", fee, spec.impact_coefficient, spread_tiers=tuple(spec.spread_tiers))
    models = {"preregistered": pre}
    if COSTS_PATH.exists():
        a1 = load_calibrated(spec=spec)
        models["calibrated"] = CostModel("A1 calibration (Binance), venue fee", fee, a1.impact_coefficient,
                                         spread_coef=a1.spread_coef)
    return models


def replication_verdict(gross_ci, fm_model1, binance_sign, critical=1.96):
    portfolio_ok = gross_ci[0] > 0
    pricing_ok = abs(fm_model1["newey_west_t"]) >= critical and np.sign(fm_model1["mean"]) == binance_sign
    if portfolio_ok and pricing_ok:
        return "Replicates"
    if portfolio_ok or pricing_ok:
        return "Partly replicates"
    return "Does not replicate"


def run(period="design", spec=SPEC, venue="hyperliquid"):
    _, fee, hedge, design_start = VENUES[venue]
    holdout = period == "holdout"
    candles, funding = load(unseal=holdout, venue=venue)
    universe = pd.read_parquet(ROOT / "data" / "raw" / venue / "universe.parquet")
    delisted = set(universe.loc[universe["delisted"], "coin"])
    all_days = day_range(design_start, spec.holdout_end if holdout else spec.design_end)
    design_days = day_range(design_start, spec.design_end)
    eval_days = day_range(spec.holdout_start, spec.holdout_end) if holdout else design_days

    floor, coverage = choose_floor(*load(venue=venue), design_days, spec, hedge)  # design-period volumes only
    p, chars = panel(candles, funding, all_days, spec, floor, hedge)
    carry = f"carry_{spec.signal_hours}h"
    keys = p.loc[p["eligible"], ["day", "symbol"]]
    cdays = contract_days(candles, funding, keys, delisted)
    weights = {"B": quintile_weights(p, carry, spec), "B_buffer": buffered_weights(p, carry, spec)}

    btc = cdays[cdays["symbol"] == hedge].set_index("day")
    market = (btc["price_factor"] - 1 + btc["funding_long"]).reindex(all_days)
    out = {"venue": venue, "period": period, "floor": floor, "floor_coverage": {f"{k:,.0f}": v for k, v in coverage.items()},
           "avg_eligible": float(p[p["eligible"] & p["day"].isin(eval_days)].groupby("day").size().mean()),
           "strategies": {}}
    report_days = None
    for name, w in weights.items():
        for key, model in cost_models(spec, fee).items():
            daily = pnl(w, cdays, chars, all_days, spec, cost_model=model)
            if report_days is None:
                traded = daily.loc[daily.index.isin(eval_days), "gross_exposure"] > 0
                report_days = eval_days[eval_days >= traded[traded].index.min()]
            f = daily.loc[report_days]
            stats = summarize(f["net"].to_numpy(), market.reindex(report_days).fillna(0.0).to_numpy(), spec)
            gross = (f["price"] + f["funding"]).to_numpy()
            stats.update({"gross_sharpe": float(gross.mean() / gross.std(ddof=1) * np.sqrt(365)),
                          "annual_funding": float(f["funding"].mean() * 365), "annual_price": float(f["price"].mean() * 365),
                          "annual_cost": float(f["cost"].mean() * 365), "avg_daily_turnover": float(f["turnover"].mean())})
            if name == "B" and key == "preregistered":
                gross_stats = summarize(gross, market.reindex(report_days).fillna(0.0).to_numpy(), spec)
                stats["gross_sharpe_ci95"] = gross_stats["sharpe_ci95"]
            out["strategies"].setdefault(name, {})[key] = stats

    long_return = cdays.assign(long_return=cdays["price_factor"] - 1 + cdays["funding_long"])
    fm = p[p["eligible"] & p["day"].isin(report_days)].merge(long_return[["day", "symbol", "long_return"]],
                                                             on=["day", "symbol"], how="left")
    pricing = pricing_test(fm, "long_return", carry, CONTROLS, spec)
    binance = json.loads((ROOT / "results" / "design" / "summary.json").read_text())["pricing"]
    sign = np.sign(binance["model1"]["coefficients"][carry]["mean"])
    out["report_days"] = [str(pd.to_datetime(report_days[0], unit="ms").date()),
                          str(pd.to_datetime(report_days[-1], unit="ms").date())]
    out["pricing"] = pricing
    out["verdicts"] = {
        "replication": replication_verdict(out["strategies"]["B"]["preregistered"]["gross_sharpe_ci95"],
                                           pricing["model1"]["coefficients"][carry], sign),
        "pricing": pricing_verdict(pricing, carry),
    }
    out_dir = RESULTS_ROOT / venue
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{period}.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def pass_through(venue, spec=SPEC):
    """Amendment A6: the §5 pass-through regression on a replication venue, over all its data."""
    from rvr.diagnostics import by_year, daily_pass_through, summarize_slopes

    _, _, hedge, design_start = VENUES[venue]
    candles, funding = load(unseal=True, venue=venue)
    universe = pd.read_parquet(ROOT / "data" / "raw" / venue / "universe.parquet")
    days = day_range(design_start, spec.holdout_end)
    floor, _ = choose_floor(*load(venue=venue), day_range(design_start, spec.design_end), spec, hedge)
    p, _ = panel(candles, funding, days, spec, floor, hedge)
    eligible = p[p["eligible"]]
    cdays = contract_days(candles, funding, eligible[["day", "symbol"]], set(universe.loc[universe["delisted"], "coin"]))
    frame = eligible.merge(cdays[["day", "symbol", "price_factor"]], on=["day", "symbol"])
    frame = frame.assign(price_return=frame["price_factor"] - 1,
                         expected_funding_cost=frame[f"carry_{spec.signal_hours}h"] * 24)
    slopes = daily_pass_through(frame, "price_return", "expected_funding_cost", spec.fm_min_contracts, spec.fm_winsor)
    whole = summarize_slopes(slopes, spec.newey_west_lags)
    falsified = whole["b"] < 0.5 and abs(whole["t_vs_1"]) >= 1.96
    return {"venue": venue, "all": whole, "by_year": by_year(slopes, spec.newey_west_lags),
            "verdict_A6": "falsified" if falsified else "holds"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--venue", choices=sorted(VENUES), default="hyperliquid")
    args = parser.parse_args(argv)
    out = run("holdout" if args.holdout else "design", venue=args.venue)
    b = out["strategies"]["B"]
    print(f"floor ${out['floor']:,.0f} (coverage {out['floor_coverage']}); {out['avg_eligible']:.0f} eligible a day")
    for name, by_cost in out["strategies"].items():
        for key, s in by_cost.items():
            print(f"{name:<9} {key:<14} net Sharpe {s['sharpe']:+.2f} CI [{s['sharpe_ci95'][0]:+.2f}, {s['sharpe_ci95'][1]:+.2f}]"
                  f"  gross {s['gross_sharpe']:+.2f}  funding {s['annual_funding']:+.1%}  price {s['annual_price']:+.1%}")
    print("gross CI (B):", b["preregistered"]["gross_sharpe_ci95"])
    print("verdicts:", out["verdicts"])


if __name__ == "__main__":
    main()

