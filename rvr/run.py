# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Run every pre-registered analysis (PREREGISTRATION.md) and write results/<period>/.

Run:  python -m rvr.run                                   # design period (2020-01 → 2025-08)
      python -m rvr.run --holdout --unseal "<reason>"     # the sealed year, once the design is frozen
      python -m rvr.run --forward --unseal "<reason>"     # Amendment A4, once 2027-02 is archived
"""
import argparse
import datetime as dt
import hashlib
import json
import time

import numpy as np
import pandas as pd

from rvr import verdict
from rvr.backtest import contract_days, pnl
from rvr.config import ROOT
from rvr.costs import COSTS_PATH, CostModel, load_calibrated, preregistered
from rvr.diagnostics import daily_pass_through, pass_through_frame, pass_through_gate, summarize_slopes
from rvr.execution import MAKER_PATH, load_maker
from rvr.factors import factor_returns, sort_characteristics, spanning_test, spanning_verdict
from rvr.features import DAY_MS, characteristics, day_range
from rvr.holdout import check_sealed, load_mark, load_panels, record_unseal
from rvr.pricing import pricing_test
from rvr.spec import SPEC
from rvr.stats import deflated_sharpe, summarize
from rvr.strategy import (add_downside_beta_hedge, band_weights, buffered_weights, decision_panel,
                          low_volatility_candidates, quintile_weights, scale_weights, volatility_scale)

RESULTS_DIR = ROOT / "results"
PRICING_CONTROLS = ["downside_beta", "coskew", "log_adv", "momentum", "vol", "max_ret"]


def _log(message, started):
    print(f"[{time.monotonic() - started:6.0f}s] {message}", flush=True)


def strategy_weights(panel, cdays, chars, days, spec):
    """Weights for every pre-registered variant (§3, §6, §9). Vol-targeted variants need their own
    unscaled returns first; volatility_scale only ever uses returns completed before each decision."""
    carry = f"carry_{spec.signal_hours}h"
    base = quintile_weights(panel, carry, spec)
    filtered = quintile_weights(panel, carry, spec, candidates=low_volatility_candidates(panel, spec))
    hedged = add_downside_beta_hedge(base, panel, spec)
    managed_unscaled = add_downside_beta_hedge(filtered, panel, spec)

    def vol_targeted(weights):
        return scale_weights(weights, volatility_scale(pnl(weights, cdays, chars, days, spec)["net"], spec))

    weights = {"B": base, "M1": filtered, "M2": hedged, "M3": vol_targeted(base), "M": vol_targeted(managed_unscaled)}
    for name, hours in spec.signal_variants.items():
        weights[name] = quintile_weights(panel, f"carry_{hours}h", spec)
    weights["B_buffer"] = buffered_weights(panel, carry, spec)  # Amendment A3, trial 9
    weights["B_core"] = band_weights(panel, carry, spec)        # Amendment A5c, trial 10 (tested forward only)
    return weights


def crash_windows(returns, dates, spec):
    out = {}
    for date in dates:
        center = int(pd.Timestamp(date, tz="UTC").value // 10**6)
        lo, hi = center + spec.crash_window[0] * DAY_MS, center + spec.crash_window[1] * DAY_MS
        out[date] = {name: float((1 + r[(r.index >= lo) & (r.index <= hi)]).prod() - 1) for name, r in returns.items()}
    return out


def run(period="design", unseal_reason=None, spec=SPEC, results_dir=RESULTS_DIR, interim=False):
    started = time.monotonic()
    holdout, forward = period == "holdout", period == "forward"
    if interim and not forward:
        raise ValueError("--interim applies to the forward window only")
    windows = {"design": (spec.design_start, spec.design_end), "holdout": (spec.holdout_start, spec.holdout_end),
               "forward": (spec.forward_start, spec.forward_end)}
    eval_start, eval_end = windows[period]
    all_days = day_range(spec.design_start, eval_end)
    last_exit = int(all_days[-1]) + DAY_MS + spec.execution_hour * 3_600_000
    check_sealed(last_exit, unseal=holdout or forward)
    if holdout or forward:
        if not unseal_reason or not unseal_reason.strip():
            raise ValueError("reading sealed data needs a stated reason")
        record_unseal(f"[{period}{' interim' if interim else ''}] {unseal_reason.strip()}")
    prices, funding = load_panels(unseal=holdout or forward)
    data_end = int(prices["close_time"].max()) + 1
    if forward and data_end < last_exit:
        if not interim:
            raise RuntimeError(f"forward data ends {pd.to_datetime(data_end, unit='ms')}; the A4 window needs "
                               f"prices through {pd.to_datetime(last_exit, unit='ms')} (use --interim to monitor)")
        last_day = (data_end - spec.execution_hour * 3_600_000 - DAY_MS) // DAY_MS * DAY_MS
        eval_end = int(last_day)
        all_days = day_range(spec.design_start, eval_end)
    if interim:
        period = "forward_interim"
    eval_days = day_range(eval_start, eval_end)
    _log(f"loaded {len(prices):,} bars, {len(funding):,} settlements", started)
    chars = characteristics(prices, all_days, spec)
    panel = decision_panel(prices, funding, chars, spec, spec.design_start, eval_end)
    _log(f"decision panel: {len(panel):,} contract-days, {int(panel['eligible'].sum()):,} eligible", started)
    keys = pd.concat([panel.loc[panel["eligible"], ["day", "symbol"]],
                      pd.DataFrame({"day": all_days, "symbol": spec.hedge_symbol})], ignore_index=True)
    cdays = contract_days(prices, funding, keys, spec)
    _log(f"priced {len(cdays):,} contract-days ({int(cdays['halted'].sum())} halts)", started)

    weights = strategy_weights(panel, cdays, chars, all_days, spec)
    daily = {name: pnl(w, cdays, chars, all_days, spec) for name, w in weights.items()}
    btc = cdays[cdays["symbol"] == spec.hedge_symbol].set_index("day")
    market = (btc["price_factor"] - 1 + btc["funding_long"]).reindex(all_days)
    baseline = daily["B"].loc[daily["B"].index.isin(eval_days)]
    traded_days = baseline["gross_exposure"] > 0
    first_trade = traded_days[traded_days].index.min()
    report_days = eval_days[eval_days >= first_trade]
    _log(f"strategies built; reporting {len(report_days)} days from {pd.to_datetime(first_trade, unit='ms').date()}",
         started)

    summary = {}
    for name, frame in daily.items():
        f = frame.loc[report_days]
        stats = summarize(f["net"].to_numpy(), market.loc[report_days].to_numpy(), spec)
        gross = (f["price"] + f["funding"]).to_numpy()
        stats.update({
            "gross_sharpe": float(gross.mean() / gross.std(ddof=1) * np.sqrt(365)),
            "annual_funding": float(f["funding"].mean() * 365),
            "annual_price": float(f["price"].mean() * 365),
            "annual_cost": float(f["cost"].mean() * 365),
            "break_even_cost_multiple": float(gross.mean() / f["cost"].mean()) if f["cost"].mean() > 0 else None,
            "avg_daily_turnover": float(f["turnover"].mean()),
            "avg_positions": float((f["n_long"] + f["n_short"]).mean()),
            "avg_gross_exposure": float(f["gross_exposure"].mean()),
            "halts": int(f["halts"].sum()),
        })
        summary[name] = stats
        _log(f"{name:<12} net Sharpe {stats['sharpe']:+.2f}  CI [{stats['sharpe_ci95'][0]:+.2f}, "
             f"{stats['sharpe_ci95'][1]:+.2f}]", started)

    capacity = {name: {f"{aum:,.0f}": summarize_capacity(weights[name], cdays, chars, all_days, report_days, spec, aum)
                       for aum in spec.capacity_aums} for name in ("B", "M")}

    long_return = cdays.assign(long_return=cdays["price_factor"] - 1 + cdays["funding_long"])
    fm_panel = panel[panel["eligible"] & panel["day"].isin(report_days)].merge(
        long_return[["day", "symbol", "long_return"]], on=["day", "symbol"], how="left")
    carry = f"carry_{spec.signal_hours}h"
    pricing = pricing_test(fm_panel, "long_return", carry, PRICING_CONTROLS, spec)

    costs, cost_nets = cost_sensitivity(weights, cdays, chars, all_days, report_days, market, spec)
    factors = factor_returns(sort_characteristics(prices, panel, all_days), cdays, all_days, spec)
    gross_b = daily["B"]["price"] + daily["B"]["funding"]
    spanning = {"gross": spanning_test(gross_b.loc[report_days], factors.loc[report_days], spec),
                "net": spanning_test(daily["B"].loc[report_days, "net"], factors.loc[report_days], spec)}
    _log(f"spanning (gross, five-factor): alpha t = {spanning['gross']['c_five_factor']['alpha_t']:+.2f}", started)

    mark = load_mark(unseal=holdout or forward)
    mark_block = (mark_price_funding(prices, funding, keys, weights, chars, all_days, report_days, market, costs, spec, mark)
                  if mark is not None and len(mark) else None)

    trials = _update_trials(results_dir, summary, period, daily, report_days)
    for name in ("B", "M"):
        summary[name]["deflated_sharpe"] = deflated_sharpe(daily[name].loc[report_days, "net"].to_numpy(), trials)
    for (key, name), net in cost_nets.items():
        if key.startswith("calibrated") or key == "maker":
            costs[key][name]["deflated_sharpe"] = deflated_sharpe(net.to_numpy(), trials)

    result = {
        "period": period,
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "preregistration_sha256": hashlib.sha256((ROOT / "PREREGISTRATION.md").read_bytes()).hexdigest(),
        "report_days": [str(pd.to_datetime(report_days[0], unit="ms").date()),
                        str(pd.to_datetime(report_days[-1], unit="ms").date())],
        "n_trials": len(trials),
        "strategies": summary,
        "capacity": capacity,
        "pricing": pricing,
        "crash_windows": crash_windows({k: daily[k].loc[report_days, "net"] for k in ("B", "M")} | {"BTC": market.loc[report_days]},
                                       spec.holdout_crash_dates if holdout else () if forward else spec.crash_dates, spec),
        "cost_sensitivity": costs,
        "mark_price_A7b": mark_block,
        "spanning": spanning,
        "verdicts": {
            "portfolio": verdict.portfolio_verdict(summary["B"], summary["M"], spec),
            "pricing": verdict.pricing_verdict(pricing, carry),
            "spanning_A2": spanning_verdict(spanning["gross"]),
        },
    }
    if "calibrated" in costs:
        result["verdicts"]["portfolio_A1"] = verdict.portfolio_verdict(
            costs["calibrated"]["B"], costs["calibrated"]["M"], spec)
    if forward:
        result["data_through"] = str(pd.to_datetime(data_end, unit="ms"))
        result["forward_A4"] = forward_predictions(panel, cdays, costs, cost_nets, report_days, spec)
        result["verdicts"] |= result["forward_A4"]["verdicts"]
    if holdout:
        design = json.loads((results_dir / "design" / "summary.json").read_text())
        result["verdicts"]["holdout_confirms"] = verdict.holdout_confirms(design["strategies"], summary)
        if "calibrated" in costs:
            result["verdicts"]["holdout_confirms_A1"] = verdict.holdout_confirms(
                design["cost_sensitivity"]["calibrated"], costs["calibrated"])
    out_dir = results_dir / period
    out_dir.mkdir(parents=True, exist_ok=True)
    factors.loc[report_days].to_parquet(out_dir / "factors.parquet")
    pd.DataFrame({f"{name}|{key}": net for (key, name), net in cost_nets.items()}).to_parquet(
        out_dir / "daily_net_by_cost_model.parquet")
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    pd.DataFrame({name: frame["net"] for name, frame in daily.items()}).loc[report_days].to_parquet(out_dir / "daily_net.parquet")
    _log(f"verdicts: {result['verdicts']}", started)
    return result


def forward_predictions(panel, cdays, costs, cost_nets, report_days, spec):
    """Amendment A4's two predictions, and the real-time pass-through gate (reported, not predicted)."""
    slopes = daily_pass_through(pass_through_frame(panel, cdays, spec), "price_return", "expected_funding_cost",
                                spec.fm_min_contracts, spec.fm_winsor)
    window = summarize_slopes(slopes[slopes.index.isin(report_days)], spec.newey_west_lags)
    gated, on = pass_through_gate(slopes, cost_nets[("calibrated", "B")].reindex(report_days), spec.forward_gate_window,
                                  spec.forward_gate_threshold)
    b = costs["calibrated"]["B"]
    dead = not (b["sharpe"] > 0 and b["sharpe_ci95"][0] > 0)
    parity = not (window["b"] < 0.5 and abs(window["t_vs_1"]) >= 1.96)
    return {"pass_through": window, "gated_B_sharpe": float(gated.mean() / gated.std() * np.sqrt(365)) if gated.std() > 0
            else 0.0, "gated_days_on": float(on.mean()),
            "verdicts": {"A4_carry_stays_dead": "holds" if dead else "falsified",
                         "A4_parity_persists": "holds" if parity else "falsified"}}


def mark_price_funding(prices, funding, keys, weights, chars, days, report_days, market, costs, spec, mark,
                       names=("B", "M", "B_buffer")):
    """
    Amendment A7b: the same weights with every settlement valued on the mark price. Reports net Sharpe under
    the §5, A1 and A5b cost models, its change from the primary (traded-close) result, and the fallback count.
    """
    cdays = contract_days(prices, funding, keys, spec, mark=mark)
    models = {k: m for k, m in cost_models(spec).items() if k in ("preregistered", "calibrated", "maker")}
    out = {"settlements_without_mark_bar": int(cdays.loc[cdays["day"].isin(report_days), "mark_fallbacks"].sum())}
    for key, model in models.items():
        out[key] = {}
        for name in names:
            f = pnl(weights[name], cdays, chars, days, spec, cost_model=model).loc[report_days]
            stats = summarize(f["net"].to_numpy(), market.loc[report_days].to_numpy(), spec)
            primary = costs[key][name]["sharpe"]
            out[key][name] = {"sharpe": stats["sharpe"], "sharpe_ci95": stats["sharpe_ci95"],
                              "annual_funding": float(f["funding"].mean() * 365),
                              "change_from_primary": stats["sharpe"] - primary}
    largest = max(abs(v["change_from_primary"]) for k in models for v in out[k].values())
    out["prediction_A7b"] = "holds" if largest <= 0.10 else "falsified"
    out["largest_change"] = largest
    return out


def cost_models(spec):
    """Every cost model the results are reported under. A1's calibration is used only if it has been fitted."""
    zero_spread = ((0.0, 0.0),)
    models = {
        "fees_only": CostModel("fees only", spec.fee, 0.0, spread_tiers=zero_spread),
        "fees_spread": CostModel("fees + §5 spread", spec.fee, 0.0, spread_tiers=tuple(spec.spread_tiers)),
        "preregistered": preregistered(spec),
    }
    if COSTS_PATH.exists():
        calibrated = load_calibrated(spec=spec)
        models |= {"calibrated": calibrated, "calibrated_x2": calibrated.scaled(2.0)}
    if COSTS_PATH.exists() and MAKER_PATH.exists():
        models["maker"] = load_maker(spec=spec)
    return models


def cost_sensitivity(weights, cdays, chars, days, report_days, market, spec,
                     names=("B", "M", "B_carry168h", "B_buffer", "B_core")):
    """
    Same weights under each cost model (A1): not trials. Calibrated costs also get a capacity curve.
    Returns (statistics, {(model, strategy): daily net returns}).
    """
    out, nets = {}, {}
    for key, model in cost_models(spec).items():
        out[key] = {"model": model.name}
        for name in names:
            f = pnl(weights[name], cdays, chars, days, spec, cost_model=model).loc[report_days]
            stats = summarize(f["net"].to_numpy(), market.loc[report_days].to_numpy(), spec)
            stats["annual_cost"] = float(f["cost"].mean() * 365)
            nets[(key, name)] = f["net"]
            out[key][name] = stats
        if key in ("calibrated", "maker"):
            out[key]["capacity"] = {
                name: {f"{aum:,.0f}": summarize_capacity(weights[name], cdays, chars, days, report_days, spec, aum,
                                                         cost_model=model) for aum in spec.capacity_aums}
                for name in ("B", "B_carry168h", "B_buffer", "B_core")}
    return out, nets


def summarize_capacity(weights, cdays, chars, days, report_days, spec, aum, cost_model=None):
    net = pnl(weights, cdays, chars, days, spec, aum=aum, cost_model=cost_model).loc[report_days, "net"].to_numpy()
    return {"sharpe": float(net.mean() / net.std(ddof=1) * np.sqrt(365)), "annual_return": float(net.mean() * 365)}


def _update_trials(results_dir, summary, period, daily, report_days):
    """
    Log this run's design variants in the trial log; return one daily Sharpe per distinct variant (§9). A variant
    is appended only when it is new or its result changed, so re-running an unchanged design adds nothing.
    """
    log = results_dir / "trials.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    latest = {}
    if log.exists():
        for line in log.read_text().splitlines():
            entry = json.loads(line)
            latest[entry["variant"]] = entry["daily_sharpe"]
    if period == "design":
        with log.open("a") as f:
            for name, stats in summary.items():
                net = daily[name].loc[report_days, "net"].to_numpy()
                sharpe = float(net.mean() / net.std(ddof=1))
                if name in latest and np.isclose(latest[name], sharpe, rtol=0, atol=1e-12):
                    continue
                latest[name] = sharpe
                f.write(json.dumps({"variant": name, "daily_sharpe": sharpe, "annual_sharpe": stats["sharpe"],
                                    "logged": dt.datetime.now(dt.timezone.utc).isoformat()}) + "\n")
    return list(latest.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--holdout", action="store_true", help="evaluate the sealed year (requires --unseal)")
    parser.add_argument("--forward", action="store_true", help="evaluate Amendment A4's window (requires --unseal)")
    parser.add_argument("--interim", action="store_true", help="with --forward: evaluate the window to date (A4 addendum)")
    parser.add_argument("--unseal", metavar="REASON", help="why sealed data is being read; logged to HOLDOUT_LOG.md")
    args = parser.parse_args(argv)
    if (args.holdout or args.forward) and not args.unseal:
        parser.error("--holdout and --forward need --unseal \"<reason>\"")
    if args.holdout and args.forward:
        parser.error("choose one of --holdout and --forward")
    run("holdout" if args.holdout else "forward" if args.forward else "design", args.unseal, interim=args.interim)


if __name__ == "__main__":
    main()
