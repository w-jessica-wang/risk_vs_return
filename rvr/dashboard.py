# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Export the forward-test dashboard's datasets (Amendment A4 and its interim-monitoring addendum) as JSON rows
in results/dashboard/, from files the pipeline already wrote. Nothing is computed here that the pipeline
didn't compute: this only reshapes.

Run:  python -m rvr.dashboard       # after `python -m rvr.run --forward --interim --unseal "<reason>"`
"""
import json

import numpy as np
import pandas as pd

from rvr.config import ROOT
from rvr.spec import SPEC

RESULTS = ROOT / "results"
OUT = RESULTS / "dashboard"
STRATEGIES = {"B": "Baseline carry", "B_buffer": "Carry with no-trade buffer", "B_core": "Tail-robust carry"}


def _day(ms):
    return pd.to_datetime(ms, unit="ms").strftime("%Y-%m-%d")


def main():
    forward = json.loads((RESULTS / "forward_interim" / "summary.json").read_text())
    nets = pd.read_parquet(RESULTS / "forward_interim" / "daily_net_by_cost_model.parquet")
    diagnostics = json.loads((RESULTS / "diagnostics.json").read_text())
    calibrated = forward["cost_sensitivity"]["calibrated"]

    cumulative = []
    for key, label in STRATEGIES.items():
        wealth = np.cumprod(1 + nets[f"{key}|calibrated"].to_numpy()) - 1
        cumulative += [{"day": _day(d), "strategy": label, "cumulative_return": float(w)}
                       for d, w in zip(nets.index, wealth)]

    summary = [{"strategy": label, "net_sharpe": calibrated[key]["sharpe"], "ci_low": calibrated[key]["sharpe_ci95"][0],
                "ci_high": calibrated[key]["sharpe_ci95"][1], "annual_return": calibrated[key]["annual_return"],
                "max_drawdown": calibrated[key]["max_drawdown"]} for key, label in STRATEGIES.items()]

    a4 = forward["forward_A4"]
    b = calibrated["B"]
    pt = a4["pass_through"]
    predictions = [
        {"id": "carry_stays_dead", "prediction": "Carry stays dead",
         "rule": "Baseline net Sharpe ≤ 0 (falsified if its 95% CI is above zero)",
         "current": b["sharpe"], "ci_low": b["sharpe_ci95"][0], "ci_high": b["sharpe_ci95"][1],
         "status": forward["verdicts"]["A4_carry_stays_dead"]},
        {"id": "parity_persists", "prediction": "Funding parity persists",
         "rule": "Mean price pass-through b ≥ 0.5 (falsified if below 0.5 and different from 1); no-carry benchmark b ≈ ρ ≈ 0.7",
         "current": pt["b"], "ci_low": pt["b"] - 1.96 * pt["se"], "ci_high": pt["b"] + 1.96 * pt["se"],
         "status": forward["verdicts"]["A4_parity_persists"]},
    ]

    years = diagnostics["pass_through"]["by_year"]
    persistence = diagnostics["signal_persistence"]["by_year"]
    total = diagnostics["total_return_slope"]["by_year"]
    perf = diagnostics["annual_performance"]["B|calibrated"]
    history = [{"year": int(y), "pass_through": r["b"], "pass_through_low": r["b"] - 1.96 * r["se"],
                "pass_through_high": r["b"] + 1.96 * r["se"], "persistence": persistence[y]["b"],
                "total_slope": total[y]["b"], "net_sharpe": perf[y]["sharpe"]}
               for y, r in sorted(years.items(), key=lambda kv: int(kv[0]))]

    window = [{"window_start": SPEC.forward_start, "window_end": SPEC.forward_end,
               "evaluated_from": forward["report_days"][0], "evaluated_to": forward["report_days"][1],
               "days": calibrated["B"]["days"], "data_through": forward["data_through"][:10],
               "gate_days_on": a4["gated_days_on"]}]

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("forward_cumulative", cumulative), ("forward_summary", summary),
                       ("predictions", predictions), ("history", history), ("window", window)):
        (OUT / f"{name}.json").write_text(json.dumps(rows, indent=1) + "\n")
        print(f"wrote {OUT / name}.json ({len(rows)} rows)")


if __name__ == "__main__":
    main()
