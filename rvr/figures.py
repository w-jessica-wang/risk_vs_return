# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Figures for README.md and RESULTS.md, drawn only from files the pipeline wrote (results/...).

Run:  python -m rvr.figures        # after rvr.run (design and holdout), rvr.costs and rvr.diagnostics
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rvr.run import RESULTS_DIR  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
HOLDOUT_FILL = "#efeeea"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "axes.labelsize": 10,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10, "legend.frameon": False,
    "legend.fontsize": 9, "legend.labelcolor": INK_2,
})


def _load(name):
    return json.loads((RESULTS_DIR / name).read_text())


def _dates(index):
    return pd.to_datetime(index, unit="ms")


def _note(fig, text):
    fig.text(0.01, 0.01, text, color=MUTED, fontsize=8, ha="left", va="bottom")


def cumulative_returns():
    daily = pd.concat([pd.read_parquet(RESULTS_DIR / p / "daily_net_by_cost_model.parquet") for p in ("design", "holdout")])
    daily.index = _dates(daily.index)
    series = [("B|fees_only", "Fees only", MUTED), ("B|calibrated", "Calibrated costs (A1)", BLUE),
              ("B|preregistered", "Pre-registered costs (§5)", ORANGE)]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    holdout = pd.Timestamp("2025-09-01")
    ax.axvspan(holdout, daily.index.max(), color=HOLDOUT_FILL, lw=0, zorder=0)
    ax.text(holdout + pd.Timedelta(days=12), 0.97, "sealed holdout\nread once", transform=ax.get_xaxis_transform(),
            va="top", color=INK_2, fontsize=9)
    for col, label, color in series:
        wealth = np.log(1 + daily[col]).cumsum()
        ax.plot(wealth.index, wealth, color=color, label=label)
        ax.annotate(label, (wealth.index[-1], wealth.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK_2, fontsize=9, va="center")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_ylabel("cumulative log return")
    ax.set_title("Baseline carry (B): the premium was a 2020–21 phenomenon")
    ax.set_xlim(daily.index.min(), daily.index.max() + pd.Timedelta(days=420))
    ax.legend(loc="lower left")
    _note(fig, "Same weights under each cost model, $1M capital. Daily, net of costs.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def pass_through():
    diag = _load("diagnostics.json")
    d, rho = diag["pass_through"], diag["signal_persistence"]["design"]["b"]
    years = sorted(d["by_year"], key=int)
    b = np.array([d["by_year"][y]["b"] for y in years])
    se = np.array([d["by_year"][y]["se"] for y in years])
    labels = [f"{y}\n(Jan–Aug)" if y == "2026" else ("2020\n(May–)" if y == "2020" else y) for y in years]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(years))
    ax.axhline(rho, color=ORANGE, lw=1.2, ls="--")
    ax.axhline(0, color=INK_2, lw=1.0)
    ax.text(-0.4, 2.9, f"dashed: b = ρ ≈ {rho:.1f}, prices absorb\nthe funding actually paid (no carry)",
            color=INK_2, fontsize=8.5, ha="left", va="top")
    ax.text(len(years) - 0.45, -0.6, "b = 0: prices ignore funding", color=INK_2, fontsize=8.5, ha="right")
    ax.errorbar(x, b, yerr=1.96 * se, fmt="o", color=BLUE, ms=8, capsize=0, elinewidth=2, mec=SURFACE, mew=2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("pass-through b")
    ax.set_title("Price moves against funding in 2020–21, with it from 2022")
    _note(fig, "Daily cross-sectional slope of a long's next-day price return on its expected funding cost, averaged per\n"
               "year; bars ±1.96 Newey-West s.e. ρ: persistence of funding (design period). Exploratory, after the holdout.")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def regime_break():
    slopes = pd.read_parquet(RESULTS_DIR / "pass_through_daily.parquet")["slope"].sort_index()
    brk = _load("breaks.json")["break"]
    rho = _load("diagnostics.json")["signal_persistence"]["design"]["b"]
    dates = _dates(slopes.index)
    rolling = slopes.rolling(90, min_periods=60).mean()
    start, ci = pd.Timestamp(brk["break_date"]), [pd.Timestamp(d) for d in brk["break_date_ci95"]]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.axvspan(ci[0], ci[1], color=HOLDOUT_FILL, lw=0, zorder=0)
    ax.axhline(rho, color=ORANGE, lw=1.2, ls="--")
    ax.axhline(0, color=INK_2, lw=1)
    ax.plot(dates, rolling, color=BLUE, lw=2, label="90-day average of daily pass-through")
    for lo, hi, mean in ((dates.min(), start, brk["mean_before"]), (start, dates.max(), brk["mean_after"])):
        ax.plot([lo, hi], [mean, mean], color=INK, lw=1.5)
        ax.annotate(f"regime mean {mean:+.2f}", (lo + (hi - lo) * 0.12, mean), xytext=(0, -16), textcoords="offset points",
                    ha="left", color=INK, fontsize=9,
                    bbox={"boxstyle": "round,pad=0.25", "fc": SURFACE, "ec": GRID, "alpha": 0.95})
    ax.axvline(start, color=INK, lw=1, ls=":")
    ax.text(start, ax.get_ylim()[1], f"  break {start:%b %Y}\n  95% CI {ci[0]:%b %Y}–{ci[1]:%b %Y}",
            va="top", color=INK_2, fontsize=9)
    ax.set_ylabel("funding pass-through b")
    ax.set_title("One break, in late 2021: from squeeze to parity")
    ax.legend(loc="lower right")
    _note(fig, f"Least-squares break in the mean of daily pass-through slopes (Bai 1997), sup-F {brk['sup_f']:.1f}, "
               f"bootstrap p < 0.001.\nDashed line: no-carry benchmark b = ρ ≈ {rho:.1f}. Exploratory: built after the holdout was read.")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def quintile_sorts():
    d = _load("diagnostics.json")["quintile_sorts"]["by_period"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, period, title in zip(axes, ("design", "holdout"), ("Design, 2020-05 → 2025-08", "Holdout, 2025-09 → 2026-08")):
        total = [d[period]["long_return"][str(q)] * 100 for q in range(1, 6)]
        ax.bar(range(1, 6), total, color=BLUE, width=0.62, edgecolor=SURFACE, linewidth=2)
        for q, v in zip(range(1, 6), total):
            ax.text(q, v + (4 if v >= 0 else -4), f"{v:+.0f}%", ha="center", va="bottom" if v >= 0 else "top",
                    color=INK_2, fontsize=9)
        ax.axhline(0, color=INK_2, lw=1)
        ax.set_xticks(range(1, 6), ["Q1\nlowest\nfunding", "Q2", "Q3", "Q4", "Q5\nhighest\nfunding"])
        ax.set_title(title, fontsize=11, pad=14)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("annualised long return, price + funding")
    fig.suptitle("Carry sorts monotonically in the design period and inverts out of sample",
                 x=0.01, ha="left", fontweight="bold", fontsize=12, color=INK)
    _note(fig, "Equal-weight mean of eligible contracts in each daily funding quintile, before costs.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


def cost_calibration():
    fit = json.loads((RESULTS_DIR / "design" / "costs.json").read_text())["impact"]
    rows = sorted(fit["by_bucket"].values(), key=lambda r: r["mean_notional"])
    notional = [r["mean_notional"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(notional, [r["mean_excess_slippage"] * 1e4 for r in rows], "-o", color=BLUE, ms=8, mec=SURFACE, mew=2,
            label="Measured: real taker orders")
    ax.plot(notional, [r["mean_model"] * 1e4 for r in rows], "-o", color=ORANGE, ms=8, mec=SURFACE, mew=2,
            label=f"Fitted √-law, Y = {fit['Y']:.2f}")
    pre = [r["mean_model"] / fit["Y"] * 1e4 for r in rows]
    ax.plot(notional, pre, "--", color=MUTED, lw=1.5, label="Pre-registered assumption, Y = 1.0")
    ax.axvline(8461, color=INK_2, lw=0.8, ls=":")
    ax.text(8461 * 1.08, 1.9, "baseline's typical\ntrade ≈ $8.5K", color=INK_2, fontsize=8.5, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("order size, USDT (log scale)")
    ax.set_ylabel("cost beyond the half-spread, bp")
    ax.set_title("Measured impact is a fifth of what was assumed")
    ax.legend(loc="upper left")
    _note(fig, f"{fit['orders']:,} taker orders on {fit['contracts']} contracts, 00:30–01:30 UTC, 5 days in "
               "2023-06 → 2024-02,\ncosted against the mid quoted just before each order. Lines join size buckets.")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def capacity():
    cost = _load("design/summary.json")["cost_sensitivity"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    lines = (("maker", "B_buffer", "Buffered carry, maker then taker (A5b)", BLUE, "-"),
             ("calibrated", "B_buffer", "Buffered carry, taker (A1)", BLUE, "--"),
             ("maker", "B", "Baseline, maker then taker", ORANGE, "-"),
             ("calibrated", "B", "Baseline, taker", ORANGE, "--"))
    for model, name, label, color, style in lines:
        cap = cost[model]["capacity"][name]
        aum = [float(k.replace(",", "")) for k in cap]
        ax.plot(aum, [v["sharpe"] for v in cap.values()], style, marker="o", color=color, ms=7, mec=SURFACE, mew=2,
                label=label)
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xscale("log")
    ax.set_xticks([1e6, 1e7, 5e7, 1e8, 2.5e8, 5e8], ["$1M", "$10M", "$50M", "$100M", "$250M", "$500M"])
    ax.set_ylabel("net Sharpe, design period")
    ax.set_title("Posting limit orders roughly triples capacity; it doesn't help small books")
    ax.legend(loc="lower left", handlelength=4)
    _note(fig, "Design period 2020-05 → 2025-08, same weights. Taker: A1 measured costs. Maker: post at the touch,\n"
               "cross after 30 min (fill model fitted on 63,910 simulated orders). Costs scale with capital.")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, make in (("cumulative", cumulative_returns), ("pass_through", pass_through), ("regime_break", regime_break),
                       ("quintile_sorts", quintile_sorts), ("cost_calibration", cost_calibration),
                       ("capacity", capacity)):
        fig = make()
        fig.savefig(FIG_DIR / f"{name}.png", dpi=160)
        plt.close(fig)
        print(f"wrote {FIG_DIR / name}.png")


if __name__ == "__main__":
    main()
