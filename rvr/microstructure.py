# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Measure what trading actually costs at the strategy's execution time (PREREGISTRATION.md Amendment A1).

Binance archives two order-book datasets for USDⓈ-M futures:

  bookTicker  every change to the best bid and ask (2023-05-16 → 2024-03-30 only)
  bookDepth   a snapshot every ~30 s of the cumulative notional resting within ±1 … ±5% of the mid
              (2023-01 onward)

and aggTrades, every fill. Pairing fills with the quote standing just before them gives what real
taker orders of each size paid, which checks the cost model built from quoted spreads and depth.

For a sample of (day, contract) pairs from the eligible universe, this keeps only the window around the
01:00 UTC execution and reduces it to a few numbers: the time-weighted quoted half-spread, the notional
at the touch, and the median depth curve. Nothing here reads a return, so calibrating costs from it
cannot tune the strategy to its own results. Only design-period days are sampled.

Run:  python -m rvr.microstructure        # samples the design-period eligible universe (§2)
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from rvr import net
from rvr.config import DATA_BASE_URL, HOUR_MS, PROCESSED_DIR
from rvr.download import ChecksumError, _run_all, verify_checksum
from rvr.holdout import HOLDOUT_START_MS
from rvr.spec import SPEC

MICRO_DIR = PROCESSED_DIR / "micro"
WINDOW_MS = 10 * 60_000                 # quoted spread and depth: ±10 minutes around execution
ORDER_WINDOW_MS = 30 * 60_000           # realised taker costs: ±30 minutes, for enough orders
DEPTH_LEVELS = (1, 2, 3, 4, 5)          # percent from mid, as archived
BOOK_TICKER_RANGE = ("2023-05-16", "2024-03-30")
BOOK_DEPTH_START = "2023-01-01"


def daily_key(dataset, symbol, day):
    return f"data/futures/um/daily/{dataset}/{symbol}/{symbol}-{dataset}-{day}.zip"


def fetch_verified(key, get_bytes=net.get_bytes, get_text=net.get_text):
    """The archive's CSV bytes, after its SHA-256 matches Binance's checksum. None if it doesn't exist."""
    url = f"{DATA_BASE_URL}/{key}"
    try:
        payload = get_bytes(url)
    except Exception as exc:  # a missing day is a 404; anything else is re-raised
        if "404" in str(exc):
            return None
        raise
    if not verify_checksum(payload, get_text(url + ".CHECKSUM")):
        raise ChecksumError(key)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return archive.read(archive.namelist()[0])


def _window(t_exec, half_width=WINDOW_MS):
    return t_exec - half_width, t_exec + half_width


def read_quotes(csv_bytes, lo, hi):
    """Best bid/ask updates in [lo, hi), plus the last one before lo. Stops reading once past the window."""
    cols = ["best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty", "transaction_time"]
    kept, before = [], None
    for chunk in pd.read_csv(io.BytesIO(csv_bytes), usecols=cols, chunksize=200_000):
        t = chunk["transaction_time"].to_numpy()
        if t[0] < lo:
            before = chunk[t < lo].iloc[[-1]]
        if t[-1] < lo:
            continue
        kept.append(chunk[(t >= lo) & (t < hi)])
        if t[-1] >= hi:
            break
    quotes = pd.concat(([before] if before is not None else []) + kept, ignore_index=True)
    return quotes[(quotes["best_ask_price"] > quotes["best_bid_price"]) & (quotes["best_bid_price"] > 0)]


def summarize_quotes(quotes, lo, hi):
    """Quoted half-spread and touch size over [lo, hi), each quote weighted by how long it stood."""
    quotes = quotes[quotes["transaction_time"] < hi]
    if quotes.empty:
        return None
    start = np.maximum(quotes["transaction_time"].to_numpy(), lo)
    duration = np.diff(np.append(start, hi)).clip(min=0).astype(float)
    if duration.sum() <= 0:
        return None
    bid, ask = quotes["best_bid_price"].to_numpy(), quotes["best_ask_price"].to_numpy()
    mid = (bid + ask) / 2
    half = (ask - bid) / (2 * mid)
    touch = (quotes["best_bid_qty"].to_numpy() * bid + quotes["best_ask_qty"].to_numpy() * ask) / 2
    prices = np.unique(np.concatenate([bid, ask]))
    steps = np.diff(prices)
    tick = float(steps[steps > 0].min()) if (steps > 0).any() else np.nan
    return {
        "half_spread": float(np.average(half, weights=duration)),
        "half_spread_median": _weighted_median(half, duration),
        "touch_notional": float(np.average(touch, weights=duration)),
        "rel_tick": tick / float(np.median(mid)),
        "quote_updates": int(len(quotes)),
    }


def taker_orders(csv_bytes, lo, hi):
    """
    Aggressive orders in [lo, hi): aggregate trades with the same timestamp and taker side are one order
    walking the book. side = +1 for a taker buy. Returns time, side, notional and VWAP.
    """
    trades = pd.read_csv(io.BytesIO(csv_bytes), usecols=["price", "quantity", "transact_time", "is_buyer_maker"])
    trades = trades[(trades["transact_time"] >= lo) & (trades["transact_time"] < hi)]
    if trades.empty:
        return pd.DataFrame(columns=["t", "side", "notional", "vwap"])
    maker_buy = trades["is_buyer_maker"].astype(str).str.lower() == "true"
    trades = trades.assign(side=np.where(maker_buy, -1, 1), value=trades["price"] * trades["quantity"])
    orders = trades.groupby(["transact_time", "side"], as_index=False).agg(
        notional=("value", "sum"), quantity=("quantity", "sum"))
    return pd.DataFrame({"t": orders["transact_time"], "side": orders["side"], "notional": orders["notional"],
                         "vwap": orders["notional"] / orders["quantity"]})


def order_slippage(orders, quotes):
    """Each order's cost against the mid quoted strictly before it: side × (VWAP / mid − 1)."""
    q = quotes.assign(mid=(quotes["best_bid_price"] + quotes["best_ask_price"]) / 2)[["transaction_time", "mid"]]
    merged = pd.merge_asof(orders.sort_values("t"), q.sort_values("transaction_time"), left_on="t",
                           right_on="transaction_time", direction="backward", allow_exact_matches=False)
    return merged.assign(slippage=merged["side"] * (merged["vwap"] / merged["mid"] - 1)).dropna(subset=["slippage"])


SIZE_BUCKETS = (0, 1e3, 3e3, 1e4, 3e4, 1e5, np.inf)


def slippage_by_size(filled):
    """Count, mean notional and mean slippage of taker orders in each notional bucket."""
    out = {}
    bucket = np.digitize(filled["notional"], SIZE_BUCKETS[1:-1])
    for b in range(len(SIZE_BUCKETS) - 1):
        g = filled[bucket == b]
        out[f"orders_b{b}"] = int(len(g))
        out[f"notional_b{b}"] = float(g["notional"].mean()) if len(g) else np.nan
        out[f"slippage_b{b}"] = float(g["slippage"].mean()) if len(g) else np.nan
    return out


def _weighted_median(values, weights):
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, cum[-1] / 2)])


def summarize_book_depth(csv_bytes, t_exec):
    """Median cumulative notional within each ±k% band, bid and ask, over the execution window's snapshots."""
    lo, hi = _window(t_exec)
    frame = pd.read_csv(io.BytesIO(csv_bytes))
    if frame.empty or "notional" not in frame.columns:
        return None
    t = pd.to_datetime(frame["timestamp"], utc=True).astype("int64") // 10**6
    frame = frame[(t >= lo) & (t < hi)]
    if frame.empty:
        return None
    level = frame.groupby("percentage")["notional"].median()
    out = {"snapshots": int(frame["timestamp"].nunique())}
    for k in DEPTH_LEVELS:
        out[f"bid_{k}"] = float(level.get(-k, np.nan))
        out[f"ask_{k}"] = float(level.get(k, np.nan))
    return out


def walk_cost(notional, half_spread, depth):
    """
    Average slippage from the mid of a market order of `notional` USDT: the half-spread, plus walking a book
    whose cumulative depth is piecewise linear through (0, 0), (1%, depth[0]), …, (5%, depth[4]). Past 5%
    the last band's density is extended. Vectorised over arrays of equal length.
    """
    notional = np.asarray(notional, dtype=float)
    depth = np.asarray(depth, dtype=float).reshape(len(notional), -1)
    grid_x = np.arange(depth.shape[1] + 1) / 100.0
    cum = np.column_stack([np.zeros(len(notional)), np.maximum.accumulate(depth, axis=1)])
    cost = np.empty(len(notional))
    for i in range(len(notional)):
        q, d = notional[i], cum[i]
        if q <= 0 or not np.isfinite(d[-1]) or d[-1] <= 0:
            cost[i] = 0.0 if q <= 0 else np.nan
            continue
        if q <= d[-1]:
            x_end = np.interp(q, d, grid_x)
        else:
            x_end = grid_x[-1] + (q - d[-1]) / ((d[-1] - d[-2]) / (grid_x[-1] - grid_x[-2]))
        # average price offset = ∫ x dD / q over the depth consumed
        xs = np.append(grid_x[grid_x < x_end], x_end)
        ds = np.interp(xs, grid_x, d, right=np.nan)
        if not np.isfinite(ds[-1]):
            ds[-1] = q
        cost[i] = np.sum(np.diff(ds) * (xs[1:] + xs[:-1]) / 2) / q
    return np.asarray(half_spread, dtype=float) + cost


def sample_jobs(eligible, days):
    """(dataset, symbol, day) for every eligible contract on each sample day, inside the design period."""
    eligible = eligible.assign(date=pd.to_datetime(eligible["day"], unit="ms").dt.strftime("%Y-%m-%d"))
    jobs = []
    for dataset, chosen in days.items():
        rows = eligible[eligible["date"].isin(chosen)]
        jobs += [(dataset, s, d) for s, d in zip(rows["symbol"], rows["date"])]
    return jobs


def default_days():
    design_end = pd.Timestamp(SPEC.design_end)
    depth = [d for d in pd.date_range(BOOK_DEPTH_START, design_end, freq="D") if d.day in (1, 15)]
    ticker = pd.to_datetime(["2023-06-01", "2023-08-01", "2023-10-01", "2023-12-01", "2024-02-01"])
    return {"bookDepth": [d.strftime("%Y-%m-%d") for d in depth],
            "execution": [d.strftime("%Y-%m-%d") for d in ticker]}


def summarize_execution(fetch, symbol, day, t_exec):
    """Quoted spread and touch (±10 min) plus realised taker slippage by order size (±30 min)."""
    quotes_csv = fetch(daily_key("bookTicker", symbol, day))
    trades_csv = fetch(daily_key("aggTrades", symbol, day))
    if quotes_csv is None:
        return None
    lo, hi = _window(t_exec, ORDER_WINDOW_MS)
    quotes = read_quotes(quotes_csv, lo, hi)
    summary = summarize_quotes(quotes, *_window(t_exec))
    if summary is None:
        return None
    if trades_csv is not None:
        summary |= slippage_by_size(order_slippage(taker_orders(trades_csv, lo, hi), quotes))
    return summary


def measure(job, out_dir=MICRO_DIR, fetch=fetch_verified):
    """Reduce one (dataset, symbol, day) to its execution-window summary, cached as a one-row parquet shard."""
    dataset, symbol, day = job
    shard = Path(out_dir) / dataset / f"{day}_{symbol}.parquet"
    if shard.exists():
        return "cached"
    t_exec = int(pd.Timestamp(day, tz="UTC").value // 10**6) + SPEC.execution_hour * HOUR_MS
    if t_exec + ORDER_WINDOW_MS >= HOLDOUT_START_MS:
        raise ValueError(f"{day} reaches the sealed holdout")
    if dataset == "execution":
        summary = summarize_execution(fetch, symbol, day, t_exec)
    else:
        csv_bytes = fetch(daily_key(dataset, symbol, day))
        summary = None if csv_bytes is None else summarize_book_depth(csv_bytes, t_exec)
    row = {"day": int(pd.Timestamp(day, tz="UTC").value // 10**6), "symbol": symbol, "available": summary is not None}
    shard.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row | (summary or {})]).to_parquet(shard)
    return "measured" if summary is not None else "missing"


def collect(dataset, out_dir=MICRO_DIR):
    shards = sorted((Path(out_dir) / dataset).glob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True) if shards else pd.DataFrame()


def design_eligible(spec=SPEC):
    """Every eligible (day, symbol) in the design period, from the same decision panel the strategies use."""
    from rvr.features import characteristics, day_range
    from rvr.holdout import load_panels
    from rvr.strategy import decision_panel

    prices, funding = load_panels()
    chars = characteristics(prices, day_range(spec.design_start, spec.design_end), spec)
    p = decision_panel(prices, funding, chars, spec, spec.design_start, spec.design_end)
    return p.loc[p["eligible"], ["day", "symbol"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eligible", help="parquet of eligible (day, symbol) pairs; default: rebuilt from §2")
    parser.add_argument("--datasets", nargs="+", default=["bookDepth", "execution"])
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)

    days = {k: v for k, v in default_days().items() if k in args.datasets}
    jobs = sample_jobs(pd.read_parquet(args.eligible) if args.eligible else design_eligible(), days)
    print(f"{len(jobs):,} archives to measure", flush=True)
    outcomes, failures = _run_all(jobs, measure, args.workers, "measured")
    print(pd.Series([o for o in outcomes if o]).value_counts().to_dict(), flush=True)
    for dataset in args.datasets:
        table = collect(dataset)
        table.to_parquet(PROCESSED_DIR / f"micro_{dataset}.parquet")
        print(f"{dataset}: {int(table['available'].sum()):,} of {len(table):,} measured", flush=True)
    if failures:
        for job, error in failures[:20]:
            print(f"  failed {job}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
