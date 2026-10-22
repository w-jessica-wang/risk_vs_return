# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Build tidy parquet panels from the raw archives in data/raw.

Outputs in data/processed/:
  funding.parquet           one row per (symbol, funding settlement)
  prices_1h/<SYMBOL>.parquet  hourly bars, one file per symbol (zero-volume bars kept)
  listings.parquet          first/last bar, last traded bar and settlements per symbol.
                            Descriptive only: it contains each contract's future (its
                            delisting), so strategy code must never pick a universe from it —
                            use rvr.pit.tradable_universe
  quality_report.json       every anomaly the parsers corrected or flagged, per symbol
  mark_1h.parquet           hourly mark-price closes, for valuing funding (Amendment A7b)

Run:  python -m rvr.build
"""
import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import pandas as pd

from rvr.config import PROCESSED_DIR, RAW_DIR
from rvr.parse import (FUNDING_COLUMNS, FUNDING_OUT, KLINE_COLUMNS, parse_funding, parse_klines, parse_mark,
                       read_zipped_csv)

LISTING_INT_COLUMNS = ["first_settle_hour", "last_settle_hour", "n_settlements",
                       "first_bar_open", "last_bar_open", "last_traded_bar_open", "n_bars"]


def _read_all(files, columns):
    return pd.concat([read_zipped_csv(f, columns) for f in files], ignore_index=True)


def build_symbol(symbol, raw_dir, prices_dir):
    """Parse one symbol's archives, write its price file → (funding frame, listing row, issues)."""
    raw_dir = Path(raw_dir)
    issues = Counter()
    row = {"symbol": symbol}
    funding = None

    funding_files = sorted((raw_dir / "fundingRate" / symbol).glob("*.zip"))
    if funding_files:
        funding, found = parse_funding(_read_all(funding_files, FUNDING_COLUMNS), symbol)
        issues.update(found)
        if len(funding):
            row.update(first_settle_hour=funding["settle_hour"].iloc[0],
                       last_settle_hour=funding["settle_hour"].iloc[-1],
                       n_settlements=len(funding),
                       funding_intervals_h=",".join(map(str, sorted(funding["interval_h"].unique()))))

    kline_files = sorted((raw_dir / "klines" / symbol / "1h").glob("*.zip"))
    if kline_files:
        prices, found = parse_klines(_read_all(kline_files, KLINE_COLUMNS), symbol)
        issues.update(found)
        if len(prices):
            prices.to_parquet(Path(prices_dir) / f"{symbol}.parquet", index=False)
            row.update(first_bar_open=prices["open_time"].iloc[0],
                       last_bar_open=prices["open_time"].iloc[-1],
                       last_traded_bar_open=prices.loc[prices["quote_volume"] > 0, "open_time"].max(),
                       n_bars=len(prices))

    return funding, row, {issue: count for issue, count in issues.items() if count}


def build(symbols, raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR, workers=1):
    processed_dir = Path(processed_dir)
    prices_dir = processed_dir / "prices_1h"
    prices_dir.mkdir(parents=True, exist_ok=True)

    jobs = (symbols, repeat(raw_dir), repeat(prices_dir))
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(build_symbol, *jobs, chunksize=8))
    else:
        results = list(map(build_symbol, *jobs))

    frames = [funding for funding, _, _ in results if funding is not None and len(funding)]
    funding = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FUNDING_OUT)
    funding.to_parquet(processed_dir / "funding.parquet", index=False)

    listings = pd.DataFrame([row for _, row, _ in results])
    for column in LISTING_INT_COLUMNS:
        listings[column] = (listings[column] if column in listings else pd.NA)
        listings[column] = listings[column].astype("Int64")
    listings.to_parquet(processed_dir / "listings.parquet", index=False)

    totals = Counter()
    for _, _, found in results:
        totals.update(found)
    report = {
        "symbols": len(symbols),
        "funding_rows": len(funding),
        "totals": dict(sorted(totals.items())),
        "by_symbol": {row["symbol"]: found for _, row, found in results if found},
    }
    (processed_dir / "quality_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_mark(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR):
    """Every symbol's mark-price bars → mark_1h.parquet, with issue counts (Amendment A7b)."""
    root = Path(raw_dir) / "markPriceKlines"
    frames, issues = [], Counter()
    for symbol_dir in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        files = sorted((symbol_dir / "1h").glob("*.zip"))
        if files:
            frame, found = parse_mark(_read_all(files, KLINE_COLUMNS), symbol_dir.name)
            frames.append(frame)
            issues.update(found)
    mark = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["symbol", "close_time", "mark"])
    mark.to_parquet(Path(processed_dir) / "mark_1h.parquet", index=False)
    return mark, dict(issues)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", help="default: every symbol under data/raw/fundingRate")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)

    symbols = args.symbols or sorted(p.name for p in (RAW_DIR / "fundingRate").iterdir() if p.is_dir())
    report = build(symbols, workers=args.workers)
    mark, mark_issues = build_mark()
    print(f"mark-price bars: {len(mark):,} across {mark['symbol'].nunique() if len(mark) else 0} symbols {mark_issues}")
    print(json.dumps(report["totals"], indent=2))
    print(f"{report['symbols']} symbols, {report['funding_rows']:,} funding settlements → {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
