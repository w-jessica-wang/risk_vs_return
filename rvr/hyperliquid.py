# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Second venue for replication (PREREGISTRATION.md Amendment A3): Hyperliquid, an on-chain perpetuals
exchange with hourly funding, reachable from US networks, whose public API still serves delisted coins.

Its API keeps only the latest 5,000 candles per interval, so hourly bars reach back only months; daily
candles cover the whole sample. Funding is paginated at 500 settlements per request. Everything is cached
per coin under data/raw/hyperliquid and re-running resumes.

Run:  python -m rvr.hyperliquid            # every listed and delisted perp, 2023-05 → 2026-08
"""
import argparse
import json
import sys
import threading
import time

import pandas as pd
import requests

from rvr.config import HOUR_MS, PROCESSED_DIR, RAW_DIR

API = "https://api.hyperliquid.xyz/info"
RAW = RAW_DIR / "hyperliquid"
START_MS = int(pd.Timestamp("2023-05-01", tz="UTC").value // 10**6)
END_MS = int(pd.Timestamp("2026-09-01", tz="UTC").value // 10**6)  # exclusive
PAGE = 500
_lock = threading.Lock()
_last_call = [0.0]
MIN_INTERVAL_S = 3.0  # ~1,200 weight a minute per IP; a full 500-row funding page weighs ~45
MIN_PEAK_ADV = 1e6    # A3's lowest possible liquidity floor: a coin never this liquid can never be eligible
LISTING_LEAD_MS = 4 * 24 * HOUR_MS  # funding starts a few days before the first candle


def post(body, attempts=8):
    for attempt in range(attempts):
        with _lock:
            wait = _last_call[0] + MIN_INTERVAL_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _last_call[0] = time.monotonic()
        try:
            r = requests.post(API, json=body, timeout=30)
            if r.status_code == 429:
                time.sleep(10)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"Hyperliquid request failed after {attempts} attempts: {body}")


def universe():
    """Every perp in the exchange's metadata, delisted ones included (flagged)."""
    meta = post({"type": "meta"})
    return pd.DataFrame([{"coin": u["name"], "delisted": bool(u.get("isDelisted", False))} for u in meta["universe"]])


def funding_history(coin, start=START_MS, end=END_MS):
    rows, cursor = [], start
    while cursor < end:
        page = post({"type": "fundingHistory", "coin": coin, "startTime": cursor, "endTime": end - 1}) or []
        rows += page
        if len(page) < PAGE:
            break
        cursor = int(page[-1]["time"]) + 1
    frame = pd.DataFrame(rows, columns=["coin", "fundingRate", "premium", "time"])
    return pd.DataFrame({"coin": coin, "calc_time": frame["time"].astype("int64"),
                         "rate": frame["fundingRate"].astype(float), "premium": frame["premium"].astype(float)}
                        ).drop_duplicates("calc_time").sort_values("calc_time").reset_index(drop=True)


def daily_candles(coin, start=START_MS, end=END_MS):
    rows = post({"type": "candleSnapshot", "req": {"coin": coin, "interval": "1d", "startTime": start,
                                                    "endTime": end - 1}}) or []
    frame = pd.DataFrame(rows, columns=["t", "T", "s", "i", "o", "c", "h", "l", "v", "n"])
    out = pd.DataFrame({"coin": coin, "open_time": frame["t"].astype("int64"), "close_time": frame["T"].astype("int64"),
                        "open": frame["o"].astype(float), "high": frame["h"].astype(float),
                        "low": frame["l"].astype(float), "close": frame["c"].astype(float),
                        "volume": frame["v"].astype(float), "trades": frame["n"].astype("int64")})
    out["quote_volume"] = out["volume"] * (out["open"] + out["close"]) / 2  # base volume × average price
    return out[out["close_time"] < end].reset_index(drop=True)


def _save(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".part"))
    path.with_suffix(".part").replace(path)


def fetch_candles(coin):
    path = RAW / "candles" / f"{coin}.parquet"
    if not path.exists():
        _save(daily_candles(coin), path)
    return pd.read_parquet(path)


def needs_funding(candles):
    """Funding is fetched only for coins whose 7-day ADV ever reaches A3's lowest floor; uses volume only."""
    if candles.empty:
        return False
    adv = candles.set_index("open_time")["quote_volume"].rolling(7, min_periods=1).mean()
    return bool(adv.max() >= MIN_PEAK_ADV)


def fetch_funding(coin, candles):
    path = RAW / "funding" / f"{coin}.parquet"
    if not path.exists():
        start = max(START_MS, int(candles["open_time"].min()) - LISTING_LEAD_MS)
        _save(funding_history(coin, start=start), path)


def build():
    """Processed panels in the Binance layout: hl_funding.parquet and hl_daily.parquet."""
    funding = pd.concat([pd.read_parquet(p) for p in sorted((RAW / "funding").glob("*.parquet"))
                         if p.stat().st_size], ignore_index=True)
    funding = funding.assign(settle_hour=((funding["calc_time"] + HOUR_MS // 2) // HOUR_MS) * HOUR_MS)
    candles = pd.concat([pd.read_parquet(p) for p in sorted((RAW / "candles").glob("*.parquet"))], ignore_index=True)
    funding.to_parquet(PROCESSED_DIR / "hl_funding.parquet")
    candles.to_parquet(PROCESSED_DIR / "hl_daily.parquet")
    return funding, candles


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coins", nargs="+")
    args = parser.parse_args(argv)
    coins = universe()
    RAW.mkdir(parents=True, exist_ok=True)
    coins.to_parquet(RAW / "universe.parquet")
    names = args.coins or coins["coin"].tolist()
    print(f"{len(names)} coins ({int(coins['delisted'].sum())} delisted)", flush=True)
    failures, candles = [], {}
    for coin in names:
        try:
            candles[coin] = fetch_candles(coin)
        except Exception as exc:  # logged; a re-run retries
            failures.append((coin, "candles", repr(exc)))
    liquid = [c for c in names if c in candles and needs_funding(candles[c])]
    print(f"candles done; {len(liquid)} coins ever reach ${MIN_PEAK_ADV:,.0f} ADV and need funding", flush=True)
    for i, coin in enumerate(liquid, 1):
        try:
            fetch_funding(coin, candles[coin])
        except Exception as exc:
            failures.append((coin, "funding", repr(exc)))
        if i % 10 == 0 or i == len(liquid):
            print(f"  funding {i}/{len(liquid)} coins, {len(failures)} failed", flush=True)
    (RAW / "manifest.json").write_text(json.dumps({"coins": len(names), "failures": failures}, indent=2))
    if failures:
        print(f"{len(failures)} coins failed; re-run to retry", file=sys.stderr)
        return 1
    build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
