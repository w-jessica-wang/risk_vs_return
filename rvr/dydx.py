# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Third venue (PREREGISTRATION.md Amendment A5): dYdX v4, an on-chain order-book exchange with hourly funding,
whose public indexer is reachable from US networks and still serves markets that have been wound down
("FINAL_SETTLEMENT"), so the universe is survivorship-free.

Writes the same processed layout as rvr/hyperliquid.py (dydx_daily.parquet, dydx_funding.parquet), so
rvr/replication.py runs on either venue unchanged.

Run:  python -m rvr.dydx
"""
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from rvr.config import HOUR_MS, PROCESSED_DIR, RAW_DIR

API = "https://indexer.dydx.trade/v4"
RAW = RAW_DIR / "dydx"
START = pd.Timestamp("2023-10-01", tz="UTC")
END = pd.Timestamp("2026-09-01", tz="UTC")  # exclusive
PAGE = 1000
MIN_INTERVAL_S = 0.1   # requests are latency-bound (~1 s each); four workers stay well under the limit
WORKERS = 4
_lock, _last = threading.Lock(), [0.0]


def get(path, params=None, attempts=8):
    for attempt in range(attempts):
        with _lock:
            wait = _last[0] + MIN_INTERVAL_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.monotonic()
        try:
            r = requests.get(f"{API}{path}", params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(10)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"dYdX request failed after {attempts} attempts: {path} {params}")


def _ms(iso):
    return int(pd.Timestamp(iso).value // 10**6)


def universe():
    markets = get("/perpetualMarkets")["markets"].values()
    return pd.DataFrame([{"coin": m["ticker"], "status": m["status"],
                          "delisted": m["status"] != "ACTIVE"} for m in markets]).sort_values("coin")


def funding_history(ticker):
    """Hourly funding, newest first from the API; paged backwards with effectiveBeforeOrAt."""
    rows, before = [], END - pd.Timedelta(milliseconds=1)
    while True:
        page = get(f"/historicalFunding/{ticker}", {"limit": PAGE, "effectiveBeforeOrAt": before.isoformat()})
        page = page.get("historicalFunding", [])
        rows += page
        if len(page) < PAGE:
            break
        before = pd.Timestamp(page[-1]["effectiveAt"]) - pd.Timedelta(milliseconds=1)
        if before < START:
            break
    frame = pd.DataFrame(rows, columns=["ticker", "rate", "price", "effectiveAt"])
    calc = frame["effectiveAt"].map(_ms).astype("int64") if len(frame) else pd.Series(dtype="int64")
    out = pd.DataFrame({"coin": ticker, "calc_time": calc, "rate": frame["rate"].astype(float),
                        "oracle_price": frame["price"].astype(float)})
    out = out[(out["calc_time"] >= START.value // 10**6) & (out["calc_time"] < END.value // 10**6)]
    return out.drop_duplicates("calc_time").sort_values("calc_time").reset_index(drop=True)


def daily_candles(ticker):
    rows, to = [], END - pd.Timedelta(milliseconds=1)
    while True:
        page = get(f"/candles/perpetualMarkets/{ticker}",
                   {"resolution": "1DAY", "limit": PAGE, "toISO": to.isoformat(), "fromISO": START.isoformat()})
        page = page.get("candles", [])
        rows += page
        if len(page) < PAGE:
            break
        to = pd.Timestamp(page[-1]["startedAt"]) - pd.Timedelta(milliseconds=1)
    frame = pd.DataFrame(rows, columns=["startedAt", "open", "close", "high", "low", "usdVolume", "trades"])
    start = frame["startedAt"].map(_ms).astype("int64") if len(frame) else pd.Series(dtype="int64")
    out = pd.DataFrame({"coin": ticker, "open_time": start, "close_time": start + 24 * HOUR_MS - 1,
                        "open": frame["open"].astype(float), "high": frame["high"].astype(float),
                        "low": frame["low"].astype(float), "close": frame["close"].astype(float),
                        "quote_volume": frame["usdVolume"].astype(float), "trades": frame["trades"].astype("int64")})
    out = out[out["close_time"] < END.value // 10**6]
    return out.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def _save(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".part"))
    path.with_suffix(".part").replace(path)


def build():
    funding = pd.concat([pd.read_parquet(p) for p in sorted((RAW / "funding").glob("*.parquet"))], ignore_index=True)
    funding = funding.assign(settle_hour=((funding["calc_time"] + HOUR_MS // 2) // HOUR_MS) * HOUR_MS)
    candles = pd.concat([pd.read_parquet(p) for p in sorted((RAW / "candles").glob("*.parquet"))], ignore_index=True)
    funding.to_parquet(PROCESSED_DIR / "dydx_funding.parquet")
    candles.to_parquet(PROCESSED_DIR / "dydx_daily.parquet")
    return funding, candles


def main():
    markets = universe()
    RAW.mkdir(parents=True, exist_ok=True)
    markets.to_parquet(RAW / "universe.parquet")
    print(f"{len(markets)} markets ({int(markets['delisted'].sum())} wound down)", flush=True)
    def fetch_market(ticker):
        errors = []
        for kind, fetch in (("candles", daily_candles), ("funding", funding_history)):
            path = RAW / kind / f"{ticker}.parquet"
            if path.exists():
                continue
            try:
                _save(fetch(ticker), path)
            except Exception as exc:  # logged; a re-run retries
                errors.append((ticker, kind, repr(exc)))
        return errors

    failures = []
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(fetch_market, t) for t in markets["coin"]]
        for i, future in enumerate(as_completed(futures), 1):
            failures += future.result()
            if i % 25 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} markets, {len(failures)} failed", flush=True)
    if failures:
        print(failures[:10], file=sys.stderr)
        return 1
    build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
