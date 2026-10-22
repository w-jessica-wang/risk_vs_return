# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Parsers for Binance USDⓈ-M monthly archives.

Every quirk in the raw files is handled here and counted, never silently absorbed:
  - early price archives have no header row; later ones do
  - funding calc_time carries a few milliseconds of publish jitter (e.g. 08:00:00.014)
  - the funding interval differs across contracts and over time (8h, 4h, 1h)
  - Binance's spot archives switched to microsecond timestamps in 2025; futures archives
    are still milliseconds, but timestamps are normalised defensively
"""
import io
import zipfile

import numpy as np
import pandas as pd

from rvr.config import HOUR_MS

FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
FUNDING_OUT = ["symbol", "calc_time", "settle_hour", "interval_h", "rate", "rate_per_hour"]

KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
KLINE_OUT = ["symbol", "open_time", "close_time", "open", "high", "low", "close", "volume",
             "quote_volume", "count", "taker_buy_quote_volume"]
_KLINE_FLOATS = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote_volume"]

# Publish jitter is a few milliseconds; anything past a minute is something else and is flagged.
MAX_JITTER_MS = 60_000


def read_zipped_csv(path, columns):
    """Read the single CSV inside a Binance archive, with or without a header row."""
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"{path}: expected exactly one CSV, found {members}")
        raw = archive.read(members[0])
    return read_csv_bytes(raw, columns)


def read_csv_bytes(raw, columns):
    """Parse archive CSV bytes into `columns`, whether or not the file has a header row."""
    raw = raw.removeprefix(b"\xef\xbb\xbf")
    if not raw.strip():
        return pd.DataFrame(columns=columns)
    first_field = raw.lstrip().split(b"\n", 1)[0].split(b",", 1)[0].strip().decode()
    has_header = not first_field.lstrip("-").isdigit()
    frame = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    if has_header:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError(f"missing columns {missing}; file has {list(frame.columns)}")
        return frame[columns]
    if frame.shape[1] != len(columns):
        raise ValueError(f"expected {len(columns)} columns, file has {frame.shape[1]}")
    frame.columns = columns
    return frame


def to_epoch_ms(values):
    """Epoch timestamps as int64 milliseconds; microsecond values (> 1e14) are divided down."""
    values = pd.Series(values).astype("int64")
    return values.where(values < 10**14, values // 1000)


def parse_funding(raw, symbol):
    """
    Tidy one symbol's funding history → (frame with FUNDING_OUT columns, issue counts).

    Sign convention (Binance): rate > 0 means longs pay shorts, so a short of notional N
    receives rate × N at settlement. rate_per_hour = rate / interval_h puts contracts on
    8h, 4h and 1h schedules on one scale.

    calc_time is kept exactly as published; rvr.pit uses it to decide when a rate became
    known. settle_hour (calc_time rounded to the nearest hour) only lines funding up with
    hourly prices when accruing P&L — it must never decide what was known.
    """
    frame = pd.DataFrame({
        "symbol": symbol,
        "calc_time": to_epoch_ms(raw["calc_time"]).to_numpy(),
        "interval_h": raw["funding_interval_hours"].astype("int64").to_numpy(),
        "rate": raw["last_funding_rate"].astype("float64").to_numpy(),
    })
    issues = {}

    invalid = ~np.isfinite(frame["rate"]) | (frame["interval_h"] <= 0)
    issues["funding_invalid_rows"] = int(invalid.sum())
    frame = frame[~invalid]

    n = len(frame)
    frame = frame.drop_duplicates()
    issues["funding_exact_duplicates"] = n - len(frame)

    frame = frame.assign(settle_hour=(frame["calc_time"] + HOUR_MS // 2) // HOUR_MS * HOUR_MS)
    jitter = (frame["calc_time"] - frame["settle_hour"]).abs()
    issues["funding_jittered_calc_time"] = int((jitter > 0).sum())
    issues["funding_jitter_over_1min"] = int((jitter > MAX_JITTER_MS).sum())

    n = len(frame)
    frame = frame.sort_values("calc_time", kind="mergesort").drop_duplicates("settle_hour", keep="last")
    issues["funding_conflicting_duplicates"] = n - len(frame)

    frame = frame.assign(rate_per_hour=frame["rate"] / frame["interval_h"])
    return frame[FUNDING_OUT].sort_values("settle_hour").reset_index(drop=True), issues


def parse_klines(raw, symbol):
    """
    Tidy one symbol's hourly bars → (frame with KLINE_OUT columns, issue counts).

    A bar is knowable only after its close_time (open_time + 1h − 1ms for a full bar; earlier
    for a contract's final, partial bar). Missing hours are counted, never filled.
    """
    frame = pd.DataFrame({
        "symbol": symbol,
        "open_time": to_epoch_ms(raw["open_time"]).to_numpy(),
        "close_time": to_epoch_ms(raw["close_time"]).to_numpy(),
        **{column: raw[column].astype("float64").to_numpy() for column in _KLINE_FLOATS},
        "count": raw["count"].astype("int64").to_numpy(),
    })
    issues = {}

    n = len(frame)
    frame = frame.sort_values("open_time", kind="mergesort").drop_duplicates("open_time", keep="last")
    issues["klines_duplicate_open_time"] = n - len(frame)

    full_close = frame["open_time"] + HOUR_MS - 1
    bad_time = ((frame["open_time"] % HOUR_MS != 0) | (frame["close_time"] < frame["open_time"])
                | (frame["close_time"] > full_close))
    issues["klines_bad_timestamps"] = int(bad_time.sum())
    frame = frame[~bad_time]
    issues["klines_partial_bar"] = int((frame["close_time"] < frame["open_time"] + HOUR_MS - 1).sum())

    ohlc = frame[["open", "high", "low", "close"]]
    bad_price = (~np.isfinite(ohlc).all(axis=1) | (frame["low"] <= 0)
                 | (frame["low"] > frame[["open", "close"]].min(axis=1))
                 | (frame["high"] < frame[["open", "close"]].max(axis=1)))
    issues["klines_inconsistent_ohlc"] = int(bad_price.sum())
    frame = frame[~bad_price]
    # Kept: Binance archives flat zero-volume bars for months after a delisting; rvr.pit ignores them.
    issues["klines_zero_volume_bars"] = int((frame["quote_volume"] <= 0).sum())

    span = (frame["open_time"].iloc[-1] - frame["open_time"].iloc[0]) // HOUR_MS + 1 if len(frame) else 0
    issues["klines_missing_hours"] = int(span - len(frame))
    return frame[KLINE_OUT].reset_index(drop=True), issues


def parse_mark(raw, symbol):
    """
    One symbol's hourly mark-price bars (Amendment A7b) → (symbol, close_time, mark) and issue counts. Same
    archive layout as trade bars; only full, on-the-hour bars with a positive close are kept.
    """
    frame = pd.DataFrame({"symbol": symbol, "open_time": to_epoch_ms(raw["open_time"]).to_numpy(),
                          "close_time": to_epoch_ms(raw["close_time"]).to_numpy(),
                          "mark": raw["close"].astype("float64").to_numpy()})
    n = len(frame)
    frame = frame.sort_values("open_time", kind="mergesort").drop_duplicates("open_time", keep="last")
    issues = {"mark_duplicate_open_time": n - len(frame)}
    good = ((frame["open_time"] % HOUR_MS == 0) & (frame["close_time"] == frame["open_time"] + HOUR_MS - 1)
            & np.isfinite(frame["mark"]) & (frame["mark"] > 0))
    issues["mark_bad_bars"] = int((~good).sum())
    return frame.loc[good, ["symbol", "close_time", "mark"]].reset_index(drop=True), issues
