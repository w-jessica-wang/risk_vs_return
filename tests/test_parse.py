# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Parser tests — every raw-archive quirk found while probing Binance's files is handled,
counted, and pinned here with the real rows that exposed it.

Run:  pytest tests/ -q
"""
import zipfile

import pandas as pd
import pytest

from rvr.config import HOUR_MS as H
from rvr.parse import (FUNDING_COLUMNS, KLINE_COLUMNS, parse_funding, parse_klines, read_csv_bytes,
                       read_zipped_csv, to_epoch_ms)

T = 1_704_067_200_000  # 2024-01-01 00:00 UTC

# First two rows of BTCUSDT-1h-2020-01.csv, an early archive with no header row.
BTC_2020_ROWS = [
    "1577836800000,7189.43,7190.52,7170.15,7171.55,2449.049,1577840399999,17576424.43970,3688,996.198,7149370.76353,0",
    "1577840400000,7171.43,7225,7171.10,7210.24,3865.038,1577843999999,27838046.00670,6635,2340.878,16860294.05346,0",
]


def _bytes(*lines):
    return ("\n".join(lines) + "\n").encode()


def _bar(hour, o=100.0, hi=101.0, lo=99.0, c=100.5, close_time=None):
    open_time = T + hour * H
    close_time = open_time + H - 1 if close_time is None else close_time
    return f"{open_time},{o},{hi},{lo},{c},1.0,{close_time},100.0,1,0.5,50.0,0"


def _klines(*rows):
    return read_csv_bytes(_bytes(*rows), KLINE_COLUMNS)


def test_headerless_and_headered_price_archives_parse_identically():
    headerless = parse_klines(read_csv_bytes(_bytes(*BTC_2020_ROWS), KLINE_COLUMNS), "BTCUSDT")[0]
    headered = parse_klines(read_csv_bytes(_bytes(",".join(KLINE_COLUMNS), *BTC_2020_ROWS), KLINE_COLUMNS),
                            "BTCUSDT")[0]
    pd.testing.assert_frame_equal(headerless, headered)
    assert headerless["open_time"].tolist() == [1577836800000, 1577840400000]
    assert headerless["close"].tolist() == [7171.55, 7210.24]


def test_microsecond_timestamps_are_normalised_to_milliseconds():
    assert to_epoch_ms([1_704_067_200_000_000, 1_704_067_200_000]).tolist() == [1_704_067_200_000] * 2


def test_funding_jitter_rounds_the_settlement_but_keeps_the_publish_time():
    # LUNAUSDT-fundingRate-2022-05.csv, during the collapse
    raw = read_csv_bytes(_bytes(",".join(FUNDING_COLUMNS), "1652342400000,8,-0.00750000",
                                "1652371200007,8,-0.00999975"), FUNDING_COLUMNS)
    funding, issues = parse_funding(raw, "LUNAUSDT")
    assert funding["calc_time"].tolist() == [1652342400000, 1652371200007]
    assert funding["settle_hour"].tolist() == [1652342400000, 1652371200000]
    assert issues["funding_jittered_calc_time"] == 1
    assert issues["funding_jitter_over_1min"] == 0


def test_rate_per_hour_puts_different_funding_schedules_on_one_scale():
    raw = pd.DataFrame({"calc_time": [T, T + 8 * H], "funding_interval_hours": [8, 4],
                        "last_funding_rate": [0.0008, 0.0004]})
    funding, _ = parse_funding(raw, "XUSDT")
    assert funding["rate_per_hour"].tolist() == pytest.approx([0.0001, 0.0001])


def test_conflicting_funding_duplicates_keep_the_latest_publication():
    raw = pd.DataFrame({"calc_time": [T, T + 5], "funding_interval_hours": [8, 8],
                        "last_funding_rate": [0.0001, 0.0002]})
    funding, issues = parse_funding(raw, "XUSDT")
    assert funding["rate"].tolist() == [0.0002]
    assert issues["funding_conflicting_duplicates"] == 1


def test_bars_with_impossible_prices_are_dropped_and_counted():
    frame, issues = parse_klines(_klines(_bar(0), _bar(1, hi=100.0, c=100.5)), "XUSDT")  # high below close
    assert frame["open_time"].tolist() == [T]
    assert issues["klines_inconsistent_ohlc"] == 1


def test_partial_final_bar_is_kept_with_its_true_close_time():
    early_close = T + H + 30 * 60_000
    frame, issues = parse_klines(_klines(_bar(0), _bar(1, close_time=early_close)), "XUSDT")
    assert frame["close_time"].iloc[-1] == early_close
    assert issues["klines_partial_bar"] == 1


def test_missing_hours_are_counted_not_filled():
    frame, issues = parse_klines(_klines(_bar(0), _bar(1), _bar(3)), "XUSDT")
    assert len(frame) == 3
    assert issues["klines_missing_hours"] == 1


def test_zero_volume_bars_are_kept_and_counted():
    zombie = f"{T + H},100.0,100.0,100.0,100.0,0.0,{T + 2 * H - 1},0.0,0,0.0,0.0,0"
    frame, issues = parse_klines(_klines(_bar(0), zombie), "FTMUSDT")
    assert len(frame) == 2
    assert issues["klines_zero_volume_bars"] == 1


def test_empty_archive_parses_to_an_empty_frame():
    funding, issues = parse_funding(read_csv_bytes(b"", FUNDING_COLUMNS), "XUSDT")
    assert funding.empty
    assert sum(issues.values()) == 0


def test_zip_roundtrip_and_multi_csv_archives_are_rejected(tmp_path):
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as archive:
        archive.writestr("BTCUSDT-1h-2020-01.csv", _bytes(*BTC_2020_ROWS))
    assert len(read_zipped_csv(good, KLINE_COLUMNS)) == 2

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("a.csv", "1")
        archive.writestr("b.csv", "2")
    with pytest.raises(ValueError, match="exactly one CSV"):
        read_zipped_csv(bad, KLINE_COLUMNS)
