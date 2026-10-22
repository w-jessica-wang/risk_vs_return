# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Build tests — synthetic archives in both historical formats go through the real build and
come out as the panels rvr.pit expects, with every anomaly in the quality report.

Run:  pytest tests/ -q
"""
import json
import zipfile

import pandas as pd
import pytest

from rvr.build import build
from rvr.config import HOUR_MS as H
from rvr.parse import FUNDING_COLUMNS, KLINE_COLUMNS

JAN = 1_704_067_200_000  # 2024-01-01 00:00 UTC
FEB = 1_706_745_600_000  # 2024-02-01 00:00 UTC
FUNDING_HEADER = ",".join(FUNDING_COLUMNS)


def _archive(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(path.with_suffix(".csv").name, text)


def _bar(open_time, close):
    return f"{open_time},{close},{close + 1},{close - 1},{close},1.0,{open_time + H - 1},100.0,1,0.5,50.0,0"


def test_build_writes_point_in_time_ready_panels(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    _archive(raw / "fundingRate/AAAUSDT/AAAUSDT-fundingRate-2024-01.zip",
             f"{FUNDING_HEADER}\n{JAN + 8 * H},8,0.0004\n{JAN + 16 * H + 14},8,0.0008\n")
    # January in the old headerless price format, February with a header
    _archive(raw / "klines/AAAUSDT/1h/AAAUSDT-1h-2024-01.zip", f"{_bar(JAN, 100.0)}\n{_bar(JAN + H, 101.0)}\n")
    _archive(raw / "klines/AAAUSDT/1h/AAAUSDT-1h-2024-02.zip", ",".join(KLINE_COLUMNS) + f"\n{_bar(FEB, 102.0)}\n")
    # a contract with funding history but no price archive still gets a listing row
    _archive(raw / "fundingRate/BBBUSDT/BBBUSDT-fundingRate-2024-01.zip", f"{FUNDING_HEADER}\n{JAN + 8 * H},4,0.0002\n")

    report = build(["AAAUSDT", "BBBUSDT"], raw_dir=raw, processed_dir=out, workers=1)

    funding = pd.read_parquet(out / "funding.parquet")
    assert funding["symbol"].tolist() == ["AAAUSDT", "AAAUSDT", "BBBUSDT"]
    assert funding["calc_time"].iloc[1] == JAN + 16 * H + 14
    assert funding["rate_per_hour"].tolist() == pytest.approx([0.00005, 0.0001, 0.00005])

    prices = pd.read_parquet(out / "prices_1h" / "AAAUSDT.parquet")
    assert prices["close"].tolist() == [100.0, 101.0, 102.0]
    assert not (out / "prices_1h" / "BBBUSDT.parquet").exists()

    listings = pd.read_parquet(out / "listings.parquet").set_index("symbol")
    assert listings.loc["AAAUSDT", "first_bar_open"] == JAN
    assert listings.loc["AAAUSDT", "last_bar_open"] == FEB
    assert listings.loc["AAAUSDT", "last_traded_bar_open"] == FEB
    assert pd.isna(listings.loc["BBBUSDT", "n_bars"])

    assert report["totals"]["klines_missing_hours"] == (FEB - JAN) // H + 1 - 3
    assert report["totals"]["funding_jittered_calc_time"] == 1
    assert json.loads((out / "quality_report.json").read_text()) == report
