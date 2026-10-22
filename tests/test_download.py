# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Download tests — an archive reaches disk only after its checksum verifies, so a file that
exists is a file that can be trusted, and a failed listing is recorded rather than fatal.
No network: fake fetchers stand in for Binance.

Run:  pytest tests/ -q
"""
import hashlib

import pytest
import requests

from rvr import download
from rvr.download import ChecksumError, fetch_file, local_path, plan, verify_checksum

KEY = "data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip"
PAYLOAD = b"PK\x03\x04 stand-in archive bytes"
GOOD_CHECKSUM = f"{hashlib.sha256(PAYLOAD).hexdigest()}  BTCUSDT-fundingRate-2024-01.zip"


def _never(url):
    raise AssertionError(f"unexpected network call: {url}")


def test_verify_checksum():
    assert verify_checksum(PAYLOAD, GOOD_CHECKSUM)
    assert not verify_checksum(PAYLOAD + b"x", GOOD_CHECKSUM)
    assert not verify_checksum(PAYLOAD, "")


def test_verified_archive_is_written_under_raw_dir(tmp_path):
    requested = []

    def get_text(url):
        requested.append(url)
        return GOOD_CHECKSUM

    assert fetch_file(KEY, raw_dir=tmp_path, get_bytes=lambda url: PAYLOAD, get_text=get_text) == "downloaded"
    assert (tmp_path / "fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip").read_bytes() == PAYLOAD
    assert requested == [f"https://data.binance.vision/{KEY}.CHECKSUM"]


def test_cached_archive_is_not_refetched(tmp_path):
    fetch_file(KEY, raw_dir=tmp_path, get_bytes=lambda url: PAYLOAD, get_text=lambda url: GOOD_CHECKSUM)
    assert fetch_file(KEY, raw_dir=tmp_path, get_bytes=_never, get_text=_never) == "cached"


def test_checksum_mismatch_leaves_nothing_behind(tmp_path):
    with pytest.raises(ChecksumError):
        fetch_file(KEY, raw_dir=tmp_path, get_bytes=lambda url: PAYLOAD, get_text=lambda url: "0" * 64 + "  x")
    assert not any(tmp_path.rglob("*"))


def test_keys_outside_the_archive_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        local_path("data/futures/um/monthly/../../../etc/passwd", tmp_path)
    with pytest.raises(ValueError):
        local_path("somewhere/else.zip", tmp_path)


def test_a_failed_listing_is_recorded_not_fatal(monkeypatch):
    def fake_list(dataset, symbol, start, end, fetch=None):
        if symbol == "BADUSDT":
            raise requests.ConnectionError("Read timed out.")
        return [(f"{dataset}/{symbol}.zip", 10)]

    monkeypatch.setattr(download.listing, "list_monthly_files", fake_list)
    files, failures = plan(["AAAUSDT", "BADUSDT"], ["funding"], "2020-01", "2026-08", workers=2)
    assert files == [("funding", "funding/AAAUSDT.zip", 10)]
    assert [job for job, _ in failures] == ["listing funding BADUSDT"]
