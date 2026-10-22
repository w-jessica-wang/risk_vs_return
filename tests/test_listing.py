# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Archive-listing tests — S3 pagination is followed to the end, and the universe keeps every
quoted perpetual the bucket has ever held. No network: a fake bucket serves the pages.

Run:  pytest tests/ -q
"""
import pytest

from rvr.listing import list_monthly_files, list_perp_symbols, list_prefix, parse_listing

NS = "http://s3.amazonaws.com/doc/2006-03-01/"
ROOT = "data/futures/um/monthly/fundingRate/"


def page(objects=(), prefixes=(), truncated=False, next_marker=None):
    parts = [f'<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns="{NS}">',
             f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>"]
    if next_marker:
        parts.append(f"<NextMarker>{next_marker}</NextMarker>")
    parts += [f"<Contents><Key>{key}</Key><Size>{size}</Size></Contents>" for key, size in objects]
    parts += [f"<CommonPrefixes><Prefix>{prefix}</Prefix></CommonPrefixes>" for prefix in prefixes]
    return "".join(parts) + "</ListBucketResult>"


class FakeBucket:
    def __init__(self, *pages):
        self.pages = list(pages)
        self.calls = []

    def __call__(self, url, params=None):
        self.calls.append(dict(params or {}))
        return self.pages.pop(0)


def test_parse_listing_reads_keys_sizes_and_prefixes():
    objects, prefixes, truncated, marker = parse_listing(
        page(objects=[("a.zip", 696)], prefixes=[ROOT + "BTCUSDT/"], truncated=True, next_marker="m"))
    assert objects == [("a.zip", 696)]
    assert prefixes == [ROOT + "BTCUSDT/"]
    assert truncated and marker == "m"


def test_pagination_follows_next_marker():
    bucket = FakeBucket(page(prefixes=[ROOT + "AUSDT/"], truncated=True, next_marker=ROOT + "AUSDT/"),
                        page(prefixes=[ROOT + "BUSDT/"]))
    _, prefixes = list_prefix(ROOT, delimiter="/", fetch=bucket)
    assert prefixes == [ROOT + "AUSDT/", ROOT + "BUSDT/"]
    assert "marker" not in bucket.calls[0]
    assert bucket.calls[1]["marker"] == ROOT + "AUSDT/"


def test_pagination_without_next_marker_resumes_after_the_last_key():
    bucket = FakeBucket(page(objects=[("k1", 1), ("k2", 1)], truncated=True), page(objects=[("k3", 1)]))
    objects, _ = list_prefix("p/", fetch=bucket)
    assert [key for key, _ in objects] == ["k1", "k2", "k3"]
    assert bucket.calls[1]["marker"] == "k2"


def test_truncated_page_with_nothing_to_resume_from_raises():
    with pytest.raises(RuntimeError, match="no way to continue"):
        list_prefix("p/", fetch=FakeBucket(page(truncated=True)))


def test_perp_symbols_keep_only_the_quote_asset_and_include_delisted_contracts():
    bucket = FakeBucket(page(prefixes=[ROOT + s + "/" for s in ["LUNAUSDT", "ADABUSD", "BTCUSDC", "BTCUSDT"]]))
    assert list_perp_symbols("USDT", fetch=bucket) == ["BTCUSDT", "LUNAUSDT"]


def test_monthly_files_skip_checksums_and_respect_the_sample_window():
    folder = "data/futures/um/monthly/klines/BTCUSDT/1h/"
    keys = [f"{folder}BTCUSDT-1h-{month}.zip{suffix}"
            for month in ["2019-12", "2020-01", "2026-08", "2026-09"] for suffix in ("", ".CHECKSUM")]
    bucket = FakeBucket(page(objects=[(key, 10) for key in keys]))
    found = list_monthly_files("klines_1h", "BTCUSDT", "2020-01", "2026-08", fetch=bucket)
    assert found == [(f"{folder}BTCUSDT-1h-2020-01.zip", 10), (f"{folder}BTCUSDT-1h-2026-08.zip", 10)]
    assert bucket.calls[0]["prefix"] == folder
