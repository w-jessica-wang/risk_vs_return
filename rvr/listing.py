# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
List what Binance has archived, straight from the S3 bucket behind data.binance.vision.

The universe comes from the bucket listing rather than the exchange's current symbol list
because the listing still contains contracts that have since been delisted. Starting from
today's symbol list would silently drop every contract that died — survivorship bias.
"""
import re
import xml.etree.ElementTree as ET

from rvr import net
from rvr.config import DATASET_PREFIXES, S3_LIST_URL, UM_MONTHLY_PREFIX

_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
_MONTH = re.compile(r"-(\d{4}-\d{2})\.zip$")


def parse_listing(xml_text):
    """One ListBucketResult page → (objects [(key, size)], common prefixes, is_truncated, next_marker)."""
    root = ET.fromstring(xml_text)
    objects = [(item.findtext("s3:Key", None, _NS), int(item.findtext("s3:Size", "0", _NS)))
               for item in root.findall("s3:Contents", _NS)]
    prefixes = [p.text for p in root.findall("s3:CommonPrefixes/s3:Prefix", _NS)]
    truncated = root.findtext("s3:IsTruncated", "false", _NS) == "true"
    return objects, prefixes, truncated, root.findtext("s3:NextMarker", None, _NS)


def list_prefix(prefix, delimiter=None, fetch=None):
    """Every object and common prefix under `prefix`, following S3 pagination."""
    fetch = fetch or net.get_text
    objects, prefixes, marker = [], [], None
    while True:
        params = {"prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        if marker:
            params["marker"] = marker
        page_objects, page_prefixes, truncated, next_marker = parse_listing(fetch(S3_LIST_URL, params))
        objects += page_objects
        prefixes += page_prefixes
        if not truncated:
            return objects, prefixes
        # S3's v1 listing returns NextMarker only when a delimiter is set; otherwise resume after the last key.
        seen = [key for key, _ in page_objects] + page_prefixes
        if not (next_marker or seen):
            raise RuntimeError(f"truncated S3 listing for {prefix!r} with no way to continue")
        marker = next_marker or max(seen)


def list_perp_symbols(quote, fetch=None):
    """Every perpetual with an archived funding history in `quote`, including delisted ones."""
    root = f"{UM_MONTHLY_PREFIX}/fundingRate/"
    _, prefixes = list_prefix(root, delimiter="/", fetch=fetch)
    symbols = (p[len(root):].strip("/") for p in prefixes)
    return sorted(s for s in symbols if s.endswith(quote))


def list_monthly_files(dataset, symbol, start, end, fetch=None):
    """(key, size) of each monthly archive for one symbol with start <= month <= end (YYYY-MM)."""
    prefix = f"{UM_MONTHLY_PREFIX}/{DATASET_PREFIXES[dataset].format(symbol=symbol)}"
    objects, _ = list_prefix(prefix, fetch=fetch)
    return sorted((key, size) for key, size in objects
                  if (month := _MONTH.search(key)) and start <= month.group(1) <= end)
