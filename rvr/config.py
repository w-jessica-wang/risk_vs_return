# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""Project-wide constants: data locations, Binance archive endpoints, sample window."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Binance publishes bulk USDⓈ-M futures history here. The live REST API (fapi.binance.com)
# returns HTTP 451 from US networks, so nothing in this project depends on it.
DATA_BASE_URL = "https://data.binance.vision"
S3_LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
UM_MONTHLY_PREFIX = "data/futures/um/monthly"

# Archive folder for each dataset, relative to UM_MONTHLY_PREFIX.
DATASET_PREFIXES = {
    "funding": "fundingRate/{symbol}/",
    "klines_1h": "klines/{symbol}/1h/",
    "mark_1h": "markPriceKlines/{symbol}/1h/",   # Amendment A7b: funding is charged on the mark price
}

QUOTE_ASSET = "USDT"
SAMPLE_START = "2020-01"  # first month Binance archives USDⓈ-M funding rates
SAMPLE_END = "2026-08"    # last complete month when the pipeline was built (2026-09-15)

HOUR_MS = 3_600_000
