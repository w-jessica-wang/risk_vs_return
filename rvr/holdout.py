# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
The sealed holdout (PREREGISTRATION.md §1, §10).

load_panels() drops every price bar and funding record at or after HOLDOUT_START unless it is
explicitly unsealed, and every unseal is appended to HOLDOUT_LOG.md with its time and reason. A
design-period run therefore cannot read the holdout even by accident: the rows are not in memory.
"""
import datetime as dt
from pathlib import Path

import pandas as pd

from rvr.config import PROCESSED_DIR, ROOT
from rvr.spec import SPEC

HOLDOUT_START_MS = int(pd.Timestamp(SPEC.holdout_start, tz="UTC").value // 10**6)
HOLDOUT_LOG = ROOT / "HOLDOUT_LOG.md"
PRICE_COLUMNS = ["symbol", "open_time", "close_time", "high", "low", "close", "quote_volume"]


class SealedHoldoutError(RuntimeError):
    """A design-period run tried to touch the holdout."""


def load_panels(processed_dir=PROCESSED_DIR, unseal=False, holdout_start_ms=HOLDOUT_START_MS):
    """(prices, funding). Without unseal, nothing at or after the holdout start is returned."""
    processed_dir = Path(processed_dir)
    prices = pd.read_parquet(processed_dir / "prices_1h", columns=PRICE_COLUMNS)
    funding = pd.read_parquet(processed_dir / "funding.parquet")
    prices["symbol"] = prices["symbol"].astype(str)
    if not unseal:
        prices = prices[prices["close_time"] < holdout_start_ms]
        funding = funding[funding["calc_time"] < holdout_start_ms]
    return prices.reset_index(drop=True), funding.reset_index(drop=True)


def load_mark(processed_dir=PROCESSED_DIR, unseal=False, holdout_start_ms=HOLDOUT_START_MS):
    """Hourly mark-price closes (Amendment A7b); nothing at or after the holdout start unless unsealed."""
    path = Path(processed_dir) / "mark_1h.parquet"
    if not path.exists():
        return None
    mark = pd.read_parquet(path)
    mark["symbol"] = mark["symbol"].astype(str)
    if not unseal:
        mark = mark[mark["close_time"] < holdout_start_ms]
    return mark.reset_index(drop=True)


def check_sealed(last_exit_ms, unseal, holdout_start_ms=HOLDOUT_START_MS):
    """Refuse any evaluation whose last position exits inside the holdout, unless unsealed."""
    if last_exit_ms >= holdout_start_ms and not unseal:
        raise SealedHoldoutError("this evaluation reaches the sealed holdout; unseal it explicitly")


def record_unseal(reason, log=HOLDOUT_LOG):
    if not reason or not reason.strip():
        raise ValueError("unsealing the holdout needs a stated reason")
    log = Path(log)
    if not log.exists():
        log.write_text("# Holdout log\n\nEvery time the sealed holdout was read, and why.\n\n")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with log.open("a") as f:
        f.write(f"- {stamp} — {reason.strip()}\n")
