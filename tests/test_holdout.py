# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Holdout tests — a sealed load never returns a holdout row, an evaluation that reaches the holdout
is refused, and every unseal is logged with a reason. Also pins the frozen non-crypto list.

Run:  pytest tests/ -q
"""
import pandas as pd
import pytest

from rvr.holdout import SealedHoldoutError, check_sealed, load_panels, record_unseal
from rvr.universe import COMMODITY, INDEX, NON_CRYPTO, STABLECOIN, TRADFI, is_crypto

CUTOFF = 1_756_684_800_000  # 2025-09-01 00:00 UTC
H = 3_600_000


def _panels(tmp_path):
    (tmp_path / "prices_1h").mkdir()
    open_time = CUTOFF + pd.Series([-3, -2, -1, 0, 1], dtype="int64") * H
    pd.DataFrame({"symbol": "AAAUSDT", "open_time": open_time, "close_time": open_time + H - 1,
                  "high": 1.0, "low": 1.0, "close": 1.0, "quote_volume": 1.0}).to_parquet(
        tmp_path / "prices_1h" / "AAAUSDT.parquet")
    pd.DataFrame({"symbol": "AAAUSDT", "calc_time": CUTOFF + pd.Series([-H, 14, H]), "settle_hour": 0,
                  "interval_h": 8, "rate": 0.0001, "rate_per_hour": 0.0}).to_parquet(tmp_path / "funding.parquet")
    return tmp_path


def test_sealed_load_returns_nothing_from_the_holdout(tmp_path):
    prices, funding = load_panels(_panels(tmp_path), holdout_start_ms=CUTOFF)
    assert (prices["close_time"] < CUTOFF).all() and len(prices) == 3  # the bar closing at 00:00 − 1ms is in
    assert (funding["calc_time"] < CUTOFF).all() and len(funding) == 1


def test_unsealed_load_returns_everything(tmp_path):
    prices, funding = load_panels(_panels(tmp_path), unseal=True, holdout_start_ms=CUTOFF)
    assert len(prices) == 5 and len(funding) == 3


def test_an_evaluation_reaching_the_holdout_is_refused_unless_unsealed():
    with pytest.raises(SealedHoldoutError):
        check_sealed(CUTOFF, unseal=False, holdout_start_ms=CUTOFF)
    check_sealed(CUTOFF - 1, unseal=False, holdout_start_ms=CUTOFF)
    check_sealed(CUTOFF, unseal=True, holdout_start_ms=CUTOFF)


def test_unsealing_is_logged_and_needs_a_reason(tmp_path):
    log = tmp_path / "HOLDOUT_LOG.md"
    with pytest.raises(ValueError):
        record_unseal("  ", log=log)
    record_unseal("design frozen; approved by Jessica", log=log)
    assert "design frozen; approved by Jessica" in log.read_text()


def test_non_crypto_list_is_frozen_and_disjoint():
    assert len(NON_CRYPTO) == 186
    assert sum(map(len, (INDEX, STABLECOIN, COMMODITY, TRADFI))) == len(NON_CRYPTO)
    assert all(is_crypto(s) for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LUNAUSDT", "INXUSDT", "USUSDT"])
    assert not any(is_crypto(s) for s in ["AAPLUSDT", "SPYUSDT", "PAXGUSDT", "BTCDOMUSDT", "USDCUSDT", "CLUSDT"])
