# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Every number PREREGISTRATION.md fixes, in one place. Section references point to that file.
Tests build smaller Spec instances; the pipeline uses SPEC.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Spec:
    # §1 samples (decision days, inclusive)
    design_start: str = "2020-01-01"
    design_end: str = "2025-08-30"
    holdout_start: str = "2025-09-01"
    holdout_end: str = "2026-08-30"
    forward_start: str = "2026-09-01"   # Amendment A4: data not downloaded when this was written
    forward_end: str = "2027-02-27"
    forward_gate_window: int = 180
    forward_gate_threshold: float = 0.5

    # §2 universe
    max_staleness_hours: int = 2
    min_history_hours: int = 720
    restart_gap_hours: int = 168
    min_adv: float = 5e6
    adv_days: int = 7
    fresh_funding_intervals: int = 2
    min_eligible: int = 20

    # §3 signal and portfolio
    decision_offset_minutes: int = 5
    execution_hour: int = 1
    signal_hours: int = 72
    quantile: float = 0.2

    # §4 execution and halts
    execution_window_hours: int = 2
    max_execution_delay_hours: int = 168
    halt_haircut: float = 0.20

    # §5 costs
    fee: float = 5.0e-4
    spread_tiers: tuple = ((1e9, 0.5e-4), (1e8, 2e-4), (2e7, 5e-4), (0.0, 10e-4))  # (ADV floor, half-spread)
    impact_coefficient: float = 1.0
    fallback_daily_vol: float = 0.05
    fallback_adv: float = 5e6
    aum: float = 1e6
    capacity_aums: tuple = (1e6, 1e7, 5e7, 1e8, 2.5e8, 5e8)

    # §6 risk controls
    vol_filter_quantile: float = 0.2
    vol_days: int = 30
    vol_min_obs: int = 20
    beta_days: int = 90
    beta_min_down_days: int = 10
    beta_min_days: int = 20
    hedge_symbol: str = "BTCUSDT"
    vol_target: float = 0.10
    vol_target_days: int = 60
    vol_target_min_obs: int = 20
    max_scale: float = 2.0
    validity_es_reduction: float = 0.25

    # §7 Fama-MacBeth
    fm_min_contracts: int = 20
    fm_winsor: float = 0.01
    momentum_days: int = 30

    # §8 statistics
    bootstrap_block: int = 20
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20261012
    newey_west_lags: int = 20
    es_levels: tuple = (0.01, 0.05)
    crash_dates: tuple = ("2020-03-12", "2021-05-19", "2022-05-09", "2022-06-13", "2022-11-08", "2024-08-05")
    holdout_crash_dates: tuple = ("2025-10-10",)
    crash_window: tuple = (-2, 5)

    # §9 trial variants: name → signal window
    signal_variants: dict = field(default_factory=lambda: {"B_carry24h": 24, "B_carry168h": 168})


SPEC = Spec()
