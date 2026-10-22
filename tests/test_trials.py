# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Trial-log tests (PREREGISTRATION.md §9) — re-running an unchanged design adds nothing, a changed or new variant
is appended, and the deflated Sharpe sees one entry per distinct variant.

Run:  pytest tests/ -q
"""
import json

import numpy as np
import pandas as pd

from rvr.run import _update_trials


def _daily(values):
    return pd.DataFrame({"net": values}, index=range(len(values)))


def test_rerunning_an_unchanged_design_adds_nothing(tmp_path):
    daily = {"B": _daily(np.array([0.01, -0.02, 0.03])), "M": _daily(np.array([0.0, 0.01, -0.01]))}
    summary = {"B": {"sharpe": 1.0}, "M": {"sharpe": 0.5}}
    days = pd.Index(range(3))
    first = _update_trials(tmp_path, summary, "design", daily, days)
    second = _update_trials(tmp_path, summary, "design", daily, days)
    assert len(first) == len(second) == 2
    assert len((tmp_path / "trials.jsonl").read_text().splitlines()) == 2


def test_a_new_or_changed_variant_is_appended_and_counted_once(tmp_path):
    days = pd.Index(range(3))
    _update_trials(tmp_path, {"B": {"sharpe": 1.0}}, "design", {"B": _daily(np.array([0.01, -0.02, 0.03]))}, days)
    trials = _update_trials(tmp_path, {"B": {"sharpe": 2.0}, "C": {"sharpe": 0.1}}, "design",
                            {"B": _daily(np.array([0.02, -0.01, 0.03])), "C": _daily(np.array([0.0, 0.01, 0.02]))}, days)
    lines = [json.loads(line) for line in (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert [entry["variant"] for entry in lines] == ["B", "B", "C"]
    assert len(trials) == 2


def test_holdout_runs_never_log(tmp_path):
    days = pd.Index(range(3))
    _update_trials(tmp_path, {"B": {"sharpe": 1.0}}, "holdout", {"B": _daily(np.array([0.01, -0.02, 0.03]))}, days)
    log = tmp_path / "trials.jsonl"
    assert not log.exists() or not log.read_text()
