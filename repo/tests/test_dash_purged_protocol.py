from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import dash_purged_protocol as pp  # noqa: E402
from run_dash_purged_validation_budget_full504 import (  # noqa: E402
    BudgetPurgedTargetProtocol,
)


def target_timestamps(
    protocol: pp.PurgedTargetProtocol, stage: int
) -> set[int]:
    origins = np.arange(protocol.origins, dtype=np.int64)
    horizons = np.arange(protocol.horizon, dtype=np.int64)
    timestamps = origins[:, None] + horizons[None, :] + 1
    return set(timestamps[protocol.mask(stage, origins, horizons)].tolist())


def test_target_timestamps_are_disjoint_for_all_main_shapes() -> None:
    val_rows = {
        "ETTh1": 3484,
        "ETTh2": 3484,
        "ETTm1": 13936,
        "ETTm2": 13936,
        "Electricity": 2632,
        "Traffic": 1756,
        "Weather": 5270,
    }
    for rows in val_rows.values():
        for horizon in (96, 192, 336, 720):
            origins = rows - horizon + 1
            protocol = pp.PurgedTargetProtocol.build(origins, horizon)
            stages = [target_timestamps(protocol, stage) for stage in range(3)]
            assert not (stages[0] & stages[1])
            assert not (stages[0] & stages[2])
            assert not (stages[1] & stages[2])
            assert set.union(*stages) == set(range(1, origins + horizon))
            assert protocol.audit()["shared_target_timestamp_count"] == 0


def test_traffic_h720_remains_feasible() -> None:
    # A full-window H-1 embargo is impossible here, but target-cell purging
    # keeps every stage non-empty without sharing any target timestamp.
    protocol = pp.PurgedTargetProtocol.build(1037, 720)
    assert 1037 - 2 * (720 - 1) < 0
    assert all(protocol.cell_count(stage) > 0 for stage in range(3))


def test_safety_rule_is_exact() -> None:
    origins = np.arange(8)
    raw = np.ones(8)
    better = np.full(8, 0.9)
    accepted, gain, positive, worst = pp.safety_decision(origins, raw, better)
    assert accepted
    assert gain > 0
    assert positive == 4
    assert worst > 0

    mixed = np.asarray([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1.2, 1.2])
    accepted, _, positive, worst = pp.safety_decision(origins, raw, mixed)
    assert not accepted
    assert positive == 3
    assert worst < -0.10


def test_budget_protocol_preserves_history_and_purges_targets() -> None:
    protocol = BudgetPurgedTargetProtocol.build(
        origins=100, horizon=24, fraction=0.25
    )
    assert protocol.origin_start == 75
    assert protocol.eligible_origins(0).min() >= 75
    assert protocol.eligible_origins(1).min() >= 75
    assert protocol.eligible_origins(2).min() >= 75
    audit = protocol.audit()
    assert audit["passed"]
    assert audit["preserves_full_observed_history"]
    assert audit["retained_validation_origins"] == 25
    assert audit["shared_target_timestamp_count"] == 0
    assert audit["unique_target_timestamp_budget"] == 48
