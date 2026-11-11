"""Shared selected DASH defaults for confirmation experiments."""
from __future__ import annotations

import copy
from typing import Any

import dash_purged_protocol as pp
import run_cf_long_memory_anchor as lm
from run_dash_purged_validation_budget_full504 import BudgetPurgedTargetProtocol

PERIODS = (96, 168, 336)
CYCLES = 3
RECENT_DECAY = 0.90
VALIDATION_FRACTION = 0.75


def branch_params(
    periods: list[int] | tuple[int, ...] = PERIODS,
    cycles: int = CYCLES,
    recent_decay: float = RECENT_DECAY,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stable = copy.deepcopy(lm.VARIANT_BY_ID["lm06_median3"])
    recent = copy.deepcopy(lm.VARIANT_BY_ID["lm07_ew3"])
    for params in (stable, recent):
        params["periods"] = list(periods)
        params["cycles"] = int(cycles)
    recent["decay"] = float(recent_decay)
    stable["variant_id"] = f"stable_selected_K{cycles}"
    recent["variant_id"] = f"recent_selected_K{cycles}_beta{recent_decay:g}"
    return stable, recent


def protocol(origins: int, horizon: int, fraction: float = VALIDATION_FRACTION):
    if float(fraction) == 1.0:
        return pp.PurgedTargetProtocol.build(origins, horizon)
    return BudgetPurgedTargetProtocol.build(origins, horizon, float(fraction))


def audit_fields() -> dict[str, Any]:
    return {
        "period_set": "P96+168+336",
        "periods": list(PERIODS),
        "cycles": CYCLES,
        "recent_decay": RECENT_DECAY,
        "validation_fraction": VALIDATION_FRACTION,
        "selected_default": True,
    }
