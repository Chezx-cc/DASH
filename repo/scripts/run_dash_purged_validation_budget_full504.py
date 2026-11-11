"""Validation-label budget sensitivity for DASH-Purged.

Set ``DASH_PURGED_VAL_BUDGET`` to 0.25, 0.50, or 0.75.  The experiment
retains the most recent fraction of validation forecast origins for label
calibration while preserving the complete observed validation history.
The retained target timeline is repartitioned into disjoint V1/V2/V3 blocks.
The 1.00 endpoint is the completed DASH-Purged main experiment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

import run_dash_purged_full504 as dash
import run_e06_new_backbones as nb


FRACTION = float(os.environ.get("DASH_PURGED_VAL_BUDGET", "0.50"))
if FRACTION not in (0.25, 0.50, 0.75):
    raise ValueError(
        "DASH_PURGED_VAL_BUDGET must be 0.25, 0.50, or 0.75; "
        "the 1.00 endpoint reuses the completed DASH-Purged main result"
    )


@dataclass(frozen=True)
class BudgetPurgedTargetProtocol:
    """Purged protocol restricted to the latest validation-origin budget."""

    origins: int
    horizon: int
    origin_start: int
    fraction: float
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]

    @classmethod
    def build(
        cls, origins: int, horizon: int, fraction: float
    ) -> "BudgetPurgedTargetProtocol":
        if origins < 1 or horizon < 1:
            raise ValueError(
                f"origins/horizon must be positive, got {origins}/{horizon}"
            )
        retained = max(3, int(round(origins * fraction)))
        retained = min(origins, retained)
        origin_start = origins - retained
        target_timestamps = np.arange(
            origin_start + 1,
            origins + horizon,
            dtype=np.int64,
        )
        chunks = np.array_split(target_timestamps, 3)
        if any(len(chunk) == 0 for chunk in chunks):
            raise ValueError("validation target timeline is too short for V1/V2/V3")
        bounds = tuple((int(chunk[0]), int(chunk[-1])) for chunk in chunks)
        return cls(
            origins=origins,
            horizon=horizon,
            origin_start=origin_start,
            fraction=float(fraction),
            bounds=bounds,  # type: ignore[arg-type]
        )

    def mask(
        self,
        stage: int,
        origin_indices: np.ndarray,
        horizon_indices: np.ndarray,
    ) -> np.ndarray:
        origin_indices = np.asarray(origin_indices, dtype=np.int64)
        low, high = self.bounds[int(stage)]
        timestamps = (
            origin_indices[:, None]
            + np.asarray(horizon_indices, dtype=np.int64)[None, :]
            + 1
        )
        return (
            (origin_indices[:, None] >= self.origin_start)
            & (timestamps >= low)
            & (timestamps <= high)
        )

    def eligible_origins(self, stage: int) -> np.ndarray:
        origins = np.arange(self.origin_start, self.origins, dtype=np.int64)
        mask = self.mask(
            stage, origins, np.arange(self.horizon, dtype=np.int64)
        )
        return origins[np.any(mask, axis=1)]

    def cell_count(self, stage: int) -> int:
        total = 0
        horizons = np.arange(self.horizon, dtype=np.int64)
        for begin in range(self.origin_start, self.origins, 2048):
            origins = np.arange(
                begin, min(self.origins, begin + 2048), dtype=np.int64
            )
            total += int(np.sum(self.mask(stage, origins, horizons)))
        return total

    def audit(self) -> dict[str, Any]:
        sets = [set(range(low, high + 1)) for low, high in self.bounds]
        intersections = {
            "V1_V2": len(sets[0] & sets[1]),
            "V1_V3": len(sets[0] & sets[2]),
            "V2_V3": len(sets[1] & sets[2]),
        }
        if any(intersections.values()):
            raise AssertionError(
                f"target timestamp overlap remains: {intersections}"
            )
        retained_origins = self.origins - self.origin_start
        return {
            "protocol": "latest_origin_budget_target_timestamp_partition",
            "target_timestamp_formula": "tau(origin,h)=origin+h+1",
            "validation_fraction": self.fraction,
            "full_validation_origins": self.origins,
            "retained_validation_origins": retained_origins,
            "first_retained_origin": self.origin_start,
            "preserves_full_observed_history": True,
            "unique_target_timestamp_budget": int(
                sum(high - low + 1 for low, high in self.bounds)
            ),
            "stage_target_bounds": {
                f"V{stage + 1}": list(bound)
                for stage, bound in enumerate(self.bounds)
            },
            "stage_origin_horizon_cells": {
                f"V{stage + 1}": self.cell_count(stage) for stage in range(3)
            },
            "cross_stage_shared_target_timestamps": intersections,
            "shared_target_timestamp_count": 0,
            "passed": True,
        }


_main_evaluate = dash.evaluate


def evaluate_budget(
    config: nb.BackboneConfig,
    val: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    protocol = BudgetPurgedTargetProtocol.build(
        len(val["preds"]), config.pred_len, FRACTION
    )
    rows = _main_evaluate(
        config,
        val,
        test,
        protocol_override=protocol,
    )
    for row in rows:
        row["validation_fraction"] = FRACTION
        row["retained_validation_origins"] = (
            len(val["preds"]) - protocol.origin_start
        )
    return rows


if __name__ == "__main__":
    suffix = int(round(FRACTION * 100))
    dash.DASH_METHOD = f"dash_purged_budget_{suffix}"
    dash.MATCHED_BASELINE = f"periodic_ridge_1008_purged_budget_{suffix}"
    dash.evaluate = evaluate_budget
    dash.main()
