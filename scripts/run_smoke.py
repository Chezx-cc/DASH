#!/usr/bin/env python3
"""Deterministic, download-free CPU smoke test for the selected DASH path."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHOD_SCRIPTS = ROOT / "repo" / "scripts"
sys.path.insert(0, str(METHOD_SCRIPTS))

import dash_selected_default as selected  # noqa: E402
import run_dash_purged_hyperparam_sensitivity as experiment  # noqa: E402
import run_e06_new_backbones as backbones  # noqa: E402


def sliding_split(
    values: np.ndarray,
    start: int,
    count: int,
    lookback: int,
    horizon: int,
) -> dict[str, np.ndarray]:
    inputs = np.stack(
        [values[i : i + lookback] for i in range(start, start + count)]
    ).astype(np.float32)
    trues = np.stack(
        [
            values[i + lookback : i + lookback + horizon]
            for i in range(start, start + count)
        ]
    ).astype(np.float32)
    preds = np.repeat(inputs[:, -1:, :], horizon, axis=1)
    return {"inputs": inputs, "trues": trues, "preds": preds}


def main() -> None:
    assert selected.PERIODS == (96, 168, 336)
    assert selected.CYCLES == 3
    assert selected.RECENT_DECAY == 0.90
    assert selected.VALIDATION_FRACTION == 0.75

    rng = np.random.default_rng(2027)
    lookback, horizon = 1008, 24
    validation_origins, test_origins = 180, 48
    length = lookback + validation_origins + test_origins + horizon
    time = np.arange(length, dtype=np.float32)
    values = np.stack(
        [
            np.sin(2.0 * np.pi * time / 96.0)
            + 0.35 * np.sin(2.0 * np.pi * time / 168.0),
            0.70 * np.cos(2.0 * np.pi * time / 96.0)
            + 0.20 * np.sin(2.0 * np.pi * time / 336.0),
        ],
        axis=1,
    ).astype(np.float32)
    values += 0.01 * rng.standard_normal(values.shape).astype(np.float32)

    validation = sliding_split(
        values, 0, validation_origins, lookback, horizon
    )
    test = sliding_split(
        values, validation_origins, test_origins, lookback, horizon
    )
    config = backbones.BackboneConfig(
        model="DLinear", dataset="ETTh1", pred_len=horizon, seed=2021
    )
    row = experiment.evaluate_effective_setting(
        config,
        validation,
        test,
        cycles=selected.CYCLES,
        decay=selected.RECENT_DECAY,
        validation_fraction=selected.VALIDATION_FRACTION,
    )
    metadata = json.loads(row["meta_json"])
    protocol = metadata["purged_protocol"]

    raw_mse = float(row["raw_mse"])
    dash_mse = float(row["method_mse"])
    assert np.isfinite(raw_mse) and np.isfinite(dash_mse)
    assert dash_mse < raw_mse
    assert row["uses_test_for_selection"] is False
    assert row["test_correction_reads_test_truth"] is False
    assert metadata["validation_history_stitched"] is True
    assert protocol["passed"] is True
    assert protocol["shared_target_timestamp_count"] == 0
    assert all(
        value == 0
        for value in protocol["cross_stage_shared_target_timestamps"].values()
    )

    print(f"Raw MSE:  {raw_mse:.6f}")
    print(f"DASH MSE: {dash_mse:.6f}")
    print("Causal checks: PASS")
    print("Smoke test: PASS")


if __name__ == "__main__":
    main()

