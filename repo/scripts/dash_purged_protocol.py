"""Target-timestamp-purged calibration utilities for DASH.

The validation loader emits unit-stride forecast origins.  Adjacent origins
therefore have overlapping target windows.  Splitting only by origin index
causes the same target timestamp to appear in more than one calibration
stage.  This module assigns every origin/horizon label cell to exactly one
chronological target-timestamp block (V1, V2, or V3).

The three stages may share forecast origins and history, but they never read
the same target timestamp.  This is important for long horizons where a
full-window embargo would leave too few (or no) origins, e.g. Traffic-H720.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

import run_cf_long_memory_anchor as lm


Stage = int
FeatureFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, dict[str, Any], np.ndarray | None],
    np.ndarray,
]


@dataclass(frozen=True)
class PurgedTargetProtocol:
    """Three chronological, disjoint target-timestamp blocks."""

    origins: int
    horizon: int
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]

    @classmethod
    def build(cls, origins: int, horizon: int) -> "PurgedTargetProtocol":
        if origins < 1 or horizon < 1:
            raise ValueError(f"origins/horizon must be positive, got {origins}/{horizon}")
        # Target timestamp for origin i and zero-based horizon h is i+h+1.
        target_count = origins + horizon - 1
        chunks = np.array_split(np.arange(1, target_count + 1, dtype=np.int64), 3)
        if any(len(chunk) == 0 for chunk in chunks):
            raise ValueError("validation target timeline is too short for three stages")
        bounds = tuple((int(chunk[0]), int(chunk[-1])) for chunk in chunks)
        return cls(origins=origins, horizon=horizon, bounds=bounds)  # type: ignore[arg-type]

    def mask(
        self,
        stage: Stage,
        origin_indices: np.ndarray,
        horizon_indices: np.ndarray,
    ) -> np.ndarray:
        low, high = self.bounds[int(stage)]
        timestamps = (
            np.asarray(origin_indices, dtype=np.int64)[:, None]
            + np.asarray(horizon_indices, dtype=np.int64)[None, :]
            + 1
        )
        return (timestamps >= low) & (timestamps <= high)

    def eligible_origins(self, stage: Stage) -> np.ndarray:
        origins = np.arange(self.origins, dtype=np.int64)
        mask = self.mask(stage, origins, np.arange(self.horizon, dtype=np.int64))
        return origins[np.any(mask, axis=1)]

    def cell_count(self, stage: Stage) -> int:
        total = 0
        horizons = np.arange(self.horizon, dtype=np.int64)
        for begin in range(0, self.origins, 2048):
            origins = np.arange(begin, min(self.origins, begin + 2048), dtype=np.int64)
            total += int(np.sum(self.mask(stage, origins, horizons)))
        return total

    def audit(self) -> dict[str, Any]:
        sets = [set(range(low, high + 1)) for low, high in self.bounds]
        intersections = {
            "V1_V2": len(sets[0] & sets[1]),
            "V1_V3": len(sets[0] & sets[2]),
            "V2_V3": len(sets[1] & sets[2]),
        }
        passed = all(value == 0 for value in intersections.values())
        if not passed:
            raise AssertionError(f"target timestamp overlap remains: {intersections}")
        return {
            "protocol": "chronological_target_timestamp_partition",
            "target_timestamp_formula": "tau(origin,h)=origin+h+1",
            "unique_target_timestamp_budget": int(
                sum(high - low + 1 for low, high in self.bounds)
            ),
            "stage_target_bounds": {
                f"V{stage + 1}": list(bound) for stage, bound in enumerate(self.bounds)
            },
            "stage_origin_horizon_cells": {
                f"V{stage + 1}": self.cell_count(stage) for stage in range(3)
            },
            "cross_stage_shared_target_timestamps": intersections,
            "shared_target_timestamp_count": 0,
            "passed": True,
        }


def sampled(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if len(values) <= count:
        return values
    positions = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    return values[positions]


def branch_feature(
    timeline: np.ndarray,
    current: np.ndarray,
    inputs: np.ndarray,
    preds: np.ndarray,
    origins: np.ndarray,
    spec: tuple[int, int],
    params: dict[str, Any],
    channels: np.ndarray | None = None,
) -> np.ndarray:
    return lm.feature_correction(
        timeline, current, inputs, preds, origins, spec, params, channels
    )


def cycle_feature(
    timeline: np.ndarray,
    current: np.ndarray,
    inputs: np.ndarray,
    preds: np.ndarray,
    origins: np.ndarray,
    spec: tuple[int, int],
    params: dict[str, Any],
    channels: np.ndarray | None = None,
) -> np.ndarray:
    """One unaggregated periodic lag feature for the matched simple baseline."""
    del inputs, params
    period, cycle = int(spec[0]), int(spec[1])
    horizon = preds.shape[1]
    phase = np.arange(horizon, dtype=np.int64) % period
    source = current[origins, None] - cycle * period + 1 + phase[None, :]
    valid = (source >= 0) & (source <= current[origins, None])
    values = timeline[np.clip(source, 0, len(timeline) - 1)]
    selected_preds = preds[origins]
    if channels is not None:
        values = values[:, :, channels]
        selected_preds = selected_preds[:, :, channels]
    anchor = np.where(valid[:, :, None], values, selected_preds)
    return (anchor - selected_preds).astype(np.float32, copy=False)


def fit_feature_weights(
    timeline: np.ndarray,
    current: np.ndarray,
    inputs: np.ndarray,
    preds: np.ndarray,
    trues: np.ndarray,
    bins: np.ndarray,
    protocol: PurgedTargetProtocol,
    stage: Stage,
    specs: list[Any],
    params: dict[str, Any],
    feature_fn: FeatureFn,
    ridge: float,
    cap: float = 1.25,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidates = protocol.eligible_origins(stage)
    max_period = max(int(spec[0]) * (int(spec[1]) if feature_fn is cycle_feature else 1) for spec in specs)
    first_available = max(0, max_period - inputs.shape[1])
    eligible = candidates[candidates >= first_available]
    if len(eligible) < 8:
        eligible = candidates
    origins = sampled(eligible, 96)
    channels = sampled(np.arange(preds.shape[2], dtype=np.int64), 128)
    num_bins = int(bins.max()) + 1
    weights = np.zeros((num_bins, len(specs)), dtype=np.float32)

    for b in range(num_bins):
        horizons = np.where(bins == b)[0].astype(np.int64)
        horizons = sampled(horizons, 48)
        valid = protocol.mask(stage, origins, horizons)
        if not np.any(valid):
            continue
        parts = [
            feature_fn(
                timeline, current, inputs, preds, origins, spec, params, channels
            )[:, horizons]
            for spec in specs
        ]
        design = np.stack(parts, axis=-1)
        target = (
            trues[np.ix_(origins, horizons, channels)]
            - preds[np.ix_(origins, horizons, channels)]
        )
        x = design[valid].reshape(-1, len(specs))
        y = target[valid].reshape(-1)
        if not len(y):
            continue
        target_clip = max(float(np.quantile(np.abs(y), 0.95)), 1e-8)
        weights[b] = lm.solve_weights(
            x,
            np.clip(y, -target_clip, target_clip),
            ridge,
            cap=cap,
        )
    return weights, {
        "fit_stage": f"V{stage + 1}",
        "fit_origins": int(len(origins)),
        "fit_first_origin": int(origins[0]) if len(origins) else None,
        "fit_last_origin": int(origins[-1]) if len(origins) else None,
        "label_protocol": "target_timestamp_purged",
    }


def feature_correction(
    timeline: np.ndarray,
    current: np.ndarray,
    inputs: np.ndarray,
    preds: np.ndarray,
    origins: np.ndarray,
    bins: np.ndarray,
    specs: list[Any],
    params: dict[str, Any],
    feature_fn: FeatureFn,
    weights: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(preds[origins], dtype=np.float32)
    for feature, spec in enumerate(specs):
        correction = feature_fn(
            timeline, current, inputs, preds, origins, spec, params, None
        )
        result += weights[bins, feature][None, :, None] * correction
    return result


def fit_clip(
    timeline: np.ndarray,
    current: np.ndarray,
    inputs: np.ndarray,
    preds: np.ndarray,
    bins: np.ndarray,
    protocol: PurgedTargetProtocol,
    stage: Stage,
    specs: list[Any],
    params: dict[str, Any],
    feature_fn: FeatureFn,
    weights: np.ndarray,
    quantile: float,
) -> float:
    origins = sampled(protocol.eligible_origins(stage), 64)
    correction = feature_correction(
        timeline,
        current,
        inputs,
        preds,
        origins,
        bins,
        specs,
        params,
        feature_fn,
        weights,
    )
    return max(float(np.quantile(np.abs(correction), quantile)), 1e-8)


def tune_gamma(
    timeline: np.ndarray,
    current: np.ndarray,
    inputs: np.ndarray,
    preds: np.ndarray,
    trues: np.ndarray,
    bins: np.ndarray,
    protocol: PurgedTargetProtocol,
    stage: Stage,
    specs: list[Any],
    params: dict[str, Any],
    feature_fn: FeatureFn,
    weights: np.ndarray,
    clip: float,
    gamma_grid: np.ndarray = lm.GAMMA_GRID,
) -> np.ndarray:
    num_bins = int(bins.max()) + 1
    sums = np.zeros((num_bins, 3), dtype=np.float64)
    origins = protocol.eligible_origins(stage)
    step = lm.chunk_size(preds)
    for begin in range(0, len(origins), step):
        selected = origins[begin : begin + step]
        correction = np.clip(
            feature_correction(
                timeline,
                current,
                inputs,
                preds,
                selected,
                bins,
                specs,
                params,
                feature_fn,
                weights,
            ),
            -clip,
            clip,
        ).astype(np.float64, copy=False)
        error = (preds[selected] - trues[selected]).astype(np.float64, copy=False)
        for b in range(num_bins):
            horizons = np.where(bins == b)[0]
            valid = protocol.mask(stage, selected, horizons)
            if not np.any(valid):
                continue
            e = error[:, horizons][valid]
            c = correction[:, horizons][valid]
            sums[b] += [np.sum(e * e), np.sum(e * c), np.sum(c * c)]
    gamma = np.zeros(num_bins, dtype=np.float32)
    for b in range(num_bins):
        candidate = (
            sums[b, 0]
            + 2.0 * gamma_grid * sums[b, 1]
            + gamma_grid**2 * sums[b, 2]
        )
        gamma[b] = float(gamma_grid[int(np.argmin(candidate))])
    return gamma


def correction_from_state(
    state: dict[str, Any],
    inputs: np.ndarray,
    preds: np.ndarray,
    origins: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    value = feature_correction(
        state["timeline"],
        state["current"],
        inputs,
        preds,
        origins,
        bins,
        state["specs"],
        state["params"],
        state["feature_fn"],
        state["weights"],
    )
    value = np.clip(value, -state["clip"], state["clip"])
    return value * state["gamma"][bins][None, :, None]


def prepare_state(
    params: dict[str, Any],
    val: dict[str, np.ndarray],
    bins: np.ndarray,
    protocol: PurgedTargetProtocol,
    *,
    specs: list[Any] | None = None,
    feature_fn: FeatureFn = branch_feature,
) -> dict[str, Any]:
    timeline, current = lm.observed_timeline(val["inputs"])
    if specs is None:
        specs = list(lm.feature_specs(params))
    weights, meta = fit_feature_weights(
        timeline,
        current,
        val["inputs"],
        val["preds"],
        val["trues"],
        bins,
        protocol,
        0,
        specs,
        params,
        feature_fn,
        ridge=float(params["ridge"]),
    )
    clip = fit_clip(
        timeline,
        current,
        val["inputs"],
        val["preds"],
        bins,
        protocol,
        0,
        specs,
        params,
        feature_fn,
        weights,
        float(params["clip_q"]),
    )
    gamma = tune_gamma(
        timeline,
        current,
        val["inputs"],
        val["preds"],
        val["trues"],
        bins,
        protocol,
        1,
        specs,
        params,
        feature_fn,
        weights,
        clip,
    )
    return {
        "params": params,
        "timeline": timeline,
        "current": current,
        "specs": specs,
        "feature_fn": feature_fn,
        "weights": weights,
        "clip": clip,
        "gamma": gamma,
        **meta,
    }


def origin_losses(
    error: np.ndarray,
    candidate: np.ndarray,
    origins: np.ndarray,
    horizons: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channels = error.shape[2]
    denom = valid.sum(axis=1).astype(np.float64) * channels
    keep = denom > 0
    base_sq = np.sum(
        np.square(error[:, horizons], dtype=np.float64)
        * valid[:, :, None],
        axis=(1, 2),
    )
    cand_sq = np.sum(
        np.square(candidate[:, horizons], dtype=np.float64)
        * valid[:, :, None],
        axis=(1, 2),
    )
    return origins[keep], base_sq[keep] / denom[keep], cand_sq[keep] / denom[keep]


def safety_decision(
    origin_ids: np.ndarray,
    base_losses: np.ndarray,
    candidate_losses: np.ndarray,
) -> tuple[bool, float, int, float]:
    if not len(origin_ids):
        return False, 0.0, 0, 0.0
    order = np.argsort(origin_ids)
    base_losses = base_losses[order]
    candidate_losses = candidate_losses[order]
    gain = 100.0 * (
        float(np.mean(base_losses)) - float(np.mean(candidate_losses))
    ) / max(float(np.mean(base_losses)), 1e-12)
    blocks: list[float] = []
    for indices in np.array_split(np.arange(len(base_losses)), min(4, len(base_losses))):
        raw = float(np.mean(base_losses[indices]))
        method = float(np.mean(candidate_losses[indices]))
        blocks.append(100.0 * (raw - method) / max(raw, 1e-12))
    positive = int(sum(value > 0.0 for value in blocks))
    worst = float(min(blocks)) if blocks else 0.0
    accepted = gain > 0.0 and positive >= 3 and worst >= -0.10
    return accepted, float(gain), positive, worst
