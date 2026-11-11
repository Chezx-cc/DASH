from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
import sys

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_cf_msrc_stream_from_checkpoints as cf
import run_cf_residual_expert as core
import run_internal_offline_candidates as base
import run_internal_strict_candidates_parallel as strict
import run_e06_new_backbones as nb


VARIANTS: list[dict[str, Any]] = [
    {"variant_id": "lm01_long168_336", "periods": [168, 336], "shifts": [0], "cycles": 1, "aggregate": "mean", "scale": "none", "bins": 4, "ridge": 0.03, "clip_q": 0.95},
    {"variant_id": "lm02_long96_168_336", "periods": [96, 168, 336], "shifts": [0], "cycles": 1, "aggregate": "mean", "scale": "none", "bins": 4, "ridge": 0.03, "clip_q": 0.95},
    {"variant_id": "lm03_long96_168_336_672", "periods": [96, 168, 336, 672], "shifts": [0], "cycles": 1, "aggregate": "mean", "scale": "none", "bins": 4, "ridge": 0.05, "clip_q": 0.95},
    {"variant_id": "lm04_phase2", "periods": [96, 168, 336], "shifts": [-2, 0, 2], "cycles": 1, "aggregate": "mean", "scale": "none", "bins": 4, "ridge": 0.10, "clip_q": 0.95},
    {"variant_id": "lm05_phase4_long", "periods": [168, 336], "shifts": [-4, -2, 0, 2, 4], "cycles": 1, "aggregate": "mean", "scale": "none", "bins": 4, "ridge": 0.10, "clip_q": 0.95},
    {"variant_id": "lm06_median3", "periods": [96, 168, 336], "shifts": [0], "cycles": 3, "aggregate": "median", "scale": "none", "bins": 4, "ridge": 0.05, "clip_q": 0.95},
    {"variant_id": "lm07_ew3", "periods": [96, 168, 336], "shifts": [0], "cycles": 3, "aggregate": "ew", "decay": 0.65, "scale": "none", "bins": 4, "ridge": 0.05, "clip_q": 0.95},
    {"variant_id": "lm08_level_median3", "periods": [96, 168, 336], "shifts": [0], "cycles": 3, "aggregate": "median", "scale": "level", "bins": 4, "ridge": 0.08, "clip_q": 0.95},
    {"variant_id": "lm09_zscore_mean2", "periods": [96, 168, 336], "shifts": [0], "cycles": 2, "aggregate": "mean", "scale": "zscore", "bins": 4, "ridge": 0.10, "clip_q": 0.95},
    {"variant_id": "lm10_channel_shrink", "periods": [96, 168, 336], "shifts": [0], "cycles": 2, "aggregate": "mean", "scale": "none", "bins": 4, "ridge": 0.15, "clip_q": 0.95, "channel_shrink": 0.90},
]
VARIANT_BY_ID = {str(item["variant_id"]): item for item in VARIANTS}
GAMMA_GRID = np.asarray([0.0, 0.10, 0.25, 0.50, 0.75, 1.0], dtype=np.float64)


def check_sliding_inputs(inputs: np.ndarray, label: str) -> None:
    if len(inputs) < 2:
        return
    sample = np.linspace(0, len(inputs) - 2, min(32, len(inputs) - 1), dtype=np.int64)
    mismatch = float(np.max(np.abs(inputs[sample + 1, :-1] - inputs[sample, 1:])))
    if mismatch > 1e-5:
        raise ValueError(f"{label} origins are not consecutive sliding windows: max mismatch={mismatch}")


def observed_timeline(inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    check_sliding_inputs(inputs, "observed")
    timeline = np.concatenate([inputs[0], inputs[1:, -1]], axis=0).astype(np.float32, copy=False)
    current = inputs.shape[1] - 1 + np.arange(len(inputs), dtype=np.int64)
    return timeline, current


def validation_known_timeline(inputs: np.ndarray, trues: np.ndarray) -> np.ndarray:
    check_sliding_inputs(inputs, "validation inputs")
    if len(trues) > 1:
        sample = np.linspace(0, len(trues) - 2, min(32, len(trues) - 1), dtype=np.int64)
        mismatch = float(np.max(np.abs(trues[sample + 1, :-1] - trues[sample, 1:])))
        if mismatch > 1e-5:
            raise ValueError(f"validation targets are not consecutive: max mismatch={mismatch}")
    bridge = float(np.max(np.abs(inputs[1:, -1] - trues[:-1, 0]))) if len(inputs) > 1 else 0.0
    if bridge > 1e-5:
        raise ValueError(f"validation input/target bridge mismatch={bridge}")
    return np.concatenate([inputs[0], trues[0], trues[1:, -1]], axis=0).astype(np.float32, copy=False)


def test_timeline(
    val_inputs: np.ndarray,
    val_trues: np.ndarray,
    test_inputs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    check_sliding_inputs(test_inputs, "test")
    known = validation_known_timeline(val_inputs, val_trues)
    first = test_inputs[0]
    lookback = len(first)
    starts: list[int] = []
    if len(known) >= lookback and np.allclose(known[-lookback:], first, rtol=0.0, atol=1e-5):
        starts = [len(known) - lookback]
    else:
        lower = max(0, len(known) - 4096)
        for start in range(lower, len(known) - lookback + 1):
            if np.allclose(known[start : start + lookback], first, rtol=0.0, atol=1e-5):
                starts.append(start)
    if starts:
        start = starts[-1]
        prefix = known[:start]
        timeline = np.concatenate([prefix, first, test_inputs[1:, -1]], axis=0).astype(np.float32, copy=False)
        current = len(prefix) + lookback - 1 + np.arange(len(test_inputs), dtype=np.int64)
        return timeline, current, {
            "validation_history_stitched": True,
            "validation_prefix_length": int(len(prefix)),
            "stitch_start": int(start),
            "stitch_matches": int(len(starts)),
        }
    timeline, current = observed_timeline(test_inputs)
    return timeline, current, {
        "validation_history_stitched": False,
        "validation_prefix_length": 0,
        "stitch_start": None,
        "stitch_matches": 0,
    }


def feature_specs(params: dict[str, Any]) -> list[tuple[int, int]]:
    return [(int(period), int(shift)) for period in params["periods"] for shift in params["shifts"]]


def causal_source_summary(
    current: np.ndarray,
    horizon: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Summarize every possible anchor read without materializing origin x horizon grids."""
    current = np.asarray(current, dtype=np.int64)
    if current.ndim != 1 or not len(current) or np.any(np.diff(current) < 0):
        raise ValueError("current indices must be a non-empty sorted vector")

    candidate_reads = 0
    valid_reads = 0
    unavailable_reads = 0
    future_reads = 0
    max_used_offset: int | None = None
    min_used_source: int | None = None
    feature_rows: list[dict[str, Any]] = []
    for period, shift in feature_specs(params):
        phase = (np.arange(horizon, dtype=np.int64) + shift) % period
        for cycle in range(1, int(params["cycles"]) + 1):
            offsets = -cycle * period + 1 + phase
            if int(offsets.max()) > 0:
                raise AssertionError("anchor source formula permits a future observation")
            feature_valid = 0
            feature_min_source: int | None = None
            for offset in offsets:
                first = int(np.searchsorted(current, -int(offset), side="left"))
                count = len(current) - first
                feature_valid += count
                if count:
                    source = int(current[first] + offset)
                    feature_min_source = source if feature_min_source is None else min(feature_min_source, source)
            feature_candidates = int(len(current) * horizon)
            feature_unavailable = feature_candidates - feature_valid
            candidate_reads += feature_candidates
            valid_reads += feature_valid
            unavailable_reads += feature_unavailable
            max_used_offset = int(offsets.max()) if max_used_offset is None else max(max_used_offset, int(offsets.max()))
            if feature_min_source is not None:
                min_used_source = (
                    feature_min_source if min_used_source is None else min(min_used_source, feature_min_source)
                )
            feature_rows.append(
                {
                    "period": period,
                    "shift": shift,
                    "cycle": cycle,
                    "candidate_reads": feature_candidates,
                    "valid_reads": int(feature_valid),
                    "unavailable_raw_fallback_reads": int(feature_unavailable),
                    "max_source_minus_current": int(offsets.max()),
                    "min_source_minus_current": int(offsets.min()),
                }
            )

    return {
        "candidate_reads": int(candidate_reads),
        "valid_observation_reads": int(valid_reads),
        "unavailable_raw_fallback_reads": int(unavailable_reads),
        "future_observation_reads": int(future_reads),
        "max_used_source_minus_current": max_used_offset,
        "min_used_source_index": min_used_source,
        "causal_source_constraint_passed": future_reads == 0 and (max_used_offset is None or max_used_offset <= 0),
        "feature_cycle_summary": feature_rows,
    }


def long_anchor(
    timeline: np.ndarray,
    current: np.ndarray,
    histories: np.ndarray,
    preds: np.ndarray,
    period: int,
    shift: int,
    cycles: int,
    aggregate: str,
    decay: float,
    scale_mode: str,
) -> np.ndarray:
    horizon = preds.shape[1]
    phase = (np.arange(horizon, dtype=np.int64) + shift) % period
    values: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    for cycle in range(1, cycles + 1):
        source = current[:, None] - cycle * period + 1 + phase[None, :]
        valid = (source >= 0) & (source <= current[:, None])
        gathered = timeline[np.clip(source, 0, len(timeline) - 1)]
        values.append(gathered.astype(np.float32, copy=False))
        valid_masks.append(valid[:, :, None])

    if aggregate == "median":
        # Older cycles are available only when every newer cycle is available,
        # so the valid masks are nested. The closed form avoids np.nanmedian's
        # expensive temporary stack on Traffic/Electricity.
        count = np.sum(np.stack(valid_masks, axis=0), axis=0)
        anchor = np.where(valid_masks[0], values[0], preds).astype(np.float32, copy=False)
        if len(values) >= 2:
            mean_two = 0.5 * (values[0] + values[1])
            anchor = np.where(count == 2, mean_two, anchor)
        if len(values) >= 3:
            minimum = np.minimum(np.minimum(values[0], values[1]), values[2])
            maximum = np.maximum(np.maximum(values[0], values[1]), values[2])
            median_three = values[0] + values[1] + values[2] - minimum - maximum
            anchor = np.where(count >= 3, median_three, anchor)
        any_valid = count > 0
    else:
        numerator = np.zeros_like(preds, dtype=np.float32)
        denominator = np.zeros((len(preds), horizon, 1), dtype=np.float32)
        for index, (value, mask) in enumerate(zip(values, valid_masks)):
            weight = float(decay**index) if aggregate == "ew" else 1.0
            numerator += weight * np.where(mask, value, 0.0)
            denominator += weight * mask.astype(np.float32)
        any_valid = denominator > 0
        anchor = numerator / np.maximum(denominator, 1e-8)
    anchor = np.where(any_valid, anchor, preds).astype(np.float32, copy=False)

    if scale_mode != "none":
        recent = histories[:, -min(96, histories.shape[1]) :, :]
        recent_mean = np.mean(recent, axis=1, keepdims=True, dtype=np.float32)
        anchor_mean = np.mean(anchor, axis=1, keepdims=True, dtype=np.float32)
        if scale_mode == "level":
            anchor = anchor - anchor_mean + recent_mean
        elif scale_mode == "zscore":
            recent_std = np.maximum(np.std(recent, axis=1, keepdims=True), 0.05)
            anchor_std = np.maximum(np.std(anchor, axis=1, keepdims=True), 0.05)
            ratio = np.clip(recent_std / anchor_std, 0.5, 2.0)
            anchor = recent_mean + ratio * (anchor - anchor_mean)
        else:
            raise ValueError(scale_mode)
        anchor = np.where(any_valid, anchor, preds).astype(np.float32, copy=False)
    return anchor


def feature_correction(
    timeline: np.ndarray,
    current: np.ndarray,
    inputs: np.ndarray,
    preds: np.ndarray,
    indices: np.ndarray,
    spec: tuple[int, int],
    params: dict[str, Any],
    channel_idx: np.ndarray | None = None,
) -> np.ndarray:
    selected_preds = preds[indices]
    selected_inputs = inputs[indices]
    if channel_idx is not None:
        selected_preds = selected_preds[:, :, channel_idx]
        selected_inputs = selected_inputs[:, :, channel_idx]
        selected_timeline = timeline[:, channel_idx]
    else:
        selected_timeline = timeline
    anchor = long_anchor(
        selected_timeline,
        current[indices],
        selected_inputs,
        selected_preds,
        period=spec[0],
        shift=spec[1],
        cycles=int(params["cycles"]),
        aggregate=str(params["aggregate"]),
        decay=float(params.get("decay", 1.0)),
        scale_mode=str(params["scale"]),
    )
    return (anchor - selected_preds).astype(np.float32, copy=False)


def solve_weights(x: np.ndarray, y: np.ndarray, ridge: float, cap: float = 1.25) -> np.ndarray:
    width = x.shape[1]
    if len(x) == 0:
        return np.zeros(width, dtype=np.float32)
    xd = x.astype(np.float64, copy=False)
    yd = y.astype(np.float64, copy=False)
    xtx = xd.T @ xd
    scale = max(float(np.trace(xtx)) / max(width, 1), 1e-9)
    system = xtx + ridge * scale * np.eye(width)
    try:
        weights = np.linalg.solve(system, xd.T @ yd)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(system, xd.T @ yd, rcond=None)[0]
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total > cap:
        weights *= cap / total
    return weights.astype(np.float32)


def sampled_indices(start: int, stop: int, count: int) -> np.ndarray:
    if stop <= start:
        return np.zeros(0, dtype=np.int64)
    return np.linspace(start, stop - 1, min(count, stop - start), dtype=np.int64)


def fit_weights(
    timeline: np.ndarray,
    current: np.ndarray,
    inputs: np.ndarray,
    preds: np.ndarray,
    trues: np.ndarray,
    fit_stop: int,
    bins: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    specs = feature_specs(params)
    max_period = max(spec[0] for spec in specs)
    first_available = max(0, max_period - inputs.shape[1])
    origin_idx = sampled_indices(first_available, fit_stop, 96)
    if len(origin_idx) < 8:
        origin_idx = sampled_indices(0, fit_stop, 96)
    channels = preds.shape[2]
    global_channels = sampled_indices(0, channels, 128)
    num_bins = int(bins.max()) + 1
    global_weights = np.zeros((num_bins, len(specs)), dtype=np.float32)
    for b in range(num_bins):
        horizon_idx = np.where(bins == b)[0]
        horizon_idx = horizon_idx[np.linspace(0, len(horizon_idx) - 1, min(48, len(horizon_idx)), dtype=np.int64)]
        features = []
        for spec in specs:
            corr = feature_correction(
                timeline, current, inputs, preds, origin_idx, spec, params, global_channels
            )
            features.append(corr[:, horizon_idx, :])
        x = np.stack(features, axis=-1).reshape(-1, len(specs))
        y = (trues[np.ix_(origin_idx, horizon_idx, global_channels)] - preds[np.ix_(origin_idx, horizon_idx, global_channels)]).reshape(-1)
        target_clip = float(np.quantile(np.abs(y), 0.95))
        global_weights[b] = solve_weights(x, np.clip(y, -target_clip, target_clip), float(params["ridge"]))

    if "channel_shrink" not in params:
        return global_weights, {
            "weight_mode": "global_horizon_bin",
            "fit_origins": int(len(origin_idx)),
            "fit_first_origin": int(origin_idx[0]) if len(origin_idx) else None,
        }

    shrink = float(params["channel_shrink"])
    weights = np.broadcast_to(global_weights[:, None, :], (num_bins, channels, len(specs))).copy()
    local_origins = origin_idx[np.linspace(0, len(origin_idx) - 1, min(64, len(origin_idx)), dtype=np.int64)]
    all_channels = np.arange(channels, dtype=np.int64)
    for b in range(num_bins):
        horizon_idx = np.where(bins == b)[0]
        horizon_idx = horizon_idx[np.linspace(0, len(horizon_idx) - 1, min(32, len(horizon_idx)), dtype=np.int64)]
        feature_parts = [
            feature_correction(timeline, current, inputs, preds, local_origins, spec, params)
            [:, horizon_idx, :]
            for spec in specs
        ]
        x = np.stack(feature_parts, axis=-1)
        y = trues[np.ix_(local_origins, horizon_idx, all_channels)] - preds[np.ix_(local_origins, horizon_idx, all_channels)]
        for channel in range(channels):
            local = solve_weights(
                x[:, :, channel, :].reshape(-1, len(specs)),
                y[:, :, channel].reshape(-1),
                float(params["ridge"]),
            )
            weights[b, channel] = shrink * global_weights[b] + (1.0 - shrink) * local
    return weights.astype(np.float32), {
        "weight_mode": "channel_shrunk_horizon_bin",
        "channel_shrink": shrink,
        "fit_origins": int(len(origin_idx)),
        "fit_first_origin": int(origin_idx[0]) if len(origin_idx) else None,
    }


def correction_chunk(
    timeline: np.ndarray,
    current: np.ndarray,
    inputs: np.ndarray,
    preds: np.ndarray,
    indices: np.ndarray,
    bins: np.ndarray,
    params: dict[str, Any],
    weights: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(preds[indices], dtype=np.float32)
    for feature, spec in enumerate(feature_specs(params)):
        corr = feature_correction(timeline, current, inputs, preds, indices, spec, params)
        if weights.ndim == 2:
            coefficient = weights[bins, feature][None, :, None]
        else:
            coefficient = weights[bins, :, feature][None, :, :]
        result += coefficient * corr
    return result


def fit_clip(
    timeline: np.ndarray,
    current: np.ndarray,
    inputs: np.ndarray,
    preds: np.ndarray,
    fit_stop: int,
    bins: np.ndarray,
    params: dict[str, Any],
    weights: np.ndarray,
) -> float:
    indices = sampled_indices(0, fit_stop, 64)
    corr = correction_chunk(timeline, current, inputs, preds, indices, bins, params, weights)
    return cf.safe_quantile_abs(corr, float(params["clip_q"]))


def chunk_size(preds: np.ndarray) -> int:
    per_origin = max(1, int(np.prod(preds.shape[1:])))
    return max(1, min(32, 8_000_000 // per_origin))


def tune_gamma(
    timeline: np.ndarray,
    current: np.ndarray,
    inputs: np.ndarray,
    preds: np.ndarray,
    trues: np.ndarray,
    indices: np.ndarray,
    bins: np.ndarray,
    params: dict[str, Any],
    weights: np.ndarray,
    clip: float,
) -> np.ndarray:
    num_bins = int(bins.max()) + 1
    sums = np.zeros((num_bins, 3), dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.int64)
    step = chunk_size(preds)
    for begin in range(0, len(indices), step):
        selected = indices[begin : begin + step]
        corr = np.clip(
            correction_chunk(timeline, current, inputs, preds, selected, bins, params, weights),
            -clip,
            clip,
        ).astype(np.float64, copy=False)
        error = (preds[selected] - trues[selected]).astype(np.float64, copy=False)
        for b in range(num_bins):
            h = np.where(bins == b)[0]
            e, c = error[:, h, :], corr[:, h, :]
            sums[b] += [np.sum(e * e), np.sum(e * c), np.sum(c * c)]
            counts[b] += e.size
    gamma = np.zeros(num_bins, dtype=np.float32)
    for b in range(num_bins):
        candidate = sums[b, 0] + 2.0 * GAMMA_GRID * sums[b, 1] + GAMMA_GRID**2 * sums[b, 2]
        gamma[b] = float(GAMMA_GRID[int(np.argmin(candidate))])
    return gamma


def evaluate_indices(
    timeline: np.ndarray,
    current: np.ndarray,
    inputs: np.ndarray,
    preds: np.ndarray,
    trues: np.ndarray,
    indices: np.ndarray,
    bins: np.ndarray,
    params: dict[str, Any],
    weights: np.ndarray,
    gamma: np.ndarray,
    clip: float,
    collect_origins: bool = False,
) -> dict[str, Any]:
    num_bins = int(bins.max()) + 1
    raw_sq = np.zeros(num_bins, dtype=np.float64)
    raw_abs = np.zeros(num_bins, dtype=np.float64)
    method_sq = np.zeros(num_bins, dtype=np.float64)
    method_abs = np.zeros(num_bins, dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.int64)
    origin_raw = np.zeros((len(indices), num_bins), dtype=np.float64) if collect_origins else None
    origin_method = np.zeros((len(indices), num_bins), dtype=np.float64) if collect_origins else None
    corr_sum, corr_max, corr_count = 0.0, 0.0, 0
    step = chunk_size(preds)
    for begin in range(0, len(indices), step):
        selected = indices[begin : begin + step]
        corr = np.clip(
            correction_chunk(timeline, current, inputs, preds, selected, bins, params, weights),
            -clip,
            clip,
        )
        corr *= gamma[bins][None, :, None]
        raw_error = preds[selected] - trues[selected]
        method_error = raw_error + corr
        abs_corr = np.abs(corr)
        corr_sum += float(np.sum(abs_corr, dtype=np.float64))
        corr_max = max(corr_max, float(abs_corr.max()) if abs_corr.size else 0.0)
        corr_count += int(abs_corr.size)
        for b in range(num_bins):
            h = np.where(bins == b)[0]
            re, me = raw_error[:, h, :], method_error[:, h, :]
            raw_sq[b] += float(np.sum(np.square(re, dtype=np.float64)))
            raw_abs[b] += float(np.sum(np.abs(re), dtype=np.float64))
            method_sq[b] += float(np.sum(np.square(me, dtype=np.float64)))
            method_abs[b] += float(np.sum(np.abs(me), dtype=np.float64))
            counts[b] += int(re.size)
            if collect_origins:
                assert origin_raw is not None and origin_method is not None
                origin_raw[begin : begin + len(selected), b] = np.mean(np.square(re, dtype=np.float64), axis=(1, 2))
                origin_method[begin : begin + len(selected), b] = np.mean(np.square(me, dtype=np.float64), axis=(1, 2))
    result: dict[str, Any] = {
        "raw_mse": float(raw_sq.sum() / counts.sum()),
        "raw_mae": float(raw_abs.sum() / counts.sum()),
        "method_mse": float(method_sq.sum() / counts.sum()),
        "method_mae": float(method_abs.sum() / counts.sum()),
        "bin_raw_mse": (raw_sq / np.maximum(counts, 1)).tolist(),
        "bin_method_mse": (method_sq / np.maximum(counts, 1)).tolist(),
        "bin_raw_mae": (raw_abs / np.maximum(counts, 1)).tolist(),
        "bin_method_mae": (method_abs / np.maximum(counts, 1)).tolist(),
        "correction_norm_mean": corr_sum / max(corr_count, 1),
        "correction_norm_max": corr_max,
    }
    if collect_origins:
        result["origin_raw_by_bin"] = origin_raw
        result["origin_method_by_bin"] = origin_method
    return result


def evaluate_pair_indices(
    timeline: np.ndarray,
    current: np.ndarray,
    inputs: np.ndarray,
    preds: np.ndarray,
    trues: np.ndarray,
    indices: np.ndarray,
    bins: np.ndarray,
    params: dict[str, Any],
    weights: np.ndarray,
    gamma_by_name: dict[str, np.ndarray],
    clip: float,
) -> dict[str, dict[str, Any]]:
    """Evaluate multiple frozen gamma vectors while constructing each correction once."""
    num_bins = int(bins.max()) + 1
    names = list(gamma_by_name)
    raw_sq = np.zeros(num_bins, dtype=np.float64)
    raw_abs = np.zeros(num_bins, dtype=np.float64)
    method_sq = {name: np.zeros(num_bins, dtype=np.float64) for name in names}
    method_abs = {name: np.zeros(num_bins, dtype=np.float64) for name in names}
    counts = np.zeros(num_bins, dtype=np.int64)
    corr_sum = {name: 0.0 for name in names}
    corr_max = {name: 0.0 for name in names}
    corr_count = 0
    step = chunk_size(preds)
    for begin in range(0, len(indices), step):
        selected = indices[begin : begin + step]
        base_corr = np.clip(
            correction_chunk(timeline, current, inputs, preds, selected, bins, params, weights),
            -clip,
            clip,
        )
        raw_error = preds[selected] - trues[selected]
        scaled: dict[str, np.ndarray] = {}
        for name, gamma in gamma_by_name.items():
            corr = base_corr * gamma[bins][None, :, None]
            scaled[name] = corr
            absolute = np.abs(corr)
            corr_sum[name] += float(np.sum(absolute, dtype=np.float64))
            corr_max[name] = max(corr_max[name], float(absolute.max()) if absolute.size else 0.0)
        corr_count += int(base_corr.size)
        for b in range(num_bins):
            h = np.where(bins == b)[0]
            re = raw_error[:, h, :]
            raw_sq[b] += float(np.sum(np.square(re, dtype=np.float64)))
            raw_abs[b] += float(np.sum(np.abs(re), dtype=np.float64))
            counts[b] += int(re.size)
            for name in names:
                me = re + scaled[name][:, h, :]
                method_sq[name][b] += float(np.sum(np.square(me, dtype=np.float64)))
                method_abs[name][b] += float(np.sum(np.abs(me), dtype=np.float64))

    results: dict[str, dict[str, Any]] = {}
    for name in names:
        results[name] = {
            "raw_mse": float(raw_sq.sum() / counts.sum()),
            "raw_mae": float(raw_abs.sum() / counts.sum()),
            "method_mse": float(method_sq[name].sum() / counts.sum()),
            "method_mae": float(method_abs[name].sum() / counts.sum()),
            "bin_raw_mse": (raw_sq / np.maximum(counts, 1)).tolist(),
            "bin_method_mse": (method_sq[name] / np.maximum(counts, 1)).tolist(),
            "bin_raw_mae": (raw_abs / np.maximum(counts, 1)).tolist(),
            "bin_method_mae": (method_abs[name] / np.maximum(counts, 1)).tolist(),
            "correction_norm_mean": corr_sum[name] / max(corr_count, 1),
            "correction_norm_max": corr_max[name],
        }
    return results


def gate_on_route(route_eval: dict[str, Any], gamma: np.ndarray) -> tuple[np.ndarray, list[float], list[int], list[float]]:
    gated = gamma.copy()
    raw = np.asarray(route_eval["origin_raw_by_bin"])
    method = np.asarray(route_eval["origin_method_by_bin"])
    gains: list[float] = []
    positives: list[int] = []
    worst: list[float] = []
    for b in range(len(gamma)):
        gain = strict.route_gain(float(np.mean(raw[:, b])), float(np.mean(method[:, b])))
        blocks = []
        for index in np.array_split(np.arange(len(raw)), min(4, len(raw))):
            if len(index):
                blocks.append(strict.route_gain(float(np.mean(raw[index, b])), float(np.mean(method[index, b]))))
        positive = int(sum(item > 0 for item in blocks))
        minimum = float(min(blocks)) if blocks else 0.0
        gains.append(float(gain))
        positives.append(positive)
        worst.append(minimum)
        if gain <= 0.0 or positive < 3 or minimum < -0.10:
            gated[b] = 0.0
    return gated, gains, positives, worst


def evaluate(
    config: nb.BackboneConfig,
    params: dict[str, Any],
    val_arrays: dict[str, np.ndarray],
    test_arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    val = {key: np.asarray(val_arrays[key], dtype=np.float32) for key in ["preds", "trues", "inputs"]}
    test = {key: np.asarray(test_arrays[key], dtype=np.float32) for key in ["preds", "trues", "inputs"]}
    n = len(val["preds"])
    c1, c2 = max(1, n // 3), min(max(n // 3 + 1, 2 * n // 3), n - 1)
    fit_idx = np.arange(0, c1, dtype=np.int64)
    tune_idx = np.arange(c1, c2, dtype=np.int64)
    route_idx = np.arange(c2, n, dtype=np.int64)
    val_timeline, val_current = observed_timeline(val["inputs"])
    test_values, test_current, stitch_meta = test_timeline(val["inputs"], val["trues"], test["inputs"])
    bins = cf.horizon_bin_ids(int(config.pred_len), int(params["bins"]))
    weights, weight_meta = fit_weights(
        val_timeline,
        val_current,
        val["inputs"],
        val["preds"],
        val["trues"],
        c1,
        bins,
        params,
    )
    clip = fit_clip(val_timeline, val_current, val["inputs"], val["preds"], c1, bins, params, weights)
    gamma = tune_gamma(
        val_timeline,
        val_current,
        val["inputs"],
        val["preds"],
        val["trues"],
        tune_idx,
        bins,
        params,
        weights,
        clip,
    )
    route_expert = evaluate_indices(
        val_timeline, val_current, val["inputs"], val["preds"], val["trues"], route_idx,
        bins, params, weights, gamma, clip, collect_origins=True,
    )
    gated, gains, positives, worst = gate_on_route(route_expert, gamma)
    route_safe = evaluate_indices(
        val_timeline, val_current, val["inputs"], val["preds"], val["trues"], route_idx,
        bins, params, weights, gated, clip,
    )
    test_idx = np.arange(len(test["preds"]), dtype=np.int64)
    test_pair = evaluate_pair_indices(
        test_values, test_current, test["inputs"], test["preds"], test["trues"], test_idx,
        bins, params, weights, {"expert": gamma, "safe": gated}, clip,
    )
    test_expert = test_pair["expert"]
    test_safe = test_pair["safe"]
    raw_mse, raw_mae = base.metrics_chunked(test["preds"], test["trues"])
    source_audit = causal_source_summary(test_current, int(config.pred_len), params)
    meta = {
        "variant": params["variant_id"],
        "periods": params["periods"],
        "phase_shifts": params["shifts"],
        "cycles": params["cycles"],
        "aggregate": params["aggregate"],
        "scale": params["scale"],
        "feature_specs": feature_specs(params),
        "weights_shape": list(weights.shape),
        "weights": np.round(weights, 5).tolist() if weights.size <= 512 else "omitted_large_channel_weights",
        "test_correction_reads_test_truth": False,
        "test_prefix_source": "validation_labels_only_when_exact_input_overlap_is_verified",
        "test_lookback": int(test["inputs"].shape[1]),
        "test_origin_count": int(len(test_current)),
        "test_current_first": int(test_current[0]),
        "test_current_last": int(test_current[-1]),
        "causal_source_audit": source_audit,
        **weight_meta,
        **stitch_meta,
    }
    name = str(params["variant_id"])
    return [
        core.make_row(config, f"{name}__expert", raw_mse, raw_mae, test_expert, route_expert, gamma, clip, meta, gains, positives, worst),
        core.make_row(config, f"{name}__safe", raw_mse, raw_mae, test_safe, route_safe, gated, clip, meta, gains, positives, worst),
    ]


def output_paths(root: Path, variant_id: str, key: str) -> tuple[Path, Path]:
    safe = key.replace("|", "__")
    return root / variant_id / "queue_status" / f"{safe}.json", root / variant_id / "rows" / f"{safe}.csv"


def run_one(payload: dict[str, Any]) -> dict[str, Any]:
    config = nb.BackboneConfig(**payload["config"])
    key = core.config_key(config)
    marker, row_path = output_paths(Path(payload["out_root"]), payload["variant_id"], key)
    if row_path.exists() and not payload["force"]:
        return {"config_key": key, "status": "skipped_complete", "failure": None}
    core.atomic_json(marker, {"config_key": key, "variant_id": payload["variant_id"], "status": "running"})
    started = time.time()
    try:
        val, test = nb.collect_val_test_arrays_in_memory(config)
        measured = base.metrics_chunked(test["preds"], test["trues"])
        expected = payload["raw_expected"]
        if not (
            np.isclose(measured[0], expected["raw_mse"], atol=payload["raw_atol"], rtol=payload["raw_rtol"])
            and np.isclose(measured[1], expected["raw_mae"], atol=payload["raw_atol"], rtol=payload["raw_rtol"])
        ):
            raise ValueError(f"Raw source mismatch: {measured} vs {expected}")
        rows = core.align_raw(evaluate(config, VARIANT_BY_ID[payload["variant_id"]], val, test), expected, measured)
        core.atomic_csv(row_path, pd.DataFrame(rows))
        core.atomic_json(marker, {"config_key": key, "variant_id": payload["variant_id"], "status": "completed", "elapsed_sec": round(time.time() - started, 3)})
        return {"config_key": key, "status": "completed", "failure": None}
    except Exception as exc:
        failure = {
            "config_key": key,
            "variant_id": payload["variant_id"],
            "status": "failed",
            "fallback_type": "technical_failure",
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc),
            "traceback": traceback.format_exc(),
        }
        core.atomic_json(marker, failure)
        return {"config_key": key, "status": "failed", "failure": failure}
    finally:
        gc.collect()


def self_test() -> None:
    rng = np.random.default_rng(2027)
    n, lookback, horizon, channels = 900, 96, 48, 4
    total = lookback + n + horizon
    time_axis = np.arange(total, dtype=np.float32)
    series = np.stack(
        [np.sin(2 * np.pi * time_axis / period) + 0.03 * rng.normal(size=total) for period in [24, 48, 96, 168]],
        axis=1,
    ).astype(np.float32)
    inputs = np.stack([series[i : i + lookback] for i in range(n)])
    trues = np.stack([series[i + lookback : i + lookback + horizon] for i in range(n)])
    preds = np.repeat(inputs[:, -1:, :], horizon, axis=1)
    val = {"preds": preds, "trues": trues, "inputs": inputs}
    test = {key: value.copy() for key, value in val.items()}
    config = nb.BackboneConfig(model="DLinear", dataset="ETTm2", pred_len=horizon, seed=2)
    for params in VARIANTS:
        rows = evaluate(config, params, val, test)
        if len(rows) != 2 or not all(np.isfinite(row["method_mse"]) for row in rows):
            raise AssertionError(params["variant_id"])
        if any(row["uses_external_outputs"] or row["uses_test_for_selection"] for row in rows):
            raise AssertionError("protocol flags")
        timeline, current = observed_timeline(inputs)
        bins = cf.horizon_bin_ids(horizon, int(params["bins"]))
        weights = np.full((int(params["bins"]), len(feature_specs(params))), 0.05, dtype=np.float32)
        expert_gamma = np.asarray([0.0, 0.25, 0.50, 0.75], dtype=np.float32)
        safe_gamma = np.asarray([0.0, 0.0, 0.25, 0.50], dtype=np.float32)
        indices = np.arange(min(16, n), dtype=np.int64)
        separate = {
            "expert": evaluate_indices(
                timeline, current, inputs, preds, trues, indices, bins, params, weights, expert_gamma, 1.0
            ),
            "safe": evaluate_indices(
                timeline, current, inputs, preds, trues, indices, bins, params, weights, safe_gamma, 1.0
            ),
        }
        paired = evaluate_pair_indices(
            timeline,
            current,
            inputs,
            preds,
            trues,
            indices,
            bins,
            params,
            weights,
            {"expert": expert_gamma, "safe": safe_gamma},
            1.0,
        )
        for operating_point in separate:
            for key in separate[operating_point]:
                if not np.allclose(
                    separate[operating_point][key], paired[operating_point][key], rtol=1e-12, atol=1e-12
                ):
                    raise AssertionError(
                        f"paired evaluation mismatch: {params['variant_id']} / {operating_point} / {key}"
                    )
    print(f"self-test passed for {len(VARIANTS)} long-memory variants")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-id", choices=sorted(VARIANT_BY_ID))
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--raw-metrics-csv", type=Path, default=Path("outputs/raw_metrics_cache/raw_metrics.csv"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/internal_only_search/cf_long_memory_anchor_seed2_v1"))
    parser.add_argument("--models", nargs="*", default=cf.BACKBONES)
    parser.add_argument("--datasets", nargs="*", default=["ETTh2", "ETTm2", "Electricity", "Traffic"])
    parser.add_argument("--pred-lens", nargs="*", type=int, default=[96, 720])
    parser.add_argument("--seeds", nargs="*", type=int, default=[2])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", default=["0"])
    parser.add_argument("--raw-match-atol", type=float, default=1e-6)
    parser.add_argument("--raw-match-rtol", type=float, default=1e-6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.variant_id or not args.checkpoint_root:
        raise ValueError("--variant-id and --checkpoint-root are required")
    raw = core.load_raw(args.raw_metrics_csv)
    datasets = [core.canonical_dataset(item) for item in args.datasets]
    configs = [
        nb.BackboneConfig(model=model, dataset=dataset, pred_len=pred_len, seed=seed)
        for model in args.models
        for dataset in datasets
        for pred_len in args.pred_lens
        for seed in args.seeds
    ]
    if args.limit is not None:
        configs = configs[: args.limit]
    payloads = []
    for config in configs:
        raw_key = (config.model, core.display_dataset(config.dataset), int(config.pred_len), int(config.seed))
        if raw_key not in raw:
            raise ValueError(f"Raw metrics missing {raw_key}")
        payloads.append(
            {
                "config": asdict(config),
                "variant_id": args.variant_id,
                "out_root": str(args.out_root),
                "raw_expected": raw[raw_key],
                "raw_atol": args.raw_match_atol,
                "raw_rtol": args.raw_match_rtol,
                "force": args.force,
            }
        )
    root = args.out_root / args.variant_id
    status = root / "runner_status.json"
    started = time.time()
    core.atomic_json(status, {"status": "running", "variant_id": args.variant_id, "expected_configs": len(configs)})
    failures: list[dict[str, Any]] = []
    with mp.Pool(
        max(1, min(args.workers, len(payloads))),
        initializer=core.worker_init,
        initargs=(args.checkpoint_root, list(args.gpu_ids)),
    ) as pool:
        for index, result in enumerate(pool.imap_unordered(run_one, payloads), 1):
            if result["failure"] is not None:
                failures.append(result["failure"])
            core.atomic_json(
                status,
                {
                    "status": "running",
                    "variant_id": args.variant_id,
                    "completed_or_seen": index,
                    "expected_configs": len(configs),
                    "failures_seen": len(failures),
                    "last_config": result["config_key"],
                    "last_status": result["status"],
                    "elapsed_sec": round(time.time() - started, 3),
                },
            )
            print(f"[{index}/{len(configs)}] {args.variant_id} {result['config_key']} {result['status']}", flush=True)
    files = sorted((root / "rows").glob("*.csv"))
    detail = pd.concat([pd.read_csv(path) for path in files], ignore_index=True) if files else pd.DataFrame()
    if not detail.empty:
        core.atomic_csv(root / "detail.csv", detail.sort_values(["config_key", "method"]))
        core.atomic_csv(root / "summary.csv", core.summarize(detail))
    core.atomic_csv(root / "failure_log.csv", pd.DataFrame(failures))
    complete = not failures and not detail.empty and detail.config_key.nunique() == len(configs)
    core.atomic_json(
        root / "manifest.json",
        {
            "script": str(Path(__file__).relative_to(ROOT)),
            "variant": VARIANT_BY_ID[args.variant_id],
            "scope": {"models": args.models, "datasets": datasets, "pred_lens": args.pred_lens, "seeds": args.seeds},
            "selection_protocol": "validation_fit_tune_route_long_memory_horizon_bin",
            "raw_explicit_fallback": True,
            "uses_external_outputs": False,
            "uses_test_for_selection": False,
            "test_correction_reads_test_truth": False,
            "validation_history_for_test_requires_exact_overlap": True,
            "technical_failure_falls_back_to_raw": False,
        },
    )
    core.atomic_json(
        status,
        {
            "status": "completed" if complete else "incomplete",
            "variant_id": args.variant_id,
            "expected_configs": len(configs),
            "completed_configs": int(detail.config_key.nunique()) if not detail.empty else 0,
            "failures_seen": len(failures),
            "elapsed_sec": round(time.time() - started, 3),
        },
    )
    if not detail.empty:
        print(pd.read_csv(root / "summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
