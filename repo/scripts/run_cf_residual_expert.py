from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
import sys

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_cf_msrc_stream_from_checkpoints as cf
import run_internal_offline_candidates as base
import run_internal_strict_candidates_parallel as strict
import run_e06_new_backbones as nb


METHOD_IDS = [
    "rb_hbin_channel",
    "rb_horizon_shared",
    "rb_svd_rank1",
    "affine_hbin_channel",
    "affine_shared",
    "context_ridge_hbin",
    "knn_regime_hbin",
    "causal_additive_ema",
    "mpa_plus_bias",
    "mpa_plus_affine",
]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def display_dataset(name: str) -> str:
    return cf.display_dataset(name)


def canonical_dataset(name: str) -> str:
    return {
        "Electricity": "electricity",
        "Traffic": "traffic",
        "Weather": "weather",
    }.get(name, name)


def config_key(config: nb.BackboneConfig) -> str:
    return cf.config_key(config)


def worker_init(checkpoint_root: str, gpu_ids: list[str]) -> None:
    nb.configure_roots(
        dirname="cf_residual_expert_memory",
        output_root="outputs/internal_only_search/_unused_residual_expert_exports",
        checkpoint_root=checkpoint_root,
    )
    identity = mp.current_process()._identity  # type: ignore[attr-defined]
    worker_idx = int(identity[0]) - 1 if identity else 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[worker_idx % len(gpu_ids)])


def sample_idx(size: int, limit: int) -> np.ndarray:
    if size <= limit:
        return np.arange(size, dtype=np.int64)
    return np.linspace(0, size - 1, limit, dtype=np.int64)


def robust_residual(preds: np.ndarray, trues: np.ndarray, q: float = 0.95) -> np.ndarray:
    resid = (trues - preds).astype(np.float32, copy=False)
    clip = cf.safe_quantile_abs(resid, q)
    return np.clip(resid, -clip, clip).astype(np.float32, copy=False)


def hbin_channel_center(resid: np.ndarray, bins: np.ndarray) -> np.ndarray:
    horizon, channels = resid.shape[1:]
    out = np.zeros((horizon, channels), dtype=np.float32)
    for b in range(int(bins.max()) + 1):
        h = np.where(bins == b)[0]
        if len(h):
            out[h] = np.mean(resid[:, h, :], axis=(0, 1), dtype=np.float64).astype(np.float32)
    return out


def horizon_shared_center(resid: np.ndarray) -> np.ndarray:
    center = np.mean(resid, axis=(0, 2), dtype=np.float64).astype(np.float32)
    return np.broadcast_to(center[:, None], resid.shape[1:]).copy()


def svd_center(resid: np.ndarray, rank: int) -> np.ndarray:
    center = np.mean(resid, axis=0, dtype=np.float64)
    global_mean = float(np.mean(center))
    centered = center - global_mean
    try:
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        r = max(1, min(rank, len(s)))
        return ((u[:, :r] * s[:r]) @ vt[:r, :] + global_mean).astype(np.float32)
    except np.linalg.LinAlgError:
        return center.astype(np.float32)


def fit_affine(
    preds: np.ndarray,
    trues: np.ndarray,
    bins: np.ndarray,
    per_channel: bool,
) -> tuple[np.ndarray, np.ndarray]:
    resid = robust_residual(preds, trues)
    num_bins = int(bins.max()) + 1
    channels = preds.shape[2]
    shape = (num_bins, channels) if per_channel else (num_bins, 1)
    intercept = np.zeros(shape, dtype=np.float32)
    slope = np.zeros(shape, dtype=np.float32)
    for b in range(num_bins):
        h = np.where(bins == b)[0]
        if not len(h):
            continue
        x = preds[:, h, :].astype(np.float64, copy=False)
        y = resid[:, h, :].astype(np.float64, copy=False)
        axes = (0, 1) if per_channel else (0, 1, 2)
        mx = np.mean(x, axis=axes, keepdims=False)
        my = np.mean(y, axis=axes, keepdims=False)
        xc = x - (mx.reshape(1, 1, -1) if per_channel else mx)
        yc = y - (my.reshape(1, 1, -1) if per_channel else my)
        cov = np.mean(xc * yc, axis=axes)
        var = np.mean(xc * xc, axis=axes)
        beta = cov / (var * 1.10 + 1e-8)
        bias = my - beta * mx
        intercept[b] = np.asarray(bias, dtype=np.float32).reshape(-1)
        slope[b] = np.asarray(beta, dtype=np.float32).reshape(-1)
    return intercept, slope


def apply_affine(preds: np.ndarray, bins: np.ndarray, params: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    intercept, slope = params
    corr = np.empty_like(preds, dtype=np.float32)
    for b in range(int(bins.max()) + 1):
        h = np.where(bins == b)[0]
        corr[:, h, :] = intercept[b].reshape(1, 1, -1) + slope[b].reshape(1, 1, -1) * preds[:, h, :]
    return corr


def context_features(preds: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    last = inputs[:, -1, :]
    lag = inputs[:, max(0, inputs.shape[1] - min(24, inputs.shape[1])), :]
    slope = last - lag
    forecast_slope = preds[:, -1, :] - preds[:, 0, :]
    return np.stack(
        [
            np.mean(last, axis=1),
            np.std(last, axis=1),
            np.mean(slope, axis=1),
            np.std(slope, axis=1),
            np.mean(preds, axis=(1, 2)),
            np.std(preds, axis=(1, 2)),
            np.mean(forecast_slope, axis=1),
            np.std(forecast_slope, axis=1),
        ],
        axis=1,
    ).astype(np.float32)


def fit_context_ridge(
    preds: np.ndarray,
    trues: np.ndarray,
    inputs: np.ndarray,
    bins: np.ndarray,
) -> list[dict[str, np.ndarray]]:
    resid = robust_residual(preds, trues)
    n_idx = sample_idx(preds.shape[0], 64)
    c_idx = sample_idx(preds.shape[2], 64)
    states: list[dict[str, np.ndarray]] = []
    last = inputs[:, -1, :]
    lag = inputs[:, max(0, inputs.shape[1] - min(24, inputs.shape[1])), :]
    slope = last - lag
    for b in range(int(bins.max()) + 1):
        h_all = np.where(bins == b)[0]
        h_idx = h_all[sample_idx(len(h_all), 32)]
        p = preds[np.ix_(n_idx, h_idx, c_idx)].astype(np.float64)
        l = last[np.ix_(n_idx, c_idx)].astype(np.float64)[:, None, :]
        s = slope[np.ix_(n_idx, c_idx)].astype(np.float64)[:, None, :]
        y = resid[np.ix_(n_idx, h_idx, c_idx)].astype(np.float64).reshape(-1)
        x = np.stack(
            [p, p - l, np.broadcast_to(l, p.shape), np.broadcast_to(s, p.shape)],
            axis=-1,
        ).reshape(-1, 4)
        mean = np.mean(x, axis=0)
        std = np.maximum(np.std(x, axis=0), 1e-5)
        z = (x - mean) / std
        design = np.column_stack([np.ones(len(z)), z])
        ridge = np.diag([1e-4, 1.0, 1.0, 1.0, 1.0])
        beta = np.linalg.solve(design.T @ design + ridge, design.T @ y)
        states.append({"mean": mean.astype(np.float32), "std": std.astype(np.float32), "beta": beta.astype(np.float32)})
    return states


def apply_context_ridge(preds: np.ndarray, inputs: np.ndarray, bins: np.ndarray, states: list[dict[str, np.ndarray]]) -> np.ndarray:
    corr = np.empty_like(preds, dtype=np.float32)
    last = inputs[:, -1, :]
    lag = inputs[:, max(0, inputs.shape[1] - min(24, inputs.shape[1])), :]
    slope = last - lag
    for b, state in enumerate(states):
        h = np.where(bins == b)[0]
        p = preds[:, h, :]
        l = last[:, None, :]
        s = slope[:, None, :]
        parts = [p, p - l, np.broadcast_to(l, p.shape), np.broadcast_to(s, p.shape)]
        out = np.full(p.shape, float(state["beta"][0]), dtype=np.float32)
        for j, part in enumerate(parts):
            out += state["beta"][j + 1] * (part - state["mean"][j]) / state["std"][j]
        corr[:, h, :] = out
    return corr


def fit_knn_state(
    preds: np.ndarray,
    trues: np.ndarray,
    inputs: np.ndarray,
    bins: np.ndarray,
) -> dict[str, np.ndarray]:
    n_idx = sample_idx(preds.shape[0], 128)
    features = context_features(preds[n_idx], inputs[n_idx]).astype(np.float64)
    mean = np.mean(features, axis=0)
    std = np.maximum(np.std(features, axis=0), 1e-5)
    features = (features - mean) / std
    resid = robust_residual(preds[n_idx], trues[n_idx])
    prototypes = np.zeros((len(n_idx), int(bins.max()) + 1, preds.shape[2]), dtype=np.float32)
    for b in range(int(bins.max()) + 1):
        h = np.where(bins == b)[0]
        prototypes[:, b, :] = np.mean(resid[:, h, :], axis=1, dtype=np.float64).astype(np.float32)
    return {"features": features.astype(np.float32), "mean": mean.astype(np.float32), "std": std.astype(np.float32), "prototypes": prototypes}


def apply_knn(preds: np.ndarray, inputs: np.ndarray, bins: np.ndarray, state: dict[str, np.ndarray], k: int = 8) -> np.ndarray:
    query = (context_features(preds, inputs) - state["mean"]) / state["std"]
    corr = np.empty_like(preds, dtype=np.float32)
    k = min(k, len(state["features"]))
    for start in range(0, len(query), 64):
        end = min(start + 64, len(query))
        dist = np.sum((query[start:end, None, :] - state["features"][None, :, :]) ** 2, axis=2)
        near = np.argpartition(dist, k - 1, axis=1)[:, :k]
        proto = np.mean(state["prototypes"][near], axis=1)
        corr[start:end] = proto[:, bins, :]
    return corr


def robust_state(resid: np.ndarray, bins: np.ndarray) -> np.ndarray:
    center = hbin_channel_center(robust_residual(np.zeros_like(resid), resid), bins)
    out = np.zeros((int(bins.max()) + 1, resid.shape[2]), dtype=np.float32)
    for b in range(len(out)):
        h = np.where(bins == b)[0]
        out[b] = np.mean(center[h], axis=0)
    return out


def additive_projection(state: np.ndarray) -> np.ndarray:
    global_mean = float(np.mean(state))
    return (np.mean(state, axis=1, keepdims=True) + np.mean(state, axis=0, keepdims=True) - global_mean).astype(np.float32)


def causal_additive_corr(
    preds: np.ndarray,
    trues: np.ndarray,
    bins: np.ndarray,
    init_state: np.ndarray,
    alpha: float,
) -> np.ndarray:
    state = init_state.astype(np.float32, copy=True)
    corr = np.empty_like(preds, dtype=np.float32)
    horizon = preds.shape[1]
    for origin in range(preds.shape[0]):
        corr[origin] = additive_projection(state)[bins]
        matured = min(horizon, origin)
        if not matured:
            continue
        h = np.arange(matured)
        src = origin - h - 1
        residual = trues[src, h, :] - preds[src, h, :]
        for b in range(state.shape[0]):
            mask = bins[h] == b
            if np.any(mask):
                update = np.mean(residual[mask], axis=0)
                state[b] = (1.0 - alpha) * state[b] + alpha * update
    return corr


def mpa_corrections(
    fit: dict[str, np.ndarray],
    tune: dict[str, np.ndarray],
    route: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    bins: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    periods = [24, 48, 96, 168, 336]
    weights = cf.fit_mpa_weights(fit["preds"], fit["trues"], fit["inputs"], periods, bins)
    corrs = {
        name: cf.mpa_corr(part["inputs"], part["preds"], periods, bins, weights)
        for name, part in [("fit", fit), ("tune", tune), ("route", route), ("test", test)]
    }
    clip = cf.safe_quantile_abs(corrs["fit"], 0.99)
    gamma = cf.tune_static_gamma_bins(tune["preds"], tune["trues"], corrs["tune"], bins, clip)
    return corrs, gamma, {"periods": periods, "period_weights": np.round(weights, 6).tolist(), "mpa_gamma": np.round(gamma, 4).tolist()}


def build_corrections(
    method_id: str,
    fit: dict[str, np.ndarray],
    tune: dict[str, np.ndarray],
    route: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    bins: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    parts = {"fit": fit, "tune": tune, "route": route, "test": test}
    fit_resid = robust_residual(fit["preds"], fit["trues"])
    if method_id == "rb_hbin_channel":
        template = hbin_channel_center(fit_resid, bins)
        return {name: np.broadcast_to(template[None], part["preds"].shape) for name, part in parts.items()}, {"template": "hbin_channel"}
    if method_id == "rb_horizon_shared":
        template = horizon_shared_center(fit_resid)
        return {name: np.broadcast_to(template[None], part["preds"].shape) for name, part in parts.items()}, {"template": "horizon_shared"}
    if method_id == "rb_svd_rank1":
        template = svd_center(fit_resid, rank=1)
        return {name: np.broadcast_to(template[None], part["preds"].shape) for name, part in parts.items()}, {"rank": 1}
    if method_id in {"affine_hbin_channel", "affine_shared"}:
        params = fit_affine(fit["preds"], fit["trues"], bins, per_channel=method_id.endswith("channel"))
        return {name: apply_affine(part["preds"], bins, params) for name, part in parts.items()}, {"affine_per_channel": method_id.endswith("channel")}
    if method_id == "context_ridge_hbin":
        state = fit_context_ridge(fit["preds"], fit["trues"], fit["inputs"], bins)
        return {name: apply_context_ridge(part["preds"], part["inputs"], bins, state) for name, part in parts.items()}, {"ridge_features": 4}
    if method_id == "knn_regime_hbin":
        state = fit_knn_state(fit["preds"], fit["trues"], fit["inputs"], bins)
        return {name: apply_knn(part["preds"], part["inputs"], bins, state, k=8) for name, part in parts.items()}, {"knn_k": 8, "fit_prototypes": int(len(state["features"]))}
    if method_id == "causal_additive_ema":
        init_fit = robust_state(fit["trues"] - fit["preds"], bins)
        best: tuple[float, float, np.ndarray] | None = None
        for alpha in [0.01, 0.03, 0.10]:
            tune_corr = causal_additive_corr(tune["preds"], tune["trues"], bins, init_fit, alpha)
            clip = cf.safe_quantile_abs(init_fit, 0.95)
            gamma = cf.tune_static_gamma_bins(tune["preds"], tune["trues"], tune_corr, bins, clip)
            ev = cf.eval_static_corr_bins(tune["preds"], tune["trues"], tune_corr, bins, gamma, clip)
            candidate = (float(ev["method_mse"]), alpha, gamma)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        _, alpha, _gamma = best
        fit_tune_resid = np.concatenate([fit["trues"] - fit["preds"], tune["trues"] - tune["preds"]], axis=0)
        full_val_resid = np.concatenate([fit_tune_resid, route["trues"] - route["preds"]], axis=0)
        init_route = robust_state(fit_tune_resid, bins)
        init_test = robust_state(full_val_resid, bins)
        return {
            "fit": np.broadcast_to(additive_projection(init_fit)[bins][None], fit["preds"].shape),
            "tune": causal_additive_corr(tune["preds"], tune["trues"], bins, init_fit, alpha),
            "route": causal_additive_corr(route["preds"], route["trues"], bins, init_route, alpha),
            "test": causal_additive_corr(test["preds"], test["trues"], bins, init_test, alpha),
        }, {"causal_alpha": alpha, "matured_rule": "source_origin+horizon+1<=current_origin"}
    if method_id in {"mpa_plus_bias", "mpa_plus_affine"}:
        mpa, mpa_gamma, meta = mpa_corrections(fit, tune, route, test, bins)
        gamma_h = mpa_gamma[bins].reshape(1, -1, 1)
        provisional = {name: part["preds"] + gamma_h * mpa[name] for name, part in parts.items()}
        if method_id == "mpa_plus_bias":
            second = hbin_channel_center(robust_residual(provisional["fit"], fit["trues"]), bins)
            total = {name: gamma_h * mpa[name] + np.broadcast_to(second[None], part["preds"].shape) for name, part in parts.items()}
            return total, {**meta, "second_stage": "hbin_channel_bias"}
        affine = fit_affine(provisional["fit"], fit["trues"], bins, per_channel=True)
        total = {name: gamma_h * mpa[name] + apply_affine(provisional[name], bins, affine) for name in parts}
        return total, {**meta, "second_stage": "hbin_channel_affine"}
    raise ValueError(f"unknown method_id: {method_id}")


def fit_correction_program(
    method_id: str,
    fit: dict[str, np.ndarray],
    tune: dict[str, np.ndarray],
    route: dict[str, np.ndarray],
    bins: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a compact correction state without retaining four dense tensors."""
    fit_resid = robust_residual(fit["preds"], fit["trues"])
    if method_id == "rb_hbin_channel":
        return {"template": hbin_channel_center(fit_resid, bins)}, {"template": "hbin_channel"}
    if method_id == "rb_horizon_shared":
        return {"template": horizon_shared_center(fit_resid)}, {"template": "horizon_shared"}
    if method_id == "rb_svd_rank1":
        return {"template": svd_center(fit_resid, rank=1)}, {"rank": 1}
    if method_id in {"affine_hbin_channel", "affine_shared"}:
        per_channel = method_id.endswith("channel")
        return {"affine": fit_affine(fit["preds"], fit["trues"], bins, per_channel)}, {
            "affine_per_channel": per_channel
        }
    if method_id == "context_ridge_hbin":
        return {
            "ridge": fit_context_ridge(fit["preds"], fit["trues"], fit["inputs"], bins)
        }, {"ridge_features": 4}
    if method_id == "knn_regime_hbin":
        state = fit_knn_state(fit["preds"], fit["trues"], fit["inputs"], bins)
        return {"knn": state}, {"knn_k": 8, "fit_prototypes": int(len(state["features"]))}
    if method_id == "causal_additive_ema":
        init_fit = robust_state(fit["trues"] - fit["preds"], bins)
        best: tuple[float, float] | None = None
        for alpha in [0.01, 0.03, 0.10]:
            corr = causal_additive_corr(tune["preds"], tune["trues"], bins, init_fit, alpha)
            clip = cf.safe_quantile_abs(init_fit, 0.95)
            gamma = cf.tune_static_gamma_bins(tune["preds"], tune["trues"], corr, bins, clip)
            evaluated = cf.eval_static_corr_bins(tune["preds"], tune["trues"], corr, bins, gamma, clip)
            candidate = (float(evaluated["method_mse"]), alpha)
            if best is None or candidate[0] < best[0]:
                best = candidate
            del corr
        assert best is not None
        fit_tune_resid = np.concatenate(
            [fit["trues"] - fit["preds"], tune["trues"] - tune["preds"]], axis=0
        )
        full_val_resid = np.concatenate(
            [fit_tune_resid, route["trues"] - route["preds"]], axis=0
        )
        return {
            "alpha": best[1],
            "init_fit": init_fit,
            "init_route": robust_state(fit_tune_resid, bins),
            "init_test": robust_state(full_val_resid, bins),
        }, {
            "causal_alpha": best[1],
            "matured_rule": "source_origin+horizon+1<=current_origin",
        }
    if method_id in {"mpa_plus_bias", "mpa_plus_affine"}:
        periods = [24, 48, 96, 168, 336]
        weights = cf.fit_mpa_weights(
            fit["preds"], fit["trues"], fit["inputs"], periods, bins
        )
        tune_mpa = cf.mpa_corr(tune["inputs"], tune["preds"], periods, bins, weights)
        mpa_clip = cf.safe_quantile_abs(tune_mpa, 0.99)
        mpa_gamma = cf.tune_static_gamma_bins(
            tune["preds"], tune["trues"], tune_mpa, bins, mpa_clip
        )
        del tune_mpa
        fit_mpa = cf.mpa_corr(fit["inputs"], fit["preds"], periods, bins, weights)
        gamma_h = mpa_gamma[bins].reshape(1, -1, 1)
        provisional_fit = fit["preds"] + gamma_h * fit_mpa
        state: dict[str, Any] = {
            "periods": periods,
            "weights": weights,
            "mpa_gamma": mpa_gamma,
        }
        if method_id == "mpa_plus_bias":
            state["second"] = hbin_channel_center(
                robust_residual(provisional_fit, fit["trues"]), bins
            )
            second_stage = "hbin_channel_bias"
        else:
            state["second"] = fit_affine(
                provisional_fit, fit["trues"], bins, per_channel=True
            )
            second_stage = "hbin_channel_affine"
        del fit_mpa, provisional_fit
        return state, {
            "periods": periods,
            "period_weights": np.round(weights, 6).tolist(),
            "mpa_gamma": np.round(mpa_gamma, 4).tolist(),
            "second_stage": second_stage,
        }
    raise ValueError(f"unknown method_id: {method_id}")


def apply_correction_program(
    method_id: str,
    state: dict[str, Any],
    part: dict[str, np.ndarray],
    bins: np.ndarray,
    phase: str,
) -> np.ndarray:
    if method_id in {"rb_hbin_channel", "rb_horizon_shared", "rb_svd_rank1"}:
        return np.broadcast_to(state["template"][None], part["preds"].shape)
    if method_id in {"affine_hbin_channel", "affine_shared"}:
        return apply_affine(part["preds"], bins, state["affine"])
    if method_id == "context_ridge_hbin":
        return apply_context_ridge(part["preds"], part["inputs"], bins, state["ridge"])
    if method_id == "knn_regime_hbin":
        return apply_knn(part["preds"], part["inputs"], bins, state["knn"], k=8)
    if method_id == "causal_additive_ema":
        if phase == "fit":
            template = additive_projection(state["init_fit"])[bins]
            return np.broadcast_to(template[None], part["preds"].shape)
        init_key = "init_fit" if phase == "tune" else "init_route" if phase == "route" else "init_test"
        return causal_additive_corr(
            part["preds"], part["trues"], bins, state[init_key], float(state["alpha"])
        )
    if method_id in {"mpa_plus_bias", "mpa_plus_affine"}:
        mpa = cf.mpa_corr(
            part["inputs"], part["preds"], state["periods"], bins, state["weights"]
        )
        gamma_h = state["mpa_gamma"][bins].reshape(1, -1, 1)
        first = gamma_h * mpa
        if method_id == "mpa_plus_bias":
            result = first + np.broadcast_to(state["second"][None], part["preds"].shape)
        else:
            result = first + apply_affine(part["preds"] + first, bins, state["second"])
        del mpa, first
        return result.astype(np.float32, copy=False)
    raise ValueError(f"unknown method_id: {method_id}")


def gate_gamma_on_route(
    route: dict[str, np.ndarray],
    corr: np.ndarray,
    bins: np.ndarray,
    gamma: np.ndarray,
    clip: float,
    min_gain: float,
) -> tuple[np.ndarray, list[float], list[int], list[float]]:
    gated = gamma.copy()
    gains: list[float] = []
    positives: list[int] = []
    worst_blocks: list[float] = []
    use = np.clip(corr, -clip, clip)
    for b in range(int(bins.max()) + 1):
        h = np.where(bins == b)[0]
        raw_err = route["preds"][:, h, :] - route["trues"][:, h, :]
        method_err = route["preds"][:, h, :] + gamma[b] * use[:, h, :] - route["trues"][:, h, :]
        raw_origin = np.mean(np.square(raw_err, dtype=np.float64), axis=(1, 2))
        method_origin = np.mean(np.square(method_err, dtype=np.float64), axis=(1, 2))
        raw_mse = float(np.mean(raw_origin))
        method_mse = float(np.mean(method_origin))
        gain = strict.route_gain(raw_mse, method_mse)
        block_gains = []
        for idx in np.array_split(np.arange(len(raw_origin)), min(4, len(raw_origin))):
            if len(idx):
                block_gains.append(strict.route_gain(float(np.mean(raw_origin[idx])), float(np.mean(method_origin[idx]))))
        gains.append(float(gain))
        positives.append(int(sum(x > 0 for x in block_gains)))
        worst_blocks.append(float(min(block_gains)) if block_gains else 0.0)
        if gain <= min_gain:
            gated[b] = 0.0
    return gated, gains, positives, worst_blocks


def make_row(
    config: nb.BackboneConfig,
    method: str,
    raw_mse: float,
    raw_mae: float,
    evaluated: dict[str, Any],
    route_eval: dict[str, Any],
    gamma: np.ndarray,
    clip: float,
    meta: dict[str, Any],
    route_gains: list[float],
    route_positive_blocks: list[int],
    route_worst_blocks: list[float],
) -> dict[str, Any]:
    method_mse = float(evaluated["method_mse"])
    method_mae = float(evaluated["method_mae"])
    delta = 100.0 * (method_mse - raw_mse) / raw_mse
    active = bool(np.any(gamma > 0))
    if not active:
        method_mse, method_mae, delta = raw_mse, raw_mae, 0.0
    return {
        "dataset": display_dataset(config.dataset),
        "backbone": config.model,
        "model": config.model,
        "pred_len": int(config.pred_len),
        "seed": int(config.seed),
        "config_key": config_key(config),
        "method": method,
        "raw_mse": raw_mse,
        "raw_mae": raw_mae,
        "method_mse": method_mse,
        "method_mae": method_mae,
        "delta_mse_pct": delta,
        "gain_pct": -delta,
        "route_gain_pct": strict.route_gain(float(route_eval["raw_mse"]), float(route_eval["method_mse"])),
        "gamma_bins_json": json.dumps(np.round(gamma, 4).tolist()),
        "clip_value": float(clip),
        "active": active,
        "fallback": not active,
        "fallback_type": "algorithmic",
        "fallback_reason": "all_route_bins_raw" if not active else "none",
        "bin_route_gain_json": json.dumps(np.round(route_gains, 6).tolist()),
        "bin_route_positive_blocks_json": json.dumps(route_positive_blocks),
        "bin_route_worst_block_json": json.dumps(np.round(route_worst_blocks, 6).tolist()),
        "bin_test_raw_mse_json": json.dumps(np.round(evaluated["bin_raw_mse"], 10).tolist()),
        "bin_test_method_mse_json": json.dumps(np.round(evaluated["bin_method_mse"], 10).tolist()),
        "correction_norm_mean": float(evaluated.get("correction_norm_mean", 0.0)),
        "correction_norm_max": float(evaluated.get("correction_norm_max", 0.0)),
        "selection_protocol": "validation_fit_tune_route_horizon_bin",
        "uses_external_outputs": False,
        "uses_test_for_selection": False,
        "meta_json": json.dumps(meta, sort_keys=True),
    }


def evaluate_method(
    config: nb.BackboneConfig,
    method_id: str,
    val_arrays: dict[str, np.ndarray],
    test_arrays: dict[str, np.ndarray],
    route_min_gain: float,
) -> list[dict[str, Any]]:
    split = strict.split_validation_three(val_arrays["preds"], val_arrays["trues"], val_arrays["inputs"])
    fit = {"preds": split[0], "trues": split[1], "inputs": split[2]}
    tune = {"preds": split[3], "trues": split[4], "inputs": split[5]}
    route = {"preds": split[6], "trues": split[7], "inputs": split[8]}
    test = {k: np.asarray(test_arrays[k], dtype=np.float32) for k in ["preds", "trues", "inputs"]}
    bins = cf.horizon_bin_ids(int(config.pred_len), num_bins=4)
    state, meta = fit_correction_program(method_id, fit, tune, route, bins)
    fit_count = min(64, len(fit["preds"]))
    fit_sample = {key: value[:fit_count] for key, value in fit.items()}
    fit_corr = apply_correction_program(method_id, state, fit_sample, bins, "fit")
    clip = cf.safe_quantile_abs(fit_corr, 0.99)
    del fit_corr, fit_sample

    tune_corr = apply_correction_program(method_id, state, tune, bins, "tune")
    gamma = cf.tune_static_gamma_bins(tune["preds"], tune["trues"], tune_corr, bins, clip)
    del tune_corr

    route_corr = apply_correction_program(method_id, state, route, bins, "route")
    route_expert = cf.eval_static_corr_bins(
        route["preds"], route["trues"], route_corr, bins, gamma, clip, collect_origin_blocks=True
    )
    gated, route_gains, positives, worst = gate_gamma_on_route(
        route, route_corr, bins, gamma, clip, route_min_gain
    )
    route_gated = cf.eval_static_corr_bins(
        route["preds"], route["trues"], route_corr, bins, gated, clip, collect_origin_blocks=True
    )
    del route_corr

    test_corr = apply_correction_program(method_id, state, test, bins, "test")
    test_expert = cf.eval_static_corr_bins(
        test["preds"], test["trues"], test_corr, bins, gamma, clip
    )
    test_gated = cf.eval_static_corr_bins(
        test["preds"], test["trues"], test_corr, bins, gated, clip
    )
    del test_corr, state
    raw_mse, raw_mae = base.metrics_chunked(test["preds"], test["trues"])
    return [
        make_row(config, f"{method_id}__expert", raw_mse, raw_mae, test_expert, route_expert, gamma, clip, meta, route_gains, positives, worst),
        make_row(config, f"{method_id}__route", raw_mse, raw_mae, test_gated, route_gated, gated, clip, meta, route_gains, positives, worst),
    ]


def align_raw(rows: list[dict[str, Any]], expected: dict[str, float], measured: tuple[float, float]) -> list[dict[str, Any]]:
    canonical_mse, canonical_mae = float(expected["raw_mse"]), float(expected["raw_mae"])
    for row in rows:
        row["raw_mse"] = canonical_mse
        row["raw_mae"] = canonical_mae
        row["raw_reexport_mse"] = measured[0]
        row["raw_reexport_mae"] = measured[1]
        row["raw_source"] = "outputs/raw_metrics_cache/raw_metrics.csv"
        if bool(row["fallback"]):
            row["method_mse"] = canonical_mse
            row["method_mae"] = canonical_mae
            row["delta_mse_pct"] = 0.0
            row["gain_pct"] = 0.0
        else:
            delta = 100.0 * (float(row["method_mse"]) - canonical_mse) / canonical_mse
            row["delta_mse_pct"] = delta
            row["gain_pct"] = -delta
    return rows


def load_raw(path: Path) -> dict[tuple[str, str, int, int], dict[str, float]]:
    frame = pd.read_csv(path)
    return {
        (str(r.model), display_dataset(str(r.dataset)), int(r.pred_len), int(r.seed)): {"raw_mse": float(r.raw_mse), "raw_mae": float(r.raw_mae)}
        for r in frame.itertuples(index=False)
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, part in frame.groupby("method"):
        delta = part["delta_mse_pct"].to_numpy(float)
        rows.append(
            {
                "method": method,
                "configs": int(part["config_key"].nunique()),
                "mean_gain_pct": float(-np.mean(delta)),
                "mean_mae_gain_pct": float(np.mean(100.0 * (part["raw_mae"] - part["method_mae"]) / part["raw_mae"])),
                "wins": int(np.sum(delta < -1e-12)),
                "ties": int(np.sum(np.abs(delta) <= 1e-12)),
                "harmful": int(np.sum(delta > 0)),
                "practical_harmful_0p1": int(np.sum(delta > 0.1)),
                "severe_harmful_1p0": int(np.sum(delta > 1.0)),
                "worst_degradation_pct": float(max(0.0, np.max(delta))),
                "active_count": int(part["active"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_gain_pct", ascending=False)


def row_paths(out_root: Path, method_id: str, key: str) -> tuple[Path, Path]:
    safe = key.replace("|", "__")
    root = out_root / method_id
    return root / "queue_status" / f"{safe}.json", root / "rows" / f"{safe}.csv"


def run_one(payload: dict[str, Any]) -> dict[str, Any]:
    config = nb.BackboneConfig(**payload["config"])
    key = config_key(config)
    marker, row_csv = row_paths(Path(payload["out_root"]), payload["method_id"], key)
    if row_csv.exists() and not payload["force"]:
        return {"config_key": key, "status": "skipped_complete", "rows": 0, "failures": []}
    atomic_json(marker, {"config_key": key, "status": "running", "method_id": payload["method_id"], "started_at": time.strftime("%F %T")})
    started = time.time()
    try:
        val, test = nb.collect_val_test_arrays_in_memory(config)
        measured = base.metrics_chunked(test["preds"], test["trues"])
        expected = payload["raw_expected"]
        if not (
            np.isclose(measured[0], expected["raw_mse"], atol=payload["raw_match_atol"], rtol=payload["raw_match_rtol"])
            and np.isclose(measured[1], expected["raw_mae"], atol=payload["raw_match_atol"], rtol=payload["raw_match_rtol"])
        ):
            raise ValueError(f"Raw source mismatch for {key}: {measured} vs {expected}")
        rows = evaluate_method(config, payload["method_id"], val, test, payload["route_min_gain"])
        rows = align_raw(rows, expected, measured)
        atomic_csv(row_csv, pd.DataFrame(rows))
        atomic_json(marker, {"config_key": key, "status": "completed", "method_id": payload["method_id"], "rows": len(rows), "elapsed_sec": round(time.time() - started, 3)})
        return {"config_key": key, "status": "completed", "rows": len(rows), "failures": []}
    except Exception as exc:
        failure = {
            "config_key": key,
            "status": "failed",
            "method_id": payload["method_id"],
            "fallback_type": "technical_failure",
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json(marker, failure)
        return {"config_key": key, "status": "failed", "rows": 0, "failures": [failure]}
    finally:
        gc.collect()


def self_test() -> None:
    rng = np.random.default_rng(2027)
    n, horizon, channels, lookback = 36, 12, 3, 24
    inputs = rng.normal(size=(n, lookback, channels)).astype(np.float32)
    preds = (inputs[:, -1:, :] + 0.1 * rng.normal(size=(n, horizon, channels))).astype(np.float32)
    pattern = np.linspace(-0.15, 0.2, horizon, dtype=np.float32)[None, :, None]
    trues = (preds + pattern + np.array([0.05, -0.03, 0.08], dtype=np.float32)[None, None, :]).astype(np.float32)
    val = {"preds": preds, "trues": trues, "inputs": inputs}
    test = {"preds": preds.copy(), "trues": trues.copy(), "inputs": inputs.copy()}
    config = nb.BackboneConfig(model="DLinear", dataset="ETTm2", pred_len=horizon, seed=2)
    split = strict.split_validation_three(val["preds"], val["trues"], val["inputs"])
    fit = {"preds": split[0], "trues": split[1], "inputs": split[2]}
    tune = {"preds": split[3], "trues": split[4], "inputs": split[5]}
    route = {"preds": split[6], "trues": split[7], "inputs": split[8]}
    bins = cf.horizon_bin_ids(horizon, 4)
    for method_id in METHOD_IDS:
        dense, _ = build_corrections(method_id, fit, tune, route, test, bins)
        state, _ = fit_correction_program(method_id, fit, tune, route, bins)
        for phase, part in (("fit", fit), ("tune", tune), ("route", route), ("test", test)):
            streamed = apply_correction_program(method_id, state, part, bins, phase)
            if not np.allclose(streamed, dense[phase], atol=1e-6, rtol=1e-6):
                raise AssertionError(f"streaming mismatch for {method_id}:{phase}")
        rows = evaluate_method(config, method_id, val, test, route_min_gain=0.0)
        if len(rows) != 2 or not all(np.isfinite(r["method_mse"]) for r in rows):
            raise AssertionError(f"invalid output for {method_id}")
    init = robust_state(trues[:12] - preds[:12], bins)
    c1 = causal_additive_corr(preds, trues, bins, init, 0.03)
    changed = trues.copy()
    changed[20:] += 100.0
    c2 = causal_additive_corr(preds, changed, bins, init, 0.03)
    if not np.array_equal(c1[:21], c2[:21]):
        raise AssertionError("future labels changed an earlier causal correction")
    print(f"self-test passed for {len(METHOD_IDS)} methods")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one strict internal-only residual expert family.")
    parser.add_argument("--method-id", choices=METHOD_IDS)
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--raw-metrics-csv", type=Path, default=Path("outputs/raw_metrics_cache/raw_metrics.csv"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/internal_only_search/cf_residual_expert_discovery"))
    parser.add_argument("--models", nargs="*", default=cf.BACKBONES)
    parser.add_argument("--datasets", nargs="*", default=["ETTh2", "ETTm2", "electricity", "traffic"])
    parser.add_argument("--pred-lens", nargs="*", type=int, default=[96, 720])
    parser.add_argument("--seeds", nargs="*", type=int, default=[2])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", default=["0"])
    parser.add_argument("--route-min-gain", type=float, default=0.0)
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
    if not args.method_id or not args.checkpoint_root:
        raise ValueError("--method-id and --checkpoint-root are required")
    raw_lookup = load_raw(args.raw_metrics_csv)
    datasets = [canonical_dataset(x) for x in args.datasets]
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
        raw_key = (config.model, display_dataset(config.dataset), int(config.pred_len), int(config.seed))
        if raw_key not in raw_lookup:
            raise ValueError(f"Raw metrics missing {raw_key}")
        payloads.append(
            {
                "config": asdict(config),
                "method_id": args.method_id,
                "out_root": str(args.out_root),
                "raw_expected": raw_lookup[raw_key],
                "route_min_gain": args.route_min_gain,
                "raw_match_atol": args.raw_match_atol,
                "raw_match_rtol": args.raw_match_rtol,
                "force": args.force,
            }
        )
    method_root = args.out_root / args.method_id
    status_path = method_root / "runner_status.json"
    started = time.time()
    atomic_json(status_path, {"status": "running", "method_id": args.method_id, "expected_configs": len(configs), "workers": args.workers})
    failures: list[dict[str, Any]] = []
    with mp.Pool(max(1, min(args.workers, len(payloads))), initializer=worker_init, initargs=(args.checkpoint_root, list(args.gpu_ids))) as pool:
        for index, result in enumerate(pool.imap_unordered(run_one, payloads), 1):
            failures.extend(result["failures"])
            atomic_json(status_path, {"status": "running", "method_id": args.method_id, "completed_or_seen": index, "expected_configs": len(configs), "failures_seen": len(failures), "last_config": result["config_key"], "last_status": result["status"], "elapsed_sec": round(time.time() - started, 3)})
            print(f"[{index}/{len(configs)}] {args.method_id} {result['config_key']} {result['status']}", flush=True)
    files = sorted((method_root / "rows").glob("*.csv"))
    detail = pd.concat([pd.read_csv(path) for path in files], ignore_index=True) if files else pd.DataFrame()
    if not detail.empty:
        atomic_csv(method_root / "detail.csv", detail.sort_values(["config_key", "method"]))
        atomic_csv(method_root / "summary.csv", summarize(detail))
    atomic_csv(method_root / "failure_log.csv", pd.DataFrame(failures))
    complete = not failures and detail["config_key"].nunique() == len(configs)
    final = {"status": "completed" if complete else "incomplete", "method_id": args.method_id, "expected_configs": len(configs), "completed_configs": int(detail["config_key"].nunique()) if not detail.empty else 0, "failures_seen": len(failures), "elapsed_sec": round(time.time() - started, 3), "uses_external_outputs": False, "uses_test_for_selection": False}
    atomic_json(status_path, final)
    atomic_json(method_root / "manifest.json", {"script": str(Path(__file__).relative_to(ROOT)), "method_id": args.method_id, "scope": {"models": args.models, "datasets": datasets, "pred_lens": args.pred_lens, "seeds": args.seeds}, "selection_protocol": "validation_fit_tune_route_horizon_bin", "matured_label_rule": "source_origin+horizon+1<=current_origin", "uses_external_outputs": False, "uses_test_for_selection": False, "technical_failure_falls_back_to_raw": False})
    if not detail.empty:
        print(pd.read_csv(method_root / "summary.csv").to_string(index=False))
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
