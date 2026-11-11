from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = ROOT / "scripts"
import sys

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_e06_new_backbones as nb
import run_internal_offline_candidates as base
import run_internal_strict_candidates_parallel as strict


DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "traffic", "weather"]
BACKBONES = ["DLinear", "PatchTST", "iTransformer", "MICN", "OLS", "FreTS"]
PRED_LENS = [96, 192, 336, 720]
SEEDS = [2, 42, 2021]
CF_METHODS = [
    "raw",
    "cf_mpa_anchor",
    "cf_frequency_residual",
    "cf_lowrank_causal_bias",
    "cf_msrc_hbin_router",
]


def env_list_int(name: str, default: list[int]) -> list[int]:
    text = os.environ.get(name, "").strip()
    if not text:
        return default
    return [int(x) for x in text.replace(",", " ").split() if x.strip()]


def env_float(name: str, default: float) -> float:
    text = os.environ.get(name, "").strip()
    return float(text) if text else default


def env_int(name: str, default: int) -> int:
    text = os.environ.get(name, "").strip()
    return int(text) if text else default


def cf_params_from_env() -> dict[str, Any]:
    return {
        "variant_id": os.environ.get("CF_VARIANT_ID", "cf_msrc_default"),
        "periods": env_list_int("CF_PERIODS", [24, 48, 96, 168, 336]),
        "freq_ms": env_list_int("CF_FREQ_MS", [4, 8, 16]),
        "num_bins": env_int("CF_NUM_BINS", 4),
        "anchor_clip_q": env_float("CF_ANCHOR_CLIP_Q", 0.95),
        "freq_clip_q": env_float("CF_FREQ_CLIP_Q", 0.95),
        "router_min_score": env_float("CF_ROUTER_MIN_SCORE", 0.0),
    }


def display_dataset(name: str) -> str:
    return {"electricity": "Electricity", "traffic": "Traffic", "weather": "Weather"}.get(name, name)


def config_key(config: nb.BackboneConfig) -> str:
    return f"{config.model}|{display_dataset(config.dataset)}|{config.pred_len}|{config.seed}"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def worker_init(export_root: str, checkpoint_root: str, gpu_ids: list[str] | None) -> None:
    nb.configure_roots(
        dirname="cf_msrc_stream_exports",
        output_root=export_root,
        checkpoint_root=checkpoint_root,
    )
    if gpu_ids:
        identity = mp.current_process()._identity  # type: ignore[attr-defined]
        worker_idx = int(identity[0]) - 1 if identity else 0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[worker_idx % len(gpu_ids)])


def horizon_bin_ids(horizon: int, num_bins: int = 4) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    raw_edges = [0, max(1, horizon // 8), max(2, horizon // 4), max(3, horizon // 2), horizon]
    if num_bins != 4:
        raw_edges = [round(i * horizon / num_bins) for i in range(num_bins + 1)]
        raw_edges[0] = 0
        raw_edges[-1] = horizon
    edges = [0]
    for edge in raw_edges[1:]:
        edge = int(min(max(edge, edges[-1] + 1), horizon))
        edges.append(edge)
        if edge == horizon:
            break
    if edges[-1] != horizon:
        edges[-1] = horizon
    bin_ids = np.zeros(horizon, dtype=np.int64)
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        bin_ids[lo:hi] = b
    return bin_ids


def safe_quantile_abs(arr: np.ndarray, q: float, max_items: int = 2_000_000) -> float:
    flat = np.ravel(np.abs(arr))
    if flat.size > max_items:
        idx = np.linspace(0, flat.size - 1, max_items, dtype=np.int64)
        flat = flat[idx]
    return float(np.quantile(flat, q))


def sample_indices(n: int, max_n: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, max_n, dtype=np.int64)


def anchor_corr(inputs: np.ndarray, preds: np.ndarray, period: int) -> np.ndarray:
    alt = base.naive_forecast_from_inputs(inputs, preds.shape[1], f"naive_seasonal{period}_hgamma_blend")
    return (alt - preds).astype(np.float32, copy=False)


def fit_mpa_weights(
    fit_preds: np.ndarray,
    fit_trues: np.ndarray,
    fit_inputs: np.ndarray,
    periods: list[int],
    bin_ids: np.ndarray,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Fit nonnegative period weights per horizon bin from validation-fit only."""
    n_fit, horizon, channels = fit_preds.shape
    origin_idx = sample_indices(n_fit, 128)
    channel_idx = sample_indices(channels, 64)
    # A seasonal anchor is only observable when the complete period is present
    # in the input window. The underlying helper otherwise falls back to the
    # last value, which would silently duplicate an unrelated expert.
    p_keep = [int(p) for p in periods if 0 < int(p) <= fit_inputs.shape[1]]
    num_bins = int(bin_ids.max()) + 1
    if not p_keep:
        return np.zeros((num_bins, len(periods)), dtype=np.float32)
    corrs = []
    fit_preds_s = fit_preds[np.ix_(origin_idx, np.arange(horizon), channel_idx)]
    fit_inputs_s = fit_inputs[np.ix_(origin_idx, np.arange(fit_inputs.shape[1]), channel_idx)]
    resid_s = fit_trues[np.ix_(origin_idx, np.arange(horizon), channel_idx)] - fit_preds_s
    for p in p_keep:
        corrs.append(anchor_corr(fit_inputs_s, fit_preds_s, int(p)))
    x = np.stack(corrs, axis=-1).astype(np.float64, copy=False)  # [N,H,C,P]
    y = resid_s.astype(np.float64, copy=False)
    weights = np.zeros((num_bins, len(p_keep)), dtype=np.float32)
    eye = np.eye(len(p_keep), dtype=np.float64)
    for b in range(num_bins):
        h_idx = np.where(bin_ids == b)[0]
        xb = x[:, h_idx, :, :].reshape(-1, len(p_keep))
        yb = y[:, h_idx, :].reshape(-1)
        xtx = xb.T @ xb + ridge * eye
        xty = xb.T @ yb
        try:
            w = np.linalg.solve(xtx, xty)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(xtx, xty, rcond=None)[0]
        w = np.maximum(w, 0.0)
        s = float(np.sum(w))
        if s > 1.0:
            w = w / s
        weights[b] = w.astype(np.float32)
    # Return dense [num_bins, original_period_count]. Missing periods keep weight zero.
    dense = np.zeros((num_bins, len(periods)), dtype=np.float32)
    for j, p in enumerate(p_keep):
        dense[:, periods.index(p)] = weights[:, j]
    return dense


def mpa_corr(inputs: np.ndarray, preds: np.ndarray, periods: list[int], bin_ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    corr = np.zeros_like(preds, dtype=np.float32)
    for p_idx, period in enumerate(periods):
        if np.allclose(weights[:, p_idx], 0.0):
            continue
        c = anchor_corr(inputs, preds, int(period))
        gamma_h = weights[bin_ids, p_idx].reshape(1, -1, 1).astype(np.float32)
        corr += gamma_h * c
    return corr


def dct_basis(horizon: int, m: int) -> np.ndarray:
    h = np.arange(horizon, dtype=np.float64)[:, None]
    k = np.arange(m, dtype=np.float64)[None, :]
    basis = np.cos(np.pi * (h + 0.5) * k / float(horizon))
    basis[:, 0] *= 1.0 / math.sqrt(2.0)
    norm = np.sqrt(np.sum(basis * basis, axis=0, keepdims=True))
    return (basis / np.maximum(norm, 1e-12)).astype(np.float32)


def fit_frequency_corr(fit_preds: np.ndarray, fit_trues: np.ndarray, m: int, ridge: float = 1e-2) -> np.ndarray:
    resid_mean = np.mean(fit_trues - fit_preds, axis=0, dtype=np.float64).astype(np.float32)  # [H,C]
    basis = dct_basis(fit_preds.shape[1], m).astype(np.float64)
    xtx = basis.T @ basis + ridge * np.eye(m, dtype=np.float64)
    xty = basis.T @ resid_mean.astype(np.float64)
    beta = np.linalg.solve(xtx, xty)
    corr = (basis @ beta).astype(np.float32)
    return corr


def tune_static_gamma_bins(
    tune_preds: np.ndarray,
    tune_trues: np.ndarray,
    corr: np.ndarray,
    bin_ids: np.ndarray,
    clip_value: float | None = None,
) -> np.ndarray:
    num_bins = int(bin_ids.max()) + 1
    gamma_options = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
    gamma = np.zeros(num_bins, dtype=np.float32)
    corr_use = corr
    if corr.ndim == 2:
        corr_use = np.broadcast_to(corr[None, :, :], tune_preds.shape)
    if clip_value is not None and np.isfinite(clip_value):
        corr_use = np.clip(corr_use, -clip_value, clip_value)
    for b in range(num_bins):
        h_idx = np.where(bin_ids == b)[0]
        raw_err = tune_preds[:, h_idx, :] - tune_trues[:, h_idx, :]
        best_mse = float(np.mean(np.square(raw_err, dtype=np.float64)))
        best_g = 0.0
        c = corr_use[:, h_idx, :]
        for g in gamma_options[1:]:
            err = tune_preds[:, h_idx, :] + float(g) * c - tune_trues[:, h_idx, :]
            mse = float(np.mean(np.square(err, dtype=np.float64)))
            if mse < best_mse:
                best_mse = mse
                best_g = float(g)
        gamma[b] = best_g
    return gamma


def eval_static_corr_bins(
    preds: np.ndarray,
    trues: np.ndarray,
    corr: np.ndarray,
    bin_ids: np.ndarray,
    gamma_bins: np.ndarray,
    clip_value: float | None = None,
    collect_origin_blocks: bool = False,
) -> dict[str, Any]:
    num_bins = int(bin_ids.max()) + 1
    sq = np.zeros(num_bins, dtype=np.float64)
    ab = np.zeros(num_bins, dtype=np.float64)
    raw_sq = np.zeros(num_bins, dtype=np.float64)
    raw_ab = np.zeros(num_bins, dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.int64)
    raw_origin: list[float] = []
    method_origin: list[float] = []
    corr_abs_sum = 0.0
    corr_abs_max = 0.0
    corr_count = 0
    if corr.ndim == 2:
        corr_view = np.broadcast_to(corr[None, :, :], preds.shape)
    else:
        corr_view = corr
    gamma_h = gamma_bins[bin_ids].reshape(1, -1, 1).astype(np.float32)
    per_origin = int(np.prod(preds.shape[1:]))
    chunk_size = max(1, min(64, 8_000_000 // max(per_origin, 1)))
    for start in range(0, preds.shape[0], chunk_size):
        end = min(start + chunk_size, preds.shape[0])
        c = corr_view[start:end]
        if clip_value is not None and np.isfinite(clip_value):
            c = np.clip(c, -clip_value, clip_value)
        pred_chunk = preds[start:end]
        true_chunk = trues[start:end]
        method_err = (pred_chunk + gamma_h * c - true_chunk).astype(np.float32, copy=False)
        raw_err = (pred_chunk - true_chunk).astype(np.float32, copy=False)
        if collect_origin_blocks:
            raw_origin.extend(np.mean(np.square(raw_err, dtype=np.float64), axis=(1, 2)).tolist())
            method_origin.extend(np.mean(np.square(method_err, dtype=np.float64), axis=(1, 2)).tolist())
        abs_c = np.abs(c)
        corr_abs_sum += float(np.sum(abs_c, dtype=np.float64))
        corr_abs_max = max(corr_abs_max, float(np.max(abs_c)) if abs_c.size else 0.0)
        corr_count += int(abs_c.size)
        for b in range(num_bins):
            h_idx = np.where(bin_ids == b)[0]
            me = method_err[:, h_idx, :]
            re = raw_err[:, h_idx, :]
            sq[b] += float(np.sum(np.square(me, dtype=np.float64)))
            ab[b] += float(np.sum(np.abs(me), dtype=np.float64))
            raw_sq[b] += float(np.sum(np.square(re, dtype=np.float64)))
            raw_ab[b] += float(np.sum(np.abs(re), dtype=np.float64))
            counts[b] += int(me.size)
    method_mse = float(np.sum(sq) / np.sum(counts))
    method_mae = float(np.sum(ab) / np.sum(counts))
    raw_mse = float(np.sum(raw_sq) / np.sum(counts))
    raw_mae = float(np.sum(raw_ab) / np.sum(counts))
    out: dict[str, Any] = {
        "raw_mse": raw_mse,
        "raw_mae": raw_mae,
        "method_mse": method_mse,
        "method_mae": method_mae,
        "bin_raw_mse": (raw_sq / np.maximum(counts, 1)).tolist(),
        "bin_method_mse": (sq / np.maximum(counts, 1)).tolist(),
        "bin_raw_mae": (raw_ab / np.maximum(counts, 1)).tolist(),
        "bin_method_mae": (ab / np.maximum(counts, 1)).tolist(),
        "bin_counts": counts.tolist(),
        "correction_norm_mean": corr_abs_sum / max(corr_count, 1),
        "correction_norm_max": corr_abs_max,
    }
    if collect_origin_blocks:
        out.update(base.block_gain_stats_from_origin_mse(np.asarray(raw_origin), np.asarray(method_origin)))
    return out


def lowrank_init_state(fit_preds: np.ndarray, fit_trues: np.ndarray, bin_ids: np.ndarray, rank: int = 2) -> np.ndarray:
    resid = (fit_trues - fit_preds).astype(np.float32, copy=False)
    num_bins = int(bin_ids.max()) + 1
    channels = fit_preds.shape[2]
    state = np.zeros((num_bins, channels), dtype=np.float32)
    for b in range(num_bins):
        h_idx = np.where(bin_ids == b)[0]
        if len(h_idx):
            state[b] = np.mean(resid[:, h_idx, :], axis=(0, 1), dtype=np.float64).astype(np.float32)
    centered = state - np.mean(state, axis=0, keepdims=True)
    try:
        u, s, vt = np.linalg.svd(centered.astype(np.float64), full_matrices=False)
        r = max(1, min(rank, len(s)))
        low = (u[:, :r] * s[:r]) @ vt[:r, :]
        state = (low + np.mean(state, axis=0, keepdims=True)).astype(np.float32)
    except np.linalg.LinAlgError:
        pass
    return state


def causal_lowrank_eval(
    preds: np.ndarray,
    trues: np.ndarray,
    init_state: np.ndarray,
    bin_ids: np.ndarray,
    gamma: float,
    alpha: float,
    clip_value: float | None = None,
    collect_origin_blocks: bool = False,
    return_state: bool = False,
) -> tuple[float, float, dict[str, Any]] | tuple[float, float, dict[str, Any], np.ndarray]:
    state = init_state.astype(np.float32, copy=True)
    n, horizon, channels = preds.shape
    num_bins = state.shape[0]
    sq = np.zeros(num_bins, dtype=np.float64)
    ab = np.zeros(num_bins, dtype=np.float64)
    raw_sq = np.zeros(num_bins, dtype=np.float64)
    raw_ab = np.zeros(num_bins, dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.int64)
    raw_origin: list[float] = []
    method_origin: list[float] = []
    updates = 0
    corr_abs_sum = 0.0
    corr_abs_max = 0.0
    corr_count = 0
    for origin in range(n):
        matured = min(horizon, origin)
        if matured:
            h_idx = np.arange(matured)
            prev_origins = origin - h_idx - 1
            residual = trues[prev_origins, h_idx, :] - preds[prev_origins, h_idx, :]
            for b in range(num_bins):
                mask = bin_ids[h_idx] == b
                if np.any(mask):
                    r = np.mean(residual[mask], axis=0)
                    state[b] = (1.0 - alpha) * state[b] + alpha * r
                    updates += int(r.size)
        corr = state[bin_ids]
        if clip_value is not None and np.isfinite(clip_value):
            corr = np.clip(corr, -clip_value, clip_value)
        method_err = preds[origin] + gamma * corr - trues[origin]
        raw_err = preds[origin] - trues[origin]
        if collect_origin_blocks:
            raw_origin.append(float(np.mean(np.square(raw_err, dtype=np.float64))))
            method_origin.append(float(np.mean(np.square(method_err, dtype=np.float64))))
        abs_corr = np.abs(corr)
        corr_abs_sum += float(np.sum(abs_corr, dtype=np.float64))
        corr_abs_max = max(corr_abs_max, float(np.max(abs_corr)) if abs_corr.size else 0.0)
        corr_count += int(abs_corr.size)
        for b in range(num_bins):
            hsel = np.where(bin_ids == b)[0]
            me = method_err[hsel, :]
            re = raw_err[hsel, :]
            sq[b] += float(np.sum(np.square(me, dtype=np.float64)))
            ab[b] += float(np.sum(np.abs(me), dtype=np.float64))
            raw_sq[b] += float(np.sum(np.square(re, dtype=np.float64)))
            raw_ab[b] += float(np.sum(np.abs(re), dtype=np.float64))
            counts[b] += int(me.size)
    mse = float(np.sum(sq) / np.sum(counts))
    mae = float(np.sum(ab) / np.sum(counts))
    meta = {
        "bin_raw_mse": (raw_sq / np.maximum(counts, 1)).tolist(),
        "bin_method_mse": (sq / np.maximum(counts, 1)).tolist(),
        "bin_raw_mae": (raw_ab / np.maximum(counts, 1)).tolist(),
        "bin_method_mae": (ab / np.maximum(counts, 1)).tolist(),
        "bin_counts": counts.tolist(),
        "update_count": int(updates),
        "correction_norm_mean": corr_abs_sum / max(corr_count, 1),
        "correction_norm_max": corr_abs_max,
    }
    if collect_origin_blocks:
        meta.update(base.block_gain_stats_from_origin_mse(np.asarray(raw_origin), np.asarray(method_origin)))
    if return_state:
        return mse, mae, meta, state
    return mse, mae, meta


def tune_lowrank_causal(
    fit_preds: np.ndarray,
    fit_trues: np.ndarray,
    tune_preds: np.ndarray,
    tune_trues: np.ndarray,
    bin_ids: np.ndarray,
) -> dict[str, Any]:
    init = lowrank_init_state(fit_preds, fit_trues, bin_ids, rank=2)
    fit_abs = np.abs(fit_trues - fit_preds)
    raw_tune_mse, _ = base.metrics_chunked(tune_preds, tune_trues)
    best = {"gamma": 0.0, "alpha": 0.0, "clip_q": None, "clip_value": np.nan, "tune_mse": raw_tune_mse, "tune_gain_pct": 0.0}
    for gamma in [0.25, 0.50, 0.75]:
        for alpha in [0.02, 0.05, 0.10]:
            for clip_q in [None, 0.90, 0.95]:
                clip_value = np.nan if clip_q is None else safe_quantile_abs(fit_abs, float(clip_q))
                mse, _mae, _meta = causal_lowrank_eval(
                    tune_preds,
                    tune_trues,
                    init,
                    bin_ids,
                    gamma=float(gamma),
                    alpha=float(alpha),
                    clip_value=clip_value if clip_q is not None else None,
                )
                if mse < best["tune_mse"]:
                    best = {
                        "gamma": float(gamma),
                        "alpha": float(alpha),
                        "clip_q": clip_q,
                        "clip_value": clip_value,
                        "tune_mse": float(mse),
                        "tune_gain_pct": strict.route_gain(raw_tune_mse, float(mse)),
                    }
    return best


def row_from_eval(pair: dict[str, Any], method: str, raw_test_mse: float, raw_test_mae: float, route_eval: dict[str, Any], test_eval: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    delta = (test_eval["method_mse"] - raw_test_mse) / raw_test_mse * 100.0 if raw_test_mse > 0 else np.nan
    return {
        **pair,
        "method": method,
        "raw_mse": raw_test_mse,
        "raw_mae": raw_test_mae,
        "method_mse": test_eval["method_mse"],
        "method_mae": test_eval["method_mae"],
        "delta_mse_pct": delta,
        "gain_pct": -delta,
        "route_raw_mse": route_eval["raw_mse"],
        "route_mse": route_eval["method_mse"],
        "route_mae": route_eval["method_mae"],
        "route_gain_pct": strict.route_gain(route_eval["raw_mse"], route_eval["method_mse"]),
        "bin_route_raw_mse_json": json.dumps([round(float(x), 8) for x in route_eval["bin_raw_mse"]]),
        "bin_route_method_mse_json": json.dumps([round(float(x), 8) for x in route_eval["bin_method_mse"]]),
        "bin_test_raw_mse_json": json.dumps([round(float(x), 8) for x in test_eval["bin_raw_mse"]]),
        "bin_test_method_mse_json": json.dumps([round(float(x), 8) for x in test_eval["bin_method_mse"]]),
        "test_correction_norm_mean": test_eval.get("correction_norm_mean", np.nan),
        "test_correction_norm_max": test_eval.get("correction_norm_max", np.nan),
        **{k: v for k, v in route_eval.items() if k.startswith("val_block_")},
        **extra,
    }


def build_router_row(
    pair: dict[str, Any],
    raw_test_mse: float,
    raw_test_mae: float,
    route_evals: dict[str, dict[str, Any]],
    test_evals: dict[str, dict[str, Any]],
    bin_ids: np.ndarray,
    router_min_score: float,
) -> dict[str, Any]:
    num_bins = int(bin_ids.max()) + 1
    selected: list[str] = []
    raw_route = route_evals["raw"]
    raw_test = test_evals["raw"]
    total_sq = 0.0
    total_abs = 0.0
    total_count = 0
    total_raw_sq = 0.0
    route_sq = 0.0
    route_abs = 0.0
    route_count = 0
    for b in range(num_bins):
        best_method = "raw"
        best_score = float(router_min_score)
        raw_mse_b = float(raw_route["bin_raw_mse"][b])
        for method, ev in route_evals.items():
            if method == "raw":
                continue
            method_mse_b = float(ev["bin_method_mse"][b])
            gain_b = strict.route_gain(raw_mse_b, method_mse_b)
            score = gain_b
            if gain_b > best_score:
                best_method = method
                best_score = gain_b
        selected.append(best_method)
        ev_test = test_evals[best_method]
        count_b = int(ev_test["bin_counts"][b])
        total_count += count_b
        total_sq += float(ev_test["bin_method_mse"][b]) * count_b
        total_abs += float(ev_test["bin_method_mae"][b]) * count_b
        total_raw_sq += float(raw_test["bin_raw_mse"][b]) * count_b
        route_count_b = int(route_evals[best_method]["bin_counts"][b])
        route_count += route_count_b
        route_sq += float(route_evals[best_method]["bin_method_mse"][b]) * route_count_b
        route_abs += float(route_evals[best_method]["bin_method_mae"][b]) * route_count_b
    method_mse = total_sq / max(total_count, 1)
    method_mae = total_abs / max(total_count, 1)
    route_mse = route_sq / max(route_count, 1)
    route_mae = route_abs / max(route_count, 1)
    delta = (method_mse - raw_test_mse) / raw_test_mse * 100.0 if raw_test_mse > 0 else np.nan
    raw_bins = raw_test["bin_raw_mse"]
    selected_bins = [test_evals[selected[b]]["bin_method_mse"][b] for b in range(num_bins)]
    route_selected_bins = [route_evals[selected[b]]["bin_method_mse"][b] for b in range(num_bins)]
    return {
        **pair,
        "method": "cf_msrc_hbin_router",
        "raw_mse": raw_test_mse,
        "raw_mae": raw_test_mae,
        "method_mse": method_mse,
        "method_mae": method_mae,
        "delta_mse_pct": delta,
        "gain_pct": -delta,
        "route_raw_mse": raw_route["raw_mse"],
        "route_mse": route_mse,
        "route_mae": route_mae,
        "route_gain_pct": strict.route_gain(raw_route["raw_mse"], route_mse),
        "fallback": all(m == "raw" for m in selected),
        "fallback_type": "algorithmic",
        "fallback_reason": "all_bins_raw" if all(m == "raw" for m in selected) else "some_bins_selected",
        "selected_experts_json": json.dumps(selected),
        "bin_test_raw_mse_json": json.dumps([round(float(x), 8) for x in raw_bins]),
        "bin_test_method_mse_json": json.dumps([round(float(x), 8) for x in selected_bins]),
        "uses_external_outputs": False,
        "uses_test_for_selection": False,
        "router_min_score": router_min_score,
    }


def compute_rows(
    config: nb.BackboneConfig,
    val_arrays: dict[str, np.ndarray],
    test_arrays: dict[str, np.ndarray],
    params_override: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    params = dict(params_override) if params_override is not None else cf_params_from_env()
    val_preds = np.asarray(val_arrays["preds"], dtype=np.float32)
    val_trues = np.asarray(val_arrays["trues"], dtype=np.float32)
    val_inputs = np.asarray(val_arrays["inputs"], dtype=np.float32)
    test_preds = np.asarray(test_arrays["preds"], dtype=np.float32)
    test_trues = np.asarray(test_arrays["trues"], dtype=np.float32)
    test_inputs = np.asarray(test_arrays["inputs"], dtype=np.float32)
    (
        fit_preds,
        fit_trues,
        fit_inputs,
        tune_preds,
        tune_trues,
        tune_inputs,
        route_preds,
        route_trues,
        route_inputs,
    ) = strict.split_validation_three(val_preds, val_trues, val_inputs)

    raw_test_mse, raw_test_mae = base.metrics_chunked(test_preds, test_trues)
    raw_route_mse, raw_route_mae = base.metrics_chunked(route_preds, route_trues)
    bin_ids = horizon_bin_ids(int(config.pred_len), num_bins=int(params["num_bins"]))
    raw_route_eval = eval_static_corr_bins(route_preds, route_trues, np.zeros_like(route_preds), bin_ids, np.zeros(int(bin_ids.max()) + 1), collect_origin_blocks=True)
    raw_test_eval = eval_static_corr_bins(test_preds, test_trues, np.zeros_like(test_preds), bin_ids, np.zeros(int(bin_ids.max()) + 1))
    pair = {
        "dataset": display_dataset(config.dataset),
        "backbone": config.model,
        "model": config.model,
        "pred_len": int(config.pred_len),
        "seed": int(config.seed),
        "config_key": config_key(config),
        "selection_protocol": "validation_fit_tune_route_horizon_bin",
        "uses_external_outputs": False,
        "uses_test_for_selection": False,
        "fit_origins": int(fit_preds.shape[0]),
        "tune_origins": int(tune_preds.shape[0]),
        "route_origins": int(route_preds.shape[0]),
        "horizon_bins_json": json.dumps([int(x) for x in bin_ids.tolist()]),
        "cf_variant_id": params["variant_id"],
        "cf_params_json": json.dumps(params, sort_keys=True),
    }
    rows: list[dict[str, Any]] = [
        {
            **pair,
            "method": "raw",
            "raw_mse": raw_test_mse,
            "raw_mae": raw_test_mae,
            "method_mse": raw_test_mse,
            "method_mae": raw_test_mae,
            "delta_mse_pct": 0.0,
            "gain_pct": 0.0,
            "route_raw_mse": raw_route_mse,
            "route_mse": raw_route_mse,
            "route_mae": raw_route_mae,
            "route_gain_pct": 0.0,
            "fallback": True,
            "fallback_type": "algorithmic",
            "fallback_reason": "raw_anchor",
            **base.block_gain_stats_from_origin_mse(
                np.mean(np.square(route_preds - route_trues, dtype=np.float64), axis=(1, 2)),
                np.mean(np.square(route_preds - route_trues, dtype=np.float64), axis=(1, 2)),
            ),
        }
    ]
    route_evals: dict[str, dict[str, Any]] = {"raw": raw_route_eval}
    test_evals: dict[str, dict[str, Any]] = {"raw": raw_test_eval}

    # Expert 2: multi-period anchor calibration.
    periods = list(params["periods"])
    weights = fit_mpa_weights(fit_preds, fit_trues, fit_inputs, periods, bin_ids)
    tune_corr = mpa_corr(tune_inputs, tune_preds, periods, bin_ids, weights)
    clip_value = safe_quantile_abs(tune_corr, float(params["anchor_clip_q"]))
    gamma_bins = tune_static_gamma_bins(tune_preds, tune_trues, tune_corr, bin_ids, clip_value=clip_value)
    route_corr = mpa_corr(route_inputs, route_preds, periods, bin_ids, weights)
    test_corr = mpa_corr(test_inputs, test_preds, periods, bin_ids, weights)
    route_eval = eval_static_corr_bins(route_preds, route_trues, route_corr, bin_ids, gamma_bins, clip_value=clip_value, collect_origin_blocks=True)
    test_eval = eval_static_corr_bins(test_preds, test_trues, test_corr, bin_ids, gamma_bins, clip_value=clip_value)
    route_evals["cf_mpa_anchor"] = route_eval
    test_evals["cf_mpa_anchor"] = test_eval
    rows.append(
        row_from_eval(
            pair,
            "cf_mpa_anchor",
            raw_test_mse,
            raw_test_mae,
            route_eval,
            test_eval,
            {
                "periods_json": json.dumps(periods),
                "period_weights_json": json.dumps(np.round(weights, 6).tolist()),
                "gamma_bins_json": json.dumps(np.round(gamma_bins, 4).tolist()),
                "clip_q": float(params["anchor_clip_q"]),
                "clip_value": clip_value,
                "fallback": bool(np.all(gamma_bins == 0)),
                "fallback_type": "algorithmic",
                "fallback_reason": "tune_selected_raw_bins" if np.all(gamma_bins == 0) else "none",
            },
        )
    )

    # Expert 3: frequency residual calibration.
    best_freq: dict[str, Any] | None = None
    for m in list(params["freq_ms"]):
        corr_const = fit_frequency_corr(fit_preds, fit_trues, m=m, ridge=1e-2)
        clip_v = safe_quantile_abs(corr_const, float(params["freq_clip_q"]))
        gamma_b = tune_static_gamma_bins(
            tune_preds,
            tune_trues,
            np.broadcast_to(corr_const[None, :, :], tune_preds.shape),
            bin_ids,
            clip_value=clip_v,
        )
        # Frequency order is a candidate hyperparameter. Select it on tune,
        # leaving validation-route untouched for the final expert router.
        tune_ev = eval_static_corr_bins(tune_preds, tune_trues, corr_const, bin_ids, gamma_b, clip_value=clip_v)
        if best_freq is None or tune_ev["method_mse"] < best_freq["tune_eval"]["method_mse"]:
            best_freq = {"m": m, "corr": corr_const, "clip": clip_v, "gamma": gamma_b, "tune_eval": tune_ev}
    assert best_freq is not None
    best_freq["route_eval"] = eval_static_corr_bins(
        route_preds,
        route_trues,
        best_freq["corr"],
        bin_ids,
        best_freq["gamma"],
        clip_value=best_freq["clip"],
        collect_origin_blocks=True,
    )
    best_freq["test_eval"] = eval_static_corr_bins(
        test_preds,
        test_trues,
        best_freq["corr"],
        bin_ids,
        best_freq["gamma"],
        clip_value=best_freq["clip"],
    )
    route_evals["cf_frequency_residual"] = best_freq["route_eval"]
    test_evals["cf_frequency_residual"] = best_freq["test_eval"]
    rows.append(
        row_from_eval(
            pair,
            "cf_frequency_residual",
            raw_test_mse,
            raw_test_mae,
            best_freq["route_eval"],
            best_freq["test_eval"],
            {
                "freq_m": int(best_freq["m"]),
                "gamma_bins_json": json.dumps(np.round(best_freq["gamma"], 4).tolist()),
                "clip_q": float(params["freq_clip_q"]),
                "clip_value": float(best_freq["clip"]),
                "fallback": bool(np.all(best_freq["gamma"] == 0)),
                "fallback_type": "algorithmic",
                "fallback_reason": "tune_selected_raw_bins" if np.all(best_freq["gamma"] == 0) else "none",
            },
        )
    )

    # Expert 4: coarse causal low-rank residual bias.
    lrb_tuned = tune_lowrank_causal(fit_preds, fit_trues, tune_preds, tune_trues, bin_ids)
    if float(lrb_tuned["gamma"]) == 0.0:
        # An algorithmic Raw fallback must be bit-for-bit Raw in the result
        # table; recomputing equivalent arithmetic can create false strict
        # harmful flags at floating-point epsilon.
        route_eval = dict(raw_route_eval)
        test_eval = dict(raw_test_eval)
        route_eval["method_mse"] = raw_route_mse
        route_eval["method_mae"] = raw_route_mae
        test_eval["method_mse"] = raw_test_mse
        test_eval["method_mae"] = raw_test_mae
    else:
        lrb_init = lowrank_init_state(fit_preds, fit_trues, bin_ids, rank=2)
        _tm, _ta, _meta, tune_state = causal_lowrank_eval(
            tune_preds,
            tune_trues,
            lrb_init,
            bin_ids,
            gamma=float(lrb_tuned["gamma"]),
            alpha=float(lrb_tuned["alpha"]),
            clip_value=lrb_tuned["clip_value"] if lrb_tuned["clip_q"] is not None else None,
            return_state=True,
        )
        route_mse, route_mae, route_meta, route_state = causal_lowrank_eval(
            route_preds,
            route_trues,
            tune_state,
            bin_ids,
            gamma=float(lrb_tuned["gamma"]),
            alpha=float(lrb_tuned["alpha"]),
            clip_value=lrb_tuned["clip_value"] if lrb_tuned["clip_q"] is not None else None,
            collect_origin_blocks=True,
            return_state=True,
        )
        test_mse, test_mae, test_meta = causal_lowrank_eval(
            test_preds,
            test_trues,
            route_state,
            bin_ids,
            gamma=float(lrb_tuned["gamma"]),
            alpha=float(lrb_tuned["alpha"]),
            clip_value=lrb_tuned["clip_value"] if lrb_tuned["clip_q"] is not None else None,
        )
        route_eval = {"raw_mse": raw_route_mse, "raw_mae": raw_route_mae, "method_mse": route_mse, "method_mae": route_mae, **route_meta}
        test_eval = {"raw_mse": raw_test_mse, "raw_mae": raw_test_mae, "method_mse": test_mse, "method_mae": test_mae, **test_meta}
    route_evals["cf_lowrank_causal_bias"] = route_eval
    test_evals["cf_lowrank_causal_bias"] = test_eval
    rows.append(
        row_from_eval(
            pair,
            "cf_lowrank_causal_bias",
            raw_test_mse,
            raw_test_mae,
            route_eval,
            test_eval,
            {
                "gamma": lrb_tuned["gamma"],
                "alpha": lrb_tuned["alpha"],
                "clip_q": lrb_tuned["clip_q"],
                "clip_value": lrb_tuned["clip_value"],
                "tune_gain_pct": lrb_tuned["tune_gain_pct"],
                "fallback": bool(lrb_tuned["gamma"] == 0),
                "fallback_type": "algorithmic",
                "fallback_reason": "tune_selected_raw" if lrb_tuned["gamma"] == 0 else "none",
            },
        )
    )

    rows.append(
        build_router_row(
            pair,
            raw_test_mse,
            raw_test_mae,
            route_evals,
            test_evals,
            bin_ids,
            router_min_score=float(params["router_min_score"]),
        )
    )
    return rows


def align_rows_to_canonical_raw(
    rows: list[dict[str, Any]],
    canonical_raw_mse: float,
    canonical_raw_mae: float,
    measured_raw_mse: float,
    measured_raw_mae: float,
) -> list[dict[str, Any]]:
    """Use raw_metrics.csv as the only reported Raw source after alignment."""
    for row in rows:
        row["raw_mse"] = canonical_raw_mse
        row["raw_mae"] = canonical_raw_mae
        row["raw_reexport_mse"] = measured_raw_mse
        row["raw_reexport_mae"] = measured_raw_mae
        row["raw_alignment_mse_abs_error"] = abs(measured_raw_mse - canonical_raw_mse)
        row["raw_alignment_mae_abs_error"] = abs(measured_raw_mae - canonical_raw_mae)
        row["raw_source"] = "outputs/raw_metrics_cache/raw_metrics.csv"
        is_algorithmic_raw = bool(row.get("fallback", False)) and row.get("fallback_type") == "algorithmic"
        if row["method"] == "raw" or is_algorithmic_raw:
            row["method_mse"] = canonical_raw_mse
            row["method_mae"] = canonical_raw_mae
            row["delta_mse_pct"] = 0.0
            row["gain_pct"] = 0.0
        else:
            delta = 100.0 * (float(row["method_mse"]) - canonical_raw_mse) / canonical_raw_mse
            row["delta_mse_pct"] = delta
            row["gain_pct"] = -delta
    return rows


def run_one(payload: dict[str, Any]) -> dict[str, Any]:
    config = nb.BackboneConfig(**payload["config"])
    key = config_key(config)
    out_dir = Path(payload["out_dir"])
    marker = out_dir / "queue_status" / f"{key.replace('|', '__')}.json"
    row_csv = out_dir / "rows" / f"{key.replace('|', '__')}.csv"
    previous_status: dict[str, Any] = {}
    if marker.exists():
        try:
            previous_status = json.loads(marker.read_text(encoding="utf-8"))
            if previous_status.get("status") == "completed" and row_csv.exists() and not payload["force"]:
                return {"config_key": key, "status": "skipped_completed", "rows": pd.read_csv(row_csv).to_dict("records"), "failures": []}
        except Exception:
            previous_status = {}
    previous_attempt = int(previous_status.get("attempt", 1 if previous_status.get("status") == "failed" else 0))
    if previous_status.get("status") == "failed" and previous_attempt >= 2 and not payload["force"]:
        failure = dict(previous_status)
        failure["retry_exhausted"] = True
        return {"config_key": key, "status": "failed_retry_exhausted", "rows": [], "failures": [failure]}
    attempt = previous_attempt + 1
    started = time.time()
    atomic_json(marker, {"config_key": key, "status": "running", "attempt": attempt, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    try:
        val_arrays = nb.collect_arrays(config, "val", force=False)
        test_arrays = nb.collect_arrays(config, "test", force=False)
        expected = payload["raw_expected"]
        measured_raw_mse, measured_raw_mae = base.metrics_chunked(
            np.asarray(test_arrays["preds"], dtype=np.float32),
            np.asarray(test_arrays["trues"], dtype=np.float32),
        )
        canonical_raw_mse = float(expected["raw_mse"])
        canonical_raw_mae = float(expected["raw_mae"])
        atol = float(payload["raw_match_atol"])
        rtol = float(payload["raw_match_rtol"])
        mse_match = bool(np.isclose(measured_raw_mse, canonical_raw_mse, atol=atol, rtol=rtol))
        mae_match = bool(np.isclose(measured_raw_mae, canonical_raw_mae, atol=atol, rtol=rtol))
        if not (mse_match and mae_match):
            raise ValueError(
                "Raw source mismatch for "
                f"{key}: reexport mse/mae={measured_raw_mse:.12g}/{measured_raw_mae:.12g}, "
                f"canonical={canonical_raw_mse:.12g}/{canonical_raw_mae:.12g}, "
                f"atol={atol:g}, rtol={rtol:g}"
            )
        rows = compute_rows(config, val_arrays, test_arrays)
        rows = align_rows_to_canonical_raw(
            rows,
            canonical_raw_mse=canonical_raw_mse,
            canonical_raw_mae=canonical_raw_mae,
            measured_raw_mse=measured_raw_mse,
            measured_raw_mae=measured_raw_mae,
        )
        row_csv.parent.mkdir(parents=True, exist_ok=True)
        tmp_csv = row_csv.with_suffix(row_csv.suffix + ".tmp")
        pd.DataFrame(rows).to_csv(tmp_csv, index=False)
        tmp_csv.replace(row_csv)
        nb.cleanup_export_dir(config)
        atomic_json(marker, {"config_key": key, "status": "completed", "attempt": attempt, "rows": len(rows), "export_dir_cleaned": True, "raw_alignment": "passed", "raw_mse_abs_error": abs(measured_raw_mse - canonical_raw_mse), "raw_mae_abs_error": abs(measured_raw_mae - canonical_raw_mae), "elapsed_sec": round(time.time() - started, 3), "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        return {"config_key": key, "status": "completed", "rows": rows, "failures": []}
    except Exception as exc:
        try:
            nb.cleanup_export_dir(config)
        except Exception:
            pass
        failure = {
            "config_key": key,
            "dataset": display_dataset(config.dataset),
            "backbone": config.model,
            "pred_len": config.pred_len,
            "seed": config.seed,
            "status": "failed",
            "attempt": attempt,
            "fallback_type": "technical_failure",
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json(marker, {**failure, "elapsed_sec": round(time.time() - started, 3), "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        return {"config_key": key, "status": "failed", "rows": [], "failures": [failure]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CF-MSRC strict internal-only streaming runner from frozen checkpoints.")
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--raw-metrics-csv", default="outputs/raw_metrics_cache/raw_metrics.csv")
    p.add_argument("--raw-match-atol", type=float, default=1e-6)
    p.add_argument("--raw-match-rtol", type=float, default=1e-6)
    p.add_argument("--temp-export-root", default="outputs/internal_full504_exports")
    p.add_argument("--out-dir", default="outputs/internal_only_search/cf_msrc_full504_stream_v1/candidates")
    p.add_argument("--models", nargs="*", default=BACKBONES)
    p.add_argument("--datasets", nargs="*", default=DATASETS)
    p.add_argument("--pred-lens", nargs="*", type=int, default=PRED_LENS)
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--gpu-ids", nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cf_params = cf_params_from_env()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        nb.BackboneConfig(model=m, dataset=d, pred_len=int(pl), seed=int(seed))
        for m in args.models
        for d in args.datasets
        for pl in args.pred_lens
        for seed in args.seeds
    ]
    raw_df = pd.read_csv(args.raw_metrics_csv)
    required_raw_columns = {"model", "dataset", "pred_len", "seed", "raw_mse", "raw_mae"}
    missing_raw_columns = sorted(required_raw_columns - set(raw_df.columns))
    if missing_raw_columns:
        raise ValueError(f"raw metrics missing columns: {missing_raw_columns}")
    raw_lookup: dict[tuple[str, str, int, int], dict[str, float]] = {}
    for row in raw_df.to_dict("records"):
        raw_key = (str(row["model"]), display_dataset(str(row["dataset"])), int(row["pred_len"]), int(row["seed"]))
        if raw_key in raw_lookup:
            raise ValueError(f"duplicate raw metrics key: {raw_key}")
        raw_lookup[raw_key] = {"raw_mse": float(row["raw_mse"]), "raw_mae": float(row["raw_mae"])}
    payloads = []
    for c in configs:
        raw_key = (c.model, display_dataset(c.dataset), int(c.pred_len), int(c.seed))
        if raw_key not in raw_lookup:
            raise ValueError(f"raw metrics missing requested config: {raw_key}")
        payloads.append(
            {
                "config": asdict(c),
                "out_dir": str(out_dir),
                "force": bool(args.force),
                "raw_expected": raw_lookup[raw_key],
                "raw_match_atol": float(args.raw_match_atol),
                "raw_match_rtol": float(args.raw_match_rtol),
            }
        )
    atomic_json(
        out_dir / "status.json",
        {
            "status": "running",
            "total": len(payloads),
            "methods": CF_METHODS,
            "cf_params": cf_params,
            "raw_metrics_csv": str(Path(args.raw_metrics_csv).resolve()),
            "raw_match_atol": float(args.raw_match_atol),
            "raw_match_rtol": float(args.raw_match_rtol),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.time()
    with mp.Pool(processes=max(1, int(args.workers)), initializer=worker_init, initargs=(args.temp_export_root, args.checkpoint_root, args.gpu_ids)) as pool:
        for idx, result in enumerate(pool.imap_unordered(run_one, payloads), start=1):
            all_rows.extend(result["rows"])
            failures.extend(result["failures"])
            pd.DataFrame(all_rows).to_csv(out_dir / "config_level_detail.partial.csv", index=False)
            pd.DataFrame(failures).to_csv(out_dir / "failure_log.partial.csv", index=False)
            atomic_json(
                out_dir / "status.json",
                {
                    "status": "running",
                    "completed_or_seen": idx,
                    "total": len(payloads),
                    "detail_rows": len(all_rows),
                    "failed_configs": len(failures),
                    "last_config": result["config_key"],
                    "last_status": result["status"],
                    "elapsed_sec": round(time.time() - started, 3),
                    "methods": CF_METHODS,
                    "cf_params": cf_params,
                },
            )
            print(f"[{idx}/{len(payloads)}] {result['config_key']} {result['status']} failures={len(failures)}", flush=True)
    detail = pd.DataFrame(all_rows)
    failure_df = pd.DataFrame(failures)
    detail.to_csv(out_dir / "config_level_detail.csv", index=False)
    failure_df.to_csv(out_dir / "failure_log.csv", index=False)
    if not detail.empty:
        strict.summarize(detail).to_csv(out_dir / "summary.csv", index=False)
    atomic_json(
        out_dir / "experiment_manifest.json",
        {
            "script": str(Path(__file__).relative_to(ROOT)),
            "protocol": "cf_msrc_internal_only_stream_from_checkpoints_then_delete_exports",
            "checkpoint_root": str(Path(args.checkpoint_root).resolve()),
            "temp_export_root": str(Path(args.temp_export_root).resolve()),
            "out_dir": str(out_dir.resolve()),
            "workers": int(args.workers),
            "gpu_ids": args.gpu_ids,
            "candidate_pool": CF_METHODS,
            "cf_params": cf_params,
            "raw_metrics_csv": str(Path(args.raw_metrics_csv).resolve()),
            "raw_match_atol": float(args.raw_match_atol),
            "raw_match_rtol": float(args.raw_match_rtol),
            "uses_external_outputs": False,
            "uses_test_for_selection": False,
            "selection_protocol": "validation_fit_tune_route_horizon_bin",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    atomic_json(
        out_dir / "status.json",
        {
            "status": "completed",
            "total": len(payloads),
            "detail_rows": len(detail),
            "failed_configs": len(failures),
            "elapsed_sec": round(time.time() - started, 3),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "methods": CF_METHODS,
            "cf_params": cf_params,
        },
    )
    print(strict.summarize(detail).to_string(index=False) if not detail.empty else "no detail rows")


if __name__ == "__main__":
    main()
