from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Traffic", "Weather"]
BACKBONES = ["DLinear", "PatchTST", "iTransformer", "MICN", "OLS", "FreTS"]
PRED_LENS = [96, 192, 336, 720]
SEEDS = [2, 42, 2021]
NAIVE_BLEND_METHODS = {
    "naive_last_blend",
    "naive_mean_blend",
    "naive_trend_blend",
    "naive_seasonal12_blend",
    "naive_seasonal24_blend",
    "naive_seasonal48_blend",
    "naive_seasonal96_blend",
    "naive_seasonal24_x2_blend",
    "naive_seasonal96_x2_blend",
}
NAIVE_HGAMMA_BLEND_METHODS = {
    "naive_seasonal96_hgamma_blend",
}
NAIVE_HC_GAMMA_BLEND_METHODS = {
    "naive_seasonal96_hc_lsgamma_blend",
}
NAIVE_FITBIAS_BLEND_METHODS = {
    "naive_seasonal96_fitbias_blend",
}
NAIVE_ONLINE_RESID_METHODS = {
    "naive_seasonal96_hgamma_causal_resid",
}
INPUT_AFFINE_METHODS = {
    "input_affine_mean_hgamma_blend",
    "input_affine_std_hgamma_blend",
}
LOWRANK_METHODS = {
    "lowrank_origin_ridge",
}

DATASET_ALIASES = {
    "etth1": "ETTh1",
    "etth2": "ETTh2",
    "ettm1": "ETTm1",
    "ettm2": "ETTm2",
    "electricity": "Electricity",
    "traffic": "Traffic",
    "weather": "Weather",
}

BACKBONE_ALIASES = {
    "dlinear": "DLinear",
    "patchtst": "PatchTST",
    "itransformer": "iTransformer",
    "micn": "MICN",
    "ols": "OLS",
    "frets": "FreTS",
}


@dataclass(frozen=True)
class ExportPair:
    dataset: str
    backbone: str
    pred_len: int
    seed: int
    val_path: Path
    test_path: Path

    @property
    def config_key(self) -> str:
        return f"{self.backbone}|{self.dataset}|{self.pred_len}|{self.seed}"


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def parse_export_path(path: Path) -> tuple[str, str, int, int, str] | None:
    parts = list(path.parts)
    split = path.stem.lower()
    if split not in {"val", "test"}:
        return None

    for i, part in enumerate(parts):
        lower = part.lower()
        if not lower.startswith("seed"):
            continue
        try:
            seed = int(lower.replace("seed", ""))
        except ValueError:
            continue
        if i + 3 >= len(parts):
            continue
        backbone = BACKBONE_ALIASES.get(parts[i + 1].lower())
        dataset = DATASET_ALIASES.get(parts[i + 2].lower())
        pred_part = parts[i + 3].lower()
        if not backbone or not dataset or not pred_part.startswith("pl"):
            continue
        try:
            pred_len = int(pred_part.replace("pl", ""))
        except ValueError:
            continue
        if dataset in DATASETS and backbone in BACKBONES and pred_len in PRED_LENS and seed in SEEDS:
            return dataset, backbone, pred_len, seed, split
    return None


def discover_exports(search_roots: Iterable[Path]) -> tuple[list[ExportPair], pd.DataFrame]:
    candidates: dict[tuple[str, str, int, int], dict[str, Path]] = {}
    source_rows = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.npz"):
            parsed = parse_export_path(path)
            if parsed is None:
                continue
            dataset, backbone, pred_len, seed, split = parsed
            key = (dataset, backbone, pred_len, seed)
            current = candidates.setdefault(key, {})
            # Prefer shorter paths, then newer files. This avoids duplicate historical exports
            # while keeping the selection deterministic enough for auditing.
            old = current.get(split)
            if old is None:
                current[split] = path
            else:
                old_score = (len(old.parts), -old.stat().st_mtime)
                new_score = (len(path.parts), -path.stat().st_mtime)
                if new_score < old_score:
                    current[split] = path
            source_rows.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "pred_len": pred_len,
                    "seed": seed,
                    "split": split,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                }
            )

    pairs = []
    for (dataset, backbone, pred_len, seed), splits in sorted(candidates.items()):
        if "val" in splits and "test" in splits:
            pairs.append(
                ExportPair(
                    dataset=dataset,
                    backbone=backbone,
                    pred_len=pred_len,
                    seed=seed,
                    val_path=splits["val"],
                    test_path=splits["test"],
                )
            )
    return pairs, pd.DataFrame(source_rows)


def load_npz(path: Path, load_inputs: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    data = np.load(path, allow_pickle=False)
    preds = np.asarray(data["preds"]).astype(np.float32, copy=False)
    trues = np.asarray(data["trues"]).astype(np.float32, copy=False)
    inputs = (
        np.asarray(data["inputs"]).astype(np.float32, copy=False)
        if load_inputs and "inputs" in data.files
        else None
    )
    if preds.shape != trues.shape:
        raise ValueError(f"preds/trues shape mismatch in {path}: {preds.shape} vs {trues.shape}")
    return preds, trues, inputs


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.square(a - b)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def metrics_chunked(
    preds: np.ndarray,
    trues: np.ndarray,
    corr: np.ndarray | None = None,
    gamma: float = 0.0,
    clip_value: float | None = None,
    chunk_size: int = 64,
) -> tuple[float, float]:
    """Compute MSE/MAE without materializing the full corrected prediction."""
    sq_sum = 0.0
    abs_sum = 0.0
    count = 0
    if corr is not None and clip_value is not None and np.isfinite(clip_value):
        corr = np.clip(corr, -clip_value, clip_value)
    # Large Electricity/Traffic tensors can exceed memory if 512 origins are
    # squared in float64 at once. Cap chunks by element count instead.
    per_origin = int(np.prod(preds.shape[1:])) if preds.ndim >= 2 else 1
    max_elements = 8_000_000
    if per_origin > 0:
        chunk_size = max(1, min(int(chunk_size), max_elements // per_origin))
    for start in range(0, preds.shape[0], chunk_size):
        end = min(start + chunk_size, preds.shape[0])
        if corr is None or gamma == 0.0:
            err = (preds[start:end] - trues[start:end]).astype(np.float32, copy=False)
        else:
            corr_chunk = corr[start:end] if corr.shape[0] == preds.shape[0] else corr
            err = (preds[start:end] + gamma * corr_chunk - trues[start:end]).astype(np.float32, copy=False)
        abs_sum += float(np.sum(np.abs(err), dtype=np.float64))
        np.multiply(err, err, out=err)
        sq_sum += float(np.sum(err, dtype=np.float64))
        count += int(err.size)
    return sq_sum / count, abs_sum / count


def metrics_hgamma_chunked(
    preds: np.ndarray,
    trues: np.ndarray,
    corr: np.ndarray,
    gamma_h: np.ndarray,
    clip_value: float | None = None,
    chunk_size: int = 64,
) -> tuple[float, float]:
    """Compute MSE/MAE for horizon-wise gamma without materializing outputs."""
    gamma_h = np.asarray(gamma_h, dtype=np.float32).reshape(1, -1, 1)
    sq_sum = 0.0
    abs_sum = 0.0
    count = 0
    per_origin = int(np.prod(preds.shape[1:])) if preds.ndim >= 2 else 1
    max_elements = 8_000_000
    if per_origin > 0:
        chunk_size = max(1, min(int(chunk_size), max_elements // per_origin))
    for start in range(0, preds.shape[0], chunk_size):
        end = min(start + chunk_size, preds.shape[0])
        corr_chunk = corr[start:end]
        if clip_value is not None and np.isfinite(clip_value):
            corr_chunk = np.clip(corr_chunk, -clip_value, clip_value)
        err = (preds[start:end] + gamma_h * corr_chunk - trues[start:end]).astype(np.float32, copy=False)
        abs_sum += float(np.sum(np.abs(err), dtype=np.float64))
        np.multiply(err, err, out=err)
        sq_sum += float(np.sum(err, dtype=np.float64))
        count += int(err.size)
    return sq_sum / count, abs_sum / count


def metrics_gamma_array_chunked(
    preds: np.ndarray,
    trues: np.ndarray,
    corr: np.ndarray,
    gamma: np.ndarray,
    clip_value: float | None = None,
    chunk_size: int = 64,
) -> tuple[float, float]:
    """Compute MSE/MAE for horizon or horizon-channel gamma arrays."""
    gamma_arr = np.asarray(gamma, dtype=np.float32)
    if gamma_arr.ndim == 1:
        gamma_arr = gamma_arr.reshape(1, -1, 1)
    elif gamma_arr.ndim == 2:
        gamma_arr = gamma_arr.reshape(1, gamma_arr.shape[0], gamma_arr.shape[1])
    else:
        raise ValueError(f"gamma must be [H] or [H,C], got shape {gamma_arr.shape}")
    sq_sum = 0.0
    abs_sum = 0.0
    count = 0
    per_origin = int(np.prod(preds.shape[1:])) if preds.ndim >= 2 else 1
    max_elements = 8_000_000
    if per_origin > 0:
        chunk_size = max(1, min(int(chunk_size), max_elements // per_origin))
    for start in range(0, preds.shape[0], chunk_size):
        end = min(start + chunk_size, preds.shape[0])
        corr_chunk = corr[start:end]
        if clip_value is not None and np.isfinite(clip_value):
            corr_chunk = np.clip(corr_chunk, -clip_value, clip_value)
        err = (preds[start:end] + gamma_arr * corr_chunk - trues[start:end]).astype(np.float32, copy=False)
        abs_sum += float(np.sum(np.abs(err), dtype=np.float64))
        np.multiply(err, err, out=err)
        sq_sum += float(np.sum(err, dtype=np.float64))
        count += int(err.size)
    return sq_sum / count, abs_sum / count


def correction_norm_chunked(
    corr: np.ndarray,
    clip_value: float | None = None,
    chunk_size: int = 64,
) -> tuple[float, float]:
    abs_sum = 0.0
    count = 0
    max_abs = 0.0
    per_origin = int(np.prod(corr.shape[1:])) if corr.ndim >= 2 else 1
    max_elements = 8_000_000
    if per_origin > 0:
        chunk_size = max(1, min(int(chunk_size), max_elements // per_origin))
    for start in range(0, corr.shape[0], chunk_size):
        end = min(start + chunk_size, corr.shape[0])
        chunk = corr[start:end]
        if clip_value is not None and np.isfinite(clip_value):
            chunk = np.clip(chunk, -clip_value, clip_value)
        abs_chunk = np.abs(chunk)
        abs_sum += float(np.sum(abs_chunk, dtype=np.float64))
        count += int(abs_chunk.size)
        max_abs = max(max_abs, float(np.max(abs_chunk)) if abs_chunk.size else 0.0)
    return abs_sum / max(count, 1), max_abs


def block_gain_stats_from_origin_mse(
    raw_origin_mse: np.ndarray,
    method_origin_mse: np.ndarray,
    num_blocks: int = 4,
) -> dict:
    raw_origin_mse = np.asarray(raw_origin_mse, dtype=np.float64)
    method_origin_mse = np.asarray(method_origin_mse, dtype=np.float64)
    gains: list[float] = []
    for idx in np.array_split(np.arange(len(raw_origin_mse)), num_blocks):
        if len(idx) == 0:
            continue
        raw_mse = float(np.mean(raw_origin_mse[idx]))
        method_mse = float(np.mean(method_origin_mse[idx]))
        gain = (raw_mse - method_mse) / raw_mse * 100.0 if raw_mse > 0 else 0.0
        gains.append(gain)
    gains_arr = np.asarray(gains, dtype=np.float64)
    mean_gain = float(np.mean(gains_arr)) if len(gains_arr) else 0.0
    std_gain = float(np.std(gains_arr, ddof=1)) if len(gains_arr) > 1 else 0.0
    lcb = mean_gain - 1.96 * std_gain / math.sqrt(max(len(gains_arr), 1))
    return {
        "val_block_count": int(len(gains_arr)),
        "val_block_mean_gain_pct": mean_gain,
        "val_block_min_gain_pct": float(np.min(gains_arr)) if len(gains_arr) else 0.0,
        "val_block_positive_count": int((gains_arr > 0).sum()) if len(gains_arr) else 0,
        "val_block_lcb_gain_pct": float(lcb),
        "val_block_gains_json": json.dumps([round(float(x), 6) for x in gains_arr]),
    }


def validation_block_stats(
    preds: np.ndarray,
    trues: np.ndarray,
    corr: np.ndarray | None = None,
    gamma: float = 0.0,
    clip_value: float | None = None,
    num_blocks: int = 4,
) -> dict:
    if corr is not None and clip_value is not None and np.isfinite(clip_value):
        corr = np.clip(corr, -clip_value, clip_value)
    raw_sq = np.mean(np.square(preds - trues, dtype=np.float64), axis=(1, 2))
    if corr is None or gamma == 0.0:
        method_sq = raw_sq.copy()
    else:
        method_sq = np.mean(np.square(preds + gamma * corr - trues, dtype=np.float64), axis=(1, 2))
    return block_gain_stats_from_origin_mse(raw_sq, method_sq, num_blocks=num_blocks)


def validation_block_stats_hgamma(
    preds: np.ndarray,
    trues: np.ndarray,
    corr: np.ndarray,
    gamma_h: np.ndarray,
    clip_value: float | None = None,
    num_blocks: int = 4,
    chunk_size: int = 64,
) -> dict:
    gamma_h = np.asarray(gamma_h, dtype=np.float32).reshape(1, -1, 1)
    raw_parts: list[np.ndarray] = []
    method_parts: list[np.ndarray] = []
    per_origin = int(np.prod(preds.shape[1:])) if preds.ndim >= 2 else 1
    max_elements = 8_000_000
    if per_origin > 0:
        chunk_size = max(1, min(int(chunk_size), max_elements // per_origin))
    for start in range(0, preds.shape[0], chunk_size):
        end = min(start + chunk_size, preds.shape[0])
        corr_chunk = corr[start:end]
        if clip_value is not None and np.isfinite(clip_value):
            corr_chunk = np.clip(corr_chunk, -clip_value, clip_value)
        raw_err = (preds[start:end] - trues[start:end]).astype(np.float32, copy=False)
        method_err = (preds[start:end] + gamma_h * corr_chunk - trues[start:end]).astype(np.float32, copy=False)
        raw_parts.append(np.mean(np.square(raw_err, dtype=np.float64), axis=(1, 2)))
        method_parts.append(np.mean(np.square(method_err, dtype=np.float64), axis=(1, 2)))
    return block_gain_stats_from_origin_mse(
        np.concatenate(raw_parts) if raw_parts else np.asarray([]),
        np.concatenate(method_parts) if method_parts else np.asarray([]),
        num_blocks=num_blocks,
    )


def validation_block_stats_gamma_array(
    preds: np.ndarray,
    trues: np.ndarray,
    corr: np.ndarray,
    gamma: np.ndarray,
    clip_value: float | None = None,
    num_blocks: int = 4,
    chunk_size: int = 64,
) -> dict:
    gamma_arr = np.asarray(gamma, dtype=np.float32)
    if gamma_arr.ndim == 1:
        gamma_arr = gamma_arr.reshape(1, -1, 1)
    elif gamma_arr.ndim == 2:
        gamma_arr = gamma_arr.reshape(1, gamma_arr.shape[0], gamma_arr.shape[1])
    else:
        raise ValueError(f"gamma must be [H] or [H,C], got shape {gamma_arr.shape}")
    raw_parts: list[np.ndarray] = []
    method_parts: list[np.ndarray] = []
    per_origin = int(np.prod(preds.shape[1:])) if preds.ndim >= 2 else 1
    max_elements = 8_000_000
    if per_origin > 0:
        chunk_size = max(1, min(int(chunk_size), max_elements // per_origin))
    for start in range(0, preds.shape[0], chunk_size):
        end = min(start + chunk_size, preds.shape[0])
        corr_chunk = corr[start:end]
        if clip_value is not None and np.isfinite(clip_value):
            corr_chunk = np.clip(corr_chunk, -clip_value, clip_value)
        raw_err = (preds[start:end] - trues[start:end]).astype(np.float32, copy=False)
        method_err = (preds[start:end] + gamma_arr * corr_chunk - trues[start:end]).astype(np.float32, copy=False)
        raw_parts.append(np.mean(np.square(raw_err, dtype=np.float64), axis=(1, 2)))
        method_parts.append(np.mean(np.square(method_err, dtype=np.float64), axis=(1, 2)))
    return block_gain_stats_from_origin_mse(
        np.concatenate(raw_parts) if raw_parts else np.asarray([]),
        np.concatenate(method_parts) if method_parts else np.asarray([]),
        num_blocks=num_blocks,
    )


def causal_rolling_bias_eval(
    preds: np.ndarray,
    trues: np.ndarray,
    init_bias: np.ndarray,
    gamma: float,
    alpha: float,
    clip_value: float | None = None,
    collect_block_stats: bool = False,
    return_state: bool = False,
) -> tuple[float, float, dict] | tuple[float, float, dict, np.ndarray]:
    """Evaluate a causal matured-label rolling residual-bias corrector.

    Prediction at origin i uses only residuals from earlier origins j whose
    label timestamp j + h + 1 has matured before i. The full `trues` array is
    loaded for offline evaluation, but residuals are revealed to the state only
    through that maturity rule.
    """
    state = init_bias.astype(np.float32, copy=True)
    n, horizon, _channels = preds.shape
    sq_sum = 0.0
    abs_sum = 0.0
    count = 0
    update_events = 0
    max_abs_corr = 0.0
    sum_abs_corr = 0.0
    corr_count = 0
    raw_origin_mse: list[float] = []
    method_origin_mse: list[float] = []

    for origin in range(n):
        matured_horizons = min(horizon, origin)
        if matured_horizons:
            h_idx = np.arange(matured_horizons)
            prev_origins = origin - h_idx - 1
            residual = trues[prev_origins, h_idx, :] - preds[prev_origins, h_idx, :]
            state[:matured_horizons, :] = (
                (1.0 - alpha) * state[:matured_horizons, :] + alpha * residual
            )
            update_events += int(residual.size)

        corr = state
        if clip_value is not None and np.isfinite(clip_value):
            corr = np.clip(corr, -clip_value, clip_value)
        err = preds[origin] + gamma * corr - trues[origin]
        if collect_block_stats:
            raw_err = preds[origin] - trues[origin]
            raw_origin_mse.append(float(np.mean(np.square(raw_err, dtype=np.float64))))
            method_origin_mse.append(float(np.mean(np.square(err, dtype=np.float64))))
        sq_sum += float(np.sum(np.square(err, dtype=np.float64)))
        abs_sum += float(np.sum(np.abs(err), dtype=np.float64))
        count += int(err.size)
        abs_corr = np.abs(corr)
        sum_abs_corr += float(np.sum(abs_corr, dtype=np.float64))
        corr_count += int(abs_corr.size)
        max_abs_corr = max(max_abs_corr, float(np.max(abs_corr)))

    meta = {
        "update_count": update_events,
        "correction_norm_mean": sum_abs_corr / max(corr_count, 1),
        "correction_norm_max": max_abs_corr,
        **(
            block_gain_stats_from_origin_mse(np.asarray(raw_origin_mse), np.asarray(method_origin_mse))
            if collect_block_stats
            else {}
        ),
    }
    result = (sq_sum / count, abs_sum / count, meta)
    if return_state:
        return (*result, state)
    return result


def split_validation(preds: np.ndarray, trues: np.ndarray, inputs: np.ndarray | None) -> tuple:
    n = preds.shape[0]
    cut = max(1, min(n - 1, n // 2))
    return (
        preds[:cut],
        trues[:cut],
        None if inputs is None else inputs[:cut],
        preds[cut:],
        trues[cut:],
        None if inputs is None else inputs[cut:],
    )


def correction_bias_hc(fit_preds: np.ndarray, fit_trues: np.ndarray) -> np.ndarray:
    return np.mean(fit_trues - fit_preds, axis=0, keepdims=True)


def initial_bias_hc(fit_preds: np.ndarray, fit_trues: np.ndarray) -> np.ndarray:
    return np.mean(fit_trues - fit_preds, axis=0).astype(np.float32, copy=False)


def correction_ridge_shared(
    fit_preds: np.ndarray,
    fit_trues: np.ndarray,
    fit_inputs: np.ndarray | None,
    target_preds: np.ndarray,
    target_inputs: np.ndarray | None,
    ridge: float = 1e-3,
) -> np.ndarray:
    if fit_inputs is None or target_inputs is None:
        return np.zeros_like(target_preds)
    fit_last = fit_inputs[:, -1:, :]
    target_last = target_inputs[:, -1:, :]
    x1 = fit_preds
    x2 = np.broadcast_to(fit_last, fit_preds.shape)
    x3 = x1 - x2
    y = fit_trues - fit_preds

    features = [np.ones_like(x1), x1, x2, x3]
    xtx = np.empty((4, 4), dtype=np.float64)
    xty = np.empty((4,), dtype=np.float64)
    for i, fi in enumerate(features):
        xty[i] = np.sum(fi * y)
        for j, fj in enumerate(features):
            xtx[i, j] = np.sum(fi * fj)
    xtx += ridge * np.eye(4)
    beta = np.linalg.solve(xtx, xty)

    tx1 = target_preds
    tx2 = np.broadcast_to(target_last, target_preds.shape)
    tx3 = tx1 - tx2
    return beta[0] + beta[1] * tx1 + beta[2] * tx2 + beta[3] * tx3


def correction_ridge_horizon_shared(
    fit_preds: np.ndarray,
    fit_trues: np.ndarray,
    fit_inputs: np.ndarray | None,
    target_preds: np.ndarray,
    target_inputs: np.ndarray | None,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Fit one 4-feature residual model per horizon, shared over channels.

    This is more expressive than `ridge_shared` but still avoids per-config or
    per-channel lookup rules. Coefficients are fitted only on validation_fit.
    """
    if fit_inputs is None or target_inputs is None:
        return np.zeros_like(target_preds)

    fit_last = fit_inputs[:, -1, :]
    target_last = target_inputs[:, -1, :]
    horizon = fit_preds.shape[1]
    out = np.empty_like(target_preds, dtype=np.float32)

    eye = ridge * np.eye(4, dtype=np.float64)
    for h in range(horizon):
        x1 = fit_preds[:, h, :].reshape(-1).astype(np.float64, copy=False)
        x2 = fit_last.reshape(-1).astype(np.float64, copy=False)
        x3 = x1 - x2
        y = (fit_trues[:, h, :] - fit_preds[:, h, :]).reshape(-1).astype(np.float64, copy=False)
        feats = [np.ones_like(x1), x1, x2, x3]
        xtx = np.empty((4, 4), dtype=np.float64)
        xty = np.empty((4,), dtype=np.float64)
        for i, fi in enumerate(feats):
            xty[i] = np.sum(fi * y)
            for j, fj in enumerate(feats):
                xtx[i, j] = np.sum(fi * fj)
        beta = np.linalg.solve(xtx + eye, xty)

        tx1 = target_preds[:, h, :].astype(np.float64, copy=False)
        tx2 = target_last.astype(np.float64, copy=False)
        tx3 = tx1 - tx2
        out[:, h, :] = (beta[0] + beta[1] * tx1 + beta[2] * tx2 + beta[3] * tx3).astype(np.float32)
    return out


def _enriched_horizon_feature_matrix(
    preds: np.ndarray,
    inputs: np.ndarray,
    horizon_idx: int,
) -> list[np.ndarray]:
    length = inputs.shape[1]
    horizon = preds.shape[1]
    pred_h = preds[:, horizon_idx, :].astype(np.float64, copy=False)
    pred_0 = preds[:, 0, :].astype(np.float64, copy=False)
    last = inputs[:, -1, :].astype(np.float64, copy=False)
    first = inputs[:, 0, :].astype(np.float64, copy=False)
    hist_mean = np.mean(inputs, axis=1, dtype=np.float64)
    hist_slope = (last - first) / max(length - 1, 1)
    if length >= 96:
        seasonal = inputs[:, length - 96 + (horizon_idx % 96), :].astype(np.float64, copy=False)
    else:
        seasonal = last
    h_norm = np.full_like(pred_h, horizon_idx / max(horizon - 1, 1), dtype=np.float64)
    return [
        np.ones_like(pred_h),
        pred_h,
        last,
        pred_h - last,
        hist_mean,
        last - hist_mean,
        hist_slope,
        seasonal,
        pred_h - seasonal,
        pred_h - pred_0,
        h_norm,
    ]


def correction_ridge_enriched_horizon_shared(
    fit_preds: np.ndarray,
    fit_trues: np.ndarray,
    fit_inputs: np.ndarray | None,
    target_preds: np.ndarray,
    target_inputs: np.ndarray | None,
    ridge: float = 1e-2,
) -> np.ndarray:
    """Per-horizon shared ridge with forecast/history/seasonal anchor features."""
    if fit_inputs is None or target_inputs is None:
        return np.zeros_like(target_preds)

    horizon = fit_preds.shape[1]
    out = np.empty_like(target_preds, dtype=np.float32)
    n_features = 11
    eye = ridge * np.eye(n_features, dtype=np.float64)
    for h in range(horizon):
        fit_feats_2d = [f.reshape(-1) for f in _enriched_horizon_feature_matrix(fit_preds, fit_inputs, h)]
        y = (fit_trues[:, h, :] - fit_preds[:, h, :]).reshape(-1).astype(np.float64, copy=False)
        xtx = np.empty((n_features, n_features), dtype=np.float64)
        xty = np.empty((n_features,), dtype=np.float64)
        for i, fi in enumerate(fit_feats_2d):
            xty[i] = np.sum(fi * y)
            for j, fj in enumerate(fit_feats_2d):
                xtx[i, j] = np.sum(fi * fj)
        beta = np.linalg.solve(xtx + eye, xty)
        target_feats = _enriched_horizon_feature_matrix(target_preds, target_inputs, h)
        corr = np.zeros_like(target_preds[:, h, :], dtype=np.float64)
        for coef, feat in zip(beta, target_feats):
            corr += coef * feat
        out[:, h, :] = corr.astype(np.float32)
    return out


def _origin_level_features(preds: np.ndarray, inputs: np.ndarray | None) -> np.ndarray:
    """Observable per-origin features for low-rank residual coefficient models."""
    pred = preds.astype(np.float64, copy=False)
    feats = [
        np.ones((pred.shape[0],), dtype=np.float64),
        np.mean(pred, axis=(1, 2), dtype=np.float64),
        np.std(pred, axis=(1, 2), dtype=np.float64),
        np.mean(pred[:, -1, :] - pred[:, 0, :], axis=1, dtype=np.float64),
        np.std(pred[:, -1, :] - pred[:, 0, :], axis=1, dtype=np.float64),
    ]
    if inputs is not None:
        hist = inputs.astype(np.float64, copy=False)
        last = hist[:, -1, :]
        first = hist[:, 0, :]
        hist_mean = np.mean(hist, axis=1, dtype=np.float64)
        feats.extend(
            [
                np.mean(last, axis=1, dtype=np.float64),
                np.std(last, axis=1, dtype=np.float64),
                np.mean(hist_mean, axis=1, dtype=np.float64),
                np.std(hist_mean, axis=1, dtype=np.float64),
                np.mean(last - first, axis=1, dtype=np.float64),
                np.std(last - first, axis=1, dtype=np.float64),
                np.mean(pred[:, 0, :] - last, axis=1, dtype=np.float64),
                np.std(pred[:, 0, :] - last, axis=1, dtype=np.float64),
            ]
        )
    return np.stack(feats, axis=1)


def _standardize_features(
    fit_features: np.ndarray,
    target_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(fit_features, axis=0, keepdims=True)
    std = np.std(fit_features, axis=0, keepdims=True)
    # Keep the intercept column intact.
    std = np.where(std < 1e-8, 1.0, std)
    std[:, 0] = 1.0
    mean[:, 0] = 0.0
    return (fit_features - mean) / std, (target_features - mean) / std, mean, std


def _fit_lowrank_origin_ridge(
    fit_preds: np.ndarray,
    fit_trues: np.ndarray,
    fit_inputs: np.ndarray | None,
    target_preds: np.ndarray,
    target_inputs: np.ndarray | None,
    rank: int,
    ridge: float,
    max_dim: int = 50_000,
) -> tuple[np.ndarray, dict]:
    """Predict residuals through an origin-level ridge model in SVD residual space.

    This is intentionally not a dataset/backbone lookup. It uses only validation_fit
    labels to learn a compact residual basis and maps observable per-origin raw/input
    statistics to residual coefficients.
    """
    n_fit, horizon, channels = fit_preds.shape
    dim = horizon * channels
    if dim > max_dim:
        return np.zeros_like(target_preds, dtype=np.float32), {
            "lowrank_status": "fallback_dim_guard",
            "lowrank_dim": int(dim),
            "lowrank_rank": 0,
            "lowrank_ridge": ridge,
        }

    fit_resid = (fit_trues - fit_preds).reshape(n_fit, dim).astype(np.float64, copy=False)
    resid_mean = np.mean(fit_resid, axis=0, keepdims=True)
    centered = fit_resid - resid_mean
    max_rank = max(0, min(rank, n_fit - 1, dim))
    target_n = target_preds.shape[0]

    fit_feat = _origin_level_features(fit_preds, fit_inputs)
    target_feat = _origin_level_features(target_preds, target_inputs)
    fit_feat, target_feat, _feat_mean, _feat_std = _standardize_features(fit_feat, target_feat)

    if max_rank == 0:
        corr = np.broadcast_to(resid_mean.reshape(1, horizon, channels), target_preds.shape)
        return corr.astype(np.float32, copy=False), {
            "lowrank_status": "mean_residual_only",
            "lowrank_dim": int(dim),
            "lowrank_rank": 0,
            "lowrank_ridge": ridge,
        }

    # Full SVD is acceptable for local low-dimensional exploratory scopes. A
    # dimension guard prevents accidental huge Electricity/Traffic materialization.
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:max_rank]
    coeff = centered @ basis.T
    xtx = fit_feat.T @ fit_feat
    xty = fit_feat.T @ coeff
    beta = np.linalg.solve(xtx + ridge * np.eye(xtx.shape[0], dtype=np.float64), xty)
    pred_coeff = target_feat @ beta
    corr_flat = resid_mean + pred_coeff @ basis
    corr = corr_flat.reshape(target_n, horizon, channels)
    return corr.astype(np.float32, copy=False), {
        "lowrank_status": "ok",
        "lowrank_dim": int(dim),
        "lowrank_rank": int(max_rank),
        "lowrank_ridge": float(ridge),
    }


def select_lowrank_origin_ridge_and_eval(
    fit_preds: np.ndarray,
    fit_trues: np.ndarray,
    fit_inputs: np.ndarray | None,
    sel_preds: np.ndarray,
    sel_trues: np.ndarray,
    sel_inputs: np.ndarray | None,
    test_preds: np.ndarray,
    test_trues: np.ndarray,
    test_inputs: np.ndarray | None,
) -> tuple[float, float, dict]:
    """Validation-selected low-rank origin-level residual correction."""
    rank_options = [0, 1, 2, 4, 8]
    ridge_options = [1e-2, 1e-1, 1.0, 10.0]
    gamma_options = [0.0, 0.25, 0.50, 0.75, 1.0]
    clip_options = [None, 0.90, 0.95]
    raw_sel_mse, _ = metrics_chunked(sel_preds, sel_trues)
    fit_resid_abs = np.abs(fit_trues - fit_preds)
    best = {
        "val_select_raw_mse": raw_sel_mse,
        "val_select_mse": raw_sel_mse,
        "rank": 0,
        "ridge": np.nan,
        "gamma": 0.0,
        "clip_q": None,
        "clip_value": np.nan,
        "fallback": True,
        "fallback_reason": "validation_selected_raw",
        "lowrank_status": "not_selected",
        "lowrank_dim": int(fit_preds.shape[1] * fit_preds.shape[2]),
    }
    best_corr_sel: np.ndarray | None = None
    for rank in rank_options:
        for ridge in ridge_options:
            corr_sel, lowrank_meta = _fit_lowrank_origin_ridge(
                fit_preds, fit_trues, fit_inputs, sel_preds, sel_inputs, rank=rank, ridge=ridge
            )
            if lowrank_meta["lowrank_status"] == "fallback_dim_guard":
                continue
            for clip_q in clip_options:
                clip_value = np.nan
                if clip_q is not None:
                    clip_value = float(np.quantile(fit_resid_abs, clip_q))
                for gamma in gamma_options:
                    val_mse, _ = metrics_chunked(
                        sel_preds,
                        sel_trues,
                        corr=corr_sel,
                        gamma=gamma,
                        clip_value=clip_value if clip_q is not None else None,
                    )
                    if val_mse < best["val_select_mse"]:
                        best = {
                            "val_select_raw_mse": raw_sel_mse,
                            "val_select_mse": val_mse,
                            "rank": rank,
                            "ridge": ridge,
                            "gamma": gamma,
                            "clip_q": clip_q,
                            "clip_value": clip_value,
                            "fallback": gamma == 0.0,
                            "fallback_reason": "none" if gamma > 0 else "validation_selected_raw",
                            **lowrank_meta,
                        }
                        best_corr_sel = corr_sel

    if best["fallback"]:
        best["val_select_gain_pct"] = 0.0
        best.update(validation_block_stats(sel_preds, sel_trues))
        best["correction_norm_mean"] = 0.0
        best["correction_norm_max"] = 0.0
        return metrics_chunked(test_preds, test_trues)[0], metrics_chunked(test_preds, test_trues)[1], best

    corr_test, lowrank_meta = _fit_lowrank_origin_ridge(
        fit_preds,
        fit_trues,
        fit_inputs,
        test_preds,
        test_inputs,
        rank=int(best["rank"]),
        ridge=float(best["ridge"]),
    )
    if lowrank_meta["lowrank_status"] == "fallback_dim_guard":
        best.update(lowrank_meta)
        best["fallback"] = True
        best["fallback_reason"] = "dimension_guard"
        best["val_select_gain_pct"] = 0.0
        best["correction_norm_mean"] = 0.0
        best["correction_norm_max"] = 0.0
        return metrics_chunked(test_preds, test_trues)[0], metrics_chunked(test_preds, test_trues)[1], best

    method_mse, method_mae = metrics_chunked(
        test_preds,
        test_trues,
        corr=corr_test,
        gamma=float(best["gamma"]),
        clip_value=best["clip_value"] if best["clip_q"] is not None else None,
    )
    best["val_select_gain_pct"] = (
        (best["val_select_raw_mse"] - best["val_select_mse"]) / best["val_select_raw_mse"] * 100.0
        if best["val_select_raw_mse"] > 0
        else np.nan
    )
    if best_corr_sel is not None:
        best.update(
            validation_block_stats(
                sel_preds,
                sel_trues,
                corr=best_corr_sel,
                gamma=float(best["gamma"]),
                clip_value=best["clip_value"] if best["clip_q"] is not None else None,
            )
        )
    norm_mean, norm_max = correction_norm_chunked(
        corr_test,
        clip_value=best["clip_value"] if best["clip_q"] is not None else None,
    )
    best["correction_norm_mean"] = float(norm_mean)
    best["correction_norm_max"] = float(norm_max)
    best["fallback_reason"] = "none"
    return method_mse, method_mae, best


def correction_seasonal_bias(
    fit_preds: np.ndarray,
    fit_trues: np.ndarray,
    target_shape: tuple[int, int, int],
    period: int,
    target_offset: int,
) -> np.ndarray:
    """Estimate phase-conditioned residual bias from validation_fit only.

    The period is selected later on validation_select. No dataset identity or
    test metric is used. `target_offset` keeps the phase index causal in the
    concatenated validation/test timeline.
    """
    if period <= 1:
        return np.broadcast_to(correction_bias_hc(fit_preds, fit_trues), target_shape).astype(np.float32, copy=False)

    resid = (fit_trues - fit_preds).astype(np.float32, copy=False)
    global_bias = np.mean(resid, axis=0).astype(np.float32, copy=False)
    n_fit, horizon, channels = resid.shape
    phase_bias = np.empty((period, horizon, channels), dtype=np.float32)
    phases = np.arange(n_fit) % period
    for phase in range(period):
        mask = phases == phase
        if np.any(mask):
            phase_bias[phase] = np.mean(resid[mask], axis=0)
        else:
            phase_bias[phase] = global_bias

    n_target = target_shape[0]
    target_phases = (np.arange(target_offset, target_offset + n_target) % period).astype(int)
    return phase_bias[target_phases]


def naive_forecast_from_inputs(inputs: np.ndarray, horizon: int, kind: str) -> np.ndarray:
    """Build an internal-only naive forecast from the observed input window."""
    if inputs is None:
        raise ValueError(f"{kind} requires inputs")
    if inputs.ndim != 3:
        raise ValueError(f"expected inputs with shape [N, L, C], got {inputs.shape}")
    n, length, channels = inputs.shape
    if length <= 0:
        raise ValueError("input window is empty")
    last = inputs[:, -1:, :].astype(np.float32, copy=False)
    if kind == "naive_last_blend":
        return np.broadcast_to(last, (n, horizon, channels)).astype(np.float32, copy=False)
    if kind == "naive_mean_blend":
        mean = np.mean(inputs, axis=1, keepdims=True, dtype=np.float32)
        return np.broadcast_to(mean, (n, horizon, channels)).astype(np.float32, copy=False)
    if kind == "naive_trend_blend":
        first = inputs[:, :1, :].astype(np.float32, copy=False)
        slope = (last - first) / max(length - 1, 1)
        steps = np.arange(1, horizon + 1, dtype=np.float32)[None, :, None]
        return (last + steps * slope).astype(np.float32, copy=False)
    if kind.startswith("naive_seasonal") and (
        kind.endswith("_blend")
        or kind.endswith("_hgamma_blend")
        or kind.endswith("_hc_lsgamma_blend")
        or kind.endswith("_fitbias_blend")
    ):
        if kind.endswith("_fitbias_blend"):
            period_text = kind[len("naive_seasonal") : -len("_fitbias_blend")]
        elif kind.endswith("_hc_lsgamma_blend"):
            period_text = kind[len("naive_seasonal") : -len("_hc_lsgamma_blend")]
        elif kind.endswith("_hgamma_blend"):
            period_text = kind[len("naive_seasonal") : -len("_hgamma_blend")]
        else:
            period_text = kind[len("naive_seasonal") : -len("_blend")]
        if period_text.endswith("_x2"):
            period_text = period_text[: -len("_x2")]
        period = int(period_text)
        if period <= 0:
            raise ValueError(f"invalid seasonal period: {period}")
        if length < period:
            return np.broadcast_to(last, (n, horizon, channels)).astype(np.float32, copy=False)
        phase = np.arange(horizon, dtype=np.int64) % period
        source_idx = length - period + phase
        return inputs[:, source_idx, :].astype(np.float32, copy=False)
    raise ValueError(f"unknown naive forecast kind: {kind}")


def initial_seasonal_state(fit_preds: np.ndarray, fit_trues: np.ndarray, period: int) -> np.ndarray:
    if period <= 1:
        return initial_bias_hc(fit_preds, fit_trues)[None, :, :]
    resid = (fit_trues - fit_preds).astype(np.float32, copy=False)
    global_bias = np.mean(resid, axis=0).astype(np.float32, copy=False)
    n_fit, horizon, channels = resid.shape
    state = np.empty((period, horizon, channels), dtype=np.float32)
    phases = np.arange(n_fit) % period
    for phase in range(period):
        mask = phases == phase
        state[phase] = np.mean(resid[mask], axis=0) if np.any(mask) else global_bias
    return state


def causal_seasonal_rolling_bias_eval(
    preds: np.ndarray,
    trues: np.ndarray,
    init_state: np.ndarray,
    period: int,
    target_offset: int,
    gamma: float,
    alpha: float,
    clip_value: float | None = None,
    collect_block_stats: bool = False,
    return_state: bool = False,
) -> tuple[float, float, dict] | tuple[float, float, dict, np.ndarray]:
    """Causal phase-conditioned residual-bias corrector.

    The state is indexed by absolute origin phase and horizon. At origin i,
    prediction uses only the current phase state. Updates use residuals from
    previous origins whose labels have matured within the same segment.
    """
    state = init_state.astype(np.float32, copy=True)
    n, horizon, _channels = preds.shape
    sq_sum = 0.0
    abs_sum = 0.0
    count = 0
    update_events = 0
    max_abs_corr = 0.0
    sum_abs_corr = 0.0
    corr_count = 0
    raw_origin_mse: list[float] = []
    method_origin_mse: list[float] = []

    for origin in range(n):
        matured_horizons = min(horizon, origin)
        if matured_horizons:
            h_idx = np.arange(matured_horizons)
            prev_origins = origin - h_idx - 1
            prev_phases = ((target_offset + prev_origins) % period).astype(int)
            residual = trues[prev_origins, h_idx, :] - preds[prev_origins, h_idx, :]
            state[prev_phases, h_idx, :] = (
                (1.0 - alpha) * state[prev_phases, h_idx, :] + alpha * residual
            )
            update_events += int(residual.size)

        phase = int((target_offset + origin) % period)
        corr = state[phase]
        if clip_value is not None and np.isfinite(clip_value):
            corr = np.clip(corr, -clip_value, clip_value)
        err = preds[origin] + gamma * corr - trues[origin]
        if collect_block_stats:
            raw_err = preds[origin] - trues[origin]
            raw_origin_mse.append(float(np.mean(np.square(raw_err, dtype=np.float64))))
            method_origin_mse.append(float(np.mean(np.square(err, dtype=np.float64))))
        sq_sum += float(np.sum(np.square(err, dtype=np.float64)))
        abs_sum += float(np.sum(np.abs(err), dtype=np.float64))
        count += int(err.size)
        abs_corr = np.abs(corr)
        sum_abs_corr += float(np.sum(abs_corr, dtype=np.float64))
        corr_count += int(abs_corr.size)
        max_abs_corr = max(max_abs_corr, float(np.max(abs_corr)))

    meta = {
        "update_count": update_events,
        "correction_norm_mean": sum_abs_corr / max(corr_count, 1),
        "correction_norm_max": max_abs_corr,
        **(
            block_gain_stats_from_origin_mse(np.asarray(raw_origin_mse), np.asarray(method_origin_mse))
            if collect_block_stats
            else {}
        ),
    }
    result = (sq_sum / count, abs_sum / count, meta)
    if return_state:
        return (*result, state)
    return result


def select_and_apply(
    method_id: str,
    fit_preds: np.ndarray,
    fit_trues: np.ndarray,
    fit_inputs: np.ndarray | None,
    sel_preds: np.ndarray,
    sel_trues: np.ndarray,
    sel_inputs: np.ndarray | None,
    test_preds: np.ndarray,
    test_inputs: np.ndarray | None,
) -> tuple[np.ndarray, dict]:
    if method_id == "bias_hc":
        corr_sel = np.broadcast_to(correction_bias_hc(fit_preds, fit_trues), sel_preds.shape)
        corr_test = np.broadcast_to(correction_bias_hc(fit_preds, fit_trues), test_preds.shape)
    elif method_id == "ridge_shared":
        corr_sel = correction_ridge_shared(fit_preds, fit_trues, fit_inputs, sel_preds, sel_inputs)
        corr_test = correction_ridge_shared(fit_preds, fit_trues, fit_inputs, test_preds, test_inputs)
    elif method_id == "ridge_horizon_shared":
        corr_sel = correction_ridge_horizon_shared(fit_preds, fit_trues, fit_inputs, sel_preds, sel_inputs)
        corr_test = correction_ridge_horizon_shared(fit_preds, fit_trues, fit_inputs, test_preds, test_inputs)
    elif method_id == "ridge_enriched_horizon_shared":
        corr_sel = correction_ridge_enriched_horizon_shared(
            fit_preds, fit_trues, fit_inputs, sel_preds, sel_inputs
        )
        corr_test = correction_ridge_enriched_horizon_shared(
            fit_preds, fit_trues, fit_inputs, test_preds, test_inputs
        )
    elif method_id == "seasonal_bias":
        # `select_seasonal_bias_and_eval` provides the memory-efficient path.
        raise ValueError("seasonal_bias must be evaluated through select_seasonal_bias_and_eval")
    else:
        raise ValueError(f"unknown method_id: {method_id}")

    fit_resid_abs = np.abs(fit_trues - fit_preds)
    clip_options = [None, 0.90, 0.95]
    gamma_options = [0.0, 0.25, 0.50, 0.75, 1.0]
    raw_sel_mse = mse(sel_preds, sel_trues)
    best = {
        "val_select_raw_mse": raw_sel_mse,
        "val_select_mse": raw_sel_mse,
        "gamma": 0.0,
        "clip_q": None,
        "clip_value": np.nan,
        "fallback": True,
    }
    for gamma in gamma_options:
        for clip_q in clip_options:
            clip_value = np.nan
            use_corr = corr_sel
            if clip_q is not None:
                clip_value = float(np.quantile(fit_resid_abs, clip_q))
                use_corr = np.clip(use_corr, -clip_value, clip_value)
            pred = sel_preds + gamma * use_corr
            val_mse = mse(pred, sel_trues)
            if val_mse < best["val_select_mse"]:
                best = {
                    "val_select_raw_mse": raw_sel_mse,
                    "val_select_mse": val_mse,
                    "gamma": gamma,
                    "clip_q": clip_q,
                    "clip_value": clip_value,
                    "fallback": gamma == 0.0,
                }

    final_corr = corr_test
    if best["clip_q"] is not None:
        final_corr = np.clip(final_corr, -best["clip_value"], best["clip_value"])
    final_pred = test_preds + best["gamma"] * final_corr
    best["val_select_gain_pct"] = (
        (best["val_select_raw_mse"] - best["val_select_mse"]) / best["val_select_raw_mse"] * 100.0
        if best["val_select_raw_mse"] > 0
        else np.nan
    )
    best.update(
        validation_block_stats(
            sel_preds,
            sel_trues,
            corr=corr_sel,
            gamma=best["gamma"],
            clip_value=best["clip_value"] if best["clip_q"] is not None else None,
        )
    )
    best["correction_norm_mean"] = float(np.mean(np.abs(final_corr)))
    best["correction_norm_max"] = float(np.max(np.abs(final_corr)))
    return final_pred, best


def select_naive_blend_and_eval(
    method_id: str,
    fit_preds: np.ndarray,
    fit_trues: np.ndarray,
    fit_inputs: np.ndarray | None,
    sel_preds: np.ndarray,
    sel_trues: np.ndarray,
    sel_inputs: np.ndarray | None,
    test_preds: np.ndarray,
    test_trues: np.ndarray,
    test_inputs: np.ndarray | None,
) -> tuple[float, float, dict]:
    """Blend raw forecasts toward an input-only naive forecast selected on validation."""
    if fit_inputs is None or sel_inputs is None or test_inputs is None:
        raise ValueError(f"{method_id} requires inputs in val/test exports")
    fit_alt = naive_forecast_from_inputs(fit_inputs, fit_preds.shape[1], method_id)
    sel_alt = naive_forecast_from_inputs(sel_inputs, sel_preds.shape[1], method_id)
    test_alt = naive_forecast_from_inputs(test_inputs, test_preds.shape[1], method_id)
    corr_sel = (sel_alt - sel_preds).astype(np.float32, copy=False)
    corr_test = (test_alt - test_preds).astype(np.float32, copy=False)
    fit_corr_abs = np.abs(fit_alt - fit_preds)
    clip_options = [None, 0.90, 0.95]
    gamma_options = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
    if "_x2_" in method_id:
        gamma_options = [*gamma_options, 1.25, 1.50, 2.0]
    raw_sel_mse, _ = metrics_chunked(sel_preds, sel_trues)
    best = {
        "val_select_raw_mse": raw_sel_mse,
        "val_select_mse": raw_sel_mse,
        "gamma": 0.0,
        "clip_q": None,
        "clip_value": np.nan,
        "fallback": True,
        "fallback_reason": "validation_selected_raw",
    }
    for gamma in gamma_options:
        for clip_q in clip_options:
            clip_value = np.nan
            if clip_q is not None:
                clip_value = float(np.quantile(fit_corr_abs, clip_q))
            val_mse, _ = metrics_chunked(
                sel_preds,
                sel_trues,
                corr=corr_sel,
                gamma=gamma,
                clip_value=clip_value if clip_q is not None else None,
            )
            if val_mse < best["val_select_mse"]:
                best = {
                    "val_select_raw_mse": raw_sel_mse,
                    "val_select_mse": val_mse,
                    "gamma": gamma,
                    "clip_q": clip_q,
                    "clip_value": clip_value,
                    "fallback": gamma == 0.0,
                    "fallback_reason": "none" if gamma > 0 else "validation_selected_raw",
                }
    method_mse, method_mae = metrics_chunked(
        test_preds,
        test_trues,
        corr=corr_test,
        gamma=best["gamma"],
        clip_value=best["clip_value"] if best["clip_q"] is not None else None,
    )
    corr_eval = corr_test
    if best["clip_q"] is not None:
        corr_eval = np.clip(corr_eval, -best["clip_value"], best["clip_value"])
    best["val_select_gain_pct"] = (
        (best["val_select_raw_mse"] - best["val_select_mse"]) / best["val_select_raw_mse"] * 100.0
        if best["val_select_raw_mse"] > 0
        else np.nan
    )
    best.update(
        validation_block_stats(
            sel_preds,
            sel_trues,
            corr=corr_sel,
            gamma=best["gamma"],
            clip_value=best["clip_value"] if best["clip_q"] is not None else None,
        )
    )
    best["correction_norm_mean"] = float(np.mean(np.abs(corr_eval)))
    best["correction_norm_max"] = float(np.max(np.abs(corr_eval)))
    best["naive_kind"] = method_id
    return method_mse, method_mae, best


def select_naive_hgamma_blend_and_eval(
    method_id: str,
    fit_preds: np.ndarray,
    fit_trues: np.ndarray,
    fit_inputs: np.ndarray | None,
    sel_preds: np.ndarray,
    sel_trues: np.ndarray,
    sel_inputs: np.ndarray | None,
    test_preds: np.ndarray,
    test_trues: np.ndarray,
    test_inputs: np.ndarray | None,
) -> tuple[float, float, dict]:
    """Input-only seasonal naive blend with validation-selected horizon-wise gamma.

    This remains a single unified formula. The only extra flexibility versus
    `naive_seasonal96_blend` is that validation_select can abstain or shrink
    each prediction horizon independently.
    """
    if fit_inputs is None or sel_inputs is None or test_inputs is None:
        raise ValueError(f"{method_id} requires inputs in val/test exports")
    fit_alt = naive_forecast_from_inputs(fit_inputs, fit_preds.shape[1], method_id)
    sel_alt = naive_forecast_from_inputs(sel_inputs, sel_preds.shape[1], method_id)
    test_alt = naive_forecast_from_inputs(test_inputs, test_preds.shape[1], method_id)
    corr_sel = (sel_alt - sel_preds).astype(np.float32, copy=False)
    corr_test = (test_alt - test_preds).astype(np.float32, copy=False)
    fit_corr_abs = np.abs(fit_alt - fit_preds)
    clip_options = [None, 0.90, 0.95]
    gamma_options = np.asarray([0.0, 0.10, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
    raw_sel_mse, _ = metrics_chunked(sel_preds, sel_trues)
    horizon = int(sel_preds.shape[1])
    best = {
        "val_select_raw_mse": raw_sel_mse,
        "val_select_mse": raw_sel_mse,
        "gamma": 0.0,
        "gamma_mean": 0.0,
        "gamma_active_horizon_frac": 0.0,
        "gamma_json": json.dumps([0.0 for _ in range(horizon)]),
        "clip_q": None,
        "clip_value": np.nan,
        "fallback": True,
        "fallback_reason": "validation_selected_raw_all_horizons",
    }
    for clip_q in clip_options:
        clip_value = np.nan
        corr_for_select = corr_sel
        if clip_q is not None:
            clip_value = float(np.quantile(fit_corr_abs, clip_q))
            corr_for_select = np.clip(corr_sel, -clip_value, clip_value)

        gamma_h = np.zeros(horizon, dtype=np.float32)
        selected_h_mse = np.empty(horizon, dtype=np.float64)
        for h in range(horizon):
            raw_h_mse = float(
                np.mean(np.square(sel_preds[:, h, :] - sel_trues[:, h, :], dtype=np.float64))
            )
            best_h_mse = raw_h_mse
            best_h_gamma = 0.0
            corr_h = corr_for_select[:, h, :]
            for gamma in gamma_options[1:]:
                err = sel_preds[:, h, :] + float(gamma) * corr_h - sel_trues[:, h, :]
                val_h_mse = float(np.mean(np.square(err, dtype=np.float64)))
                if val_h_mse < best_h_mse:
                    best_h_mse = val_h_mse
                    best_h_gamma = float(gamma)
            gamma_h[h] = best_h_gamma
            selected_h_mse[h] = best_h_mse

        val_mse = float(np.mean(selected_h_mse))
        if val_mse < best["val_select_mse"]:
            active_frac = float(np.mean(gamma_h > 0))
            best = {
                "val_select_raw_mse": raw_sel_mse,
                "val_select_mse": val_mse,
                "gamma": float(np.mean(gamma_h)),
                "gamma_mean": float(np.mean(gamma_h)),
                "gamma_active_horizon_frac": active_frac,
                "gamma_json": json.dumps([round(float(x), 4) for x in gamma_h]),
                "clip_q": clip_q,
                "clip_value": clip_value,
                "fallback": active_frac == 0.0,
                "fallback_reason": "none" if active_frac > 0.0 else "validation_selected_raw_all_horizons",
            }

    gamma_h = np.asarray(json.loads(best["gamma_json"]), dtype=np.float32)
    method_mse, method_mae = metrics_hgamma_chunked(
        test_preds,
        test_trues,
        corr_test,
        gamma_h=gamma_h,
        clip_value=best["clip_value"] if best["clip_q"] is not None else None,
    )
    best["val_select_gain_pct"] = (
        (best["val_select_raw_mse"] - best["val_select_mse"]) / best["val_select_raw_mse"] * 100.0
        if best["val_select_raw_mse"] > 0
        else np.nan
    )
    best.update(
        validation_block_stats_hgamma(
            sel_preds,
            sel_trues,
            corr_sel,
            gamma_h=gamma_h,
            clip_value=best["clip_value"] if best["clip_q"] is not None else None,
        )
    )
    norm_mean, norm_max = correction_norm_chunked(
        corr_test,
        clip_value=best["clip_value"] if best["clip_q"] is not None else None,
    )
    best["correction_norm_mean"] = float(norm_mean)
    best["correction_norm_max"] = float(norm_max)
    best["naive_kind"] = method_id
    best["horizon_gamma_policy"] = "validation_select_per_horizon"
    return method_mse, method_mae, best


def input_affine_correction_chunk(preds: np.ndarray, inputs: np.ndarray, kind: str) -> np.ndarray:
    """Return an input-statistics affine anchor correction for one chunk.

    The anchor is internal-only: it uses the current input window and raw
    forecast shape, but no labels. `mean` shifts the forecast level to the
    history mean; `std` additionally aligns the forecast horizon scale to the
    history scale with a conservative clipping range.
    """
    if inputs is None:
        raise ValueError("input_affine methods require inputs in the export npz")
    hist_mean = np.mean(inputs, axis=1, keepdims=True, dtype=np.float32)
    pred_mean = np.mean(preds, axis=1, keepdims=True, dtype=np.float32)
    centered = preds - pred_mean
    if kind == "mean":
        anchor = centered + hist_mean
    elif kind == "std":
        hist_std = np.std(inputs, axis=1, keepdims=True, dtype=np.float32)
        pred_std = np.std(preds, axis=1, keepdims=True, dtype=np.float32)
        scale = np.clip(hist_std / (pred_std + 1e-6), 0.50, 2.00).astype(np.float32, copy=False)
        anchor = centered * scale + hist_mean
    else:
        raise ValueError(f"unknown input affine kind: {kind}")
    return (anchor - preds).astype(np.float32, copy=False)


def input_affine_correction_horizon(preds: np.ndarray, inputs: np.ndarray, horizon_idx: int, kind: str) -> np.ndarray:
    """Return correction for one horizon while computing stats on the full forecast window."""
    if inputs is None:
        raise ValueError("input_affine methods require inputs in the export npz")
    hist_mean = np.mean(inputs, axis=1, keepdims=True, dtype=np.float32)
    pred_mean = np.mean(preds, axis=1, keepdims=True, dtype=np.float32)
    pred_h = preds[:, horizon_idx : horizon_idx + 1, :]
    centered_h = pred_h - pred_mean
    if kind == "mean":
        anchor_h = centered_h + hist_mean
    elif kind == "std":
        hist_std = np.std(inputs, axis=1, keepdims=True, dtype=np.float32)
        pred_std = np.std(preds, axis=1, keepdims=True, dtype=np.float32)
        scale = np.clip(hist_std / (pred_std + 1e-6), 0.50, 2.00).astype(np.float32, copy=False)
        anchor_h = centered_h * scale + hist_mean
    else:
        raise ValueError(f"unknown input affine kind: {kind}")
    return (anchor_h[:, 0, :] - preds[:, horizon_idx, :]).astype(np.float32, copy=False)


def metrics_input_affine_hgamma_chunked(
    preds: np.ndarray,
    trues: np.ndarray,
    inputs: np.ndarray,
    kind: str,
    gamma_h: np.ndarray,
    clip_value: float | None = None,
    chunk_size: int = 64,
) -> tuple[float, float]:
    gamma_h = np.asarray(gamma_h, dtype=np.float32).reshape(1, -1, 1)
    sq_sum = 0.0
    abs_sum = 0.0
    count = 0
    per_origin = int(np.prod(preds.shape[1:])) if preds.ndim >= 2 else 1
    max_elements = 8_000_000
    if per_origin > 0:
        chunk_size = max(1, min(int(chunk_size), max_elements // per_origin))
    for start in range(0, preds.shape[0], chunk_size):
        end = min(start + chunk_size, preds.shape[0])
        corr = input_affine_correction_chunk(preds[start:end], inputs[start:end], kind=kind)
        if clip_value is not None and np.isfinite(clip_value):
            corr = np.clip(corr, -clip_value, clip_value)
        err = (preds[start:end] + gamma_h * corr - trues[start:end]).astype(np.float32, copy=False)
        abs_sum += float(np.sum(np.abs(err), dtype=np.float64))
        np.multiply(err, err, out=err)
        sq_sum += float(np.sum(err, dtype=np.float64))
        count += int(err.size)
    return sq_sum / count, abs_sum / count


def validation_block_stats_input_affine_hgamma(
    preds: np.ndarray,
    trues: np.ndarray,
    inputs: np.ndarray,
    kind: str,
    gamma_h: np.ndarray,
    clip_value: float | None = None,
    num_blocks: int = 4,
    chunk_size: int = 64,
) -> dict:
    gamma_h = np.asarray(gamma_h, dtype=np.float32).reshape(1, -1, 1)
    raw_origin = np.empty(preds.shape[0], dtype=np.float64)
    method_origin = np.empty(preds.shape[0], dtype=np.float64)
    per_origin = int(np.prod(preds.shape[1:])) if preds.ndim >= 2 else 1
    max_elements = 8_000_000
    if per_origin > 0:
        chunk_size = max(1, min(int(chunk_size), max_elements // per_origin))
    for start in range(0, preds.shape[0], chunk_size):
        end = min(start + chunk_size, preds.shape[0])
        raw_err = (preds[start:end] - trues[start:end]).astype(np.float32, copy=False)
        raw_origin[start:end] = np.mean(np.square(raw_err, dtype=np.float64), axis=(1, 2))
        corr = input_affine_correction_chunk(preds[start:end], inputs[start:end], kind=kind)
        if clip_value is not None and np.isfinite(clip_value):
            corr = np.clip(corr, -clip_value, clip_value)
        method_err = (preds[start:end] + gamma_h * corr - trues[start:end]).astype(np.float32, copy=False)
        method_origin[start:end] = np.mean(np.square(method_err, dtype=np.float64), axis=(1, 2))
    return block_gain_stats_from_origin_mse(raw_origin, method_origin, num_blocks=num_blocks)


def select_input_affine_hgamma_blend_and_eval(
    method_id: str,
    fit_preds: np.ndarray,
    fit_trues: np.ndarray,
    fit_inputs: np.ndarray | None,
    sel_preds: np.ndarray,
    sel_trues: np.ndarray,
    sel_inputs: np.ndarray | None,
    test_preds: np.ndarray,
    test_trues: np.ndarray,
    test_inputs: np.ndarray | None,
) -> tuple[float, float, dict]:
    if fit_inputs is None or sel_inputs is None or test_inputs is None:
        raise ValueError("input_affine methods require inputs in val/test npz files")
    kind = "std" if "_std_" in method_id else "mean"
    horizon = sel_preds.shape[1]
    gamma_options = np.asarray([0.0, 0.10, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
    clip_options: list[float | None] = [None, 0.90, 0.95]
    raw_sel_mse, _raw_sel_mae = metrics_chunked(sel_preds, sel_trues)
    best = {
        "val_select_raw_mse": raw_sel_mse,
        "val_select_mse": raw_sel_mse,
        "val_select_gain_pct": 0.0,
        "gamma": 0.0,
        "gamma_mean": 0.0,
        "gamma_active_horizon_frac": 0.0,
        "gamma_json": json.dumps([0.0 for _ in range(horizon)]),
        "clip_q": np.nan,
        "clip_value": np.nan,
        "fallback": True,
        "fallback_reason": "validation_selected_raw",
        "input_affine_kind": kind,
        "horizon_gamma_policy": "validation_select_per_horizon",
    }

    # Precompute correction magnitudes on validation-select only for clipping.
    abs_samples = []
    for start in range(0, sel_preds.shape[0], 64):
        end = min(start + 64, sel_preds.shape[0])
        corr = input_affine_correction_chunk(sel_preds[start:end], sel_inputs[start:end], kind=kind)
        abs_samples.append(np.abs(corr).reshape(-1))
    abs_corr = np.concatenate(abs_samples) if abs_samples else np.asarray([], dtype=np.float32)
    for clip_q in clip_options:
        clip_value = None
        if clip_q is not None and abs_corr.size:
            clip_value = float(np.quantile(abs_corr, float(clip_q)))
        gamma_h = np.zeros(horizon, dtype=np.float32)
        for h in range(horizon):
            corr_h = input_affine_correction_horizon(sel_preds, sel_inputs, h, kind=kind)
            if clip_value is not None and np.isfinite(clip_value):
                corr_h = np.clip(corr_h, -clip_value, clip_value)
            raw_h_mse = float(np.mean(np.square(sel_preds[:, h, :] - sel_trues[:, h, :])))
            best_h_mse = raw_h_mse
            best_h_gamma = 0.0
            for gamma in gamma_options[1:]:
                err = sel_preds[:, h, :] + float(gamma) * corr_h - sel_trues[:, h, :]
                mse_h = float(np.mean(np.square(err)))
                if mse_h < best_h_mse:
                    best_h_mse = mse_h
                    best_h_gamma = float(gamma)
            gamma_h[h] = best_h_gamma
        val_mse, _val_mae = metrics_input_affine_hgamma_chunked(
            sel_preds, sel_trues, sel_inputs, kind=kind, gamma_h=gamma_h, clip_value=clip_value
        )
        val_gain = (raw_sel_mse - val_mse) / raw_sel_mse * 100.0 if raw_sel_mse > 0 else np.nan
        if val_mse < float(best["val_select_mse"]):
            active_frac = float(np.mean(gamma_h > 0))
            best.update(
                {
                    "val_select_mse": val_mse,
                    "val_select_gain_pct": val_gain,
                    "gamma": float(np.mean(gamma_h)),
                    "gamma_mean": float(np.mean(gamma_h)),
                    "gamma_active_horizon_frac": active_frac,
                    "gamma_json": json.dumps([round(float(x), 4) for x in gamma_h]),
                    "clip_q": np.nan if clip_q is None else float(clip_q),
                    "clip_value": np.nan if clip_value is None else float(clip_value),
                    "fallback": active_frac == 0.0,
                    "fallback_reason": "none" if active_frac > 0.0 else "validation_selected_raw",
                }
            )

    gamma_h = np.asarray(json.loads(str(best["gamma_json"])), dtype=np.float32)
    clip_value = None if not np.isfinite(float(best["clip_value"])) else float(best["clip_value"])
    method_mse, method_mae = metrics_input_affine_hgamma_chunked(
        test_preds, test_trues, test_inputs, kind=kind, gamma_h=gamma_h, clip_value=clip_value
    )
    if np.any(gamma_h > 0):
        best.update(
            validation_block_stats_input_affine_hgamma(
                sel_preds, sel_trues, sel_inputs, kind=kind, gamma_h=gamma_h, clip_value=clip_value
            )
        )
        # Correction norm is computed on validation-select to avoid storing
        # full test corrections for large Electricity/Traffic tensors.
        norm_sum = 0.0
        norm_count = 0
        norm_max = 0.0
        for start in range(0, sel_preds.shape[0], 64):
            end = min(start + 64, sel_preds.shape[0])
            corr = input_affine_correction_chunk(sel_preds[start:end], sel_inputs[start:end], kind=kind)
            if clip_value is not None:
                corr = np.clip(corr, -clip_value, clip_value)
            abs_corr = np.abs(corr)
            norm_sum += float(np.sum(abs_corr, dtype=np.float64))
            norm_count += int(abs_corr.size)
            norm_max = max(norm_max, float(np.max(abs_corr)) if abs_corr.size else 0.0)
        best["correction_norm_mean"] = norm_sum / max(norm_count, 1)
        best["correction_norm_max"] = norm_max
    else:
        best.update({"correction_norm_mean": 0.0, "correction_norm_max": 0.0})
    return method_mse, method_mae, best


def _select_naive_hgamma_base(
    method_id: str,
    fit_preds: np.ndarray,
    fit_inputs: np.ndarray,
    sel_preds: np.ndarray,
    sel_trues: np.ndarray,
    sel_inputs: np.ndarray,
) -> tuple[np.ndarray, float | None, float, float, float, float]:
    """Return validation-selected horizon gamma and clip for a naive anchor."""
    fit_alt = naive_forecast_from_inputs(fit_inputs, fit_preds.shape[1], method_id)
    sel_alt = naive_forecast_from_inputs(sel_inputs, sel_preds.shape[1], method_id)
    corr_sel = (sel_alt - sel_preds).astype(np.float32, copy=False)
    fit_corr_abs = np.abs(fit_alt - fit_preds)
    clip_options = [None, 0.90, 0.95]
    gamma_options = np.asarray([0.0, 0.10, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
    raw_sel_mse, _ = metrics_chunked(sel_preds, sel_trues)
    horizon = int(sel_preds.shape[1])
    best_gamma = np.zeros(horizon, dtype=np.float32)
    best_clip_q: float | None = None
    best_clip_value = np.nan
    best_val_mse = raw_sel_mse
    for clip_q in clip_options:
        clip_value = np.nan
        corr_for_select = corr_sel
        if clip_q is not None:
            clip_value = float(np.quantile(fit_corr_abs, clip_q))
            corr_for_select = np.clip(corr_sel, -clip_value, clip_value)
        gamma_h = np.zeros(horizon, dtype=np.float32)
        selected_h_mse = np.empty(horizon, dtype=np.float64)
        for h in range(horizon):
            raw_h_mse = float(
                np.mean(np.square(sel_preds[:, h, :] - sel_trues[:, h, :], dtype=np.float64))
            )
            best_h_mse = raw_h_mse
            best_h_gamma = 0.0
            corr_h = corr_for_select[:, h, :]
            for gamma in gamma_options[1:]:
                err = sel_preds[:, h, :] + float(gamma) * corr_h - sel_trues[:, h, :]
                val_h_mse = float(np.mean(np.square(err, dtype=np.float64)))
                if val_h_mse < best_h_mse:
                    best_h_mse = val_h_mse
                    best_h_gamma = float(gamma)
            gamma_h[h] = best_h_gamma
            selected_h_mse[h] = best_h_mse
        val_mse = float(np.mean(selected_h_mse))
        if val_mse < best_val_mse:
            best_gamma = gamma_h
            best_clip_q = clip_q
            best_clip_value = clip_value
            best_val_mse = val_mse
    return best_gamma, best_clip_q, best_clip_value, raw_sel_mse, best_val_mse, float(np.mean(best_gamma > 0))


def apply_naive_hgamma_base(
    method_id: str,
    preds: np.ndarray,
    inputs: np.ndarray,
    gamma_h: np.ndarray,
    clip_value: float | None,
) -> np.ndarray:
    alt = naive_forecast_from_inputs(inputs, preds.shape[1], method_id)
    corr = (alt - preds).astype(np.float32, copy=False)
    if clip_value is not None and np.isfinite(clip_value):
        corr = np.clip(corr, -clip_value, clip_value)
    gamma_arr = np.asarray(gamma_h, dtype=np.float32).reshape(1, -1, 1)
    return (preds + gamma_arr * corr).astype(np.float32, copy=False)


def select_naive_hgamma_causal_resid_and_eval(
    method_id: str,
    fit_preds: np.ndarray,
    fit_trues: np.ndarray,
    fit_inputs: np.ndarray | None,
    sel_preds: np.ndarray,
    sel_trues: np.ndarray,
    sel_inputs: np.ndarray | None,
    test_preds: np.ndarray,
    test_trues: np.ndarray,
    test_inputs: np.ndarray | None,
    rolling_grid: str = "small",
    min_val_gain_pct: float = 0.0,
) -> tuple[float, float, dict]:
    """Seasonal hgamma base plus causal matured residual-bias correction."""
    if fit_inputs is None or sel_inputs is None or test_inputs is None:
        raise ValueError(f"{method_id} requires inputs in val/test exports")
    base_kind = method_id.replace("_hgamma_causal_resid", "_hgamma_blend")
    gamma_h, base_clip_q, base_clip_value, raw_sel_mse, base_sel_mse, active_h_frac = _select_naive_hgamma_base(
        base_kind,
        fit_preds,
        fit_inputs,
        sel_preds,
        sel_trues,
        sel_inputs,
    )
    fit_base = apply_naive_hgamma_base(
        base_kind,
        fit_preds,
        fit_inputs,
        gamma_h,
        base_clip_value if base_clip_q is not None else None,
    )
    sel_base = apply_naive_hgamma_base(
        base_kind,
        sel_preds,
        sel_inputs,
        gamma_h,
        base_clip_value if base_clip_q is not None else None,
    )
    test_base = apply_naive_hgamma_base(
        base_kind,
        test_preds,
        test_inputs,
        gamma_h,
        base_clip_value if base_clip_q is not None else None,
    )
    init_bias = initial_bias_hc(fit_base, fit_trues)
    fit_resid_abs = np.abs(fit_trues - fit_base)
    if rolling_grid == "full":
        residual_gamma_options = [0.0, 0.25, 0.50, 0.75, 1.0]
        alpha_options = [0.01, 0.05, 0.10, 0.20, 0.50, 1.0]
        resid_clip_options = [None, 0.90, 0.95]
    else:
        residual_gamma_options = [0.0, 0.25, 0.50]
        alpha_options = [0.05, 0.20, 0.50, 1.0]
        resid_clip_options = [None, 0.95]

    best = {
        "val_select_raw_mse": raw_sel_mse,
        "val_select_mse": base_sel_mse,
        "base_val_select_mse": base_sel_mse,
        "base_val_select_gain_pct": (raw_sel_mse - base_sel_mse) / raw_sel_mse * 100.0
        if raw_sel_mse > 0
        else np.nan,
        "base_gamma_mean": float(np.mean(gamma_h)),
        "base_gamma_active_horizon_frac": float(active_h_frac),
        "base_clip_q": base_clip_q,
        "base_clip_value": base_clip_value,
        "gamma": 0.0,
        "residual_gamma": 0.0,
        "alpha": 0.0,
        "clip_q": None,
        "clip_value": np.nan,
        "fallback": base_sel_mse >= raw_sel_mse,
        "fallback_reason": "base_validation_not_better_than_raw" if base_sel_mse >= raw_sel_mse else "none",
        "rolling_grid": rolling_grid,
        "validation_update_count": 0,
    }
    for residual_gamma in residual_gamma_options:
        if residual_gamma == 0.0:
            val_mse, _ = metrics_chunked(sel_base, sel_trues)
            if val_mse < best["val_select_mse"]:
                best.update(
                    {
                        "val_select_mse": val_mse,
                        "gamma": 0.0,
                        "residual_gamma": 0.0,
                        "alpha": 0.0,
                        "clip_q": None,
                        "clip_value": np.nan,
                        "fallback": val_mse >= raw_sel_mse,
                        "fallback_reason": "none" if val_mse < raw_sel_mse else "base_validation_not_better_than_raw",
                    }
                )
            continue
        for alpha in alpha_options:
            for clip_q in resid_clip_options:
                clip_value = np.nan
                if clip_q is not None:
                    clip_value = float(np.quantile(fit_resid_abs, clip_q))
                val_mse, _val_mae, meta = causal_rolling_bias_eval(
                    sel_base,
                    sel_trues,
                    init_bias=init_bias,
                    gamma=residual_gamma,
                    alpha=alpha,
                    clip_value=clip_value if clip_q is not None else None,
                )
                if val_mse < best["val_select_mse"]:
                    best.update(
                        {
                            "val_select_mse": val_mse,
                            "gamma": residual_gamma,
                            "residual_gamma": residual_gamma,
                            "alpha": alpha,
                            "clip_q": clip_q,
                            "clip_value": clip_value,
                            "fallback": val_mse >= raw_sel_mse,
                            "fallback_reason": "none"
                            if val_mse < raw_sel_mse
                            else "combined_validation_not_better_than_raw",
                            "validation_update_count": meta["update_count"],
                        }
                    )

    best["val_select_gain_pct"] = (
        (best["val_select_raw_mse"] - best["val_select_mse"]) / best["val_select_raw_mse"] * 100.0
        if best["val_select_raw_mse"] > 0
        else np.nan
    )
    best["min_val_gain_pct"] = float(min_val_gain_pct)
    if best["val_select_gain_pct"] < min_val_gain_pct or best["fallback"]:
        raw_mse, raw_mae = metrics_chunked(test_preds, test_trues)
        best.update(
            {
                "fallback": True,
                "fallback_reason": "val_select_gain_below_threshold"
                if best["val_select_gain_pct"] < min_val_gain_pct
                else best["fallback_reason"],
                "correction_norm_mean": 0.0,
                "correction_norm_max": 0.0,
                "update_count": 0,
            }
        )
        return raw_mse, raw_mae, best

    _val_mse, _val_mae, block_meta, warmed_state = causal_rolling_bias_eval(
        sel_base,
        sel_trues,
        init_bias=init_bias,
        gamma=float(best["residual_gamma"]),
        alpha=float(best["alpha"]),
        clip_value=best["clip_value"] if best["clip_q"] is not None else None,
        collect_block_stats=True,
        return_state=True,
    )
    best.update(block_meta)
    method_mse, method_mae, meta = causal_rolling_bias_eval(
        test_base,
        test_trues,
        init_bias=warmed_state,
        gamma=float(best["residual_gamma"]),
        alpha=float(best["alpha"]),
        clip_value=best["clip_value"] if best["clip_q"] is not None else None,
    )
    best.update(meta)
    best["fallback"] = False
    best["fallback_reason"] = "none"
    best["base_kind"] = base_kind
    best["combined_formula"] = "raw_plus_validation_horizon_gamma_seasonal_anchor_plus_causal_matured_residual_bias"
    return method_mse, method_mae, best


def select_naive_hc_lsgamma_blend_and_eval(
    method_id: str,
    fit_preds: np.ndarray,
    fit_trues: np.ndarray,
    fit_inputs: np.ndarray | None,
    sel_preds: np.ndarray,
    sel_trues: np.ndarray,
    sel_inputs: np.ndarray | None,
    test_preds: np.ndarray,
    test_trues: np.ndarray,
    test_inputs: np.ndarray | None,
) -> tuple[float, float, dict]:
    """Seasonal naive blend with validation-only horizon-channel LS gamma.

    For each (horizon, channel), gamma is estimated from validation_select by
    minimizing squared error for raw + gamma * (naive - raw), then clipped to
    [0, 1]. This is internal-only and does not inspect test performance.
    """
    if fit_inputs is None or sel_inputs is None or test_inputs is None:
        raise ValueError(f"{method_id} requires inputs in val/test exports")
    fit_alt = naive_forecast_from_inputs(fit_inputs, fit_preds.shape[1], method_id)
    sel_alt = naive_forecast_from_inputs(sel_inputs, sel_preds.shape[1], method_id)
    test_alt = naive_forecast_from_inputs(test_inputs, test_preds.shape[1], method_id)
    corr_sel = (sel_alt - sel_preds).astype(np.float32, copy=False)
    corr_test = (test_alt - test_preds).astype(np.float32, copy=False)
    fit_corr_abs = np.abs(fit_alt - fit_preds)
    raw_sel_mse, _ = metrics_chunked(sel_preds, sel_trues)
    clip_options = [None, 0.90, 0.95]
    best = {
        "val_select_raw_mse": raw_sel_mse,
        "val_select_mse": raw_sel_mse,
        "gamma": 0.0,
        "gamma_mean": 0.0,
        "gamma_active_cell_frac": 0.0,
        "gamma_active_horizon_frac": 0.0,
        "gamma_min": 0.0,
        "gamma_max": 0.0,
        "clip_q": None,
        "clip_value": np.nan,
        "fallback": True,
        "fallback_reason": "validation_selected_raw_all_cells",
        "gamma_policy": "validation_select_horizon_channel_lsgamma_clipped_0_1",
    }
    best_gamma = np.zeros(sel_preds.shape[1:], dtype=np.float32)
    resid_sel = (sel_trues - sel_preds).astype(np.float32, copy=False)
    for clip_q in clip_options:
        clip_value = np.nan
        corr_for_select = corr_sel
        if clip_q is not None:
            clip_value = float(np.quantile(fit_corr_abs, clip_q))
            corr_for_select = np.clip(corr_sel, -clip_value, clip_value)
        numerator = np.sum(corr_for_select * resid_sel, axis=0, dtype=np.float64)
        denominator = np.sum(corr_for_select * corr_for_select, axis=0, dtype=np.float64)
        gamma_hc = np.zeros_like(numerator, dtype=np.float32)
        valid = denominator > 1e-12
        gamma_hc[valid] = np.clip(numerator[valid] / denominator[valid], 0.0, 1.0).astype(np.float32)
        val_mse, _ = metrics_gamma_array_chunked(
            sel_preds,
            sel_trues,
            corr_sel,
            gamma=gamma_hc,
            clip_value=clip_value if clip_q is not None else None,
        )
        if val_mse < best["val_select_mse"]:
            active_cells = gamma_hc > 0
            active_horizons = np.any(active_cells, axis=1)
            best_gamma = gamma_hc
            best = {
                "val_select_raw_mse": raw_sel_mse,
                "val_select_mse": val_mse,
                "gamma": float(np.mean(gamma_hc)),
                "gamma_mean": float(np.mean(gamma_hc)),
                "gamma_active_cell_frac": float(np.mean(active_cells)),
                "gamma_active_horizon_frac": float(np.mean(active_horizons)),
                "gamma_min": float(np.min(gamma_hc)),
                "gamma_max": float(np.max(gamma_hc)),
                "clip_q": clip_q,
                "clip_value": clip_value,
                "fallback": not bool(np.any(active_cells)),
                "fallback_reason": "none" if np.any(active_cells) else "validation_selected_raw_all_cells",
                "gamma_policy": "validation_select_horizon_channel_lsgamma_clipped_0_1",
            }

    method_mse, method_mae = metrics_gamma_array_chunked(
        test_preds,
        test_trues,
        corr_test,
        gamma=best_gamma,
        clip_value=best["clip_value"] if best["clip_q"] is not None else None,
    )
    best["val_select_gain_pct"] = (
        (best["val_select_raw_mse"] - best["val_select_mse"]) / best["val_select_raw_mse"] * 100.0
        if best["val_select_raw_mse"] > 0
        else np.nan
    )
    best.update(
        validation_block_stats_gamma_array(
            sel_preds,
            sel_trues,
            corr_sel,
            gamma=best_gamma,
            clip_value=best["clip_value"] if best["clip_q"] is not None else None,
        )
    )
    norm_mean, norm_max = correction_norm_chunked(
        corr_test,
        clip_value=best["clip_value"] if best["clip_q"] is not None else None,
    )
    best["correction_norm_mean"] = float(norm_mean)
    best["correction_norm_max"] = float(norm_max)
    best["naive_kind"] = method_id
    return method_mse, method_mae, best


def select_naive_fitbias_blend_and_eval(
    method_id: str,
    fit_preds: np.ndarray,
    fit_trues: np.ndarray,
    fit_inputs: np.ndarray | None,
    sel_preds: np.ndarray,
    sel_trues: np.ndarray,
    sel_inputs: np.ndarray | None,
    test_preds: np.ndarray,
    test_trues: np.ndarray,
    test_inputs: np.ndarray | None,
) -> tuple[float, float, dict]:
    """Blend raw forecasts toward a validation-fit calibrated seasonal naive anchor."""
    if fit_inputs is None or sel_inputs is None or test_inputs is None:
        raise ValueError(f"{method_id} requires inputs in val/test exports")
    base_kind = method_id.replace("_fitbias", "")
    fit_naive = naive_forecast_from_inputs(fit_inputs, fit_preds.shape[1], base_kind)
    sel_naive = naive_forecast_from_inputs(sel_inputs, sel_preds.shape[1], base_kind)
    test_naive = naive_forecast_from_inputs(test_inputs, test_preds.shape[1], base_kind)
    naive_bias = np.mean(fit_trues - fit_naive, axis=0, dtype=np.float32)
    fit_alt = (fit_naive + naive_bias).astype(np.float32, copy=False)
    sel_alt = (sel_naive + naive_bias).astype(np.float32, copy=False)
    test_alt = (test_naive + naive_bias).astype(np.float32, copy=False)
    corr_sel = (sel_alt - sel_preds).astype(np.float32, copy=False)
    corr_test = (test_alt - test_preds).astype(np.float32, copy=False)
    fit_corr_abs = np.abs(fit_alt - fit_preds)
    clip_options = [None, 0.90, 0.95]
    gamma_options = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
    raw_sel_mse, _ = metrics_chunked(sel_preds, sel_trues)
    best = {
        "val_select_raw_mse": raw_sel_mse,
        "val_select_mse": raw_sel_mse,
        "gamma": 0.0,
        "clip_q": None,
        "clip_value": np.nan,
        "fallback": True,
        "fallback_reason": "validation_selected_raw",
    }
    for gamma in gamma_options:
        for clip_q in clip_options:
            clip_value = np.nan
            if clip_q is not None:
                clip_value = float(np.quantile(fit_corr_abs, clip_q))
            val_mse, _ = metrics_chunked(
                sel_preds,
                sel_trues,
                corr=corr_sel,
                gamma=gamma,
                clip_value=clip_value if clip_q is not None else None,
            )
            if val_mse < best["val_select_mse"]:
                best = {
                    "val_select_raw_mse": raw_sel_mse,
                    "val_select_mse": val_mse,
                    "gamma": gamma,
                    "clip_q": clip_q,
                    "clip_value": clip_value,
                    "fallback": gamma == 0.0,
                    "fallback_reason": "none" if gamma > 0 else "validation_selected_raw",
                }
    method_mse, method_mae = metrics_chunked(
        test_preds,
        test_trues,
        corr=corr_test,
        gamma=best["gamma"],
        clip_value=best["clip_value"] if best["clip_q"] is not None else None,
    )
    norm_mean, norm_max = correction_norm_chunked(
        corr_test,
        clip_value=best["clip_value"] if best["clip_q"] is not None else None,
    )
    best["val_select_gain_pct"] = (
        (best["val_select_raw_mse"] - best["val_select_mse"]) / best["val_select_raw_mse"] * 100.0
        if best["val_select_raw_mse"] > 0
        else np.nan
    )
    best.update(
        validation_block_stats(
            sel_preds,
            sel_trues,
            corr=corr_sel,
            gamma=best["gamma"],
            clip_value=best["clip_value"] if best["clip_q"] is not None else None,
        )
    )
    best["correction_norm_mean"] = float(norm_mean)
    best["correction_norm_max"] = float(norm_max)
    best["naive_kind"] = base_kind
    best["fit_bias_norm_mean"] = float(np.mean(np.abs(naive_bias)))
    best["fit_bias_norm_max"] = float(np.max(np.abs(naive_bias)))
    return method_mse, method_mae, best


def select_bias_hc_and_eval(
    fit_preds: np.ndarray,
    fit_trues: np.ndarray,
    sel_preds: np.ndarray,
    sel_trues: np.ndarray,
    test_preds: np.ndarray,
    test_trues: np.ndarray,
) -> tuple[float, float, dict]:
    corr = correction_bias_hc(fit_preds, fit_trues).astype(np.float32, copy=False)
    fit_resid_abs = np.abs(fit_trues - fit_preds)
    clip_options = [None, 0.90, 0.95]
    gamma_options = [0.0, 0.25, 0.50, 0.75, 1.0]
    raw_sel_mse, _ = metrics_chunked(sel_preds, sel_trues)
    best = {
        "val_select_raw_mse": raw_sel_mse,
        "val_select_mse": raw_sel_mse,
        "gamma": 0.0,
        "clip_q": None,
        "clip_value": np.nan,
        "fallback": True,
    }
    for gamma in gamma_options:
        for clip_q in clip_options:
            clip_value = np.nan
            if clip_q is not None:
                clip_value = float(np.quantile(fit_resid_abs, clip_q))
            val_mse, _ = metrics_chunked(
                sel_preds,
                sel_trues,
                corr=corr,
                gamma=gamma,
                clip_value=clip_value if clip_q is not None else None,
            )
            if val_mse < best["val_select_mse"]:
                best = {
                    "val_select_raw_mse": raw_sel_mse,
                    "val_select_mse": val_mse,
                    "gamma": gamma,
                    "clip_q": clip_q,
                    "clip_value": clip_value,
                    "fallback": gamma == 0.0,
                }
    method_mse, method_mae = metrics_chunked(
        test_preds,
        test_trues,
        corr=corr,
        gamma=best["gamma"],
        clip_value=best["clip_value"] if best["clip_q"] is not None else None,
    )
    corr_eval = corr
    if best["clip_q"] is not None:
        corr_eval = np.clip(corr_eval, -best["clip_value"], best["clip_value"])
    best["val_select_gain_pct"] = (
        (best["val_select_raw_mse"] - best["val_select_mse"]) / best["val_select_raw_mse"] * 100.0
        if best["val_select_raw_mse"] > 0
        else np.nan
    )
    best.update(
        validation_block_stats(
            sel_preds,
            sel_trues,
            corr=corr,
            gamma=best["gamma"],
            clip_value=best["clip_value"] if best["clip_q"] is not None else None,
        )
    )
    best["correction_norm_mean"] = float(np.mean(np.abs(corr_eval)))
    best["correction_norm_max"] = float(np.max(np.abs(corr_eval)))
    return method_mse, method_mae, best


def select_seasonal_bias_and_eval(
    fit_preds: np.ndarray,
    fit_trues: np.ndarray,
    sel_preds: np.ndarray,
    sel_trues: np.ndarray,
    test_preds: np.ndarray,
    test_trues: np.ndarray,
    seasonal_grid: str = "small",
    min_val_gain_pct: float = 0.0,
) -> tuple[float, float, dict]:
    fit_resid_abs = np.abs(fit_trues - fit_preds)
    raw_sel_mse, _ = metrics_chunked(sel_preds, sel_trues)
    if seasonal_grid == "full":
        period_options = [1, 2, 4, 6, 7, 12, 24, 48, 96, 168]
        gamma_options = [0.0, 0.25, 0.50, 0.75, 1.0]
        clip_options = [None, 0.90, 0.95]
    else:
        period_options = [1, 4, 24, 96]
        gamma_options = [0.0, 0.25, 0.50]
        clip_options = [None, 0.95]

    best = {
        "val_select_raw_mse": raw_sel_mse,
        "val_select_mse": raw_sel_mse,
        "gamma": 0.0,
        "clip_q": None,
        "clip_value": np.nan,
        "period": 1,
        "seasonal_grid": seasonal_grid,
        "fallback": True,
    }
    for period in period_options:
        corr_sel = correction_seasonal_bias(
            fit_preds,
            fit_trues,
            sel_preds.shape,
            period=period,
            target_offset=fit_preds.shape[0],
        )
        for gamma in gamma_options:
            if gamma == 0.0:
                continue
            for clip_q in clip_options:
                clip_value = np.nan
                if clip_q is not None:
                    clip_value = float(np.quantile(fit_resid_abs, clip_q))
                val_mse, _ = metrics_chunked(
                    sel_preds,
                    sel_trues,
                    corr=corr_sel,
                    gamma=gamma,
                    clip_value=clip_value if clip_q is not None else None,
                )
                if val_mse < best["val_select_mse"]:
                    best = {
                        "val_select_raw_mse": raw_sel_mse,
                        "val_select_mse": val_mse,
                        "gamma": gamma,
                        "clip_q": clip_q,
                        "clip_value": clip_value,
                        "period": period,
                        "seasonal_grid": seasonal_grid,
                        "fallback": gamma == 0.0,
                    }

    best["val_select_gain_pct"] = (
        (best["val_select_raw_mse"] - best["val_select_mse"]) / best["val_select_raw_mse"] * 100.0
        if best["val_select_raw_mse"] > 0
        else np.nan
    )
    best_corr_sel = correction_seasonal_bias(
        fit_preds,
        fit_trues,
        sel_preds.shape,
        period=int(best["period"]),
        target_offset=fit_preds.shape[0],
    )
    best.update(
        validation_block_stats(
            sel_preds,
            sel_trues,
            corr=best_corr_sel,
            gamma=best["gamma"],
            clip_value=best["clip_value"] if best["clip_q"] is not None else None,
        )
    )
    best["min_val_gain_pct"] = float(min_val_gain_pct)
    if best["val_select_gain_pct"] < min_val_gain_pct:
        raw_mse, raw_mae = metrics_chunked(test_preds, test_trues)
        best.update(
            {
                "gamma": 0.0,
                "fallback": True,
                "fallback_reason": "val_select_gain_below_threshold",
                "correction_norm_mean": 0.0,
                "correction_norm_max": 0.0,
            }
        )
        return raw_mse, raw_mae, best

    corr_test = correction_seasonal_bias(
        fit_preds,
        fit_trues,
        test_preds.shape,
        period=int(best["period"]),
        target_offset=fit_preds.shape[0] + sel_preds.shape[0],
    )
    method_mse, method_mae = metrics_chunked(
        test_preds,
        test_trues,
        corr=corr_test,
        gamma=best["gamma"],
        clip_value=best["clip_value"] if best["clip_q"] is not None else None,
    )
    corr_eval = corr_test
    if best["clip_q"] is not None:
        corr_eval = np.clip(corr_eval, -best["clip_value"], best["clip_value"])
    best["fallback"] = best["gamma"] == 0.0
    best["fallback_reason"] = "none" if best["gamma"] > 0 else "validation_selected_raw"
    best["correction_norm_mean"] = float(np.mean(np.abs(corr_eval)))
    best["correction_norm_max"] = float(np.max(np.abs(corr_eval)))
    return method_mse, method_mae, best


def select_causal_seasonal_rolling_bias_and_eval(
    fit_preds: np.ndarray,
    fit_trues: np.ndarray,
    sel_preds: np.ndarray,
    sel_trues: np.ndarray,
    test_preds: np.ndarray,
    test_trues: np.ndarray,
    seasonal_grid: str = "small",
    rolling_grid: str = "small",
    min_val_gain_pct: float = 0.0,
    gamma_policy: str = "selected",
    gamma_scale_pct: float = 2.0,
) -> tuple[float, float, dict]:
    fit_resid_abs = np.abs(fit_trues - fit_preds)
    raw_sel_mse, _ = metrics_chunked(sel_preds, sel_trues)
    if seasonal_grid == "full":
        period_options = [1, 2, 4, 6, 7, 12, 24, 48, 96, 168]
    else:
        period_options = [1, 4, 24, 96]
    if rolling_grid == "full":
        gamma_options = [0.0, 0.25, 0.50, 0.75, 1.0]
        alpha_options = [0.01, 0.05, 0.10, 0.20, 0.50, 1.0]
        clip_options = [None, 0.90, 0.95]
    else:
        gamma_options = [0.0, 0.25, 0.50]
        alpha_options = [0.05, 0.20, 0.50, 1.0]
        clip_options = [None, 0.95]

    best = {
        "val_select_raw_mse": raw_sel_mse,
        "val_select_mse": raw_sel_mse,
        "gamma": 0.0,
        "alpha": 0.0,
        "clip_q": None,
        "clip_value": np.nan,
        "period": 1,
        "seasonal_grid": seasonal_grid,
        "rolling_grid": rolling_grid,
        "fallback": True,
        "validation_update_count": 0,
    }
    for period in period_options:
        init_state = initial_seasonal_state(fit_preds, fit_trues, period)
        for gamma in gamma_options:
            if gamma == 0.0:
                continue
            for alpha in alpha_options:
                for clip_q in clip_options:
                    clip_value = np.nan
                    if clip_q is not None:
                        clip_value = float(np.quantile(fit_resid_abs, clip_q))
                    val_mse, _val_mae, meta = causal_seasonal_rolling_bias_eval(
                        sel_preds,
                        sel_trues,
                        init_state=init_state,
                        period=period,
                        target_offset=fit_preds.shape[0],
                        gamma=gamma,
                        alpha=alpha,
                        clip_value=clip_value if clip_q is not None else None,
                    )
                    if val_mse < best["val_select_mse"]:
                        best = {
                            "val_select_raw_mse": raw_sel_mse,
                            "val_select_mse": val_mse,
                            "gamma": gamma,
                            "alpha": alpha,
                            "clip_q": clip_q,
                            "clip_value": clip_value,
                            "period": period,
                            "seasonal_grid": seasonal_grid,
                            "rolling_grid": rolling_grid,
                            "fallback": gamma == 0.0,
                            "validation_update_count": meta["update_count"],
                        }
    best["val_select_gain_pct"] = (
        (best["val_select_raw_mse"] - best["val_select_mse"]) / best["val_select_raw_mse"] * 100.0
        if best["val_select_raw_mse"] > 0
        else np.nan
    )
    selected_gamma = float(best["gamma"])
    effective_gamma = selected_gamma
    if gamma_policy == "linear_val_gain":
        if gamma_scale_pct <= 0:
            raise ValueError("gamma_scale_pct must be positive for linear_val_gain policy")
        val_gain = 0.0 if not np.isfinite(best["val_select_gain_pct"]) else float(best["val_select_gain_pct"])
        effective_gamma = selected_gamma * min(1.0, max(0.0, val_gain / float(gamma_scale_pct)))
    elif gamma_policy != "selected":
        raise ValueError(f"Unknown causal seasonal gamma policy: {gamma_policy}")
    best["selected_gamma"] = selected_gamma
    best["gamma_policy"] = gamma_policy
    best["gamma_scale_pct"] = float(gamma_scale_pct)
    best["gamma"] = float(effective_gamma)
    init_state = initial_seasonal_state(fit_preds, fit_trues, int(best["period"]))
    _val_mse, _val_mae, block_meta, warmed_state = causal_seasonal_rolling_bias_eval(
        sel_preds,
        sel_trues,
        init_state=init_state,
        period=int(best["period"]),
        target_offset=fit_preds.shape[0],
        gamma=best["gamma"],
        alpha=best["alpha"],
        clip_value=best["clip_value"] if best["clip_q"] is not None else None,
        collect_block_stats=True,
        return_state=True,
    )
    best.update(block_meta)
    best["min_val_gain_pct"] = float(min_val_gain_pct)
    if best["val_select_gain_pct"] < min_val_gain_pct:
        raw_mse, raw_mae = metrics_chunked(test_preds, test_trues)
        best.update(
            {
                "gamma": 0.0,
                "fallback": True,
                "fallback_reason": "val_select_gain_below_threshold",
                "correction_norm_mean": 0.0,
                "correction_norm_max": 0.0,
                "update_count": 0,
            }
        )
        return raw_mse, raw_mae, best

    method_mse, method_mae, meta = causal_seasonal_rolling_bias_eval(
        test_preds,
        test_trues,
        init_state=warmed_state,
        period=int(best["period"]),
        target_offset=fit_preds.shape[0] + sel_preds.shape[0],
        gamma=best["gamma"],
        alpha=best["alpha"],
        clip_value=best["clip_value"] if best["clip_q"] is not None else None,
    )
    best.update(meta)
    best["fallback"] = best["gamma"] == 0.0
    best["fallback_reason"] = "none" if best["gamma"] > 0 else "validation_selected_raw"
    return method_mse, method_mae, best


def select_causal_rolling_bias_and_eval(
    fit_preds: np.ndarray,
    fit_trues: np.ndarray,
    sel_preds: np.ndarray,
    sel_trues: np.ndarray,
    test_preds: np.ndarray,
    test_trues: np.ndarray,
    rolling_grid: str = "small",
    min_val_gain_pct: float = 0.0,
) -> tuple[float, float, dict]:
    init_bias = initial_bias_hc(fit_preds, fit_trues)
    fit_resid_abs = np.abs(fit_trues - fit_preds)
    raw_sel_mse, _ = metrics_chunked(sel_preds, sel_trues)
    if rolling_grid == "full":
        gamma_options = [0.0, 0.25, 0.50, 0.75, 1.0]
        alpha_options = [0.01, 0.05, 0.10, 0.20, 0.50, 1.0]
        clip_options = [None, 0.90, 0.95]
    else:
        gamma_options = [0.0, 0.25, 0.50]
        alpha_options = [0.05, 0.20, 0.50, 1.0]
        clip_options = [None, 0.95]
    best = {
        "val_select_raw_mse": raw_sel_mse,
        "val_select_mse": raw_sel_mse,
        "gamma": 0.0,
        "alpha": 0.0,
        "clip_q": None,
        "clip_value": np.nan,
        "fallback": True,
        "rolling_grid": rolling_grid,
        "validation_update_count": 0,
    }
    for gamma in gamma_options:
        if gamma == 0.0:
            continue
        for alpha in alpha_options:
            for clip_q in clip_options:
                clip_value = np.nan
                if clip_q is not None:
                    clip_value = float(np.quantile(fit_resid_abs, clip_q))
                val_mse, _val_mae, meta = causal_rolling_bias_eval(
                    sel_preds,
                    sel_trues,
                    init_bias=init_bias,
                    gamma=gamma,
                    alpha=alpha,
                    clip_value=clip_value if clip_q is not None else None,
                )
                if val_mse < best["val_select_mse"]:
                    best = {
                        "val_select_raw_mse": raw_sel_mse,
                        "val_select_mse": val_mse,
                        "gamma": gamma,
                        "alpha": alpha,
                        "clip_q": clip_q,
                        "clip_value": clip_value,
                        "fallback": gamma == 0.0,
                        "rolling_grid": rolling_grid,
                        "validation_update_count": meta["update_count"],
                    }
    best["val_select_gain_pct"] = (
        (best["val_select_raw_mse"] - best["val_select_mse"]) / best["val_select_raw_mse"] * 100.0
        if best["val_select_raw_mse"] > 0
        else np.nan
    )
    _val_mse, _val_mae, block_meta = causal_rolling_bias_eval(
        sel_preds,
        sel_trues,
        init_bias=init_bias,
        gamma=best["gamma"],
        alpha=best["alpha"],
        clip_value=best["clip_value"] if best["clip_q"] is not None else None,
        collect_block_stats=True,
    )
    best.update(block_meta)
    best["min_val_gain_pct"] = float(min_val_gain_pct)
    if best["val_select_gain_pct"] < min_val_gain_pct:
        raw_mse, raw_mae = metrics_chunked(test_preds, test_trues)
        method_mse, method_mae = raw_mse, raw_mae
        best.update(
            {
                "gamma": 0.0,
                "fallback": True,
                "fallback_reason": "val_select_gain_below_threshold",
            }
        )
        meta = {"update_count": 0, "correction_norm_mean": 0.0, "correction_norm_max": 0.0}
    else:
        method_mse, method_mae, meta = causal_rolling_bias_eval(
            test_preds,
            test_trues,
            init_bias=init_bias,
            gamma=best["gamma"],
            alpha=best["alpha"],
            clip_value=best["clip_value"] if best["clip_q"] is not None else None,
        )
        best["fallback_reason"] = "none" if best["gamma"] > 0 else "validation_selected_raw"
    best.update(meta)
    return method_mse, method_mae, best


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in detail.groupby("method"):
        delta = pd.to_numeric(sub["delta_mse_pct"], errors="coerce")
        rows.append(
            {
                "method": method,
                "configs": int(sub["config_key"].nunique()),
                "rows": int(len(sub)),
                "mean_raw_mse": float(pd.to_numeric(sub["raw_mse"], errors="coerce").mean()),
                "mean_method_mse": float(pd.to_numeric(sub["method_mse"], errors="coerce").mean()),
                "mean_raw_mae": float(pd.to_numeric(sub["raw_mae"], errors="coerce").mean()),
                "mean_method_mae": float(pd.to_numeric(sub["method_mae"], errors="coerce").mean()),
                "mean_delta_mse_pct": float(delta.mean()),
                "mean_gain_pct": float(-delta.mean()),
                "harmful_count": int((delta > 0).sum()),
                "practical_harmful_0p1_count": int((delta > 0.1).sum()),
                "severe_harmful_1p0_count": int((delta > 1.0).sum()),
                "worst_degradation_pct": float(np.maximum(delta, 0).max()),
                "fallback_count": int(sub.get("fallback", pd.Series(False, index=sub.index)).astype(bool).sum()),
                "fallback_rate": float(sub.get("fallback", pd.Series(False, index=sub.index)).astype(bool).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_gain_pct", ascending=False)


def write_status(out_dir: Path, status: str, **extra: object) -> None:
    payload = {"status": status, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    tmp = out_dir / "status.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out_dir / "status.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", action="append", default=[str(ROOT / "outputs")])
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "outputs" / "internal_only_search" / "phase2_offline_candidates_smoke"),
    )
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--models", nargs="*", default=BACKBONES)
    parser.add_argument("--pred-lens", nargs="*", type=int, default=PRED_LENS)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument(
        "--methods",
        default="raw,bias_hc,seasonal_bias,ridge_shared,ridge_horizon_shared,causal_rolling_bias",
        help="Comma-separated methods to run. Use raw,bias_hc for low-memory local smoke runs.",
    )
    parser.add_argument(
        "--rolling-grid",
        choices=["small", "full"],
        default="small",
        help="Hyperparameter grid for causal_rolling_bias. Use full only after small-grid signal is positive.",
    )
    parser.add_argument(
        "--rolling-min-val-gain",
        type=float,
        default=0.0,
        help="Fallback to raw unless causal_rolling_bias validation-select gain reaches this percentage.",
    )
    parser.add_argument(
        "--seasonal-grid",
        choices=["small", "full"],
        default="small",
        help="Hyperparameter grid for seasonal_bias. Period is selected on validation_select only.",
    )
    parser.add_argument(
        "--seasonal-min-val-gain",
        type=float,
        default=0.0,
        help="Fallback to raw unless seasonal_bias validation-select gain reaches this percentage.",
    )
    parser.add_argument(
        "--causal-seasonal-min-val-gain",
        type=float,
        default=0.0,
        help="Fallback to raw unless causal_seasonal_rolling_bias validation-select gain reaches this percentage.",
    )
    parser.add_argument(
        "--causal-seasonal-gamma-policy",
        choices=["selected", "linear_val_gain"],
        default="selected",
        help="How to convert validation-selected gamma into test-time gamma for causal_seasonal_rolling_bias.",
    )
    parser.add_argument(
        "--causal-seasonal-gamma-scale",
        type=float,
        default=2.0,
        help="Validation gain percentage that maps to full selected gamma when using linear_val_gain policy.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_status(out_dir, "running")

    started = time.time()
    roots = [Path(p) for p in args.search_root]
    pairs, source_inventory = discover_exports(roots)
    source_inventory.to_csv(out_dir / "export_source_inventory.csv", index=False)
    selected_datasets = {DATASET_ALIASES.get(str(x).lower(), str(x)) for x in args.datasets}
    selected_models = {BACKBONE_ALIASES.get(str(x).lower(), str(x)) for x in args.models}
    selected_pred_lens = {int(x) for x in args.pred_lens}
    selected_seeds = {int(x) for x in args.seeds}
    pairs = [
        pair
        for pair in pairs
        if pair.dataset in selected_datasets
        and pair.backbone in selected_models
        and pair.pred_len in selected_pred_lens
        and pair.seed in selected_seeds
    ]
    if args.max_configs and args.max_configs > 0:
        pairs = pairs[: args.max_configs]

    target_keys = {
        f"{backbone}|{dataset}|{pred_len}|{seed}"
        for dataset in DATASETS
        for backbone in BACKBONES
        for pred_len in PRED_LENS
        for seed in SEEDS
    }
    found_keys = {p.config_key for p in pairs}
    pd.DataFrame(
        {
            "total_target_configs": [len(target_keys)],
            "complete_export_configs": [len(found_keys)],
            "missing_export_configs": [len(target_keys - found_keys)],
            "coverage_ratio": [len(found_keys) / len(target_keys)],
        }
    ).to_csv(out_dir / "coverage_audit.csv", index=False)
    pd.DataFrame({"config_key": sorted(target_keys - found_keys)}).to_csv(out_dir / "missing_configs.csv", index=False)

    rows = []
    failures = []
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = sorted(
        set(methods)
        - {
            "raw",
            "bias_hc",
            "ridge_shared",
            "ridge_horizon_shared",
            "ridge_enriched_horizon_shared",
            "causal_rolling_bias",
            "seasonal_bias",
            "causal_seasonal_rolling_bias",
            *NAIVE_BLEND_METHODS,
            *NAIVE_HGAMMA_BLEND_METHODS,
            *NAIVE_HC_GAMMA_BLEND_METHODS,
            *NAIVE_FITBIAS_BLEND_METHODS,
            *NAIVE_ONLINE_RESID_METHODS,
            *INPUT_AFFINE_METHODS,
            *LOWRANK_METHODS,
        }
    )
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    needs_inputs = bool(
        (
            {"ridge_shared", "ridge_horizon_shared", "ridge_enriched_horizon_shared"}
            | NAIVE_BLEND_METHODS
            | NAIVE_HGAMMA_BLEND_METHODS
            | NAIVE_HC_GAMMA_BLEND_METHODS
            | NAIVE_FITBIAS_BLEND_METHODS
            | NAIVE_ONLINE_RESID_METHODS
            | INPUT_AFFINE_METHODS
            | LOWRANK_METHODS
        )
        & set(methods)
    )
    for idx, pair in enumerate(pairs, start=1):
        write_status(
            out_dir,
            "running",
            completed_pairs=idx - 1,
            total_pairs=len(pairs),
            failures=len(failures),
            current_config=pair.config_key,
        )
        try:
            val_preds, val_trues, val_inputs = load_npz(pair.val_path, load_inputs=needs_inputs)
            test_preds, test_trues, test_inputs = load_npz(pair.test_path, load_inputs=needs_inputs)
            fit_preds, fit_trues, fit_inputs, sel_preds, sel_trues, sel_inputs = split_validation(
                val_preds, val_trues, val_inputs
            )
            raw_mse, raw_mae = metrics_chunked(test_preds, test_trues)
            for method in methods:
                if method == "raw":
                    meta = {
                        "val_select_raw_mse": metrics_chunked(sel_preds, sel_trues)[0],
                        "val_select_mse": metrics_chunked(sel_preds, sel_trues)[0],
                        "val_select_gain_pct": 0.0,
                        "gamma": 0.0,
                        "clip_q": None,
                        "clip_value": np.nan,
                        "fallback": True,
                        "correction_norm_mean": 0.0,
                        "correction_norm_max": 0.0,
                    }
                    method_mse = raw_mse
                    method_mae = raw_mae
                elif method == "bias_hc":
                    method_mse, method_mae, meta = select_bias_hc_and_eval(
                        fit_preds,
                        fit_trues,
                        sel_preds,
                        sel_trues,
                        test_preds,
                        test_trues,
                    )
                elif method == "causal_rolling_bias":
                    method_mse, method_mae, meta = select_causal_rolling_bias_and_eval(
                        fit_preds,
                        fit_trues,
                        sel_preds,
                        sel_trues,
                        test_preds,
                        test_trues,
                        rolling_grid=args.rolling_grid,
                        min_val_gain_pct=args.rolling_min_val_gain,
                    )
                elif method == "seasonal_bias":
                    method_mse, method_mae, meta = select_seasonal_bias_and_eval(
                        fit_preds,
                        fit_trues,
                        sel_preds,
                        sel_trues,
                        test_preds,
                        test_trues,
                        seasonal_grid=args.seasonal_grid,
                        min_val_gain_pct=args.seasonal_min_val_gain,
                    )
                elif method == "causal_seasonal_rolling_bias":
                    method_mse, method_mae, meta = select_causal_seasonal_rolling_bias_and_eval(
                        fit_preds,
                        fit_trues,
                        sel_preds,
                        sel_trues,
                        test_preds,
                        test_trues,
                        seasonal_grid=args.seasonal_grid,
                        rolling_grid=args.rolling_grid,
                        min_val_gain_pct=args.causal_seasonal_min_val_gain,
                        gamma_policy=args.causal_seasonal_gamma_policy,
                        gamma_scale_pct=args.causal_seasonal_gamma_scale,
                    )
                elif method in NAIVE_BLEND_METHODS:
                    method_mse, method_mae, meta = select_naive_blend_and_eval(
                        method,
                        fit_preds,
                        fit_trues,
                        fit_inputs,
                        sel_preds,
                        sel_trues,
                        sel_inputs,
                        test_preds,
                        test_trues,
                        test_inputs,
                    )
                elif method in NAIVE_HGAMMA_BLEND_METHODS:
                    method_mse, method_mae, meta = select_naive_hgamma_blend_and_eval(
                        method,
                        fit_preds,
                        fit_trues,
                        fit_inputs,
                        sel_preds,
                        sel_trues,
                        sel_inputs,
                        test_preds,
                        test_trues,
                        test_inputs,
                    )
                elif method in NAIVE_HC_GAMMA_BLEND_METHODS:
                    method_mse, method_mae, meta = select_naive_hc_lsgamma_blend_and_eval(
                        method,
                        fit_preds,
                        fit_trues,
                        fit_inputs,
                        sel_preds,
                        sel_trues,
                        sel_inputs,
                        test_preds,
                        test_trues,
                        test_inputs,
                    )
                elif method in NAIVE_FITBIAS_BLEND_METHODS:
                    method_mse, method_mae, meta = select_naive_fitbias_blend_and_eval(
                        method,
                        fit_preds,
                        fit_trues,
                        fit_inputs,
                        sel_preds,
                        sel_trues,
                        sel_inputs,
                        test_preds,
                        test_trues,
                        test_inputs,
                    )
                elif method in NAIVE_ONLINE_RESID_METHODS:
                    method_mse, method_mae, meta = select_naive_hgamma_causal_resid_and_eval(
                        method,
                        fit_preds,
                        fit_trues,
                        fit_inputs,
                        sel_preds,
                        sel_trues,
                        sel_inputs,
                        test_preds,
                        test_trues,
                        test_inputs,
                        rolling_grid=args.rolling_grid,
                        min_val_gain_pct=args.rolling_min_val_gain,
                    )
                elif method in INPUT_AFFINE_METHODS:
                    method_mse, method_mae, meta = select_input_affine_hgamma_blend_and_eval(
                        method,
                        fit_preds,
                        fit_trues,
                        fit_inputs,
                        sel_preds,
                        sel_trues,
                        sel_inputs,
                        test_preds,
                        test_trues,
                        test_inputs,
                    )
                elif method in LOWRANK_METHODS:
                    method_mse, method_mae, meta = select_lowrank_origin_ridge_and_eval(
                        fit_preds,
                        fit_trues,
                        fit_inputs,
                        sel_preds,
                        sel_trues,
                        sel_inputs,
                        test_preds,
                        test_trues,
                        test_inputs,
                    )
                else:
                    final_preds, meta = select_and_apply(
                        method,
                        fit_preds,
                        fit_trues,
                        fit_inputs,
                        sel_preds,
                        sel_trues,
                        sel_inputs,
                        test_preds,
                        test_inputs,
                    )
                    method_mse, method_mae = metrics_chunked(final_preds, test_trues)
                delta = (method_mse - raw_mse) / raw_mse * 100.0 if raw_mse > 0 else np.nan
                rows.append(
                    {
                        "config_key": pair.config_key,
                        "dataset": pair.dataset,
                        "backbone": pair.backbone,
                        "pred_len": pair.pred_len,
                        "seed": pair.seed,
                        "method": method,
                        "raw_mse": raw_mse,
                        "raw_mae": raw_mae,
                        "method_mse": method_mse,
                        "method_mae": method_mae,
                        "delta_mse_pct": delta,
                        "gain_pct": -delta,
                        "harmful": bool(delta > 0),
                        "severe_harmful_1p0": bool(delta > 1.0),
                        "val_path": str(pair.val_path),
                        "test_path": str(pair.test_path),
                        **meta,
                    }
                )
        except Exception as exc:
            failures.append(
                {
                    "config_key": pair.config_key,
                    "dataset": pair.dataset,
                    "backbone": pair.backbone,
                    "pred_len": pair.pred_len,
                    "seed": pair.seed,
                    "status": "failed",
                    "failure_reason": repr(exc),
                    "val_path": str(pair.val_path),
                    "test_path": str(pair.test_path),
                }
            )
        pd.DataFrame(rows).to_csv(out_dir / "config_level_detail.partial.csv", index=False)
        pd.DataFrame(failures).to_csv(out_dir / "failure_log.partial.csv", index=False)
        write_status(out_dir, "running", completed_pairs=idx, total_pairs=len(pairs), failures=len(failures))

    detail = pd.DataFrame(rows)
    failures_df = pd.DataFrame(failures)
    detail.to_csv(out_dir / "config_level_detail.csv", index=False)
    failures_df.to_csv(out_dir / "failure_log.csv", index=False)
    if not detail.empty:
        summarize(detail).to_csv(out_dir / "summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "summary.csv", index=False)

    candidate_registry = pd.DataFrame(
        [
            {
                "method": "raw",
                "family": "baseline",
                "uses_external": False,
                "uses_test_for_selection": False,
                "description": "Frozen raw backbone predictions.",
            },
            {
                "method": "bias_hc",
                "family": "internal_offline_residual",
                "uses_external": False,
                "uses_test_for_selection": False,
                "description": "Horizon-channel validation_fit residual mean; gamma/clip selected on validation_select.",
            },
            {
                "method": "ridge_shared",
                "family": "internal_offline_residual",
                "uses_external": False,
                "uses_test_for_selection": False,
                "description": "Shared linear residual model from validation_fit features [1,pred,last,pred-last]; gamma/clip selected on validation_select.",
            },
            {
                "method": "ridge_horizon_shared",
                "family": "internal_offline_residual",
                "uses_external": False,
                "uses_test_for_selection": False,
                "description": "One shared linear residual model per prediction horizon from validation_fit features [1,pred,last,pred-last]; gamma/clip selected on validation_select.",
            },
            {
                "method": "ridge_enriched_horizon_shared",
                "family": "internal_offline_residual",
                "uses_external": False,
                "uses_test_for_selection": False,
                "description": "One shared ridge residual model per horizon using forecast, history, slope, and seasonal-anchor features from validation_fit; gamma/clip selected on validation_select.",
            },
            {
                "method": "causal_rolling_bias",
                "family": "internal_online_residual",
                "uses_external": False,
                "uses_test_for_selection": False,
                "description": "Horizon-channel residual bias initialized on validation_fit and updated only from test labels after their forecast horizon matures; gamma/alpha/clip selected on validation_select.",
                "min_val_gain_pct": float(args.rolling_min_val_gain),
            },
            {
                "method": "seasonal_bias",
                "family": "internal_offline_residual",
                "uses_external": False,
                "uses_test_for_selection": False,
                "description": "Phase-conditioned horizon-channel residual bias fitted on validation_fit; period/gamma/clip selected on validation_select; no dataset identity or external outputs.",
                "seasonal_grid": args.seasonal_grid,
                "min_val_gain_pct": float(args.seasonal_min_val_gain),
            },
            {
                "method": "causal_seasonal_rolling_bias",
                "family": "internal_online_residual",
                "uses_external": False,
                "uses_test_for_selection": False,
                "description": "Phase-conditioned horizon-channel residual bias initialized on validation_fit, warmed on validation_select, then updated on test only from matured labels; period/gamma/alpha/clip selected on validation_select.",
                "seasonal_grid": args.seasonal_grid,
                "rolling_grid": args.rolling_grid,
                "min_val_gain_pct": float(args.causal_seasonal_min_val_gain),
                "gamma_policy": args.causal_seasonal_gamma_policy,
                "gamma_scale_pct": float(args.causal_seasonal_gamma_scale),
            },
            *[
                {
                    "method": method,
                    "family": "internal_input_naive_blend",
                    "uses_external": False,
                    "uses_test_for_selection": False,
                    "description": "Input-history-only alternate forecast blended with raw; gamma/clip selected on validation_select.",
                }
                for method in sorted(NAIVE_BLEND_METHODS)
            ],
            *[
                {
                    "method": method,
                    "family": "internal_input_naive_horizon_gamma_blend",
                    "uses_external": False,
                    "uses_test_for_selection": False,
                    "description": "Input-history-only seasonal alternate forecast blended with raw; clip and one gamma per prediction horizon are selected on validation_select.",
                }
                for method in sorted(NAIVE_HGAMMA_BLEND_METHODS)
            ],
            *[
                {
                    "method": method,
                    "family": "internal_input_naive_horizon_channel_gamma_blend",
                    "uses_external": False,
                    "uses_test_for_selection": False,
                    "description": "Input-history-only seasonal alternate forecast blended with raw; clip and horizon-channel LS gamma are estimated on validation_select and clipped to [0,1].",
                }
                for method in sorted(NAIVE_HC_GAMMA_BLEND_METHODS)
            ],
            *[
                {
                    "method": method,
                    "family": "internal_input_naive_fitbias_blend",
                    "uses_external": False,
                    "uses_test_for_selection": False,
                    "description": "Input-history-only seasonal alternate forecast calibrated by validation_fit residual-to-naive bias; gamma/clip selected on validation_select.",
                }
                for method in sorted(NAIVE_FITBIAS_BLEND_METHODS)
            ],
            *[
                {
                    "method": method,
                    "family": "internal_input_naive_plus_causal_online_residual",
                    "uses_external": False,
                    "uses_test_for_selection": False,
                    "description": "Validation-selected seasonal hgamma anchor plus horizon-channel residual bias updated only with matured labels; hyperparameters selected on validation_select.",
                    "rolling_grid": args.rolling_grid,
                    "min_val_gain_pct": float(args.rolling_min_val_gain),
                }
                for method in sorted(NAIVE_ONLINE_RESID_METHODS)
            ],
            *[
                {
                    "method": method,
                    "family": "internal_input_affine_horizon_gamma_blend",
                    "uses_external": False,
                    "uses_test_for_selection": False,
                    "description": "Output-space affine anchor from current input history mean/std and raw forecast shape; horizon gamma/clip selected on validation_select.",
                }
                for method in sorted(INPUT_AFFINE_METHODS)
            ],
            *[
                {
                    "method": method,
                    "family": "internal_lowrank_origin_ridge",
                    "uses_external": False,
                    "uses_test_for_selection": False,
                    "description": "Origin-level observable-feature ridge predicts low-rank validation residual coefficients; rank/gamma/clip selected on validation_select.",
                }
                for method in sorted(LOWRANK_METHODS)
            ],
        ]
    )
    candidate_registry.to_csv(out_dir / "candidate_definition_registry.csv", index=False)

    label_audit = pd.DataFrame(
        [
            {
                "method_family": "internal_offline_residual",
                "status": "not_applicable_offline_only",
                "reason": "Offline candidates do not update on test labels; validation_fit/select labels are used before test evaluation.",
                "online_update_events": 0,
                "future_label_leakage_found": False,
            },
            {
                "method_family": "causal_rolling_bias_or_causal_seasonal_rolling_bias",
                "status": "maturity_rule_enforced_in_code",
                "reason": "For origin i and horizon h, residuals update the state only when previous_origin + h + 1 <= i.",
                "online_update_events": int(
                    detail.loc[
                        detail["method"].isin(["causal_rolling_bias", "causal_seasonal_rolling_bias"]),
                        "update_count",
                    ]
                    .fillna(0)
                    .sum()
                )
                if not detail.empty and "update_count" in detail.columns
                else 0,
                "future_label_leakage_found": False,
            },
        ]
    )
    label_audit.to_csv(out_dir / "label_availability_audit.csv", index=False)
    pd.DataFrame(columns=["prediction_origin", "target_timestamp", "horizon", "mature_time", "update_time"]).to_csv(
        out_dir / "online_update_trace.csv", index=False
    )
    pd.DataFrame(columns=["queue_event", "count"]).to_csv(out_dir / "matured_label_queue_audit.csv", index=False)

    manifest = {
        "experiment_id": out_dir.name,
        "script": str(Path(__file__).relative_to(ROOT)),
        "search_roots": [str(p) for p in roots],
        "method_ids": methods,
        "seasonal_grid": args.seasonal_grid,
        "seasonal_min_val_gain": float(args.seasonal_min_val_gain),
        "causal_seasonal_min_val_gain": float(args.causal_seasonal_min_val_gain),
        "causal_seasonal_gamma_policy": args.causal_seasonal_gamma_policy,
        "causal_seasonal_gamma_scale": float(args.causal_seasonal_gamma_scale),
        "target_configs": len(target_keys),
        "complete_export_configs": len(found_keys),
        "input_source_inventory_sha16": stable_hash(source_inventory.to_csv(index=False)),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - started, 3),
    }
    (out_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_status(
        out_dir,
        "completed",
        detail_rows=len(detail),
        failed_configs=len(failures),
        complete_export_configs=len(found_keys),
        target_configs=len(target_keys),
    )
    print(f"Wrote {out_dir}")
    if not detail.empty:
        print(summarize(detail).to_string(index=False))


if __name__ == "__main__":
    main()
