from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import ts_posthoc_w1 as base
import ts_posthoc_w3 as w3
from e06_formal_protocol import (
    E06_SAFE_ALPHA_GRID,
    E06_SAFE_FORGETTING_GRID,
    E06_SAFE_GATE_REQUIRED_IMPROVED,
    E06_SAFE_MIN_TUNE_GAIN_PCT,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
E06_ANALYSIS_ROOT = PROJECT_ROOT / "outputs" / "e06_analysis"

EMA_ALPHA_GRID = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
RIDGE_ALPHA_GRID = [300.0, 1000.0, 3000.0, 10000.0, 30000.0]
E03_MIN_TUNE_GAIN_PCT = 5.0
E06_ALPHA_GRID = list(E06_SAFE_ALPHA_GRID)
E06_FORGETTING_GRID = list(E06_SAFE_FORGETTING_GRID)
E06_MIN_TUNE_GAIN_PCT = E06_SAFE_MIN_TUNE_GAIN_PCT
CP_ALPHA = 0.1


def run_key_of(config: base.RunConfig) -> str:
    return f"{config.model}-{config.dataset}-pl{config.pred_len}"


def config_from_run_key(run_key: str) -> base.RunConfig:
    for config in base.w1_default_configs():
        if run_key_of(config) == run_key:
            return config
    raise KeyError(f"Unknown run key: {run_key}")


def load_w1_metrics(config: base.RunConfig) -> dict[str, Any]:
    metrics_path = config.export_dir / "posthoc_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing W1 metrics: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def thirds(num_samples: int) -> list[tuple[str, int, int]]:
    first = num_samples // 3
    second = (2 * num_samples) // 3
    return [
        ("early", 0, first),
        ("mid", first, second),
        ("late", second, num_samples),
    ]


def quarter_spans(num_samples: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, num_samples, 5, dtype=int)
    return [(int(edges[idx]), int(edges[idx + 1])) for idx in range(4)]


def delta_pct(mse: float, raw_mse: float) -> float:
    if abs(raw_mse) < 1e-12:
        return 0.0
    return 100.0 * (mse - raw_mse) / raw_mse


def cli_namespace() -> SimpleNamespace:
    return SimpleNamespace(
        ema_alpha_grid=EMA_ALPHA_GRID,
        ridge_alpha_grid=RIDGE_ALPHA_GRID,
        e03_min_tune_gain_pct=E03_MIN_TUNE_GAIN_PCT,
        e06_alpha_grid=E06_ALPHA_GRID,
        e06_forgetting_grid=E06_FORGETTING_GRID,
        e06_min_tune_gain_pct=E06_MIN_TUNE_GAIN_PCT,
        cp_alpha=CP_ALPHA,
    )


def parse_variant(variant: str) -> tuple[float, float]:
    alpha_part, lambda_part = variant.split("|")
    alpha = float(alpha_part.split("=")[1])
    forgetting = float(lambda_part.split("=")[1])
    return alpha, forgetting


def fit_online_model(
    preds: np.ndarray,
    inputs: np.ndarray,
    trues: np.ndarray,
    alpha: float,
    forgetting: float,
) -> dict[str, Any]:
    offline = base.fit_linear_residual_adapter(preds, inputs, trues, alpha, feature_mode="weak")
    return {
        "kind": "online_linear",
        "alpha": alpha,
        "forgetting": forgetting,
        "state": w3.rls_state_from_linear_model(offline, prior_alpha=alpha),
    }


def segment_rows(
    run_key: str,
    config: base.RunConfig,
    method_name: str,
    raw_preds: np.ndarray,
    method_preds: np.ndarray,
    trues: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment_name, start_idx, end_idx in thirds(trues.shape[0]):
        raw_slice = raw_preds[start_idx:end_idx]
        pred_slice = method_preds[start_idx:end_idx]
        true_slice = trues[start_idx:end_idx]
        raw_mae, raw_mse = base.metric(raw_slice, true_slice)
        method_mae, method_mse = base.metric(pred_slice, true_slice)
        rows.append(
            {
                "run_key": run_key,
                "model": config.model,
                "dataset": config.dataset,
                "pred_len": config.pred_len,
                "method": method_name,
                "segment_name": segment_name,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "segment_size": end_idx - start_idx,
                "segment_start_ratio": start_idx / trues.shape[0],
                "segment_end_ratio": end_idx / trues.shape[0],
                "raw_mae": raw_mae,
                "raw_mse": raw_mse,
                "method_mae": method_mae,
                "method_mse": method_mse,
                "delta_mse_pct": delta_pct(method_mse, raw_mse),
            }
        )
    return rows


def selection_scores_for_e06(preds: np.ndarray, inputs: np.ndarray, trues: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    cli = cli_namespace()
    return w3.select_e06_model(
        preds,
        inputs,
        trues,
        cli.e06_alpha_grid,
        cli.e06_forgetting_grid,
        cli.e06_min_tune_gain_pct,
    )


def forced_params_from_arrays(preds: np.ndarray, inputs: np.ndarray, trues: np.ndarray) -> tuple[float, float, dict[str, Any]]:
    _, diag = selection_scores_for_e06(preds, inputs, trues)
    candidates = diag["candidates"]
    best_variant = min(candidates, key=lambda name: candidates[name]["tune_mse"])
    alpha, forgetting = parse_variant(best_variant)
    return alpha, forgetting, diag


def fit_bundle_raw(_: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    return {"kind": "raw"}, {"selected": "raw"}


def fit_bundle_e02(fit_arrays: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    alpha, scores = base.select_ema_alpha(fit_arrays["preds"], fit_arrays["trues"], EMA_ALPHA_GRID)
    _, state, seen = base.streaming_ema_correction(fit_arrays["preds"], fit_arrays["trues"], alpha)
    return {
        "kind": "E02",
        "alpha": alpha,
        "state": state,
        "seen": seen,
    }, {"selected": f"alpha={alpha:g}", "alpha_scores": scores}


def fit_bundle_e03(fit_arrays: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    fitted, diag = base.select_e03_model(
        fit_arrays["preds"],
        fit_arrays["inputs"],
        fit_arrays["trues"],
        RIDGE_ALPHA_GRID,
        E03_MIN_TUNE_GAIN_PCT,
    )
    return {"kind": "E03", "fitted": fitted}, diag


def fit_bundle_e06_base(fit_arrays: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    fitted, diag = selection_scores_for_e06(fit_arrays["preds"], fit_arrays["inputs"], fit_arrays["trues"])
    return fitted, diag


def fit_bundle_e06_forced(fit_arrays: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    alpha, forgetting, diag = forced_params_from_arrays(fit_arrays["preds"], fit_arrays["inputs"], fit_arrays["trues"])
    fitted = fit_online_model(fit_arrays["preds"], fit_arrays["inputs"], fit_arrays["trues"], alpha, forgetting)
    diag = dict(diag)
    diag["selected_forced"] = f"alpha={alpha:g}|lambda={forgetting:g}"
    diag["selected_forced_alpha"] = alpha
    diag["selected_forced_forgetting"] = forgetting
    return fitted, diag


def apply_bundle(bundle: dict[str, Any], arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    kind = bundle["kind"]
    if kind == "raw":
        return arrays["preds"].copy(), {"variant": "raw"}
    if kind == "E02":
        pred, state, seen = base.streaming_ema_correction(
            arrays["preds"],
            arrays["trues"],
            bundle["alpha"],
            init_state=bundle["state"],
            init_seen=bundle["seen"],
        )
        return pred, {"variant": f"alpha={bundle['alpha']:g}", "state": state, "seen": seen}
    if kind == "E03":
        return base.apply_linear_residual_adapter(arrays["preds"], arrays["inputs"], bundle["fitted"]), {
            "variant": bundle["fitted"]["kind"]
        }
    if kind in {"identity", "online_linear"}:
        pred, next_bundle, updates = w3.apply_e06_model(arrays["preds"], arrays["inputs"], arrays["trues"], bundle)
        return pred, {"variant": next_bundle["kind"], "bundle_after": next_bundle, "updates": updates}
    raise ValueError(f"Unknown bundle kind: {kind}")


def forced_test_predictions(fit_arrays: dict[str, np.ndarray], test_arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    bundle, diag = fit_bundle_e06_forced(fit_arrays)
    pred, info = apply_bundle(bundle, test_arrays)
    return pred, {"fit_diag": diag, **info}


def block_deltas(raw_preds: np.ndarray, corrected_preds: np.ndarray, trues: np.ndarray) -> list[float]:
    raw_err = np.empty(trues.shape[0], dtype=np.float64)
    corrected_err = np.empty(trues.shape[0], dtype=np.float64)
    chunk = max(1, min(trues.shape[0], 64))
    for left in range(0, trues.shape[0], chunk):
        right = min(trues.shape[0], left + chunk)
        true_chunk = trues[left:right].astype(np.float32, copy=False)
        raw_chunk = raw_preds[left:right].astype(np.float32, copy=False)
        corrected_chunk = corrected_preds[left:right].astype(np.float32, copy=False)
        raw_err[left:right] = np.square(true_chunk - raw_chunk, dtype=np.float32).mean(axis=(1, 2), dtype=np.float64)
        corrected_err[left:right] = np.square(true_chunk - corrected_chunk, dtype=np.float32).mean(axis=(1, 2), dtype=np.float64)
    deltas: list[float] = []
    for left, right in quarter_spans(raw_err.shape[0]):
        raw_mse = float(raw_err[left:right].mean())
        corrected_mse = float(corrected_err[left:right].mean())
        deltas.append(delta_pct(corrected_mse, raw_mse))
    return deltas


def block_gate_enabled(
    block_gain_pct: list[float],
    required_improved: int = E06_SAFE_GATE_REQUIRED_IMPROVED,
) -> bool:
    return sum(1 for value in block_gain_pct if value < 0.0) >= required_improved


def late_weighted_gate_enabled(block_gain_pct: list[float], weights: list[float]) -> tuple[bool, float]:
    weighted_score = float(np.average(np.asarray(block_gain_pct, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64)))
    return weighted_score < 0.0, weighted_score


def blend_with_ramp(raw_preds: np.ndarray, corrected_preds: np.ndarray, ramp_windows: int) -> np.ndarray:
    if ramp_windows <= 0:
        return corrected_preds.copy()
    weights = np.minimum(1.0, (np.arange(raw_preds.shape[0], dtype=np.float32) + 1.0) / float(ramp_windows))
    return raw_preds + weights[:, None, None] * (corrected_preds - raw_preds)


def fit_gate_variants(
    fit_source_arrays: dict[str, np.ndarray],
    weights: list[float] | None = None,
    ramp_windows: int | None = None,
    mode: str = "block_gate",
) -> tuple[dict[str, Any], dict[str, Any]]:
    split_idx = base.temporal_split_index(fit_source_arrays["preds"].shape[0])
    fit_arrays, tune_arrays = w3.split_arrays(fit_source_arrays, split_idx)
    alpha, forgetting, diag = forced_params_from_arrays(fit_source_arrays["preds"], fit_source_arrays["inputs"], fit_source_arrays["trues"])
    forced_fit_on_tune = fit_online_model(fit_arrays["preds"], fit_arrays["inputs"], fit_arrays["trues"], alpha, forgetting)
    forced_tune_pred, _, _ = w3.apply_e06_model(
        tune_arrays["preds"],
        tune_arrays["inputs"],
        tune_arrays["trues"],
        forced_fit_on_tune,
    )
    block_gain = block_deltas(tune_arrays["preds"], forced_tune_pred, tune_arrays["trues"])
    enabled = False
    weighted_score: float | None = None
    if mode == "block_gate":
        enabled = block_gate_enabled(block_gain)
    elif mode == "late_weighted":
        if weights is None:
            raise ValueError("weights must be provided for late_weighted")
        enabled, weighted_score = late_weighted_gate_enabled(block_gain, weights)
    elif mode == "warmup_ramp":
        enabled = True
    elif mode == "block_plus_warmup":
        enabled = block_gate_enabled(block_gain)
    else:
        raise ValueError(f"Unknown gate mode: {mode}")

    full_bundle = fit_online_model(
        fit_source_arrays["preds"],
        fit_source_arrays["inputs"],
        fit_source_arrays["trues"],
        alpha,
        forgetting,
    )
    return {
        "kind": mode,
        "forced_bundle": full_bundle,
        "forced_alpha": alpha,
        "forced_forgetting": forgetting,
        "enabled": enabled,
        "weights": weights,
        "ramp_windows": ramp_windows,
    }, {
        "selection_scores": diag,
        "block_gain_pct": block_gain,
        "weighted_score": weighted_score,
        "enabled": enabled,
    }


def apply_gate_bundle(bundle: dict[str, Any], arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    kind = bundle["kind"]
    if kind in {"block_gate", "late_weighted"} and not bundle["enabled"]:
        return arrays["preds"].copy(), {
            "enabled": bundle["enabled"],
            "forced_alpha": bundle["forced_alpha"],
            "forced_forgetting": bundle["forced_forgetting"],
            "ramp_windows": bundle["ramp_windows"],
            "weights": bundle["weights"],
            "updates": 0,
        }
    if kind == "block_plus_warmup" and not bundle["enabled"]:
        return arrays["preds"].copy(), {
            "enabled": bundle["enabled"],
            "forced_alpha": bundle["forced_alpha"],
            "forced_forgetting": bundle["forced_forgetting"],
            "ramp_windows": bundle["ramp_windows"],
            "weights": bundle["weights"],
            "updates": 0,
        }

    forced_pred, _, updates = w3.apply_e06_model(
        arrays["preds"],
        arrays["inputs"],
        arrays["trues"],
        bundle["forced_bundle"],
    )

    if kind == "block_gate":
        pred = forced_pred
    elif kind == "late_weighted":
        pred = forced_pred
    elif kind == "warmup_ramp":
        pred = blend_with_ramp(arrays["preds"], forced_pred, int(bundle["ramp_windows"]))
    elif kind == "block_plus_warmup":
        pred = blend_with_ramp(arrays["preds"], forced_pred, int(bundle["ramp_windows"]))
    else:
        raise ValueError(f"Unknown gate bundle kind: {kind}")
    return pred, {
        "enabled": bundle["enabled"],
        "forced_alpha": bundle["forced_alpha"],
        "forced_forgetting": bundle["forced_forgetting"],
        "ramp_windows": bundle["ramp_windows"],
        "weights": bundle["weights"],
        "updates": updates,
    }


def horizon_table_rows(
    run_key: str,
    config: base.RunConfig,
    raw_preds: np.ndarray,
    method_preds: np.ndarray,
    trues: np.ndarray,
    method_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_h = base.horizon_mse(raw_preds, trues)
    method_h = base.horizon_mse(method_preds, trues)
    for idx, (raw_value, method_value) in enumerate(zip(raw_h, method_h), start=1):
        rows.append(
            {
                "run_key": run_key,
                "model": config.model,
                "dataset": config.dataset,
                "pred_len": config.pred_len,
                "method": method_name,
                "horizon_index": idx,
                "raw_horizon_mse": float(raw_value),
                "method_horizon_mse": float(method_value),
                "delta_mse_pct": delta_pct(float(method_value), float(raw_value)),
            }
        )
    return rows


def interval_metrics_from_quantiles(
    preds: np.ndarray,
    trues: np.ndarray,
    quantiles: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    low = preds - quantiles[:, :, None]
    high = preds + quantiles[:, :, None]
    inside = (trues >= low) & (trues <= high)
    coverage = float(np.mean(inside))
    width = float(np.mean(high - low))
    penalty_low = np.maximum(low - trues, 0.0)
    penalty_high = np.maximum(trues - high, 0.0)
    interval_score = (high - low) + (2.0 / alpha) * penalty_low + (2.0 / alpha) * penalty_high
    wis = float(np.mean(interval_score))
    return {
        "coverage": coverage,
        "mean_interval_width": width,
        "interval_score": wis,
        "wis": wis,
    }


def online_conformal_with_quantiles(
    preds: np.ndarray,
    trues: np.ndarray,
    init_abs_residuals: list[np.ndarray],
    alpha: float,
) -> tuple[np.ndarray, dict[str, float]]:
    samples, horizon, channels = preds.shape
    banks = [res.copy().astype(np.float32).reshape(-1) for res in init_abs_residuals]
    quantiles = np.zeros((samples, horizon), dtype=np.float32)
    for idx in range(samples):
        for h_idx in range(horizon):
            src_idx = idx - (h_idx + 1)
            if src_idx >= 0:
                revealed = np.abs(trues[src_idx, h_idx, :] - preds[src_idx, h_idx, :]).reshape(-1).astype(np.float32)
                banks[h_idx] = np.concatenate([banks[h_idx], revealed], axis=0)
            q_level = w3.quantile_level(len(banks[h_idx]), alpha)
            quantiles[idx, h_idx] = float(np.quantile(banks[h_idx], q_level))
    return quantiles, interval_metrics_from_quantiles(preds, trues, quantiles, alpha)
