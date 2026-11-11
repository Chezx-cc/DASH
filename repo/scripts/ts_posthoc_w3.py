from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import torch
try:
    from numba import njit
except ImportError:  # pragma: no cover - optional accelerator
    njit = None

import ts_posthoc_stage_a as stage_a
import ts_posthoc_w1 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
WORKBOOK_PATH = PROJECT_ROOT / "ts_posthoc_experiment_matrix.xlsx"
RESULTS_CSV = OUTPUT_ROOT / "w3_results.csv"
SUMMARY_JSON = OUTPUT_ROOT / "w3_summary.json"
SNAP_EXPORT_ROOT = OUTPUT_ROOT / "w3_snapshots"
METHODS = ["E06", "E07", "E08"]
TARGET_COVERAGE = 0.9


if njit is not None:
    @njit(cache=True)
    def _streaming_rls_core(
        preds: np.ndarray,
        trues: np.ndarray,
        features: np.ndarray,
        weights: np.ndarray,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        cov: np.ndarray,
        forgetting: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        samples, horizon, channels, feat_dim = features.shape
        width = feat_dim + 1
        corrected = np.empty_like(preds)
        update_count = 0

        for idx in range(samples):
            for h_idx in range(horizon):
                src_idx = idx - (h_idx + 1)
                if src_idx < 0:
                    continue
                for c_idx in range(channels):
                    x_aug = np.empty(width, dtype=np.float32)
                    x_aug[0] = 1.0
                    for f_idx in range(feat_dim):
                        x_aug[f_idx + 1] = (
                            (features[src_idx, h_idx, c_idx, f_idx] - feat_mean[h_idx, c_idx, f_idx])
                            / feat_std[h_idx, c_idx, f_idx]
                        )

                    px = np.empty(width, dtype=np.float32)
                    denom = forgetting
                    pred_res = 0.0
                    for row_idx in range(width):
                        tmp = 0.0
                        for col_idx in range(width):
                            tmp += cov[h_idx, c_idx, row_idx, col_idx] * x_aug[col_idx]
                        px[row_idx] = tmp
                        denom += x_aug[row_idx] * tmp
                        pred_res += x_aug[row_idx] * weights[h_idx, c_idx, row_idx]

                    residual = trues[src_idx, h_idx, c_idx] - preds[src_idx, h_idx, c_idx]
                    err = residual - pred_res

                    x_tp = np.empty(width, dtype=np.float32)
                    for col_idx in range(width):
                        tmp = 0.0
                        for row_idx in range(width):
                            tmp += x_aug[row_idx] * cov[h_idx, c_idx, row_idx, col_idx]
                        x_tp[col_idx] = tmp

                    for row_idx in range(width):
                        gain = px[row_idx] / denom
                        weights[h_idx, c_idx, row_idx] = weights[h_idx, c_idx, row_idx] + gain * err
                        for col_idx in range(width):
                            cov[h_idx, c_idx, row_idx, col_idx] = (
                                cov[h_idx, c_idx, row_idx, col_idx] - gain * x_tp[col_idx]
                            ) / forgetting
                    update_count += 1

            for h_idx in range(horizon):
                for c_idx in range(channels):
                    correction = weights[h_idx, c_idx, 0]
                    for f_idx in range(feat_dim):
                        x_norm = (
                            (features[idx, h_idx, c_idx, f_idx] - feat_mean[h_idx, c_idx, f_idx])
                            / feat_std[h_idx, c_idx, f_idx]
                        )
                        correction += x_norm * weights[h_idx, c_idx, f_idx + 1]
                    corrected[idx, h_idx, c_idx] = preds[idx, h_idx, c_idx] + correction

        return corrected, weights, cov, update_count


def split_arrays(arrays: dict[str, np.ndarray], split_idx: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    left = {key: value[:split_idx] for key, value in arrays.items()}
    right = {key: value[split_idx:] for key, value in arrays.items()}
    return left, right


def load_stage_a_best_methods() -> dict[str, str]:
    summary_path = OUTPUT_ROOT / "stage_a_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Stage A summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)["per_run"]

    best: dict[str, str] = {}
    for run_key, run in summary.items():
        best_method = min(["E01", "E02", "E03", "E04", "E05"], key=lambda name: run[name]["mse"])
        best[run_key] = best_method
    return best


def rls_state_from_linear_model(model: dict[str, np.ndarray], prior_alpha: float) -> dict[str, np.ndarray]:
    weights = model["weights"].copy().astype(np.float32)
    horizon, channels, width = weights.shape
    eye = np.eye(width, dtype=np.float32) / prior_alpha
    cov = np.repeat(eye[None, None, :, :], horizon, axis=0)
    cov = np.repeat(cov, channels, axis=1)
    return {
        "kind": "online_linear",
        "feature_mode": model.get("feature_mode", "weak"),
        "weights": weights,
        "feat_mean": model["feat_mean"].copy().astype(np.float32),
        "feat_std": model["feat_std"].copy().astype(np.float32),
        "cov": cov.astype(np.float32),
    }


def streaming_rls_correction(
    preds: np.ndarray,
    inputs: np.ndarray,
    trues: np.ndarray,
    state: dict[str, np.ndarray],
    forgetting: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    if state["kind"] == "identity":
        return preds.copy(), {"kind": "identity"}, 0

    features = base.build_residual_features(preds, inputs, mode=state["feature_mode"])
    corrected = np.empty_like(preds)
    weights = state["weights"].copy()
    feat_mean = state["feat_mean"]
    feat_std = state["feat_std"]
    cov = state["cov"].copy()
    samples, horizon, channels, _ = features.shape
    update_count = 0

    if njit is not None:
        corrected, weights, cov, update_count = _streaming_rls_core(
            preds.astype(np.float32, copy=False),
            trues.astype(np.float32, copy=False),
            features.astype(np.float32, copy=False),
            weights.astype(np.float32, copy=False),
            feat_mean.astype(np.float32, copy=False),
            feat_std.astype(np.float32, copy=False),
            cov.astype(np.float32, copy=False),
            float(forgetting),
        )
    else:
        for idx in range(samples):
            for h_idx in range(horizon):
                src_idx = idx - (h_idx + 1)
                if src_idx < 0:
                    continue
                residual = trues[src_idx, h_idx, :] - preds[src_idx, h_idx, :]
                x_raw = features[src_idx, h_idx, :, :]
                x_norm = (x_raw - feat_mean[h_idx]) / feat_std[h_idx]
                x_aug = np.concatenate([np.ones((channels, 1), dtype=np.float32), x_norm.astype(np.float32)], axis=1)
                p_mat = cov[h_idx]
                w_mat = weights[h_idx]
                px = np.einsum("cij,cj->ci", p_mat, x_aug, optimize=True)
                denom = forgetting + np.einsum("ci,ci->c", x_aug, px, optimize=True)
                gain = px / denom[:, None]
                err = residual - np.einsum("ci,ci->c", x_aug, w_mat, optimize=True)
                weights[h_idx] = (w_mat + gain * err[:, None]).astype(np.float32)
                x_tp = np.einsum("ci,cij->cj", x_aug, p_mat, optimize=True)
                cov[h_idx] = ((p_mat - gain[:, :, None] * x_tp[:, None, :]) / forgetting).astype(np.float32)
                update_count += channels

            x_cur = (features[idx] - feat_mean) / feat_std
            x_cur = np.concatenate(
                [np.ones((horizon, channels, 1), dtype=np.float32), x_cur.astype(np.float32)],
                axis=-1,
            )
            correction = np.sum(x_cur * weights, axis=-1)
            corrected[idx] = preds[idx] + correction

    new_state = {
        "kind": "online_linear",
        "feature_mode": state["feature_mode"],
        "weights": weights,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "cov": cov,
    }
    return corrected, new_state, update_count


def select_e06_model(
    preds: np.ndarray,
    inputs: np.ndarray,
    trues: np.ndarray,
    alpha_grid: list[float],
    forgetting_grid: list[float],
    min_tune_gain_pct: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    split_idx = base.temporal_split_index(preds.shape[0])
    fit_arrays, tune_arrays = split_arrays(
        {"preds": preds, "inputs": inputs, "trues": trues},
        split_idx,
    )
    _, raw_tune_mse = base.metric(tune_arrays["preds"], tune_arrays["trues"])
    diagnostics: dict[str, Any] = {
        "raw_tune_mse": raw_tune_mse,
        "min_tune_gain_pct": min_tune_gain_pct,
        "candidates": {},
    }
    best_choice: tuple[float, float] | None = None
    best_score = raw_tune_mse

    for alpha in alpha_grid:
        offline_model = base.fit_linear_residual_adapter(
            fit_arrays["preds"],
            fit_arrays["inputs"],
            fit_arrays["trues"],
            alpha,
            feature_mode="weak",
        )
        for forgetting in forgetting_grid:
            state = rls_state_from_linear_model(offline_model, prior_alpha=alpha)
            corrected, _, updates = streaming_rls_correction(
                tune_arrays["preds"],
                tune_arrays["inputs"],
                tune_arrays["trues"],
                state,
                forgetting,
            )
            _, tune_mse = base.metric(corrected, tune_arrays["trues"])
            key = f"alpha={alpha:g}|lambda={forgetting:g}"
            tune_gain_pct = 100.0 * (raw_tune_mse - tune_mse) / raw_tune_mse
            diagnostics["candidates"][key] = {
                "tune_mse": tune_mse,
                "tune_gain_pct": tune_gain_pct,
                "updates": updates,
            }
            if tune_mse < best_score:
                best_score = tune_mse
                best_choice = (alpha, forgetting)

    if best_choice is None:
        diagnostics["selected"] = "identity"
        return {"kind": "identity"}, diagnostics

    best_gain_pct = 100.0 * (raw_tune_mse - best_score) / raw_tune_mse
    diagnostics["best_tune_gain_pct"] = best_gain_pct
    if best_gain_pct < min_tune_gain_pct:
        diagnostics["selected"] = "identity"
        diagnostics["rejected_reason"] = "best candidate did not clear gain gate"
        return {"kind": "identity"}, diagnostics

    alpha, forgetting = best_choice
    diagnostics["selected"] = f"alpha={alpha:g}|lambda={forgetting:g}"
    diagnostics["selected_alpha"] = alpha
    diagnostics["selected_forgetting"] = forgetting
    offline_model = base.fit_linear_residual_adapter(preds, inputs, trues, alpha, feature_mode="weak")
    return {
        "kind": "online_linear",
        "alpha": alpha,
        "forgetting": forgetting,
        "state": rls_state_from_linear_model(offline_model, prior_alpha=alpha),
    }, diagnostics


def apply_e06_model(
    preds: np.ndarray,
    inputs: np.ndarray,
    trues: np.ndarray,
    fitted: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], int]:
    if fitted["kind"] == "identity":
        return preds.copy(), {"kind": "identity"}, 0
    corrected, state, updates = streaming_rls_correction(
        preds,
        inputs,
        trues,
        fitted["state"],
        fitted["forgetting"],
    )
    new_fitted = {
        "kind": "online_linear",
        "alpha": fitted["alpha"],
        "forgetting": fitted["forgetting"],
        "state": state,
    }
    return corrected, new_fitted, updates


def fit_stage_method(
    method: str,
    fit_arrays: dict[str, np.ndarray],
    config: base.RunConfig,
    cli: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if method == "E01":
        fitted, scores = base.select_e01_calibrator(fit_arrays["preds"], fit_arrays["trues"])
        return {"kind": "E01", "fitted": fitted}, {"selected": fitted["kind"], "scores": scores}
    if method == "E02":
        alpha, alpha_scores = base.select_ema_alpha(fit_arrays["preds"], fit_arrays["trues"], cli.ema_alpha_grid)
        _, state, seen = base.streaming_ema_correction(fit_arrays["preds"], fit_arrays["trues"], alpha)
        return {
            "kind": "E02",
            "alpha": alpha,
            "state": state,
            "seen": seen,
        }, {"selected": f"alpha={alpha:g}", "alpha_scores": alpha_scores}
    if method == "E03":
        fitted, diag = base.select_e03_model(
            fit_arrays["preds"],
            fit_arrays["inputs"],
            fit_arrays["trues"],
            cli.ridge_alpha_grid,
            cli.e03_min_tune_gain_pct,
        )
        return {"kind": "E03", "fitted": fitted}, diag
    if method == "E04":
        fitted, diag = stage_a.select_e04_model(
            fit_arrays["preds"],
            fit_arrays["inputs"],
            fit_arrays["trues"],
            cli.ridge_alpha_grid,
            cli.gate_alpha_grid,
        )
        return {"kind": "E04", "fitted": fitted}, diag
    if method == "E05":
        fitted, diag = stage_a.select_e05_model(
            fit_arrays["preds"],
            fit_arrays["inputs"],
            fit_arrays["trues"],
            cli.ridge_alpha_grid,
            cli.revise_fractions,
            cli.detector_alpha,
        )
        return {"kind": "E05", "fitted": fitted}, diag
    if method == "E06":
        fitted, diag = select_e06_model(
            fit_arrays["preds"],
            fit_arrays["inputs"],
            fit_arrays["trues"],
            cli.e06_alpha_grid,
            cli.e06_forgetting_grid,
            cli.e06_min_tune_gain_pct,
        )
        return fitted, diag
    raise ValueError(f"Unknown method: {method}")


def predict_stage_method(
    bundle: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    kind = bundle["kind"]
    if kind == "E01":
        pred = base.apply_e01_calibrator(arrays["preds"], bundle["fitted"])
        return pred, bundle, {"variant": bundle["fitted"]["kind"]}
    if kind == "E02":
        pred, state, seen = base.streaming_ema_correction(
            arrays["preds"],
            arrays["trues"],
            bundle["alpha"],
            init_state=bundle["state"],
            init_seen=bundle["seen"],
        )
        next_bundle = {"kind": "E02", "alpha": bundle["alpha"], "state": state, "seen": seen}
        return pred, next_bundle, {"variant": f"alpha={bundle['alpha']:g}"}
    if kind == "E03":
        pred = base.apply_linear_residual_adapter(arrays["preds"], arrays["inputs"], bundle["fitted"])
        return pred, bundle, {"variant": bundle["fitted"]["kind"]}
    if kind == "E04":
        pred, gate = stage_a.apply_drift_gated_adapter(arrays["preds"], arrays["inputs"], bundle["fitted"])
        return pred, bundle, {
            "variant": bundle["fitted"]["kind"],
            "gate_mean": float(gate.mean()),
            "gate_p90": float(np.quantile(gate, 0.9)),
        }
    if kind == "E05":
        fitted = bundle["fitted"]
        if fitted["kind"] == "identity":
            pred = arrays["preds"].copy()
            return pred, bundle, {"variant": "identity", "revised_fraction": 0.0}
        pred, risk, mask = stage_a.apply_selective_revision(
            arrays["preds"],
            arrays["inputs"],
            fitted["reviser"],
            fitted["detector"],
            fitted["revise_fraction"],
        )
        return pred, bundle, {
            "variant": fitted["kind"],
            "revised_fraction": float(mask.mean()),
            "risk_mean": float(risk.mean()),
        }
    if kind == "identity":
        return arrays["preds"].copy(), bundle, {"variant": "identity"}
    if kind == "online_linear":
        pred, next_bundle, updates = apply_e06_model(
            arrays["preds"],
            arrays["inputs"],
            arrays["trues"],
            bundle,
        )
        return pred, next_bundle, {
            "variant": f"alpha={bundle['alpha']:g}|lambda={bundle['forgetting']:g}",
            "updates": updates,
        }
    raise ValueError(f"Unknown prediction bundle kind: {kind}")


def quantile_level(n: int, alpha: float) -> float:
    if n <= 0:
        return 1.0
    return min(1.0, float(np.ceil((n + 1) * (1 - alpha)) / n))


def online_conformal_metrics(
    preds: np.ndarray,
    trues: np.ndarray,
    init_abs_residuals: list[np.ndarray],
    alpha: float,
) -> dict[str, Any]:
    samples, horizon, channels = preds.shape
    banks = [res.copy().astype(np.float32).reshape(-1) for res in init_abs_residuals]
    covered = 0
    total = samples * horizon * channels
    width_sum = 0.0
    quantiles = np.zeros((samples, horizon), dtype=np.float32)

    for idx in range(samples):
        for h_idx in range(horizon):
            src_idx = idx - (h_idx + 1)
            if src_idx >= 0:
                revealed = np.abs(trues[src_idx, h_idx, :] - preds[src_idx, h_idx, :]).reshape(-1).astype(np.float32)
                banks[h_idx] = np.concatenate([banks[h_idx], revealed], axis=0)
            q_level = quantile_level(len(banks[h_idx]), alpha)
            q_value = float(np.quantile(banks[h_idx], q_level))
            quantiles[idx, h_idx] = q_value
            low = preds[idx, h_idx, :] - q_value
            high = preds[idx, h_idx, :] + q_value
            covered += int(np.sum((trues[idx, h_idx, :] >= low) & (trues[idx, h_idx, :] <= high)))
            width_sum += float(2.0 * q_value * channels)

    return {
        "coverage": covered / total,
        "avg_width": width_sum / total,
        "quantile_mean": float(quantiles.mean()),
        "quantile_p90": float(np.quantile(quantiles, 0.9)),
    }


def evaluate_e06(
    config: base.RunConfig,
    val_arrays: dict[str, np.ndarray],
    test_arrays: dict[str, np.ndarray],
    cli: argparse.Namespace,
) -> dict[str, Any]:
    raw_mae, raw_mse = base.metric(test_arrays["preds"], test_arrays["trues"])
    fitted, diag = select_e06_model(
        val_arrays["preds"],
        val_arrays["inputs"],
        val_arrays["trues"],
        cli.e06_alpha_grid,
        cli.e06_forgetting_grid,
        cli.e06_min_tune_gain_pct,
    )
    start = time.perf_counter()
    pred, _, updates = apply_e06_model(test_arrays["preds"], test_arrays["inputs"], test_arrays["trues"], fitted)
    latency_ms = (time.perf_counter() - start) * 1000.0 / test_arrays["preds"].shape[0]
    mae, mse = base.metric(pred, test_arrays["trues"])
    params_added = 0
    if fitted["kind"] != "identity":
        params_added = int(np.prod(fitted["state"]["weights"].shape))
    return {
        "raw": {"mae": raw_mae, "mse": raw_mse},
        "E06": {
            "mae": mae,
            "mse": mse,
            "delta_mse_pct": 100.0 * (mse - raw_mse) / raw_mse,
            "latency_ms_per_window": latency_ms,
            "variant": diag["selected"],
            "params_added": params_added,
            "selection_scores": diag,
            "update_count": updates,
            "horizon_mse": base.horizon_mse(pred, test_arrays["trues"]).tolist(),
        },
    }


def evaluate_e07(
    config: base.RunConfig,
    val_arrays: dict[str, np.ndarray],
    test_arrays: dict[str, np.ndarray],
    point_method: str,
    cli: argparse.Namespace,
) -> dict[str, Any]:
    split_idx = base.temporal_split_index(val_arrays["preds"].shape[0])
    fit_arrays, calib_arrays = split_arrays(val_arrays, split_idx)

    candidates: dict[str, dict[str, Any]] = {}
    for candidate_method in [point_method, "E06"]:
        bundle, fit_diag = fit_stage_method(candidate_method, fit_arrays, config, cli)
        calib_pred, bundle_after_calib, calib_info = predict_stage_method(bundle, calib_arrays)
        _, calib_mse = base.metric(calib_pred, calib_arrays["trues"])
        candidates[candidate_method] = {
            "bundle_after_calib": bundle_after_calib,
            "calib_pred": calib_pred,
            "fit_diag": fit_diag,
            "calib_info": calib_info,
            "calib_mse": calib_mse,
        }

    chosen_method = min(candidates, key=lambda name: candidates[name]["calib_mse"])
    chosen = candidates[chosen_method]
    test_point_pred, _, test_info = predict_stage_method(chosen["bundle_after_calib"], test_arrays)

    raw_cp = online_conformal_metrics(
        test_arrays["preds"],
        test_arrays["trues"],
        [np.abs(calib_arrays["trues"][:, h_idx, :] - calib_arrays["preds"][:, h_idx, :]) for h_idx in range(config.pred_len)],
        cli.cp_alpha,
    )
    corrected_cp = online_conformal_metrics(
        test_point_pred,
        test_arrays["trues"],
        [np.abs(calib_arrays["trues"][:, h_idx, :] - chosen["calib_pred"][:, h_idx, :]) for h_idx in range(config.pred_len)],
        cli.cp_alpha,
    )
    raw_mae, raw_mse = base.metric(test_arrays["preds"], test_arrays["trues"])
    point_mae, point_mse = base.metric(test_point_pred, test_arrays["trues"])
    return {
        "raw": {"mae": raw_mae, "mse": raw_mse},
        "E07": {
            "mae": point_mae,
            "mse": point_mse,
            "delta_mse_pct": 100.0 * (point_mse - raw_mse) / raw_mse,
            "latency_ms_per_window": 0.0,
            "params_added": 0,
            "variant": chosen_method,
            "coverage": corrected_cp["coverage"],
            "avg_width": corrected_cp["avg_width"],
            "selection_scores": {
                "requested_point_method": point_method,
                "chosen_point_method": chosen_method,
                "candidate_calib_mse": {name: info["calib_mse"] for name, info in candidates.items()},
                "fit_diag": chosen["fit_diag"],
                "calib_info": chosen["calib_info"],
                "test_info": test_info,
                "raw_cp": raw_cp,
                "corrected_cp": corrected_cp,
                "coverage_error_abs": abs(corrected_cp["coverage"] - TARGET_COVERAGE),
            },
        },
    }


def snapshot_config(config: base.RunConfig) -> base.RunConfig:
    return base.replace_config(config, des="w3e08", train_epochs=10, patience=3)


def snapshot_setting(config: base.RunConfig, seed: int) -> str:
    args = base.arg_namespace(config)
    args.seed = seed
    return base.tslib_setting(args)


def run_training_with_snapshots(config: base.RunConfig, seed: int, force: bool = False) -> str:
    setting = snapshot_setting(config, seed)
    checkpoint_dir = base.TSLIB_ROOT / "checkpoints" / setting
    epoch_files = list(checkpoint_dir.glob("epoch_*.pth"))
    if checkpoint_dir.exists() and epoch_files and not force:
        return setting

    command = [
        sys.executable,
        "-u",
        "run.py",
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--root_path",
        config.root_path,
        "--data_path",
        config.data_path,
        "--model_id",
        config.model_id,
        "--model",
        config.model,
        "--data",
        config.dataset,
        "--features",
        config.features,
        "--freq",
        config.freq,
        "--seq_len",
        str(config.seq_len),
        "--label_len",
        str(config.label_len),
        "--pred_len",
        str(config.pred_len),
        "--e_layers",
        str(config.e_layers),
        "--d_layers",
        str(config.d_layers),
        "--factor",
        str(config.factor),
        "--enc_in",
        str(config.enc_in),
        "--dec_in",
        str(config.dec_in),
        "--c_out",
        str(config.c_out),
        "--des",
        config.des,
        "--d_model",
        str(config.d_model),
        "--d_ff",
        str(config.d_ff),
        "--n_heads",
        str(config.n_heads),
        "--batch_size",
        str(config.batch_size),
        "--train_epochs",
        str(config.train_epochs),
        "--patience",
        str(config.patience),
        "--learning_rate",
        str(config.learning_rate),
        "--num_workers",
        str(config.num_workers),
        "--itr",
        str(config.itr),
        "--gpu",
        str(config.gpu),
        "--seed",
        str(seed),
    ]
    if config.use_amp:
        command.append("--use_amp")
    env = os.environ.copy()
    env["TSLIB_SAVE_EPOCH_CHECKPOINTS"] = "1"
    subprocess.run(command, cwd=base.TSLIB_ROOT, env=env, check=True)
    return setting


def collect_arrays_from_checkpoint(
    config: base.RunConfig,
    setting: str,
    checkpoint_path: Path,
    split: str,
    cache_dir: Path,
    force: bool = False,
) -> dict[str, np.ndarray]:
    out_path = cache_dir / f"{split}.npz"
    if out_path.exists() and not force:
        with np.load(out_path) as saved:
            return {key: saved[key] for key in saved.files}

    cache_dir.mkdir(parents=True, exist_ok=True)
    args = base.arg_namespace(config)
    device = torch.device(f"cuda:{config.gpu}" if config.use_gpu and torch.cuda.is_available() else "cpu")

    with base.pushd(base.TSLIB_ROOT):
        model = base.build_model(args, device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        _, loader = base.build_loader(args, split)
        preds: list[np.ndarray] = []
        trues: list[np.ndarray] = []
        inputs: list[np.ndarray] = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(device)
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -args.pred_len :, :]
                batch_y = batch_y[:, -args.pred_len :, :]

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
                inputs.append(batch_x.detach().cpu().numpy())

    arrays = {
        "preds": np.concatenate(preds, axis=0),
        "trues": np.concatenate(trues, axis=0),
        "inputs": np.concatenate(inputs, axis=0),
    }
    np.savez_compressed(out_path, **arrays)
    with (cache_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "setting": setting,
                "checkpoint": str(checkpoint_path),
                "split": split,
                "pred_shape": list(arrays["preds"].shape),
            },
            fh,
            indent=2,
        )
    return arrays


def evaluate_posthoc_method_on_test(
    config: base.RunConfig,
    val_arrays: dict[str, np.ndarray],
    test_arrays: dict[str, np.ndarray],
    method: str,
    cli: argparse.Namespace,
) -> tuple[float, float]:
    if method == "E01":
        fitted, _ = base.select_e01_calibrator(val_arrays["preds"], val_arrays["trues"])
        pred = base.apply_e01_calibrator(test_arrays["preds"], fitted)
    elif method == "E02":
        alpha, _ = base.select_ema_alpha(val_arrays["preds"], val_arrays["trues"], cli.ema_alpha_grid)
        _, state, seen = base.streaming_ema_correction(val_arrays["preds"], val_arrays["trues"], alpha)
        pred, _, _ = base.streaming_ema_correction(
            test_arrays["preds"],
            test_arrays["trues"],
            alpha,
            init_state=state,
            init_seen=seen,
        )
    elif method == "E03":
        fitted, _ = base.select_e03_model(
            val_arrays["preds"],
            val_arrays["inputs"],
            val_arrays["trues"],
            cli.ridge_alpha_grid,
            cli.e03_min_tune_gain_pct,
        )
        pred = base.apply_linear_residual_adapter(test_arrays["preds"], test_arrays["inputs"], fitted)
    elif method == "E04":
        fitted, _ = stage_a.select_e04_model(
            val_arrays["preds"],
            val_arrays["inputs"],
            val_arrays["trues"],
            cli.ridge_alpha_grid,
            cli.gate_alpha_grid,
        )
        pred, _ = stage_a.apply_drift_gated_adapter(test_arrays["preds"], test_arrays["inputs"], fitted)
    elif method == "E05":
        fitted, _ = stage_a.select_e05_model(
            val_arrays["preds"],
            val_arrays["inputs"],
            val_arrays["trues"],
            cli.ridge_alpha_grid,
            cli.revise_fractions,
            cli.detector_alpha,
        )
        if fitted["kind"] == "identity":
            pred = test_arrays["preds"].copy()
        else:
            pred, _, _ = stage_a.apply_selective_revision(
                test_arrays["preds"],
                test_arrays["inputs"],
                fitted["reviser"],
                fitted["detector"],
                fitted["revise_fraction"],
            )
    else:
        raise ValueError(f"Unsupported method for E08: {method}")

    mae, mse = base.metric(pred, test_arrays["trues"])
    return mae, mse


def evaluate_e08(
    config: base.RunConfig,
    point_method: str,
    cli: argparse.Namespace,
) -> dict[str, Any]:
    snap_cfg = snapshot_config(config)
    setting = run_training_with_snapshots(snap_cfg, cli.seed, force=cli.force_snapshot_train)
    checkpoint_dir = base.TSLIB_ROOT / "checkpoints" / setting
    epoch_files = sorted(checkpoint_dir.glob("epoch_*.pth"))
    if not epoch_files:
        raise FileNotFoundError(f"No epoch snapshots found under {checkpoint_dir}")

    candidate_rows: list[dict[str, Any]] = []
    cache_root = SNAP_EXPORT_ROOT / config.model / config.dataset / f"pl{config.pred_len}" / f"seed{cli.seed}"
    for checkpoint_path in epoch_files:
        epoch_name = checkpoint_path.stem
        val_arrays = collect_arrays_from_checkpoint(
            snap_cfg,
            setting,
            checkpoint_path,
            "val",
            cache_root / epoch_name,
            force=cli.force_export,
        )
        split_idx = base.temporal_split_index(val_arrays["preds"].shape[0])
        fit_arrays, tune_arrays = split_arrays(val_arrays, split_idx)
        raw_tune_mae, raw_tune_mse = base.metric(tune_arrays["preds"], tune_arrays["trues"])
        posthoc_tune_mae, posthoc_tune_mse = evaluate_posthoc_method_on_test(
            config,
            fit_arrays,
            tune_arrays,
            point_method,
            cli,
        )
        candidate_rows.append(
            {
                "epoch": epoch_name,
                "checkpoint": str(checkpoint_path),
                "raw_tune_mae": raw_tune_mae,
                "raw_tune_mse": raw_tune_mse,
                "posthoc_tune_mae": posthoc_tune_mae,
                "posthoc_tune_mse": posthoc_tune_mse,
            }
        )

    raw_choice = min(candidate_rows, key=lambda row: row["raw_tune_mse"])
    posthoc_choice = min(candidate_rows, key=lambda row: row["posthoc_tune_mse"])
    raw_val_arrays = collect_arrays_from_checkpoint(snap_cfg, setting, Path(raw_choice["checkpoint"]), "val", cache_root / raw_choice["epoch"], force=cli.force_export)
    raw_test_arrays = collect_arrays_from_checkpoint(snap_cfg, setting, Path(raw_choice["checkpoint"]), "test", cache_root / raw_choice["epoch"], force=cli.force_export)
    post_val_arrays = collect_arrays_from_checkpoint(snap_cfg, setting, Path(posthoc_choice["checkpoint"]), "val", cache_root / posthoc_choice["epoch"], force=cli.force_export)
    post_test_arrays = collect_arrays_from_checkpoint(snap_cfg, setting, Path(posthoc_choice["checkpoint"]), "test", cache_root / posthoc_choice["epoch"], force=cli.force_export)

    raw_base_mae, raw_base_mse = evaluate_posthoc_method_on_test(config, raw_val_arrays, raw_test_arrays, point_method, cli)
    post_mae, post_mse = evaluate_posthoc_method_on_test(config, post_val_arrays, post_test_arrays, point_method, cli)
    return {
        "raw": {"mae": raw_base_mae, "mse": raw_base_mse},
        "E08": {
            "mae": post_mae,
            "mse": post_mse,
            "delta_mse_pct": 100.0 * (post_mse - raw_base_mse) / raw_base_mse,
            "latency_ms_per_window": 0.0,
            "params_added": 0,
            "variant": point_method,
            "selection_scores": {
                "point_method": point_method,
                "raw_selected_epoch": raw_choice["epoch"],
                "posthoc_selected_epoch": posthoc_choice["epoch"],
                "candidates": candidate_rows,
            },
        },
    }


def tracker_row(
    config: base.RunConfig,
    method: str,
    method_result: dict[str, Any],
    raw_result: dict[str, Any],
    seed: int,
) -> list[Any]:
    scheme_map = {
        "E06": "Online update of tiny corrector",
        "E07": "Correct-then-conformal intervals",
        "E08": "Post-hoc checkpoint selection",
    }
    delta = method_result["delta_mse_pct"]
    kept = "yes" if delta < 0 else "no"
    failure_note = "" if delta < 0 else "No gain on this config"
    coverage = None
    avg_width = None
    if method == "E07":
        coverage = method_result["coverage"]
        avg_width = method_result["avg_width"]
        if delta < 0 and abs(coverage - TARGET_COVERAGE) <= 0.03:
            next_action = "Promote with intervals"
        elif delta < 0:
            next_action = "Retune CP alpha"
        else:
            next_action = "Retune or drop"
    elif delta <= -1.0:
        next_action = "Promote to next stage"
    elif delta < 0:
        next_action = "Keep as secondary baseline"
    else:
        next_action = "Retune or drop"

    return [
        date.today().isoformat(),
        method,
        scheme_map[method],
        config.model,
        config.dataset,
        config.pred_len,
        "standard test",
        seed,
        "Yes" if method == "E06" else "No",
        "Yes" if method == "E06" else "No",
        raw_result["mse"],
        method_result["mse"],
        delta,
        raw_result["mae"],
        method_result["mae"],
        coverage,
        avg_width,
        method_result["latency_ms_per_window"],
        method_result["params_added"],
        kept,
        failure_note,
        next_action,
    ]


def write_csv(rows: list[list[Any]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    header = [
        "Date",
        "Exp ID",
        "Scheme",
        "Backbone",
        "Dataset",
        "Horizon",
        "Split Type",
        "Seed",
        "Uses Recent GT?",
        "Online Update?",
        "Raw MSE",
        "Post-hoc MSE",
        "Delta MSE %",
        "Raw MAE",
        "Post-hoc MAE",
        "Coverage",
        "Avg Width",
        "Latency (ms)",
        "Params Added",
        "Kept?",
        "Failure Notes",
        "Next Action",
    ]
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def write_summary(summary: dict[str, Any]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSON.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def aggregate_summary(per_run: dict[str, Any]) -> dict[str, Any]:
    method_scores = {method: [] for method in METHODS}
    wins = {method: 0 for method in METHODS}
    for run_result in per_run.values():
        best_method = min(METHODS, key=lambda name: run_result[name]["mse"])
        wins[best_method] += 1
        for method in METHODS:
            method_scores[method].append(run_result[method]["delta_mse_pct"])
    return {
        "per_run": per_run,
        "aggregate": {
            method: {
                "mean_delta_mse_pct": float(np.mean(scores)),
                "wins": wins[method],
            }
            for method, scores in method_scores.items()
        },
    }


def update_workbook(all_rows: list[list[Any]], expected_runs_per_method: int) -> None:
    if not WORKBOOK_PATH.exists():
        return
    wb = openpyxl.load_workbook(WORKBOOK_PATH)
    ws = wb["Run_Tracker"]
    if ws.max_row > 1:
        ws.delete_rows(2, ws.max_row - 1)
    for row in all_rows:
        ws.append(row)

    counts = {method: 0 for method in METHODS}
    for row in all_rows:
        exp_id = row[1]
        if exp_id in counts:
            counts[exp_id] += 1

    exp_ws = wb["Experiment_Matrix"]
    for row_idx in range(2, exp_ws.max_row + 1):
        exp_id = exp_ws.cell(row=row_idx, column=1).value
        if exp_id in counts:
            count = counts[exp_id]
            status = "done" if count >= expected_runs_per_method else f"partial ({count}/{expected_runs_per_method})"
            exp_ws.cell(row=row_idx, column=17).value = status
    wb.save(WORKBOOK_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run W3 experiments on selected post-hoc lines.")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of models to run.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Subset of datasets to run.")
    parser.add_argument("--pred-lens", nargs="*", type=int, default=None, help="Subset of pred_lens to run.")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--force-export", action="store_true", help="Re-export snapshot val/test arrays even if cached.")
    parser.add_argument("--force-snapshot-train", action="store_true", help="Retrain snapshot runs even if epoch checkpoints exist.")
    parser.add_argument("--skip-e08", action="store_true", help="Skip checkpoint-selection experiments.")
    parser.add_argument("--no-workbook", action="store_true", help="Skip workbook updates.")
    parser.add_argument("--ema-alpha-grid", nargs="*", type=float, default=[0.0003, 0.001, 0.003, 0.01, 0.03, 0.1])
    parser.add_argument("--ridge-alpha-grid", nargs="*", type=float, default=[300.0, 1000.0, 3000.0, 10000.0, 30000.0])
    parser.add_argument("--gate-alpha-grid", nargs="*", type=float, default=[10.0, 100.0, 300.0])
    parser.add_argument("--revise-fractions", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--detector-alpha", type=float, default=100.0)
    parser.add_argument("--e03-min-tune-gain-pct", type=float, default=5.0)
    parser.add_argument("--e06-alpha-grid", nargs="*", type=float, default=[300.0, 1000.0, 3000.0, 10000.0])
    parser.add_argument("--e06-forgetting-grid", nargs="*", type=float, default=[1.0, 0.995])
    parser.add_argument("--e06-min-tune-gain-pct", type=float, default=0.5)
    parser.add_argument("--cp-alpha", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    best_methods = load_stage_a_best_methods()
    configs = base.filtered_configs(base.w1_default_configs(), cli.models, cli.datasets, cli.pred_lens)
    if not configs:
        raise SystemExit("No configs selected.")

    per_run: dict[str, Any] = {}
    tracker_rows: list[list[Any]] = []
    for config in configs:
        run_key = f"{config.model}-{config.dataset}-pl{config.pred_len}"
        point_method = best_methods[run_key]
        val_arrays = base.collect_split_arrays(config, "val", force=False)
        test_arrays = base.collect_split_arrays(config, "test", force=False)
        e06_result = evaluate_e06(config, val_arrays, test_arrays, cli)
        e07_result = evaluate_e07(config, val_arrays, test_arrays, point_method, cli)
        if cli.skip_e08:
            raw_mae, raw_mse = e07_result["raw"]["mae"], e07_result["raw"]["mse"]
            e08_result = {
                "raw": {"mae": raw_mae, "mse": raw_mse},
                "E08": {
                    "mae": raw_mae,
                    "mse": raw_mse,
                    "delta_mse_pct": 0.0,
                    "latency_ms_per_window": 0.0,
                    "params_added": 0,
                    "variant": point_method,
                    "selection_scores": {"skipped": True},
                },
            }
        else:
            e08_result = evaluate_e08(config, point_method, cli)

        run_result = {
            "raw": e06_result["raw"],
            "point_method_for_e07_e08": point_method,
            "E06": e06_result["E06"],
            "E07": e07_result["E07"],
            "E08": e08_result["E08"],
        }
        per_run[run_key] = run_result
        for method in METHODS:
            raw_result = e08_result["raw"] if method == "E08" else run_result["raw"]
            tracker_rows.append(tracker_row(config, method, run_result[method], raw_result, cli.seed))

        metrics_path = config.export_dir / "posthoc_w3_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(run_result, fh, indent=2)

    write_csv(tracker_rows)
    write_summary(aggregate_summary(per_run))
    if not cli.no_workbook:
        update_workbook(tracker_rows, expected_runs_per_method=len(configs))


if __name__ == "__main__":
    main()
