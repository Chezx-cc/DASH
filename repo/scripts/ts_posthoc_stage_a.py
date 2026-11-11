from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl

import ts_posthoc_w1 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
WORKBOOK_PATH = PROJECT_ROOT / "ts_posthoc_experiment_matrix.xlsx"
RESULTS_CSV = OUTPUT_ROOT / "stage_a_results.csv"
SUMMARY_JSON = OUTPUT_ROOT / "stage_a_summary.json"
METHODS = ["E01", "E02", "E03", "E04", "E05"]


def sample_mse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.mean((true - pred) ** 2, axis=(1, 2)).astype(np.float32)


def fit_scalar_ridge(features: np.ndarray, targets: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    return base.ridge_fit(features.astype(np.float32), targets.astype(np.float32), alpha)


def predict_scalar_ridge(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    predicted = base.ridge_predict(model, features.astype(np.float32))
    return predicted.reshape(-1).astype(np.float32)


def build_context_features(preds: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    half_idx = inputs.shape[1] // 2
    early = inputs[:, :half_idx, :]
    late = inputs[:, half_idx:, :]
    first = inputs[:, 0, :]
    last = inputs[:, -1, :]
    pred_first = preds[:, 0, :]
    pred_last = preds[:, -1, :]

    scalar_features = [
        inputs.mean(axis=(1, 2)),
        inputs.std(axis=(1, 2)),
        early.mean(axis=(1, 2)),
        late.mean(axis=(1, 2)),
        late.mean(axis=(1, 2)) - early.mean(axis=(1, 2)),
        early.std(axis=(1, 2)),
        late.std(axis=(1, 2)),
        late.std(axis=(1, 2)) - early.std(axis=(1, 2)),
        (last - first).mean(axis=1),
        np.abs(last - first).mean(axis=1),
        preds.mean(axis=(1, 2)),
        preds.std(axis=(1, 2)),
        (pred_last - pred_first).mean(axis=1),
        (pred_first - last).mean(axis=1),
        np.abs(pred_first - last).mean(axis=1),
    ]
    per_channel_features = [
        late.mean(axis=1) - early.mean(axis=1),
        late.std(axis=1) - early.std(axis=1),
        preds.mean(axis=1) - last,
        pred_first - last,
    ]
    pieces = [feat[:, None].astype(np.float32) for feat in scalar_features]
    pieces.extend(feat.astype(np.float32) for feat in per_channel_features)
    return np.concatenate(pieces, axis=1)


def optimal_gate_targets(
    preds: np.ndarray,
    corrected: np.ndarray,
    trues: np.ndarray,
) -> np.ndarray:
    correction = corrected - preds
    residual = trues - preds
    numer = np.sum(correction * residual, axis=(1, 2))
    denom = np.sum(correction * correction, axis=(1, 2)) + 1e-6
    targets = numer / denom
    return np.clip(targets, 0.0, 1.0).astype(np.float32)


def fit_drift_gated_adapter(
    preds: np.ndarray,
    inputs: np.ndarray,
    trues: np.ndarray,
    head_alpha: float,
    gate_alpha: float,
) -> dict[str, Any]:
    head_model = base.fit_linear_residual_adapter(
        preds,
        inputs,
        trues,
        head_alpha,
        feature_mode="weak",
    )
    fully_corrected = base.apply_linear_residual_adapter(preds, inputs, head_model)
    gate_targets = optimal_gate_targets(preds, fully_corrected, trues)
    gate_features = build_context_features(preds, inputs)
    gate_model = fit_scalar_ridge(gate_features, gate_targets, gate_alpha)
    return {
        "kind": "drift_gated",
        "head_model": head_model,
        "gate_model": gate_model,
        "head_alpha": head_alpha,
        "gate_alpha": gate_alpha,
    }


def apply_drift_gated_adapter(
    preds: np.ndarray,
    inputs: np.ndarray,
    fitted: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if fitted["kind"] == "identity":
        return preds.copy(), np.zeros(preds.shape[0], dtype=np.float32)

    fully_corrected = base.apply_linear_residual_adapter(preds, inputs, fitted["head_model"])
    correction = fully_corrected - preds
    gate = predict_scalar_ridge(fitted["gate_model"], build_context_features(preds, inputs))
    gate = np.clip(gate, 0.0, 1.0)
    corrected = preds + gate[:, None, None] * correction
    return corrected, gate


def select_e04_model(
    preds: np.ndarray,
    inputs: np.ndarray,
    trues: np.ndarray,
    head_alpha_grid: list[float],
    gate_alpha_grid: list[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    split_idx = base.temporal_split_index(preds.shape[0])
    fit_preds, tune_preds = preds[:split_idx], preds[split_idx:]
    fit_inputs, tune_inputs = inputs[:split_idx], inputs[split_idx:]
    fit_trues, tune_trues = trues[:split_idx], trues[split_idx:]

    _, raw_tune_mse = base.metric(tune_preds, tune_trues)
    diagnostics: dict[str, Any] = {
        "raw_tune_mse": raw_tune_mse,
        "candidates": {},
    }
    best_choice: tuple[float, float] | None = None
    best_score = raw_tune_mse

    for head_alpha in head_alpha_grid:
        for gate_alpha in gate_alpha_grid:
            fitted = fit_drift_gated_adapter(fit_preds, fit_inputs, fit_trues, head_alpha, gate_alpha)
            corrected, gate = apply_drift_gated_adapter(tune_preds, tune_inputs, fitted)
            _, tune_mse = base.metric(corrected, tune_trues)
            key = f"head={head_alpha:g}|gate={gate_alpha:g}"
            diagnostics["candidates"][key] = {
                "tune_mse": tune_mse,
                "tune_gain_pct": 100.0 * (raw_tune_mse - tune_mse) / raw_tune_mse,
                "gate_mean": float(gate.mean()),
                "gate_p90": float(np.quantile(gate, 0.9)),
            }
            if tune_mse < best_score:
                best_score = tune_mse
                best_choice = (head_alpha, gate_alpha)

    if best_choice is None:
        diagnostics["selected"] = "identity"
        return {"kind": "identity"}, diagnostics

    head_alpha, gate_alpha = best_choice
    diagnostics["selected"] = f"head={head_alpha:g}|gate={gate_alpha:g}"
    return fit_drift_gated_adapter(preds, inputs, trues, head_alpha, gate_alpha), diagnostics


def fit_revision_gain_detector(
    preds: np.ndarray,
    inputs: np.ndarray,
    gain_targets: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    features = build_context_features(preds, inputs)
    return {
        "kind": "revision_gain_detector",
        "alpha": alpha,
        "model": fit_scalar_ridge(features, gain_targets.astype(np.float32), alpha),
    }


def predict_failure_risk(
    preds: np.ndarray,
    inputs: np.ndarray,
    fitted: dict[str, Any],
) -> np.ndarray:
    return predict_scalar_ridge(fitted["model"], build_context_features(preds, inputs))


def apply_selective_revision(
    preds: np.ndarray,
    inputs: np.ndarray,
    reviser: dict[str, Any],
    detector: dict[str, Any],
    revise_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if revise_fraction <= 0.0:
        return preds.copy(), np.zeros(preds.shape[0], dtype=np.float32), np.zeros(preds.shape[0], dtype=bool)

    risk = predict_failure_risk(preds, inputs, detector)
    corrected = base.apply_linear_residual_adapter(preds, inputs, reviser)
    num_samples = preds.shape[0]
    top_k = max(1, int(math.ceil(num_samples * revise_fraction)))
    chosen = np.argpartition(risk, -top_k)[-top_k:]
    mask = np.zeros(num_samples, dtype=bool)
    mask[chosen] = True
    mixed = preds.copy()
    mixed[mask] = corrected[mask]
    return mixed, risk, mask


def select_e05_model(
    preds: np.ndarray,
    inputs: np.ndarray,
    trues: np.ndarray,
    head_alpha_grid: list[float],
    revise_fraction_grid: list[float],
    detector_alpha: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    split_idx = base.temporal_split_index(preds.shape[0])
    fit_preds, tune_preds = preds[:split_idx], preds[split_idx:]
    fit_inputs, tune_inputs = inputs[:split_idx], inputs[split_idx:]
    fit_trues, tune_trues = trues[:split_idx], trues[split_idx:]

    _, raw_tune_mse = base.metric(tune_preds, tune_trues)
    diagnostics: dict[str, Any] = {
        "raw_tune_mse": raw_tune_mse,
        "detector_alpha": detector_alpha,
        "candidates": {},
    }
    best_choice: tuple[float, float] | None = None
    best_score = raw_tune_mse

    for head_alpha in head_alpha_grid:
        reviser = base.fit_linear_residual_adapter(
            fit_preds,
            fit_inputs,
            fit_trues,
            head_alpha,
            feature_mode="weak",
        )
        fit_corrected = base.apply_linear_residual_adapter(fit_preds, fit_inputs, reviser)
        gain_targets = sample_mse(fit_preds, fit_trues) - sample_mse(fit_corrected, fit_trues)
        detector = fit_revision_gain_detector(fit_preds, fit_inputs, gain_targets, detector_alpha)
        for revise_fraction in revise_fraction_grid:
            corrected, risk, mask = apply_selective_revision(
                tune_preds,
                tune_inputs,
                reviser,
                detector,
                revise_fraction,
            )
            _, tune_mse = base.metric(corrected, tune_trues)
            key = f"head={head_alpha:g}|frac={revise_fraction:.2f}"
            diagnostics["candidates"][key] = {
                "tune_mse": tune_mse,
                "tune_gain_pct": 100.0 * (raw_tune_mse - tune_mse) / raw_tune_mse,
                "revised_fraction": float(mask.mean()),
                "risk_mean": float(risk.mean()),
            }
            if tune_mse < best_score:
                best_score = tune_mse
                best_choice = (head_alpha, revise_fraction)

    if best_choice is None:
        diagnostics["selected"] = "identity"
        return {"kind": "identity"}, diagnostics

    head_alpha, revise_fraction = best_choice
    diagnostics["selected"] = f"head={head_alpha:g}|frac={revise_fraction:.2f}"
    reviser = base.fit_linear_residual_adapter(
        preds,
        inputs,
        trues,
        head_alpha,
        feature_mode="weak",
    )
    full_corrected = base.apply_linear_residual_adapter(preds, inputs, reviser)
    full_gain_targets = sample_mse(preds, trues) - sample_mse(full_corrected, trues)
    detector = fit_revision_gain_detector(preds, inputs, full_gain_targets, detector_alpha)
    return {
        "kind": "selective_revision",
        "reviser": reviser,
        "detector": detector,
        "head_alpha": head_alpha,
        "revise_fraction": revise_fraction,
    }, diagnostics


def evaluate_stage_a(
    config: base.RunConfig,
    val_arrays: dict[str, np.ndarray],
    test_arrays: dict[str, np.ndarray],
    alpha_grid_ema: list[float],
    alpha_grid_ridge: list[float],
    e03_min_tune_gain_pct: float,
    gate_alpha_grid: list[float],
    revise_fraction_grid: list[float],
    detector_alpha: float,
) -> dict[str, Any]:
    results = base.evaluate_methods(
        config,
        val_arrays,
        test_arrays,
        alpha_grid_ema=alpha_grid_ema,
        alpha_grid_ridge=alpha_grid_ridge,
        e03_min_tune_gain_pct=e03_min_tune_gain_pct,
    )
    raw_mae = results["raw"]["mae"]
    raw_mse = results["raw"]["mse"]

    e04_fit, e04_diag = select_e04_model(
        val_arrays["preds"],
        val_arrays["inputs"],
        val_arrays["trues"],
        alpha_grid_ridge,
        gate_alpha_grid,
    )
    start = time.perf_counter()
    e04_pred, e04_gate = apply_drift_gated_adapter(test_arrays["preds"], test_arrays["inputs"], e04_fit)
    e04_latency_ms = (time.perf_counter() - start) * 1000.0 / test_arrays["preds"].shape[0]
    e04_mae, e04_mse = base.metric(e04_pred, test_arrays["trues"])
    results["E04"] = {
        "mae": e04_mae,
        "mse": e04_mse,
        "delta_mse_pct": 100.0 * (e04_mse - raw_mse) / raw_mse,
        "latency_ms_per_window": e04_latency_ms,
        "variant": e04_diag["selected"],
        "params_added": 0 if e04_fit["kind"] == "identity" else (
            int(np.prod(e04_fit["head_model"]["weights"].shape)) + int(e04_fit["gate_model"]["weight"].size)
        ),
        "selection_scores": e04_diag,
        "gate_mean": float(e04_gate.mean()),
        "gate_p90": float(np.quantile(e04_gate, 0.9)),
        "horizon_mse": base.horizon_mse(e04_pred, test_arrays["trues"]).tolist(),
    }

    e05_fit, e05_diag = select_e05_model(
        val_arrays["preds"],
        val_arrays["inputs"],
        val_arrays["trues"],
        alpha_grid_ridge,
        revise_fraction_grid,
        detector_alpha,
    )
    start = time.perf_counter()
    if e05_fit["kind"] == "identity":
        e05_pred = test_arrays["preds"].copy()
        e05_risk = np.zeros(test_arrays["preds"].shape[0], dtype=np.float32)
        e05_mask = np.zeros(test_arrays["preds"].shape[0], dtype=bool)
    else:
        e05_pred, e05_risk, e05_mask = apply_selective_revision(
            test_arrays["preds"],
            test_arrays["inputs"],
            e05_fit["reviser"],
            e05_fit["detector"],
            e05_fit["revise_fraction"],
        )
    e05_latency_ms = (time.perf_counter() - start) * 1000.0 / test_arrays["preds"].shape[0]
    e05_mae, e05_mse = base.metric(e05_pred, test_arrays["trues"])
    results["E05"] = {
        "mae": e05_mae,
        "mse": e05_mse,
        "delta_mse_pct": 100.0 * (e05_mse - raw_mse) / raw_mse,
        "latency_ms_per_window": e05_latency_ms,
        "variant": e05_diag["selected"],
        "params_added": 0 if e05_fit["kind"] == "identity" else (
            int(np.prod(e05_fit["reviser"]["weights"].shape)) + int(e05_fit["detector"]["model"]["weight"].size)
        ),
        "selection_scores": e05_diag,
        "revised_fraction": float(e05_mask.mean()),
        "risk_mean": float(e05_risk.mean()),
        "horizon_mse": base.horizon_mse(e05_pred, test_arrays["trues"]).tolist(),
    }

    results["raw"]["mae"] = raw_mae
    results["raw"]["mse"] = raw_mse
    return results


def tracker_row(
    config: base.RunConfig,
    method: str,
    method_result: dict[str, Any],
    raw_result: dict[str, Any],
) -> list[Any]:
    scheme_map = {
        "E01": "Horizon-wise bias / scale calibrator",
        "E02": "EMA residual memory",
        "E03": "Linear residual adapter",
        "E04": "Drift-gated residual adapter",
        "E05": "Failure detector + selective revision",
    }
    uses_recent_gt = "Yes" if method == "E02" else "No"
    online_update = "Yes" if method == "E02" else "No"
    delta = method_result["delta_mse_pct"]
    kept = "yes" if delta < 0 else "no"
    failure_note = "" if delta < 0 else "No gain on this config"
    if delta <= -1.0:
        next_action = "Promote to next stage"
    elif delta < 0:
        next_action = "Keep as secondary baseline"
    else:
        next_action = "Retune or drop"
    params_added = method_result["params_added"]
    if method == "E02":
        params_added = f"0 (state={method_result['state_size']})"
    return [
        date.today().isoformat(),
        method,
        scheme_map[method],
        config.model,
        config.dataset,
        config.pred_len,
        "standard test",
        2021,
        uses_recent_gt,
        online_update,
        raw_result["mse"],
        method_result["mse"],
        delta,
        raw_result["mae"],
        method_result["mae"],
        None,
        None,
        method_result["latency_ms_per_window"],
        params_added,
        kept,
        failure_note,
        next_action,
    ]


def update_workbook(all_rows: list[list[Any]], expected_runs_per_method: int) -> None:
    if not WORKBOOK_PATH.exists():
        return

    wb = openpyxl.load_workbook(WORKBOOK_PATH)
    tracker_ws = wb["Run_Tracker"]
    if tracker_ws.max_row > 1:
        tracker_ws.delete_rows(2, tracker_ws.max_row - 1)
    for row in all_rows:
        tracker_ws.append(row)

    counts = {method: 0 for method in METHODS}
    for row in all_rows:
        exp_id = row[1]
        if exp_id in counts:
            counts[exp_id] += 1

    matrix_ws = wb["Experiment_Matrix"]
    for row_idx in range(2, matrix_ws.max_row + 1):
        exp_id = matrix_ws.cell(row=row_idx, column=1).value
        if exp_id in counts:
            count = counts[exp_id]
            status = "done" if count >= expected_runs_per_method else f"partial ({count}/{expected_runs_per_method})"
            matrix_ws.cell(row=row_idx, column=17).value = status
    wb.save(WORKBOOK_PATH)


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
        raw_mse = run_result["raw"]["mse"]
        best_method = min(METHODS, key=lambda name: run_result[name]["mse"])
        wins[best_method] += 1
        for method in METHODS:
            method_scores[method].append(100.0 * (run_result[method]["mse"] - raw_mse) / raw_mse)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage A post-hoc experiments (E01-E05) on TSLib backbones.")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of models to run.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Subset of datasets to run.")
    parser.add_argument("--pred-lens", nargs="*", type=int, default=None, help="Subset of pred_lens to run.")
    parser.add_argument("--train-epochs", type=int, default=10, help="Epochs for raw backbones.")
    parser.add_argument("--force-train", action="store_true", help="Retrain raw backbones even if checkpoints exist.")
    parser.add_argument("--force-export", action="store_true", help="Re-export val/test arrays even if cached.")
    parser.add_argument("--skip-train", action="store_true", help="Only run post-hoc stages on existing checkpoints.")
    parser.add_argument("--no-workbook", action="store_true", help="Skip workbook updates.")
    parser.add_argument("--ema-alpha-grid", nargs="*", type=float, default=[0.0003, 0.001, 0.003, 0.01, 0.03, 0.1])
    parser.add_argument("--ridge-alpha-grid", nargs="*", type=float, default=[300.0, 1000.0, 3000.0, 10000.0, 30000.0])
    parser.add_argument("--gate-alpha-grid", nargs="*", type=float, default=[10.0, 100.0, 300.0])
    parser.add_argument("--revise-fractions", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--detector-alpha", type=float, default=100.0)
    parser.add_argument("--e03-min-tune-gain-pct", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    configs = base.filtered_configs(
        base.w1_default_configs(train_epochs=cli.train_epochs),
        cli.models,
        cli.datasets,
        cli.pred_lens,
    )
    if not configs:
        raise SystemExit("No configs selected.")

    per_run: dict[str, Any] = {}
    tracker_rows: list[list[Any]] = []

    for config in configs:
        if not cli.skip_train:
            base.run_training(config, force=cli.force_train)
        val_arrays = base.collect_split_arrays(config, "val", force=cli.force_export)
        test_arrays = base.collect_split_arrays(config, "test", force=cli.force_export)
        results = evaluate_stage_a(
            config,
            val_arrays,
            test_arrays,
            alpha_grid_ema=cli.ema_alpha_grid,
            alpha_grid_ridge=cli.ridge_alpha_grid,
            e03_min_tune_gain_pct=cli.e03_min_tune_gain_pct,
            gate_alpha_grid=cli.gate_alpha_grid,
            revise_fraction_grid=cli.revise_fractions,
            detector_alpha=cli.detector_alpha,
        )
        run_key = f"{config.model}-{config.dataset}-pl{config.pred_len}"
        per_run[run_key] = results
        for method in METHODS:
            tracker_rows.append(tracker_row(config, method, results[method], results["raw"]))

        metrics_path = config.export_dir / "posthoc_stage_a_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    write_csv(tracker_rows)
    write_summary(aggregate_summary(per_run))
    if not cli.no_workbook:
        update_workbook(tracker_rows, expected_runs_per_method=len(base.w1_default_configs()))


if __name__ == "__main__":
    main()
