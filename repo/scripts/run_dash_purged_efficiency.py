"""Controlled efficiency benchmark for leakage-audited DASH.

The default scope is a representative 18-configuration panel:
3 datasets x 3 backbones x 2 horizons x 1 seed.  Each configuration is run
four times in a single-GPU serial queue; repeat zero warms filesystem/runtime
caches and repeats one to three are the reported measurements.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dash_purged_protocol as pp  # noqa: E402
import run_cf_long_memory_anchor as lm  # noqa: E402
import run_cf_residual_expert as core  # noqa: E402
import run_dash_purged_full504 as dash  # noqa: E402
import run_internal_offline_candidates as base  # noqa: E402
import run_e06_new_backbones as nb  # noqa: E402
import dash_selected_default as selected_default  # noqa: E402


def rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024.0**2)
    except Exception:
        return float("nan")


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def prepare_dash(
    val: dict[str, np.ndarray],
    horizon: int,
) -> dict[str, Any]:
    bins = lm.cf.horizon_bin_ids(horizon, 4)
    protocol = selected_default.protocol(len(val["preds"]), horizon)
    stable_params, recent_params = selected_default.branch_params()
    stable = pp.prepare_state(stable_params, val, bins, protocol)
    recent = pp.prepare_state(recent_params, val, bins, protocol)
    thresholds = dash.fit_thresholds(stable, recent, val, bins, protocol)
    quantiles, alphas = dash.tune_router(
        stable, recent, val, bins, protocol, thresholds
    )
    quantiles, alphas, route = dash.route_once_on_v3(
        stable,
        recent,
        val,
        bins,
        protocol,
        thresholds,
        quantiles,
        alphas,
    )
    return {
        "bins": bins,
        "stable": stable,
        "recent": recent,
        "thresholds": thresholds,
        "quantiles": quantiles,
        "alphas": alphas,
        "route": route,
    }


def correct_test(
    val: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    method: dict[str, Any],
) -> tuple[float, float, float]:
    bins = method["bins"]
    stable = method["stable"]
    recent = method["recent"]
    timeline, current, _ = lm.test_timeline(
        val["inputs"], val["trues"], test["inputs"]
    )
    for state in (stable, recent):
        state["timeline"] = timeline
        state["current"] = current
    sq = ab = count = 0.0
    origins = np.arange(len(test["preds"]), dtype=np.int64)
    step = lm.chunk_size(test["preds"])
    for begin in range(0, len(origins), step):
        selected = origins[begin : begin + step]
        stable_correction = pp.correction_from_state(
            stable, test["inputs"], test["preds"], selected, bins
        )
        recent_correction = pp.correction_from_state(
            recent, test["inputs"], test["preds"], selected, bins
        )
        scores = dash.shock_score(
            recent_correction - stable_correction, bins
        )
        correction = recent_correction.copy()
        for bin_index in range(4):
            horizons = np.where(bins == bin_index)[0]
            quantile = method["quantiles"][bin_index]
            trigger = (
                np.zeros(len(selected), dtype=bool)
                if quantile == "never"
                else scores[:, bin_index]
                > method["thresholds"][float(quantile)][bin_index]
            )
            correction[:, horizons] += (
                method["alphas"][bin_index]
                * trigger[:, None, None]
                * (
                    stable_correction[:, horizons]
                    - recent_correction[:, horizons]
                )
            )
        error = test["preds"][selected] + correction - test["trues"][selected]
        sq += float(np.sum(np.square(error, dtype=np.float64)))
        ab += float(np.sum(np.abs(error), dtype=np.float64))
        count += int(error.size)
    return sq / count, ab / count, float(len(origins))


def prepare_matched(
    val: dict[str, np.ndarray],
    horizon: int,
) -> dict[str, Any]:
    bins = lm.cf.horizon_bin_ids(horizon, 4)
    protocol = pp.PurgedTargetProtocol.build(len(val["preds"]), horizon)
    state = dash.matched_baseline_state(val, bins, protocol)
    gamma, route = dash.gate_matched_baseline_on_v3(
        state, val, bins, protocol
    )
    state["gamma"] = gamma
    return {"bins": bins, "state": state, "route": route}


def correct_matched(
    val: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    method: dict[str, Any],
) -> tuple[float, float, float]:
    state = method["state"]
    timeline, current, _ = lm.test_timeline(
        val["inputs"], val["trues"], test["inputs"]
    )
    state["timeline"] = timeline
    state["current"] = current
    sq = ab = count = 0.0
    origins = np.arange(len(test["preds"]), dtype=np.int64)
    step = lm.chunk_size(test["preds"])
    for begin in range(0, len(origins), step):
        selected = origins[begin : begin + step]
        correction = pp.correction_from_state(
            state,
            test["inputs"],
            test["preds"],
            selected,
            method["bins"],
        )
        error = test["preds"][selected] + correction - test["trues"][selected]
        sq += float(np.sum(np.square(error, dtype=np.float64)))
        ab += float(np.sum(np.abs(error), dtype=np.float64))
        count += int(error.size)
    return sq / count, ab / count, float(len(origins))


def row_path(root: Path, config: nb.BackboneConfig, repeat: int) -> Path:
    safe = core.config_key(config).replace("|", "__")
    return root / "rows" / f"{safe}__repeat{repeat}.csv"


def run_one(
    config: nb.BackboneConfig,
    repeat: int,
    out_root: Path,
    expected: dict[str, float],
    force: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    path = row_path(out_root, config, repeat)
    if path.exists() and not force:
        return None, None
    started = time.perf_counter()
    rss_start = rss_mb()
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        synchronize()
        export_started = time.perf_counter()
        val0, test0 = nb.collect_val_test_arrays_in_memory(config)
        synchronize()
        export_sec = time.perf_counter() - export_started
        val = {
            name: np.asarray(val0[name], dtype=np.float32)
            for name in ("preds", "trues", "inputs")
        }
        test = {
            name: np.asarray(test0[name], dtype=np.float32)
            for name in ("preds", "trues", "inputs")
        }
        raw_mse, raw_mae = base.metrics_chunked(test["preds"], test["trues"])
        if not (
            np.isclose(raw_mse, expected["raw_mse"], atol=1e-6, rtol=1e-6)
            and np.isclose(raw_mae, expected["raw_mae"], atol=1e-6, rtol=1e-6)
        ):
            raise ValueError(
                f"Raw mismatch {(raw_mse, raw_mae)} vs {expected}"
            )

        dash_calibration_started = time.perf_counter()
        dash_method = prepare_dash(val, config.pred_len)
        dash_calibration_sec = (
            time.perf_counter() - dash_calibration_started
        )
        dash_correction_started = time.perf_counter()
        dash_mse, dash_mae, origins = correct_test(val, test, dash_method)
        synchronize()
        dash_correction_sec = time.perf_counter() - dash_correction_started

        matched_calibration_started = time.perf_counter()
        matched_method = prepare_matched(val, config.pred_len)
        matched_calibration_sec = (
            time.perf_counter() - matched_calibration_started
        )
        matched_correction_started = time.perf_counter()
        matched_mse, matched_mae, matched_origins = correct_matched(
            val, test, matched_method
        )
        synchronize()
        matched_correction_sec = (
            time.perf_counter() - matched_correction_started
        )
        if origins != matched_origins:
            raise ValueError("DASH/matched origin counts differ")

        common = {
            "dataset": core.display_dataset(config.dataset),
            "backbone": config.model,
            "pred_len": config.pred_len,
            "seed": config.seed,
            "config_key": core.config_key(config),
            "repeat": repeat,
            "is_warmup": repeat == 0,
            "device": (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available()
                else "CPU"
            ),
            "raw_export_sec": export_sec,
            "validation_origins": len(val["preds"]),
            "test_origins": int(origins),
            "gpu_peak_allocated_mb": (
                torch.cuda.max_memory_allocated() / (1024.0**2)
                if torch.cuda.is_available()
                else 0.0
            ),
            "rss_start_mb": rss_start,
            "rss_end_mb": rss_mb(),
            "raw_mse": raw_mse,
            "raw_mae": raw_mae,
            "trainable_parameters_added": 0,
            "test_time_backward_calls": 0,
            "history_steps": 1008,
            "selection_protocol": "target_timestamp_purged_V1_V2_V3",
            "cycles": selected_default.CYCLES,
            "recent_decay": selected_default.RECENT_DECAY,
            "period_set": "P96+168+336",
            "validation_fraction": selected_default.VALIDATION_FRACTION,
        }
        rows = []
        for (
            method_name,
            calibration_sec,
            correction_sec,
            method_mse,
            method_mae,
        ) in (
            ("Raw", 0.0, 0.0, raw_mse, raw_mae),
            (
                dash.DASH_METHOD,
                dash_calibration_sec,
                dash_correction_sec,
                dash_mse,
                dash_mae,
            ),
            (
                dash.MATCHED_BASELINE,
                matched_calibration_sec,
                matched_correction_sec,
                matched_mse,
                matched_mae,
            ),
        ):
            isolated_total = export_sec + calibration_sec + correction_sec
            rows.append(
                {
                    **common,
                    "method": method_name,
                    "calibration_sec": calibration_sec,
                    "test_correction_sec": correction_sec,
                    "end_to_end_sec": isolated_total,
                    "calibration_ms_per_validation_origin": 1000.0
                    * calibration_sec
                    / max(len(val["preds"]), 1),
                    "correction_ms_per_test_origin": 1000.0
                    * correction_sec
                    / max(origins, 1),
                    "end_to_end_test_origins_per_sec": origins
                    / max(isolated_total, 1e-12),
                    "method_mse": method_mse,
                    "method_mae": method_mae,
                    "timing_boundary": (
                        "validation_plus_test_backbone_export"
                        if method_name == "Raw"
                        else "backbone_export_plus_validation_calibration_plus_test"
                    ),
                }
            )
        core.atomic_csv(path, pd.DataFrame(rows))
        return {"rows": rows, "wall_sec": time.perf_counter() - started}, None
    except Exception as exc:
        return None, {
            "config_key": core.config_key(config),
            "repeat": repeat,
            "type": type(exc).__name__,
            "reason": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--raw-metrics-csv", type=Path, required=True)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs/dash_purged_efficiency_20260725"),
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["DLinear", "iTransformer", "OLS"],
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["ETTh1", "Traffic", "Weather"],
    )
    parser.add_argument("--pred-lens", nargs="*", type=int, default=[96, 720])
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.repeats < 2:
        raise ValueError("Use at least one warmup and one measured repeat")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    core.worker_init(args.checkpoint_root, [str(args.gpu_id)])
    raw = core.load_raw(args.raw_metrics_csv)
    configs = [
        nb.BackboneConfig(
            model=model,
            dataset=core.canonical_dataset(dataset),
            pred_len=horizon,
            seed=args.seed,
        )
        for model in args.models
        for dataset in args.datasets
        for horizon in args.pred_lens
    ]
    args.out_root.mkdir(parents=True, exist_ok=True)
    task_count = len(configs) * args.repeats
    failures: list[dict[str, Any]] = []
    started = time.time()
    seen = 0
    for repeat in range(args.repeats):
        for config in configs:
            expected = raw[
                (
                    config.model,
                    core.display_dataset(config.dataset),
                    config.pred_len,
                    config.seed,
                )
            ]
            _, failure = run_one(
                config,
                repeat,
                args.out_root,
                expected,
                args.force,
            )
            seen += 1
            if failure:
                failures.append(failure)
            core.atomic_json(
                args.out_root / "runner_status.json",
                {
                    "status": "running",
                    "seen": seen,
                    "expected": task_count,
                    "failures": len(failures),
                    "last": core.config_key(config),
                    "repeat": repeat,
                    "elapsed_sec": round(time.time() - started, 2),
                },
            )
            print(
                seen,
                task_count,
                core.config_key(config),
                repeat,
                "failed" if failure else "completed",
                flush=True,
            )

    files = sorted((args.out_root / "rows").glob("*.csv"))
    detail = (
        pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        if files
        else pd.DataFrame()
    )
    core.atomic_csv(args.out_root / "detail.csv", detail)
    measured = (
        detail[~detail["is_warmup"].astype(bool)].copy()
        if not detail.empty
        else detail
    )
    summary = (
        measured.groupby("method", observed=True)
        .agg(
            runs=("config_key", "count"),
            configs=("config_key", "nunique"),
            median_end_to_end_sec=("end_to_end_sec", "median"),
            mean_end_to_end_sec=("end_to_end_sec", "mean"),
            median_calibration_sec=("calibration_sec", "median"),
            median_test_correction_sec=("test_correction_sec", "median"),
            median_test_origins_per_sec=(
                "end_to_end_test_origins_per_sec",
                "median",
            ),
            median_gpu_peak_mb=("gpu_peak_allocated_mb", "median"),
            max_gpu_peak_mb=("gpu_peak_allocated_mb", "max"),
        )
        .reset_index()
        if not measured.empty
        else pd.DataFrame()
    )
    core.atomic_csv(args.out_root / "summary.csv", summary)
    core.atomic_csv(args.out_root / "failures.csv", pd.DataFrame(failures))
    core.atomic_json(
        args.out_root / "runner_status.json",
        {
            "status": "completed" if not failures else "incomplete",
            "tasks": task_count,
            "method_rows": len(detail),
            "expected_method_rows": task_count * 3,
            "measured_method_rows": len(measured),
            "configs": len(configs),
            "failures": len(failures),
            "elapsed_sec": round(time.time() - started, 2),
        },
    )


if __name__ == "__main__":
    main()
