"""Prespecified period-set sensitivity under the purged DASH protocol."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import multiprocessing as mp
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dash_purged_protocol as pp  # noqa: E402
import run_cf_long_memory_anchor as lm  # noqa: E402
import run_cf_residual_expert as core  # noqa: E402
import run_dash_purged_full504 as dash  # noqa: E402
from run_dash_purged_validation_budget_full504 import (  # noqa: E402
    BudgetPurgedTargetProtocol,
)
import run_internal_offline_candidates as base  # noqa: E402
import run_e06_new_backbones as nb  # noqa: E402


PERIOD_SETS: dict[str, list[int]] = {
    "P96": [96],
    "P168": [168],
    "P336": [336],
    "P96+168": [96, 168],
    "P96+336": [96, 336],
    "P168+336": [168, 336],
    "P96+168+336": [96, 168, 336],
}


def evaluate_period_set(
    config: nb.BackboneConfig,
    val: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    name: str,
    periods: list[int],
    validation_fraction: float,
    cycles: int,
    recent_decay: float,
) -> dict[str, Any]:
    bins = lm.cf.horizon_bin_ids(config.pred_len, 4)
    protocol = (
        pp.PurgedTargetProtocol.build(len(val["preds"]), config.pred_len)
        if validation_fraction == 1.0
        else BudgetPurgedTargetProtocol.build(
            len(val["preds"]), config.pred_len, validation_fraction
        )
    )
    stable_params = copy.deepcopy(lm.VARIANT_BY_ID["lm06_median3"])
    recent_params = copy.deepcopy(lm.VARIANT_BY_ID["lm07_ew3"])
    stable_params["periods"] = list(periods)
    recent_params["periods"] = list(periods)
    stable_params["cycles"] = int(cycles)
    recent_params["cycles"] = int(cycles)
    recent_params["decay"] = float(recent_decay)
    stable_params["variant_id"] = f"stable_{name}"
    recent_params["variant_id"] = f"recent_{name}"
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

    timeline, current, stitch = lm.test_timeline(
        val["inputs"], val["trues"], test["inputs"]
    )
    for state in (stable, recent):
        state["timeline"] = timeline
        state["current"] = current

    raw_sq = raw_ab = method_sq = method_ab = 0.0
    count = trigger_count = 0
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
            quantile = quantiles[bin_index]
            trigger = (
                np.zeros(len(selected), dtype=bool)
                if quantile == "never"
                else scores[:, bin_index]
                > thresholds[float(quantile)][bin_index]
            )
            correction[:, horizons] += (
                alphas[bin_index]
                * trigger[:, None, None]
                * (
                    stable_correction[:, horizons]
                    - recent_correction[:, horizons]
                )
            )
            trigger_count += int(np.sum(trigger))
        raw_error = test["preds"][selected] - test["trues"][selected]
        method_error = raw_error + correction
        raw_sq += float(np.sum(np.square(raw_error, dtype=np.float64)))
        raw_ab += float(np.sum(np.abs(raw_error), dtype=np.float64))
        method_sq += float(
            np.sum(np.square(method_error, dtype=np.float64))
        )
        method_ab += float(np.sum(np.abs(method_error), dtype=np.float64))
        count += int(raw_error.size)

    raw_mse = raw_sq / count
    raw_mae = raw_ab / count
    method_mse = method_sq / count
    method_mae = method_ab / count
    return {
        "dataset": core.display_dataset(config.dataset),
        "backbone": config.model,
        "pred_len": config.pred_len,
        "seed": config.seed,
        "config_key": core.config_key(config),
        "period_set": name,
        "periods_json": json.dumps(periods),
        "period_count": len(periods),
        "available_history_steps": 1008,
        "maximum_periodic_lag": max(periods) * 3,
        "raw_mse": raw_mse,
        "raw_mae": raw_mae,
        "method_mse": method_mse,
        "method_mae": method_mae,
        "gain_pct": 100.0 * (raw_mse - method_mse) / raw_mse,
        "mae_gain_pct": 100.0 * (raw_mae - method_mae) / raw_mae,
        "trigger_rate": trigger_count / max(len(origins) * 4, 1),
        "validation_fraction": validation_fraction,
        "cycles": int(cycles),
        "recent_decay": float(recent_decay),
        "retained_validation_origins": int(
            len(val["preds"]) - getattr(protocol, "origin_start", 0)
        ),
        "selection_protocol": (
            "target_timestamp_purged_V1_V2_V3"
            if validation_fraction == 1.0
            else "latest_origin_budget_target_timestamp_purged_V1_V2_V3"
        ),
        "uses_test_for_period_selection": False,
        "route_json": json.dumps(route, sort_keys=True),
        "stitch_json": json.dumps(stitch, sort_keys=True),
    }


def paths(root: Path, key: str) -> tuple[Path, Path]:
    safe = key.replace("|", "__")
    return root / "status" / f"{safe}.json", root / "rows" / f"{safe}.csv"


def run_one(payload: dict[str, Any]) -> dict[str, Any]:
    config = nb.BackboneConfig(**payload["config"])
    key = core.config_key(config)
    status_path, row_path = paths(Path(payload["out_root"]), key)
    if row_path.exists() and not payload["force"]:
        return {"key": key, "status": "skipped", "failure": None}
    started = time.time()
    try:
        val0, test0 = nb.collect_val_test_arrays_in_memory(config)
        val = {
            name: np.asarray(val0[name], dtype=np.float32)
            for name in ("preds", "trues", "inputs")
        }
        test = {
            name: np.asarray(test0[name], dtype=np.float32)
            for name in ("preds", "trues", "inputs")
        }
        measured = base.metrics_chunked(test["preds"], test["trues"])
        expected = payload["raw"]
        if not (
            np.isclose(measured[0], expected["raw_mse"], atol=1e-6, rtol=1e-6)
            and np.isclose(measured[1], expected["raw_mae"], atol=1e-6, rtol=1e-6)
        ):
            raise ValueError(f"Raw mismatch {measured} vs {expected}")
        rows = [
            evaluate_period_set(
                config,
                val,
                test,
                name,
                periods,
                float(payload["validation_fraction"]),
                int(payload["cycles"]),
                float(payload["recent_decay"]),
            )
            for name, periods in PERIOD_SETS.items()
        ]
        core.atomic_csv(row_path, pd.DataFrame(rows))
        core.atomic_json(
            status_path,
            {
                "status": "completed",
                "period_sets": len(rows),
                "elapsed_sec": round(time.time() - started, 3),
            },
        )
        return {"key": key, "status": "completed", "failure": None}
    except Exception as exc:
        failure = {
            "config_key": key,
            "status": "failed",
            "type": type(exc).__name__,
            "reason": str(exc),
            "traceback": traceback.format_exc(),
        }
        core.atomic_json(status_path, failure)
        return {"key": key, "status": "failed", "failure": failure}
    finally:
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--raw-metrics-csv", type=Path, required=True)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs/dash_purged_period_sensitivity_20260725"),
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
    parser.add_argument("--seeds", nargs="*", type=int, default=[2, 42, 2021])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu-ids", nargs="*", default=[str(index) for index in range(8)])
    parser.add_argument(
        "--validation-fraction",
        type=float,
        choices=[0.25, 0.50, 0.75, 1.0],
        default=1.0,
        help="Most recent validation-origin fraction used for calibration.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--recent-decay", type=float, default=0.65)
    args = parser.parse_args()

    raw = core.load_raw(args.raw_metrics_csv)
    configs = [
        nb.BackboneConfig(
            model=model,
            dataset=core.canonical_dataset(dataset),
            pred_len=horizon,
            seed=seed,
        )
        for model in args.models
        for dataset in args.datasets
        for horizon in args.pred_lens
        for seed in args.seeds
    ]
    if args.limit:
        configs = configs[: args.limit]
    payloads = [
        {
            "config": asdict(config),
            "raw": raw[
                (
                    config.model,
                    core.display_dataset(config.dataset),
                    config.pred_len,
                    config.seed,
                )
            ],
            "out_root": str(args.out_root),
            "validation_fraction": args.validation_fraction,
            "cycles": args.cycles,
            "recent_decay": args.recent_decay,
            "force": args.force,
        }
        for config in configs
    ]
    args.out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    failures: list[dict[str, Any]] = []
    core.atomic_json(
        args.out_root / "runner_status.json",
        {
            "status": "running",
            "expected_configs": len(configs),
            "period_sets": list(PERIOD_SETS),
            "validation_fraction": args.validation_fraction,
        },
    )
    with mp.Pool(
        min(args.workers, len(payloads)),
        initializer=core.worker_init,
        initargs=(args.checkpoint_root, list(args.gpu_ids)),
    ) as pool:
        for seen, result in enumerate(pool.imap_unordered(run_one, payloads), 1):
            if result["failure"]:
                failures.append(result["failure"])
            core.atomic_json(
                args.out_root / "runner_status.json",
                {
                    "status": "running",
                    "seen_configs": seen,
                    "expected_configs": len(configs),
                    "period_sets": len(PERIOD_SETS),
                    "failures": len(failures),
                    "validation_fraction": args.validation_fraction,
                    "last": result["key"],
                    "elapsed_sec": round(time.time() - started, 2),
                },
            )
            print(seen, len(configs), result["key"], result["status"], flush=True)

    files = sorted((args.out_root / "rows").glob("*.csv"))
    detail = (
        pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        if files
        else pd.DataFrame()
    )
    core.atomic_csv(args.out_root / "detail.csv", detail)
    summary = (
        detail.groupby("period_set", observed=True)
        .agg(
            configs=("config_key", "nunique"),
            mean_mse=("method_mse", "mean"),
            mean_mae=("method_mae", "mean"),
            mean_gain_pct=("gain_pct", "mean"),
            mean_mae_gain_pct=("mae_gain_pct", "mean"),
            harmful=("gain_pct", lambda values: int((values < 0.0).sum())),
        )
        .reset_index()
        if not detail.empty
        else pd.DataFrame()
    )
    core.atomic_csv(args.out_root / "summary.csv", summary)
    core.atomic_csv(args.out_root / "failures.csv", pd.DataFrame(failures))
    done = int(detail["config_key"].nunique()) if not detail.empty else 0
    core.atomic_json(
        args.out_root / "runner_status.json",
        {
            "status": (
                "completed"
                if not failures and done == len(configs)
                else "incomplete"
            ),
            "configs": done,
            "expected_configs": len(configs),
            "rows": len(detail),
            "expected_rows": len(configs) * len(PERIOD_SETS),
            "failures": len(failures),
            "validation_fraction": args.validation_fraction,
            "elapsed_sec": round(time.time() - started, 2),
        },
    )


if __name__ == "__main__":
    main()
