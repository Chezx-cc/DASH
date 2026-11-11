"""Prespecified beta sensitivity on the 54-config DASH-Purged panel.

The already completed period-set sensitivity is intentionally not rerun.
This script varies recent decay beta in {0.40, 0.55, 0.65, 0.75, 0.90},
with K fixed at 3 and the complete period set fixed at {96, 168, 336}.

Every setting refits V1, retunes V2, performs one V3 route-safety decision,
freezes calibration, and only then evaluates the test set.
"""
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


SETTINGS: tuple[dict[str, Any], ...] = (
    {
        "axis": "recent_decay",
        "value": 0.40,
        "cycles": 3,
        "decay": 0.40,
    },
    {
        "axis": "recent_decay",
        "value": 0.55,
        "cycles": 3,
        "decay": 0.55,
    },
    {
        "axis": "recent_decay",
        "value": 0.65,
        "cycles": 3,
        "decay": 0.65,
    },
    {
        "axis": "recent_decay",
        "value": 0.75,
        "cycles": 3,
        "decay": 0.75,
    },
    {
        "axis": "recent_decay",
        "value": 0.90,
        "cycles": 3,
        "decay": 0.90,
    },
)
FULL_PERIOD_SET = [96, 168, 336]


def evaluate_effective_setting(
    config: nb.BackboneConfig,
    val: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    cycles: int,
    decay: float,
    validation_fraction: float,
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
    stable_params["periods"] = list(FULL_PERIOD_SET)
    recent_params["periods"] = list(FULL_PERIOD_SET)
    stable_params["cycles"] = int(cycles)
    recent_params["cycles"] = int(cycles)
    recent_params["decay"] = float(decay)
    stable_params["variant_id"] = f"stable_K{cycles}"
    recent_params["variant_id"] = f"recent_K{cycles}_beta{decay:.2f}"

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
        state["timeline"], state["current"] = timeline, current

    raw_sq = raw_ab = method_sq = method_ab = correction_abs = 0.0
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
        for b in range(int(bins.max()) + 1):
            horizons = np.where(bins == b)[0]
            if quantiles[b] == "never":
                trigger = np.zeros(len(selected), dtype=bool)
            else:
                trigger = (
                    scores[:, b] > thresholds[float(quantiles[b])][b]
                )
            correction[:, horizons] += (
                alphas[b]
                * trigger[:, None, None]
                * (
                    stable_correction[:, horizons]
                    - recent_correction[:, horizons]
                )
            )
            trigger_count += int(np.sum(trigger))
        raw_error = test["preds"][selected] - test["trues"][selected]
        error = raw_error + correction
        raw_sq += float(np.sum(np.square(raw_error, dtype=np.float64)))
        raw_ab += float(np.sum(np.abs(raw_error), dtype=np.float64))
        method_sq += float(np.sum(np.square(error, dtype=np.float64)))
        method_ab += float(np.sum(np.abs(error), dtype=np.float64))
        correction_abs += float(np.sum(np.abs(correction), dtype=np.float64))
        count += int(error.size)

    raw_mse, raw_mae = raw_sq / count, raw_ab / count
    mse, mae = method_sq / count, method_ab / count
    return {
        "dataset": core.display_dataset(config.dataset),
        "backbone": config.model,
        "model": config.model,
        "pred_len": config.pred_len,
        "seed": config.seed,
        "config_key": core.config_key(config),
        "raw_mse": raw_mse,
        "raw_mae": raw_mae,
        "method_mse": mse,
        "method_mae": mae,
        "delta_mse_pct": 100.0 * (mse - raw_mse) / raw_mse,
        "gain_pct": 100.0 * (raw_mse - mse) / raw_mse,
        "mae_gain_pct": 100.0 * (raw_mae - mae) / raw_mae,
        "cycles": int(cycles),
        "recent_decay": float(decay),
        "period_set": "P96+168+336",
        "periods_json": json.dumps(FULL_PERIOD_SET),
        "validation_fraction": float(validation_fraction),
        "retained_validation_origins": (
            len(val["preds"])
            if validation_fraction == 1.0
            else len(val["preds"]) - protocol.origin_start
        ),
        "active": True,
        "fallback": False,
        "fallback_type": "none",
        "fallback_reason": "none",
        "selection_protocol": "target_timestamp_purged_V1_V2_V3",
        "uses_external_outputs": False,
        "uses_test_for_selection": False,
        "test_correction_reads_test_truth": False,
        "correction_norm_mean": correction_abs / count,
        "meta_json": json.dumps(
            {
                **stitch,
                "purged_protocol": protocol.audit(),
                "threshold_quantiles": quantiles,
                "alpha": alphas.tolist(),
                "route": route,
                "stable_gamma": stable["gamma"].tolist(),
                "recent_gamma": recent["gamma"].tolist(),
                "triggered_origin_bins": trigger_count,
                "one_factor_at_a_time": True,
                "full_period_set_fixed": True,
                "uses_test_for_hyperparameter_selection": False,
            },
            sort_keys=True,
        ),
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
            and np.isclose(
                measured[1], expected["raw_mae"], atol=1e-6, rtol=1e-6
            )
        ):
            raise ValueError(f"Raw mismatch {measured} vs {expected}")

        cache: dict[tuple[int, float], dict[str, Any]] = {}
        rows: list[dict[str, Any]] = []
        for setting in SETTINGS:
            signature = (int(setting["cycles"]), float(setting["decay"]))
            if signature not in cache:
                cache[signature] = evaluate_effective_setting(
                    config,
                    val,
                    test,
                    *signature,
                    float(payload["validation_fraction"]),
                )
            row = dict(cache[signature])
            row["sensitivity_axis"] = setting["axis"]
            row["sensitivity_value"] = setting["value"]
            row["is_default"] = bool(
                setting.get(
                    "is_default",
                    setting["axis"] == "recent_decay"
                    and setting["value"] == 0.65,
                )
            )
            row["method"] = (
                f"dash_purged_{setting['axis']}_{setting['value']}"
            )
            rows.append(row)
        rows = core.align_raw(rows, expected, measured)
        core.atomic_csv(row_path, pd.DataFrame(rows))
        core.atomic_json(
            status_path,
            {
                "status": "completed",
                "rows": len(rows),
                "effective_settings": len(cache),
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
        return {
            "key": key,
            "status": "failed",
            "failure": failure,
        }
    finally:
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--raw-metrics-csv", type=Path, required=True)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs/dash_purged_hyperparam_20260725"),
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
    parser.add_argument(
        "--pred-lens", nargs="*", type=int, default=[96, 720]
    )
    parser.add_argument(
        "--seeds", nargs="*", type=int, default=[2, 42, 2021]
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--gpu-ids",
        nargs="*",
        default=[str(index) for index in range(8)],
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        choices=[0.25, 0.50, 0.75, 1.0],
        default=1.0,
    )
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
            "force": args.force,
            "validation_fraction": args.validation_fraction,
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
            "settings": len(SETTINGS),
            "expected_rows": len(configs) * len(SETTINGS),
            "validation_fraction": args.validation_fraction,
            "period_set": "P96+168+336",
        },
    )
    with mp.Pool(
        min(args.workers, len(payloads)),
        initializer=core.worker_init,
        initargs=(args.checkpoint_root, list(args.gpu_ids)),
    ) as pool:
        for seen, result in enumerate(
            pool.imap_unordered(run_one, payloads), 1
        ):
            if result["failure"]:
                failures.append(result["failure"])
            core.atomic_json(
                args.out_root / "runner_status.json",
                {
                    "status": "running",
                    "seen_configs": seen,
                    "expected_configs": len(configs),
                    "settings": len(SETTINGS),
                    "failures": len(failures),
                    "last": result["key"],
                    "elapsed_sec": round(time.time() - started, 2),
                    "validation_fraction": args.validation_fraction,
                },
            )
            print(
                seen,
                len(configs),
                result["key"],
                result["status"],
                flush=True,
            )

    files = sorted((args.out_root / "rows").glob("*.csv"))
    detail = (
        pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        if files
        else pd.DataFrame()
    )
    core.atomic_csv(args.out_root / "detail.csv", detail)
    summary = (
        detail.groupby(
            ["sensitivity_axis", "sensitivity_value"], observed=True
        )
        .agg(
            configs=("config_key", "nunique"),
            mean_mse=("method_mse", "mean"),
            mean_mae=("method_mae", "mean"),
            mean_gain_pct=("gain_pct", "mean"),
            mean_mae_gain_pct=("mae_gain_pct", "mean"),
            harmful=(
                "gain_pct",
                lambda values: int((values < 0.0).sum()),
            ),
        )
        .reset_index()
        if not detail.empty
        else pd.DataFrame()
    )
    core.atomic_csv(args.out_root / "summary.csv", summary)
    core.atomic_csv(args.out_root / "failures.csv", pd.DataFrame(failures))
    done = (
        int(detail["config_key"].nunique()) if not detail.empty else 0
    )
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
            "expected_rows": len(configs) * len(SETTINGS),
            "failures": len(failures),
            "elapsed_sec": round(time.time() - started, 2),
            "validation_fraction": args.validation_fraction,
            "period_set": "P96+168+336",
        },
    )


if __name__ == "__main__":
    main()
