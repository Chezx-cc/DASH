"""Full-504 timing and memory audit for the frozen DASH inference pipeline."""
from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_cf_long_memory_anchor as lm
import run_cf_residual_expert as core
import run_dash_shock_guard_full504 as shock
import run_final_full504 as prior
import run_internal_offline_candidates as base
import run_e06_new_backbones as nb


def current_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024.0**2)
    except Exception:
        return float("nan")


def paths(root: Path, key: str):
    safe = key.replace("|", "__")
    return root / "status" / f"{safe}.json", root / "rows" / f"{safe}.csv"


def run_one(payload):
    config = nb.BackboneConfig(**payload["config"])
    key = core.config_key(config)
    status_path, row_path = paths(Path(payload["out_root"]), key)
    if row_path.exists() and not payload["force"]:
        return {"key": key, "status": "skipped", "failure": None}
    started = time.perf_counter()
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        rss_start = current_rss_mb()
        raw_started = time.perf_counter()
        val0, test0 = nb.collect_val_test_arrays_in_memory(config)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        raw_export_sec = time.perf_counter() - raw_started
        val = {key_: np.asarray(val0[key_], dtype=np.float32) for key_ in ("preds", "trues", "inputs")}
        test = {key_: np.asarray(test0[key_], dtype=np.float32) for key_ in ("preds", "trues", "inputs")}
        measured = base.metrics_chunked(test["preds"], test["trues"])
        expected = payload["raw"]
        if not (
            np.isclose(measured[0], expected["raw_mse"], atol=1e-6, rtol=1e-6)
            and np.isclose(measured[1], expected["raw_mae"], atol=1e-6, rtol=1e-6)
        ):
            raise ValueError(f"Raw mismatch {measured} vs {expected}")

        calibration_started = time.perf_counter()
        bins = lm.cf.horizon_bin_ids(config.pred_len, 4)
        stable = prior.prepare(lm.VARIANT_BY_ID["lm06_median3"], val, bins)
        recent = prior.prepare(lm.VARIANT_BY_ID["lm07_ew3"], val, bins)
        thresholds = shock.fit_thresholds(stable, recent, val, bins)
        q_selected, alpha = shock.tune(stable, recent, val, bins, thresholds)
        q_selected, alpha, _ = shock.route(
            stable, recent, val, bins, thresholds, q_selected, alpha
        )
        calibration_sec = time.perf_counter() - calibration_started

        correction_started = time.perf_counter()
        timeline, current, _ = lm.test_timeline(val["inputs"], val["trues"], test["inputs"])
        for state in (stable, recent):
            state["timeline"], state["current"] = timeline, current
        sq = ab = count = trigger_count = 0.0
        indices = np.arange(len(test["preds"]))
        step = lm.chunk_size(test["preds"])
        for start in range(0, len(indices), step):
            ii = indices[start : start + step]
            c6e = prior.corr(stable, test["inputs"], test["preds"], ii, bins)
            c6s = prior.corr(stable, test["inputs"], test["preds"], ii, bins, True)
            c7 = prior.corr(recent, test["inputs"], test["preds"], ii, bins)
            scores = shock.shock_score(c7 - c6e, bins)
            correction = c7.copy()
            for b in range(4):
                h = np.where(bins == b)[0]
                trigger = (
                    np.zeros(len(ii), dtype=bool)
                    if q_selected[b] == "never"
                    else scores[:, b] > thresholds[q_selected[b]][b]
                )
                correction[:, h] += (
                    alpha[b] * trigger[:, None, None] * (c6s[:, h] - c7[:, h])
                )
                trigger_count += int(trigger.sum())
            error = test["preds"][ii] + correction - test["trues"][ii]
            sq += np.sum(np.square(error, dtype=np.float64))
            ab += np.sum(np.abs(error), dtype=np.float64)
            count += error.size
        correction_sec = time.perf_counter() - correction_started
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_sec = time.perf_counter() - started
        row = {
            "dataset": core.display_dataset(config.dataset),
            "backbone": config.model,
            "pred_len": config.pred_len,
            "seed": config.seed,
            "config_key": key,
            "val_origins": len(val["preds"]),
            "test_origins": len(test["preds"]),
            "raw_export_sec": raw_export_sec,
            "calibration_sec": calibration_sec,
            "test_correction_sec": correction_sec,
            "total_sec": total_sec,
            "raw_export_ms_per_origin": 1000.0
            * raw_export_sec
            / max(len(val["preds"]) + len(test["preds"]), 1),
            "dash_correction_ms_per_test_origin": 1000.0
            * correction_sec
            / max(len(test["preds"]), 1),
            "calibration_ms_per_val_origin": 1000.0
            * calibration_sec
            / max(len(val["preds"]), 1),
            "correction_to_raw_export_time_pct": 100.0 * correction_sec / max(raw_export_sec, 1e-12),
            "method_mse": sq / count,
            "method_mae": ab / count,
            "trigger_rate": trigger_count / max(len(test["preds"]) * 4, 1),
            "trainable_parameters_added": 0,
            "test_time_backward_calls": 0,
            "gpu_peak_allocated_mb": (
                torch.cuda.max_memory_allocated() / (1024.0**2)
                if torch.cuda.is_available()
                else 0.0
            ),
            "rss_start_mb": rss_start,
            "rss_end_mb": current_rss_mb(),
        }
        core.atomic_csv(row_path, pd.DataFrame([row]))
        core.atomic_json(status_path, {"status": "completed", "elapsed_sec": round(total_sec, 3)})
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--raw-metrics-csv", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("outputs/dash_efficiency_full504_20260723"))
    parser.add_argument("--models", nargs="*", default=lm.cf.BACKBONES)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Traffic", "Weather"],
    )
    parser.add_argument("--pred-lens", nargs="*", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--seeds", nargs="*", type=int, default=[2, 42, 2021])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu-ids", nargs="*", default=[str(i) for i in range(8)])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    raw = core.load_raw(args.raw_metrics_csv)
    datasets = [core.canonical_dataset(x) for x in args.datasets]
    configs = [
        nb.BackboneConfig(model=m, dataset=d, pred_len=h, seed=s)
        for m in args.models
        for d in datasets
        for h in args.pred_lens
        for s in args.seeds
    ]
    if args.limit:
        configs = configs[: args.limit]
    payloads = [
        {
            "config": asdict(config),
            "raw": raw[(config.model, core.display_dataset(config.dataset), config.pred_len, config.seed)],
            "out_root": str(args.out_root),
            "force": args.force,
        }
        for config in configs
    ]
    started = time.time()
    failures = []
    core.atomic_json(args.out_root / "runner_status.json", {"status": "running", "expected": len(configs)})
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
                    "seen": seen,
                    "expected": len(configs),
                    "failures": len(failures),
                    "last": result["key"],
                    "elapsed_sec": round(time.time() - started, 2),
                },
            )
            print(seen, len(configs), result["key"], result["status"], flush=True)
    files = sorted((args.out_root / "rows").glob("*.csv"))
    detail = pd.concat([pd.read_csv(path) for path in files], ignore_index=True) if files else pd.DataFrame()
    core.atomic_csv(args.out_root / "detail.csv", detail)
    core.atomic_csv(args.out_root / "failures.csv", pd.DataFrame(failures))
    done = int(detail["config_key"].nunique()) if not detail.empty else 0
    core.atomic_json(
        args.out_root / "runner_status.json",
        {
            "status": "completed" if not failures and done == len(configs) else "incomplete",
            "configs": done,
            "expected": len(configs),
            "failures": len(failures),
            "elapsed_sec": round(time.time() - started, 2),
        },
    )


if __name__ == "__main__":
    main()
