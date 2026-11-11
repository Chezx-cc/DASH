"""Leakage-audited DASH main experiment.

Changes from the superseded round-1 runner:
1. V1/V2/V3 are disjoint in target-timestamp space, not merely origin space.
2. V3 makes exactly one decision: accept/reject the complete fallback route.
   The stable branch is not separately gated on V3.
3. A matched PeriodicRidge-1008 baseline receives the same 1008-step history
   and the same V1/V2/V3 label-timestamp budget.
"""
from __future__ import annotations

import argparse
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
import run_internal_offline_candidates as base  # noqa: E402
import run_e06_new_backbones as nb  # noqa: E402


DASH_METHOD = "dash_purged_v2"
MATCHED_BASELINE = "periodic_ridge_1008_purged"
QUANTILES = (0.80, 0.90, 0.95, 0.99)
ALPHAS = (0.25, 0.50, 0.75, 1.00)
CANDIDATES = (("never", 0.0),) + tuple(
    (q, alpha) for q in reversed(QUANTILES) for alpha in ALPHAS
)


def shock_score(disagreement: np.ndarray, bins: np.ndarray) -> np.ndarray:
    out = np.zeros((len(disagreement), int(bins.max()) + 1), dtype=np.float32)
    for b in range(out.shape[1]):
        horizons = np.where(bins == b)[0]
        out[:, b] = np.sqrt(
            np.mean(
                np.square(disagreement[:, horizons], dtype=np.float64),
                axis=(1, 2),
            )
        )
    return out


def fit_thresholds(
    stable: dict[str, Any],
    recent: dict[str, Any],
    val: dict[str, np.ndarray],
    bins: np.ndarray,
    protocol: pp.PurgedTargetProtocol,
) -> dict[float, np.ndarray]:
    origins = pp.sampled(protocol.eligible_origins(0), 96)
    stable_correction = pp.correction_from_state(
        stable, val["inputs"], val["preds"], origins, bins
    )
    recent_correction = pp.correction_from_state(
        recent, val["inputs"], val["preds"], origins, bins
    )
    scores = shock_score(recent_correction - stable_correction, bins)
    return {
        q: np.quantile(scores, q, axis=0).astype(np.float32) for q in QUANTILES
    }


def tune_router(
    stable: dict[str, Any],
    recent: dict[str, Any],
    val: dict[str, np.ndarray],
    bins: np.ndarray,
    protocol: pp.PurgedTargetProtocol,
    thresholds: dict[float, np.ndarray],
) -> tuple[list[float | str], np.ndarray]:
    num_bins = int(bins.max()) + 1
    sums = np.zeros((num_bins, len(CANDIDATES)), dtype=np.float64)
    origins = protocol.eligible_origins(1)
    step = lm.chunk_size(val["preds"])
    for begin in range(0, len(origins), step):
        selected = origins[begin : begin + step]
        stable_correction = pp.correction_from_state(
            stable, val["inputs"], val["preds"], selected, bins
        )
        recent_correction = pp.correction_from_state(
            recent, val["inputs"], val["preds"], selected, bins
        )
        scores = shock_score(recent_correction - stable_correction, bins)
        for b in range(num_bins):
            horizons = np.where(bins == b)[0]
            valid = protocol.mask(1, selected, horizons)
            if not np.any(valid):
                continue
            error = (
                val["preds"][selected][:, horizons]
                + recent_correction[:, horizons]
                - val["trues"][selected][:, horizons]
            )
            difference = stable_correction[:, horizons] - recent_correction[:, horizons]
            for candidate_index, (quantile, alpha) in enumerate(CANDIDATES):
                if quantile == "never":
                    scale = np.zeros(len(selected), dtype=np.float64)
                else:
                    scale = alpha * (scores[:, b] > thresholds[float(quantile)][b])
                candidate = error + scale[:, None, None] * difference
                sums[b, candidate_index] += float(
                    np.sum(np.square(candidate[valid], dtype=np.float64))
                )
    selected_quantiles: list[float | str] = []
    selected_alphas = np.zeros(num_bins, dtype=np.float32)
    for b in range(num_bins):
        index = int(np.argmin(sums[b]))
        quantile, alpha = CANDIDATES[index]
        selected_quantiles.append(quantile)
        selected_alphas[b] = float(alpha)
    return selected_quantiles, selected_alphas


def route_once_on_v3(
    stable: dict[str, Any],
    recent: dict[str, Any],
    val: dict[str, np.ndarray],
    bins: np.ndarray,
    protocol: pp.PurgedTargetProtocol,
    thresholds: dict[float, np.ndarray],
    quantiles: list[float | str],
    alphas: np.ndarray,
) -> tuple[list[float | str], np.ndarray, dict[str, list[Any]]]:
    """Use V3 once to accept/reject the complete fallback candidate."""
    num_bins = int(bins.max()) + 1
    origin_parts: list[list[np.ndarray]] = [[] for _ in range(num_bins)]
    base_parts: list[list[np.ndarray]] = [[] for _ in range(num_bins)]
    candidate_parts: list[list[np.ndarray]] = [[] for _ in range(num_bins)]
    origins = protocol.eligible_origins(2)
    step = lm.chunk_size(val["preds"])
    for begin in range(0, len(origins), step):
        selected = origins[begin : begin + step]
        stable_correction = pp.correction_from_state(
            stable, val["inputs"], val["preds"], selected, bins
        )
        recent_correction = pp.correction_from_state(
            recent, val["inputs"], val["preds"], selected, bins
        )
        scores = shock_score(recent_correction - stable_correction, bins)
        full_error = val["preds"][selected] + recent_correction - val["trues"][selected]
        for b in range(num_bins):
            horizons = np.where(bins == b)[0]
            valid = protocol.mask(2, selected, horizons)
            if quantiles[b] == "never":
                trigger = np.zeros(len(selected), dtype=bool)
            else:
                trigger = scores[:, b] > thresholds[float(quantiles[b])][b]
            candidate_error = full_error.copy()
            candidate_error[:, horizons] += (
                alphas[b]
                * trigger[:, None, None]
                * (stable_correction[:, horizons] - recent_correction[:, horizons])
            )
            oid, raw_loss, method_loss = pp.origin_losses(
                full_error,
                candidate_error,
                selected,
                horizons,
                valid,
            )
            origin_parts[b].append(oid)
            base_parts[b].append(raw_loss)
            candidate_parts[b].append(method_loss)

    gated_quantiles = list(quantiles)
    gated_alphas = alphas.copy()
    audit: dict[str, list[Any]] = {
        "accepted": [],
        "gain": [],
        "positive": [],
        "worst": [],
    }
    for b in range(num_bins):
        ids = np.concatenate(origin_parts[b]) if origin_parts[b] else np.zeros(0)
        raw = np.concatenate(base_parts[b]) if base_parts[b] else np.zeros(0)
        method = (
            np.concatenate(candidate_parts[b]) if candidate_parts[b] else np.zeros(0)
        )
        accepted, gain, positive, worst = pp.safety_decision(ids, raw, method)
        if not accepted:
            gated_quantiles[b] = "never"
            gated_alphas[b] = 0.0
        audit["accepted"].append(bool(accepted))
        audit["gain"].append(float(gain))
        audit["positive"].append(int(positive))
        audit["worst"].append(float(worst))
    return gated_quantiles, gated_alphas, audit


def matched_baseline_state(
    val: dict[str, np.ndarray],
    bins: np.ndarray,
    protocol: pp.PurgedTargetProtocol,
) -> dict[str, Any]:
    specs = [(period, cycle) for period in (96, 168, 336) for cycle in (1, 2, 3)]
    params = {
        "variant_id": MATCHED_BASELINE,
        "periods": [96, 168, 336],
        "cycles": 3,
        "ridge": 0.05,
        "clip_q": 0.95,
    }
    return pp.prepare_state(
        params,
        val,
        bins,
        protocol,
        specs=specs,
        feature_fn=pp.cycle_feature,
    )


def gate_matched_baseline_on_v3(
    state: dict[str, Any],
    val: dict[str, np.ndarray],
    bins: np.ndarray,
    protocol: pp.PurgedTargetProtocol,
) -> tuple[np.ndarray, dict[str, list[Any]]]:
    num_bins = int(bins.max()) + 1
    origins = protocol.eligible_origins(2)
    origin_parts: list[list[np.ndarray]] = [[] for _ in range(num_bins)]
    base_parts: list[list[np.ndarray]] = [[] for _ in range(num_bins)]
    candidate_parts: list[list[np.ndarray]] = [[] for _ in range(num_bins)]
    step = lm.chunk_size(val["preds"])
    for begin in range(0, len(origins), step):
        selected = origins[begin : begin + step]
        correction = pp.correction_from_state(
            state, val["inputs"], val["preds"], selected, bins
        )
        raw_error = val["preds"][selected] - val["trues"][selected]
        candidate_error = raw_error + correction
        for b in range(num_bins):
            horizons = np.where(bins == b)[0]
            valid = protocol.mask(2, selected, horizons)
            oid, raw_loss, method_loss = pp.origin_losses(
                raw_error,
                candidate_error,
                selected,
                horizons,
                valid,
            )
            origin_parts[b].append(oid)
            base_parts[b].append(raw_loss)
            candidate_parts[b].append(method_loss)
    gated = state["gamma"].copy()
    audit: dict[str, list[Any]] = {
        "accepted": [],
        "gain": [],
        "positive": [],
        "worst": [],
    }
    for b in range(num_bins):
        accepted, gain, positive, worst = pp.safety_decision(
            np.concatenate(origin_parts[b]),
            np.concatenate(base_parts[b]),
            np.concatenate(candidate_parts[b]),
        )
        if not accepted:
            gated[b] = 0.0
        audit["accepted"].append(bool(accepted))
        audit["gain"].append(float(gain))
        audit["positive"].append(int(positive))
        audit["worst"].append(float(worst))
    return gated, audit


def evaluate(
    config: nb.BackboneConfig,
    val0: dict[str, np.ndarray],
    test0: dict[str, np.ndarray],
    *,
    protocol_override: Any | None = None,
) -> list[dict[str, Any]]:
    val = {
        key: np.asarray(val0[key], dtype=np.float32)
        for key in ("preds", "trues", "inputs")
    }
    test = {
        key: np.asarray(test0[key], dtype=np.float32)
        for key in ("preds", "trues", "inputs")
    }
    bins = lm.cf.horizon_bin_ids(config.pred_len, 4)
    protocol = (
        protocol_override
        if protocol_override is not None
        else pp.PurgedTargetProtocol.build(len(val["preds"]), config.pred_len)
    )
    protocol_audit = protocol.audit()

    stable = pp.prepare_state(
        lm.VARIANT_BY_ID["lm06_median3"], val, bins, protocol
    )
    recent = pp.prepare_state(
        lm.VARIANT_BY_ID["lm07_ew3"], val, bins, protocol
    )
    thresholds = fit_thresholds(stable, recent, val, bins, protocol)
    quantiles, alphas = tune_router(
        stable, recent, val, bins, protocol, thresholds
    )
    quantiles, alphas, route_audit = route_once_on_v3(
        stable,
        recent,
        val,
        bins,
        protocol,
        thresholds,
        quantiles,
        alphas,
    )

    matched = matched_baseline_state(val, bins, protocol)
    matched_gamma, matched_audit = gate_matched_baseline_on_v3(
        matched, val, bins, protocol
    )
    matched["gamma"] = matched_gamma

    timeline, current, stitch = lm.test_timeline(
        val["inputs"], val["trues"], test["inputs"]
    )
    for state in (stable, recent, matched):
        state["timeline"], state["current"] = timeline, current

    totals = {
        DASH_METHOD: {"sq": 0.0, "ab": 0.0, "corr": 0.0, "count": 0},
        MATCHED_BASELINE: {"sq": 0.0, "ab": 0.0, "corr": 0.0, "count": 0},
    }
    raw_sq = raw_ab = 0.0
    raw_count = trigger_count = 0
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
        scores = shock_score(recent_correction - stable_correction, bins)
        dash_correction = recent_correction.copy()
        for b in range(4):
            horizons = np.where(bins == b)[0]
            if quantiles[b] == "never":
                trigger = np.zeros(len(selected), dtype=bool)
            else:
                trigger = scores[:, b] > thresholds[float(quantiles[b])][b]
            dash_correction[:, horizons] += (
                alphas[b]
                * trigger[:, None, None]
                * (stable_correction[:, horizons] - recent_correction[:, horizons])
            )
            trigger_count += int(np.sum(trigger))
        baseline_correction = pp.correction_from_state(
            matched, test["inputs"], test["preds"], selected, bins
        )
        raw_error = test["preds"][selected] - test["trues"][selected]
        raw_sq += float(np.sum(np.square(raw_error, dtype=np.float64)))
        raw_ab += float(np.sum(np.abs(raw_error), dtype=np.float64))
        raw_count += int(raw_error.size)
        for method, correction in (
            (DASH_METHOD, dash_correction),
            (MATCHED_BASELINE, baseline_correction),
        ):
            error = raw_error + correction
            totals[method]["sq"] += float(np.sum(np.square(error, dtype=np.float64)))
            totals[method]["ab"] += float(np.sum(np.abs(error), dtype=np.float64))
            totals[method]["corr"] += float(np.sum(np.abs(correction), dtype=np.float64))
            totals[method]["count"] += int(error.size)

    raw_mse, raw_mae = raw_sq / raw_count, raw_ab / raw_count
    common_meta = {
        **stitch,
        "purged_protocol": protocol_audit,
        "history_steps": 1008,
        "validation_unique_target_budget": int(
            protocol_audit.get(
                "unique_target_timestamp_budget",
                config.pred_len + len(val["preds"]) - 1,
            )
        ),
        "v3_decisions_for_dash": ["complete_fallback_route"],
        "stable_v3_gate_removed": True,
        "uses_test_targets_for_selection": False,
    }
    method_meta = {
        DASH_METHOD: {
            **common_meta,
            "threshold_quantiles": quantiles,
            "alpha": alphas.tolist(),
            "route": route_audit,
            "triggered_origin_bins": int(trigger_count),
            "stable_gamma": stable["gamma"].tolist(),
            "recent_gamma": recent["gamma"].tolist(),
        },
        MATCHED_BASELINE: {
            **common_meta,
            "description": "nine unaggregated period-cycle lag features plus nonnegative ridge",
            "period_cycle_features": matched["specs"],
            "gamma": matched_gamma.tolist(),
            "v3_route": matched_audit,
        },
    }
    rows = []
    for method in (DASH_METHOD, MATCHED_BASELINE):
        total = totals[method]
        mse = total["sq"] / total["count"]
        mae = total["ab"] / total["count"]
        rows.append(
            {
                "dataset": core.display_dataset(config.dataset),
                "backbone": config.model,
                "model": config.model,
                "pred_len": config.pred_len,
                "seed": config.seed,
                "config_key": core.config_key(config),
                "method": method,
                "raw_mse": raw_mse,
                "raw_mae": raw_mae,
                "method_mse": mse,
                "method_mae": mae,
                "delta_mse_pct": 100.0 * (mse - raw_mse) / raw_mse,
                "gain_pct": 100.0 * (raw_mse - mse) / raw_mse,
                "mae_gain_pct": 100.0 * (raw_mae - mae) / raw_mae,
                "active": True,
                "fallback": False,
                "selection_protocol": "target_timestamp_purged_V1_V2_V3",
                "uses_external_outputs": False,
                "uses_test_for_selection": False,
                "test_correction_reads_test_truth": False,
                "correction_norm_mean": total["corr"] / total["count"],
                "meta_json": json.dumps(method_meta[method], sort_keys=True),
            }
        )
    return rows


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
        val, test = nb.collect_val_test_arrays_in_memory(config)
        measured = base.metrics_chunked(test["preds"], test["trues"])
        expected = payload["raw"]
        if not (
            np.isclose(measured[0], expected["raw_mse"], atol=1e-6, rtol=1e-6)
            and np.isclose(measured[1], expected["raw_mae"], atol=1e-6, rtol=1e-6)
        ):
            raise ValueError(f"Raw mismatch {measured} vs {expected}")
        rows = core.align_raw(evaluate(config, val, test), expected, measured)
        core.atomic_csv(row_path, pd.DataFrame(rows))
        core.atomic_json(
            status_path,
            {"status": "completed", "elapsed_sec": round(time.time() - started, 3)},
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
        default=Path("outputs/dash_purged_v2_full504_20260725"),
    )
    parser.add_argument("--models", nargs="*", default=lm.cf.BACKBONES)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[
            "ETTh1",
            "ETTh2",
            "ETTm1",
            "ETTm2",
            "Electricity",
            "Traffic",
            "Weather",
        ],
    )
    parser.add_argument("--pred-lens", nargs="*", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--seeds", nargs="*", type=int, default=[2, 42, 2021])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu-ids", nargs="*", default=[str(index) for index in range(8)])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw = core.load_raw(args.raw_metrics_csv)
    datasets = [core.canonical_dataset(value) for value in args.datasets]
    configs = [
        nb.BackboneConfig(model=model, dataset=dataset, pred_len=horizon, seed=seed)
        for model in args.models
        for dataset in datasets
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
        }
        for config in configs
    ]
    args.out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    failures = []
    core.atomic_json(
        args.out_root / "runner_status.json",
        {"status": "running", "expected": len(configs)},
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
                    "seen": seen,
                    "expected": len(configs),
                    "failures": len(failures),
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
    for method in (DASH_METHOD, MATCHED_BASELINE):
        selected = detail[detail["method"].eq(method)].copy()
        if not selected.empty:
            core.atomic_csv(
                args.out_root / f"{method}_detail.csv",
                selected,
            )
            core.atomic_csv(
                args.out_root / f"{method}_summary.csv",
                core.summarize(selected),
            )
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
            "expected": len(configs),
            "rows": len(detail),
            "failures": len(failures),
            "elapsed_sec": round(time.time() - started, 2),
        },
    )


if __name__ == "__main__":
    main()
