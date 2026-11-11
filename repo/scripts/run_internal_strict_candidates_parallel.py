from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = ROOT / "scripts"
import sys

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_internal_offline_candidates as base


STRICT_METHODS = {
    "raw",
    "naive_seasonal96_hgamma_blend",
    "causal_seasonal_rolling_bias",
}


def write_status(out_dir: Path, status: str, **extra: object) -> None:
    payload = {"status": status, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    tmp = out_dir / "status.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out_dir / "status.json")


def split_validation_three(
    preds: np.ndarray,
    trues: np.ndarray,
    inputs: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray | None]:
    """Chronological fit/tune/route split for validation-only selection.

    fit: residual initialization / state fitting.
    tune: candidate hyperparameter selection.
    route: frozen-candidate evidence and block LCB.
    """
    n = int(preds.shape[0])
    if n < 6:
        raise ValueError(f"validation too short for strict three-way split: n={n}")
    c1 = max(1, n // 3)
    c2 = max(c1 + 1, (2 * n) // 3)
    c2 = min(c2, n - 1)
    return (
        preds[:c1],
        trues[:c1],
        None if inputs is None else inputs[:c1],
        preds[c1:c2],
        trues[c1:c2],
        None if inputs is None else inputs[c1:c2],
        preds[c2:],
        trues[c2:],
        None if inputs is None else inputs[c2:],
    )


def route_gain(raw_mse: float, method_mse: float) -> float:
    return (raw_mse - method_mse) / raw_mse * 100.0 if raw_mse > 0 else 0.0


def select_hgamma_on_tune(
    fit_preds: np.ndarray,
    fit_inputs: np.ndarray,
    tune_preds: np.ndarray,
    tune_trues: np.ndarray,
    tune_inputs: np.ndarray,
    method_id: str,
) -> dict[str, Any]:
    fit_alt = base.naive_forecast_from_inputs(fit_inputs, fit_preds.shape[1], method_id)
    tune_alt = base.naive_forecast_from_inputs(tune_inputs, tune_preds.shape[1], method_id)
    corr_tune = (tune_alt - tune_preds).astype(np.float32, copy=False)
    fit_corr_abs = np.abs(fit_alt - fit_preds)
    clip_options = [None, 0.90, 0.95]
    gamma_options = np.asarray([0.0, 0.10, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
    raw_tune_mse, _ = base.metrics_chunked(tune_preds, tune_trues)
    horizon = int(tune_preds.shape[1])
    best = {
        "tune_raw_mse": raw_tune_mse,
        "tune_mse": raw_tune_mse,
        "tune_gain_pct": 0.0,
        "gamma_json": json.dumps([0.0 for _ in range(horizon)]),
        "gamma_mean": 0.0,
        "gamma_active_horizon_frac": 0.0,
        "clip_q": None,
        "clip_value": np.nan,
        "fallback": True,
        "fallback_reason": "tune_selected_raw_all_horizons",
    }
    for clip_q in clip_options:
        clip_value = np.nan
        corr_for_tune = corr_tune
        if clip_q is not None:
            clip_value = float(np.quantile(fit_corr_abs, clip_q))
            corr_for_tune = np.clip(corr_tune, -clip_value, clip_value)
        gamma_h = np.zeros(horizon, dtype=np.float32)
        selected_h_mse = np.empty(horizon, dtype=np.float64)
        for h in range(horizon):
            raw_h_mse = float(np.mean(np.square(tune_preds[:, h, :] - tune_trues[:, h, :], dtype=np.float64)))
            best_h_mse = raw_h_mse
            best_h_gamma = 0.0
            corr_h = corr_for_tune[:, h, :]
            for gamma in gamma_options[1:]:
                err = tune_preds[:, h, :] + float(gamma) * corr_h - tune_trues[:, h, :]
                val_h_mse = float(np.mean(np.square(err, dtype=np.float64)))
                if val_h_mse < best_h_mse:
                    best_h_mse = val_h_mse
                    best_h_gamma = float(gamma)
            gamma_h[h] = best_h_gamma
            selected_h_mse[h] = best_h_mse
        tune_mse = float(np.mean(selected_h_mse))
        if tune_mse < float(best["tune_mse"]):
            active_frac = float(np.mean(gamma_h > 0))
            best.update(
                {
                    "tune_mse": tune_mse,
                    "tune_gain_pct": route_gain(raw_tune_mse, tune_mse),
                    "gamma_json": json.dumps([round(float(x), 4) for x in gamma_h]),
                    "gamma_mean": float(np.mean(gamma_h)),
                    "gamma_active_horizon_frac": active_frac,
                    "clip_q": clip_q,
                    "clip_value": clip_value,
                    "fallback": active_frac == 0.0,
                    "fallback_reason": "none" if active_frac > 0.0 else "tune_selected_raw_all_horizons",
                }
            )
    return best


def eval_hgamma(
    preds: np.ndarray,
    trues: np.ndarray,
    inputs: np.ndarray,
    method_id: str,
    gamma_json: str,
    clip_value: float | None,
    collect_blocks: bool = False,
) -> tuple[float, float, dict[str, Any]]:
    alt = base.naive_forecast_from_inputs(inputs, preds.shape[1], method_id)
    corr = (alt - preds).astype(np.float32, copy=False)
    gamma_h = np.asarray(json.loads(gamma_json), dtype=np.float32)
    mse_val, mae_val = base.metrics_hgamma_chunked(
        preds,
        trues,
        corr,
        gamma_h=gamma_h,
        clip_value=clip_value,
    )
    meta: dict[str, Any] = {}
    if collect_blocks:
        meta.update(
            base.validation_block_stats_hgamma(
                preds,
                trues,
                corr,
                gamma_h=gamma_h,
                clip_value=clip_value,
            )
        )
    norm_mean, norm_max = base.correction_norm_chunked(corr, clip_value=clip_value)
    meta["correction_norm_mean"] = float(norm_mean)
    meta["correction_norm_max"] = float(norm_max)
    return mse_val, mae_val, meta


def select_causal_on_tune(
    fit_preds: np.ndarray,
    fit_trues: np.ndarray,
    tune_preds: np.ndarray,
    tune_trues: np.ndarray,
    seasonal_grid: str,
    rolling_grid: str,
) -> dict[str, Any]:
    fit_resid_abs = np.abs(fit_trues - fit_preds)
    raw_tune_mse, _ = base.metrics_chunked(tune_preds, tune_trues)
    period_options = [1, 2, 4, 6, 7, 12, 24, 48, 96, 168] if seasonal_grid == "full" else [1, 4, 24, 96]
    if rolling_grid == "full":
        gamma_options = [0.0, 0.25, 0.50, 0.75, 1.0]
        alpha_options = [0.01, 0.05, 0.10, 0.20, 0.50, 1.0]
        clip_options = [None, 0.90, 0.95]
    else:
        gamma_options = [0.0, 0.25, 0.50]
        alpha_options = [0.05, 0.20, 0.50, 1.0]
        clip_options = [None, 0.95]
    best = {
        "tune_raw_mse": raw_tune_mse,
        "tune_mse": raw_tune_mse,
        "tune_gain_pct": 0.0,
        "gamma": 0.0,
        "alpha": 0.0,
        "period": 1,
        "clip_q": None,
        "clip_value": np.nan,
        "fallback": True,
        "fallback_reason": "tune_selected_raw",
    }
    for period in period_options:
        init_state = base.initial_seasonal_state(fit_preds, fit_trues, int(period))
        for gamma in gamma_options:
            if gamma == 0.0:
                continue
            for alpha in alpha_options:
                for clip_q in clip_options:
                    clip_value = np.nan if clip_q is None else float(np.quantile(fit_resid_abs, clip_q))
                    tune_mse, _mae, _meta = base.causal_seasonal_rolling_bias_eval(
                        tune_preds,
                        tune_trues,
                        init_state=init_state,
                        period=int(period),
                        target_offset=fit_preds.shape[0],
                        gamma=float(gamma),
                        alpha=float(alpha),
                        clip_value=clip_value if clip_q is not None else None,
                    )
                    if tune_mse < float(best["tune_mse"]):
                        best.update(
                            {
                                "tune_mse": tune_mse,
                                "tune_gain_pct": route_gain(raw_tune_mse, tune_mse),
                                "gamma": float(gamma),
                                "alpha": float(alpha),
                                "period": int(period),
                                "clip_q": clip_q,
                                "clip_value": clip_value,
                                "fallback": False,
                                "fallback_reason": "none",
                            }
                        )
    return best


def run_pair(payload: dict[str, Any]) -> dict[str, Any]:
    pair = payload["pair"]
    methods = payload["methods"]
    key = f"{pair['backbone']}|{pair['dataset']}|{pair['pred_len']}|{pair['seed']}"
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        val_preds, val_trues, val_inputs = base.load_npz(Path(pair["val_path"]), load_inputs=True)
        test_preds, test_trues, test_inputs = base.load_npz(Path(pair["test_path"]), load_inputs=True)
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
        ) = split_validation_three(val_preds, val_trues, val_inputs)
        raw_test_mse, raw_test_mae = base.metrics_chunked(test_preds, test_trues)
        raw_route_mse, raw_route_mae = base.metrics_chunked(route_preds, route_trues)
        split_meta = {
            "fit_origins": int(fit_preds.shape[0]),
            "tune_origins": int(tune_preds.shape[0]),
            "route_origins": int(route_preds.shape[0]),
            "selection_protocol": "validation_fit_tune_route",
            "uses_external_outputs": False,
            "uses_test_for_selection": False,
        }
        if "raw" in methods:
            block = base.block_gain_stats_from_origin_mse(
                np.mean(np.square(route_preds - route_trues, dtype=np.float64), axis=(1, 2)),
                np.mean(np.square(route_preds - route_trues, dtype=np.float64), axis=(1, 2)),
            )
            rows.append(
                {
                    **pair,
                    "config_key": key,
                    "method": "raw",
                    "raw_mse": raw_test_mse,
                    "raw_mae": raw_test_mae,
                    "method_mse": raw_test_mse,
                    "method_mae": raw_test_mae,
                    "delta_mse_pct": 0.0,
                    "gain_pct": 0.0,
                    "route_raw_mse": raw_route_mse,
                    "route_mse": raw_route_mse,
                    "route_gain_pct": 0.0,
                    "route_mae": raw_route_mae,
                    "tune_gain_pct": 0.0,
                    "fallback": True,
                    "fallback_reason": "raw_anchor",
                    **block,
                    **split_meta,
                }
            )
        if "naive_seasonal96_hgamma_blend" in methods:
            method = "naive_seasonal96_hgamma_blend"
            tuned = select_hgamma_on_tune(fit_preds, fit_inputs, tune_preds, tune_trues, tune_inputs, method)
            clip_value = tuned["clip_value"] if tuned["clip_q"] is not None else None
            route_mse, route_mae, route_meta = eval_hgamma(
                route_preds,
                route_trues,
                route_inputs,
                method,
                tuned["gamma_json"],
                clip_value,
                collect_blocks=True,
            )
            test_mse, test_mae, test_meta = eval_hgamma(
                test_preds,
                test_trues,
                test_inputs,
                method,
                tuned["gamma_json"],
                clip_value,
                collect_blocks=False,
            )
            delta = (test_mse - raw_test_mse) / raw_test_mse * 100.0 if raw_test_mse > 0 else np.nan
            rows.append(
                {
                    **pair,
                    "config_key": key,
                    "method": method,
                    "raw_mse": raw_test_mse,
                    "raw_mae": raw_test_mae,
                    "method_mse": test_mse,
                    "method_mae": test_mae,
                    "delta_mse_pct": delta,
                    "gain_pct": -delta,
                    "route_raw_mse": raw_route_mse,
                    "route_mse": route_mse,
                    "route_mae": route_mae,
                    "route_gain_pct": route_gain(raw_route_mse, route_mse),
                    **{f"test_{k}": v for k, v in test_meta.items()},
                    **route_meta,
                    **tuned,
                    **split_meta,
                }
            )
        if "causal_seasonal_rolling_bias" in methods:
            method = "causal_seasonal_rolling_bias"
            tuned = select_causal_on_tune(
                fit_preds,
                fit_trues,
                tune_preds,
                tune_trues,
                payload["seasonal_grid"],
                payload["rolling_grid"],
            )
            clip_value = tuned["clip_value"] if tuned["clip_q"] is not None else None
            init_state = base.initial_seasonal_state(fit_preds, fit_trues, int(tuned["period"]))
            _tune_mse, _tune_mae, _tune_meta, tune_state = base.causal_seasonal_rolling_bias_eval(
                tune_preds,
                tune_trues,
                init_state=init_state,
                period=int(tuned["period"]),
                target_offset=fit_preds.shape[0],
                gamma=float(tuned["gamma"]),
                alpha=float(tuned["alpha"]),
                clip_value=clip_value,
                return_state=True,
            )
            route_mse, route_mae, route_meta, route_state = base.causal_seasonal_rolling_bias_eval(
                route_preds,
                route_trues,
                init_state=tune_state,
                period=int(tuned["period"]),
                target_offset=fit_preds.shape[0] + tune_preds.shape[0],
                gamma=float(tuned["gamma"]),
                alpha=float(tuned["alpha"]),
                clip_value=clip_value,
                collect_block_stats=True,
                return_state=True,
            )
            test_mse, test_mae, test_meta = base.causal_seasonal_rolling_bias_eval(
                test_preds,
                test_trues,
                init_state=route_state,
                period=int(tuned["period"]),
                target_offset=fit_preds.shape[0] + tune_preds.shape[0] + route_preds.shape[0],
                gamma=float(tuned["gamma"]),
                alpha=float(tuned["alpha"]),
                clip_value=clip_value,
            )
            delta = (test_mse - raw_test_mse) / raw_test_mse * 100.0 if raw_test_mse > 0 else np.nan
            rows.append(
                {
                    **pair,
                    "config_key": key,
                    "method": method,
                    "raw_mse": raw_test_mse,
                    "raw_mae": raw_test_mae,
                    "method_mse": test_mse,
                    "method_mae": test_mae,
                    "delta_mse_pct": delta,
                    "gain_pct": -delta,
                    "route_raw_mse": raw_route_mse,
                    "route_mse": route_mse,
                    "route_mae": route_mae,
                    "route_gain_pct": route_gain(raw_route_mse, route_mse),
                    **{f"test_{k}": v for k, v in test_meta.items()},
                    **route_meta,
                    **tuned,
                    **split_meta,
                }
            )
    except Exception as exc:
        failures.append(
            {
                **pair,
                "config_key": key,
                "status": "failed",
                "failure_reason": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return {"config_key": key, "rows": rows, "failures": failures}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict validation-fit/tune/route candidate runner.")
    p.add_argument("--search-root", action="append", default=[str(ROOT / "outputs")])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--datasets", nargs="*", default=base.DATASETS)
    p.add_argument("--models", nargs="*", default=base.BACKBONES)
    p.add_argument("--pred-lens", nargs="*", type=int, default=base.PRED_LENS)
    p.add_argument("--seeds", nargs="*", type=int, default=base.SEEDS)
    p.add_argument("--methods", default="raw,naive_seasonal96_hgamma_blend,causal_seasonal_rolling_bias")
    p.add_argument("--workers", type=int, default=max(1, min(32, mp.cpu_count() // 2)))
    p.add_argument("--seasonal-grid", choices=["small", "full"], default="small")
    p.add_argument("--rolling-grid", choices=["small", "full"], default="small")
    return p.parse_args()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby("method", dropna=False):
        delta = pd.to_numeric(g["delta_mse_pct"], errors="coerce")
        rows.append(
            {
                "method": method,
                "configs": int(len(g)),
                "mean_gain_pct": float(-delta.mean()),
                "harmful_count": int((delta > 0).sum()),
                "practical_harmful_0p1_count": int((delta > 0.1).sum()),
                "severe_harmful_1p0_count": int((delta > 1.0).sum()),
                "worst_degradation_pct": float(delta.clip(lower=0).max()),
                "route_mean_gain_pct": float(pd.to_numeric(g["route_gain_pct"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_gain_pct", "harmful_count"], ascending=[False, True])


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_status(out_dir, "running")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = sorted(set(methods) - STRICT_METHODS)
    if unknown:
        raise ValueError(f"Unknown strict methods: {unknown}")
    roots = [Path(p) for p in args.search_root]
    pairs, source_inventory = base.discover_exports(roots)
    source_inventory.to_csv(out_dir / "export_source_inventory.csv", index=False)
    datasets = {base.DATASET_ALIASES.get(str(x).lower(), str(x)) for x in args.datasets}
    models = {base.BACKBONE_ALIASES.get(str(x).lower(), str(x)) for x in args.models}
    pred_lens = {int(x) for x in args.pred_lens}
    seeds = {int(x) for x in args.seeds}
    pairs = [
        p
        for p in pairs
        if p.dataset in datasets and p.backbone in models and p.pred_len in pred_lens and p.seed in seeds
    ]
    payloads = [
        {
            "pair": {
                "dataset": p.dataset,
                "backbone": p.backbone,
                "model": p.backbone,
                "pred_len": int(p.pred_len),
                "seed": int(p.seed),
                "val_path": str(p.val_path),
                "test_path": str(p.test_path),
            },
            "methods": methods,
            "seasonal_grid": args.seasonal_grid,
            "rolling_grid": args.rolling_grid,
        }
        for p in pairs
    ]
    all_rows: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    started = time.time()
    with mp.Pool(processes=max(1, int(args.workers))) as pool:
        for idx, result in enumerate(pool.imap_unordered(run_pair, payloads), start=1):
            all_rows.extend(result["rows"])
            all_failures.extend(result["failures"])
            pd.DataFrame(all_rows).to_csv(out_dir / "config_level_detail.partial.csv", index=False)
            pd.DataFrame(all_failures).to_csv(out_dir / "failure_log.partial.csv", index=False)
            write_status(
                out_dir,
                "running",
                completed_pairs=idx,
                total_pairs=len(payloads),
                detail_rows=len(all_rows),
                failures=len(all_failures),
                current_config=result["config_key"],
                elapsed_sec=round(time.time() - started, 3),
            )
            print(f"[{idx}/{len(payloads)}] {result['config_key']} rows={len(result['rows'])}", flush=True)
    detail = pd.DataFrame(all_rows)
    failures = pd.DataFrame(all_failures)
    detail.to_csv(out_dir / "config_level_detail.csv", index=False)
    failures.to_csv(out_dir / "failure_log.csv", index=False)
    if not detail.empty:
        summarize(detail).to_csv(out_dir / "summary.csv", index=False)
    manifest = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "protocol": "strict_validation_fit_tune_route",
        "candidate_pool": methods,
        "raw_explicit_candidate": "raw" in methods,
        "feature_selected_role": "not_generated_here; use only with independent route evidence or as empirical reference",
        "uses_external_outputs": False,
        "uses_test_for_selection": False,
        "seed_independence": "each config uses only its own validation split for candidate parameters/evidence",
        "search_roots": [str(p) for p in roots],
        "workers": int(args.workers),
        "elapsed_sec": round(time.time() - started, 3),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_status(
        out_dir,
        "completed",
        selected_export_configs=len(pairs),
        detail_rows=len(detail),
        failed_configs=len(failures),
        elapsed_sec=round(time.time() - started, 3),
    )
    print(f"Wrote {out_dir}")
    if not detail.empty:
        print(summarize(detail).to_string(index=False))


if __name__ == "__main__":
    main()
