"""Hierarchical paired inference for leakage-audited DASH.

The primary inferential unit is the dataset.  Backbone, horizon, and seed
results within a dataset are dependent repeated measurements and are not
treated as 504 independent samples.  A nested
dataset -> backbone -> horizon -> seed bootstrap is reported as a secondary
uncertainty analysis.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Traffic", "Weather"]
BACKBONES = ["DLinear", "PatchTST", "iTransformer", "MICN", "OLS", "FreTS"]
HORIZONS = [96, 192, 336, 720]
SEEDS = [2, 42, 2021]
EXTERNAL = ["TAFAS", "PETSA", "COSA-F", "COSA-P"]
OURS = "dash_purged_v2"
MATCHED = "periodic_ridge_1008_purged"
KEYS = ["dataset", "backbone", "pred_len", "seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dash-detail", type=Path, required=True)
    parser.add_argument("--external-detail", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "backbone" not in out.columns and "model" in out.columns:
        out["backbone"] = out["model"]
    if "config_key" not in out.columns:
        out["config_key"] = [
            f"{model}|{dataset}|{int(horizon)}|{int(seed)}"
            for dataset, model, horizon, seed in out[KEYS].itertuples(index=False, name=None)
        ]
    for column in ("pred_len", "seed"):
        out[column] = pd.to_numeric(out[column], errors="raise").astype(int)
    for column in ("raw_mse", "raw_mae", "method_mse", "method_mae"):
        out[column] = pd.to_numeric(out[column], errors="raise")
    out["mse_gain_pct"] = 100.0 * (out["raw_mse"] - out["method_mse"]) / out["raw_mse"]
    out["mae_gain_pct"] = 100.0 * (out["raw_mae"] - out["method_mae"]) / out["raw_mae"]
    return out


def validate_scope(frame: pd.DataFrame, methods: list[str]) -> None:
    subset = frame[frame["method"].isin(methods)]
    expected = len(DATASETS) * len(BACKBONES) * len(HORIZONS) * len(SEEDS)
    counts = subset.groupby("method", observed=True).size().to_dict()
    if any(int(counts.get(method, 0)) != expected for method in methods):
        raise ValueError(f"Expected {expected} rows per method, observed {counts}")
    if subset.duplicated(KEYS + ["method"]).any():
        raise ValueError("Duplicate configuration/method rows")
    observed = {
        (row.dataset, row.backbone, int(row.pred_len), int(row.seed))
        for row in subset[KEYS].itertuples(index=False)
    }
    expected_keys = set(itertools.product(DATASETS, BACKBONES, HORIZONS, SEEDS))
    if observed != expected_keys:
        raise ValueError(
            f"Scope mismatch: missing={len(expected_keys - observed)}, "
            f"extra={len(observed - expected_keys)}"
        )


def exact_sign_flip(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    observed = float(np.mean(values))
    signs = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=len(values))),
        dtype=np.float64,
    )
    null = np.mean(signs * values[None, :], axis=1)
    one_sided = (np.count_nonzero(null >= observed) + 1.0) / (len(null) + 1.0)
    two_sided = (
        np.count_nonzero(np.abs(null) >= abs(observed)) + 1.0
    ) / (len(null) + 1.0)
    return float(one_sided), float(two_sided)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def nested_bootstrap(
    paired: pd.DataFrame,
    value_column: str,
    iterations: int,
    seed: int,
) -> np.ndarray:
    """Balanced four-level resampling preserving the experimental hierarchy."""
    datasets = [value for value in DATASETS if value in set(paired["dataset"])]
    backbones = [value for value in BACKBONES if value in set(paired["backbone"])]
    horizons = [value for value in HORIZONS if value in set(paired["pred_len"])]
    seeds = [value for value in SEEDS if value in set(paired["seed"])]
    index = pd.MultiIndex.from_product(
        [datasets, backbones, horizons, seeds], names=KEYS
    )
    values = paired.set_index(KEYS)[value_column].reindex(index)
    if values.isna().any():
        raise ValueError("Nested bootstrap requires a balanced hierarchy")
    cube = values.to_numpy(np.float64).reshape(
        len(datasets), len(backbones), len(horizons), len(seeds)
    )
    rng = np.random.default_rng(seed)
    result = np.empty(iterations, dtype=np.float64)
    # Chunked vectorization keeps the exact nested resampling structure while
    # avoiding millions of Python-level inner loops.
    for begin in range(0, iterations, 1000):
        count = min(1000, iterations - begin)
        d_index = rng.integers(
            0, len(datasets), size=(count, len(datasets))
        )
        b_index = rng.integers(
            0,
            len(backbones),
            size=(count, len(datasets), len(backbones)),
        )
        h_index = rng.integers(
            0,
            len(horizons),
            size=(count, len(datasets), len(backbones), len(horizons)),
        )
        s_index = rng.integers(
            0,
            len(seeds),
            size=(
                count,
                len(datasets),
                len(backbones),
                len(horizons),
                len(seeds),
            ),
        )
        sampled = cube[
            d_index[:, :, None, None, None],
            b_index[:, :, :, None, None],
            h_index[:, :, :, :, None],
            s_index,
        ]
        result[begin : begin + count] = sampled.mean(axis=(1, 2, 3, 4))
    return result


def dataset_backbone_bootstrap(
    paired: pd.DataFrame,
    value_column: str,
    iterations: int,
    seed: int,
) -> np.ndarray:
    cluster_means = (
        paired.groupby(["dataset", "backbone"], observed=True)[value_column]
        .mean()
        .to_numpy(np.float64)
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(cluster_means), size=(iterations, len(cluster_means))
    )
    return cluster_means[indices].mean(axis=1)


def comparison_rows(
    ours: pd.DataFrame,
    baseline: pd.DataFrame,
    baseline_name: str,
    scope: str,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    left = ours.set_index(KEYS).sort_index()
    right = baseline.set_index(KEYS).sort_index().loc[left.index]
    metadata = left.reset_index()[KEYS]
    rows: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(("mse_gain_pct", "mae_gain_pct")):
        paired = metadata.copy()
        paired["difference_pp"] = (
            left[metric].to_numpy(np.float64)
            - right[metric].to_numpy(np.float64)
        )
        dataset_effects = (
            paired.groupby("dataset", observed=True)["difference_pp"]
            .mean()
            .reindex(DATASETS)
            .dropna()
            .to_numpy(np.float64)
        )
        p_one, p_two = exact_sign_flip(dataset_effects)
        nested = nested_bootstrap(
            paired,
            "difference_pp",
            iterations,
            seed + metric_index * 10,
        )
        supplementary = dataset_backbone_bootstrap(
            paired,
            "difference_pp",
            iterations,
            seed + metric_index * 10 + 1,
        )
        differences = paired["difference_pp"].to_numpy(np.float64)
        rows.append(
            {
                "scope": scope,
                "metric": metric.replace("_gain_pct", "").upper(),
                "ours": OURS,
                "baseline": baseline_name,
                "configs": len(paired),
                "datasets": len(dataset_effects),
                "difference_pp": float(np.mean(differences)),
                "dataset_equal_weight_difference_pp": float(
                    np.mean(dataset_effects)
                ),
                "nested_ci2p5_pp": float(np.quantile(nested, 0.025)),
                "nested_ci97p5_pp": float(np.quantile(nested, 0.975)),
                "dataset_backbone_ci2p5_pp": float(
                    np.quantile(supplementary, 0.025)
                ),
                "dataset_backbone_ci97p5_pp": float(
                    np.quantile(supplementary, 0.975)
                ),
                "dataset_exact_signflip_p_one_sided": p_one,
                "dataset_exact_signflip_p_two_sided": p_two,
                "wins": int(np.count_nonzero(differences > 1e-12)),
                "ties": int(np.count_nonzero(np.abs(differences) <= 1e-12)),
                "losses": int(np.count_nonzero(differences < -1e-12)),
            }
        )
    return rows


def grouped_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for grouping in (["dataset"], ["backbone"], ["pred_len"]):
        for keys, group in frame.groupby(grouping, sort=False, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row: dict[str, Any] = {
                "grouping": " x ".join(grouping),
                "configs": len(group),
                "mean_mse": float(group["method_mse"].mean()),
                "mean_mae": float(group["method_mae"].mean()),
                "mean_mse_gain_pct": float(group["mse_gain_pct"].mean()),
                "mean_mae_gain_pct": float(group["mae_gain_pct"].mean()),
                "mse_harmful": int((group["mse_gain_pct"] < 0.0).sum()),
                "mae_harmful": int((group["mae_gain_pct"] < 0.0).sum()),
            }
            row.update(dict(zip(grouping, keys)))
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    dash = normalize(pd.read_csv(args.dash_detail))
    dash = dash[dash["method"].isin([OURS, MATCHED])].copy()
    external = normalize(pd.read_csv(args.external_detail))
    external = external[external["method"].isin(EXTERNAL)].copy()
    validate_scope(dash, [OURS, MATCHED])
    validate_scope(external, EXTERNAL)

    ours_full = dash[dash["method"].eq(OURS)].copy()
    comparisons: list[dict[str, Any]] = []
    run_index = 0
    candidates = [
        (MATCHED, dash[dash["method"].eq(MATCHED)].copy()),
        *[
            (method, external[external["method"].eq(method)].copy())
            for method in EXTERNAL
        ],
    ]
    for method, baseline in candidates:
        comparisons.extend(
            comparison_rows(
                ours_full,
                baseline,
                method,
                "full_504",
                args.iterations,
                args.seed + run_index * 100,
            )
        )
        run_index += 1

    # Electricity and Traffic use official code with fallback/default
    # hyperparameters because the released scripts do not provide those
    # dataset-specific settings.  The strict scope reports only configurations
    # covered by released dataset-specific scripts.
    strict_datasets = [value for value in DATASETS if value not in {"Electricity", "Traffic"}]
    ours_strict = ours_full[ours_full["dataset"].isin(strict_datasets)].copy()
    for method in EXTERNAL:
        baseline = external[
            external["method"].eq(method)
            & external["dataset"].isin(strict_datasets)
        ].copy()
        comparisons.extend(
            comparison_rows(
                ours_strict,
                baseline,
                method,
                "strict_official_script_360",
                args.iterations,
                args.seed + run_index * 100,
            )
        )
        run_index += 1

    results = pd.DataFrame(comparisons)
    for (scope, metric), indices in results.groupby(
        ["scope", "metric"], sort=False
    ).groups.items():
        selected = list(indices)
        results.loc[selected, "holm_p_one_sided"] = holm_adjust(
            results.loc[selected, "dataset_exact_signflip_p_one_sided"].tolist()
        )
        results.loc[selected, "holm_p_two_sided"] = holm_adjust(
            results.loc[selected, "dataset_exact_signflip_p_two_sided"].tolist()
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / "hierarchical_paired_inference.csv", index=False)
    grouped_summary(ours_full).to_csv(
        args.out_dir / "dash_grouped_summary.csv", index=False
    )
    protocol = (
        external[external["backbone"].eq("OLS")]
        .groupby(["method", "dataset", "hp_source"], dropna=False, observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    protocol["protocol_class"] = np.where(
        protocol["hp_source"].astype(str).str.contains("fallback"),
        "official_code_default_hp_extension",
        np.where(
            protocol["dataset"].isin(["Electricity", "Traffic"]),
            "official_code_non_script_dataset_extension",
            "released_script_protocol",
        ),
    )
    protocol.to_csv(args.out_dir / "ols_external_protocol_scope.csv", index=False)
    audit = {
        "ours": OURS,
        "matched_baseline": MATCHED,
        "external_methods": EXTERNAL,
        "full_configs": int(len(ours_full)),
        "strict_configs": int(len(ours_strict)),
        "iterations": int(args.iterations),
        "primary_test": "exact paired sign flip over seven dataset-level effects",
        "uncertainty": "nested dataset->backbone->horizon->seed bootstrap",
        "multiplicity": "Holm correction within scope and metric",
        "strict_exclusion": ["Electricity", "Traffic"],
    }
    (args.out_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
