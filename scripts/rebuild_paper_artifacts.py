#!/usr/bin/env python3
"""Rebuild Tables 1--4 and Figures 2--4 from frozen result CSVs."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper_results"
METHOD_ORDER = ["Raw", "TAFAS", "PETSA", "COSA-F", "COSA-P", "DASH"]
DATASET_ORDER = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Electricity",
    "Traffic",
    "Weather",
]


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise AssertionError(f"{name} is missing columns: {missing}")


def verify_checksums() -> None:
    manifest = RESULTS / "SHA256SUMS"
    if not manifest.exists():
        raise FileNotFoundError("paper_results/SHA256SUMS is missing")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        path = RESULTS / relative
        if not path.is_file():
            raise FileNotFoundError(f"checksum input is missing: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise AssertionError(f"checksum mismatch for {relative}")


def load_and_validate() -> dict[str, pd.DataFrame]:
    verify_checksums()
    names = [
        "main_config_504",
        "main_aggregate_168",
        "raw_metrics_504",
        "external_config_2016",
        "overall_summary",
        "ablation_summary",
        "efficiency_summary",
        "efficiency_config_216",
        "hierarchical_comparison",
        "hierarchical_paired_inference",
        "validation_budget_summary",
        "validation_budget_config_2016",
        "period_sensitivity_summary",
        "period_sensitivity_config_378",
        "k_sensitivity_summary",
        "k_sensitivity_config_162",
        "beta_sensitivity_summary",
        "beta_sensitivity_config_270",
    ]
    frames = {name: pd.read_csv(RESULTS / f"{name}.csv") for name in names}

    detail = frames["main_config_504"]
    require_columns(
        detail,
        {
            "dataset",
            "backbone",
            "pred_len",
            "seed",
            "config_key",
            "raw_mse",
            "raw_mae",
            "method_mse",
            "method_mae",
            "gain_pct",
            "mae_gain_pct",
            "cycles",
            "recent_decay",
            "period_set",
            "validation_fraction",
            "uses_test_for_selection",
            "test_correction_reads_test_truth",
        },
        "main_config_504.csv",
    )
    assert len(detail) == 504
    assert detail["config_key"].nunique() == 504
    assert len(detail[["dataset", "backbone", "pred_len", "seed"]].drop_duplicates()) == 504
    assert set(detail["dataset"]) == set(DATASET_ORDER)
    assert detail["backbone"].nunique() == 6
    assert set(detail["pred_len"]) == {96, 192, 336, 720}
    assert set(detail["seed"]) == {2, 42, 2021}
    assert set(detail["cycles"]) == {3}
    assert np.allclose(detail["recent_decay"], 0.90)
    assert np.allclose(detail["validation_fraction"], 0.75)
    assert set(detail["period_set"]) == {"P96+168+336"}
    for column in ("uses_test_for_selection", "test_correction_reads_test_truth"):
        values = detail[column].astype(str).str.strip().str.lower()
        assert set(values) <= {"false", "0"}, f"causal flag failed: {column}"
    assert np.isclose(detail["raw_mse"].mean(), 0.38388868986223246, atol=5e-7)
    assert np.isclose(detail["method_mse"].mean(), 0.3430844513552633, atol=5e-7)
    assert np.isclose(detail["gain_pct"].mean(), 10.050, atol=0.002)
    assert (detail["method_mse"] < detail["raw_mse"]).all()

    raw = frames["raw_metrics_504"]
    require_columns(
        raw,
        {"model", "dataset", "pred_len", "seed", "raw_mse", "raw_mae", "status"},
        "raw_metrics_504.csv",
    )
    assert len(raw) == 504
    assert len(raw[["model", "dataset", "pred_len", "seed"]].drop_duplicates()) == 504
    raw_join = detail.merge(
        raw,
        left_on=["backbone", "dataset", "pred_len", "seed"],
        right_on=["model", "dataset", "pred_len", "seed"],
        suffixes=("_detail", "_cache"),
        validate="one_to_one",
    )
    assert np.allclose(raw_join["raw_mse_detail"], raw_join["raw_mse_cache"], atol=1e-9)
    assert np.allclose(raw_join["raw_mae_detail"], raw_join["raw_mae_cache"], atol=1e-9)

    external_detail = frames["external_config_2016"]
    require_columns(
        external_detail,
        {
            "config_key",
            "backbone",
            "dataset",
            "pred_len",
            "seed",
            "method",
            "status",
            "raw_mse",
            "raw_mae",
            "method_mse",
            "method_mae",
            "gain_pct",
        },
        "external_config_2016.csv",
    )
    assert len(external_detail) == 2016
    assert set(external_detail["method"]) == {"TAFAS", "PETSA", "COSA-F", "COSA-P"}
    assert external_detail.groupby("method").size().eq(504).all()
    assert external_detail.groupby("method")["config_key"].nunique().eq(504).all()
    assert set(external_detail["status"].astype(str).str.lower()) == {"ok"}

    aggregate = frames["main_aggregate_168"]
    require_columns(
        aggregate,
        {"dataset", "backbone", "pred_len", "method", "mse_mean", "mae_mean"},
        "main_aggregate_168.csv",
    )
    assert len(aggregate) == 1008
    assert set(aggregate["method"]) == set(METHOD_ORDER)
    assert aggregate.groupby("method").size().eq(168).all()
    means = aggregate.groupby("method")[["mse_mean", "mae_mean"]].mean()
    assert np.isclose(means.loc["Raw", "mse_mean"], 0.38388868986223246, atol=5e-7)
    assert np.isclose(means.loc["DASH", "mse_mean"], 0.3430844513552633, atol=5e-7)

    overall = frames["overall_summary"].set_index("method")
    assert set(overall.index) == set(METHOD_ORDER)
    assert np.allclose(
        overall.loc[METHOD_ORDER, "mse"], means.loc[METHOD_ORDER, "mse_mean"], atol=5e-7
    )
    assert np.allclose(
        overall.loc[METHOD_ORDER, "mae"], means.loc[METHOD_ORDER, "mae_mean"], atol=5e-7
    )
    external = overall.loc[["TAFAS", "PETSA", "COSA-F", "COSA-P"], "mse"]
    advantage = 100.0 * (external - overall.loc["DASH", "mse"]) / external
    assert advantage.min() >= 5.82 and advantage.max() <= 6.78
    for method, group in external_detail.groupby("method"):
        expected = overall.loc[method]
        wins = 100.0 * float((group["method_mse"] < group["raw_mse"]).mean())
        ties = 100.0 * float(np.isclose(group["method_mse"], group["raw_mse"], atol=1e-12).mean())
        assert np.isclose(group["method_mse"].mean(), expected["mse"], atol=5e-7)
        assert np.isclose(group["method_mae"].mean(), expected["mae"], atol=5e-7)
        assert np.isclose(group["gain_pct"].mean(), expected["mse_gain_pct"], atol=5e-4)
        assert np.isclose(wins, expected["win_pct"], atol=0.01)
        assert np.isclose(ties, expected["tie_pct"], atol=0.01)

    combined = pd.concat(
        [
            detail.assign(method="Raw", mse_mean=detail["raw_mse"], mae_mean=detail["raw_mae"]),
            detail.assign(method="DASH", mse_mean=detail["method_mse"], mae_mean=detail["method_mae"]),
            external_detail.assign(
                mse_mean=external_detail["method_mse"],
                mae_mean=external_detail["method_mae"],
            ),
        ],
        ignore_index=True,
    )
    rebuilt_aggregate = (
        combined.groupby(["dataset", "backbone", "pred_len", "method"], as_index=False)[
            ["mse_mean", "mae_mean"]
        ]
        .mean()
    )
    aggregate_check = aggregate.merge(
        rebuilt_aggregate,
        on=["dataset", "backbone", "pred_len", "method"],
        suffixes=("_frozen", "_rebuilt"),
        validate="one_to_one",
    )
    assert len(aggregate_check) == 1008
    assert np.allclose(aggregate_check["mse_mean_frozen"], aggregate_check["mse_mean_rebuilt"], atol=5e-7)
    assert np.allclose(aggregate_check["mae_mean_frozen"], aggregate_check["mae_mean_rebuilt"], atol=5e-7)

    ablation = frames["ablation_summary"].set_index("variant")
    assert np.isclose(ablation.loc["Raw", "mse"], 0.3839, atol=5e-5)
    assert np.isclose(ablation.loc["Full DASH", "mse"], 0.3431, atol=5e-5)
    assert ablation.loc["Full DASH", "mse"] == ablation["mse"].min()

    efficiency = frames["efficiency_summary"].set_index("method")
    assert np.isclose(efficiency.loc["DASH", "test_origins_per_sec"], 347.8286, atol=5e-4)
    efficiency_detail = frames["efficiency_config_216"]
    assert len(efficiency_detail) == 216
    measured = efficiency_detail.loc[~efficiency_detail["is_warmup"].astype(bool)].copy()
    assert measured.groupby("method").size().eq(54).all()
    method_names = {
        "Raw": "Raw",
        "dash_purged_v2": "DASH",
        "periodic_ridge_1008_purged": "Periodic ridge",
    }
    for source_name, paper_name in method_names.items():
        group = measured.loc[measured["method"] == source_name]
        assert np.isclose(group["end_to_end_sec"].median(), efficiency.loc[paper_name, "end_to_end_sec"], atol=5e-4)
        assert np.isclose(group["end_to_end_test_origins_per_sec"].median(), efficiency.loc[paper_name, "test_origins_per_sec"], atol=5e-4)

    budget = frames["validation_budget_summary"]
    assert len(budget) == 4
    assert np.isclose(
        budget.loc[budget["mse_gain_pct"].idxmax(), "validation_fraction"], 0.75
    )
    budget_detail = frames["validation_budget_config_2016"]
    assert len(budget_detail) == 2016
    assert budget_detail.groupby("validation_fraction").size().eq(504).all()
    assert not budget_detail["uses_test_for_selection"].astype(bool).any()
    assert not budget_detail["test_correction_reads_test_truth"].astype(bool).any()
    budget_means = budget_detail.groupby("validation_fraction")[["gain_pct", "mae_gain_pct"]].mean()
    for row in budget.itertuples(index=False):
        actual = budget_means.loc[row.validation_fraction]
        assert np.isclose(actual["gain_pct"], row.mse_gain_pct, atol=5e-4)
        assert np.isclose(actual["mae_gain_pct"], row.mae_gain_pct, atol=5e-4)

    periods = frames["period_sensitivity_summary"]
    assert str(periods.loc[periods["mean_mse"].idxmin(), "period_set"]) == "P96+168+336"
    period_detail = frames["period_sensitivity_config_378"]
    assert len(period_detail) == 378
    assert period_detail.groupby("period_set").size().eq(54).all()
    period_means = period_detail.groupby("period_set")[["method_mse", "gain_pct"]].mean()
    for row in periods.itertuples(index=False):
        assert np.isclose(period_means.loc[row.period_set, "method_mse"], row.mean_mse, atol=5e-6)
        assert np.isclose(period_means.loc[row.period_set, "gain_pct"], row.mse_gain_pct, atol=5e-4)

    assert int(frames["k_sensitivity_summary"].loc[lambda x: x["is_default"], "cycles"].iloc[0]) == 3
    k_detail = frames["k_sensitivity_config_162"]
    assert len(k_detail) == 162 and k_detail.groupby("cycles").size().eq(54).all()
    k_means = k_detail.groupby("cycles")[["gain_pct", "mae_gain_pct"]].mean()
    for row in frames["k_sensitivity_summary"].itertuples(index=False):
        assert np.isclose(k_means.loc[row.cycles, "gain_pct"], row.mse_gain_pct, atol=5e-4)
        assert np.isclose(k_means.loc[row.cycles, "mae_gain_pct"], row.mae_gain_pct, atol=5e-4)

    assert np.isclose(
        frames["beta_sensitivity_summary"].loc[lambda x: x["is_default"], "recent_decay"].iloc[0],
        0.90,
    )
    beta_detail = frames["beta_sensitivity_config_270"]
    assert len(beta_detail) == 270 and beta_detail.groupby("recent_decay").size().eq(54).all()
    beta_means = beta_detail.groupby("recent_decay")[["gain_pct", "mae_gain_pct"]].mean()
    for row in frames["beta_sensitivity_summary"].itertuples(index=False):
        assert np.isclose(beta_means.loc[row.recent_decay, "gain_pct"], row.mse_gain_pct, atol=5e-4)
        assert np.isclose(beta_means.loc[row.recent_decay, "mae_gain_pct"], row.mae_gain_pct, atol=5e-4)

    hierarchy = frames["hierarchical_paired_inference"]
    hierarchy = hierarchy.loc[
        (hierarchy["scope"] == "full_504")
        & (hierarchy["metric"] == "MSE")
        & hierarchy["baseline"].isin(["TAFAS", "PETSA", "COSA-F", "COSA-P"])
    ].set_index("baseline")
    frozen_hierarchy = frames["hierarchical_comparison"].set_index("method")
    assert set(hierarchy.index) == set(frozen_hierarchy.index)
    assert np.allclose(hierarchy.loc[frozen_hierarchy.index, "difference_pp"], frozen_hierarchy["delta_gain_pp"], atol=5e-4)
    assert np.allclose(hierarchy.loc[frozen_hierarchy.index, "nested_ci2p5_pp"], frozen_hierarchy["ci_low"], atol=5e-4)
    assert np.allclose(hierarchy.loc[frozen_hierarchy.index, "nested_ci97p5_pp"], frozen_hierarchy["ci_high"], atol=5e-4)
    return frames


def write_tables(frames: dict[str, pd.DataFrame], output: Path) -> None:
    overall = frames["overall_summary"].copy()
    overall["method"] = pd.Categorical(overall["method"], METHOD_ORDER, ordered=True)
    overall.sort_values("method").to_csv(output / "table1_overall.csv", index=False)

    aggregate = frames["main_aggregate_168"]
    by_dataset = (
        aggregate.groupby(["dataset", "method"], as_index=False)["mse_mean"]
        .mean()
        .pivot(index="dataset", columns="method", values="mse_mean")
        .loc[DATASET_ORDER, METHOD_ORDER]
        .reset_index()
    )
    by_dataset.to_csv(output / "table2_by_dataset.csv", index=False)
    frames["ablation_summary"].to_csv(output / "table3_ablation.csv", index=False)

    efficiency = frames["efficiency_summary"].copy()
    raw_e2e = float(efficiency.loc[efficiency["method"] == "Raw", "end_to_end_sec"].iloc[0])
    efficiency.insert(2, "e2e_difference_vs_raw_sec", efficiency["end_to_end_sec"] - raw_e2e)
    efficiency.to_csv(output / "table4_efficiency.csv", index=False)


def setup_plots() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf")
    fig.savefig(output / f"{stem}.png", dpi=220)
    plt.close(fig)


def figure2_hierarchical(frame: pd.DataFrame, output: Path) -> None:
    y = np.arange(len(frame))
    delta = frame["delta_gain_pp"].to_numpy()
    low = frame["ci_low"].to_numpy()
    high = frame["ci_high"].to_numpy()
    wins = frame["win_pct"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2), gridspec_kw={"width_ratios": [1.2, 1]})
    axes[0].errorbar(
        delta,
        y,
        xerr=np.vstack([delta - low, high - delta]),
        fmt="o",
        color="#2E54A1",
        ecolor="#2E54A1",
        capsize=3,
    )
    axes[0].axvline(0, color="#999999", linestyle="--", linewidth=0.8)
    axes[0].set_yticks(y, frame["method"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("MSE-gain difference (pp)")
    axes[0].set_title("Nested 95% confidence interval", loc="left")
    axes[0].grid(axis="x", color="#E5E5E5")
    axes[1].barh(y, wins, color="#00A651", label="Win")
    axes[1].barh(y, 100.0 - wins, left=wins, color="#D0D0D0", label="Loss")
    axes[1].invert_yaxis()
    axes[1].set_yticks([])
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Paired outcomes (%)")
    axes[1].set_title("DASH versus external method", loc="left")
    axes[1].legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, output, "figure2_hierarchical")


def figure3_budget(frame: pd.DataFrame, output: Path) -> None:
    x = 100.0 * frame["validation_fraction"].to_numpy()
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.plot(x, frame["mse_gain_pct"], "o-", color="#2E54A1", label="MSE gain")
    ax.plot(x, frame["mae_gain_pct"], "s-", color="#E5A900", label="MAE gain")
    ax.axvline(75, color="#999999", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, [f"{value:g}%" for value in x])
    ax.set_xlabel("Validation budget")
    ax.set_ylabel("Gain (%)")
    ax.grid(axis="y", color="#E5E5E5")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output, "figure3_validation_budget")


def figure4_sensitivity(frames: dict[str, pd.DataFrame], output: Path) -> None:
    periods = frames["period_sensitivity_summary"]
    k = frames["k_sensitivity_summary"]
    beta = frames["beta_sensitivity_summary"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6))
    y = np.arange(len(periods))
    axes[0, 0].scatter(periods["mean_mse"], y, color="#2E54A1")
    axes[0, 0].set_yticks(y, periods["period_set"])
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("Mean MSE (lower is better)")
    axes[0, 0].set_title("(a) Period-set error", loc="left")
    axes[0, 1].barh(y, periods["mse_gain_pct"], color="#7DA7D9")
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_yticks(y, periods["period_set"])
    axes[0, 1].set_xlabel("MSE gain (%)")
    axes[0, 1].set_title("(b) Period-set gain", loc="left")
    axes[1, 0].plot(k["cycles"], k["mse_gain_pct"], "o-", label="MSE gain")
    axes[1, 0].plot(k["cycles"], k["mae_gain_pct"], "s-", label="MAE gain")
    axes[1, 0].set_xticks(k["cycles"])
    axes[1, 0].set_xlabel("Memory depth K")
    axes[1, 0].set_ylabel("Gain (%)")
    axes[1, 0].set_title("(c) Memory depth", loc="left")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].plot(beta["recent_decay"], beta["mse_gain_pct"], "o-", label="MSE gain")
    axes[1, 1].plot(beta["recent_decay"], beta["mae_gain_pct"], "s-", label="MAE gain")
    axes[1, 1].set_xticks(beta["recent_decay"])
    axes[1, 1].set_xlabel("Recent retention beta")
    axes[1, 1].set_ylabel("Gain (%)")
    axes[1, 1].set_title("(d) Recent-memory retention", loc="left")
    axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.grid(axis="y", color="#E5E5E5")
    fig.tight_layout()
    save_figure(fig, output, "figure4_sensitivity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reproduced")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup_plots()
    frames = load_and_validate()
    write_tables(frames, args.output)
    figure2_hierarchical(frames["hierarchical_comparison"], args.output)
    figure3_budget(frames["validation_budget_summary"], args.output)
    figure4_sensitivity(frames, args.output)
    print("Paper numeric assertions: PASS")
    print(f"Rebuilt Tables 1-4 and Figures 2-4 in: {args.output}")


if __name__ == "__main__":
    main()
