from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import openpyxl
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TSLIB_ROOT = PROJECT_ROOT / "external" / "Time-Series-Library"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EXPORT_ROOT = OUTPUT_ROOT / "exports"
WORKBOOK_PATH = PROJECT_ROOT / "ts_posthoc_experiment_matrix.xlsx"
RESULTS_CSV = OUTPUT_ROOT / "w1_results.csv"
SUMMARY_JSON = OUTPUT_ROOT / "w1_summary.json"


@dataclass(frozen=True)
class RunConfig:
    model: str
    dataset: str
    pred_len: int
    data_loader_name: str | None = None
    root_path_override: str | None = None
    data_path_override: str | None = None
    freq_override: str | None = None
    seq_len: int = 96
    label_len: int = 48
    batch_size: int = 32
    e_layers: int = 2
    d_layers: int = 1
    d_model: int = 512
    d_ff: int = 2048
    n_heads: int = 8
    factor: int = 3
    train_epochs: int = 10
    patience: int = 3
    learning_rate: float = 1e-4
    features: str = "M"
    target: str = "OT"
    enc_in: int = 7
    dec_in: int = 7
    c_out: int = 7
    des: str = "w1"
    num_workers: int = 0
    itr: int = 1
    gpu: int = 0
    use_amp: bool = False
    embed: str = "timeF"
    dropout: float = 0.1
    moving_avg: int = 25
    use_gpu: bool = True

    @property
    def data_path(self) -> str:
        if self.data_path_override is not None:
            return self.data_path_override
        return f"{self.dataset}.csv"

    @property
    def root_path(self) -> str:
        if self.root_path_override is not None:
            return self.root_path_override
        return "./dataset/ETT-small/"

    @property
    def freq(self) -> str:
        if self.freq_override is not None:
            return self.freq_override
        return "h" if self.dataset.startswith("ETTh") else "t"

    @property
    def data_name(self) -> str:
        if self.data_loader_name is not None:
            return self.data_loader_name
        return self.dataset

    @property
    def model_id(self) -> str:
        return f"W1_{self.dataset}_{self.seq_len}_{self.pred_len}"

    @property
    def export_dir(self) -> Path:
        return EXPORT_ROOT / self.model / self.dataset / f"pl{self.pred_len}"


def w1_default_configs(train_epochs: int | None = None) -> list[RunConfig]:
    configs = [
        RunConfig(model="DLinear", dataset="ETTh1", pred_len=96),
        RunConfig(model="DLinear", dataset="ETTh1", pred_len=336),
        RunConfig(model="DLinear", dataset="ETTm2", pred_len=96),
        RunConfig(model="DLinear", dataset="ETTm2", pred_len=336),
        RunConfig(model="PatchTST", dataset="ETTh1", pred_len=96, e_layers=1, n_heads=2),
        RunConfig(model="PatchTST", dataset="ETTh1", pred_len=336, e_layers=1, n_heads=8),
        RunConfig(model="PatchTST", dataset="ETTm2", pred_len=96, e_layers=3, n_heads=16),
        RunConfig(model="PatchTST", dataset="ETTm2", pred_len=336, e_layers=1, n_heads=4),
    ]
    if train_epochs is None:
        return configs
    return [replace_config(cfg, train_epochs=train_epochs) for cfg in configs]


def replace_config(config: RunConfig, **kwargs: Any) -> RunConfig:
    data = {field: getattr(config, field) for field in config.__dataclass_fields__}
    data.update(kwargs)
    return RunConfig(**data)


def metric(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, true={true.shape}")
    total_abs = 0.0
    total_sq = 0.0
    total_count = 0
    chunk = max(1, min(pred.shape[0], 64))
    for left in range(0, pred.shape[0], chunk):
        right = min(pred.shape[0], left + chunk)
        pred_chunk = pred[left:right].astype(np.float32, copy=False)
        true_chunk = true[left:right].astype(np.float32, copy=False)
        err = true_chunk - pred_chunk
        total_abs += float(np.abs(err).sum(dtype=np.float64))
        total_sq += float(np.square(err, dtype=np.float32).sum(dtype=np.float64))
        total_count += int(err.size)
    return total_abs / total_count, total_sq / total_count


def horizon_mse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, true={true.shape}")
    sq_sum = np.zeros(pred.shape[1], dtype=np.float64)
    count = 0
    chunk = max(1, min(pred.shape[0], 64))
    for left in range(0, pred.shape[0], chunk):
        right = min(pred.shape[0], left + chunk)
        pred_chunk = pred[left:right].astype(np.float32, copy=False)
        true_chunk = true[left:right].astype(np.float32, copy=False)
        err = true_chunk - pred_chunk
        sq_sum += np.square(err, dtype=np.float32).sum(axis=(0, 2), dtype=np.float64)
        count += int(err.shape[0] * err.shape[2])
    return (sq_sum / count).astype(np.float64)


def temporal_split_index(num_samples: int, ratio: float = 0.7, min_tail: int = 64) -> int:
    split_idx = max(128, int(num_samples * ratio))
    split_idx = min(split_idx, num_samples - min_tail)
    return split_idx


def param_count(state_dict: dict[str, torch.Tensor]) -> int:
    total = 0
    for tensor in state_dict.values():
        total += int(tensor.numel())
    return total


def arg_namespace(config: RunConfig) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        is_training=1,
        model_id=config.model_id,
        model=config.model,
        data=config.data_name,
        root_path=config.root_path,
        data_path=config.data_path,
        features=config.features,
        target=config.target,
        freq=config.freq,
        checkpoints="./checkpoints/",
        seq_len=config.seq_len,
        label_len=config.label_len,
        pred_len=config.pred_len,
        seasonal_patterns="Monthly",
        inverse=False,
        mask_rate=0.25,
        anomaly_ratio=0.25,
        expand=2,
        d_conv=4,
        tv_dt=0,
        tv_B=0,
        tv_C=0,
        use_D=0,
        top_k=5,
        num_kernels=6,
        enc_in=config.enc_in,
        dec_in=config.dec_in,
        c_out=config.c_out,
        d_model=config.d_model,
        n_heads=config.n_heads,
        e_layers=config.e_layers,
        d_layers=config.d_layers,
        d_ff=config.d_ff,
        moving_avg=config.moving_avg,
        factor=config.factor,
        distil=True,
        dropout=config.dropout,
        embed=config.embed,
        activation="gelu",
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
        seg_len=96,
        num_workers=config.num_workers,
        itr=config.itr,
        train_epochs=config.train_epochs,
        batch_size=config.batch_size,
        patience=config.patience,
        learning_rate=config.learning_rate,
        des=config.des,
        loss="MSE",
        lradj="type1",
        use_amp=config.use_amp,
        use_gpu=config.use_gpu,
        gpu=config.gpu,
        gpu_type="cuda",
        use_multi_gpu=False,
        devices=str(config.gpu),
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        use_dtw=False,
        augmentation_ratio=0,
        seed=2,
        jitter=False,
        scaling=False,
        permutation=False,
        randompermutation=False,
        magwarp=False,
        timewarp=False,
        windowslice=False,
        windowwarp=False,
        rotation=False,
        spawner=False,
        dtwwarp=False,
        shapedtwwarp=False,
        wdba=False,
        discdtw=False,
        discsdtw=False,
        extra_tag="",
        patch_len=16,
        node_dim=10,
        gcn_depth=2,
        gcn_dropout=0.3,
        propalpha=0.3,
        conv_channel=32,
        skip_channel=32,
        individual=False,
        alpha=0.1,
        top_p=0.5,
        pos=1,
    )


def tslib_setting(args: SimpleNamespace, trial: int = 0) -> str:
    return f"{args.task_name}_{args.model_id}_{args.model}_{args.des}_{trial}"


def raw_model_params(config: RunConfig) -> int:
    args = arg_namespace(config)
    with pushd(TSLIB_ROOT):
        ensure_tslib_on_path()
        model = build_model(args, torch.device("cpu"))
        return sum(int(p.numel()) for p in model.parameters())


def ensure_tslib_on_path() -> None:
    root = str(TSLIB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    purge_conflicting_tslib_modules()


def purge_conflicting_tslib_modules() -> None:
    prefixes = ("data_provider", "datasets", "exp", "layers", "models", "utils")
    root = str(TSLIB_ROOT.resolve())
    for name in list(sys.modules):
        if not any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            continue
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            sys.modules.pop(name, None)
            continue
        module_path = str(Path(module_file).resolve())
        if not module_path.startswith(root):
            sys.modules.pop(name, None)


@contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def build_model(args: SimpleNamespace, device: torch.device) -> torch.nn.Module:
    ensure_tslib_on_path()
    module = importlib.import_module(f"models.{args.model}")
    if hasattr(module, "Model"):
        model_cls = module.Model
    else:
        model_cls = getattr(module, args.model)
    model = model_cls(args).float().to(device)
    return model


def build_loader(args: SimpleNamespace, split: str) -> tuple[Any, DataLoader]:
    ensure_tslib_on_path()
    from data_provider.data_factory import data_dict

    data_cls = data_dict[args.data]
    timeenc = 0 if args.embed != "timeF" else 1
    dataset = data_cls(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=split,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        seasonal_patterns=args.seasonal_patterns,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return dataset, loader


def collect_split_arrays(config: RunConfig, split: str, force: bool = False) -> dict[str, np.ndarray]:
    args = arg_namespace(config)
    setting = tslib_setting(args)
    out_path = config.export_dir / f"{split}.npz"
    meta_path = config.export_dir / "metadata.json"
    if out_path.exists() and not force:
        with np.load(out_path) as saved:
            return {key: saved[key] for key in saved.files}

    config.export_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{config.gpu}" if config.use_gpu and torch.cuda.is_available() else "cpu")
    checkpoint_path = TSLIB_ROOT / "checkpoints" / setting / "checkpoint.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    with pushd(TSLIB_ROOT):
        model = build_model(args, device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        dataset, loader = build_loader(args, split)

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
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "setting": setting,
                "checkpoint": str(checkpoint_path),
                "raw_model_params": raw_model_params(config),
                "checkpoint_params": param_count(state_dict),
                "pred_shape": list(arrays["preds"].shape),
            },
            fh,
            indent=2,
        )
    return arrays


def fit_horizon_affine(preds: np.ndarray, trues: np.ndarray) -> dict[str, np.ndarray]:
    _, horizon, _ = preds.shape
    slope = np.ones(horizon, dtype=np.float32)
    bias = np.zeros(horizon, dtype=np.float32)
    for h_idx in range(horizon):
        x = preds[:, h_idx, :].reshape(-1).astype(np.float64)
        y = trues[:, h_idx, :].reshape(-1).astype(np.float64)
        x_mean = float(x.mean())
        y_mean = float(y.mean())
        denom = float(np.sum((x - x_mean) ** 2))
        if denom < 1e-12:
            slope[h_idx] = 1.0
            bias[h_idx] = y_mean - x_mean
            continue
        cov = float(np.sum((x - x_mean) * (y - y_mean)))
        slope[h_idx] = cov / denom
        bias[h_idx] = y_mean - slope[h_idx] * x_mean
    return {"slope": slope, "bias": bias}


def apply_horizon_affine(preds: np.ndarray, fitted: dict[str, np.ndarray]) -> np.ndarray:
    slope = fitted["slope"][None, :, None]
    bias = fitted["bias"][None, :, None]
    return preds * slope + bias


def fit_bias_calibrator(preds: np.ndarray, trues: np.ndarray, per_channel: bool) -> dict[str, Any]:
    residual = trues - preds
    if per_channel:
        bias = residual.mean(axis=0).astype(np.float32)
    else:
        bias = residual.mean(axis=(0, 2)).astype(np.float32)
    return {
        "kind": "bias",
        "per_channel": per_channel,
        "bias": bias,
    }


def apply_bias_calibrator(preds: np.ndarray, fitted: dict[str, Any]) -> np.ndarray:
    if fitted["per_channel"]:
        return preds + fitted["bias"][None, :, :]
    return preds + fitted["bias"][None, :, None]


def select_e01_calibrator(preds: np.ndarray, trues: np.ndarray) -> tuple[dict[str, Any], dict[str, float]]:
    split_idx = temporal_split_index(preds.shape[0])
    fit_preds, tune_preds = preds[:split_idx], preds[split_idx:]
    fit_trues, tune_trues = trues[:split_idx], trues[split_idx:]

    candidates: dict[str, dict[str, Any]] = {
        "identity": {"kind": "identity"},
        "global_bias": fit_bias_calibrator(fit_preds, fit_trues, per_channel=False),
        "per_channel_bias": fit_bias_calibrator(fit_preds, fit_trues, per_channel=True),
    }
    scores: dict[str, float] = {}
    for name, fitted in candidates.items():
        corrected = tune_preds if fitted["kind"] == "identity" else apply_bias_calibrator(tune_preds, fitted)
        _, mse = metric(corrected, tune_trues)
        scores[name] = mse

    best_name = min(scores, key=scores.get)
    if best_name == "identity":
        return {"kind": "identity"}, scores

    best_per_channel = best_name == "per_channel_bias"
    return fit_bias_calibrator(preds, trues, per_channel=best_per_channel), scores


def apply_e01_calibrator(preds: np.ndarray, fitted: dict[str, Any]) -> np.ndarray:
    if fitted["kind"] == "identity":
        return preds.copy()
    return apply_bias_calibrator(preds, fitted)


def build_residual_features(preds: np.ndarray, inputs: np.ndarray, mode: str = "full") -> np.ndarray:
    last = inputs[:, -1, :]
    horizon = preds.shape[1]
    last_h = np.repeat(last[:, None, :], horizon, axis=1)
    pred_delta = preds - last[:, None, :]
    if mode == "weak":
        features = np.stack([preds, pred_delta, last_h], axis=-1)
        return features.astype(np.float32)

    first = inputs[:, 0, :]
    mean = inputs.mean(axis=1)
    std = inputs.std(axis=1)
    mean_h = np.repeat(mean[:, None, :], horizon, axis=1)
    std_h = np.repeat(std[:, None, :], horizon, axis=1)
    slope_h = np.repeat((last - first)[:, None, :], horizon, axis=1)
    features = np.stack([preds, pred_delta, last_h, mean_h, std_h, slope_h], axis=-1)
    return features.astype(np.float32)


def ridge_fit(
    train_x: np.ndarray,
    train_y: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    feat_mean = train_x.mean(axis=0)
    feat_std = train_x.std(axis=0)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
    x_norm = (train_x - feat_mean) / feat_std
    x_aug = np.concatenate([np.ones((x_norm.shape[0], 1), dtype=np.float32), x_norm], axis=1)
    reg = np.eye(x_aug.shape[1], dtype=np.float32)
    reg[0, 0] = 0.0
    lhs = x_aug.T @ x_aug + alpha * reg
    rhs = x_aug.T @ train_y
    weight = np.linalg.solve(lhs, rhs)
    return {
        "weight": weight.astype(np.float32),
        "feat_mean": feat_mean.astype(np.float32),
        "feat_std": feat_std.astype(np.float32),
    }


def ridge_predict(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    x_norm = (features - model["feat_mean"]) / model["feat_std"]
    x_aug = np.concatenate([np.ones((x_norm.shape[0], 1), dtype=np.float32), x_norm], axis=1)
    return x_aug @ model["weight"]


def fit_linear_residual_adapter(
    preds: np.ndarray,
    inputs: np.ndarray,
    trues: np.ndarray,
    alpha: float,
    feature_mode: str = "full",
) -> dict[str, np.ndarray]:
    features = build_residual_features(preds, inputs, mode=feature_mode)
    residual = trues - preds
    samples, horizon, channels, feat_dim = features.shape
    weights = np.zeros((horizon, channels, feat_dim + 1), dtype=np.float32)
    feat_mean = np.zeros((horizon, channels, feat_dim), dtype=np.float32)
    feat_std = np.zeros((horizon, channels, feat_dim), dtype=np.float32)
    for h_idx in range(horizon):
        for c_idx in range(channels):
            model = ridge_fit(features[:, h_idx, c_idx, :], residual[:, h_idx, c_idx], alpha)
            weights[h_idx, c_idx] = model["weight"]
            feat_mean[h_idx, c_idx] = model["feat_mean"]
            feat_std[h_idx, c_idx] = model["feat_std"]
    return {
        "kind": "linear",
        "feature_mode": feature_mode,
        "weights": weights,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
    }


def apply_linear_residual_adapter(
    preds: np.ndarray,
    inputs: np.ndarray,
    fitted: dict[str, np.ndarray],
) -> np.ndarray:
    if fitted["kind"] == "identity":
        return preds.copy()

    features = build_residual_features(preds, inputs, mode=fitted.get("feature_mode", "full"))
    corrected = preds.copy()
    samples, horizon, channels, _ = features.shape
    for h_idx in range(horizon):
        for c_idx in range(channels):
            feature_slice = features[:, h_idx, c_idx, :]
            model = {
                "weight": fitted["weights"][h_idx, c_idx],
                "feat_mean": fitted["feat_mean"][h_idx, c_idx],
                "feat_std": fitted["feat_std"][h_idx, c_idx],
            }
            corrected[:, h_idx, c_idx] += ridge_predict(model, feature_slice)
    return corrected


def streaming_ema_correction(
    preds: np.ndarray,
    trues: np.ndarray,
    alpha: float,
    init_state: np.ndarray | None = None,
    init_seen: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples, horizon, channels = preds.shape
    state = np.zeros((horizon, channels), dtype=np.float32) if init_state is None else init_state.copy()
    seen = np.zeros((horizon, channels), dtype=bool) if init_seen is None else init_seen.copy()
    corrected = np.empty_like(preds)
    for idx in range(samples):
        for h_idx in range(horizon):
            src_idx = idx - (h_idx + 1)
            if src_idx < 0:
                continue
            residual = trues[src_idx, h_idx, :] - preds[src_idx, h_idx, :]
            new_mask = ~seen[h_idx]
            state[h_idx, new_mask] = residual[new_mask]
            state[h_idx, ~new_mask] = alpha * residual[~new_mask] + (1.0 - alpha) * state[h_idx, ~new_mask]
            seen[h_idx] = True
        corrected[idx] = preds[idx] + state
    return corrected, state, seen


def select_ema_alpha(preds: np.ndarray, trues: np.ndarray, alpha_grid: list[float]) -> tuple[float, dict[str, float]]:
    warm_idx = max(preds.shape[1] + 1, int(preds.shape[0] * 0.3))
    warm_idx = min(warm_idx, preds.shape[0] - 1)
    scores: dict[str, float] = {}
    best_alpha = alpha_grid[0]
    best_score = float("inf")
    for alpha in alpha_grid:
        corrected, _, _ = streaming_ema_correction(preds, trues, alpha)
        _, mse = metric(corrected[warm_idx:], trues[warm_idx:])
        scores[f"{alpha:g}"] = mse
        if mse < best_score:
            best_score = mse
            best_alpha = alpha
    return best_alpha, scores


def select_ridge_alpha(
    preds: np.ndarray,
    inputs: np.ndarray,
    trues: np.ndarray,
    alpha_grid: list[float],
) -> tuple[float, dict[str, float]]:
    split_idx = max(128, int(preds.shape[0] * 0.7))
    split_idx = min(split_idx, preds.shape[0] - 64)
    train_preds, tune_preds = preds[:split_idx], preds[split_idx:]
    train_inputs, tune_inputs = inputs[:split_idx], inputs[split_idx:]
    train_trues, tune_trues = trues[:split_idx], trues[split_idx:]
    scores: dict[str, float] = {}
    best_alpha = alpha_grid[0]
    best_score = float("inf")
    for alpha in alpha_grid:
        fitted = fit_linear_residual_adapter(train_preds, train_inputs, train_trues, alpha)
        corrected = apply_linear_residual_adapter(tune_preds, tune_inputs, fitted)
        _, mse = metric(corrected, tune_trues)
        scores[f"{alpha:g}"] = mse
        if mse < best_score:
            best_score = mse
            best_alpha = alpha
    return best_alpha, scores


def select_e03_model(
    preds: np.ndarray,
    inputs: np.ndarray,
    trues: np.ndarray,
    alpha_grid: list[float],
    min_tune_gain_pct: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    split_idx = temporal_split_index(preds.shape[0])
    fit_preds, tune_preds = preds[:split_idx], preds[split_idx:]
    fit_inputs, tune_inputs = inputs[:split_idx], inputs[split_idx:]
    fit_trues, tune_trues = trues[:split_idx], trues[split_idx:]
    tail_alpha = max(alpha_grid)

    _, raw_tune_mse = metric(tune_preds, tune_trues)
    diagnostics: dict[str, Any] = {
        "raw_tune_mse": raw_tune_mse,
        "min_tune_gain_pct": min_tune_gain_pct,
        "selection_rule": "tail_alpha_gate",
        "tail_alpha": tail_alpha,
        "candidates": {},
    }
    for alpha in alpha_grid:
        fitted = fit_linear_residual_adapter(
            fit_preds,
            fit_inputs,
            fit_trues,
            alpha,
            feature_mode="weak",
        )
        corrected = apply_linear_residual_adapter(tune_preds, tune_inputs, fitted)
        _, tune_mse = metric(corrected, tune_trues)
        tune_gain_pct = 100.0 * (raw_tune_mse - tune_mse) / raw_tune_mse
        diagnostics["candidates"][f"{alpha:g}"] = {
            "tune_mse": tune_mse,
            "tune_gain_pct": tune_gain_pct,
        }

    tail_gain_pct = diagnostics["candidates"][f"{tail_alpha:g}"]["tune_gain_pct"]
    diagnostics["tail_gain_pct"] = tail_gain_pct
    diagnostics["tail_alpha_qualified"] = tail_gain_pct >= min_tune_gain_pct
    if tail_gain_pct < min_tune_gain_pct:
        diagnostics["selected"] = "identity"
        diagnostics["rejected_reason"] = "tail alpha did not clear tune gain gate"
        return {"kind": "identity"}, diagnostics

    diagnostics["selected"] = f"weak_linear_{tail_alpha:g}"
    diagnostics["selected_alpha"] = tail_alpha
    selected_model = fit_linear_residual_adapter(
        preds,
        inputs,
        trues,
        tail_alpha,
        feature_mode="weak",
    )
    return selected_model, diagnostics


def run_training(config: RunConfig, force: bool = False) -> str:
    args = arg_namespace(config)
    setting = tslib_setting(args)
    checkpoint_path = TSLIB_ROOT / "checkpoints" / setting / "checkpoint.pth"
    if checkpoint_path.exists() and not force:
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
        config.data_name,
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
    ]
    if config.use_amp:
        command.append("--use_amp")
    subprocess.run(command, cwd=TSLIB_ROOT, check=True)
    return setting


def evaluate_methods(
    config: RunConfig,
    val_arrays: dict[str, np.ndarray],
    test_arrays: dict[str, np.ndarray],
    alpha_grid_ema: list[float],
    alpha_grid_ridge: list[float],
    e03_min_tune_gain_pct: float,
) -> dict[str, Any]:
    raw_mae, raw_mse = metric(test_arrays["preds"], test_arrays["trues"])
    raw_horizon = horizon_mse(test_arrays["preds"], test_arrays["trues"]).tolist()

    e01_fit, e01_scores = select_e01_calibrator(val_arrays["preds"], val_arrays["trues"])
    start = time.perf_counter()
    e01_pred = apply_e01_calibrator(test_arrays["preds"], e01_fit)
    e01_latency_ms = (time.perf_counter() - start) * 1000.0 / test_arrays["preds"].shape[0]
    e01_mae, e01_mse = metric(e01_pred, test_arrays["trues"])

    best_ema_alpha, ema_scores = select_ema_alpha(val_arrays["preds"], val_arrays["trues"], alpha_grid_ema)
    _, ema_state, ema_seen = streaming_ema_correction(val_arrays["preds"], val_arrays["trues"], best_ema_alpha)
    start = time.perf_counter()
    e02_pred, _, _ = streaming_ema_correction(
        test_arrays["preds"], test_arrays["trues"], best_ema_alpha, init_state=ema_state, init_seen=ema_seen
    )
    e02_latency_ms = (time.perf_counter() - start) * 1000.0 / test_arrays["preds"].shape[0]
    e02_mae, e02_mse = metric(e02_pred, test_arrays["trues"])

    e03_fit, e03_diag = select_e03_model(
        val_arrays["preds"],
        val_arrays["inputs"],
        val_arrays["trues"],
        alpha_grid_ridge,
        e03_min_tune_gain_pct,
    )
    start = time.perf_counter()
    e03_pred = apply_linear_residual_adapter(test_arrays["preds"], test_arrays["inputs"], e03_fit)
    e03_latency_ms = (time.perf_counter() - start) * 1000.0 / test_arrays["preds"].shape[0]
    e03_mae, e03_mse = metric(e03_pred, test_arrays["trues"])

    return {
        "raw": {
            "mae": raw_mae,
            "mse": raw_mse,
            "horizon_mse": raw_horizon,
        },
        "E01": {
            "mae": e01_mae,
            "mse": e01_mse,
            "delta_mse_pct": 100.0 * (e01_mse - raw_mse) / raw_mse,
            "latency_ms_per_window": e01_latency_ms,
            "variant": e01_fit["kind"] if e01_fit["kind"] == "identity" else (
                "per_channel_bias" if e01_fit["per_channel"] else "global_bias"
            ),
            "selection_scores": e01_scores,
            "params_added": 0 if e01_fit["kind"] == "identity" else (
                int(config.pred_len * config.c_out) if e01_fit["per_channel"] else int(config.pred_len)
            ),
            "horizon_mse": horizon_mse(e01_pred, test_arrays["trues"]).tolist(),
        },
        "E02": {
            "mae": e02_mae,
            "mse": e02_mse,
            "delta_mse_pct": 100.0 * (e02_mse - raw_mse) / raw_mse,
            "latency_ms_per_window": e02_latency_ms,
            "params_added": 0,
            "state_size": int(np.prod(ema_state.shape)),
            "alpha": best_ema_alpha,
            "alpha_scores": ema_scores,
            "horizon_mse": horizon_mse(e02_pred, test_arrays["trues"]).tolist(),
        },
        "E03": {
            "mae": e03_mae,
            "mse": e03_mse,
            "delta_mse_pct": 100.0 * (e03_mse - raw_mse) / raw_mse,
            "latency_ms_per_window": e03_latency_ms,
            "variant": e03_diag["selected"],
            "params_added": 0 if e03_fit["kind"] == "identity" else int(np.prod(e03_fit["weights"].shape)),
            "selection_scores": e03_diag,
            "horizon_mse": horizon_mse(e03_pred, test_arrays["trues"]).tolist(),
        },
    }


def tracker_row(
    config: RunConfig,
    method: str,
    method_result: dict[str, Any],
    raw_result: dict[str, Any],
) -> list[Any]:
    scheme_map = {
        "E01": "Horizon-wise bias / scale calibrator",
        "E02": "EMA residual memory",
        "E03": "Linear residual adapter",
    }
    uses_recent_gt = "Yes" if method == "E02" else "No"
    online_update = "Yes" if method == "E02" else "No"
    delta = method_result["delta_mse_pct"]
    kept = "yes" if delta < 0 else "no"
    failure_note = "" if delta < 0 else "No gain on this config"
    if delta <= -1.0:
        next_action = "Promote to W2 compare set"
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
    ws = wb["Run_Tracker"]
    if ws.max_row > 1:
        ws.delete_rows(2, ws.max_row - 1)
    for row in all_rows:
        ws.append(row)

    counts: dict[str, int] = {"E01": 0, "E02": 0, "E03": 0}
    for row in all_rows:
        exp_id = row[1]
        if exp_id in counts:
            counts[exp_id] += 1

    exp_ws = wb["Experiment_Matrix"]
    for row_idx in range(2, exp_ws.max_row + 1):
        exp_id = exp_ws.cell(row=row_idx, column=1).value
        if exp_id in counts:
            status = "done" if counts[exp_id] >= expected_runs_per_method else f"partial ({counts[exp_id]}/{expected_runs_per_method})"
            exp_ws.cell(row=row_idx, column=17).value = status
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
    method_scores: dict[str, list[float]] = {"E01": [], "E02": [], "E03": []}
    wins: dict[str, int] = {"E01": 0, "E02": 0, "E03": 0}
    for run_key, run_result in per_run.items():
        raw_mse = run_result["raw"]["mse"]
        best_method = min(["E01", "E02", "E03"], key=lambda name: run_result[name]["mse"])
        wins[best_method] += 1
        for method in ["E01", "E02", "E03"]:
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


def filtered_configs(
    configs: list[RunConfig],
    models: list[str] | None,
    datasets: list[str] | None,
    pred_lens: list[int] | None,
) -> list[RunConfig]:
    selected: list[RunConfig] = []
    for cfg in configs:
        if models is not None and cfg.model not in models:
            continue
        if datasets is not None and cfg.dataset not in datasets:
            continue
        if pred_lens is not None and cfg.pred_len not in pred_lens:
            continue
        selected.append(cfg)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run W1 post-hoc experiments on TSLib backbones.")
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
    parser.add_argument("--e03-min-tune-gain-pct", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

    configs = filtered_configs(
        w1_default_configs(train_epochs=cli.train_epochs),
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
            run_training(config, force=cli.force_train)
        val_arrays = collect_split_arrays(config, "val", force=cli.force_export)
        test_arrays = collect_split_arrays(config, "test", force=cli.force_export)
        results = evaluate_methods(
            config,
            val_arrays,
            test_arrays,
            alpha_grid_ema=cli.ema_alpha_grid,
            alpha_grid_ridge=cli.ridge_alpha_grid,
            e03_min_tune_gain_pct=cli.e03_min_tune_gain_pct,
        )
        run_key = f"{config.model}-{config.dataset}-pl{config.pred_len}"
        per_run[run_key] = results
        for method in ["E01", "E02", "E03"]:
            tracker_rows.append(tracker_row(config, method, results[method], results["raw"]))

        metrics_path = config.export_dir / "posthoc_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    write_csv(tracker_rows)
    write_summary(aggregate_summary(per_run))
    if not cli.no_workbook:
        update_workbook(tracker_rows, expected_runs_per_method=len(w1_default_configs()))


if __name__ == "__main__":
    main()
