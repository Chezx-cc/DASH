from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import e06_followup_utils as e06
import ts_posthoc_w1 as base
from e06_formal_protocol import (
    FORMAL_APPENDIX_BACKBONES,
    FORMAL_BACKBONE_FAMILY,
    FORMAL_BACKBONE_STAGE1_DATASETS,
    FORMAL_BACKBONE_STAGE1_PRED_LENS,
    FORMAL_MAIN_BACKBONES,
)


COSA_ROOT = PROJECT_ROOT.parent / "base"
TSLIB_DATA_ROOT = PROJECT_ROOT / "external" / "Time-Series-Library" / "dataset" / "ETT-small"
PYTHON = sys.executable

DEFAULT_MODELS = list(FORMAL_BACKBONE_FAMILY)
DEFAULT_DATASETS = list(FORMAL_BACKBONE_STAGE1_DATASETS)
DEFAULT_PRED_LENS = list(FORMAL_BACKBONE_STAGE1_PRED_LENS)

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "e06_new_backbones"
CACHE_ROOT = OUTPUT_ROOT / "exports"
LOG_ROOT = OUTPUT_ROOT / "logs"
CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "_checkpoints_e06safe"
LEGACY_CHECKPOINT_ROOT = COSA_ROOT / "checkpoints_e06safe"


@dataclass(frozen=True)
class BackboneConfig:
    model: str
    dataset: str
    pred_len: int
    seq_len: int = 96
    label_len: int = 48
    train_epochs: int = 10
    batch_size: int = 32
    seed: int = 2021
    learning_rate: float = 1e-4

    @property
    def run_key(self) -> str:
        return f"{self.model}-{self.dataset}-pl{self.pred_len}"

    @property
    def checkpoint_dir_rel(self) -> str:
        return f"./checkpoints_e06safe/{self.model}/{self.dataset}_{self.pred_len}_seed{self.seed}/"

    @property
    def checkpoint_dir_abs(self) -> Path:
        return CHECKPOINT_ROOT / self.model / f"{self.dataset}_{self.pred_len}_seed{self.seed}"

    @property
    def export_dir(self) -> Path:
        return CACHE_ROOT / f"seed{self.seed}" / self.model / self.dataset / f"pl{self.pred_len}"

    @property
    def train_batch_size(self) -> int:
        if self.model == "MICN":
            if self.dataset == "traffic":
                return 2 if self.pred_len >= 336 else 4
            if self.dataset == "electricity":
                return 4 if self.pred_len >= 336 else 8
            if self.dataset == "weather":
                return 16
            return self.batch_size
        if self.model == "FreTS":
            if self.dataset == "traffic":
                return 4
            if self.dataset == "electricity":
                return 8
            if self.dataset == "weather":
                return 16
            return self.batch_size
        return self.batch_size

    @property
    def eval_batch_size(self) -> int:
        if self.dataset == "traffic":
            return 4
        if self.dataset == "electricity":
            return 4
        if self.dataset == "weather":
            return 16
        if self.model == "iTransformer" and self.dataset == "traffic":
            return 16
        if self.model == "MICN":
            if self.dataset == "traffic":
                return 2 if self.pred_len >= 336 else 4
            if self.dataset == "electricity":
                return 4 if self.pred_len >= 336 else 8
            if self.dataset == "weather":
                return 16
            return 128
        if self.model == "FreTS":
            if self.dataset == "traffic":
                return 8
            if self.dataset == "electricity":
                return 16
            if self.dataset == "weather":
                return 64
            return 128
        if self.model == "OLS":
            if self.dataset == "traffic":
                return 16
            if self.dataset == "electricity":
                return 32
        return 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train new backbone baselines and evaluate the current E06-safe method."
    )
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--scope", choices=["all", "main", "appendix"], default="all")
    parser.add_argument(
        "--allow-extra-models",
        action="store_true",
        help="Allow models outside the locked 6-backbone family.",
    )
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--pred-lens", nargs="*", type=int, default=DEFAULT_PRED_LENS)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--output-dirname", default="e06_new_backbones")
    parser.add_argument("--output-root", default=None, help="Root directory for logs, arrays, and result tables.")
    parser.add_argument("--checkpoint-root", default=None, help="Root directory for trained raw backbone checkpoints.")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--force-e06", action="store_true")
    parser.add_argument(
        "--cleanup-exports",
        action="store_true",
        help="Delete cached val/test export arrays after each config is fully evaluated.",
    )
    return parser.parse_args()


def resolve_root(path_str: str | None, fallback: Path) -> Path:
    if path_str is None:
        return fallback
    path = Path(path_str)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def configure_roots(dirname: str, output_root: str | None, checkpoint_root: str | None) -> None:
    global OUTPUT_ROOT, CACHE_ROOT, LOG_ROOT, CHECKPOINT_ROOT
    OUTPUT_ROOT = resolve_root(output_root, PROJECT_ROOT / "outputs" / dirname)
    CACHE_ROOT = OUTPUT_ROOT / "exports"
    LOG_ROOT = OUTPUT_ROOT / "logs"
    CHECKPOINT_ROOT = resolve_root(checkpoint_root, PROJECT_ROOT / "outputs" / "_checkpoints_e06safe")


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def resolve_models(cli: argparse.Namespace) -> list[str]:
    models = dedupe_keep_order(list(cli.models))
    if cli.scope == "main":
        models = [model for model in models if model in FORMAL_MAIN_BACKBONES]
    elif cli.scope == "appendix":
        models = [model for model in models if model in FORMAL_APPENDIX_BACKBONES]

    if not cli.allow_extra_models:
        unknown = [model for model in models if model not in FORMAL_BACKBONE_FAMILY]
        if unknown:
            raise SystemExit(
                "Unsupported model(s) in locked backbone family mode: "
                f"{unknown}. Pass --allow-extra-models to override."
            )

    if not models:
        raise SystemExit("No models selected after applying scope/filter.")
    return models


def model_track(model: str) -> str:
    if model in FORMAL_MAIN_BACKBONES:
        return "main"
    if model in FORMAL_APPENDIX_BACKBONES:
        return "appendix"
    return "extra"


def selected_configs(cli: argparse.Namespace, models: list[str]) -> list[BackboneConfig]:
    configs: list[BackboneConfig] = []
    for model in models:
        for dataset in cli.datasets:
            for pred_len in cli.pred_lens:
                configs.append(
                    BackboneConfig(
                        model=model,
                        dataset=dataset,
                        pred_len=pred_len,
                        seq_len=cli.seq_len,
                        label_len=cli.label_len,
                        train_epochs=cli.train_epochs,
                        batch_size=cli.batch_size,
                        seed=cli.seed,
                        learning_rate=cli.learning_rate,
                    )
                )
    return configs


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cleanup_export_dir(config: BackboneConfig) -> None:
    if not config.export_dir.exists():
        return
    for child in config.export_dir.iterdir():
        if child.is_file():
            child.unlink(missing_ok=True)
    try:
        config.export_dir.rmdir()
    except OSError:
        pass


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_logged_command(command: list[str], cwd: Path, log_path: Path) -> tuple[str, float]:
    ensure_dir(log_path.parent)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - started
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    if process.returncode != 0:
        tail = "\n".join(text.splitlines()[-60:])
        raise RuntimeError(
            f"Command failed ({process.returncode}) in {cwd}:\n"
            f"{' '.join(command)}\n"
            f"---- log tail ----\n{tail}"
        )
    return text, elapsed


def copy_if_needed(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    dst.write_bytes(src.read_bytes())


def prepare_external_data() -> None:
    mapping = [
        (TSLIB_DATA_ROOT / "ETTh1.csv", COSA_ROOT / "data" / "ETTh1" / "ETTh1.csv"),
        (TSLIB_DATA_ROOT / "ETTm2.csv", COSA_ROOT / "data" / "ETTm2" / "ETTm2.csv"),
        (TSLIB_DATA_ROOT / "ETTh2.csv", COSA_ROOT / "data" / "ETTh2" / "ETTh2.csv"),
        (TSLIB_DATA_ROOT / "ETTm1.csv", COSA_ROOT / "data" / "ETTm1" / "ETTm1.csv"),
        (
            PROJECT_ROOT
            / "external"
            / "Time-Series-Library"
            / "dataset"
            / "auto_ltsf"
            / "electricity"
            / "electricity.csv",
            COSA_ROOT / "data" / "electricity" / "electricity.csv",
        ),
        (
            PROJECT_ROOT
            / "external"
            / "Time-Series-Library"
            / "dataset"
            / "auto_ltsf"
            / "traffic"
            / "traffic.csv",
            COSA_ROOT / "data" / "traffic" / "traffic.csv",
        ),
        (
            PROJECT_ROOT
            / "external"
            / "Time-Series-Library"
            / "dataset"
            / "auto_ltsf"
            / "weather"
            / "weather.csv",
            COSA_ROOT / "data" / "weather" / "weather.csv",
        ),
        (
            PROJECT_ROOT
            / "external"
            / "Time-Series-Library"
            / "dataset"
            / "auto_ltsf"
            / "exchange"
            / "exchange_rate.csv",
            COSA_ROOT / "data" / "exchange_rate" / "exchange_rate.csv",
        ),
        (
            PROJECT_ROOT
            / "external"
            / "Time-Series-Library"
            / "dataset"
            / "auto_ltsf"
            / "ili"
            / "illness.csv",
            COSA_ROOT / "data" / "illness" / "illness.csv",
        ),
    ]
    for src, dst in mapping:
        if not src.exists() and dst.exists():
            continue
        copy_if_needed(src, dst)


def purge_conflicting_cosa_modules() -> None:
    prefixes = ("config", "datasets", "layers", "models", "tta", "utils")
    root = str(COSA_ROOT.resolve())
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


def external_modules() -> dict[str, Any]:
    if str(COSA_ROOT) not in sys.path:
        sys.path.insert(0, str(COSA_ROOT))
    purge_conflicting_cosa_modules()
    from config import get_cfg_defaults
    from datasets.build import update_cfg_from_dataset
    from datasets.loader import get_val_dataloader, get_test_dataloader
    from models.build import build_model, load_best_model
    from models.forecast import forecast
    from utils.misc import prepare_inputs, set_devices, set_seeds

    return {
        "get_cfg_defaults": get_cfg_defaults,
        "update_cfg_from_dataset": update_cfg_from_dataset,
        "get_val_dataloader": get_val_dataloader,
        "get_test_dataloader": get_test_dataloader,
        "build_model": build_model,
        "load_best_model": load_best_model,
        "forecast": forecast,
        "prepare_inputs": prepare_inputs,
        "set_devices": set_devices,
        "set_seeds": set_seeds,
    }


def build_cfg(config: BackboneConfig) -> Any:
    mods = external_modules()
    cfg = mods["get_cfg_defaults"]()
    mods["update_cfg_from_dataset"](cfg, config.dataset)
    cfg.SEED = config.seed
    cfg.VISIBLE_DEVICES = 0
    cfg.DATA.BASE_DIR = str(COSA_ROOT / "data")
    cfg.DATA.NAME = config.dataset
    cfg.DATA.SEQ_LEN = config.seq_len
    cfg.DATA.LABEL_LEN = config.label_len
    cfg.DATA.PRED_LEN = config.pred_len
    cfg.MODEL.NAME = config.model
    cfg.MODEL.task_name = "long_term_forecast"
    cfg.MODEL.seq_len = config.seq_len
    cfg.MODEL.label_len = config.label_len
    cfg.MODEL.pred_len = config.pred_len
    cfg.TRAIN.CHECKPOINT_DIR = str(config.checkpoint_dir_abs)
    cfg.TRAIN.BATCH_SIZE = config.train_batch_size
    cfg.VAL.BATCH_SIZE = config.eval_batch_size
    cfg.TEST.BATCH_SIZE = config.eval_batch_size
    cfg.TTA.TAFAS.BATCH_SIZE = config.eval_batch_size
    cfg.DATA_LOADER.NUM_WORKERS = 0
    return cfg


def checkpoint_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        _ = torch.load(path, map_location="cpu")
        return True
    except Exception:
        return False


def ensure_backbone(config: BackboneConfig, force: bool = False) -> Path:
    checkpoint_path = config.checkpoint_dir_abs / "checkpoint_best.pth"
    if checkpoint_path.exists() and not force and checkpoint_is_valid(checkpoint_path):
        return checkpoint_path
    if checkpoint_path.exists() and not checkpoint_is_valid(checkpoint_path):
        checkpoint_path.unlink(missing_ok=True)

    legacy_checkpoint_path = (
        LEGACY_CHECKPOINT_ROOT
        / config.model
        / f"{config.dataset}_{config.pred_len}_seed{config.seed}"
        / "checkpoint_best.pth"
    )
    if legacy_checkpoint_path.exists() and not force and checkpoint_is_valid(legacy_checkpoint_path):
        ensure_dir(checkpoint_path.parent)
        copy_if_needed(legacy_checkpoint_path, checkpoint_path)
        if checkpoint_is_valid(checkpoint_path):
            return checkpoint_path
    if checkpoint_path.exists() and not checkpoint_is_valid(checkpoint_path):
        checkpoint_path.unlink(missing_ok=True)

    log_path = LOG_ROOT / "train" / f"{config.run_key}.log"
    command = [
        PYTHON,
        "main.py",
        "SEED",
        str(config.seed),
        "DATA.NAME",
        config.dataset,
        "DATA.SEQ_LEN",
        str(config.seq_len),
        "DATA.LABEL_LEN",
        str(config.label_len),
        "DATA.PRED_LEN",
        str(config.pred_len),
        "MODEL.NAME",
        config.model,
        "MODEL.task_name",
        "long_term_forecast",
        "MODEL.seq_len",
        str(config.seq_len),
        "MODEL.label_len",
        str(config.label_len),
        "MODEL.pred_len",
        str(config.pred_len),
        "TRAIN.ENABLE",
        "True",
        "TEST.ENABLE",
        "False",
        "TTA.ENABLE",
        "False",
        "TRAIN.CHECKPOINT_DIR",
        str(config.checkpoint_dir_abs),
        "TRAIN.BATCH_SIZE",
        str(config.train_batch_size),
        "VAL.BATCH_SIZE",
        str(config.eval_batch_size),
        "TEST.BATCH_SIZE",
        str(config.eval_batch_size),
        "TRAIN.EVAL_PERIOD",
        "1",
        "SOLVER.MAX_EPOCH",
        str(config.train_epochs),
        "SOLVER.BASE_LR",
        f"{config.learning_rate:g}",
        "SOLVER.WEIGHT_DECAY",
        "0.0",
        "DATA_LOADER.NUM_WORKERS",
        "0",
        "VISIBLE_DEVICES",
        "0",
        "RESULT_DIR",
        "./results/E06_SAFE_RAW/",
    ]
    run_logged_command(command, COSA_ROOT, log_path)
    if not checkpoint_path.exists() or not checkpoint_is_valid(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint after training: {checkpoint_path}")
    return checkpoint_path


def collect_predictor_like_arrays(
    cfg: Any,
    loader: Any,
    forecast: Any,
    prepare_inputs: Any,
    model: Any,
) -> dict[str, np.ndarray]:
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    inputs: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(batch)
            batch_start = 0
            batch_end = 0
            while batch_end < len(enc_window_all):
                batch_end = min(batch_start + cfg.TTA.TAFAS.BATCH_SIZE, len(enc_window_all))
                sub_batch = (
                    enc_window_all[batch_start:batch_end],
                    enc_window_stamp_all[batch_start:batch_end],
                    dec_window_all[batch_start:batch_end],
                    dec_window_stamp_all[batch_start:batch_end],
                )
                pred, true = forecast(cfg, sub_batch, model, None)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                inputs.append(enc_window_all[batch_start:batch_end].detach().cpu().numpy())
                batch_start = batch_end

    return {
        "preds": np.concatenate(preds, axis=0),
        "trues": np.concatenate(trues, axis=0),
        "inputs": np.concatenate(inputs, axis=0),
    }


def cache_paths(config: BackboneConfig, split: str) -> tuple[Path, Path]:
    return config.export_dir / f"{split}.npz", config.export_dir / f"{split}_metadata.json"


def collect_arrays(config: BackboneConfig, split: str, force: bool = False) -> dict[str, np.ndarray]:
    npz_path, meta_path = cache_paths(config, split)
    if npz_path.exists() and not force:
        try:
            with np.load(npz_path) as bundle:
                return {key: bundle[key] for key in bundle.files}
        except (OSError, ValueError, zipfile.BadZipFile):
            npz_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)

    ensure_dir(config.export_dir)
    ensure_backbone(config, force=False)
    cfg = build_cfg(config)
    mods = external_modules()
    mods["set_devices"](cfg.VISIBLE_DEVICES)
    mods["set_seeds"](cfg.SEED)

    model = mods["build_model"](cfg)
    model = mods["load_best_model"](cfg, model)
    loader = mods["get_val_dataloader"](cfg) if split == "val" else mods["get_test_dataloader"](cfg)

    arrays = collect_predictor_like_arrays(
        cfg=cfg,
        loader=loader,
        forecast=mods["forecast"],
        prepare_inputs=mods["prepare_inputs"],
        model=model,
    )
    np.savez_compressed(npz_path, **arrays)
    mae, mse = base.metric(arrays["preds"], arrays["trues"])
    write_json(
        meta_path,
        {
            "run_key": config.run_key,
            "split": split,
            "seed": config.seed,
            "pred_shape": list(arrays["preds"].shape),
            "input_shape": list(arrays["inputs"].shape),
            "mse": float(mse),
            "mae": float(mae),
            "checkpoint_dir": str(config.checkpoint_dir_abs),
        },
    )
    return arrays


def collect_val_test_arrays_in_memory(
    config: BackboneConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Export validation and test predictions without writing large NPZ files.

    Each split gets a freshly seeded and freshly loaded model.  The upstream
    backbone loader leaves dropout-bearing models in training mode, so running
    validation first would otherwise consume RNG state and change the canonical
    test prediction.  Separate model instances also prevent split-to-split
    state contamination from any stateful layer.
    """
    ensure_backbone(config, force=False)
    cfg = build_cfg(config)
    mods = external_modules()
    mods["set_devices"](cfg.VISIBLE_DEVICES)

    def collect_split(split: str) -> dict[str, np.ndarray]:
        mods["set_seeds"](cfg.SEED)
        model = mods["build_model"](cfg)
        model = mods["load_best_model"](cfg, model)
        loader = mods["get_val_dataloader"](cfg) if split == "val" else mods["get_test_dataloader"](cfg)
        try:
            return collect_predictor_like_arrays(
                cfg=cfg,
                loader=loader,
                forecast=mods["forecast"],
                prepare_inputs=mods["prepare_inputs"],
                model=model,
            )
        finally:
            del loader
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    val_arrays = collect_split("val")
    test_arrays = collect_split("test")
    return val_arrays, test_arrays


def run_e06_safe(
    config: BackboneConfig,
    val_arrays: dict[str, np.ndarray],
    test_arrays: dict[str, np.ndarray],
    force: bool = False,
) -> dict[str, Any]:
    cache_path = OUTPUT_ROOT / "parsed" / f"e06_safe_{config.run_key}_seed{config.seed}.json"
    if cache_path.exists() and not force:
        return read_json(cache_path)

    bundle, diag = e06.fit_gate_variants(val_arrays, mode="block_gate")
    started = time.perf_counter()
    pred, info = e06.apply_gate_bundle(bundle, test_arrays)
    latency_ms = (time.perf_counter() - started) * 1000.0 / test_arrays["preds"].shape[0]
    mae, mse = base.metric(pred, test_arrays["trues"])
    payload = {
        "run_key": config.run_key,
        "model": config.model,
        "dataset": config.dataset,
        "pred_len": config.pred_len,
        "seed": config.seed,
        "method": "E06-safe",
        "mse": float(mse),
        "mae": float(mae),
        "latency_ms_per_window": float(latency_ms),
        "gate_enabled": bool(info["enabled"]),
        "update_count": int(info["updates"]),
        "forced_alpha": float(info["forced_alpha"]),
        "forced_forgetting": float(info["forced_forgetting"]),
        "block_gain_pct": [float(x) for x in diag["block_gain_pct"]],
        "status": "ok",
    }
    write_json(cache_path, payload)
    return payload


def method_row(
    config: BackboneConfig,
    method: str,
    mse: float,
    mae: float,
    raw_mse: float,
    latency_ms_per_window: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "run_key": config.run_key,
        "model": config.model,
        "track": model_track(config.model),
        "dataset": config.dataset,
        "pred_len": config.pred_len,
        "seed": config.seed,
        "method": method,
        "mse": float(mse),
        "mae": float(mae),
        "raw_mse": float(raw_mse),
        "delta_mse_pct": e06.delta_pct(float(mse), float(raw_mse)),
        "harmful": int(float(mse) > float(raw_mse)),
        "latency_ms_per_window": float(latency_ms_per_window),
        "status": "ok",
    }
    if extra:
        row.update(extra)
    return row


def summarize(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, sub in detail_df.groupby("method", observed=False):
        rows.append(
            {
                "method": method,
                "runs": int(sub.shape[0]),
                "harmful_runs": int(sub["harmful"].sum()),
                "mean_delta_mse_pct": float(sub["delta_mse_pct"].mean()),
                "mean_latency_ms_per_window": float(sub["latency_ms_per_window"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def summarize_by_model(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, method), sub in detail_df.groupby(["model", "method"], observed=False):
        rows.append(
            {
                "model": model,
                "method": method,
                "runs": int(sub.shape[0]),
                "harmful_runs": int(sub["harmful"].sum()),
                "mean_delta_mse_pct": float(sub["delta_mse_pct"].mean()),
                "mean_latency_ms_per_window": float(sub["latency_ms_per_window"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "method"]).reset_index(drop=True)


def summarize_by_track(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (track, method), sub in detail_df.groupby(["track", "method"], observed=False):
        rows.append(
            {
                "track": track,
                "method": method,
                "runs": int(sub.shape[0]),
                "harmful_runs": int(sub["harmful"].sum()),
                "mean_delta_mse_pct": float(sub["delta_mse_pct"].mean()),
                "mean_latency_ms_per_window": float(sub["latency_ms_per_window"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["track", "method"]).reset_index(drop=True)


def main() -> None:
    cli = parse_args()
    configure_roots(cli.output_dirname, cli.output_root, cli.checkpoint_root)
    ensure_dir(OUTPUT_ROOT)
    ensure_dir(CACHE_ROOT)
    ensure_dir(LOG_ROOT)
    prepare_external_data()

    selected_models = resolve_models(cli)
    configs = selected_configs(cli, selected_models)
    if not configs:
        raise SystemExit("No configs selected.")

    detail_rows: list[dict[str, Any]] = []
    for config in configs:
        print(f"[e06-new] {config.run_key}")
        ensure_backbone(config, force=cli.force_train)
        val_arrays = collect_arrays(config, "val", force=cli.force_export)
        test_arrays = collect_arrays(config, "test", force=cli.force_export)
        raw_mae, raw_mse = base.metric(test_arrays["preds"], test_arrays["trues"])
        detail_rows.append(
            method_row(
                config,
                "raw",
                mse=float(raw_mse),
                mae=float(raw_mae),
                raw_mse=float(raw_mse),
                latency_ms_per_window=0.0,
                extra={"gate_enabled": None, "update_count": 0},
            )
        )

        e06_row = run_e06_safe(config, val_arrays, test_arrays, force=cli.force_e06)
        detail_rows.append(
            method_row(
                config,
                "E06-safe",
                mse=float(e06_row["mse"]),
                mae=float(e06_row["mae"]),
                raw_mse=float(raw_mse),
                latency_ms_per_window=float(e06_row["latency_ms_per_window"]),
                extra={
                    "gate_enabled": bool(e06_row["gate_enabled"]),
                    "update_count": int(e06_row["update_count"]),
                    "forced_alpha": float(e06_row["forced_alpha"]),
                    "forced_forgetting": float(e06_row["forced_forgetting"]),
                    "block_gain_pct": ",".join(f"{x:.4f}" for x in e06_row["block_gain_pct"]),
                },
            )
        )
        if cli.cleanup_exports:
            cleanup_export_dir(config)

    detail_df = pd.DataFrame(detail_rows).sort_values(["model", "dataset", "pred_len", "method"]).reset_index(drop=True)
    summary_df = summarize(detail_df)
    model_summary_df = summarize_by_model(detail_df)
    track_summary_df = summarize_by_track(detail_df)

    table_df = (
        detail_df.pivot_table(
            index=["run_key", "model", "track", "dataset", "pred_len"],
            columns="method",
            values=["mse", "mae", "delta_mse_pct", "latency_ms_per_window", "harmful"],
            aggfunc="first",
        )
        .sort_index()
    )
    table_df.columns = [f"{method}_{metric}" for metric, method in table_df.columns]
    table_df = table_df.reset_index()
    main_table_df = table_df[table_df["track"] == "main"].reset_index(drop=True)
    appendix_table_df = table_df[table_df["track"] == "appendix"].reset_index(drop=True)

    artifact_prefix = OUTPUT_ROOT.name
    detail_path = OUTPUT_ROOT / f"{artifact_prefix}_detail.csv"
    summary_path = OUTPUT_ROOT / f"{artifact_prefix}_summary.csv"
    model_summary_path = OUTPUT_ROOT / f"{artifact_prefix}_model_summary.csv"
    track_summary_path = OUTPUT_ROOT / f"{artifact_prefix}_track_summary.csv"
    table_path = OUTPUT_ROOT / f"{artifact_prefix}_table.csv"
    main_table_path = OUTPUT_ROOT / f"{artifact_prefix}_main_table.csv"
    appendix_table_path = OUTPUT_ROOT / f"{artifact_prefix}_appendix_table.csv"

    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    model_summary_df.to_csv(model_summary_path, index=False)
    track_summary_df.to_csv(track_summary_path, index=False)
    table_df.to_csv(table_path, index=False)
    main_table_df.to_csv(main_table_path, index=False)
    appendix_table_df.to_csv(appendix_table_path, index=False)

    write_json(
        OUTPUT_ROOT / f"{artifact_prefix}_summary.json",
        {
            "settings": {
                "models": selected_models,
                "scope": cli.scope,
                "allow_extra_models": bool(cli.allow_extra_models),
                "datasets": cli.datasets,
                "pred_lens": cli.pred_lens,
                "seq_len": cli.seq_len,
                "label_len": cli.label_len,
                "train_epochs": cli.train_epochs,
                "batch_size": cli.batch_size,
                "learning_rate": cli.learning_rate,
                "seed": cli.seed,
                "cleanup_exports": bool(cli.cleanup_exports),
                "e06_safe_variant": "block_gate",
                "main_backbones": FORMAL_MAIN_BACKBONES,
                "appendix_backbones": FORMAL_APPENDIX_BACKBONES,
            },
            "detail_csv": str(detail_path),
            "summary_csv": str(summary_path),
            "model_summary_csv": str(model_summary_path),
            "track_summary_csv": str(track_summary_path),
            "table_csv": str(table_path),
            "main_table_csv": str(main_table_path),
            "appendix_table_csv": str(appendix_table_path),
            "configs": [config.run_key for config in configs],
        },
    )


if __name__ == "__main__":
    main()
