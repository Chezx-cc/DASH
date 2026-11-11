from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORMAL_PROTOCOL_NAME = "E06-safe-formal-v1"
FORMAL_PROTOCOL_LOCK_DATE = "2026-04-18"

E06_SAFE_ALPHA_GRID = [300.0, 1000.0, 3000.0, 10000.0]
E06_SAFE_FORGETTING_GRID = [1.0, 0.995]
E06_SAFE_MIN_TUNE_GAIN_PCT = 0.5
E06_SAFE_GATE_MODE = "block_gate"
E06_SAFE_GATE_BLOCKS = 4
E06_SAFE_GATE_REQUIRED_IMPROVED = 3

FORMAL_POINT_METHODS = ["raw", "e03", "e06_safe"]
FORMAL_EXTERNAL_METHODS = ["raw", "E06-safe", "TAFAS", "COSA"]
FORMAL_DEFAULT_SEEDS = [2, 42, 2021]
FORMAL_POSITIVE_PRED_LENS = [96, 336, 720]
FORMAL_CONTROL_RUN_KEY = "PatchTST-ETTh1-pl336"

# Stage-1 backbone family coverage (paper-focused scope control).
FORMAL_MAIN_BACKBONES = ["DLinear", "PatchTST", "iTransformer", "MICN"]
FORMAL_APPENDIX_BACKBONES = ["OLS", "FreTS"]
FORMAL_BACKBONE_FAMILY = FORMAL_MAIN_BACKBONES + FORMAL_APPENDIX_BACKBONES
FORMAL_BACKBONE_STAGE1_DATASETS = ["ETTh1", "ETTm2"]
FORMAL_BACKBONE_STAGE1_PRED_LENS = [96, 336]

FORMAL_POSITIVE_RUN_SPECS = [
    {"model": "DLinear", "dataset": "ETTm2", "pred_len": 96, "stack": "tslib", "role": "positive"},
    {"model": "DLinear", "dataset": "ETTm2", "pred_len": 336, "stack": "tslib", "role": "positive"},
    {"model": "DLinear", "dataset": "ETTm2", "pred_len": 720, "stack": "tslib", "role": "positive"},
    {"model": "iTransformer", "dataset": "ETTm2", "pred_len": 96, "stack": "cosa", "role": "positive"},
    {"model": "iTransformer", "dataset": "ETTm2", "pred_len": 336, "stack": "cosa", "role": "positive"},
    {"model": "iTransformer", "dataset": "ETTm2", "pred_len": 720, "stack": "cosa", "role": "positive"},
]

FORMAL_CONTROL_SPECS = [
    {"model": "PatchTST", "dataset": "ETTh1", "pred_len": 336, "stack": "tslib", "role": "control"},
]


def protocol_payload() -> dict[str, Any]:
    return {
        "protocol_name": FORMAL_PROTOCOL_NAME,
        "locked_on": FORMAL_PROTOCOL_LOCK_DATE,
        "e06_safe": {
            "alpha_grid": E06_SAFE_ALPHA_GRID,
            "forgetting_grid": E06_SAFE_FORGETTING_GRID,
            "min_tune_gain_pct": E06_SAFE_MIN_TUNE_GAIN_PCT,
            "gate_mode": E06_SAFE_GATE_MODE,
            "gate_blocks": E06_SAFE_GATE_BLOCKS,
            "gate_required_improved": E06_SAFE_GATE_REQUIRED_IMPROVED,
        },
        "formal_methods": FORMAL_POINT_METHODS,
        "external_methods": FORMAL_EXTERNAL_METHODS,
        "default_seeds": FORMAL_DEFAULT_SEEDS,
        "backbone_family_stage1": {
            "main_backbones": FORMAL_MAIN_BACKBONES,
            "appendix_backbones": FORMAL_APPENDIX_BACKBONES,
            "all_backbones": FORMAL_BACKBONE_FAMILY,
            "datasets": FORMAL_BACKBONE_STAGE1_DATASETS,
            "pred_lens": FORMAL_BACKBONE_STAGE1_PRED_LENS,
        },
        "positive_run_specs": FORMAL_POSITIVE_RUN_SPECS,
        "control_specs": FORMAL_CONTROL_SPECS,
        "notes": [
            "Core algorithm is locked. Future formal batches should only extend evidence.",
            "Do not introduce warmup, late-weighted, lambda, or other new E06 variants into formal runs.",
            "Formal point-method comparisons only keep raw / E03 / E06-safe.",
            "Backbone expansion should stay within the 6-model family (4 main + 2 appendix).",
        ],
    }


def run_key(model: str, dataset: str, pred_len: int) -> str:
    return f"{model}-{dataset}-pl{pred_len}"


def protocol_banner() -> str:
    return (
        f"{FORMAL_PROTOCOL_NAME}: alpha_grid={E06_SAFE_ALPHA_GRID}, "
        f"forgetting_grid={E06_SAFE_FORGETTING_GRID}, "
        f"min_tune_gain_pct={E06_SAFE_MIN_TUNE_GAIN_PCT}, "
        f"{E06_SAFE_GATE_MODE}={E06_SAFE_GATE_BLOCKS} blocks require "
        f"{E06_SAFE_GATE_REQUIRED_IMPROVED}/{E06_SAFE_GATE_BLOCKS} improvements"
    )
