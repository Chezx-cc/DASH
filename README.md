# DASH Reproducibility Package

This repository provides the code supplement for **DASH: Stable–Dynamic
Periodic Memory for Test-Time Adaptation in Time-Series Forecasting**. It
contains the DASH implementation, experiment launchers, a deterministic CPU
smoke test, and one example checkpoint.

## Contributors

- chezhuoxun
- chenyiyun

## Quick verification

After installing the dependencies below, run from the repository root:

```bash
python scripts/run_smoke.py
```

Windows users can instead double-click `smoke_test.bat` or run it from
PowerShell.  Linux/macOS users can run `bash smoke_test.sh`.

The packaged unit tests can be run from the same directory:

```bash
python -m pytest repo/tests -q
```


The exact values are deterministic up to normal floating-point variation.  The
test creates a small synthetic periodic stream in memory, runs the same frozen
`K=3`, `beta=0.90`, 75%-validation DASH calibration/evaluation functions used
by the main experiments, and checks that the three calibration stages have no
shared target timestamps.  It needs neither datasets nor checkpoints.

## Environment

The original full experiments used Ubuntu 22.04, Python 3.10.8, PyTorch
2.1.2+cu121, and CUDA 12.1.  The review-only smoke test is CPU-compatible and
was additionally checked on Windows.  We recommend Python 3.10.

Conda setup:

```bash
conda env create -f environment.yml
conda activate dash-review
python scripts/run_smoke.py
```

Pip setup in an existing Python 3.10 virtual environment:

```bash
python -m pip install -r requirements-review.txt
python scripts/run_smoke.py
```

`requirements-review.txt` freezes the versions used for the release checks.
For a GPU-specific PyTorch wheel, install PyTorch 2.1.2 from the index matching
the local CUDA driver before installing the remaining requirements.

Typical CPU runtimes are under 10 seconds for the smoke test. Full 504-run
reproduction is hardware-dependent, and the supplied launcher is designed for
eight GPUs.

## Frozen defaults

- validation fraction: `0.75`
- completed memory cycles: `K=3`
- recent-memory retention: `beta=0.90`
- period set: `{96, 168, 336}`
- calibration protocol: chronological target-timestamp-purged `V1/V2/V3`

The shared definitions are in `repo/scripts/dash_selected_default.py`.  The
main selected-setting entry point is
`repo/scripts/run_dash_purged_default_full_period_val75.py`, and the
eight-GPU launcher is
`repo/scripts/run_dash_main_full504_K3_beta090_val75_8gpu.sh`.

## Paper results and artifact mapping

The rebuild script requires the experiment CSVs and `SHA256SUMS` in
`paper_results/`. Supply those inputs separately before running:

```bash
python scripts/rebuild_paper_artifacts.py
```

## Full experiment reproduction

Supply the public benchmark datasets and full pretrained checkpoint collection
with this layout:

```text
base/data/ETTh1/ETTh1.csv
base/data/ETTh2/ETTh2.csv
base/data/ETTm1/ETTm1.csv
base/data/ETTm2/ETTm2.csv
base/data/electricity/electricity.csv
base/data/traffic/traffic.csv
base/data/weather/weather.csv
checkpoints/<BACKBONE>/<DATASET>_<HORIZON>_seed<SEED>/checkpoint_best.pth
```

The panel is 7 datasets x 6 backbones x 4 horizons x 3 seeds = 504 runs.
Backbones are DLinear, PatchTST, iTransformer, MICN, OLS, and FreTS; horizons
are 96, 192, 336, and 720; seeds are 2, 42, and 2021.

After placing data and checkpoints, launch the selected configuration:

```bash
bash repo/scripts/run_dash_main_full504_K3_beta090_val75_8gpu.sh
```

The launcher expects `paper_results/raw_metrics_504.csv` by default. Set
`RAW_METRICS_CSV` to use another copy of that file. `PYTHON` and
`CHECKPOINT_ROOT` are also configurable. The launcher first performs one real
checkpoint smoke run, then starts the 504-run panel. Per-run CSVs, failure
records, audit metadata, and merged summaries are written below `repo/outputs/`.
Calibration parameters are frozen before the chronological test stream is
evaluated; the test correction never reads future or current test targets.

## Package layout

```text
README.md
environment.yml
requirements-review.txt
scripts/                    # one-command smoke and paper rebuild
repo/                       # DASH implementation, launchers, and tests
base/                       # integrated forecasting runtime
sample_checkpoint/          # one representative, non-required checkpoint
```

The synthetic smoke test runs without the representative checkpoint. The full
experiment panel requires the complete checkpoint collection.
