#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
CKPT="${CHECKPOINT_ROOT:-$REPO/../checkpoints}"
RAW="${RAW_METRICS_CSV:-$REPO/../paper_results/raw_metrics_504.csv}"
SMOKE=outputs/dash_selected_smoke
OUT=outputs/dash_selected_full504
STATUS=outputs/dash_selected_full504.launch.json
MODELS=(DLinear PatchTST iTransformer MICN OLS FreTS)
DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Electricity Traffic Weather)
HORIZONS=(96 192 336 720)
SEEDS=(2 42 2021)

cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE

write_status() {
  "$PY" - "$STATUS" "$1" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"stage":sys.argv[2],"updated_at":datetime.now(timezone.utc).isoformat()},indent=2)+"\n",encoding="utf-8")
PY
}

run_experiment() {
  local out_root=$1
  shift
  "$PY" -u scripts/run_dash_purged_default_full_period_val75.py \
    --checkpoint-root "$CKPT" --raw-metrics-csv "$RAW" --out-root "$out_root" \
    --models "${MODELS[@]}" --datasets "${DATASETS[@]}" \
    --pred-lens "${HORIZONS[@]}" --seeds "${SEEDS[@]}" \
    --validation-fraction 0.75 "$@"
}

write_status smoke_running
run_experiment "$SMOKE" --workers 1 --gpu-ids 0 --limit 1 > "${SMOKE}.log" 2>&1
"$PY" - "$SMOKE" 1 <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
r=Path(sys.argv[1]); expected=int(sys.argv[2])
s=json.loads((r/'runner_status.json').read_text()); d=pd.read_csv(r/'detail.csv')
assert s['status']=='completed' and s['configs']==expected and s['rows']==expected and s['failures']==0, s
assert d.config_key.nunique()==expected and d.groupby('config_key').size().eq(1).all()
assert set(d.cycles)=={3} and set(d.recent_decay.round(8))=={0.9} and d.is_default.all()
assert set(d.period_set)=={'P96+168+336'} and set(d.validation_fraction.round(8))=={0.75}
assert not d.uses_test_for_selection.any() and not d.test_correction_reads_test_truth.any()
PY

write_status full_running
run_experiment "$OUT" --workers 8 --gpu-ids 0 1 2 3 4 5 6 7 > "${OUT}.log" 2>&1
"$PY" - "$OUT" 504 <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
r=Path(sys.argv[1]); expected=int(sys.argv[2])
s=json.loads((r/'runner_status.json').read_text()); d=pd.read_csv(r/'detail.csv')
assert s['status']=='completed' and s['configs']==expected and s['rows']==expected and s['failures']==0, s
assert d.config_key.nunique()==expected and d.groupby('config_key').size().eq(1).all()
assert d[['dataset','backbone','pred_len','seed']].drop_duplicates().shape[0]==expected
assert set(d.cycles)=={3} and set(d.recent_decay.round(8))=={0.9} and d.is_default.all()
assert set(d.period_set)=={'P96+168+336'} and set(d.validation_fraction.round(8))=={0.75}
assert not d.uses_test_for_selection.any() and not d.test_correction_reads_test_truth.any()
assert not d[['raw_mse','raw_mae','method_mse','method_mae','gain_pct','mae_gain_pct']].isna().any().any()
PY
write_status completed
