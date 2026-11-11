"""Frozen round: recent memory by default, stable fallback on validation-supported shocks."""
from __future__ import annotations

import argparse, gc, json, multiprocessing as mp, sys, time, traceback
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

import run_cf_long_memory_anchor as lm
import run_cf_residual_expert as core
import run_final_full504 as prior
import run_internal_offline_candidates as base
import run_e06_new_backbones as nb

METHOD = "dash_shock_guard_round1"
QUANTILES = (0.80, 0.90, 0.95, 0.99)
ALPHAS = (0.25, 0.50, 0.75, 1.00)
CANDIDATES = (("never", 0.0),) + tuple((q, a) for q in reversed(QUANTILES) for a in ALPHAS)


def shock_score(dynamic: np.ndarray, bins: np.ndarray) -> np.ndarray:
    out = np.zeros((len(dynamic), 4), dtype=np.float32)
    for b in range(4):
        h = np.where(bins == b)[0]
        out[:, b] = np.sqrt(np.mean(np.square(dynamic[:, h], dtype=np.float64), axis=(1, 2)))
    return out


def fit_thresholds(stable, recent, val, bins):
    idx = lm.sampled_indices(0, stable["c1"], 96)
    c6 = prior.corr(stable, val["inputs"], val["preds"], idx, bins)
    c7 = prior.corr(recent, val["inputs"], val["preds"], idx, bins)
    scores = shock_score(c7 - c6, bins)
    return {q: np.quantile(scores, q, axis=0).astype(np.float32) for q in QUANTILES}


def tune(stable, recent, val, bins, thresholds):
    sums = np.zeros((4, len(CANDIDATES)), dtype=np.float64)
    counts = np.zeros(4, dtype=np.int64)
    idx = np.arange(stable["c1"], stable["c2"])
    step = lm.chunk_size(val["preds"])
    for start in range(0, len(idx), step):
        ii = idx[start:start + step]
        c6 = prior.corr(stable, val["inputs"], val["preds"], ii, bins)
        c7 = prior.corr(recent, val["inputs"], val["preds"], ii, bins)
        scores = shock_score(c7 - c6, bins)
        for b in range(4):
            h = np.where(bins == b)[0]
            err = val["preds"][ii][:, h] + c7[:, h] - val["trues"][ii][:, h]
            diff = c6[:, h] - c7[:, h]
            e2 = np.sum(np.square(err, dtype=np.float64), axis=(1, 2))
            ed = np.sum(err.astype(np.float64) * diff.astype(np.float64), axis=(1, 2))
            d2 = np.sum(np.square(diff, dtype=np.float64), axis=(1, 2))
            for j, (q, alpha) in enumerate(CANDIDATES):
                scale = np.zeros(len(ii), dtype=np.float64) if q == "never" else alpha * (scores[:, b] > thresholds[q][b])
                sums[b, j] += np.sum(e2 + 2.0 * scale * ed + np.square(scale) * d2)
            counts[b] += err.size
    q_selected, alpha = [], np.zeros(4, dtype=np.float32)
    for b in range(4):
        j = int(np.argmin(sums[b]))
        q, a = CANDIDATES[j]
        q_selected.append(q)
        alpha[b] = float(a)
        if stable["gated"][b] <= 0:
            q_selected[-1], alpha[b] = "never", 0.0
    return q_selected, alpha


def route(stable, recent, val, bins, thresholds, q_selected, alpha):
    idx = np.arange(stable["c2"], len(val["preds"]))
    base_o = np.zeros((len(idx), 4)); cand_o = np.zeros_like(base_o)
    step = lm.chunk_size(val["preds"])
    for start in range(0, len(idx), step):
        ii = idx[start:start + step]
        c6 = prior.corr(stable, val["inputs"], val["preds"], ii, bins, True)
        c7 = prior.corr(recent, val["inputs"], val["preds"], ii, bins)
        scores = shock_score(c7 - prior.corr(stable, val["inputs"], val["preds"], ii, bins), bins)
        for b in range(4):
            h = np.where(bins == b)[0]
            err = val["preds"][ii][:, h] + c7[:, h] - val["trues"][ii][:, h]
            trigger = np.zeros(len(ii), dtype=bool) if q_selected[b] == "never" else scores[:, b] > thresholds[q_selected[b]][b]
            candidate = err + alpha[b] * trigger[:, None, None] * (c6[:, h] - c7[:, h])
            base_o[start:start + len(ii), b] = np.mean(np.square(err, dtype=np.float64), axis=(1, 2))
            cand_o[start:start + len(ii), b] = np.mean(np.square(candidate, dtype=np.float64), axis=(1, 2))
    audit = {"gain": [], "positive": [], "worst": []}
    gated = alpha.copy()
    for b in range(4):
        blocks=[]
        for jj in np.array_split(np.arange(len(idx)), min(4, len(idx))):
            a=float(np.mean(base_o[jj,b])); m=float(np.mean(cand_o[jj,b])); blocks.append(100*(a-m)/max(a,1e-12))
        a=float(np.mean(base_o[:,b]));m=float(np.mean(cand_o[:,b]));gain=100*(a-m)/max(a,1e-12)
        positive=sum(x>0 for x in blocks);worst=min(blocks)
        if gain <= 0 or positive < 3 or worst < -0.10:
            gated[b]=0.0;q_selected[b]="never"
        audit["gain"].append(gain);audit["positive"].append(positive);audit["worst"].append(worst)
    return q_selected, gated, audit


def evaluate(config, val0, test0):
    val={k:np.asarray(val0[k],dtype=np.float32) for k in ("preds","trues","inputs")}
    test={k:np.asarray(test0[k],dtype=np.float32) for k in ("preds","trues","inputs")}
    bins=lm.cf.horizon_bin_ids(config.pred_len,4)
    stable=prior.prepare(lm.VARIANT_BY_ID["lm06_median3"],val,bins)
    recent=prior.prepare(lm.VARIANT_BY_ID["lm07_ew3"],val,bins)
    thresholds=fit_thresholds(stable,recent,val,bins)
    q_selected,alpha=tune(stable,recent,val,bins,thresholds)
    q_selected,alpha,route_audit=route(stable,recent,val,bins,thresholds,q_selected,alpha)
    timeline,current,stitch=lm.test_timeline(val["inputs"],val["trues"],test["inputs"])
    for state in (stable,recent):state["timeline"],state["current"]=timeline,current
    raw_sq=raw_ab=met_sq=met_ab=corr_abs=0.;count=triggers=0
    idx=np.arange(len(test["preds"]));step=lm.chunk_size(test["preds"])
    for start in range(0,len(idx),step):
        ii=idx[start:start+step]
        c6e=prior.corr(stable,test["inputs"],test["preds"],ii,bins)
        c6s=prior.corr(stable,test["inputs"],test["preds"],ii,bins,True)
        c7=prior.corr(recent,test["inputs"],test["preds"],ii,bins)
        scores=shock_score(c7-c6e,bins); correction=c7.copy()
        for b in range(4):
            h=np.where(bins==b)[0]
            trigger=np.zeros(len(ii),dtype=bool) if q_selected[b]=="never" else scores[:,b]>thresholds[q_selected[b]][b]
            correction[:,h]+=alpha[b]*trigger[:,None,None]*(c6s[:,h]-c7[:,h]);triggers+=int(trigger.sum())
        rawerr=test["preds"][ii]-test["trues"][ii];err=rawerr+correction
        raw_sq+=np.sum(np.square(rawerr,dtype=np.float64));raw_ab+=np.sum(np.abs(rawerr),dtype=np.float64)
        met_sq+=np.sum(np.square(err,dtype=np.float64));met_ab+=np.sum(np.abs(err),dtype=np.float64)
        corr_abs+=np.sum(np.abs(correction),dtype=np.float64);count+=rawerr.size
    raw_mse,raw_mae=raw_sq/count,raw_ab/count;mse,mae=met_sq/count,met_ab/count
    meta={**stitch,"threshold_quantiles":q_selected,"alpha":alpha.tolist(),"route":route_audit,"triggered_origin_bins":triggers,"registry":"configs/dash_shock_guard_round1_frozen_20260723.json"}
    return [{"dataset":core.display_dataset(config.dataset),"backbone":config.model,"model":config.model,"pred_len":config.pred_len,"seed":config.seed,"config_key":core.config_key(config),"method":METHOD,"raw_mse":raw_mse,"raw_mae":raw_mae,"method_mse":mse,"method_mae":mae,"delta_mse_pct":100*(mse-raw_mse)/raw_mse,"gain_pct":100*(raw_mse-mse)/raw_mse,"mae_gain_pct":100*(raw_mae-mae)/raw_mae,"threshold_json":json.dumps(q_selected),"alpha_json":json.dumps(alpha.tolist()),"stable_gamma_json":json.dumps(stable["gated"].tolist()),"route_json":json.dumps(route_audit),"active":True,"fallback":False,"fallback_type":"none","fallback_reason":"none","selection_protocol":"validation_fit_threshold_tune_route","uses_external_outputs":False,"uses_test_for_selection":False,"test_correction_reads_test_truth":False,"correction_norm_mean":corr_abs/count,"correction_norm_max":float("nan"),"meta_json":json.dumps(meta,sort_keys=True)}]


def paths(root,key):
    safe=key.replace("|","__");return root/"status"/f"{safe}.json",root/"rows"/f"{safe}.csv"


def run_one(payload):
    cfg=nb.BackboneConfig(**payload["config"]);key=core.config_key(cfg);status,row=paths(Path(payload["out_root"]),key)
    if row.exists() and not payload["force"]:return {"key":key,"status":"skipped","failure":None}
    started=time.time()
    try:
        val,test=nb.collect_val_test_arrays_in_memory(cfg);measured=base.metrics_chunked(test["preds"],test["trues"]);expected=payload["raw"]
        if not(np.isclose(measured[0],expected["raw_mse"],atol=1e-6,rtol=1e-6) and np.isclose(measured[1],expected["raw_mae"],atol=1e-6,rtol=1e-6)):raise ValueError(f"Raw mismatch {measured} vs {expected}")
        rows=core.align_raw(evaluate(cfg,val,test),expected,measured);core.atomic_csv(row,pd.DataFrame(rows));core.atomic_json(status,{"status":"completed","elapsed_sec":round(time.time()-started,3)});return {"key":key,"status":"completed","failure":None}
    except Exception as e:
        f={"config_key":key,"status":"failed","type":type(e).__name__,"reason":str(e),"traceback":traceback.format_exc()};core.atomic_json(status,f);return {"key":key,"status":"failed","failure":f}
    finally:gc.collect()


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint-root",required=True);p.add_argument("--raw-metrics-csv",type=Path,required=True);p.add_argument("--out-root",type=Path,default=Path("outputs/dash_shock_guard_round1_full504_20260723"));p.add_argument("--models",nargs="*",default=lm.cf.BACKBONES);p.add_argument("--datasets",nargs="*",default=["ETTh1","ETTh2","ETTm1","ETTm2","Electricity","Traffic","Weather"]);p.add_argument("--pred-lens",nargs="*",type=int,default=[96,192,336,720]);p.add_argument("--seeds",nargs="*",type=int,default=[2,42,2021]);p.add_argument("--workers",type=int,default=10);p.add_argument("--gpu-ids",nargs="*",default=[str(i) for i in range(10)]);p.add_argument("--limit",type=int);p.add_argument("--force",action="store_true");a=p.parse_args()
    raw=core.load_raw(a.raw_metrics_csv);datasets=[core.canonical_dataset(x) for x in a.datasets];cfgs=[nb.BackboneConfig(model=m,dataset=d,pred_len=h,seed=s) for m in a.models for d in datasets for h in a.pred_lens for s in a.seeds];cfgs=cfgs[:a.limit] if a.limit else cfgs
    payload=[{"config":asdict(c),"raw":raw[(c.model,core.display_dataset(c.dataset),c.pred_len,c.seed)],"out_root":str(a.out_root),"force":a.force} for c in cfgs]
    started=time.time();fail=[];core.atomic_json(a.out_root/"runner_status.json",{"status":"running","expected":len(cfgs)})
    with mp.Pool(min(a.workers,len(payload)),initializer=core.worker_init,initargs=(a.checkpoint_root,list(a.gpu_ids))) as pool:
        for i,r in enumerate(pool.imap_unordered(run_one,payload),1):
            if r["failure"]:fail.append(r["failure"])
            core.atomic_json(a.out_root/"runner_status.json",{"status":"running","seen":i,"expected":len(cfgs),"failures":len(fail),"last":r["key"],"elapsed_sec":round(time.time()-started,2)});print(i,len(cfgs),r["key"],r["status"],flush=True)
    files=sorted((a.out_root/"rows").glob("*.csv"));detail=pd.concat([pd.read_csv(x) for x in files],ignore_index=True) if files else pd.DataFrame();core.atomic_csv(a.out_root/"detail.csv",detail)
    if not detail.empty:core.atomic_csv(a.out_root/"summary.csv",core.summarize(detail))
    core.atomic_csv(a.out_root/"failures.csv",pd.DataFrame(fail));done=int(detail.config_key.nunique()) if not detail.empty else 0;core.atomic_json(a.out_root/"runner_status.json",{"status":"completed" if not fail and done==len(cfgs) else "incomplete","configs":done,"expected":len(cfgs),"failures":len(fail),"elapsed_sec":round(time.time()-started,2)})

if __name__=="__main__":main()
