from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_cf_long_memory_anchor as lm
import run_cf_residual_expert as core
import run_internal_offline_candidates as base
import run_e06_new_backbones as nb

POLICIES = ["final_fixed25", "final_val_tuned", "final_route_safe", "final_disagreement_shrink"]
RHO_GRID = np.asarray([0.0, .25, .5, .75, 1.0], dtype=np.float64)


def prepare(params: dict[str, Any], val: dict[str, np.ndarray], bins: np.ndarray) -> dict[str, Any]:
    n = len(val["preds"]); c1, c2 = max(1, n // 3), min(max(n // 3 + 1, 2 * n // 3), n - 1)
    timeline, current = lm.observed_timeline(val["inputs"])
    weights, meta = lm.fit_weights(timeline, current, val["inputs"], val["preds"], val["trues"], c1, bins, params)
    clip = lm.fit_clip(timeline, current, val["inputs"], val["preds"], c1, bins, params, weights)
    gamma = lm.tune_gamma(timeline, current, val["inputs"], val["preds"], val["trues"], np.arange(c1, c2), bins, params, weights, clip)
    route = lm.evaluate_indices(timeline, current, val["inputs"], val["preds"], val["trues"], np.arange(c2, n), bins, params, weights, gamma, clip, collect_origins=True)
    gated, gains, positives, worst = lm.gate_on_route(route, gamma)
    return {"params": params, "timeline": timeline, "current": current, "weights": weights, "clip": clip,
            "gamma": gamma, "gated": gated, "route_gain": gains, "positive": positives, "worst": worst,
            "c1": c1, "c2": c2, **meta}


def corr(state: dict[str, Any], inputs: np.ndarray, preds: np.ndarray, indices: np.ndarray,
         bins: np.ndarray, safe: bool = False) -> np.ndarray:
    value = lm.correction_chunk(state["timeline"], state["current"], inputs, preds, indices, bins,
                                state["params"], state["weights"])
    value = np.clip(value, -state["clip"], state["clip"])
    gamma = state["gated"] if safe else state["gamma"]
    return value * gamma[bins][None, :, None]


def dynamic_clip(l6: dict[str, Any], l7: dict[str, Any], val: dict[str, np.ndarray], bins: np.ndarray) -> float:
    idx = lm.sampled_indices(0, l6["c1"], 64)
    d = corr(l7, val["inputs"], val["preds"], idx, bins) - corr(l6, val["inputs"], val["preds"], idx, bins)
    return max(float(np.quantile(np.abs(d), .90)), 1e-8)


def reliability(stable_corr: np.ndarray, dynamic: np.ndarray, bins: np.ndarray) -> np.ndarray:
    out = np.ones((len(stable_corr), 4), dtype=np.float32)
    for b in range(4):
        h = np.where(bins == b)[0]
        s = np.sqrt(np.mean(np.square(stable_corr[:, h], dtype=np.float64), axis=(1, 2)))
        d = np.sqrt(np.mean(np.square(dynamic[:, h], dtype=np.float64), axis=(1, 2)))
        out[:, b] = np.clip(s / (s + d + 1e-8), 0, 1)
    return out


def tune_rho(l6: dict[str, Any], l7: dict[str, Any], val: dict[str, np.ndarray], bins: np.ndarray,
             dclip: float, shrink: bool) -> np.ndarray:
    sums = np.zeros((4, len(RHO_GRID)), dtype=np.float64)
    idx = np.arange(l6["c1"], l6["c2"]); step = lm.chunk_size(val["preds"])
    for start in range(0, len(idx), step):
        ii = idx[start:start+step]
        c6 = corr(l6, val["inputs"], val["preds"], ii, bins)
        d = np.clip(corr(l7, val["inputs"], val["preds"], ii, bins) - c6, -dclip, dclip)
        rel = reliability(c6, d, bins) if shrink else np.ones((len(ii), 4), dtype=np.float32)
        for b in range(4):
            h = np.where(bins == b)[0]; e = val["preds"][ii][:, h] + c6[:, h] - val["trues"][ii][:, h]
            dd = d[:, h] * rel[:, b, None, None]
            for j, r in enumerate(RHO_GRID): sums[b, j] += np.sum(np.square(e + r * dd, dtype=np.float64))
    return np.asarray([RHO_GRID[np.argmin(sums[b])] for b in range(4)], dtype=np.float32)


def route_rho(l6: dict[str, Any], l7: dict[str, Any], val: dict[str, np.ndarray], bins: np.ndarray,
              dclip: float, rho: np.ndarray, shrink: bool) -> tuple[np.ndarray, dict[str, Any]]:
    idx = np.arange(l6["c2"], len(val["preds"])); raw_o = np.zeros((len(idx), 4)); met_o = np.zeros_like(raw_o)
    step = lm.chunk_size(val["preds"])
    for start in range(0, len(idx), step):
        ii=idx[start:start+step]; c6=corr(l6,val["inputs"],val["preds"],ii,bins)
        d=np.clip(corr(l7,val["inputs"],val["preds"],ii,bins)-c6,-dclip,dclip)
        rel=reliability(c6,d,bins) if shrink else np.ones((len(ii),4),dtype=np.float32)
        for b in range(4):
            h=np.where(bins==b)[0]; e=val["preds"][ii][:,h]+c6[:,h]-val["trues"][ii][:,h]
            em=e+rho[b]*rel[:,b,None,None]*d[:,h]
            raw_o[start:start+len(ii),b]=np.mean(np.square(e,dtype=np.float64),axis=(1,2))
            met_o[start:start+len(ii),b]=np.mean(np.square(em,dtype=np.float64),axis=(1,2))
    gated=rho.copy(); gains=[]; pos=[]; worst=[]
    for b in range(4):
        blocks=[]
        for jj in np.array_split(np.arange(len(idx)),min(4,len(idx))):
            a=float(np.mean(raw_o[jj,b])); m=float(np.mean(met_o[jj,b])); blocks.append(100*(a-m)/max(a,1e-12))
        a=float(np.mean(raw_o[:,b]));m=float(np.mean(met_o[:,b]));g=100*(a-m)/max(a,1e-12)
        p=sum(x>0 for x in blocks);w=min(blocks)
        if g<=0 or p<3 or w<-.10:gated[b]=0
        gains.append(g);pos.append(p);worst.append(w)
    return gated,{"gain":gains,"positive":pos,"worst":worst}


def evaluate(config: nb.BackboneConfig, val0: dict[str,np.ndarray], test0: dict[str,np.ndarray]) -> list[dict[str,Any]]:
    val={k:np.asarray(val0[k],dtype=np.float32) for k in ("preds","trues","inputs")};test={k:np.asarray(test0[k],dtype=np.float32) for k in ("preds","trues","inputs")}
    bins=lm.cf.horizon_bin_ids(config.pred_len,4); l6=prepare(lm.VARIANT_BY_ID["lm06_median3"],val,bins);l7=prepare(lm.VARIANT_BY_ID["lm07_ew3"],val,bins)
    dclip=dynamic_clip(l6,l7,val,bins); tuned=tune_rho(l6,l7,val,bins,dclip,False); tuned_shrink=tune_rho(l6,l7,val,bins,dclip,True)
    route,ra=route_rho(l6,l7,val,bins,dclip,tuned,False);route_shrink,ras=route_rho(l6,l7,val,bins,dclip,tuned_shrink,True)
    rhos={"final_fixed25":np.full(4,.25,dtype=np.float32),"final_val_tuned":tuned,"final_route_safe":route,"final_disagreement_shrink":route_shrink}
    test_timeline,test_current,stitch=lm.test_timeline(val["inputs"],val["trues"],test["inputs"])
    for state in (l6,l7):state["timeline"],state["current"]=test_timeline,test_current
    totals={p:{"sq":0.,"ab":0.,"corr":0.,"count":0} for p in POLICIES}; raw_sq=raw_ab=0.; count=0
    idx=np.arange(len(test["preds"]));step=lm.chunk_size(test["preds"])
    for start in range(0,len(idx),step):
        ii=idx[start:start+step]; c6e=corr(l6,test["inputs"],test["preds"],ii,bins);c6s=corr(l6,test["inputs"],test["preds"],ii,bins,True)
        dyn=np.clip(corr(l7,test["inputs"],test["preds"],ii,bins)-c6e,-dclip,dclip); rel=reliability(c6e,dyn,bins)
        rawerr=test["preds"][ii]-test["trues"][ii];raw_sq+=np.sum(np.square(rawerr,dtype=np.float64));raw_ab+=np.sum(np.abs(rawerr),dtype=np.float64);count+=rawerr.size
        stable_active=(l6["gated"]>0)[bins][None,:,None]
        for p in POLICIES:
            rr=rel if p=="final_disagreement_shrink" else np.ones((len(ii),4),dtype=np.float32)
            add=rhos[p][bins][None,:,None]*rr[:,bins,None]*dyn*stable_active
            cc=c6s+add;err=rawerr+cc;t=totals[p];t["sq"]+=np.sum(np.square(err,dtype=np.float64));t["ab"]+=np.sum(np.abs(err),dtype=np.float64);t["corr"]+=np.sum(np.abs(cc),dtype=np.float64);t["count"]+=cc.size
    raw_mse=raw_sq/count;raw_mae=raw_ab/count;rows=[]
    for p in POLICIES:
        t=totals[p];mse=t["sq"]/count;mae=t["ab"]/count
        active=bool(np.any(l6["gated"]>0) or np.any(rhos[p]>0))
        rows.append({"dataset":core.display_dataset(config.dataset),"backbone":config.model,"model":config.model,"pred_len":config.pred_len,"seed":config.seed,"config_key":core.config_key(config),"method":p,"raw_mse":raw_mse,"raw_mae":raw_mae,"method_mse":mse,"method_mae":mae,"delta_mse_pct":100*(mse-raw_mse)/raw_mse,"gain_pct":100*(raw_mse-mse)/raw_mse,"mae_gain_pct":100*(raw_mae-mae)/raw_mae,"dynamic_clip":dclip,"rho_json":json.dumps(rhos[p].tolist()),"stable_gamma_json":json.dumps(l6["gated"].tolist()),"dynamic_route_json":json.dumps(ras if p=="final_disagreement_shrink" else ra),"active":active,"fallback":not active,"fallback_type":"algorithmic" if not active else "none","fallback_reason":"stable_and_dynamic_rejected" if not active else "none","selection_protocol":"validation_fit_tune_incremental_route","uses_external_outputs":False,"uses_test_for_selection":False,"test_correction_reads_test_truth":False,"correction_norm_mean":t["corr"]/t["count"],"correction_norm_max":float("nan"),"meta_json":json.dumps(stitch,sort_keys=True)})
    return rows


def paths(root:Path,key:str):
    s=key.replace("|","__");return root/"status"/f"{s}.json",root/"rows"/f"{s}.csv"


def run_one(payload:dict[str,Any])->dict[str,Any]:
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
    p=argparse.ArgumentParser();p.add_argument("--checkpoint-root",required=True);p.add_argument("--raw-metrics-csv",type=Path,required=True);p.add_argument("--out-root",type=Path,default=Path("outputs/final_full504_20260722"));p.add_argument("--models",nargs="*",default=lm.cf.BACKBONES);p.add_argument("--datasets",nargs="*",default=["ETTh1","ETTh2","ETTm1","ETTm2","Electricity","Traffic","Weather"]);p.add_argument("--pred-lens",nargs="*",type=int,default=[96,192,336,720]);p.add_argument("--seeds",nargs="*",type=int,default=[2,42,2021]);p.add_argument("--workers",type=int,default=10);p.add_argument("--gpu-ids",nargs="*",default=[str(i) for i in range(10)]);p.add_argument("--limit",type=int);p.add_argument("--force",action="store_true");a=p.parse_args()
    raw=core.load_raw(a.raw_metrics_csv);datasets=[core.canonical_dataset(x) for x in a.datasets];cfgs=[nb.BackboneConfig(model=m,dataset=d,pred_len=h,seed=s) for m in a.models for d in datasets for h in a.pred_lens for s in a.seeds];cfgs=cfgs[:a.limit] if a.limit else cfgs
    payload=[]
    for c in cfgs:
        k=(c.model,core.display_dataset(c.dataset),c.pred_len,c.seed);payload.append({"config":asdict(c),"raw":raw[k],"out_root":str(a.out_root),"force":a.force})
    started=time.time();fail=[];core.atomic_json(a.out_root/"runner_status.json",{"status":"running","expected":len(cfgs)})
    with mp.Pool(min(a.workers,len(payload)),initializer=core.worker_init,initargs=(a.checkpoint_root,list(a.gpu_ids))) as pool:
        for i,r in enumerate(pool.imap_unordered(run_one,payload),1):
            if r["failure"]:fail.append(r["failure"])
            core.atomic_json(a.out_root/"runner_status.json",{"status":"running","seen":i,"expected":len(cfgs),"failures":len(fail),"last":r["key"],"elapsed_sec":round(time.time()-started,2)});print(i,len(cfgs),r["key"],r["status"],flush=True)
    files=sorted((a.out_root/"rows").glob("*.csv"));detail=pd.concat([pd.read_csv(x) for x in files],ignore_index=True) if files else pd.DataFrame();core.atomic_csv(a.out_root/"detail.csv",detail)
    if not detail.empty:core.atomic_csv(a.out_root/"summary.csv",core.summarize(detail))
    core.atomic_csv(a.out_root/"failures.csv",pd.DataFrame(fail));done=int(detail.config_key.nunique()) if not detail.empty else 0;core.atomic_json(a.out_root/"runner_status.json",{"status":"completed" if not fail and done==len(cfgs) else "incomplete","configs":done,"expected":len(cfgs),"failures":len(fail),"elapsed_sec":round(time.time()-started,2)})

if __name__=="__main__":main()
