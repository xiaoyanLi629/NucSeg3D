"""Supplementary Data 2: per-volume test results of all models and seeds, both protocols.
Read-only reuse of $NUCSEG3D_SCRATCH/r6/test_r6.recs (rule T for S1 and 2D; fixed threshold -2 for the original)
and the result directories recorded in $NUCSEG3D_SCRATCH/nc/stats/ref/s5_test_{gt,fp}_5seeds_basefixed.json.
nnU-Net / StarDist-3D: baselines/results/eval/{nnunet,stardistB}_test_{gt,fp}/rows.json at the validation-selected setting
(selection_val.json). Self-check: macro F1 recomputed from the CSV must equal the reference JSONs."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import csv, json, sys
import numpy as np
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/five_seed'))
from test_r6 import recs  # noqa: E402

REF = f"{_NSP_SCRATCH}/nc/stats/ref"
BL = f"{_NSP_SHARED}/baselines/results/eval"
OUT = f"{_NSP_SCRATCH}/nc/source_data/supplementary_data_2.csv"
SEEDS = [0, 1000, 2000, 3000, 4000]
SEL = json.load(open(f"{BL}/selection_val.json"))
def one(job):
    proto, model, seed, d, fixed = job
    R = recs(d, fixed=fixed)
    return [dict(protocol="gt" if proto == "gt" else "full_pipeline", dataset=ds, volume_id=vid, model=model, seed=seed,
                 cellprob_threshold=r["t"], min_volume_fraction=0.15, n_gt=r["n_gt"], n_pred=r["n_pred"],
                 tp_iou05=r["tp5"], tp_iou07=r["tp7"],
                 f1_iou05=round(2 * r["tp5"] / max(r["n"], 1), 6), f1_iou07=round(2 * r["tp7"] / max(r["n"], 1), 6))
            for (ds, vid), r in sorted(R.items())]
jobs = []
for proto in ("gt", "fp"):
    D = json.load(open(f"{REF}/s5_test_{proto}_5seeds_basefixed.json"))["dirs"]
    jobs += [(proto, "S1", s, d, False) for s, d in zip(SEEDS, D["S1"])] + [(proto, "2D control", s, d, False) for s, d in zip(SEEDS, D["2D"])] \
        + [(proto, "Original Cellpose-SAM", "", D["base"][0], True)]
from multiprocessing import Pool
with Pool(11) as pool:
    parts = pool.map(one, jobs)
rows = [r for p in parts for r in p]
print("cellpose runs done", len(rows), flush=True)
for proto in ("gt", "fp"):
    for model, tag, sub in (("nnU-Net", "nnU-Net", "nnunet"), ("StarDist-3D", "StarDist", "stardistB")):
        s = SEL[tag]
        thr = float(s["key"].split("_")[0][2:]); frac = s["frac"]
        X = [x for x in json.load(open(f"{BL}/{sub}_test_{proto}/rows.json")) if x["thr"] == thr and x["frac"] == frac]
        for x in sorted(X, key=lambda x: (x["ds"], x["id"])):
            n = x["n_gt"] + x["n_pred"]
            rows.append(dict(protocol="gt" if proto == "gt" else "full_pipeline", dataset=x["ds"], volume_id=x["id"], model=model, seed=0,
                             cellprob_threshold=thr, min_volume_fraction=frac, n_gt=x["n_gt"], n_pred=x["n_pred"],
                             tp_iou05=x["tp5"], tp_iou07=x["tp7"],
                             f1_iou05=round(2 * x["tp5"] / max(n, 1), 6), f1_iou07=round(2 * x["tp7"] / max(n, 1), 6)))
        print(proto, model, len(X), flush=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
print("rows", len(rows))

# self-check against the reference macro values
def macro(proto, model, seed, k, excl_meso=False):
    P = "gt" if proto == "gt" else "full_pipeline"
    sel = [r for r in rows if r["protocol"] == P and r["model"] == model and r["seed"] == seed and not (excl_meso and r["dataset"] == "mesospim")]
    out = []
    for ds in sorted({r["dataset"] for r in sel}):
        v = [r for r in sel if r["dataset"] == ds]
        if proto == "gt":
            out.append(2 * sum(r[f"tp_iou0{k}"] for r in v) / max(sum(r["n_gt"] + r["n_pred"] for r in v), 1))
        else:
            out.append(np.mean([2 * r[f"tp_iou0{k}"] / max(r["n_gt"] + r["n_pred"], 1) for r in v]))
    return float(np.mean(out))
bad = 0
for proto in ("gt", "fp"):
    ref = json.load(open(f"{REF}/s5_test_{proto}_5seeds_basefixed.json"))["macro"]
    cmpj = json.load(open(f"{BL}/compare_test_{proto}.json"))
    for k, lab in ((5, "F1@0.5"), (7, "F1@0.7")):
        e = ref[lab]["13 套"]
        for i, s in enumerate(SEEDS):
            for model, key in (("S1", "S1_seeds"), ("2D control", "two_d_seeds")):
                a, b = macro(proto, model, s, k), e[key][i]
                if abs(a - b) > 1e-9: bad += 1; print("MISMATCH", proto, lab, model, s, a, b)
        a, b = macro(proto, "Original Cellpose-SAM", "", k), e["base"]
        if abs(a - b) > 1e-9: bad += 1; print("MISMATCH base", proto, lab, a, b)
        for model, key in (("nnU-Net", "nnU-Net"), ("StarDist-3D", "StarDist")):
            m13 = cmpj[lab].get("macro13", {})
            if key in m13:
                a = macro(proto, model, 0, k)
                print("check", proto, lab, model, round(a, 6), round(m13[key], 6), "OK" if abs(a - m13[key]) < 1e-6 else "DIFF")
print("SELF-CHECK mismatches:", bad)
