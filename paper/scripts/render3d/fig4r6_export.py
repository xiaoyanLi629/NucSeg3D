"""R6 Fig. 4 export (rule FIG4R6_RULE.md). Run on hostA (CPU only, a few minutes):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 19 $NUCSEG3D_SHARED/env_torch/bin/python fig4r6_export.py
Writes $NUCSEG3D_SCRATCH/nc/figs/fig4r6/{fig4r6_degrade.npz, fig4r6_perds.json, fig4r6_export_log.json}.
(1) degradation grid: gx_eval.degrade (imported) on the full evaluated volume, then the F5 crop; (2) per-dataset pooled F1 of every
G4 model instance with g4_summary.py's pick() logic, and a check that the 6-dataset macro reproduces g4_summary.json."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import glob
import hashlib
import json
import os
import sys

import numpy as np

SRC = _nsp_os.path.join(_NSP_REPO, 'src')
sys.path.insert(0, SRC)
sys.path.insert(1, _nsp_os.path.join(_NSP_REPO, 'src'))
from eval_cpsam import load_case  # noqa: E402
from stddata import load_manifest  # noqa: E402
import gx_eval  # noqa: E402  (degrade only)

O = f"{_NSP_SCRATCH}/nc/figs/fig4r6"
os.makedirs(O, exist_ok=True)
KEY = ("celegans", "pha4I2L_0408072")
F5LOG = f"{_NSP_SCRATCH}/nc/figs/render3d/export/f5_log.json"
KS, CS = (1, 2, 4, 8), ("0", "0.25", "0.5")
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()  # noqa: E731

# ---- (1) degradation grid
f5 = json.load(open(F5LOG)) if os.path.exists(F5LOG) else None
y0, x0, H = (f5["crop_yx"] if f5 else [31, 150, 96])
r0 = [x for x in load_manifest(split="test", dim="3d") if (x["ds"], x["id"]) == KEY][0]
img0, gt0, box, ign0 = load_case(r0, 384)
out, sig = {}, {}
for k in KS:
    for c in CS:
        r = dict(r0)
        img, gt, ign = gx_eval.degrade(img0.copy(), gt0.copy(), None if ign0 is None else ign0.copy(), r, k, float(c))
        out[f"img_k{k}_c{c}"] = np.asarray(img, np.float32)[:, y0:y0 + H, x0:x0 + H]
        out[f"gt_k{k}"] = np.asarray(gt, np.int32)[:, y0:y0 + H, x0:x0 + H]
        sig[f"k{k}_c{c}"] = dict(sigma=r.get("gx_noise_sigma"), contrast=r.get("gx_contrast"), vox=list(r["voxel"]))
np.savez_compressed(f"{O}/fig4r6_degrade.npz", **out, vox0=np.array(r0["voxel"], float), crop=np.array([y0, x0, H]))
# sanity: k-subsampled crop GT equals the F5 export GT (same crop)
f5chk = {}
for k in KS:
    p = f"{_NSP_SCRATCH}/nc/figs/render3d/export/f5_S1_k{k}.npz"
    if os.path.exists(p):
        f5chk[f"k{k}"] = bool(np.array_equal(np.load(p)["gt"], out[f"gt_k{k}"]))

# ---- (2) per-dataset values (g4_summary.py pick logic)
R = f"{_NSP_ROOT}/results/"
S = json.load(open(f"{_NSP_SCRATCH}/r6/sched_state.json"))
KEYS = [tuple(l.strip().split("__", 1)) for l in open(f"{_NSP_SCRATCH}/nc/gpu/lists/g4_keys.txt") if l.strip()]
DS = sorted({k[0] for k in KEYS})
INST = [("S1", s) for s in (0, 1000, 2000)] + [("2D", s) for s in (0, 1000, 2000)] + [("base", None)]
G = json.load(open(f"{_NSP_SCRATCH}/nc/gpu/g4/g4_summary.json"))


def rdir(key):
    return glob.glob(R + "orx_" + S[key]["run"] + "*")[0]


def pick(d, ruleT):
    rows = {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in json.load(open(d + "/rows.json"))}
    t = {(r["ds"], r["id"]): r["t"] for r in json.load(open(d + "/ruleT.json"))} if ruleT else {}
    o = {}
    for kk in KEYS:
        x = rows[(*kk, t.get(kk, -2.0) if ruleT else -2.0, 0.15)]
        o[kk] = (x["tp5"], x["tp7"], x["n_gt"] + x["n_pred"])
    return o


def per_ds(v):
    res = {}
    for ds in DS:
        L = [v[kk] for kk in KEYS if kk[0] == ds]; den = max(sum(x[2] for x in L), 1)
        res[ds] = (2 * sum(x[0] for x in L) / den, 2 * sum(x[1] for x in L) / den)
    return res


perds, worst = {}, 0.0
for k in KS:
    for c in CS:
        cell = {}
        for m, s in INST:
            key = f"G4_{m}_s{s}_z{k}" if s is not None else f"G4_base_z{k}"
            cell[f"{m}_{s}"] = per_ds(pick(f"{rdir(key)}/noise_{c}", ruleT=m != "base"))
        ent = {"n_vol": {ds: sum(kk[0] == ds for kk in KEYS) for ds in DS}}
        for i, met in enumerate(("f05", "f07")):
            for ds in DS:
                dd = [cell[f"S1_{s}"][ds][i] - cell[f"2D_{s}"][ds][i] for s in (0, 1000, 2000)]
                ent.setdefault(met, {})[ds] = dict(S1=float(np.mean([cell[f"S1_{s}"][ds][i] for s in (0, 1000, 2000)])),
                                                   D2=float(np.mean([cell[f"2D_{s}"][ds][i] for s in (0, 1000, 2000)])),
                                                   base=float(cell["base_None"][ds][i]),
                                                   diff=[float(np.mean(dd)), float(min(dd)), float(max(dd))], diff_seeds=[float(x) for x in dd])
            ref = G["conditions"][f"z{k}_n{c}"][f"all_{met}"]
            mS1 = np.mean([ent[met][ds]["S1"] for ds in DS]); mD2 = np.mean([ent[met][ds]["D2"] for ds in DS])
            mb = np.mean([ent[met][ds]["base"] for ds in DS])
            worst = max(worst, abs(mS1 - ref["S1"][0]), abs(mD2 - ref["D2"][0]), abs(mb - ref["base"]),
                        abs(np.mean([ent[met][ds]["diff"][0] for ds in DS]) - ref["diff"][0]))
        perds[f"z{k}_n{c}"] = ent
json.dump(dict(datasets=DS, conditions=perds, macro_check_max_abs_diff=float(worst)), open(f"{O}/fig4r6_perds.json", "w"), indent=1)
log = dict(key=KEY, crop_yx_H=[y0, x0, H], f5_log_used=bool(f5), noise=sig, f5_gt_equal=f5chk, macro_check_max_abs_diff=float(worst),
           md5={"g4_summary.json": md5(f"{_NSP_SCRATCH}/nc/gpu/g4/g4_summary.json"), "gx_eval.py": md5(f"{_NSP_SCRATCH}/nc/gpu/gx_eval.py"),
                "g4_keys.txt": md5(f"{_NSP_SCRATCH}/nc/gpu/lists/g4_keys.txt"), "fig4r6_degrade.npz": md5(f"{O}/fig4r6_degrade.npz"),
                "fig4r6_perds.json": md5(f"{O}/fig4r6_perds.json")})
json.dump(log, open(f"{O}/fig4r6_export_log.json", "w"), indent=1, default=str)
print(json.dumps(log, indent=1, default=str))
