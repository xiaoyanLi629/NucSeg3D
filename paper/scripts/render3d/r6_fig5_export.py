"""R6 Fig. 5: select one representative test volume per held-out dataset (rule: R6_FIG5_RULE.md) and export it for rendering.
Runs on the server (CPU only, read-only on results). Writes $NUCSEG3D_SCRATCH/nc/figs/render3d/export/r6f5_<ds>.npz
(img uint8 on the eval grid, gt, ret = retrained S1 seed 0 final mask, prod = production S1 seed 0 final mask, vox, ratio, d)
and r6f5_select.json (all per-volume differences, the median, the pick, per-volume F1 from l1s_vol.json).
Also writes r6f5_l1vol.json: per-volume F1@0.5/0.7 of all L1 / production runs (seeds 0, 1000, 2000) for the distribution panel,
recomputed from the same l1s_vol.json counts (no new evaluation)."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import csv
import glob
import json
import os
import sys

import numpy as np
import tifffile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/errors'))
import common  # noqa: E402

O = f"{_NSP_SCRATCH}/nc/figs/render3d/export"
STD = f"{_NSP_ROOT}/data/std"
R = f"{_NSP_ROOT}/results"
V = json.load(open(f"{_NSP_SCRATCH}/nc/heldout/l1cmp/l1s_vol.json"))
MAN = {(r["ds"], r["id"]): r for r in csv.DictReader(open(f"{STD}/manifest.csv"))}
RUN = {"ret": "99107759", "prod": "07992b55"}
KEY = {"ret": "tGT|L1_S1_s0", "prod": "tGT|F_S1_s0"}
SETS = ("bbbc050", "blastospim", "nucverse3d")
f1 = lambda r, k: 2 * r[k] / max(r["n_gt"] + r["n_pred"], 1)  # noqa: E731


def best_box(lab, size):
    """Copied from render_export.py (= eval_cpsam.best_box)."""
    H, W = lab.shape[-2:]
    if H <= size and W <= size:
        return None
    integ = np.pad((lab > 0).sum(0).astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sy, sx = min(size, H), min(size, W)
    v = integ[sy:, sx:] - integ[:-sy, sx:] - integ[sy:, :-sx] + integ[:-sy, :-sx]
    y, x = np.unravel_index(np.argmax(v), v.shape)
    return int(y), int(x)


def final_mask(npz, t, frac=0.15):
    """Identical to render_export.final_mask."""
    z = np.load(npz)
    gt = z["gt"]; d = float(z["d"]); vox = z["vox"]
    if t == -2.0:
        pm = z["pred"]
    else:
        run = os.path.dirname(os.path.dirname(npz))
        ds, vid = os.path.basename(npz)[:-4].split("__", 1)
        pm = common.recompute(z, common.load_rows(run)[(ds, vid)], t)
    ev = np.pi / 6 * d ** 3 * vox[1] / vox[0]
    cnt = np.bincount(pm.ravel())
    small = np.nonzero(cnt < frac * ev)[0]
    pm = np.where(np.isin(pm, small[small > 0]), 0, pm)
    return gt.astype(np.int32), pm.astype(np.int32), d, np.asarray(vox, float)


# ---- per-volume F1 for the distribution panel
dist = {}
for n in ("L1_S1", "L1_2D", "F_S1", "F_2D"):
    for s in (0, 1000, 2000):
        dist[f"{n}_s{s}"] = {k: [f1(v["r2"], "tp5"), f1(v["r2"], "tp7"), v["r2"]["n_gt"]] for k, v in V[f"tGT|{n}_s{s}"].items()}
dist["base"] = {k: [f1(v["fix"], "tp5"), f1(v["fix"], "tp7"), v["fix"]["n_gt"]] for k, v in V["tGT|base"].items()}
json.dump(dist, open(f"{O}/r6f5_l1vol.json", "w"))

# ---- selection (R6_FIG5_RULE.md)
sel = {}
for ds in SETS:
    ks = sorted(k for k in V[KEY["ret"]] if k.startswith(ds + "__") and V[KEY["ret"]][k]["r2"]["n_gt"] >= 10)
    d = {k: f1(V[KEY["ret"]][k]["r2"], "tp7") - f1(V[KEY["prod"]][k]["r2"], "tp7") for k in ks}
    med = float(np.median(list(d.values())))
    pick = min(ks, key=lambda k: (abs(d[k] - med), -V[KEY["ret"]][k]["r2"]["n_gt"], k))
    sel[ds] = dict(n_eligible=len(ks), n_all=sum(k.startswith(ds + "__") for k in V[KEY["ret"]]), median=med, pick=pick, d=d[pick],
                   f1={m: [f1(V[KEY[m]][pick]["r2"], "tp5"), f1(V[KEY[m]][pick]["r2"], "tp7")] for m in KEY},
                   thr={m: V[KEY[m]][pick]["t"] for m in KEY}, n_gt=V[KEY["ret"]][pick]["r2"]["n_gt"], all_d=d)
    print(ds, len(ks), "median", round(med, 4), "pick", pick, round(d[pick], 4), sel[ds]["f1"], flush=True)

# ---- export
for ds in SETS:
    k = sel[ds]["pick"]; vid = k.split("__", 1)[1]
    out = {}
    for m in ("ret", "prod"):
        npz = glob.glob(f"{R}/orx_{RUN[m]}*_thr_eval/masks/{k}.npz")
        assert len(npz) == 1, npz
        gt, pm, d, vox = final_mask(npz[0], float(sel[ds]["thr"][m]))
        out[m] = pm
        res, G, P = common.analyse(gt, pm, vox)
        chk = [2 * res["tp5"] / max(res["n_gt"] + res["n_pred"], 1)]
        sel[ds].setdefault("check_f1_05", {})[m] = chk[0]
    man = MAN[(ds, vid)]
    img = tifffile.imread(f"{STD}/{man['image']}").astype(np.float32).squeeze()
    if img.shape != gt.shape:
        lab = tifffile.imread(f"{STD}/{man['mask']}").squeeze().astype(np.int32)
        y0, x0 = best_box(lab, 384)
        assert np.array_equal(lab[:, y0:y0 + 384, x0:x0 + 384], gt)
        img = img[:, y0:y0 + 384, x0:x0 + 384]
    lo, hi = np.percentile(img, (1, 99.8))
    ratio = float(vox[0] / vox[1])
    np.savez_compressed(f"{O}/r6f5_{ds}.npz", img=np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8),
                        gt=gt, ret=out["ret"], prod=out["prod"], vox=vox, ratio=ratio, d=d)
    sel[ds].update(shape=list(gt.shape), vox=vox.tolist(), ratio=ratio, lut_percentiles=[1, 99.8])
    print(ds, "exported", gt.shape, "check F1@0.5", sel[ds]["check_f1_05"], "stored", {m: sel[ds]["f1"][m][0] for m in KEY}, flush=True)
json.dump(sel, open(f"{O}/r6f5_select.json", "w"), indent=1)
print("R6F5_DONE")
