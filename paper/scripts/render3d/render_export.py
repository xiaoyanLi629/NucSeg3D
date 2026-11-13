"""Export showcase volumes for 3D rendering (runs on the server, CPU only, read-only on results).

For each selected test volume (median-difference rule of ortho/RULE.md, seed 0, gt protocol) writes
$NUCSEG3D_SCRATCH/nc/figs/render3d/export/<ds>.npz with: img (eval grid), gt, S1, 2D, base (final masks exactly as in the
evaluation: rule-T threshold from test_r6.recs, recomputed from stored flows if t != -2, fragments < 0.15 x expected
volume removed), vox, ratio, and per-model GT-nucleus error classes from $NUCSEG3D_SCRATCH/nc/errors/common.analyse
(hit/miss/merge/split/boundary, same taxonomy as the paper's error decomposition) plus z-split flags.
No new evaluation numbers are produced; the per-volume F1 printed here is only a consistency check against selection.json.
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import csv
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
SEL = json.load(open(f"{_NSP_SCRATCH}/nc/figs/ortho/selection.json"))
MAN = {(r["ds"], r["id"]): r for r in csv.DictReader(open(f"{STD}/manifest.csv"))}
os.makedirs(O, exist_ok=True)


def best_box(lab, size):
    """Copied from $NUCSEG3D_SCRATCH/orx_wt_t3d/src/eval_cpsam.py."""
    H, W = lab.shape[-2:]
    if H <= size and W <= size:
        return None
    integ = np.pad((lab > 0).sum(0).astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sy, sx = min(size, H), min(size, W)
    v = integ[sy:, sx:] - integ[:-sy, sx:] - integ[sy:, :-sx] + integ[:-sy, :-sx]
    y, x = np.unravel_index(np.argmax(v), v.shape)
    return int(y), int(x)


def final_mask(npz, t, frac=0.15):
    """Identical to the mask construction inside $NUCSEG3D_SCRATCH/r6/test_r6.py hits()."""
    z = np.load(npz)
    gt = z["gt"]; d = float(z["d"]); vox = z["vox"]
    if t == -2.0:
        pm = z["pred"]
    else:
        # same recompute as the paper's error analysis (errors/common.recompute: isotropic interpolation for z/xy >= 2)
        run = os.path.dirname(os.path.dirname(npz))
        ds, vid = os.path.basename(npz)[:-4].split("__", 1)
        pm = common.recompute(z, common.load_rows(run)[(ds, vid)], t)
    ev = np.pi / 6 * d ** 3 * vox[1] / vox[0]
    cnt = np.bincount(pm.ravel())
    small = np.nonzero(cnt < frac * ev)[0]
    pm = np.where(np.isin(pm, small[small > 0]), 0, pm)
    return gt.astype(np.int32), pm.astype(np.int32), d, np.asarray(vox, float)


log = {}
for ds in sys.argv[1:]:
    v = SEL["volumes"][ds]
    m = MAN[(ds, v["id"])]
    img = tifffile.imread(f"{STD}/{m['image']}").astype(np.float32).squeeze()
    out = {}
    for mod in ("S1", "2D", "base"):
        gt, pm, d, vox = final_mask(v["npz"][mod], v["thr"][mod])
        out[mod] = pm
    out["gt"] = gt
    if img.shape != gt.shape:
        lab = tifffile.imread(f"{STD}/{m['mask']}").squeeze().astype(np.int32)
        y0, x0 = best_box(lab, 384)
        assert np.array_equal(lab[:, y0:y0 + 384, x0:x0 + 384], gt)
        img = img[:, y0:y0 + 384, x0:x0 + 384]
    lo, hi = np.percentile(img, (1, 99.8))
    out["img"] = np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    info = {}
    for mod in ("S1", "2D", "base"):
        res, G, P = common.analyse(gt, out[mod], vox)
        out[f"cls_{mod}"] = G["cls"].astype(np.int8)          # index into common.CLS, order of G["id"]
        out[f"zsplit_{mod}"] = G["zsplit"]
        out[f"zmerge_{mod}"] = G["zmerge"]
        f1 = 2 * res["tp5"] / max(res["n_gt"] + res["n_pred"], 1)
        info[mod] = {"f1_05_check": f1, "f1_05_selection": v["f1"][mod], "counts": {k: res[k] for k in common.CLS}, "n_pred": res["n_pred"]}
    out["gt_ids"] = G["id"].astype(np.int32)
    np.savez_compressed(f"{O}/{ds}.npz", vox=vox, ratio=float(v["ratio"]), d=d, **out)
    log[ds] = dict(id=v["id"], shape=list(gt.shape), vox=vox.tolist(), ratio=v["ratio"], d=d, cls_names=list(common.CLS), models=info)
    print(ds, v["id"], {k: (round(i["f1_05_check"], 4), round(i["f1_05_selection"], 4), i["counts"]) for k, i in info.items()}, flush=True)
    old = json.load(open(f"{O}/export_log.json")) if os.path.exists(f"{O}/export_log.json") else {}
    old.update(log)
    json.dump(old, open(f"{O}/export_log.json", "w"), indent=1)
