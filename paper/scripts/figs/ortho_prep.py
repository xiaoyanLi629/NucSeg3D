"""Prepare centre-section crops for the orthogonal-view panels (rule: ortho/RULE.md).

Reads ortho/selection.json; for each selected volume loads the image (data/std manifest), the ground truth and the three
models' final masks (threshold t from test_r6.recs; t != -2 recomputed from stored dP/cp exactly as test_r6.hits;
fragments < 0.15 x expected volume removed), and writes ortho/crops/<ds>.npz plus ortho/prep_log.json.
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

O = f"{_NSP_SCRATCH}/nc/figs/ortho"
STD = f"{_NSP_ROOT}/data/std"
S = json.load(open(f"{O}/selection.json"))
MAN = {(r["ds"], r["id"]): r for r in csv.DictReader(open(f"{STD}/manifest.csv"))}
os.makedirs(f"{O}/crops", exist_ok=True)


def best_box(lab, size):
    """Copied from $NUCSEG3D_SCRATCH/orx_wt_t3d/src/eval_cpsam.py (xy window with the most foreground)."""
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
        # same recompute as the paper's error analysis (errors/common.recompute; isotropic interpolation for z/xy >= 2)
        sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/errors'))
        import common
        run = os.path.dirname(os.path.dirname(npz))
        ds_, vid_ = os.path.basename(npz)[:-4].split("__", 1)
        pm = common.recompute(z, common.load_rows(run)[(ds_, vid_)], t)
        assert pm.shape == gt.shape, (pm.shape, gt.shape)
    ev = np.pi / 6 * d ** 3 * vox[1] / vox[0]
    cnt = np.bincount(pm.ravel())
    small = np.nonzero(cnt < frac * ev)[0]
    pm = np.where(np.isin(pm, small[small > 0]), 0, pm)
    return gt, pm, d


only = sys.argv[1:] or sorted(S["volumes"])
log = json.load(open(f"{O}/prep_log.json")) if os.path.exists(f"{O}/prep_log.json") else {}
for ds in only:
    v = S["volumes"][ds]
    m = MAN[(ds, v["id"])]
    img = tifffile.imread(f"{STD}/{m['image']}").astype(np.float32).squeeze()
    labs, info = {}, {}
    for mod in ("S1", "2D", "base"):
        gt, pm, d = final_mask(v["npz"][mod], v["thr"][mod])
        labs[mod] = pm
        n_pred = int((np.unique(pm) > 0).sum())
        info[mod] = dict(t=v["thr"][mod], n_pred=n_pred, n_pred_rec=v["n_pred_rec"][mod], recomputed=v["thr"][mod] != -2.0)
    labs["gt"] = gt
    resized = None
    if img.shape != gt.shape:
        # evaluation used a 384 x 384 xy box with the most foreground (eval_cpsam.best_box, copied below);
        # reproduce it on the standard mask and check that the crop equals the stored ground truth exactly
        lab = tifffile.imread(f"{STD}/{m['mask']}").squeeze().astype(np.int32)
        y0, x0 = best_box(lab, 384)
        assert np.array_equal(lab[:, y0:y0 + 384, x0:x0 + 384], gt), (ds, "box mismatch")
        img = img[:, y0:y0 + 384, x0:x0 + 384]
        resized = {"box_yx": [y0, x0], "from": list(lab.shape)}
    Z, Y, X = gt.shape
    ratio = float(v["ratio"])
    W = int(min(max(Y, X), round(8 * d)))
    hz = int(round(W / ratio / 2))
    # RULE.md addendum 2: section planes and window centre at the rounded centroid of the ground-truth foreground
    cz, cy, cx = (int(round(c)) for c in np.argwhere(gt > 0).mean(0))
    cy = min(max(cy, W // 2), Y - W // 2); cx = min(max(cx, W // 2), X - W // 2)
    sy = slice(max(cy - W // 2, 0), min(cy + W // 2, Y)); sx = slice(max(cx - W // 2, 0), min(cx + W // 2, X))
    sz = slice(max(cz - hz, 0), min(cz + hz, Z))
    out = {"ratio": ratio, "d": d}
    for view, sl in (("xy", (cz, sy, sx)), ("xz", (sz, cy, sx)), ("yz", (sz, sy, cx))):
        im = img[sl]
        lo, hi = np.percentile(im, (1, 99.5))
        out[f"img_{view}"] = np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
        for k, L in labs.items():
            out[f"{k}_{view}"] = L[sl].astype(np.int32)
    np.savez_compressed(f"{O}/crops/{ds}.npz", **out)
    log[ds] = dict(id=v["id"], shape=[Z, Y, X], ratio=ratio, d=d, window_xy=W, z_slices=[sz.start, sz.stop],
                   centre=[cz, cy, cx], models=info, image=m["image"], eval_box=resized)
    print(ds, v["id"], {k: (i["t"], i["n_pred"], i["n_pred_rec"]) for k, i in info.items()}, flush=True)
    json.dump(log, open(f"{O}/prep_log.json", "w"), indent=1)
