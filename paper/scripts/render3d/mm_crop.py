"""Server side of the multi-model strip (CPU): apply the MM_RULE.md callout rule on the full evaluated masks and save only the
callout crops of every model (export/mmcrop_<ds>.npz), so that the small files can be copied over a slow link."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import sys

import numpy as np

E = f"{_NSP_SCRATCH}/nc/figs/render3d/export"


def gt_classes(gt, pm):
    """GT-side class per nucleus (hit / miss / merge / split / boundary), as in the paper's taxonomy (simplified for selection)."""
    from collections import defaultdict
    m = (gt > 0) & (pm > 0)
    gsz = np.bincount(gt.ravel()); psz = np.bincount(pm.ravel())
    K = int(pm.max()) + 1
    u, c = np.unique(gt[m].astype(np.int64) * K + pm[m], return_counts=True)
    g, p_ = u // K, u % K
    iou = c / (gsz[g] + psz[p_] - c); cov = c / gsz[g]
    best = defaultdict(float); bcov = defaultdict(float); parts = defaultdict(list); col = defaultdict(list)
    for gi, pj, v, cv in zip(g, p_, iou, cov):
        best[int(gi)] = max(best[int(gi)], v); bcov[int(gi)] = max(bcov[int(gi)], cv)
        if cv >= 0.15:
            parts[int(gi)].append(int(pj))
        col[int(pj)].append((int(gi), cv))
    out = {}
    for gi in np.unique(gt)[1:]:
        gi = int(gi)
        if best[gi] >= 0.5:
            out[gi] = "hit"
        elif best[gi] < 0.1 and bcov[gi] < 0.15:
            out[gi] = "miss"
        elif any(sum(1 for (h, cv) in col[pj] if cv > 0.5) >= 2 for pj in parts[gi]):
            out[gi] = "merge"
        elif len(parts[gi]) >= 2:
            out[gi] = "split"
        else:
            out[gi] = "boundary"
    return out



for ds in sys.argv[1:]:
    Z = np.load(f"{E}/{ds}.npz"); M = np.load(f"{E}/mm_{ds}.npz")
    gt, ratio, d = Z["gt"], float(Z["ratio"]), float(Z["d"])
    labs = {"gt": gt, "base": Z["base"], "2D": Z["2D"], "S1": Z["S1"]}
    for k in ("nnunet", "stardist3d", "useg_2d"):
        if k in M.files:
            labs[k] = M[k]
    cl = {k: gt_classes(gt, v) for k, v in labs.items() if k != "gt"}
    sizes = np.bincount(gt.ravel())
    ids = np.unique(gt)[1:]
    score = [(len({cl[k][int(i)] for k in cl}), int(sizes[i]), int(i)) for i in ids]
    nmax, _, tgt = max(score)
    zz, yy, xx = np.nonzero(gt == tgt)
    cz, cy, cx = int(zz.mean()), int(yy.mean()), int(xx.mean())
    w = int(round(3 * d)); hz = int(round(1.5 * d / ratio))
    sl = (slice(max(cz - hz, 0), cz + hz + 1), slice(max(cy - w // 2, 0), cy + w // 2), slice(max(cx - w // 2, 0), cx + w // 2))
    np.savez_compressed(f"{E}/mmcrop_{ds}.npz", origin=np.array([s.start for s in sl]), target=tgt, n_classes=nmax, ratio=ratio, d=d,
                        img=Z["img"][sl], **{k: v[sl] for k, v in labs.items()},
                        target_classes=json.dumps({k: cl[k][tgt] for k in cl}))
    print(ds, "target", tgt, "classes", {k: cl[k][tgt] for k in cl}, "crop", [s.start for s in sl], labs["gt"][sl].shape, flush=True)
