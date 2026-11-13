"""R6 Fig. 5 renders (local, pyvista offscreen). Volumes and masks: export/r6f5/r6f5_<ds>.npz from r6_fig5_export.py
(rule R6_FIG5_RULE.md). Same pipeline as the finalised Fig. 2e / Fig. 3 (r6_render.py): black background, vivid error colours
(figures/palette_vivid_dark.json), depth peeling, SSAO, 2x+ supersampling with Lanczos downsampling, one orthographic
camera and one crop box per volume so production and retrained renders are directly comparable.
Error colouring against GT: rstyle.classify_preds (colouring only; counts returned for the caption, no metric reported)."""
import os

import numpy as np
import pyvista as pv
from PIL import Image

import r6_render as V
import rstyle as R

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "export", "r6f5")
CACHE = os.path.join(HERE, "out", "r6f5_cache")
os.makedirs(CACHE, exist_ok=True)
CLS = ("match", "iou57", "boundary", "split", "merge", "other")
# matched at IoU 0.5 but not at 0.7 (counts for F1@0.5, fails F1@0.7): one extra colour. Hue of #FDB462 (team lead 2026-11-10)
# kept, lightness / saturation set by the same rule as palette_vivid_dark.json (HLS L 0.72, S x 1.4) -> #FFBC70; CVD search over
# NPG hues (scratch cvd_pick.py): minimum CAM02-UCS distance to every other error colour and to black >= 15.1 under normal,
# deuteranomaly, protanomaly, tritanomaly (#FDB462 itself: 13.0). Full check recomputed in fig5_r6.py and written to the provenance.
IOU57 = "#FFBC70"
COL = dict(V.ERRV, iou57=IOU57)


def pred_best_iou(gt, pm):
    m = (gt > 0) & (pm > 0)
    gsz = np.bincount(gt.ravel()); psz = np.bincount(pm.ravel())
    K = int(pm.max()) + 1
    u, c = np.unique(gt[m].astype(np.int64) * K + pm[m], return_counts=True)
    g, p_ = u // K, u % K
    iou = c / (gsz[g] + psz[p_] - c)
    best = np.zeros(K)
    np.maximum.at(best, p_, iou)
    return best


def load(ds):
    Z = np.load(os.path.join(EXP, f"r6f5_{ds}.npz"))
    return {k: Z[k] for k in Z.files}


def gt_crop(Z, margin=6):
    """Crop to the GT-foreground bounding box (+margin voxels in xy, +1 plane in z); same box for all masks."""
    g = Z["gt"] > 0
    zz, yy, xx = [np.nonzero(g.any(axis=a))[0] for a in ((1, 2), (0, 2), (0, 1))]
    sl = (slice(max(zz.min() - 1, 0), zz.max() + 2), slice(max(yy.min() - margin, 0), yy.max() + margin + 1),
          slice(max(xx.min() - margin, 0), xx.max() + margin + 1))
    return {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim == 3 else v) for k, v in Z.items()}, sl


def render_pair(ds, px=2600, kinds=("prod", "ret"), azim=35.0, elev=25.0):
    """{kind: RGBA (transparent black)}, {kind: class counts}. Cached."""
    key = os.path.join(CACHE, f"{ds}_{'-'.join(kinds)}_{px}_v3")
    if all(os.path.exists(f"{key}_{k}.png") for k in kinds) and os.path.exists(f"{key}_counts.npy"):
        return {k: Image.open(f"{key}_{k}.png") for k in kinds}, np.load(f"{key}_counts.npy", allow_pickle=True).item()
    Z, _ = gt_crop(load(ds))
    gt, ratio = Z["gt"], float(Z["ratio"])
    shell = R.shell_surface(gt > 0, ratio, 0.6 * float(Z["d"]))
    ims, counts = {}, {}
    for k in kinds:
        p = V.plotter(px)
        if k == "gt":
            s, bl = R.label_surface(gt, ratio)
            V.mesh(p, s, [pv.Color(V.QUALV[b % len(V.QUALV)]).float_rgb for b in bl])
        else:
            lab = Z[k]
            s, bl = R.label_surface(lab, ratio)
            cls, missed, _ = R.classify_preds(gt, lab)
            pb = pred_best_iou(gt, lab)
            cls = {j: ("iou57" if (c == "match" and pb[j] < 0.7) else c) for j, c in cls.items()}
            kk = np.array([cls.get(int(b), "other") for b in bl])
            for c in CLS:
                sel = np.nonzero(kk == c)[0]
                if len(sel):
                    sub = s.extract_cells(sel).extract_surface(algorithm="dataset_surface")
                    V.mesh(p, sub, np.tile(pv.Color(COL[c]).float_rgb, (sub.n_cells, 1)), V.MATCH_OP if c == "match" else 1.0)
            if missed:
                gs, _ = R.label_surface(gt, ratio, missed)
                p.add_mesh(gs, color=V.ERRV["miss"], opacity=V.GHOST_OP, smooth_shading=True, **V.MAT)
            counts[k] = {c: int(sum(v == c for v in cls.values())) for c in CLS}
            counts[k]["miss"] = len(missed)
        p.add_mesh(shell, color=R.SHELL, opacity=V.SHELL_OP, smooth_shading=True)
        R.ortho_camera(p, azim, elev)
        ims[k] = V.shoot(p)
    a = np.max([np.asarray(i)[..., 3] for i in ims.values()], 0)
    ys, xs = np.nonzero(a > 8); m = int(0.012 * px)
    box = (max(xs.min() - m, 0), max(ys.min() - m, 0), min(xs.max() + m, a.shape[1]), min(ys.max() + m, a.shape[0]))
    ims = {k: i.crop(box) for k, i in ims.items()}
    for k, i in ims.items():
        i.save(f"{key}_{k}.png")
    np.save(f"{key}_counts.npy", counts, allow_pickle=True)
    return ims, counts


def xy_plane(ds, margin=10):
    """Raw plane with the most GT foreground (eval grid, 1-99.8 % LUT applied at export), GT and both masks, cropped like the render."""
    Z, sl = gt_crop(load(ds), margin=margin)
    k = int(np.argmax([(Z["gt"][i] > 0).sum() for i in range(Z["gt"].shape[0])]))
    return dict(img=Z["img"][k].astype(np.float32) / 255.0, gt=Z["gt"][k], ret=Z["ret"][k], prod=Z["prod"][k],
                z=k + sl[0].start, vox=np.asarray(Z["vox"], float), ratio=float(Z["ratio"]))


if __name__ == "__main__":
    for ds in ("bbbc050", "blastospim", "nucverse3d"):
        ims, c = render_pair(ds)
        print(ds, {k: i.size for k, i in ims.items()}, c, flush=True)
