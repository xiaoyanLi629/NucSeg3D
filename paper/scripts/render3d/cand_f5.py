"""F5 anisotropy-degradation illustration (gallery F5-a light, F5-b dark). Rule and run: F5_RULE.md, f5_infer.py (CPU
re-inference on one 96-px crop of the rule-selected G4 volume; no metrics shown). Rows: 2D control, S1; columns: k = 1, 2, 4, 8
(every k-th plane kept). Each tile: the crop's masks rendered in 3D at physical aspect, coloured by error class against the
subsampled GT (colouring only), plus a central xz strip with GT (magenta) and model outlines."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

import figcheck  # noqa: E402
import rstyle as R  # noqa: E402

for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6.5})
HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(HERE, "export")
GA = os.path.abspath(os.path.join(HERE, "..", "..", "figures", "gallery", "assets"))
KS = (1, 2, 4, 8)
GTC = R.PALETTE["gt_outline_on_image"]


def zcrop(*arrs):
    g = arrs[0]; zz = np.nonzero((g > 0).any((1, 2)))[0]
    sl = slice(max(zz.min() - 1, 0), zz.max() + 2)
    return [a[sl] for a in arrs]


def render(gt, pm, ratio, dark):
    gt, pm = zcrop(gt, pm)
    p = R.plotter((1100, 1100), dark)
    s, bl = R.label_surface(pm, ratio)
    cls, missed, _ = R.classify_preds(gt, pm)
    kk = np.array([cls.get(int(b), "other") for b in bl])
    for k in ("match", "boundary", "split", "merge", "other"):
        sel = np.nonzero(kk == k)[0]
        if len(sel):
            sub = s.extract_cells(sel).extract_surface(algorithm="dataset_surface")
            R.add_rgb_mesh(p, sub, np.tile(pv.Color(R.ERR[k]).float_rgb, (sub.n_cells, 1)), opacity=0.7 if k == "match" else 1.0)
    box = pv.Box(bounds=(0, gt.shape[2], 0, gt.shape[1], 0, gt.shape[0] * ratio))
    p.add_mesh(box.extract_feature_edges(), color="#9a9a96" if not dark else "#5b606b", line_width=1)
    ctr = np.array([gt.shape[2] / 2, gt.shape[1] / 2, gt.shape[0] * ratio / 2]); rad = np.linalg.norm(ctr)
    p.camera_position = [tuple(ctr + rad * np.array([1.5, -2.0, 1.2])), tuple(ctr), (0, 0, 1)]
    p.enable_parallel_projection(); p.reset_camera()
    return R.autocrop(R.shot(p))


def xz(img, gt, pm, col, ratio):
    gt, pm, img = zcrop(gt, pm, img)
    y = gt.shape[1] // 2
    up = lambda a: np.repeat(np.repeat(a, max(int(round(4 * ratio)), 1), 0), 4, 1)  # noqa: E731  nearest upsampling, physical aspect
    im = up(img[:, y].astype(float)); g2, p2 = up(gt[:, y]), up(pm[:, y])
    lo, hi = np.percentile(im, (1, 99.8))
    rgb = np.repeat(np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)[..., None], 3, -1)
    rgb[find_boundaries(g2, mode="inner")] = to_rgb(GTC)
    rgb[find_boundaries(p2, mode="inner")] = to_rgb(col)
    return rgb


def main(dark, cid):
    bg = R.DARK_BG[0] if dark else "white"; tc = "#e8e8ea" if dark else R.FG; mc = "#a0a0a8" if dark else R.MUTED
    fig = plt.figure(figsize=(183 / 25.4, 100 / 25.4), dpi=600, facecolor=bg)
    L, tw = 0.10, 0.895 / 4
    for r, (m, lab, col) in enumerate((("2D", "2D control", R.D2), ("S1", "S1", R.S1))):
        for j, k in enumerate(KS):
            Z = np.load(os.path.join(E, f"f5_{m}_k{k}.npz"))
            gt, pm, img, ratio = Z["gt"], Z["pred"], Z["img"], float(Z["ratio"])
            y0 = 0.53 - r * 0.47
            a = fig.add_axes([L + j * tw + 0.004, y0 + 0.13, tw - 0.008, 0.30])
            a.imshow(R.tile(R.autocrop(render(gt, pm, ratio, dark)), dark, size=(1000, 760), pad=0.03)); a.axis("off")
            b = fig.add_axes([L + j * tw + 0.004, y0, tw - 0.008, 0.12])
            b.imshow(xz(img, gt, pm, col, ratio), interpolation="nearest"); b.set_xticks([]); b.set_yticks([])
            for sp in b.spines.values():
                sp.set_color(col); sp.set_linewidth(0.6)
            if r == 0:
                fig.text(L + j * tw + tw / 2, 0.975, f"k = {k}" + ("  (all planes)" if k == 1 else ("  (every 2nd plane)" if k == 2 else f"  (every {k}th plane)")),
                         ha="center", va="top", fontsize=6.5, fontweight="bold", color=tc)
        fig.text(0.01, 0.53 - r * 0.47 + 0.2, lab, fontsize=7, fontweight="bold", color=col, va="center")
    leg = [Patch(fc=R.ERR[k], label=R.ERR_NAME[k]) for k in ("match", "boundary", "split", "merge", "other")] + \
          [Patch(fc="white", ec="#1d1d1f", lw=0.4, label="GT outline (xz, white)")]
    fig.legend(handles=leg, loc="lower center", ncol=6, fontsize=5.5, frameon=False, labelcolor=tc, bbox_to_anchor=(0.55, -0.01))
    fn = os.path.join(GA, f"{cid}.png"); fig.savefig(fn, facecolor=bg); fig.savefig(fn.replace(".png", ".pdf"), facecolor=bg)
    rep = figcheck.check(fig)
    log = json.load(open(os.path.join(E, "f5_log.json")))
    json.dump({"rule": open(os.path.join(HERE, "F5_RULE.md")).read(), "run_log": log, "overlap_issues": rep,
               "note": "illustration only; masks re-inferred on the crop, so they need not equal the G4 full-volume masks; no numbers shown"},
              open(os.path.join(GA, f"{cid}.provenance.json"), "w"), indent=1, default=str)
    plt.close(fig)
    print("saved", cid, rep)


if __name__ == "__main__":
    main(False, "F5-a")
    main(True, "F5-b")
