"""Draw orthogonal-view grids from ortho/crops/<ds>.npz (prepared by ortho_prep.py).

Each row = one dataset; columns = xy | xz | yz centre sections, each shown for S1, 2D and the original model.
Every tile: image (1-99.5 percentile stretch), ground-truth outlines in magenta, the model's outlines in its colour.
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json

import numpy as np
from matplotlib.colors import to_rgb
from skimage.segmentation import find_boundaries

import nc_style as S

O = f"{_NSP_SCRATCH}/nc/figs/ortho"
MODS = ("S1", "2D", "base")
_E0 = {"S1": "#2E9BFF", "2D": S.C["2D"], "base": "#C8C8C8"}   # brighter variants for visibility on dark images
EDGE = {k: S.C[k] for k in ("S1", "2D", "base")} if S.PALETTE else _E0
GTC = S.PALETTE.get("gt_outline_on_image", "#ffffff") if S.PALETTE else "#FF4FD8"   # ground truth outlines
MNAME = {"S1": "S1", "2D": "2D", "base": "Orig."}


def tile(img, gt, pm, col):
    rgb = np.repeat(img[..., None], 3, axis=2) * 0.85
    bg = find_boundaries(gt, mode="inner")
    bp = find_boundaries(pm, mode="inner")
    rgb[bg] = to_rgb(GTC)
    rgb[bp] = to_rgb(col)
    rgb[bg & bp] = np.array(to_rgb(col)) * 0.5 + np.array(to_rgb(GTC)) * 0.5
    return rgb


def draw(fig, gs, datasets, label_x=None, fs=5.2, show_f1=True):
    sel = json.load(open(f"{O}/selection.json"))["volumes"]
    log = json.load(open(f"{O}/prep_log.json"))
    sub = gs.subgridspec(len(datasets), 11, width_ratios=[1, 1, 1, 0.12, 1, 1, 1, 0.12, 1, 1, 1], wspace=0.04, hspace=0.12)
    axes = []
    for r, ds in enumerate(datasets):
        C = np.load(f"{O}/crops/{ds}.npz")
        ratio = float(C["ratio"])
        for vi, view in enumerate(("xy", "xz", "yz")):
            for mi, m in enumerate(MODS):
                ax = fig.add_subplot(sub[r, vi * 4 + mi])
                rgb = tile(C[f"img_{view}"], C[f"gt_{view}"], C[f"{m}_{view}"], EDGE[m])
                ax.imshow(rgb, interpolation="nearest", aspect=1.0 if view == "xy" else ratio)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_linewidth(0.3); sp.set_color("0.6")
                if r == 0:
                    ax.set_title(MNAME[m], fontsize=fs, color=S.C[m] if m != "base" else "0.35", pad=1.5)
                    if mi == 1:
                        ax.text(0.5, 1.30, view, transform=ax.transAxes,
                                ha="center", va="bottom", fontsize=fs + 0.3, fontweight="bold")
                if vi == 0 and mi == 0:
                    v = sel[ds]
                    txt = f"{S.DS_NAME[ds]}\nz/xy {ratio:.2f}"
                    if show_f1:
                        txt += f"\nF1 {v['f1']['S1']:.2f} / {v['f1']['2D']:.2f} / {v['f1']['base']:.2f}"
                    ax.text(-0.08, 0.5, txt, transform=ax.transAxes, ha="right", va="center", fontsize=fs - 0.2, linespacing=1.15)
                axes.append(ax)
    return axes
