"""Multi-sample grids (gallery T-grid, H-grid, F6-grid, MM-grid; light and dark). Datasets and volumes: MM_RULE.md amendment
(Fig. 2d rule: NIS3D, Tapenade, NISNet3D zebrafish, mouse embryo, BlastoSPIM; median-rule volume each). New palette
(paper/figures/palette.json). Per-volume F1@0.5 = values already in selection.json / mm_log.json (no new numbers).
Within a comparison row all renders share one orthographic camera and one crop box.
Usage: python grids.py [T H F6 MM] [--dark]
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

import figcheck  # noqa: E402
import rstyle as R  # noqa: E402

for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6.5})
HERE = os.path.dirname(os.path.abspath(__file__))
GA = os.path.abspath(os.path.join(HERE, "..", "..", "figures", "gallery", "assets"))
CACHE = os.path.join(HERE, "out", "grid_cache"); os.makedirs(CACHE, exist_ok=True)
SEL = json.load(open(os.path.join(HERE, "..", "figs", "ortho_selection.json")))["volumes"]
MMLOG = json.load(open(os.path.join(HERE, "export", "mm_log.json")))
DSS = ["nis3d", "tapenade", "nisnet_zf", "mouse_embryo", "blastospim"]
NAME = {d: R.DS_NAME[d] for d in DSS}
DIV = mcolors.LinearSegmentedColormap.from_list("npgdiv", R.PALETTE["diverging"])


def add_err(p, gt, lab, ratio, matched_op=0.45, ghosts=True):
    s, bl = R.label_surface(lab, ratio)
    cls, missed, _ = R.classify_preds(gt, lab)
    kk = np.array([cls.get(int(b), "other") for b in bl])
    for k in ("match", "boundary", "split", "merge", "other"):
        sel = np.nonzero(kk == k)[0]
        if len(sel):
            sub = s.extract_cells(sel).extract_surface(algorithm="dataset_surface")
            R.add_rgb_mesh(p, sub, np.tile(pv.Color(R.ERR[k]).float_rgb, (sub.n_cells, 1)), opacity=matched_op if k == "match" else 1.0)
    if ghosts and missed:
        gs, _ = R.label_surface(gt, ratio, missed)
        p.add_mesh(gs, color=R.ERR["miss"], opacity=0.5, smooth_shading=True)


def shot_set(ds, kinds, dark, size=1300):
    """Render several panes of one volume with one camera; return RGBA images cropped to their common bbox."""
    Z = np.load(os.path.join(HERE, "export", f"{ds}.npz"))
    gt, ratio = Z["gt"], float(Z["ratio"])
    key = os.path.join(CACHE, f"{ds}_{'-'.join(kinds)}_{'d' if dark else 'l'}")
    if all(os.path.exists(f"{key}_{k}.png") for k in kinds):
        return [Image.open(f"{key}_{k}.png") for k in kinds]
    shell = R.shell_surface(gt > 0, ratio, 0.6 * float(Z["d"]))
    ims = []
    for k in kinds:
        p = R.plotter((size, size), dark)
        if k == "gt":
            s, bl = R.label_surface(gt, ratio)
            R.add_rgb_mesh(p, s, [pv.Color(R.QUAL[b % len(R.QUAL)]).float_rgb for b in bl])
        elif k == "gain":
            b1, b2 = R.gt_best_iou(gt, Z["S1"]), R.gt_best_iou(gt, Z["2D"])
            s, bl = R.label_surface(gt, ratio)
            v = np.clip((b1 - b2)[bl] / 0.3, -1, 1) * 0.5 + 0.5
            R.add_rgb_mesh(p, s, DIV(v)[:, :3])
        else:
            add_err(p, gt, Z[k], ratio)
        p.add_mesh(shell, color=R.SHELL, opacity=0.12 if dark else 0.16, smooth_shading=True)
        R.ortho_camera(p, 35, 25)
        ims.append(R.shot(p))
    a = np.max([np.asarray(i)[..., 3] for i in ims], 0)
    ys, xs = np.nonzero(a > 8); m = 12
    box = (max(xs.min() - m, 0), max(ys.min() - m, 0), min(xs.max() + m, a.shape[1]), min(ys.max() + m, a.shape[0]))
    ims = [i.crop(box) for i in ims]
    for k, i in zip(kinds, ims):
        i.save(f"{key}_{k}.png")
    return ims


def colors(dark):
    return (R.DARK_BG[0] if dark else "white", "#e8e8ea" if dark else R.FG, "#a0a0a8" if dark else R.MUTED)


def err_legend(fig, y, tc, ncol=6, x=0.5):
    leg = [Patch(fc=R.ERR[k], label=R.ERR_NAME[k]) for k in ("match", "boundary", "split", "merge", "other", "miss")]
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(x, y), ncol=ncol, fontsize=5.6, frameon=False, labelcolor=tc,
               handlelength=1.1, columnspacing=1.2)


def finish(fig, cid, bg, prov):
    fig.savefig(os.path.join(GA, f"{cid}.png"), facecolor=bg)
    fig.savefig(os.path.join(GA, f"{cid}.pdf"), facecolor=bg)
    prov["overlap_issues"] = figcheck.check(fig)
    prov.update(rule="MM_RULE.md amendment 2026-11-10 (Fig. 2d dataset rule; median-rule volumes)", palette="paper/figures/palette.json",
                render=R.RENDER)
    json.dump(prov, open(os.path.join(GA, f"{cid}.provenance.json"), "w"), indent=1, default=str)
    plt.close(fig); print("saved", cid, prov["overlap_issues"])


def grid_T(dark):
    bg, tc, mc = colors(dark)
    cols = [("gt", R.LAB["gt"], tc), ("2D", R.LAB["2D"], R.D2), ("S1", R.LAB["S1"], R.S1)]
    W = 183 / 25.4; L = 0.13; g = 0.004
    tw = (0.995 - L) / 3; tile_px = (1000, 760)
    th_in = tw * W * tile_px[1] / tile_px[0]
    H = len(DSS) * th_in + 0.55 + (len(DSS) - 1) * g * W
    fig = plt.figure(figsize=(W, H), dpi=500, facecolor=bg)
    th = th_in / H; gy = g * W / H; top = 1 - 0.3 / H
    for r, ds in enumerate(DSS):
        ims = shot_set(ds, ["gt", "2D", "S1"], dark)
        for c, ((k, lab, col), im) in enumerate(zip(cols, ims)):
            ax = fig.add_axes([L + c * tw + g / 2, top - (r + 1) * th - r * gy, tw - g, th])
            ax.imshow(R.tile(im, dark, size=tile_px, pad=0.03)); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(k == "S1"); sp.set_color(R.S1); sp.set_linewidth(0.8)
            if r == 0:
                fig.text(L + c * tw + tw / 2, top + 0.006, lab, ha="center", va="bottom", fontsize=7, fontweight="bold", color=col)
            if k != "gt":
                ax.text(0.97, 0.04, f"F1@0.5 {SEL[ds]['f1'][k]:.3f}", transform=ax.transAxes, ha="right", fontsize=5.6,
                        color=col if k == "S1" else mc, fontweight="bold" if k == "S1" else "normal")
        fig.text(L - 0.01, top - (r + 0.5) * th - r * gy, f"{NAME[ds]}\nz/xy {SEL[ds]['ratio']:.2f}", ha="right", va="center", fontsize=6.3, color=tc, linespacing=1.2)
    err_legend(fig, 0.0, tc, x=L + 1.5 * tw)
    finish(fig, "T-grid" + ("-dark" if dark else ""), bg, {"desc": "GT | 2D | S1 over 5 datasets", "datasets": DSS})


def grid_H(dark):
    bg, tc, mc = colors(dark)
    W = 183 / 25.4; n = len(DSS); g = 0.004; tw = (1 - (n + 1) * g) / n; tile_px = (800, 1000)
    th_in = tw * W * tile_px[1] / tile_px[0]; H = th_in + 0.5
    fig = plt.figure(figsize=(W, H), dpi=500, facecolor=bg); th = th_in / H; top = 1 - 0.08 / H
    for c, ds in enumerate(DSS):
        im = shot_set(ds, ["S1"], dark)[0]
        ax = fig.add_axes([g + c * (tw + g), top - th, tw, th]); ax.imshow(R.tile(im, dark, size=tile_px, pad=0.03)); ax.axis("off")
        ax.text(0.05, 0.97, NAME[ds], transform=ax.transAxes, fontsize=6.3, fontweight="bold", color=tc, va="top")
        ax.text(0.05, 0.915, f"z/xy {SEL[ds]['ratio']:.2f}", transform=ax.transAxes, fontsize=5.4, color=mc, va="top")
        ax.text(0.95, 0.03, f"S1  F1@0.5 {SEL[ds]['f1']['S1']:.3f}", transform=ax.transAxes, fontsize=5.4, color=R.S1, ha="right")
    err_legend(fig, 0.0, tc)
    finish(fig, "H-grid" + ("-dark" if dark else ""), bg, {"desc": "S1 hero renders with error colouring, 5 datasets", "datasets": DSS})


def grid_F6(dark):
    bg, tc, mc = colors(dark)
    W = 183 / 25.4; n = len(DSS); g = 0.004; tw = (1 - (n + 1) * g) / n; tile_px = (800, 1000)
    th_in = tw * W * tile_px[1] / tile_px[0]; H = th_in + 0.75
    fig = plt.figure(figsize=(W, H), dpi=500, facecolor=bg); th = th_in / H; top = 1 - 0.08 / H
    prov = {}
    for c, ds in enumerate(DSS):
        im = shot_set(ds, ["gain"], dark)[0]
        ax = fig.add_axes([g + c * (tw + g), top - th, tw, th]); ax.imshow(R.tile(im, dark, size=tile_px, pad=0.03)); ax.axis("off")
        ax.text(0.05, 0.97, NAME[ds], transform=ax.transAxes, fontsize=6.3, fontweight="bold", color=tc, va="top")
        Z = np.load(os.path.join(HERE, "export", f"{ds}.npz")); gt = Z["gt"]
        d = (R.gt_best_iou(gt, Z["S1"]) - R.gt_best_iou(gt, Z["2D"]))[np.unique(gt)[1:]]
        prov[ds] = {"n_nuclei": int(len(d)), "n_S1_better_0.05": int((d > 0.05).sum()), "n_2D_better_0.05": int((d < -0.05).sum())}
    cax = fig.add_axes([0.32, 0.17, 0.36, 0.035])
    cb = plt.colorbar(plt.cm.ScalarMappable(mcolors.Normalize(-0.3, 0.3), DIV), cax=cax, orientation="horizontal")
    cb.set_ticks([-0.3, 0, 0.3]); cb.ax.tick_params(labelsize=5.5, colors=tc, length=1.5, width=0.4); cb.outline.set_linewidth(0.4)
    cb.set_label("IoU(S1) − IoU(2D) per GT nucleus (clipped at ±0.3): blue = 2D better, red = S1 better", fontsize=5.6, color=tc)
    finish(fig, "F6-grid" + ("-dark" if dark else ""), bg, {"desc": "per-nucleus gain maps, 5 datasets", "datasets": DSS,
                                                             "counts_colouring_only": prov})


MMCOLS = [("gt", R.LAB["gt"]), ("base", R.LAB["base"].replace(" ", "\n", 1)), ("stardist3d", R.LAB["stardist3d"]),
          ("useg_2d", "u-Segment3D consensus\nof " + R.LAB["2D"]), ("nnunet", R.LAB["nnunet"]), ("2D", R.LAB["2D"]), ("S1", R.LAB["S1"])]


def mm_tile(ds, k, dark):
    fn = os.path.join(CACHE, f"mm_{ds}_{k}_{'d' if dark else 'l'}.png")
    if os.path.exists(fn):
        return Image.open(fn)
    C = np.load(os.path.join(HERE, "export", f"mmcrop_{ds}.npz"))
    if k not in C.files:
        return None
    gt, ratio = C["gt"], float(C["ratio"])
    p = R.plotter((800, 800), dark)
    if k == "gt":
        s, bl = R.label_surface(gt, ratio); R.add_rgb_mesh(p, s, [pv.Color(R.QUAL[b % len(R.QUAL)]).float_rgb for b in bl])
    else:
        add_err(p, gt, C[k], ratio, matched_op=0.6)
    box = pv.Box(bounds=(0, gt.shape[2], 0, gt.shape[1], 0, gt.shape[0] * ratio))
    p.add_mesh(box.extract_feature_edges(), color="#9a9a96" if not dark else "#5b606b", line_width=1)
    ctr = np.array([gt.shape[2] / 2, gt.shape[1] / 2, gt.shape[0] * ratio / 2]); rad = np.linalg.norm(ctr)
    p.camera_position = [tuple(ctr + rad * np.array([1.4, -1.9, 1.5])), tuple(ctr), (0, 0, 1)]
    p.enable_parallel_projection(); p.reset_camera()
    im = R.shot(p); im.save(fn)
    return im


def grid_MM(dark):
    bg, tc, mc = colors(dark)
    W = 183 / 25.4; L = 0.125; n = len(MMCOLS); g = 0.003; tw = (0.995 - L) / n; tile_px = (800, 800)
    th_in = tw * W; Hrow = th_in + 0.12
    H = len(DSS) * Hrow + 0.78
    fig = plt.figure(figsize=(W, H), dpi=500, facecolor=bg); top = 1 - 0.36 / H
    for r, ds in enumerate(DSS):
        y = top - (r + 1) * Hrow / H + 0.12 / H
        for c, (k, lab) in enumerate(MMCOLS):
            ax = fig.add_axes([L + c * tw + g / 2, y, tw - g, th_in / H]); ax.set_xticks([]); ax.set_yticks([])
            im = mm_tile(ds, k, dark)
            if im is None:
                ax.set_facecolor("#26292f" if dark else "#f7f7f5"); ax.text(0.5, 0.5, "n/a", ha="center", va="center", color=mc, fontsize=5.6, transform=ax.transAxes)
            else:
                ax.imshow(R.tile(R.autocrop(im), dark, size=tile_px, pad=0.04))
            for sp in ax.spines.values():
                sp.set_visible(k == "S1"); sp.set_color(R.S1); sp.set_linewidth(0.75)
            if r == 0:
                fig.text(L + c * tw + tw / 2, top + 0.004, lab, ha="center", va="bottom", linespacing=1.0,
                         fontweight="heavy" if k == "S1" else ("bold" if k == "gt" else "normal"), color=R.S1 if k == "S1" else tc,
                         fontsize=7.0 if k == "S1" else 6.0)
            f1 = None
            if k in ("S1", "2D", "base"):
                f1 = SEL[ds]["f1"][k]
            elif isinstance(MMLOG.get(ds, {}).get("models", {}).get(k), dict):
                f1 = MMLOG[ds]["models"][k]["f1_05"]
            if f1 is not None:
                fig.text(L + c * tw + tw / 2, y - 0.01 / H * 3, f"{f1:.3f}", ha="center", va="top", fontsize=5.4,
                         color=R.S1 if k == "S1" else mc, fontweight="bold" if k == "S1" else "normal")
        fig.text(L - 0.008, y + th_in / H / 2, f"{NAME[ds]}\nz/xy {SEL[ds]['ratio']:.2f}", ha="right", va="center", fontsize=6.0, color=tc, linespacing=1.2)
    err_legend(fig, -0.012, tc, x=L + (0.995 - L) / 2)
    CNT = json.load(open(os.path.join(HERE, "export", "mm_counts.json")))
    finish(fig, "MM-grid" + ("-dark" if dark else ""), bg, {"desc": "multi-model callout grid, 5 datasets", "datasets": DSS,
                                                           "row_counts_check": {ds: {m: ({k: v[k] for k in ("hit", "n_gt", "n_pred", "f1")} if isinstance(v, dict) else v)
                                                                                     for m, v in CNT[ds].items()} for ds in DSS},
                                                           "equal_f1_note": "NIS3D: 2D 0.41406 (tp 318, n_pred 653) vs u-Segment3D 0.41440 (tp 328, n_pred 700) - different, both round to 0.414. BlastoSPIM: original, 2D, u-Segment3D and S1 all tp 59 / n_gt 60 / n_pred 59 -> 0.99160, genuinely equal.",
                                                           "f1_sources": {ds: MMLOG.get(ds) for ds in DSS}})


if __name__ == "__main__":
    dark = "--dark" in sys.argv
    for k in [a for a in sys.argv[1:] if not a.startswith("--")] or ["T", "H", "F6", "MM"]:
        if k in ("T", "H", "F6", "MM"):
            {"T": grid_T, "H": grid_H, "F6": grid_F6, "MM": grid_MM}[k](dark)


def grid_Twide(dark=False, kinds=("gt", "2D", "S1"), tile_px=(800, 900), cid="T-grid-wide", white_page=False, width_mm=183.0):
    """Transposed T grid for Fig. 2d: rows GT / 2D / S1, columns = the five datasets (fills 183 mm width).
    white_page: dark image tiles (if dark) on a white page with dark text (NC: page white, image panels may be dark)."""
    bg, tc, mc = colors(dark and not white_page)
    rows = [r for r in [("gt", R.LAB["gt"], tc), ("2D", R.LAB["2D"], R.D2), ("S1", R.LAB["S1"], R.S1)] if r[0] in kinds]
    W = width_mm / 25.4; L = 0.075 * 183 / width_mm; n = len(DSS); g = 0.003; tw = (0.995 - L) / n
    narrow = width_mm < 170; hx = 0.1 if narrow else 0.0          # narrow: two-line column headers, footnote moved to the caption
    th_in = tw * W * tile_px[1] / tile_px[0]; H = len(rows) * th_in + 0.55 + hx
    fig = plt.figure(figsize=(W, H), dpi=500, facecolor=bg); th = th_in / H; top = 1 - (0.22 + hx) / H
    for c, ds in enumerate(DSS):
        ims = shot_set(ds, ["gt", "2D", "S1"], dark)
        ims = [im for (k_, im) in zip(("gt", "2D", "S1"), ims) if k_ in kinds]
        fig.text(L + c * tw + tw / 2, top + 0.004, f"{NAME[ds]}" + ("\n" if narrow else "  ·  ") + f"z/xy {SEL[ds]['ratio']:.2f}", ha="center", linespacing=1.15, va="bottom", fontsize=6.0, fontweight="bold", color=tc)
        for r, ((k, lab, col), im) in enumerate(zip(rows, ims)):
            ax = fig.add_axes([L + c * tw + g / 2, top - (r + 1) * th, tw - g, th - g * W / H])
            ax.imshow(R.tile(im, dark, size=tile_px, pad=0.03)); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(k == "S1"); sp.set_color(R.S1); sp.set_linewidth(0.8)
            if k != "gt":
                ax.text(0.96, 0.03, f"{SEL[ds]['f1'][k]:.3f}", transform=ax.transAxes, ha="right", fontsize=5.6,
                        color=col if k == "S1" else ("#c8c8cc" if dark else mc), fontweight="bold" if k == "S1" else "normal")
            if c == 0:
                fig.text(L - 0.006, top - (r + 0.5) * th, lab, ha="right", va="center", fontsize=6.3, fontweight="bold", color=col, rotation=90)
    err_legend(fig, 0.0, tc, x=L + (0.995 - L) / 2)
    if not narrow:
        fig.text(0.995, 0.012, "numbers: per-volume F1@0.5", ha="right", fontsize=5.2, color=mc)
    finish(fig, cid if white_page else cid + ("-dark" if dark else ""), bg,
           {"desc": "transposed T grid for Fig. 2d", "datasets": DSS, "tiles": "dark" if dark else "light", "page": bg,
            "volumes": {d: SEL[d]["id"] for d in DSS}, "runs_seed0": "ortho_selection.json runs (S1 07992b55, 2D 81116913)"})


if __name__ == "__main__" and "Twide" in sys.argv:
    grid_Twide("--dark" in sys.argv)
if __name__ == "__main__" and "T2row" in sys.argv:
    grid_Twide("--dark" in sys.argv, kinds=("2D", "S1"), tile_px=(800, 720), cid="T-grid-2row")
if __name__ == "__main__" and "T3row" in sys.argv:   # round 5: Fig. 2d three-row candidates (GT / 2D / S1), white page
    grid_Twide(True, kinds=("gt", "2D", "S1"), tile_px=(800, 624), cid="R5-T3row-darktiles", white_page=True)
    grid_Twide(False, kinds=("gt", "2D", "S1"), tile_px=(800, 624), cid="R5-T3row-lighttiles", white_page=True)
if __name__ == "__main__" and "T2row132" in sys.argv:   # round 5: two-row grid at its native width next to the cube panel (Fig. 2e)
    grid_Twide(False, kinds=("2D", "S1"), tile_px=(800, 720), cid="R5-T2row-132", width_mm=132.0)
