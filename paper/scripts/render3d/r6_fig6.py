"""R6 Fig. 6: multi-model 3D comparison, two versions (A: Fig. 6 datasets; B: datasets by the g2ds rule).
Rules: R6_FIG6_RULE.md (written before rendering). Data: export/r6f6/r6f6_<ds>.npz and r6f6_log.json (server script
r6f6_server_export.py). Render style copied from r6_render.py (nc-figures; that file is not modified): black tiles, vivid
palette (figures/palette_vivid_dark.json), 2x supersampling + Lanczos, depth peeling, tiles embedded unresampled at 1000 ppi.
Usage: python r6_fig6.py A|B [--draft]
"""
import hashlib
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

import figcheck  # noqa: E402
import rstyle as R  # noqa: E402

pv.OFF_SCREEN = True
for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6.0, "pdf.fonttype": 42})
HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.abspath(os.path.join(HERE, "..", "..", "figures"))
GA = os.path.join(FIGD, "gallery", "assets")
EXP = os.path.join(HERE, "export", "r6f6")
CACHE = os.path.join(HERE, "out", "r6f6_cache"); os.makedirs(CACHE, exist_ok=True)
LOG = json.load(open(os.path.join(EXP, "r6f6_log.json")))
VIV = json.load(open(os.path.join(FIGD, "palette_vivid_dark.json")))
ERRV, QUALV = VIV["errors_vivid_dark"], VIV["qualitative_instances_vivid"]
MAT = VIV["material"]; OP = VIV["opacity"]
PAL = R.PALETTE
RULE = os.path.join(HERE, "R6_FIG6_RULE.md")
PPI = 1000
COLS = [("gt", "Ground truth"), ("base", "Original\nCellpose-SAM"), ("nnunet", "nnU-Net"), ("2D", "2D control"), ("S1", "S1")]
SETS = {"A": ["nis3d", "tapenade", "nisnet_zf", "mouse_embryo", "blastospim"],
        "B": ["cellos", "parhyale", "nucverse3d", "mesospim", "blastospim"]}
NN_BETTER = {"mesospim", "blastospim"}            # version B rows where nnU-Net is better (g2ds, F1@0.7)
G2DS = json.load(open(os.path.join(FIGD, "..", "data_snapshot", "paired_stats_gt.json")))["nnU-Net|13|F1@0.7"]["per_ds"]


# ---------------------------------------------------------------- render primitives (copied from r6_render.py)
def plotter(px):
    p = pv.Plotter(off_screen=True, window_size=(px, px), lighting="none")
    p.set_background("black")
    for l_ in (pv.Light(position=(-1, 1, 1.4), focal_point=(0, 0, 0), intensity=0.95, light_type="camera light"),
               pv.Light(position=(1, 0.2, 0.6), focal_point=(0, 0, 0), intensity=0.35, light_type="camera light"),
               pv.Light(position=(0.2, -1, -0.8), focal_point=(0, 0, 0), intensity=0.45, light_type="camera light"),
               pv.Light(light_type="headlight", intensity=0.2)):
        p.add_light(l_)
    p.enable_depth_peeling(number_of_peels=10, occlusion_ratio=0.0)
    return p


def mesh(p, surf, rgb, opacity=1.0):
    surf = surf.copy()
    surf.cell_data["rgb"] = (np.asarray(rgb) * 255).astype(np.uint8)
    p.add_mesh(surf, scalars="rgb", rgb=True, smooth_shading=True, opacity=opacity, **MAT)


def shoot(p):
    p.enable_ssao(**R.RENDER["ssao"])
    im = p.screenshot(transparent_background=True, return_img=True)
    p.close()
    return Image.fromarray(im)


def camera(p, bounds):
    """Orthographic view, azimuth 35, elevation 25 (as r6_render); fitted to fixed bounds so that all panes of a row match."""
    p.camera_position = "xy"; p.camera.azimuth = 35; p.camera.elevation = -25
    p.enable_parallel_projection(); p.reset_camera(bounds=bounds)


# ---------------------------------------------------------------- rules
def gt_box(gt, d, ratio):
    """Rule 3: bounding box of GT foreground expanded by d/2 (z: d/2/ratio), clipped."""
    z, y, x = [np.nonzero((gt > 0).any(axis=tuple(a for a in range(3) if a != k)))[0] for k in range(3)]
    mz, mxy = int(np.ceil(0.5 * d / ratio)), int(np.ceil(0.5 * d))
    return tuple(slice(max(a.min() - m, 0), min(a.max() + m + 1, n)) for a, m, n in zip((z, y, x), (mz, mxy, mxy), gt.shape))


def zoom_region(Z):
    """Rule 5: disagreement-weighted window of 3 nuclear diameters; returns (slices in volume coords, record)."""
    gt, d, ratio = Z["gt"], float(Z["d"]), float(Z["ratio"])
    ids = np.unique(gt); ids = ids[ids > 0]
    best = {k: R.gt_best_iou(gt, Z[k]) for k in ("S1", "base", "nnunet", "2D")}
    s = sum(np.abs(best["S1"][ids] - best[k][ids]) for k in ("base", "nnunet", "2D"))
    cen = np.array(ndi.center_of_mass(np.ones_like(gt, np.uint8), gt, ids)).reshape(-1, 3) * np.array([ratio, 1, 1])
    D = np.linalg.norm(cen[:, None] - cen[None], axis=2)
    inwin = D <= 1.5 * d
    score = (inwin * s[None]).sum(1); n = inwin.sum(1)
    order = sorted(range(len(ids)), key=lambda i: (-round(float(score[i]), 9), -int(n[i]), int(ids[i])))
    i = order[0]; c = cen[i] / np.array([ratio, 1, 1])
    h = np.array([1.5 * d / ratio, 1.5 * d, 1.5 * d])
    sl = tuple(slice(int(max(np.floor(c[k] - h[k]), 0)), int(min(np.ceil(c[k] + h[k]) + 1, gt.shape[k]))) for k in range(3))
    return sl, dict(centre_label=int(ids[i]), score=float(score[i]), n_nuclei_in_window=int(n[i]),
                    s_per_nucleus_in_window={int(ids[j]): round(float(s[j]), 4) for j in np.nonzero(inwin[i])[0]},
                    window_voxels=[[x.start, x.stop] for x in sl])


# ---------------------------------------------------------------- rendering one pane
def surf_or_none(lab, ratio, keep=None):
    if keep is not None and not len(keep):
        return None, None
    if (lab if keep is None else np.isin(lab, keep)).max() == 0:
        return None, None
    try:
        s, bl = R.label_surface(lab, ratio, keep)
    except KeyError:                      # no surface produced (e.g. only sub-voxel fragments in the crop)
        return None, None
    return (s, bl) if s.n_cells else (None, None)


def bounds_of(sl, ratio, origin):
    z, y, x = [(s.start - o, s.stop - o) for s, o in zip(sl, origin)]
    return (x[0], x[1], y[0], y[1], z[0] * ratio, z[1] * ratio)


def render_pane(Z, k, sl, px, marker=None, faint_box=True, tag=""):
    """Render model k (or 'gt') inside slices sl. Error classes are computed on the whole evaluated volume."""
    fn = os.path.join(CACHE, f"{tag}_{k}_{px}.png")
    if os.path.exists(fn):
        return Image.open(fn)
    gt_full, ratio = Z["gt"], float(Z["ratio"])
    gt = gt_full[sl]
    p = plotter(px)
    if k == "gt":
        s, bl = surf_or_none(gt, ratio)
        if s is not None:
            mesh(p, s, [pv.Color(QUALV[b % len(QUALV)]).float_rgb for b in bl])
    else:
        cls, missed, _ = R.classify_preds(gt_full, Z[k])
        lab = Z[k][sl]
        s, bl = surf_or_none(lab, ratio)
        if s is not None:
            kk = np.array([cls.get(int(b), "other") for b in bl])
            for c in ("match", "boundary", "split", "merge", "other"):
                idx = np.nonzero(kk == c)[0]
                if len(idx):
                    sub = s.extract_cells(idx).extract_surface(algorithm="dataset_surface")
                    mesh(p, sub, np.tile(pv.Color(ERRV[c]).float_rgb, (sub.n_cells, 1)), OP["matched"] if c == "match" else 1.0)
        miss_here = [m for m in missed if (gt == m).any()]
        if miss_here:
            gs, _ = surf_or_none(gt, ratio, miss_here)
            if gs is not None:
                p.add_mesh(gs, color=ERRV["miss"], opacity=OP["missed_ghost"], smooth_shading=True, **MAT)
    B = (0, gt.shape[2], 0, gt.shape[1], 0, gt.shape[0] * ratio)
    edge_w = max(1.0, px / 900)
    if faint_box:
        p.add_mesh(pv.Box(bounds=B).extract_feature_edges(), color="#5b606b", line_width=edge_w)
    if marker is not None:
        p.add_mesh(pv.Box(bounds=marker).extract_feature_edges(), color="#ffffff", line_width=2.2 * edge_w)
    camera(p, B)
    im = shoot(p); im.save(fn)
    return im


def common_crop(ims, margin=0.01):
    a = np.max([np.asarray(i)[..., 3] for i in ims], 0)
    ys, xs = np.nonzero(a > 8); m = int(margin * a.shape[0])
    box = (max(xs.min() - m, 0), max(ys.min() - m, 0), min(xs.max() + m, a.shape[1]), min(ys.max() + m, a.shape[0]))
    return [i.crop(box) for i in ims]


def fit(im, w, h):
    sc = min(w / im.width, h / im.height)
    return im.resize((max(1, int(round(im.width * sc))), max(1, int(round(im.height * sc)))), Image.LANCZOS)


def compose(main, inset, w, h, iw, pad):
    """Black tile w x h px: main render fitted to the tile (top-left aligned region), zoom inset (iw px square) bottom-right."""
    t = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    m = fit(main, w - 2 * pad - iw // 4, h - 2 * pad)        # main render shifted left so the inset hides less of it
    t.alpha_composite(m, (pad + (w - 2 * pad - iw // 4 - m.width) // 2, pad + (h - 2 * pad - m.height) // 2))
    box = Image.new("RGBA", (iw, iw), (0, 0, 0, 255))
    ins = fit(inset, iw - 6, iw - 6)
    box.alpha_composite(ins, ((iw - ins.width) // 2, (iw - ins.height) // 2))
    bw = max(2, iw // 120)
    arr = np.asarray(box).copy(); arr[:bw] = arr[-bw:] = 255; arr[:, :bw] = arr[:, -bw:] = 255
    t.alpha_composite(Image.fromarray(arr), (w - iw - pad // 2, h - iw - pad // 2))
    return t.convert("RGB")


# ---------------------------------------------------------------- figure
def build(ver, draft=False):
    dss = SETS[ver]
    W_mm, Lmm, gap = 183.0, 18.0, 0.6
    tw_mm = (W_mm - Lmm - gap * 4) / 5; th_mm = 0.86 * tw_mm
    head_mm, foot_mm, rowgap = 6.0, 8.5, 1.4
    H_mm = head_mm + len(dss) * th_mm + (len(dss) - 1) * rowgap + foot_mm
    assert H_mm <= 170, H_mm
    ppi = 300 if draft else PPI
    tw_px, th_px = int(round(tw_mm / 25.4 * ppi)), int(round(th_mm / 25.4 * ppi))
    rpx = 2 * max(tw_px, th_px) + 200                     # render >= 2x final pixel size (supersampling)
    iw = int(0.42 * tw_px); pad = int(0.02 * tw_px)
    fig = plt.figure(figsize=(W_mm / 25.4, H_mm / 25.4), dpi=ppi, facecolor="white")
    X = lambda mm: mm / W_mm; Y = lambda mm: 1 - mm / H_mm  # noqa: E731
    prov = {"version": ver, "rule_file": "paper/scripts/render3d/R6_FIG6_RULE.md",
            "rule_md5": hashlib.md5(open(RULE, "rb").read()).hexdigest(), "datasets": dss, "rows": {},
            "render": {"supersampling": f"rendered at {rpx} px square, Lanczos-downsampled to {tw_px}x{th_px} px tiles",
                       "ppi_embedded": ppi, "depth_peeling": 10, "ssao": R.RENDER["ssao"], "smoothing": [R.RENDER["smoothing_iterations"], R.RENDER["smoothing_relaxation"]],
                       "material": MAT, "opacity": OP, "camera": "orthographic, azimuth 35, elevation 25, fitted to the shared box",
                       "colours": "figures/palette_vivid_dark.json (hue of palette.json NPG kept; only lightness/saturation changed)"},
            "export": "paper/scripts/render3d/r6f6_server_export.py -> export/r6f6 (masks identical to export/<ds>.npz where both exist)"}
    for c, (k, lab) in enumerate(COLS):
        x0 = Lmm + c * (tw_mm + gap)
        fig.text(X(x0 + tw_mm / 2), Y(head_mm - 0.6), lab, ha="center", va="bottom", fontsize=7.0 if k == "S1" else 6.0,
                 fontweight="bold" if k in ("S1", "gt") else "normal", color=PAL["models"]["S1"] if k == "S1" else PAL["text"], linespacing=1.0)
    for r, ds in enumerate(dss):
        Z = dict(np.load(os.path.join(EXP, f"r6f6_{ds}.npz")))
        info = LOG[ds]
        d, ratio = float(Z["d"]), float(Z["ratio"])
        box = gt_box(Z["gt"], d, ratio)
        zsl, zrec = zoom_region(Z)
        org = [b.start for b in box]
        zin = tuple(slice(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(zsl, box))
        marker = bounds_of(zin, ratio, org)
        tag = f"{ds}_{info['id']}"
        mains = common_crop([render_pane(Z, k, box, rpx, marker=marker, tag=tag + "_full") for k, _ in COLS])
        insets = common_crop([render_pane(Z, k, zsl, rpx // 2, tag=tag + "_zoom") for k, _ in COLS])
        y0 = head_mm + r * (th_mm + rowgap)
        for c, (k, lab) in enumerate(COLS):
            x0 = Lmm + c * (tw_mm + gap)
            ax = fig.add_axes([X(x0), Y(y0 + th_mm), X(tw_mm), th_mm / H_mm])
            ax.imshow(compose(mains[c], insets[c], tw_px, th_px, iw, pad), interpolation="none")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(k == "S1"); sp.set_color(PAL["models"]["S1"]); sp.set_linewidth(1.0)
            if k == "gt":
                ax.text(0.04, 0.96, f"{info['models']['S1']['n_gt']} nuclei", transform=ax.transAxes, ha="left", va="top", fontsize=5.2, color="#e8e8ea")
            else:
                mi = info["models"][k]
                ax.text(0.04, 0.96, f"F1@0.5 {mi['f1_05']:.3f}\nF1@0.7 {mi['f1_07']:.3f}", transform=ax.transAxes, ha="left", va="top",
                        fontsize=5.4, linespacing=1.15, color="#ffffff" if k == "S1" else "#d6d9e1", fontweight="bold" if k == "S1" else "normal",
                        bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.6))
        name = R.DS_NAME[ds].replace("NISNet3D zebrafish", "NISNet3D\nzebrafish")
        sub = f"z/xy {ratio:.2f}"
        if ver == "B":
            dd = G2DS[ds]["d"]
            sub += "\nS1 − nnU-Net\nF1@0.7 " + f"{dd:+.3f}".replace("-", "−")
            if ds == "mesospim":
                sub += "\nsingle volume; not in\n12-dataset analyses"
        lc = PAL["text"]
        yl = Y(y0 + th_mm / 2)
        fig.text(X(Lmm - 1.2), yl + 0.012 * (1 if ver == "B" else 0.5), name, ha="right", va="bottom", fontsize=6.3, fontweight="bold", color=lc,
                 style="italic" if ds == "parhyale" else "normal")
        fig.text(X(Lmm - 1.2), yl + 0.012 * (1 if ver == "B" else 0.5) - 0.002, sub, ha="right", va="top", fontsize=5.4, color=PAL["muted"], linespacing=1.15)
        if ver == "B":
            better = "nnU-Net better" if ds in NN_BETTER else "S1 better"
            col = PAL["models"]["nnunet"] if ds in NN_BETTER else PAL["models"]["S1"]
            fig.text(X(Lmm - 1.2), Y(y0 + th_mm - 0.6), better, ha="right", va="bottom", fontsize=5.6, fontweight="bold", color=col)
        prov["rows"][ds] = dict(volume=info["id"], shape=info["shape"], d=d, z_xy=ratio, display_box=[[b.start, b.stop] for b in box], zoom=zrec,
                                f1={k: {kk: info["models"][k][kk] for kk in ("f1_05", "f1_07", "tp5", "tp7", "n_gt", "n_pred", "source")} for k in ("base", "nnunet", "2D", "S1")},
                                **({"g2ds_S1_minus_nnunet_F1@0.7": G2DS[ds]["d"], "g2ds_ci": G2DS[ds].get("ci"), "row_label": "nnU-Net better" if ds in NN_BETTER else "S1 better"} if ver == "B" else {}))
    leg = [Patch(fc=ERRV[k], ec="none", label=PAL["error_names"][k]) for k in ("match", "boundary", "split", "merge", "other", "miss")]
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(X(Lmm + (W_mm - Lmm) / 2), 1.2 / H_mm), ncol=6, fontsize=5.6, frameon=False,
               handlelength=1.1, columnspacing=1.3, labelcolor=PAL["text"])
    fig.text(X(Lmm), 5.6 / H_mm, "Inset: zoom of the white box (3 nuclear diameters; region of largest S1-vs-other disagreement, rule-based)",
             ha="left", va="bottom", fontsize=5.2, color=PAL["muted"])
    cid = f"R6-Fig6-{ver}" + ("-draft" if draft else "")
    prov["overlap_and_font_check"] = figcheck.check(fig)
    prov["size_mm"] = [W_mm, round(H_mm, 2)]
    fig.savefig(os.path.join(GA, f"{cid}.png"), dpi=ppi, facecolor="white")
    if not draft:
        fig.savefig(os.path.join(GA, f"{cid}.pdf"), dpi=ppi, facecolor="white")
    json.dump(prov, open(os.path.join(GA, f"{cid}.provenance.json"), "w"), indent=1, ensure_ascii=False, default=str)
    plt.close(fig)
    print("saved", cid, prov["size_mm"], prov["overlap_and_font_check"])


if __name__ == "__main__":
    build(sys.argv[1], draft="--draft" in sys.argv)
