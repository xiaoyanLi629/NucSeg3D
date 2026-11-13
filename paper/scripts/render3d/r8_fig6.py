"""R8 Fig. 6 (gallery R8-Fig6): R7 with the colour scheme of Supplementary_Data_3D_viewer.html as the user saw it (2026-11-11):
warm grey panes #EDECE9, matched steel blue, boundary ochre, split purple, merge dark red, false positive light grey, missed grey-blue.

Based on R7 Fig. 6 (gallery R7-Fig6): R6-Fig6-A content (rule R6_FIG6_RULE.md, version A datasets; whole evaluated volume per model
with a rule-based zoom inset) redrawn in the light style of the installed Fig. 4 R9 / Fig. 5 R7 (user 2026-11-11): every tile on
the soft blue-grey tint (#EEF2F7) instead of black, zoom inset on white with a dark frame, the zoom marker and the faint volume box
in dark / light grey, dark text. Rendering, selection, numbers and data are those of r6_fig6.py (imported, not modified); only
the background, marker colours and the compositing change. Near-black low-alpha SSAO shading (< 40/255) is dropped before
compositing, as in Fig. 5 R7, because it shows as a grey halo on a light background.
Usage: python r7_fig6.py [--draft]"""
import hashlib
import json
import os
import sys

import numpy as np
import pyvista as pv
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from PIL import Image

import r6_fig6 as B
from r6_fig6 import COLS, ERRV, EXP, LOG, MAT, OP, PAL, QUALV, R, figcheck, plt

TINT = "#edece9"
VIEWER = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures", "palette_viewer.json")))["errors"]
ERRV.update(VIEWER)
OP["matched"] = 0.9
QUALV[:] = ["#8FA3C9", "#E9A99A", "#8CC7BB", "#B9B2D3", "#D9C3A6", "#9FD3E2", "#C8A49B", "#A9BFA0", "#E3C59A", "#B3B7C9"]   # palette.json qualitative_instances, as in the viewer
TINT_RGBA = tuple(int(round(255 * c)) for c in to_rgb(TINT)) + (255,)
MARK = "#2f3036"
EDGE = "#b4b8c2"


def render_pane_l(Z, k, sl, px, marker=None, faint_box=True, tag=""):
    fn = os.path.join(B.CACHE, f"V_{tag}_{k}_{px}.png")
    if os.path.exists(fn):
        return Image.open(fn)
    gt_full, ratio = Z["gt"], float(Z["ratio"])
    gt = gt_full[sl]
    p = B.plotter(px)
    if k == "gt":
        s, bl = B.surf_or_none(gt, ratio)
        if s is not None:
            B.mesh(p, s, [pv.Color(QUALV[b % len(QUALV)]).float_rgb for b in bl])
    else:
        cls, missed, _ = R.classify_preds(gt_full, Z[k])
        s, bl = B.surf_or_none(Z[k][sl], ratio)
        if s is not None:
            kk = np.array([cls.get(int(b), "other") for b in bl])
            for c in ("match", "boundary", "split", "merge", "other"):
                idx = np.nonzero(kk == c)[0]
                if len(idx):
                    sub = s.extract_cells(idx).extract_surface(algorithm="dataset_surface")
                    B.mesh(p, sub, np.tile(pv.Color(ERRV[c]).float_rgb, (sub.n_cells, 1)), OP["matched"] if c == "match" else 1.0)
        miss_here = [m for m in missed if (gt == m).any()]
        if miss_here:
            gs, _ = B.surf_or_none(gt, ratio, miss_here)
            if gs is not None:
                p.add_mesh(gs, color=ERRV["miss"], opacity=OP["missed_ghost"], smooth_shading=True, **MAT)
    Bx = (0, gt.shape[2], 0, gt.shape[1], 0, gt.shape[0] * ratio)
    ew = max(1.0, px / 900)
    if faint_box:
        p.add_mesh(pv.Box(bounds=Bx).extract_feature_edges(), color=EDGE, line_width=ew)
    if marker is not None:
        p.add_mesh(pv.Box(bounds=marker).extract_feature_edges(), color=MARK, line_width=2.2 * ew)
    B.camera(p, Bx)
    im = B.shoot(p); im.save(fn)
    return im


def dehalo(im):
    a = np.asarray(im).copy(); a[..., 3][a[..., 3] < 40] = 0
    return Image.fromarray(a)


def _place_main(main, w, h, iw, pad):
    t = Image.new("RGBA", (w, h), TINT_RGBA)
    m = B.fit(dehalo(main), w - 2 * pad - iw // 4, h - 2 * pad)
    t.alpha_composite(m, (pad + (w - 2 * pad - iw // 4 - m.width) // 2, pad + (h - 2 * pad - m.height) // 2))
    return t


def pick_corner(main, w, h, iw, pad):
    """Corner for the inset that hides the fewest pixels of the dark zoom-marker box (R8 fix: in three rows the bottom-right
    inset covered the marker)."""
    a = np.asarray(_place_main(main, w, h, iw, pad)).astype(int)
    mc = np.array([int(MARK[i:i + 2], 16) for i in (1, 3, 5)])
    mk = (np.abs(a[..., :3] - mc).sum(-1) < 40)
    best = None
    for name, (x0, y0) in {"br": (w - iw - pad // 2, h - iw - pad // 2), "bl": (pad // 2, h - iw - pad // 2),
                           }.items():   # bottom corners only: the top corners carry the F1 labels
        cov = int(mk[y0:y0 + iw, x0:x0 + iw].sum())
        if best is None or cov < best[0]:
            best = (cov, name, (x0, y0))
    return best


def compose_l(main, inset, w, h, iw, pad, corner=None):
    t = _place_main(main, w, h, iw, pad)
    box = Image.new("RGBA", (iw, iw), (255, 255, 255, 255))
    ins = B.fit(dehalo(inset), iw - 6, iw - 6)
    box.alpha_composite(ins, ((iw - ins.width) // 2, (iw - ins.height) // 2))
    bw = max(2, iw // 140)
    arr = np.asarray(box).copy(); mc = np.array(to_rgb(MARK)) * 255
    arr[:bw, :, :3] = arr[-bw:, :, :3] = mc; arr[:, :bw, :3] = arr[:, -bw:, :3] = mc
    xy = corner[2] if corner else (w - iw - pad // 2, h - iw - pad // 2)
    t.alpha_composite(Image.fromarray(arr), xy)
    return t.convert("RGB")


def build(draft=False):
    dss = B.SETS["A"]
    W_mm, Lmm, gap = 183.0, 18.0, 0.8
    tw_mm = (W_mm - Lmm - gap * 4) / 5; th_mm = 0.86 * tw_mm
    head_mm, foot_mm, rowgap = 6.0, 8.5, 0.8
    H_mm = head_mm + len(dss) * th_mm + (len(dss) - 1) * rowgap + foot_mm
    ppi = 300 if draft else B.PPI
    tw_px, th_px = int(round(tw_mm / 25.4 * ppi)), int(round(th_mm / 25.4 * ppi))
    rpx = 2 * max(int(round(tw_mm / 25.4 * B.PPI)), int(round(th_mm / 25.4 * B.PPI))) + 200   # same render size as R6 (cache reuse)
    iw = int(0.42 * tw_px); pad = int(0.02 * tw_px)
    fig = plt.figure(figsize=(W_mm / 25.4, H_mm / 25.4), dpi=ppi, facecolor="white")
    X = lambda mm: mm / W_mm; Y = lambda mm: 1 - mm / H_mm  # noqa: E731
    prov = {"figure": "R8-Fig6", "script": "paper/scripts/render3d/r8_fig6.py (renders and rules from r6_fig6.py, version A)",
            "rule_file": "paper/scripts/render3d/R6_FIG6_RULE.md", "rule_md5": hashlib.md5(open(B.RULE, "rb").read()).hexdigest(),
            "datasets": dss, "rows": {}, "style": {"tile": TINT, "inset": "white, frame " + MARK, "marker": MARK, "volume_box": EDGE,
                                                  "dehalo": "alpha < 40/255 set to 0 before compositing"}}
    for c, (k, lab) in enumerate(COLS):
        x0 = Lmm + c * (tw_mm + gap)
        fig.text(X(x0 + tw_mm / 2), Y(head_mm - 0.6), lab, ha="center", va="bottom", fontsize=7.0 if k == "S1" else 6.0,
                 fontweight="bold" if k in ("S1", "gt") else "normal", color=PAL["models"]["S1"] if k == "S1" else PAL["text"], linespacing=1.0)
    for r, ds in enumerate(dss):
        Z = dict(np.load(os.path.join(EXP, f"r6f6_{ds}.npz")))
        info = LOG[ds]
        d, ratio = float(Z["d"]), float(Z["ratio"])
        box = B.gt_box(Z["gt"], d, ratio)
        zsl, zrec = B.zoom_region(Z)
        org = [b.start for b in box]
        zin = tuple(slice(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(zsl, box))
        marker = B.bounds_of(zin, ratio, org)
        tag = f"{ds}_{info['id']}"
        mains = B.common_crop([render_pane_l(Z, k, box, rpx, marker=marker, tag=tag + "_full") for k, _ in COLS])
        insets = B.common_crop([render_pane_l(Z, k, zsl, rpx // 2, tag=tag + "_zoom") for k, _ in COLS])
        y0 = head_mm + r * (th_mm + rowgap)
        row_corner = pick_corner(mains[1], tw_px, th_px, iw, pad)   # one corner per row (same camera in all panes)
        prov["rows"].setdefault(ds, {})
        prov.setdefault("inset_corner", {})[ds] = [row_corner[1], row_corner[0]]
        for c, (k, lab) in enumerate(COLS):
            x0 = Lmm + c * (tw_mm + gap)
            ax = fig.add_axes([X(x0), Y(y0 + th_mm), X(tw_mm), th_mm / H_mm])
            ax.imshow(compose_l(mains[c], insets[c], tw_px, th_px, iw, pad, corner=row_corner), interpolation="none")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(k == "S1"); sp.set_color(PAL["models"]["S1"]); sp.set_linewidth(1.0)
            if k == "gt":
                ax.text(0.04, 0.96, f"{info['models']['S1']['n_gt']} nuclei", transform=ax.transAxes, ha="left", va="top", fontsize=5.4,
                        color=PAL["text"], fontweight="bold", bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.3", fc="white", ec="none", alpha=0.82))
            else:
                mi = info["models"][k]
                ax.text(0.04, 0.96, f"F1@0.5 {mi['f1_05']:.3f}\nF1@0.7 {mi['f1_07']:.3f}", transform=ax.transAxes, ha="left", va="top",
                        fontsize=5.4, linespacing=1.15, color=PAL["models"]["S1"] if k == "S1" else PAL["text"],
                        fontweight="bold" if k == "S1" else "normal", bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.3", fc="white", ec="none", alpha=0.82))
        name = R.DS_NAME[ds].replace("NISNet3D zebrafish", "NISNet3D\nzebrafish")
        yl = Y(y0 + th_mm / 2)
        fig.text(X(Lmm - 1.2), yl + 0.006, name, ha="right", va="bottom", fontsize=6.3, fontweight="bold", color=PAL["text"])
        fig.text(X(Lmm - 1.2), yl + 0.004, f"z/xy {ratio:.2f}", ha="right", va="top", fontsize=5.4, color=PAL["muted"])
        prov["rows"][ds] = dict(volume=info["id"], z_xy=ratio, display_box=[[b.start, b.stop] for b in box], zoom=zrec,
                                f1={k: {kk: info["models"][k][kk] for kk in ("f1_05", "f1_07", "tp5", "tp7", "n_gt", "n_pred", "source")}
                                    for k in ("base", "nnunet", "2D", "S1")})
    leg = [Patch(fc=ERRV[k], ec="#9a9a96", lw=0.3, label=PAL["error_names"][k]) for k in ("match", "boundary", "split", "merge", "other", "miss")]
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(X(Lmm + (W_mm - Lmm) / 2), 1.2 / H_mm), ncol=6, fontsize=5.6, frameon=False,
               handlelength=1.1, columnspacing=1.3, labelcolor=PAL["text"])
    fig.text(X(Lmm), 5.6 / H_mm, "Inset: zoom of the dark box (3 nuclear diameters; region of largest S1-vs-other disagreement, rule-based)",
             ha="left", va="bottom", fontsize=5.2, color=PAL["muted"])
    cid = "R8-Fig6" + ("-draft" if draft else "")
    prov["overlap_and_font_check"] = figcheck.check(fig); prov["size_mm"] = [W_mm, round(H_mm, 2)]
    fig.savefig(os.path.join(B.GA, f"{cid}.png"), dpi=ppi, facecolor="white")
    if not draft:
        fig.savefig(os.path.join(B.GA, f"{cid}.pdf"), dpi=ppi, facecolor="white")
    json.dump(prov, open(os.path.join(B.GA, f"{cid}.provenance.json"), "w"), indent=1, ensure_ascii=False, default=str)
    plt.close(fig)
    print("saved", cid, prov["size_mm"], prov["overlap_and_font_check"])


if __name__ == "__main__":
    build(draft="--draft" in sys.argv)
