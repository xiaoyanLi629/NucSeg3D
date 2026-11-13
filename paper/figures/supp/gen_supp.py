"""R9 supplementary figures in one system (183 mm wide; Arial/Helvetica 5-8 pt with the main-figure hierarchy; #EDECE9 panes;
viewer error colours; model colours from palette.json; GT outlines white on images; no data, selection or number changes).

Hand-built SVG -> vector PDF + 600-dpi PNG through paper/figures/arch/gen_fig.export (Chrome; text/text, text/image, text/line
overlap check and minimum font size). Raster tiles: make_supp_assets.py (tiles/). Outputs: figures/gallery/assets/R9-FigS_<name>.*
Usage: python gen_supp.py [ortho Tgrid mm7 aniso mech]
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import os
import re
import shutil
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(PAPER, "figures", "arch"))
import gen_fig as GF  # noqa: E402
from gen_fig import LW, Fig, tint  # noqa: E402

OUTD = os.path.join(HERE, "out"); os.makedirs(OUTD, exist_ok=True)
GF.HERE = OUTD                                   # export() writes html / overlap / pdf / png here
GAL = os.path.join(PAPER, "figures", "gallery", "assets")
TL = os.path.join(HERE, "tiles")
PJ = json.load(open(os.path.join(PAPER, "figures", "palette.json")))
PVW = json.load(open(os.path.join(PAPER, "figures", "palette_viewer.json")))
PT = 0.3528
F_LET, F_T, F_S, F_L, F_N = 8.0, 7.0, 6.0, 5.5, 5.0
INK, MUTE = PJ["text"], PJ["muted"]
PANE, TAB = PVW["pane"], PVW["tab"]
M = PJ["models"]
MC = dict(S1=M["S1"], **{"2D": M["2D"]}, base=M["base"], nnunet=M["nnunet"], stardist3d=M["stardist3d"], useg_2d=M["useg"])
MC_TXT = dict(MC, **{"2D": M["2D_dark"], "S1": M["S1"], "useg_2d": "#3f8f80", "base": "#5d6680"})   # darker text variants for legibility
ERRK = ("match", "boundary", "split", "merge", "other", "miss")
ERRN = PJ["error_names"]
DSN = {"bbbc050": "BBBC050", "blastospim": "BlastoSPIM", "celegans": "C. elegans", "cellos": "Cellos", "gonuclear": "GoNuclear",
       "mesospim": "mesoSPIM", "mouse_embryo": "Mouse embryo", "nis3d": "NIS3D", "nisnet_zf": "NISNet3D zebrafish",
       "nucverse3d": "NucVerse3D", "parhyale": "Parhyale", "purdue": "Purdue", "tapenade": "Tapenade"}
REL = os.path.relpath(TL, GF.AS)                 # Fig.image paths are relative to arch/assets


def T(f, x, y, s, size=F_L, w=400, c=INK, a="start", **k):
    return f.text(x, y, s, size, w, c=c, a=a, **k)


def tw(s, size, bold=False):
    s = re.sub(r"[_^]\{([^}]*)\}", r"\1", s).replace("*", "")
    return len(s) * size * PT * (0.56 if bold else 0.52)


def letter(f, x, y, L, title=None):
    T(f, x, y, L, F_LET, 700)
    if title:
        T(f, x + 3.8, y, title, F_T, 700)


def pane(f, x, y, w, h, tab=None, accent=None):
    f.rect(x, y, w, h, rx=1.2, fill=PANE, stroke="none", chk=False)
    if tab:
        a0 = 1.6 if accent else 0.0
        twd = tw(tab, F_S, True) + 3.2 + a0
        f.add(f'<path d="M{x:.2f},{y + 4.2:.2f} V{y + 1.2:.2f} Q{x:.2f},{y:.2f} {x + 1.2:.2f},{y:.2f} H{x + twd:.2f} V{y + 4.2:.2f} Z" '
              f'fill="{TAB}" stroke="none"/>')
        if accent:
            f.rect(x + 1.2, y + 1.0, 1.1, 2.3, rx=0.3, fill=accent, stroke="none", chk=False)
        T(f, x + 1.5 + a0, y + 3.05, tab, F_S, 700, c="#3a3a3e")


def img(f, x, y, w, h, name, fit="xMidYMid meet", border=None, bw=0.5):
    f.image(x, y, w, h, os.path.join(REL, name), fit=fit)
    if border:
        f.rect(x, y, w, h, rx=0, fill="none", stroke=border, sw=bw * LW, chk=False)


def imsize(name):
    return Image.open(os.path.join(TL, name)).size


def err_legend(f, xc, y, keys=ERRK, extra=None):
    items = [(PVW["errors"][k], ERRN[k]) for k in keys]
    ws = [3.4 + 1.2 + tw(lab, F_L) * 0.92 + 2.6 for _, lab in items]
    if extra:
        ws.append(3.4 + 1.2 + tw(extra, F_L))
    x = xc - sum(ws) / 2
    for (col, lab), wd in zip(items, ws):
        f.rect(x, y - 2.3, 3.4, 2.4, rx=0.5, fill=col, stroke="none", chk=False)
        T(f, x + 4.6, y, lab, F_L)
        x += wd
    if extra:
        f.rect(x, y - 2.3, 3.4, 2.4, rx=0.3, fill="#ffffff", stroke=INK, sw=0.5 * LW, chk=False)
        T(f, x + 4.6, y, extra, F_L)


def finish(f, name, prov):
    cid = f"R9-FigS_{name}"
    res = GF.export(f, cid)
    prov.update(figure=cid, size_mm=[f.W, round(f.H, 1)], generator="paper/figures/supp/gen_supp.py; tiles: make_supp_assets.py",
                export_check=dict(issues=res.get("issues", []), min_font_pt=res.get("min_font_pt")),
                style="183 mm; Arial/Helvetica 5-8 pt (letter 8 bold, titles 7 bold, sub-titles 6 bold, labels 5.5, notes 5 muted); "
                      "panes #EDECE9; viewer error colours (palette_viewer.json); model colours palette.json; GT outlines white",
                items=f.prov)
    json.dump(prov, open(os.path.join(OUTD, f"{cid}.provenance.json"), "w"), indent=1, ensure_ascii=False, default=str)
    for ext in ("pdf", "png", "provenance.json"):
        shutil.copy(os.path.join(OUTD, f"{cid}.{ext}"), os.path.join(GAL, f"{cid}.{ext}"))
    print(cid, f"{f.W:.0f} x {f.H:.1f} mm", "issues", len(res.get("issues", [])), "min font", res.get("min_font_pt"))
    for s in res.get("issues", [])[:40]:
        print("   ", s)


# ======================================================================== orthogonal views
def ortho():
    OD = os.path.join(HERE, "data_ortho")
    SEL = json.load(open(os.path.join(OD, "selection.json")))
    VOL = SEL["volumes"]
    dss = SEL["pool_sorted_by_ratio"] + ["mesospim"]
    for part, sub in (("ortho_1", dss[:7]), ("ortho_2", dss[7:])):
        L, gi, gg, p = 25.0, 0.7, 2.0, 1.0
        c = (183.0 - L - 3 * (2 * gi + 2 * p) - 2 * gg) / 9
        gw = 3 * c + 2 * gi + 2 * p
        y0, rg = 0.5, 1.2
        ty0 = y0 + 8.6
        H = ty0 + len(sub) * (c + rg) - rg + p + 0.5
        f = Fig(183.0, H)
        gx = [L + k * (gw + gg) for k in range(3)]
        for k, (view, lab) in enumerate((("xy", "xy section"), ("xz", "xz section"), ("yz", "yz section"))):
            pane(f, gx[k], y0, gw, H - y0 - 0.5, tab=lab)
            for mi, (m, ml) in enumerate((("S1", "S1"), ("2D", "2D control"), ("base", "original"))):
                T(f, gx[k] + p + mi * (c + gi) + c / 2, ty0 - 1.3, ml, F_L, 700, c=MC_TXT[m], a="middle")
        T(f, 0.5, ty0 - 4.0, "per-volume F1@0.5:", F_N, c=MUTE)
        xx = 0.5
        for m, ml in (("S1", "S1"), ("2D", "2D"), ("base", "orig.")):
            T(f, xx, ty0 - 1.3, ml, F_N, 700, c=MC_TXT[m]); xx += tw(ml, F_N, True) + 2.4
        for r, ds in enumerate(sub):
            Cz = np.load(os.path.join(OD, "crops", f"{ds}.npz")); ratio = float(Cz["ratio"])
            ty = ty0 + r * (c + rg)
            v = VOL[ds]
            T(f, 0.5, ty + c / 2 - 2.0, DSN[ds], F_L, 700)
            T(f, 0.5, ty + c / 2 + 0.4, f"z/xy {ratio:.2f}", F_N, c=MUTE)
            xx = 0.5
            for m in ("S1", "2D", "base"):
                s = f"{v['f1'][m]:.2f}"
                T(f, xx, ty + c / 2 + 2.8, s, F_N, 700 if m == "S1" else 400, c=MC_TXT[m]); xx += tw(s, F_N, m == "S1") + 2.4
            for k, view in enumerate(("xy", "xz", "yz")):
                for mi, m in enumerate(("S1", "2D", "base")):
                    nm = f"O_{ds}_{view}_{m}.png"
                    iw, ih = imsize(nm)
                    sc = min(c / iw, c / ih)
                    w_, h_ = iw * sc, ih * sc
                    x_ = gx[k] + p + mi * (c + gi) + (c - w_) / 2
                    img(f, x_, ty + (c - h_) / 2, w_, h_, nm, border="#B9B8B2", bw=0.5)
        f.prov_add("tiles: same crops (ortho_prep.py), sections, images and outline rule as scripts/figs/ortho_plot.py; GT white, S1 / 2D control / original in palette colours; "
                   "one-voxel outlines as before, nearest-upsampled for display; xz and yz at physical aspect (z upsampled by z/xy), letterboxed in equal cells",
                   f"figures/supp/make_supp_assets.py (O); data_ortho copied read-only from server {_NSP_SCRATCH}/nc/figs/ortho")
        f.prov_add("row numbers: per-volume F1@0.5 of S1 / 2D control / original and z/xy, unchanged", "data_ortho/selection.json volumes[ds].f1; crops ratio")
        finish(f, part, dict(rule=SEL["rule"], order=sub, old=f"figures/final/FigS_{part}.pdf",
                             volumes={ds: {k: VOL[ds][k] for k in ("id", "ratio", "diff", "median_diff", "f1", "thr")} for ds in sub}))


# ======================================================================== 3D render grids
R3 = os.path.join(PAPER, "scripts", "render3d")
SELR = json.load(open(os.path.join(PAPER, "scripts", "figs", "ortho_selection.json")))["volumes"]
DSS5 = ["nis3d", "tapenade", "nisnet_zf", "mouse_embryo", "blastospim"]


def Tgrid():
    L, g = 17.0, 1.0
    cw = (183.0 - L - 4 * g) / 5
    ch = cw * 900 / 800
    nh = 3.0
    y0 = 8.0
    rows = (("gt", "Ground truth", INK), ("2D", "2D control", MC_TXT["2D"]), ("S1", "S1", MC_TXT["S1"]))
    H = y0 + 3 * ch + 2 * nh + 2 * 0.8 + 9.0
    f = Fig(183.0, H)
    for c_, ds in enumerate(DSS5):
        x = L + c_ * (cw + g)
        T(f, x + cw / 2, y0 - 4.0, DSN[ds], F_S, 700, a="middle")
        T(f, x + cw / 2, y0 - 1.4, f"z/xy {SELR[ds]['ratio']:.2f}", F_N, c=MUTE, a="middle")
    y = y0
    for r, (k, lab, col) in enumerate(rows):
        T(f, 0.5, y + ch / 2 + 0.7, lab, F_S, 700, c=col)
        for c_, ds in enumerate(DSS5):
            x = L + c_ * (cw + g)
            img(f, x, y, cw, ch, f"T_{ds}_{k}.png", fit="none slice" if False else "xMidYMid meet",
                border=M["S1"] if k == "S1" else None, bw=1.6)
            if k != "gt":
                T(f, x + cw - 0.6, y + ch + 2.3, f"{SELR[ds]['f1'][k]:.3f}", F_L, 700 if k == "S1" else 400,
                  c=MC_TXT[k] if k == "S1" else MUTE, a="end")
        y += ch + (nh if k != "gt" else 0.0) + 0.8
    err_legend(f, L + (183.0 - L) / 2 - 12.0, H - 2.2)
    T(f, 182.5, H - 2.2, "numbers: per-volume F1@0.5", F_N, c=MUTE, a="end")
    f.prov_add("renders: R8-T-grid-wide tiles (grid_cache_viewer, viewer palette), unchanged; numbers per-volume F1@0.5", "scripts/figs/ortho_selection.json volumes[ds].f1")
    finish(f, "Tgrid", dict(old="figures/final/FigS_Tgrid.pdf (R8-T-grid-wide)", datasets=DSS5, volumes={d: SELR[d]["id"] for d in DSS5}))


def mm7():
    MMLOG = json.load(open(os.path.join(R3, "export", "mm_log.json")))
    cols = [("gt", ["Ground truth"], INK, 700), ("base", ["Original", "Cellpose-SAM"], MC_TXT["base"], 400),
            ("stardist3d", ["StarDist-3D"], MC_TXT["stardist3d"], 400), ("useg_2d", ["u-Segment3D consensus", "of 2D control"], MC_TXT["useg_2d"], 400),
            ("nnunet", ["nnU-Net"], MC_TXT["nnunet"], 400), ("2D", ["2D control"], MC_TXT["2D"], 400), ("S1", ["S1"], MC_TXT["S1"], 700)]
    L, g = 22.0, 0.8
    cw = (183.0 - L - 6 * g) / 7
    y0, nh, rg = 9.0, 3.0, 1.0
    H = y0 + 5 * (cw + nh + rg) - rg + 8.5
    f = Fig(183.0, H)
    for j, (k, lab, col, wt) in enumerate(cols):
        x = L + j * (cw + g) + cw / 2
        T(f, x, y0 - 1.4 - (len(lab) - 1) * 2.6, lab, F_L if k != "S1" else F_S, wt, c=col, a="middle", ls=1.36)
    for r, ds in enumerate(DSS5):
        y = y0 + r * (cw + nh + rg)
        T(f, L - 1.2, y + cw / 2 - 0.4, DSN[ds], F_L, 700, a="end")
        T(f, L - 1.2, y + cw / 2 + 2.0, f"z/xy {SELR[ds]['ratio']:.2f}", F_N, c=MUTE, a="end")
        for j, (k, lab, col, wt) in enumerate(cols):
            x = L + j * (cw + g)
            nm = f"MM_{ds}_{k}.png"
            if os.path.exists(os.path.join(TL, nm)):
                img(f, x, y, cw, cw, nm, border=M["S1"] if k == "S1" else None, bw=1.6)
            else:
                f.rect(x, y, cw, cw, rx=0, fill=PANE, stroke="none", chk=False)
                T(f, x + cw / 2, y + cw / 2 + 0.7, "n/a", F_N, c=MUTE, a="middle")
            f1 = SELR[ds]["f1"][k] if k in ("S1", "2D", "base") else (MMLOG.get(ds, {}).get("models", {}).get(k) or {}).get("f1_05") \
                if isinstance(MMLOG.get(ds, {}).get("models", {}).get(k), dict) else None
            if f1 is not None:
                T(f, x + cw / 2, y + cw + 2.4, f"{f1:.3f}", F_L, 700 if k == "S1" else 400, c=MC_TXT["S1"] if k == "S1" else MUTE, a="middle")
    err_legend(f, L + (183.0 - L) / 2 - 12.0, H - 2.2)
    T(f, 182.5, H - 2.2, "numbers: per-volume F1@0.5", F_N, c=MUTE, a="end")
    f.prov_add("renders: R8-MM-grid tiles (grid_cache_viewer, viewer palette), unchanged; F1@0.5 from ortho_selection.json (S1, 2D, original) and export/mm_log.json (others)",
               "scripts/render3d/grids.py grid_MM (same lookup)")
    finish(f, "mm7", dict(old="figures/final/FigS_mm7.pdf (R8-MM-grid)", datasets=DSS5))


def aniso():
    L, g = 17.0, 1.2
    cw = (183.0 - L - 3 * g) / 4
    rh = cw * 760 / 1000
    sh = 11.5
    ch = rh + sh + 2.2
    y0 = 6.0
    H = y0 + 2 * ch + 2.0 + 8.0
    f = Fig(183.0, H)
    for j, k in enumerate((1, 2, 4, 8)):
        sub = "all planes" if k == 1 else ("every 2nd plane" if k == 2 else f"every {k}th plane")
        x = L + j * (cw + g) + cw / 2
        T(f, x - 1.2, y0 - 1.6, f"k = {k}", F_S, 700, a="end")
        T(f, x - 0.4, y0 - 1.6, f"({sub})", F_L, c=MUTE)
    for r, (m, lab) in enumerate((("2D", "2D control"), ("S1", "S1"))):
        y = y0 + r * (ch + 2.0)
        T(f, 0.5, y + ch / 2 + 0.7, lab, F_S, 700, c=MC_TXT[m])
        for j, k in enumerate((1, 2, 4, 8)):
            x = L + j * (cw + g)
            f.rect(x, y, cw, ch, rx=1.2, fill=PANE, stroke="none", chk=False)
            img(f, x, y, cw, rh, f"F5_{m}_k{k}.png")
            nm = f"F5x_{m}_k{k}.png"; iw, ih = imsize(nm)
            ww = sh * iw / ih
            img(f, x + (cw - ww) / 2, y + rh + 0.9, ww, sh, nm, border=MC[m], bw=1.4)
    err_legend(f, 91.5, H - 2.4, keys=("match", "boundary", "split", "merge", "other"), extra="GT outline (xz, white)")
    f.prov_add("renders and xz strips: cand_f5.render / cand_f5.xz with the viewer palette (as R8-F5-a), unchanged data and crop", "figures/supp/make_supp_assets.py F5; scripts/render3d/F5_RULE.md")
    finish(f, "anisotropy3D", dict(old="figures/final/FigS_anisotropy3D.pdf (R8-F5-a)", rule="scripts/render3d/F5_RULE.md",
                                   note="illustration only; masks re-inferred on the crop; no numbers shown"))


# ======================================================================== mechanism
def axes(f, x, y, w, h, xt, yt, xl, yl, fx, fy, xlab=None, ylab=None):
    """Plot frame in the main-figure style: white area, light gridlines, left/bottom axes, ticks outside."""
    f.rect(x, y, w, h, rx=0, fill="#ffffff", stroke="none", chk=False)
    for v in xt:
        f.add(f'<line x1="{fx(v):.2f}" y1="{y:.2f}" x2="{fx(v):.2f}" y2="{y + h:.2f}" stroke="#E4E4E6" stroke-width="{0.5 * LW:.3f}"/>')
        f.add(f'<line x1="{fx(v):.2f}" y1="{y + h:.2f}" x2="{fx(v):.2f}" y2="{y + h + 0.8:.2f}" stroke="{INK}" stroke-width="{0.5 * LW:.3f}"/>')
    for v in yt:
        f.add(f'<line x1="{x:.2f}" y1="{fy(v):.2f}" x2="{x + w:.2f}" y2="{fy(v):.2f}" stroke="#E4E4E6" stroke-width="{0.5 * LW:.3f}"/>')
        f.add(f'<line x1="{x - 0.8:.2f}" y1="{fy(v):.2f}" x2="{x:.2f}" y2="{fy(v):.2f}" stroke="{INK}" stroke-width="{0.5 * LW:.3f}"/>')
    f.add(f'<path d="M{x:.2f},{y:.2f} V{y + h:.2f} H{x + w:.2f}" stroke="{INK}" stroke-width="{0.5 * LW:.3f}" fill="none"/>')
    for v, s in zip(xt, xl):
        T(f, fx(v), y + h + 3.0, s, F_N, a="middle")
    for v, s in zip(yt, yl):
        T(f, x - 1.3, fy(v) + 0.7, s, F_N, a="end")
    if xlab:
        T(f, x + w / 2, y + h + 6.0, xlab, F_L, a="middle")


def line(f, pts, col, sw=0.9, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    f.add('<path d="M' + " L".join(f"{a:.2f},{b:.2f}" for a, b in pts) + f'" stroke="{col}" stroke-width="{sw * LW * 2 / 2:.3f}" fill="none"{d} '
          f'stroke-linejoin="round"/>')


def band(f, x, lo, hi, fx, fy, col, op):
    pts = [(fx(a), fy(b)) for a, b in zip(x, hi)] + [(fx(a), fy(b)) for a, b in zip(x[::-1], lo[::-1])]
    f.add('<path d="M' + " L".join(f"{a:.2f},{b:.2f}" for a, b in pts) + f' Z" fill="{col}" opacity="{op}" stroke="none"/>')


def marker(f, x, y, kind, col, s=0.75):
    if kind == "o":
        f.circle(x, y, s * 0.62, fill=col, stroke="#ffffff", sw=0.3 * LW, chk=False)
    elif kind == "s":
        f.rect(x - s * 0.55, y - s * 0.55, s * 1.1, s * 1.1, rx=0.1, fill=col, stroke="#ffffff", sw=0.3 * LW, chk=False)
    else:
        f.poly([(x, y - s * 0.75), (x + s * 0.7, y + s * 0.5), (x - s * 0.7, y + s * 0.5)], fill=col, stroke="#ffffff", sw=0.3 * LW, chk=False)


def ylabel(f, x, y, s, size=F_L):
    """Rotated y-axis label."""
    f.add(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size * PT:.3f}" fill="{INK}" text-anchor="middle" transform="rotate(-90 {x:.2f} {y:.2f})">'
          f'{GF.mark(s, size)}</text>')


def mech():
    MJ = json.load(open(os.path.join(TL, "mech.json")))
    EJ = os.path.join(PAPER, "scripts", "figs", "localdata", "errors_results.json")
    E = json.load(open(EJ))
    s4 = [v for k, v in E["gt"]["S4"]["all"].items() if "mesospim" in k][0]
    W = 183.0
    # ---------------- a
    tw_mm, th_mm = 33.0, 33.0 * MJ["tile_mm"][1] / MJ["tile_mm"][0]
    ya = 9.0
    H = 105.0
    f = Fig(W, H)
    letter(f, 0, 3.4, "a", "One nucleus: GT-derived 3D flow target and each model's angle (mouse embryo, nucleus 62)")
    pw = 3 * tw_mm + 4 * 1.6 + 16.0
    pane(f, 0.5, 5.4, pw, th_mm + 12.6)
    titles = (("GT", "GT-derived 3D flow target"), ("2D", "2D control (3 views fused)"), ("S1", "S1"))
    for j, (m, t) in enumerate(titles):
        x = 0.5 + 1.6 + j * (tw_mm + 1.6)
        T(f, x + tw_mm / 2, ya + 0.6, t, F_L, 700, c=MC_TXT.get(m, INK) if m != "GT" else INK, a="middle")
        img(f, x, ya + 2.2, tw_mm, th_mm, f"M_{m}.png")
    # scale bar 3 um under the S1 tile (tile spans tile_um[0] um)
    um_mm = tw_mm / MJ["tile_um"][0]
    xs1 = 0.5 + 1.6 + 2 * (tw_mm + 1.6)
    yb = ya + 2.2 + th_mm + 1.6
    f.add(f'<line x1="{xs1 + tw_mm - 3 * um_mm:.2f}" y1="{yb:.2f}" x2="{xs1 + tw_mm:.2f}" y2="{yb:.2f}" stroke="{INK}" stroke-width="{1.0 * LW * 2:.3f}"/>')
    T(f, xs1 + tw_mm - 1.5 * um_mm, yb + 2.6, "3 µm", F_N, a="middle")
    T(f, 0.5 + 1.6, yb + 2.6, "target arrows coloured by u_{z}: navy up, red down; outline = GT nucleus 62", F_N, c=MUTE)
    # angle colour bar
    cbx = 0.5 + 1.6 + 3 * (tw_mm + 1.6) + 1.6
    gid = "angM"
    stops = "".join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in ((0, "#ffffff"), (0.33, "#F39B7F"), (0.67, "#E64B35"), (1, "#7a1f12")))
    f.defs.append(f'<linearGradient id="{gid}" x1="0" x2="0" y1="1" y2="0">{stops}</linearGradient>')
    f.add(f'<rect x="{cbx:.2f}" y="{ya + 2.6:.2f}" width="1.6" height="{th_mm - 0.8:.2f}" rx="0.3" fill="url(#{gid})" stroke="#B9B8B2" stroke-width="{0.5 * LW:.3f}"/>')
    for v in (0, 30, 60, 90):
        yy = ya + 2.6 + (th_mm - 0.8) * (1 - v / 90)
        T(f, cbx + 2.4, yy + 0.7, f"{v}", F_N)
    T(f, cbx + 0.8, ya + 0.6, "angle (°)", F_N, c=MUTE, a="middle")
    f.prov_add("a: nucleus 62 (mouse embryo E15_Ab2, seed 0, xz section y = 126): GT-derived 3D flow target arrows (colour u_z) | per-voxel angle of the 2D control (3 views fused) and S1 to the target with flow arrows; arrows every 2 px, every plane; tiles re-rendered without text by the code of v2_detail3.py (V2-i3)",
               "figures/supp/make_supp_assets.py M; figures/assets_viz/V2-i3_nucleus62.provenance.json")
    # ---------------- b
    bx0 = 0.5 + pw + 4.0
    letter(f, bx0, 3.4, "b", "Where along z the error sits")
    zp = MJ["zprofile"]
    px, py, pw_, ph = bx0 + 10.0, 12.0, W - 0.5 - (bx0 + 10.0), 27.0
    xc = (np.arange(10) + 0.5) / 10
    fx = lambda v: px + pw_ * v  # noqa: E731
    fy = lambda v: py + ph - ph * v / 35.0  # noqa: E731
    axes(f, px, py, pw_, ph, [0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 10, 20, 30], ["0", "0.2", "0.4", "0.6", "0.8", "1"], ["0", "10", "20", "30"],
         fx, fy, "relative z in nucleus (0 = first plane)")
    ylabel(f, px - 6.6, py + ph / 2, "median angle (°)")
    for m in ("2D", "S1"):
        band(f, xc, zp[m]["q1"], zp[m]["q3"], fx, fy, MC[m], 0.14)
    for m in ("2D", "S1"):
        line(f, [(fx(a), fy(b)) for a, b in zip(xc, zp[m]["med"])], MC[m], 1.6)
        for a, b in zip(xc, zp[m]["med"]):
            marker(f, fx(a), fy(b), "o" if m == "S1" else "s", MC[m])
    T(f, px + 1.5, py + 2.4, "2D control", F_L, 700, c=MC_TXT["2D"])
    T(f, px + 1.5 + tw("2D control", F_L, True) + 2.0, py + 2.4, "S1", F_L, 700, c=MC_TXT["S1"])
    T(f, px + pw_, py + ph + 9.0, f"median over nuclei, band = IQR · E15_Ab2 · {MJ['nuclei_used']} nuclei · single volume, descriptive",
      F_N, c=MUTE, a="end")
    f.prov_add(f"b: per-nucleus z-profile of the median angle to the GT-derived 3D flow (10 bins of relative z; line = median over {MJ['nuclei_used']} nuclei, band = IQR)",
               "code of figures/assets_viz/scripts/v2_detail2.py V2-k2 recomputed in make_supp_assets.py M (values identical to V2-k2)")
    # ---------------- c
    yc = 51.0
    letter(f, 0, yc, "c", "F1 across IoU thresholds (gt protocol, 12 datasets, five seeds)")
    gth = np.round(0.5 + 0.05 * np.arange(10), 2)
    cx, cy_, cwid, chh = 12.0, yc + 6.0, 80.0, 38.0
    fx = lambda v: cx + cwid * (v - 0.48) / 0.49  # noqa: E731
    fy = lambda v: cy_ + chh - chh * v / 0.8  # noqa: E731
    xt = [0.5, 0.6, 0.7, 0.8, 0.9]
    axes(f, cx, cy_, cwid, chh, xt, [0, 0.2, 0.4, 0.6, 0.8], [f"{v:.1f}" for v in xt], ["0", "0.2", "0.4", "0.6", "0.8"], fx, fy, "IoU threshold")
    ylabel(f, cx - 8.0, cy_ + chh / 2, "macro F1")
    MK = {"S1": "o", "2D": "s", "base": "^"}
    for m in ("base", "2D", "S1"):
        line(f, [(fx(a), fy(b)) for a, b in zip(gth, s4[m]["curve"])], MC[m], 1.6)
        for a, b in zip(gth, s4[m]["curve"]):
            marker(f, fx(a), fy(b), MK[m], MC[m])
    lx, ly = cx + cwid - 34.0, cy_ + 3.0
    for i, (m, lab) in enumerate((("S1", "S1"), ("2D", "2D control"), ("base", "Original Cellpose-SAM"))):
        yy = ly + i * 2.7
        line(f, [(lx, yy - 0.7), (lx + 4.0, yy - 0.7)], MC[m], 1.6); marker(f, lx + 2.0, yy - 0.7, MK[m], MC[m])
        T(f, lx + 5.2, yy, lab, F_L, c=INK)
    # right: S1 - 2D with min-max over paired seeds
    dx0, dw = cx + cwid + 17.0, W - 0.5 - (cx + cwid + 17.0)
    fx2 = lambda v: dx0 + dw * (v - 0.48) / 0.49  # noqa: E731
    fy2 = lambda v: cy_ + chh - chh * (v + 0.005) / 0.05  # noqa: E731
    yt2 = [0, 0.01, 0.02, 0.03, 0.04]
    axes(f, dx0, cy_, dw, chh, xt, yt2, [f"{v:.1f}" for v in xt], ["0", "0.01", "0.02", "0.03", "0.04"], fx2, fy2, "IoU threshold")
    ylabel(f, dx0 - 9.0, cy_ + chh / 2, "S1 − 2D control (macro F1)")
    d_seed = np.array(s4["S1"]["curve_seeds"]) - np.array(s4["2D"]["curve_seeds"])
    band(f, gth, d_seed.min(0), d_seed.max(0), fx2, fy2, MC["S1"], 0.20)
    f.add(f'<line x1="{dx0:.2f}" y1="{fy2(0):.2f}" x2="{dx0 + dw:.2f}" y2="{fy2(0):.2f}" stroke="{MUTE}" stroke-width="{0.5 * LW:.3f}" stroke-dasharray="1.0 0.7"/>')
    line(f, [(fx2(a), fy2(b)) for a, b in zip(gth, s4["S1_minus_2D_curve"])], MC["S1"], 1.6)
    for a, b in zip(gth, s4["S1_minus_2D_curve"]):
        marker(f, fx2(a), fy2(b), "o", MC["S1"])
    T(f, dx0 + dw, cy_ + 2.4, "line: mean of five paired seeds; band: min–max", F_N, c=MUTE, a="end")
    f.prov_add("c: macro F1 over 12 datasets (no mesoSPIM) at IoU 0.50-0.95 for S1, 2D control and original (gt protocol, five seeds), and S1 - 2D control with the min-max over the five paired seed differences",
               "scripts/figs/localdata/errors_results.json gt/S4/all['12 ...'] (md5 01b39857...), as figS_mech_final.py")
    finish(f, "mech", dict(old="figures/final/FigS_mech.pdf (scripts/figs/figS_mech_final.py)"))


if __name__ == "__main__":
    todo = sys.argv[1:] or ["ortho", "Tgrid", "mm7", "aniso", "mech"]
    for k in todo:
        {"ortho": ortho, "Tgrid": Tgrid, "mm7": mm7, "aniso": aniso, "mech": mech}[k]()
