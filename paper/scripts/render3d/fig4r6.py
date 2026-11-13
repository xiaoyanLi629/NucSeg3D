"""R6 Fig. 4 candidates (G4: axial subsampling and noise). Rule: FIG4R6_RULE.md. Style (team-lead 2026-11-10): statistical panels on
white, images and 3D renders on black, vivid error colours from figures/palette_vivid_dark.json (via r6_render.ERRV), diverging ramp
from palette.json. Advanced forms are the main body (team-lead, user request "热力图、星形图"):
  R6-Fig4-A  a degradation grid (xz, k x noise, black) | b k x noise heat maps of S1 - 2D (F1@0.5, F1@0.7)
             c dataset x condition heat maps (6 datasets + macro row x 12 conditions), F1@0.5 and F1@0.7 side by side
             d black band: S1 vs 2D on the same crop at k = 8, no noise (3D renders + xz sections), error colours
  R6-Fig4-B  a, b as A (b side by side) | e radar charts of macro F1 for S1 and 2D over the 12 conditions (common scale)
             c one dataset x condition heat map with both metrics interleaved | d black band as A
Inputs (export/fig4r6/): g4_summary.json (md5 3ea0513f...), fig4r6_perds.json, fig4r6_degrade.npz (fig4r6_export.py on hostA);
export/f5_{S1,2D}_k{1,8}.npz (existing F5 crop re-inference, F5_RULE.md). Run: .venv/bin/python fig4r6.py"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import hashlib
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

import figcheck  # noqa: E402
import r6_render as V  # noqa: E402  (plotter / mesh / shoot / on_black and the vivid colours; not modified)
import rstyle as R  # noqa: E402

for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 1.8, "ytick.major.size": 1.8,
                     "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "axes.labelsize": 6, "axes.titlesize": 6.5,
                     "legend.fontsize": 5.5, "legend.frameon": False, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "text.color": R.FG, "axes.labelcolor": R.FG, "xtick.color": R.FG,
                     "ytick.color": R.FG, "axes.edgecolor": R.MUTED, "lines.linewidth": 0.8, "mathtext.default": "regular"})
HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(HERE, "export"); E4 = os.path.join(E, "fig4r6")
GA = os.path.abspath(os.path.join(HERE, "..", "..", "figures", "gallery", "assets"))
P = R.PALETTE
PV = json.load(open(os.path.join(HERE, "..", "..", "figures", "palette_vivid_dark.json")))
ERRV = PV["errors_vivid_dark"]
G = json.load(open(os.path.join(E4, "g4_summary.json")))
PD = json.load(open(os.path.join(E4, "fig4r6_perds.json")))
Z = np.load(os.path.join(E4, "fig4r6_degrade.npz"))
KS, CS = (1, 2, 4, 8), ("0", "0.25", "0.5")
COND = [(k, c) for c in CS for k in KS]            # 12 conditions, grouped by noise
MET = (("f05", "F1@0.5"), ("f07", "F1@0.7"))
DSN = {"bbbc050": "BBBC050", "celegans": "C. elegans", "nis3d": "NIS3D", "nucverse3d": "NucVerse3D", "purdue": "Purdue", "tapenade": "Tapenade"}
CMAP = LinearSegmentedColormap.from_list("div", P["diverging"])
S1C, D2C = P["models"]["S1"], P["models"]["2D"]
WT, MT = "#e8e8ea", "#a0a0a8"                       # text on black
MM = 1 / 25.4
W, H = 183, 170
DPI = 600
RATIO0 = float(Z["vox0"][0] / Z["vox0"][1])
Z0, Z1 = 33, 102          # k = 1 GT z extent (planes 34-101) +-1 plane; the same physical window for every k
YS = 48                   # central y of the 96-px crop (as F5)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()  # noqa: E731
NUMS = {}


def mm_ax(fig, x, y, w, h, **kw):
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)


def text(fig, x, y, s, **kw):
    kw.setdefault("fontsize", 6); kw.setdefault("va", "top")
    return fig.text(x / W, 1 - y / H, s, **kw)


def letter(fig, x, y, s, color=None):
    text(fig, x, y, s, fontsize=8, fontweight="bold", ha="left", color=color or R.FG)


def black(fig, x, y, w, h):
    fig.add_artist(Rectangle((x / W, 1 - (y + h) / H), w / W, h / H, transform=fig.transFigure, fc="black", ec="none", zorder=-10))


def phys(stack, k, fx):
    """xz section at YS on the k = 1 physical z window (nearest kept plane), nearest replication with fx px per xy pixel."""
    rows = [min(int(round(j / k)), stack.shape[0] - 1) for j in range(Z0, Z1 + 1)]
    sec = stack[rows][:, YS]
    fz = max(int(round(fx * RATIO0)), 1)
    return np.repeat(np.repeat(sec, fz, 0), fx, 1)


LO, HI = np.percentile(Z["img_k1_c0"], (1, 99.8))


def gray(sec):
    g = np.clip((sec.astype(np.float32) - LO) / max(HI - LO, 1e-6), 0, 1)
    return np.repeat(g[..., None], 3, -1)


def outline(lab, w):
    b = find_boundaries(lab, mode="inner")
    return (ndi.binary_dilation(b, iterations=w - 1) & (lab > 0)) if w > 1 else b


def fx_for(w_mm):
    return int(np.ceil(w_mm * MM * DPI / 96))


def tile_frame(ax, col, lw=0.6):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(col); sp.set_linewidth(lw)


# ---------------------------------------------------------------- a: degradation grid (black)
def panel_a(fig, x, y, tw, gap):
    fx = fx_for(tw)
    th = None
    for i, k in enumerate(KS):
        for j, c in enumerate(CS):
            im = gray(phys(Z[f"img_k{k}_c{c}"], k, fx))
            g = phys(Z[f"gt_k{k}"], k, fx)
            im[outline(g, max(fx // 4, 1))] = to_rgb("#ffffff")
            th = tw * im.shape[0] / im.shape[1]
            ax = mm_ax(fig, x + j * (tw + gap), y + i * (th + gap), tw, th)
            ax.imshow(im, interpolation="none", aspect="auto"); tile_frame(ax, "#3a3a3e", 0.4)
            if i == 0:
                text(fig, x + j * (tw + gap) + tw / 2, y - 1.0, "no noise" if c == "0" else f"noise c = {c}", ha="center", va="bottom", color=WT)
            if j == 0:
                text(fig, x - 1.2, y + i * (th + gap) + th / 2, f"k = {k}", ha="right", va="center", color=WT)
    sb = 5 / float(Z["vox0"][1]) / 96 * tw
    xb = x + 2 * (tw + gap) + tw - sb - 1.5; yb = y + 3 * (th + gap) + th - 1.5
    fig.add_artist(Line2D([xb / W, (xb + sb) / W], [1 - yb / H] * 2, color="white", lw=1.0, transform=fig.transFigure))
    text(fig, xb + sb / 2, yb - 0.6, "5 µm", ha="center", va="bottom", fontsize=5.5, color="white",
         bbox=dict(fc="black", ec="none", pad=0.6))
    return y + 4 * th + 3 * gap


# ---------------------------------------------------------------- d: S1 vs 2D at k = 8 (black)
def classify(m, k):
    f5 = np.load(os.path.join(E, f"f5_{m}_k{k}.npz"))
    gt, pm = f5["gt"], f5["pred"]
    assert np.array_equal(gt, Z[f"gt_k{k}"])
    cls, missed, _ = R.classify_preds(gt, pm)
    return gt, pm, float(f5["ratio"]), cls, missed


def xz_overlay(m, k, fx):
    gt, pm, _, cls, missed = classify(m, k)
    im = gray(phys(Z[f"img_k{k}_c0"], k, fx)) * 0.8
    ps, gs = phys(pm, k, fx), phys(gt, k, fx)
    for lab in np.unique(ps)[1:]:
        msk = ps == lab
        im[msk] = im[msk] * 0.35 + np.array(to_rgb(ERRV[cls.get(int(lab), "other")])) * 0.65
    for gi in missed:
        msk = gs == gi
        im[msk] = im[msk] * 0.35 + np.array(to_rgb(ERRV["miss"])) * 0.65
    w = max(fx // 4, 1)
    im[outline(ps, w)] *= 0.25
    im[outline(gs, w)] = to_rgb("#ffffff")
    return im, {kk: sum(v == kk for v in cls.values()) for kk in ("match", "boundary", "split", "merge", "other")} | {"miss": len(missed)}


def render3d(m, k, px):
    cache = os.path.join(HERE, "out", f"fig4r6_nobox_{m}_k{k}_{px}.png")
    if os.path.exists(cache):
        return Image.open(cache)
    gt, pm, ratio, cls, missed = classify(m, k)
    zz = np.nonzero((gt > 0).any((1, 2)))[0]; sl = slice(max(zz.min() - 1, 0), zz.max() + 2)
    gt, pm = gt[sl], pm[sl]
    p = V.plotter(px)
    s, bl = R.label_surface(pm, ratio)
    kk = np.array([cls.get(int(b), "other") for b in bl])
    for c_ in ("match", "boundary", "split", "merge", "other"):
        sel = np.nonzero(kk == c_)[0]
        if len(sel):
            sub = s.extract_cells(sel).extract_surface(algorithm="dataset_surface")
            V.mesh(p, sub, np.tile(pv.Color(ERRV[c_]).float_rgb, (sub.n_cells, 1)), V.MATCH_OP if c_ == "match" else 1.0)
    if missed:
        gs, _ = R.label_surface(gt, ratio, missed)
        p.add_mesh(gs, color=ERRV["miss"], opacity=V.GHOST_OP, smooth_shading=True, **V.MAT)
    R.ortho_camera(p, 35, 25)
    im = V.shoot(p)
    a = np.asarray(im)[..., 3]; ys, xs = np.nonzero(a > 8); mg = int(0.01 * px)
    im = im.crop((max(xs.min() - mg, 0), max(ys.min() - mg, 0), min(xs.max() + mg, a.shape[1]), min(ys.max() + mg, a.shape[0])))
    im.save(cache)
    return im


def panel_d(fig, x, y, h, gap):
    """Four black tiles: 2D 3D, S1 3D, 2D xz, S1 xz (k = 8, no noise). Returns right edge (mm) and counts."""
    cnt = {}
    rw = h * 1.2
    xx = x
    for m, lab, col in (("2D", R.LAB["2D"], D2C), ("S1", R.LAB["S1"], S1C)):
        ax = mm_ax(fig, xx, y + 4, rw, h - 4)
        px = (V.mm_px(rw, DPI), V.mm_px(h - 4, DPI))
        tile, _ = V.on_black(render3d(m, 8, 2 * max(px)), px)
        ax.imshow(np.asarray(tile), interpolation="none", aspect="auto"); tile_frame(ax, col, 0.8)
        text(fig, xx + rw / 2, y + 3.4, f"{lab}, 3D", ha="center", va="bottom", fontweight="bold", color=col)
        xx += rw + gap
    for m, lab, col in (("2D", R.LAB["2D"], D2C), ("S1", R.LAB["S1"], S1C)):
        fx = fx_for(h * 0.62 * 96 / 72)
        im, c_ = xz_overlay(m, 8, fx); cnt[f"{m}_k8"] = c_
        th = h - 4; tw = th * im.shape[1] / im.shape[0]
        ax = mm_ax(fig, xx, y + 4, tw, th)
        ax.imshow(im, interpolation="none", aspect="auto"); tile_frame(ax, col, 0.8)
        text(fig, xx + tw / 2, y + 3.4, f"{lab}, xz", ha="center", va="bottom", fontweight="bold", color=col)
        xx += tw + gap
    return xx, cnt


def err_legend(fig, x, y, ncol, dark=True):
    hs = [Patch(fc=ERRV[k], ec="none", label=P["error_names"][k]) for k in ("match", "boundary", "split", "merge", "other", "miss")]
    hs.append(Patch(fc="black", ec="white", lw=0.6, label="GT outline (xz)"))
    lg = fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(x / W, 1 - y / H), ncol=ncol, fontsize=5.5, handlelength=1.2,
                    handleheight=0.8, columnspacing=0.8, borderaxespad=0, labelspacing=0.35, labelcolor=WT if dark else R.FG)
    return lg


# ---------------------------------------------------------------- b: k x noise heat maps
def cell_text(v, nd=3):
    return f"{v:+.{nd}f}".replace("-", "−")


def heat_kn(fig, x, y, w, h, met, lab, vmax):
    ax = mm_ax(fig, x, y, w, h)
    M = np.array([[G["conditions"][f"z{k}_n{c}"][f"all_{met}"]["diff"][0] for c in CS] for k in KS])
    Rg = [[G["conditions"][f"z{k}_n{c}"][f"all_{met}"]["diff"][1:] for c in CS] for k in KS]
    ax.imshow(M, cmap=CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(4):
        for j in range(3):
            v = M[i, j]; lo, hi = Rg[i][j]; ex = lo > 0 or hi < 0
            ax.text(j, i, cell_text(v), ha="center", va="center", fontsize=6, color="white" if abs(v) > 0.62 * vmax else R.FG,
                    fontweight="bold" if ex else "normal")
            if ex:
                ax.add_patch(Rectangle((j - 0.45, i - 0.44), 0.9, 0.88, fill=False, ec=R.FG, lw=0.7))
            NUMS[f"b/{met}/k{KS[i]}/c{CS[j]}"] = dict(mean=float(v), seed_range=[float(lo), float(hi)], framed=bool(ex))
    ax.set_xticks(range(3), ["0", "0.25", "0.5"]); ax.set_yticks(range(4), [str(k) for k in KS])
    ax.set_xlabel("Noise c (× nuclear contrast)", labelpad=1.5); ax.set_ylabel("Axial subsampling k", labelpad=1.5)
    ax.tick_params(length=0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Δ macro {lab} (S1 − 2D)", fontsize=6.5, pad=2.5, fontweight="bold")
    return ax


def cbar(fig, x, y, h, vmax, label=None):
    cax = mm_ax(fig, x, y, 1.4, h)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(-vmax, vmax)), cax=cax)
    cb.set_ticks([-vmax, 0, vmax], labels=[cell_text(-vmax, 2), "0", cell_text(vmax, 2)])
    cb.ax.tick_params(labelsize=5, length=1.2, width=0.4, pad=1); cb.outline.set_linewidth(0.4)
    if label:
        cb.set_label(label, fontsize=5.5, labelpad=1.5)


def vmax_macro(met):
    return float(np.ceil(max(abs(G["conditions"][f"z{k}_n{c}"][f"all_{met}"]["diff"][0]) for k in KS for c in CS) * 100) / 100)


# ---------------------------------------------------------------- c: dataset x condition heat maps
def big_rows(met):
    rows = []
    for ds in PD["datasets"]:
        rows.append((f"{DSN[ds]} ({PD['conditions']['z1_n0']['n_vol'][ds]})", [PD["conditions"][f"z{k}_n{c}"][met][ds]["diff"] for k, c in COND], False))
    rows.append(("Macro (6 sets)", [G["conditions"][f"z{k}_n{c}"][f"all_{met}"]["diff"] for k, c in COND], True))
    return rows


def heat_big(fig, x, y, w, h, blocks, vmax, show_cond=True, nd=2, fs=5.2, labels=True):
    """blocks: list of (block label or None, met, rows). Column groups = noise level; within group k = 1, 2, 4, 8."""
    allrows = []
    for bl, met, rows in blocks:
        for r in rows:
            allrows.append((bl, met, *r))
    n = len(allrows)
    ax = mm_ax(fig, x, y, w, h)
    ax.set_xlim(-0.5, 11.5 + 0.0); ax.set_ylim(n - 0.5, -0.5)
    ax.axis("off")
    for i, (bl, met, lab, vals, macro) in enumerate(allrows):
        for j, d in enumerate(vals):
            v, lo, hi = d; ex = lo > 0 or hi < 0
            xj = j + 0.18 * (j // 4)
            ax.add_patch(Rectangle((xj - 0.47, i - 0.45), 0.94, 0.9, fc=CMAP((v + vmax) / (2 * vmax)), ec="none"))
            ax.text(xj, i, cell_text(v, nd), ha="center", va="center", fontsize=fs, color="white" if abs(v) > 0.62 * vmax else R.FG,
                    fontweight="bold" if macro else "normal")
            if False:  # round-7: no frames in the dataset x condition map (team-lead 2026-11-10); same-sign flag kept in provenance
                ax.add_patch(Rectangle((xj - 0.44, i - 0.42), 0.88, 0.84, fill=False, ec=R.FG, lw=0.55))
            k, c = COND[j]
            NUMS[f"c/{met}/{'macro' if macro else lab}/k{k}/c{c}"] = dict(mean=float(v), seed_range=[float(lo), float(hi)], framed=bool(ex))
        if labels:
            ax.text(-0.62, i, lab, ha="right", va="center", fontsize=5.5, fontweight="bold" if macro else "normal")
        if macro and i < n - 1:
            ax.axhline(i + 0.5, color=R.MUTED, lw=0.4)
    ax.set_xlim(-0.5, 11.5 + 0.36)
    if show_cond:
        for j, (k, c) in enumerate(COND):
            xj = j + 0.18 * (j // 4)
            ax.text(xj, -0.62, f"k = {k}", ha="center", va="bottom", fontsize=5.2, color=R.MUTED)
        for g, c in enumerate(CS):
            xc = g * 4 + 1.5 + 0.18 * g
            ax.text(xc, -1.55, "no noise" if c == "0" else f"noise {c}", ha="center", va="bottom", fontsize=5.8)
            ax.plot([xc - 1.9, xc + 1.9], [-1.45, -1.45], color=R.MUTED, lw=0.4, clip_on=False)
    return ax


# ---------------------------------------------------------------- e: radar (B only)
def radar(fig, x, y, s, met, lab, rlim):
    ax = fig.add_axes([x / W, 1 - (y + s) / H, s / W, s / H], projection="polar")
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    for key, col, nm in (("D2", D2C, "2D"), ("S1", S1C, "S1")):
        v = [G["conditions"][f"z{k}_n{c}"][f"all_{met}"][key][0] for k, c in COND]
        NUMS[f"e/{met}/{nm}"] = v
        a = np.r_[ang, ang[:1]]; vv = np.r_[v, v[:1]]
        ax.plot(a, vv, color=col, lw=0.8); ax.fill(a, vv, color=col, alpha=0.12, lw=0)
    ax.set_ylim(*rlim); ax.set_yticks(np.linspace(rlim[0], rlim[1], 3)[1:])
    ax.set_yticklabels([f"{t:.2f}" for t in np.linspace(rlim[0], rlim[1], 3)[1:]], fontsize=4.8, color=R.MUTED)
    ax.set_xticks(ang, [f"k{k}\n{'·' if c == '0' else c}" for k, c in COND], fontsize=4.8)
    ax.tick_params(pad=0)
    ax.grid(lw=0.3, color=P["rule"]); ax.spines["polar"].set_linewidth(0.4); ax.spines["polar"].set_color(R.MUTED)
    ax.set_title(f"macro {lab}", fontsize=6, pad=7)
    return ax


def dumbbell(fig, x, y, w, h, met, lab, ylabels=True):
    ax = mm_ax(fig, x, y, w, h)
    for i, (k, c) in enumerate(COND):
        e = G["conditions"][f"z{k}_n{c}"][f"all_{met}"]
        yy = i + 0.35 * (i // 4)
        ax.plot([e["S1"][0], e["D2"][0]], [yy, yy], color=R.MUTED, lw=0.6, zorder=1)
        ax.plot(e["base"], yy, "^", color=P["models"]["base"], ms=2.2, mew=0, zorder=2)
        ax.plot(e["D2"][0], yy, "s", color=D2C, ms=2.2, mew=0, zorder=3)
        ax.plot(e["S1"][0], yy, "o", color=S1C, ms=2.4, mew=0, zorder=4)
        NUMS[f"e/{met}/k{k}/c{c}"] = dict(S1=e["S1"][0], D2=e["D2"][0], base=e["base"])
    yt = [i + 0.35 * (i // 4) for i in range(12)]
    ax.set_yticks(yt, [f"k = {k}, c = {c}" if ylabels else "" for k, c in COND], fontsize=5)
    ax.set_ylim(12 + 0.35 * 2, -0.8); ax.tick_params(axis="y", length=0, pad=1)
    ax.set_xlim(0.0, 0.72 if met == "f05" else 0.52); ax.set_xticks([0, 0.2, 0.4, 0.6] if met == "f05" else [0, 0.2, 0.4])
    ax.grid(axis="x", lw=0.25, color=P["rule"]); ax.tick_params(axis="x", labelsize=5, pad=1)
    ax.set_xlabel(f"macro {lab}", labelpad=1)
    for g in (1, 2):
        ax.axhline(4 * g - 0.5 + 0.35 * (g - 0.5), color=P["rule"], lw=0.5)
    return ax


# ---------------------------------------------------------------- save
def save(fig, cid, extra):
    rep = figcheck.check(fig)
    fn = os.path.join(GA, f"{cid}.png")
    fig.savefig(fn, dpi=DPI, facecolor="white"); fig.savefig(fn.replace(".png", ".pdf"), dpi=DPI, facecolor="white")
    src = {p: md5(os.path.join(E4, p)) for p in ("g4_summary.json", "fig4r6_perds.json", "fig4r6_degrade.npz", "fig4r6_export_log.json")}
    src.update({f"f5_{m}_k{k}.npz": md5(os.path.join(E, f"f5_{m}_k{k}.npz")) for m in ("S1", "2D") for k in (1, 8)})
    prov = {"figure": cid, "rule": "scripts/render3d/FIG4R6_RULE.md", "rule_md5": md5(os.path.join(HERE, "FIG4R6_RULE.md")),
            "script": "scripts/render3d/fig4r6.py", "script_md5": md5(__file__), "inputs_md5": src,
            "palettes": {"palette.json": md5(os.path.join(HERE, "..", "..", "figures", "palette.json")),
                         "palette_vivid_dark.json": md5(os.path.join(HERE, "..", "..", "figures", "palette_vivid_dark.json"))},
            "server_sources": {"g4_summary.json": f"{_NSP_SCRATCH}/nc/gpu/g4/g4_summary.json (md5 3ea0513f2de64cc9dc7e24e8233a1c5c); same values as NUMBERS.md G4 and final/Fig4.provenance.json",
                               "per-dataset": f"{_NSP_SCRATCH}/nc/figs/fig4r6/fig4r6_perds.json (fig4r6_export.py, g4_summary.py pick logic; 6-dataset macro reproduces g4_summary to 2.2e-16)",
                               "degradation": "gx_eval.degrade imported on hostA (degrade() identical to the pre-M1 file used by the G4 runs); same noise stream",
                               "masks": "existing F5 crop re-inference (F5_RULE.md, f5_infer.py), seed 0, rule-T threshold of the G4 run; colouring only, no metric"},
            "frame_rule": "cell framed when all three paired seed differences S1 - 2D have the same sign (diff range excludes 0); descriptive, no test",
            "export_log": json.load(open(os.path.join(E4, "fig4r6_export_log.json"))),
            "numbers_drawn": NUMS, "size_mm": [W, H], "dpi": DPI, "overlap_and_font_check": rep,
            "generative_ai": "none", "notes": extra}
    json.dump(prov, open(fn.replace(".png", ".provenance.json"), "w"), indent=1, ensure_ascii=False, default=str)
    plt.close(fig)
    print("saved", cid, "check:", rep)


def top_band(fig, b_side_by_side):
    """a (black, left) + b (white, right). Returns bottom y of the band."""
    black(fig, 0, 0, 88, 90)
    letter(fig, 1.5, 2, "a", "white")
    text(fig, 6.5, 2.2, "One test volume, degraded as in G4 (xz)", fontsize=6.5, fontweight="bold", color=WT)
    ya = panel_a(fig, 12, 11, 23.5, 0.8)
    text(fig, 3, ya + 1.4, "C. elegans test crop (median rule); every k-th plane kept;\nnoise σ = c × (nucleus − background median); white: GT",
         fontsize=5.3, color=MT, ha="left")
    letter(fig, 91, 2, "b")
    if b_side_by_side:
        for i, (met, lab) in enumerate(MET):
            heat_kn(fig, 99 + i * 43, 9, 33, 30, met, lab, vmax_macro(met)); cbar(fig, 99 + i * 43 + 34.2, 9, 30, vmax_macro(met))
        text(fig, 91, 48.5, "6 datasets, 20 volumes, 3 paired seeds; red: S1 better; framed / bold: all three seed\n"
             "differences share the sign (descriptive, no test)", fontsize=5.3, color=R.MUTED, ha="left")
        return 90
    for i, (met, lab) in enumerate(MET):
        heat_kn(fig, 106, 9 + i * 40, 50, 30, met, lab, vmax_macro(met)); cbar(fig, 158, 9 + i * 40, 30, vmax_macro(met))
    text(fig, 168, 9, "6 datasets\n20 volumes\n3 paired\nseeds\n\nred: S1\nbetter\n\nframed,\nbold: all\nthree seed\ndifferences\nshare the\nsign\n(descriptive,\nno test)",
         fontsize=5.2, color=R.MUTED, ha="left")
    return 90


def bottom_band(fig, y0, let="d"):
    black(fig, 0, y0, W, H - y0)
    letter(fig, 1.5, y0 + 1.5, let, "white")
    text(fig, 6.5, y0 + 1.7, "Same crop at k = 8, no noise: S1 vs 2D control (seed 0)", fontsize=6.5, fontweight="bold", color=WT)
    xr, cnt = panel_d(fig, 6, y0 + 6.5, H - y0 - 8.5, 2)
    err_legend(fig, xr + 3, y0 + 8.5, 2)
    text(fig, xr + 3, y0 + 19.5, "colour: error class of each predicted nucleus\nvs. the subsampled GT (one crop, no metric)",
         fontsize=5.3, color=MT, ha="left")
    return cnt


RADAR_NOTE = ("A radar (12 condition axes, S1 and 2D macro F1 polygons on one common radial scale 0.1-0.7) was drawn and rejected: "
              "the S1 - 2D differences (at most 0.072) are 1-2 % of the radius, so the two polygons coincide and no difference can be read, "
              "while the same information is legible in the heat maps; the test render is R6-Fig4-radar-rejected.png. "
              "In B the absolute levels are shown as a dumbbell plot instead.")


def radar_test():
    NUMS.clear()
    fig = plt.figure(figsize=(90 * MM, 45 * MM), facecolor="white")
    global W, H
    W0, H0 = W, H; W, H = 90, 45
    for i, (met, lab) in enumerate(MET):
        radar(fig, 8 + i * 45, 9, 28, met, lab, (0.1, 0.7))
    hs = [Line2D([], [], color=S1C, lw=0.8, label="S1"), Line2D([], [], color=D2C, lw=0.8, label="2D control")]
    fig.legend(handles=hs, loc="lower center", ncol=2, fontsize=5.5)
    fig.savefig(os.path.join(GA, "R6-Fig4-radar-rejected.png"), dpi=DPI, facecolor="white"); plt.close(fig)
    W, H = W0, H0


def version_a():
    NUMS.clear()
    fig = plt.figure(figsize=(W * MM, H * MM), facecolor="white")
    top_band(fig, False)
    letter(fig, 1.5, 93, "c")
    text(fig, 6.5, 93.2, "Dataset × condition: Δ macro F1 (S1 − 2D) per dataset; 12 conditions = 3 noise levels × 4 subsampling factors",
         fontsize=6.5, fontweight="bold")
    v = max(abs(d[0]) for met, _ in MET for _, vals, _ in big_rows(met) for d in vals)
    vmax = float(np.ceil(v * 20) / 20)
    for i, (met, lab) in enumerate(MET):
        heat_big(fig, 24 + i * 76, 105, 71, 26, [(None, met, big_rows(met))], vmax, labels=i == 0)
        text(fig, 24 + i * 76 + 35.5, 132.6, lab, ha="center", fontsize=6.5, fontweight="bold")
    cbar(fig, 173.5, 105, 26, vmax)
    text(fig, 3, 136.5, "(n) = volumes", fontsize=5.2, color=R.MUTED, ha="left")
    cnt = bottom_band(fig, 141)
    save(fig, "R6-Fig4-A", dict(radar_decision=RADAR_NOTE, c_frames="no frames in the dataset x condition maps (c); 'framed' in numbers_drawn records only whether the three seed differences share the sign", version="A: k x noise heat maps stacked (b), dataset x condition heat maps per metric side by side (c), "
                                         "black band d (3D + xz at k = 8). Radar not used in A. The macro line panels of the old Fig. 4a-b are "
                                         "superseded by the heat maps (values identical) and can move to the supplement.",
                                 d_error_counts_on_crop=cnt))


def version_b():
    NUMS.clear()
    fig = plt.figure(figsize=(W * MM, H * MM), facecolor="white")
    top_band(fig, True)
    letter(fig, 91, 53, "c")
    text(fig, 96, 53.2, "Absolute macro F1 in the 12 conditions (S1 vs 2D)", fontsize=6.5, fontweight="bold")
    for i, (met, lab) in enumerate(MET):
        dumbbell(fig, 111 + i * 37, 62, 33, 26, met, lab, ylabels=i == 0)
    hs = [Line2D([], [], color=S1C, marker="o", ms=2.4, lw=0, label="S1"), Line2D([], [], color=D2C, marker="s", ms=2.2, lw=0, label="2D control"),
          Line2D([], [], color=P["models"]["base"], marker="^", ms=2.2, lw=0, label="original")]
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(110 / W, 1 - 57 / H), ncol=3, fontsize=5.3, handletextpad=0.2, borderaxespad=0,
               columnspacing=1.2)
    letter(fig, 1.5, 93, "d")
    text(fig, 6.5, 93.2, "Dataset × condition: Δ (S1 − 2D) per dataset, both metrics; (n) = volumes", fontsize=6.5, fontweight="bold")
    v = max(abs(d[0]) for met, _ in MET for _, vals, _ in big_rows(met) for d in vals)
    vmax = float(np.ceil(v * 20) / 20)
    # interleaved: one row per dataset x metric
    allr = []
    for ds_i in range(7):
        for mi, (met, lab) in enumerate(MET):
            br = big_rows(met)[ds_i]
            allr.append((br[0] if mi == 0 else "", lab, br[1], br[2], met))
    n = len(allr)
    ax = mm_ax(fig, 40, 105, 132, 34)
    ax.set_xlim(-0.5, 11.86); ax.set_ylim(n - 0.5, -0.5); ax.axis("off")
    for i, (dsl, ml, vals, macro, met) in enumerate(allr):
        for j, d in enumerate(vals):
            v_, lo_, hi_ = d; ex = lo_ > 0 or hi_ < 0; xj = j + 0.18 * (j // 4)
            ax.add_patch(Rectangle((xj - 0.47, i - 0.44), 0.94, 0.88, fc=CMAP((v_ + vmax) / (2 * vmax)), ec="none"))
            ax.text(xj, i, cell_text(v_, 2), ha="center", va="center", fontsize=5.0, color="white" if abs(v_) > 0.62 * vmax else R.FG,
                    fontweight="bold" if macro else "normal")
            if False:  # no frames here (team-lead 2026-11-10)
                ax.add_patch(Rectangle((xj - 0.44, i - 0.40), 0.88, 0.80, fill=False, ec=R.FG, lw=0.5))
            k, c = COND[j]
            NUMS[f"c/{met}/{'macro' if macro else dsl or allr[i - 1][0]}/k{k}/c{c}"] = dict(mean=float(v_), seed_range=[float(lo_), float(hi_)], framed=bool(ex))
        ax.text(-0.62, i, ml, ha="right", va="center", fontsize=5, color=R.MUTED)
        if dsl:
            ax.text(-2.15, i + 0.5, dsl, ha="right", va="center", fontsize=5.5, fontweight="bold" if macro else "normal")
        if i % 2 == 1 and i < n - 1:
            ax.axhline(i + 0.5, color="white", lw=0.8)
    ax.axhline(n - 2.5, color=R.MUTED, lw=0.4)
    for j, (k, c) in enumerate(COND):
        ax.text(j + 0.18 * (j // 4), -0.62, f"k = {k}", ha="center", va="bottom", fontsize=5.2, color=R.MUTED)
    for g, c in enumerate(CS):
        xc = g * 4 + 1.5 + 0.18 * g
        ax.text(xc, -1.7, "no noise" if c == "0" else f"noise {c}", ha="center", va="bottom", fontsize=5.8)
        ax.plot([xc - 1.9, xc + 1.9], [-1.6, -1.6], color=R.MUTED, lw=0.4, clip_on=False)
    cbar(fig, 175, 105, 34, vmax)
    cnt = bottom_band(fig, 143, "e")
    remap = {k: v for k, v in NUMS.items()}
    NUMS.clear()
    NUMS.update({({"e/": "c/", "c/": "d/"}.get(k[:2], k[:2]) + k[2:]): v for k, v in remap.items()})
    save(fig, "R6-Fig4-B", dict(radar_decision=RADAR_NOTE, panel_letters="a degradation grid, b k x noise heat maps, c dumbbell (absolute macro F1), "
                                         "d dataset x condition heat map, e black band (renders + xz); numbers_drawn keys follow these letters", c_frames="no frames in the dataset x condition map; 'framed' in numbers_drawn "
                                         "records only whether the three seed differences share the sign",
                                 version="B: k x noise heat maps side by side (b), dumbbell of absolute macro F1 (c), one interleaved dataset x metric x "
                                         "condition heat map (d), black band e. Old macro line panels move to the supplement.",
                                 d_error_counts_on_crop=cnt))


if __name__ == "__main__":
    radar_test()
    version_a()
    version_b()
