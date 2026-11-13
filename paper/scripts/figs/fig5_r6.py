"""Fig. 5, round 6 (gallery R6-Fig5-A / R6-Fig5-B): datasets withheld from training (L1) and data efficiency (L2).
Only existing L1 / L2 results; no new statistic. Style as the finalised Fig. 2 / Fig. 3 (R7): statistics on white,
renders and images on black with the vivid palette (figures/palette_vivid_dark.json via render3d/r6_render.py).

Sources (copied read-only from the server into render3d/export/r6f5/, md5 recorded in the provenance):
  l1s_vol.json      per-volume counts of every L1 / production run, gt protocol ($NUCSEG3D_SCRATCH/nc/heldout/l1cmp/)
  l1s_ana.out       registered L1 analysis output (paired seeds 0/1000/2000; confirmation rule); used for the cross-check below
  l2_cmp.json       L2 pooled F1 (rule T 'r2'; original 'fix'), seed 0, 1,000 fine-tuning steps; production seeds 0-4
  r6f5_l1vol.json   per-volume F1 recomputed from l1s_vol.json counts (r6_fig5_export.py)
  r6f5_select.json  representative volumes (rule render3d/R6_FIG5_RULE.md, written before selection)
  r6f5_<ds>.npz     evaluated final masks of the two S1 runs on those volumes (r6_fig5_export.py)
Pooled F1 = 2 tp / (n_gt + n_pred) summed over the dataset's volumes (as l1s_ana.py / fig5_heldout.py).
Radar chart considered (team lead 2026-11-10) and not drawn: of the suggested axes, nucleus-volume bias and diameter
sensitivity were never measured for the retrained models; only F1@0.5, F1@0.7 and counts exist, so the comparison is
shown as heatmaps and slope charts instead (recorded in the provenance).
Usage: FIG5_LAYOUT=A|B python fig5_r6.py
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import hashlib
import json
import os
import re
import sys

os.environ["NC_PALETTE"] = "npg"
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
import shutil  # noqa: E402
shutil.copy(os.path.join(HERE, "..", "..", "figures", "palette.json"), os.path.join(HERE, "palette.json"))
import nc_style as S  # noqa: E402
from nc_style import plt  # noqa: E402

RD = os.path.join(HERE, "..", "render3d")
sys.path.insert(0, RD)
import figcheck  # noqa: E402
import r6_fig5_render as F  # noqa: E402
import r6_render as V  # noqa: E402

S.setup()
from matplotlib import font_manager  # noqa: E402
for f_ in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
    if os.path.exists(f_):
        font_manager.fontManager.addfont(f_)
plt.rcParams.update({"font.sans-serif": ["Arial", "DejaVu Sans"]})
P = S.PALETTE
FG, MUT, RULE = P["text"], P["muted"], P["rule"]
CS1, C2D, CB = S.C["S1"], S.C["2D"], S.C["base"]
DIV = LinearSegmentedColormap.from_list("npgdiv", P["diverging"])
EXP = os.path.join(RD, "export", "r6f5")
GA = os.path.abspath(os.path.join(HERE, "..", "..", "figures", "gallery", "assets"))
SRC = {k: os.path.join(EXP, v) for k, v in dict(l1vol="l1s_vol.json", l1ana="l1s_ana.out", l2="l2_cmp.json", pervol="r6f5_l1vol.json",
                                                    select="r6f5_select.json").items()}
MD5 = {k: hashlib.md5(open(v, "rb").read()).hexdigest() for k, v in SRC.items()}
VOL = json.load(open(SRC["l1vol"]))
L2 = json.load(open(SRC["l2"]))
PV = json.load(open(SRC["pervol"]))
SEL = json.load(open(SRC["select"]))
SETS = ("bbbc050", "blastospim", "nucverse3d")
SS = (0, 1000, 2000)
SS5 = (0, 1000, 2000, 3000, 4000)
KS = (1, 2, 5, 10)
MOD = {"bbbc050": "spinning-disk confocal", "blastospim": "light-sheet", "nucverse3d": "mainly multiphoton"}
NM = dict(S.DS_NAME)
LAY = os.environ.get("FIG5_LAYOUT", "A")
CID = f"R6-Fig5-{LAY}"
W = 183.0
H = 170.0


# ------------------------------------------------------------------ data
def grp(g):
    if g == "liver":
        return lambda k: k.startswith("nucverse3d__") and "Drosophila" not in k
    if g == "fly":
        return lambda k: k.startswith("nucverse3d__") and "Drosophila" in k
    return lambda k: k.startswith(g + "__")


def pooled(key, g, i):
    rule = "fix" if key.endswith("base") else "r2"
    vs = [v[rule] for k, v in VOL[f"tGT|{key}"].items() if grp(g)(k)]
    return 2 * sum(x["tp7" if i else "tp5"] for x in vs) / max(sum(x["n_gt"] + x["n_pred"] for x in vs), 1)


def G(key, g, i):
    return float(np.mean([pooled(key, s, i) for s in SETS])) if g == "macro" else pooled(key, g, i)


def seeds(n, g, i):
    return [G(f"{n}_s{s}", g, i) for s in SS]


def confirmed(d):
    m, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    return abs(m) > 2 * sd and abs(m) >= 0.01


# cross-check against the registered analysis output (l1s_ana.out): every "损失 S1 / 损失二维" mean must match to 3 decimals
ANA = open(SRC["l1ana"], encoding="utf-8").read().split("== tFP")[0]
CHK = []
for m_ in re.finditer(r"^ (\S+)\s+F1@(0\.[57]).*?\n.*?\n\s+损失 S1 ([+-]\d\.\d{3}).*?损失二维 ([+-]\d\.\d{3})", ANA, re.M):
    g = {"宏": "macro", "nv_肝脏": "liver", "nv_果蝇": "fly"}.get(m_.group(1), m_.group(1))
    i = 0 if m_.group(2) == "0.5" else 1
    for n, txt in (("S1", m_.group(3)), ("2D", m_.group(4))):
        v = float(np.mean([a - b for a, b in zip(seeds(f"L1_{n}", g, i), seeds(f"F_{n}", g, i))]))
        assert abs(v - float(txt)) < 0.0006, (g, i, n, v, txt)
        CHK.append((g, i, n, txt))
assert len(CHK) == 24, len(CHK)
# advantage change ("优势变化") for every group and threshold must match l1s_ana.out to 3 decimals
for m_ in re.finditer(r"^ (\S+)\s+F1@(0\.[57]).*?\n.*?优势变化 ([+-]\d\.\d{3})±(\d\.\d{3})（.*?）(\S+)", ANA, re.M):
    g = {"宏": "macro", "nv_肝脏": "liver", "nv_果蝇": "fly"}.get(m_.group(1), m_.group(1))
    i = 0 if m_.group(2) == "0.5" else 1
    d = [(a - b) - (c - e) for a, b, c, e in zip(seeds("L1_S1", g, i), seeds("L1_2D", g, i), seeds("F_S1", g, i), seeds("F_2D", g, i))]
    assert abs(np.mean(d) - float(m_.group(3))) < 0.0006 and abs(np.std(d, ddof=1) - float(m_.group(4))) < 0.0006, (g, i, d, m_.groups())
    assert confirmed(d) == (m_.group(5) != "未确认"), (g, i, m_.group(5))
    CHK.append((g, i, "adv", m_.group(3)))
assert len(CHK) == 36, len(CHK)
# manuscript sentences (main.tex sec:res-heldout)
_adv = lambda g, i: [(a - b) - (c - e) for a, b, c, e in zip(seeds("L1_S1", g, i), seeds("L1_2D", g, i), seeds("F_S1", g, i), seeds("F_2D", g, i))]  # noqa: E731
_r = lambda g, i: [a - b for a, b in zip(seeds("L1_S1", g, i), seeds("L1_2D", g, i))]  # noqa: E731
for (g, i, fn, val, sd) in (("macro", 0, _adv, -0.016, 0.002), ("macro", 1, _adv, -0.034, 0.014), ("macro", 0, _r, -0.007, 0.005),
                            ("macro", 1, _r, -0.015, 0.004), ("nucverse3d", 1, _r, -0.068, 0.008), ("liver", 1, _r, -0.086, 0.008),
                            ("bbbc050", 1, _r, 0.005, 0.027), ("blastospim", 1, _r, 0.019, 0.016), ("blastospim", 0, _r, -0.020, 0.005)):
    v = fn(g, i); assert abs(np.mean(v) - val) < 0.0006 and abs(np.std(v, ddof=1) - sd) < 0.0006, (g, i, v, val)
_l = lambda n, g, i: float(np.mean([a - b for a, b in zip(seeds(f"L1_{n}", g, i), seeds(f"F_{n}", g, i))]))  # noqa: E731
assert abs(_l("S1", "liver", 1) + 0.109) < 0.0006 and abs(_l("2D", "liver", 1) + 0.007) < 0.0006
assert abs(_l("S1", "macro", 1) / _l("2D", "macro", 1) - 1.7) < 0.05

# ------------------------------------------------------------------ figure helpers
fig = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor="white")
prov = {"figure": CID, "sources": {k: {"path": v, "md5": MD5[k]} for k, v in SRC.items()},
        "server_sources": {"l1vol": f"{_NSP_SCRATCH}/nc/heldout/l1cmp/l1s_vol.json", "l2": f"{_NSP_SCRATCH}/nc/heldout/l1cmp/l2_cmp.json",
                           "l1ana": f"{_NSP_SCRATCH}/nc/heldout/l1cmp/l1s_ana.out",
                           "export": f"{_NSP_SCRATCH}/nc/figs/render3d/export/r6f5_* (r6_fig5_export.py)"},
        "cross_check": f"{len(CHK)} values reproduced from l1s_vol.json match l1s_ana.out to 3 decimals (24 per-model losses, 12 "
                       "advantage changes incl. SD and confirmation call); manuscript sec:res-heldout numbers asserted in the script",
        "radar": "not drawn: nucleus-volume bias and diameter sensitivity were never measured for the retrained models (only F1@0.5, "
                 "F1@0.7 and counts exist); heatmaps and slope charts used instead", "panels": {}, "ai_generated_content": "none", "thumbnails_resolution": "raw xy planes embedded at native pixel resolution (interpolation none); no upsampling"}


def ax_mm(x, y, w, h, **kw):
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)


def letter(x, y, s, title=None, color="black", fs_t=6.5):
    fig.text(x / W, 1 - y / H, s, fontsize=8, fontweight="bold", va="top", ha="left", color=color)
    if title:
        fig.text((x + 3.6) / W, 1 - (y + 0.35) / H, title, fontsize=fs_t, fontweight="bold", va="top", ha="left", color=color)


def txt(x, y, s, **kw):
    kw.setdefault("fontsize", 5.5); kw.setdefault("color", FG)
    return fig.text(x / W, 1 - y / H, s, **kw)


def heat(ax, M, rows, cols, vmax, fmt="{:+.3f}", frames=None, hatch=None, fs=5.2, scale=1.0):
    ax.imshow(M, cmap=DIV, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            v = M[r, c]
            if np.isnan(v):
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fc="white", ec=RULE, lw=0.3))
                continue
            ax.text(c, r, fmt.format(v * scale).replace("-", "−"), ha="center", va="center", fontsize=fs, color="white" if abs(v) > 0.62 * vmax else FG)
            if frames is not None and frames[r][c]:
                ax.add_patch(Rectangle((c - 0.45, r - 0.42), 0.9, 0.84, fc="none", ec=FG, lw=0.8))
    ax.set_xticks(range(len(cols)), cols, fontsize=5.3); ax.xaxis.tick_top(); ax.tick_params(length=0, pad=1.5)
    ax.set_yticks(range(len(rows)), rows, fontsize=5.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True); ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", lw=1.0); ax.tick_params(which="minor", length=0)


def colorbar(x, y, w, h, vmax, label, pp=False):
    cax = ax_mm(x, y, w, h)
    cax.imshow(np.linspace(-1, 1, 200)[None], cmap=DIV, aspect="auto", extent=(-vmax, vmax, 0, 1))
    tl = [f"−{vmax * 100:.0f}", "0", f"+{vmax * 100:.0f}"] if pp else [f"−{vmax:.2f}", "0", f"+{vmax:.2f}"]
    cax.set_yticks([]); cax.set_xticks([-vmax, 0, vmax], tl, fontsize=5.0)
    cax.tick_params(length=1.2, pad=1)
    for sp in cax.spines.values():
        sp.set_linewidth(0.3)
    cax.set_xlabel(label, fontsize=5.0, labelpad=1.2, color=MUT)


def yt(ax, n=4):
    """Explicit y ticks inside the limits (steps 1/2/5 x 10^k), two decimals."""
    from matplotlib.ticker import MaxNLocator
    lo, hi = ax.get_ylim()
    t = [v for v in MaxNLocator(n, steps=[1, 2, 5, 10]).tick_values(lo, hi) if lo <= v <= hi]
    ax.set_yticks(t, [f"{v:.2f}".replace("-", "−") for v in t])


# ------------------------------------------------------------------ panel a: design schematic
def panel_a(x, y, w, h, cw):
    letter(x, y, "a", "Experiment design: three datasets withheld, one per imaging modality")
    txt(x + 3.6, y + 3.9, "chosen in advance from data attributes only; tested on all their test volumes (gt protocol)", va="top", fontsize=5.0, color=MUT)
    y0 = y + 7.6
    gap = 1.4; th = h - (y0 - y) - 8.2
    rec = {}
    for j, ds in enumerate(SETS):
        cx = x + j * (cw + gap)
        pl = F.xy_plane(ds)
        im = pl["img"]
        lo, hi = np.percentile(im, (0.5, 99.8)); im = np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)
        a = ax_mm(cx, y0 + 5.0, cw, th)
        a.set_facecolor("black")
        a.imshow(im, cmap="gray", interpolation="none", aspect="equal")
        a.set_xticks([]); a.set_yticks([])
        a.set_xlim(-0.5, im.shape[1] - 0.5); a.set_ylim(im.shape[0] - 0.5, -0.5)
        for sp in a.spines.values():
            sp.set_visible(False)
        vx = pl["vox"][2]
        if ds == "bbbc050":
            L, lab = 20, "20 px"            # voxel size not reported for BBBC050 (manifest: xy recorded as 1 px)
        else:
            L = 10 / vx; lab = "10 µm"
        x1 = im.shape[1] * 0.06; yb = im.shape[0] * 0.93
        a.plot([x1, x1 + L], [yb, yb], color="white", lw=1.0, solid_capstyle="butt")
        a.text(x1, yb - im.shape[0] * 0.05, lab, color="white", fontsize=5.0, ha="left", va="bottom")
        txt(cx + cw / 2, y0 + 0.4, NM[ds], ha="center", va="top", fontsize=6.0, fontweight="bold")
        txt(cx + cw / 2, y0 + 2.9, MOD[ds], ha="center", va="top", fontsize=5.3, color=MUT)
        nt = sum(k.startswith(ds + "__") for k in VOL["tGT|L1_S1_s0"])
        txt(cx + cw / 2, y0 + 5.0 + th + 0.7, f"{nt} test volumes", ha="center", va="top", fontsize=5.2, color=FG)
        rec[ds] = {"plane_z": pl["z"], "volume": SEL[ds]["pick"], "lut": "0.5-99.8 percentile of the plane (display only)",
                   "scalebar": lab, "n_test": nt}
    fx = x + 3 * (cw + gap) + 3.0
    bw = w - (fx - x)

    def box(by, bh, title, sub, col, fc):
        fig.add_artist(FancyBboxPatch((fx / W, 1 - (by + bh) / H), bw / W, bh / H, boxstyle="round,pad=0,rounding_size=0.006",
                                      transform=fig.transFigure, fc=fc, ec=col, lw=0.6))
        txt(fx + 1.3, by + 0.9, title, va="top", fontsize=5.8, fontweight="bold", color=col)
        txt(fx + 1.3, by + 3.5, sub, va="top", fontsize=5.1, color=FG, linespacing=1.2)

    def arrow(x0, y0_, x1, y1, col, ls="-"):
        fig.add_artist(FancyArrowPatch((x0 / W, 1 - y0_ / H), (x1 / W, 1 - y1 / H), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=5, lw=0.6, color=col, linestyle=ls))
    gy = 1.6; bh = (h - (y0 - y) - 2 * gy) / 3
    wide = bw > 70
    T = {"prod": ("training splits of all datasets, incl. these three; " if wide else "training splits of all datasets\n")
         + "2D → A1 → C, X1 → S1 = ½(C + X1); seeds 0–4",
         "ret": ("whole pipeline retrained without the three datasets; " if wide else
                 "whole pipeline retrained without\nthe three; ") + "seeds 0, 1000, 2000, paired by seed",
         "l2": ("add back k = 1, 2, 5, 10 training volumes of one dataset; " if wide else "add back k = 1, 2, 5, 10 volumes of one\ndataset; ")
         + "fine-tune 1,000 steps; seed 0"}
    box(y0, bh, "Production (paper model)", T["prod"], FG, "#f4f4f2")
    box(y0 + bh + gy, bh, "Retrained (L1)", T["ret"], CS1, "#fdf1ee")
    box(y0 + 2 * (bh + gy), bh, "Adding back volumes (L2, exploratory)", T["l2"], MUT, "white")
    arrow(fx + bw * 0.5, y0 + bh + 0.05, fx + bw * 0.5, y0 + bh + gy - 0.05, CS1, "--")
    arrow(fx + bw * 0.5, y0 + 2 * bh + gy + 0.05, fx + bw * 0.5, y0 + 2 * bh + 2 * gy - 0.05, MUT, "--")
    arrow(x + 3 * (cw + gap) - 0.4, y0 + bh + gy + bh / 2, fx - 0.3, y0 + bh + gy + bh / 2, CS1)
    txt(x + 3 * (cw + gap) + 1.3, y0 + bh + gy + bh / 2 - 0.5, "−", ha="center", va="bottom", fontsize=7, fontweight="bold", color=CS1)
    prov["panels"]["a"] = {"thumbnails": rec, "note": "schematic; thumbnails are raw xy planes (plane with most GT foreground) of the "
                           "representative volumes shown as 3D renders"}


# ------------------------------------------------------------------ slope chart production -> retrained
DSM = {"bbbc050": "o", "blastospim": "^", "nucverse3d": "s"}


def panel_slope(x, y, w, h, lt):
    """3D advantage S1 - 2D (pooled F1, gt), production -> retrained, paired seeds 0/1000/2000 (main.tex sec:res-heldout).
    Groups: macro, the three datasets, and the NucVerse3D liver volumes (named in the text). Filled end point = confirmed by the
    registered rule (|mean| > 2 SD and >= 0.01, as l1s_ana.out). NucVerse3D (largest loss of advantage) in the accent colour."""
    letter(x, y, lt, "3D advantage, production → retrained")
    txt(x + 3.6, y + 3.6, "S1 − 2D, pooled F1; mean of 3 paired seeds (thin: each seed)", va="top", fontsize=5.0, color=MUT)
    GR = [("macro", "Macro (3 datasets)", FG, 1.0, "-", "D"), ("bbbc050", "BBBC050", MUT, 0.6, "-", "o"),
          ("blastospim", "BlastoSPIM", MUT, 0.6, "-", "^"), ("nucverse3d", "NucVerse3D", CS1, 1.0, "-", "s"),
          ("liver", "NucVerse3D liver (3 vol.)", CS1, 0.6, (0, (2.5, 1.2)), "s")]
    rec = {}
    pw = (w - 9.0) / 2
    for i in (0, 1):
        ax = ax_mm(x + 9.0 + i * pw, y + 8.5, pw - 12.5, h - 21.0)
        ax.axhline(0, color=RULE, lw=0.6, zorder=0)
        labs, allv = [], []
        for g, lab, col, lw_, ls, mk in GR:
            dp = [a - b for a, b in zip(seeds("F_S1", g, i), seeds("F_2D", g, i))]
            dr = [a - b for a, b in zip(seeds("L1_S1", g, i), seeds("L1_2D", g, i))]
            for a_, b_ in zip(dp, dr):
                ax.plot([0, 1], [a_, b_], color=col, lw=0.25, alpha=0.4, ls=ls)
            mp, mr = float(np.mean(dp)), float(np.mean(dr))
            cp, cr = confirmed(dp), confirmed(dr)
            ax.plot([0, 1], [mp, mr], color=col, lw=lw_, ls=ls, zorder=3)
            for xx, m, c in ((0, mp, cp), (1, mr, cr)):
                ax.plot([xx], [m], marker=mk, ms=2.6 if lw_ > 0.8 else 2.1, mfc=col if c else "white", mec=col, mew=0.6, zorder=4)
            labs.append(ax.text(1.1, mr, f"{mr:+.3f}".replace("-", "−"), fontsize=5.0, va="center", ha="left",
                                color=col, fontweight="bold" if g == "nucverse3d" else "normal"))
            allv += dp + dr
            rec[f"F1@{0.5 if i == 0 else 0.7}/{g}"] = {"production": {"seeds": [round(v, 4) for v in dp], "mean": round(mp, 4),
                                                                      "sd": round(float(np.std(dp, ddof=1)), 4), "confirmed": cp},
                                                       "retrained": {"seeds": [round(v, 4) for v in dr], "mean": round(mr, 4),
                                                                     "sd": round(float(np.std(dr, ddof=1)), 4), "confirmed": cr},
                                                       "change": round(mr - mp, 4)}
        lo, hi = min(allv), max(allv); pad = (hi - lo) * 0.08
        ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(-0.15, 1.05)
        yt(ax, 5)
        labs.sort(key=lambda t: t.get_position()[1]); dmin = 0.085 * (hi - lo + 2 * pad)
        for k in range(1, len(labs)):
            if labs[k].get_position()[1] - labs[k - 1].get_position()[1] < dmin:
                labs[k].set_position((1.1, labs[k - 1].get_position()[1] + dmin))
        ax.set_xticks([0, 1], ["prod.", "retr."], fontsize=5.0)
        ax.set_title(f"F1@{0.5 if i == 0 else 0.7}", fontsize=5.6, pad=2)
        if i == 0:
            ax.set_ylabel("S1 − 2D", fontsize=5.5, labelpad=1.5)
        ax.tick_params(labelsize=5.0, length=1.5, pad=1)
        ax.spines["bottom"].set_linewidth(0.4); ax.spines["left"].set_linewidth(0.4)
    hl = [Line2D([], [], color=c_, lw=l_, ls=s_, marker=m_, ms=2.3, mfc=c_, mec=c_, label=n_) for _, n_, c_, l_, s_, m_ in GR]
    hl.append(Line2D([], [], marker="o", ls="", ms=2.3, mfc="white", mec=FG, mew=0.6, label="open: not confirmed"))
    fig.legend(handles=hl, loc="lower center", bbox_to_anchor=((x + w / 2) / W, 1 - (y + h) / H), ncol=2, fontsize=5.0, frameon=False,
               handletextpad=0.3, columnspacing=0.8, handlelength=1.8, labelspacing=0.25)
    prov["panels"][lt] = {"kind": "slope chart of the 3D advantage S1 - 2D (pooled F1, gt protocol) from production to retrained, "
                          "mean of paired seeds 0/1000/2000 with per-seed lines; filled = confirmed by the registered rule",
                          "text": "main.tex sec:res-heldout: production +0.010/+0.019, retrained -0.007/-0.015, NucVerse3D F1@0.7 -0.068, "
                                  "liver -0.086, BBBC050 +0.005, BlastoSPIM +0.019 (F1@0.5 -0.020, confirmed)",
                          "fly_subset": "NucVerse3D Drosophila volumes (n = 2) not drawn separately; not named in the text; values in panel c",
                          "values": rec}


# ------------------------------------------------------------------ retraining-loss heatmap (3 seeds, confirmation rule)
def panel_loss(x, y, w, h, lt):
    letter(x, y, lt, "Cost of withholding")
    txt(x + 3.6, y + 3.6, "retrained − production, pooled F1", va="top", fontsize=5.0, color=MUT)
    rows = [("macro", "Macro (3)"), ("bbbc050", "BBBC050"), ("blastospim", "BlastoSPIM"), ("nucverse3d", "NucVerse3D"),
            ("liver", "liver (3 vol.)"), ("fly", "fly (2 vol.)")]
    cols = [(0, "S1"), (0, "2D"), (0, "Δ"), (1, "S1"), (1, "2D"), (1, "Δ")]
    M = np.zeros((len(rows), len(cols))); Fr = [[False] * len(cols) for _ in rows]; rec = {}
    loss = lambda n, g, i: [a - b for a, b in zip(seeds(f"L1_{n}", g, i), seeds(f"F_{n}", g, i))]  # noqa: E731
    for r, (g, _) in enumerate(rows):
        for c, (i, n) in enumerate(cols):
            d = loss(n, g, i) if n != "Δ" else [a - b for a, b in zip(loss("S1", g, i), loss("2D", g, i))]
            M[r, c] = np.mean(d); Fr[r][c] = confirmed(d)
            rec[f"{g}/F1@{0.5 if i == 0 else 0.7}/{n}"] = {"seeds": [round(v, 4) for v in d], "mean": round(float(np.mean(d)), 4),
                                                             "sd": round(float(np.std(d, ddof=1)), 4), "confirmed": Fr[r][c]}
    lw = 15.0
    ax = ax_mm(x + lw, y + 12.0, w - lw - 0.5, h - 21.5)
    vmax = 0.12
    heat(ax, M, [r[1] for r in rows], [{"Δ": "Δ(S1−2D)"}.get(n, n) for _, n in cols], vmax, frames=Fr, fs=5.0)
    for c, (i, n) in enumerate(cols):
        ax.get_xticklabels()[c].set_color({"S1": CS1, "2D": C2D}.get(n, FG)); ax.get_xticklabels()[c].set_fontweight("bold")
        if n == "Δ":
            ax.get_xticklabels()[c].set_fontsize(5.0)
    for t in ax.get_yticklabels()[4:]:
        t.set_color(MUT); t.set_fontsize(5.0)
    aw = w - lw - 0.5
    txt(x + lw + aw * 0.25, y + 7.6, "F1@0.5", ha="center", va="bottom", fontsize=5.5, fontweight="bold")
    txt(x + lw + aw * 0.75, y + 7.6, "F1@0.7", ha="center", va="bottom", fontsize=5.5, fontweight="bold")
    ax.axvline(2.5, color="white", lw=1.0); ax.axhline(0.5, color="white", lw=1.0); ax.axhline(3.5, color="white", lw=0.8, ls=":")
    for c in (1.5, 4.5):
        ax.axvline(c, color="white", lw=1.0, ls=(0, (1, 1)))
    colorbar(x + lw, y + h - 7.6, aw, 1.2, vmax, "3 paired seeds; Δ = S1 loss − 2D loss; framed: confirmed")
    prov["panels"][lt] = {"kind": "heatmap, cell = mean over seeds 0/1000/2000 of (retrained − production), pooled F1, gt protocol; "
                                  "column Δ(S1−2D) = per-seed S1 loss − 2D loss = change in the 3D advantage (retrained S1−2D minus "
                                  "production S1−2D), as '优势变化' in l1s_ana.out / 'change in advantage' in main.tex sec:res-heldout",
                          "frame": "registered confirmation rule (|mean| > 2 SD and >= 0.01), same as l1s_ana.out; liver/fly are the "
                                   "NucVerse3D subsets of l1s_ana.out", "values": rec}


# ------------------------------------------------------------------ per-volume distributions
def panel_dist(x, y, w, h, lt, ncol_leg=6):
    letter(x, y, lt, "Per-volume F1")
    order = [("F_S1", CS1, True), ("L1_S1", CS1, False), ("F_2D", C2D, True), ("L1_2D", C2D, False)]
    rec = {}
    leg_h = 4.5 if ncol_leg >= 6 else 8.0
    pw = (w - 7.5) / 2
    for i in (0, 1):
        ax = ax_mm(x + 6.5 + i * (pw + 1.0), y + 5.5, pw - 1.0, h - 12.0 - leg_h)
        for gi, ds in enumerate(SETS):
            for oi, (n, col, prod) in enumerate(order):
                xc = gi + (oi - 1.5) * 0.22
                ks = sorted(k for k in PV[f"{n}_s0"] if k.startswith(ds + "__"))
                v = np.array([np.mean([PV[f"{n}_s{s}"][k][i] for s in SS]) for k in ks])
                parts = ax.violinplot([v], positions=[xc], widths=0.2, showextrema=False, bw_method=0.3)
                for b in parts["bodies"]:
                    b.set_facecolor(col if prod else "white"); b.set_edgecolor(col); b.set_alpha(0.55 if prod else 1.0); b.set_linewidth(0.5)
                    vv = b.get_paths()[0].vertices; vv[:, 1] = np.clip(vv[:, 1], 0, 1)
                ax.hlines(np.median(v), xc - 0.07, xc + 0.07, color=FG, lw=0.6, zorder=4)
                ps = seeds(n, ds, i)
                ax.scatter([xc + 0.08] * 3, ps, s=3.5, marker="o", color=col, edgecolor=FG, lw=0.3, zorder=5)
                rec[f"F1@{0.5 if i == 0 else 0.7}/{ds}/{n}"] = {"n_vol": len(v), "median_per_volume": round(float(np.median(v)), 4),
                                                                  "pooled_seeds": [round(p, 4) for p in ps]}
            ax.hlines(G("base", ds, i), gi - 0.45, gi + 0.45, color=CB, lw=0.6, ls=(0, (3, 1.5)), zorder=1)
            if gi:
                ax.axvline(gi - 0.5, color=RULE, lw=0.4)
        nv = {d: len([k for k in PV["F_S1_s0"] if k.startswith(d + "__")]) for d in SETS}
        ax.set_xticks(range(3), [f"{NM[d]}\n({nv[d]})" for d in SETS], fontsize=5.0)
        ax.tick_params(axis="x", length=0, pad=1.2); ax.tick_params(axis="y", labelsize=5.0, length=1.5, pad=1)
        ax.set_xlim(-0.5, 2.5); ax.set_ylim(0, 1.03); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(f"F1@{0.5 if i == 0 else 0.7}", fontsize=5.6, pad=1.5)
        if i == 0:
            ax.set_ylabel("F1 per volume", fontsize=5.5, labelpad=1)
        else:
            ax.set_yticklabels([])
        ax.spines["bottom"].set_linewidth(0.4); ax.spines["left"].set_linewidth(0.4)
    hl = [Patch(fc=CS1, ec=CS1, alpha=0.55, label="S1 production"), Patch(fc="white", ec=CS1, label="S1 retrained"),
          Patch(fc=C2D, ec=C2D, alpha=0.55, label="2D production"), Patch(fc="white", ec=C2D, label="2D retrained"),
          Line2D([], [], marker="o", ls="", ms=2.2, mfc="0.7", mec=FG, mew=0.3, label="pooled F1, one dot per seed"),
          Line2D([], [], color=CB, lw=0.6, ls=(0, (3, 1.5)), label="original (pooled)")]
    fig.legend(handles=hl, loc="lower center", bbox_to_anchor=((x + w / 2) / W, 1 - (y + h) / H), ncol=ncol_leg, fontsize=5.0,
               handlelength=1.1, columnspacing=0.8, handletextpad=0.3, frameon=False)
    prov["panels"][lt] = {"kind": "violins of per-volume F1 (mean over seeds 0/1000/2000 per volume, clipped to [0,1], bw 0.3), black tick = "
                          "median; dots = pooled F1 per seed; dashed = original Cellpose-SAM pooled", "values": rec}


# ------------------------------------------------------------------ L2
def l2(name, d, i, rule="r2"):
    return L2[name][rule][0][d][i]


def l2_series(m, d, i):
    """[retrained (no fine-tuning), k0 control, k1, k2, k5, k10]"""
    return [l2(f"L1_{'S1' if m == '3D' else '2D'}", d, i), l2(f"{m}_k0", d, i)] + [l2(f"{m}_{d}_k{k}", d, i) for k in KS]


L2COLS = ["retr.", "k=0", "1", "2", "5", "10"]


def panel_l2_curves(x, y, w, h, lt):
    letter(x, y, lt, "Adding back training volumes (L2): single seed (seed 0), descriptive only")
    rec = {}
    pw = (w - 8) / 3; ph = (h - 19.7) / 2
    xs = [-0.8, 0, 1, 2, 5, 10]
    for j, d in enumerate(SETS):
        for i in (0, 1):
            ax = ax_mm(x + 7.5 + j * pw + 3.6 * j / 2, y + 7.0 + i * (ph + 3.2), pw - 5.4, ph)
            allv = []
            for m, n5, col, mk in (("3D", "S1", CS1, "o"), ("2D", "2D", C2D, "s")):
                ys = l2_series(m, d, i)
                prod = [l2(f"{n5}_s{s}", d, i) for s in SS5]
                ax.axhspan(min(prod), max(prod), color=col, alpha=0.18, lw=0)
                ax.plot(xs[1:], ys[1:], color=col, lw=0.8, marker=mk, ms=2.2, mew=0)
                ax.plot([xs[0]], [ys[0]], marker="x", color=col, ms=3, mew=0.7, ls="none")
                allv += ys + prod
                rec[f"{d}/{n5}/F1@{0.5 if i == 0 else 0.7}"] = {"retrained_k_series": [round(v, 4) for v in ys],
                                                               "production_5seeds": [round(v, 4) for v in prod]}
            lo, hi = min(allv), max(allv); pad = (hi - lo) * 0.12
            ax.set_ylim(lo - pad, hi + pad)
            yt(ax)
            ax.set_xscale("symlog", linthresh=1); ax.minorticks_off()
            ax.set_xticks(xs, ["r", "0", "1", "2", "5", "10"] if i == 1 else [""] * 6, fontsize=5.0)
            ax.set_xlim(-1.2, 13)
            ax.tick_params(labelsize=5.0, length=1.5, pad=1)
            ax.spines["bottom"].set_linewidth(0.4); ax.spines["left"].set_linewidth(0.4)
            if j == 0:
                ax.set_ylabel(f"F1@{0.5 if i == 0 else 0.7}", fontsize=5.5, labelpad=1)
            if i == 0:
                ax.set_title(f"{NM[d]} ({L2['S1_s0']['r2'][0][d][-1]} vol.)", fontsize=5.6, pad=2)
    txt(x + 7.5 + 1.5 * pw, y + h - 5.9, "training volumes added back, k  (r = retrained, no fine-tuning; 0 = 1,000 steps, no data)",
        ha="center", va="top", fontsize=5.0)
    hl = [Line2D([], [], color=CS1, marker="o", ms=2.2, lw=0.8, label="S1"), Line2D([], [], color=C2D, marker="s", ms=2.2, lw=0.8, label="2D control"),
          Line2D([], [], color="0.4", marker="x", ls="", ms=3, label="retrained, no fine-tuning"),
          Patch(fc="0.6", alpha=0.3, label="production, range of 5 seeds")]
    fig.legend(handles=hl, loc="lower center", bbox_to_anchor=((x + w / 2) / W, 1 - (y + h) / H), ncol=4, fontsize=5.0, frameon=False,
               handlelength=1.3, columnspacing=1.0, handletextpad=0.35)
    prov["panels"][lt] = {"kind": "L2 curves, seed 0; x = 'r' retrained model without fine-tuning, 0 = 1,000 steps without added data; "
                          "band = production min-max over seeds 0-4 on the same L2 test volumes", "values": rec}


def l2_matrix(kind):
    rows, M, rec = [], [], {}
    for d in SETS:
        for i in (0, 1):
            s1, d2 = l2_series("3D", d, i), l2_series("2D", d, i)
            p1 = float(np.mean([l2(f"S1_s{s}", d, i) for s in SS5])); p2 = float(np.mean([l2(f"2D_s{s}", d, i) for s in SS5]))
            row = {"diff": [a - b for a, b in zip(s1, d2)] + [p1 - p2], "gapS1": [a - p1 for a in s1], "gap2D": [a - p2 for a in d2]}[kind]
            M.append(row); rows.append(f"{NM[d]}  F1@{0.5 if i == 0 else 0.7}")
            rec[rows[-1]] = [round(v, 4) for v in row]
    return np.array(M), rows, rec


def draw_l2_heat(x, y, w, h, kind, show_rows, vmax, title, tcol, cbar_label):
    M, rows, rec = l2_matrix(kind)
    cols = L2COLS + (["prod."] if kind == "diff" else [])
    lw = 22.0 if show_rows else 0.5
    ax = ax_mm(x + lw, y + 7.5, w - lw - 0.5, h - 15.0)
    heat(ax, M, rows, cols, vmax, fs=5.0, fmt="{:+.1f}", scale=100)
    if not show_rows:
        ax.set_yticklabels([])
    for r in (1.5, 3.5):
        ax.axhline(r, color="white", lw=1.0)
    if kind == "diff":
        ax.axvline(len(L2COLS) - 0.5, color="white", lw=1.0)
    txt(x + lw + (w - lw) / 2, y + 2.6, title, ha="center", va="top", fontsize=5.6, fontweight="bold", color=tcol)
    colorbar(x + lw + 1.5, y + h - 5.6, w - lw - 3.5, 1.2, vmax, cbar_label, pp=True)
    return {"cols": cols, "values": rec}


def panel_l2_heat_A(x, y, w, h, lt):
    letter(x, y, lt, "S1 − 2D after adding back (seed 0)")
    prov["panels"][lt] = {"diff": draw_l2_heat(x, y + 3.5, w, h - 3.5, "diff", True, 0.08, "pooled F1; prod. = production, 5 seeds", MUT,
                                               "S1 − 2D, percentage points"),
                          "note": "seed 0; 'retr.' = retrained model without fine-tuning; 'k=0' = 1,000 steps without added data; "
                                  "production = mean of seeds 0-4 on the same L2 test volumes"}


def panel_l2_heat_B(x, y, w, h, lt):
    letter(x, y, lt, "Adding back training volumes (L2): single seed (seed 0), descriptive only")
    w1 = 22.0 + (w - 22.0 - 4) * 6 / 19; w2 = (w - 22.0 - 4) * 6 / 19; w3 = (w - 22.0 - 4) * 7 / 19
    rec = {"gapS1": draw_l2_heat(x, y + 4.0, w1, h - 4.0, "gapS1", True, 0.12, "S1: gap to production", CS1, "gap to production mean, percentage points"),
           "gap2D": draw_l2_heat(x + w1 + 2, y + 4.0, w2, h - 4.0, "gap2D", False, 0.12, "2D: gap to production", C2D, "gap to production mean, percentage points"),
           "diff": draw_l2_heat(x + w1 + w2 + 4, y + 4.0, w3, h - 4.0, "diff", False, 0.08, "S1 − 2D", FG, "S1 − 2D (percentage points)")}
    rec["note"] = ("seed 0; columns: retr. = retrained model without fine-tuning; k=0 = 1,000 steps without added data; k = training volumes "
                   "added back; production = mean of seeds 0-4 on the same L2 test volumes (44 / 44 / 5)")
    prov["panels"][lt] = rec


# ------------------------------------------------------------------ renders (black band)
def panel_renders(x, y, w, h, lt):
    fig.add_artist(Rectangle((0, 1 - (y + h) / H), 1, (h + 0.8) / H, transform=fig.transFigure, fc="black", ec="none", zorder=-10))
    letter(x, y + 1.0, lt, "Representative test volumes (median rule, seed 0): production vs retrained S1", color="#e8e8ea")
    rec = {}
    head = 11.0; leg = 5.0; foot = 3.2; g = 1.0; gw = 2.5
    cw = (w - 2 * gw) / 3; tw = (cw - g) / 2; th = h - head - leg - foot
    tpx = (V.mm_px(tw), V.mm_px(th))
    for j, ds in enumerate(SETS):
        x0 = x + j * (cw + gw)
        ims, cnt = F.render_pair(ds)
        sel = SEL[ds]
        vid = sel["pick"].split("__", 1)[1]
        short = vid.replace("BlastoSPIM1_test_", "").replace("_image_0001", "").replace("2_livernuclei_20221025_30_Control_12w_", "liver ")
        txt(x0 + cw / 2, y + 5.3, f"{NM[ds]}  ·  {MOD[ds]}", ha="center", va="top", fontsize=6.0, fontweight="bold", color="#e8e8ea")
        txt(x0 + cw / 2, y + 7.9, f"{short}, {sel['n_gt']} GT nuclei, z/xy {sel['ratio']:.2f}", ha="center", va="top", fontsize=5.0, color="#a0a0a8")
        for c, (k, lab) in enumerate((("prod", "production"), ("ret", "retrained"))):
            ax = ax_mm(x0 + c * (tw + g), y + head, tw, th)
            t, sc = V.on_black(ims[k], tpx)
            ax.imshow(np.asarray(t), interpolation="none"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            f05, f07 = sel["f1"][k]
            txt(x0 + c * (tw + g) + tw / 2, y + head + th + 0.5, f"{lab}  {f05:.3f} / {f07:.3f}", ha="center", va="top", fontsize=5.1,
                color="#e8e8ea" if k == "prod" else V.ERRV["boundary"], fontweight="bold")
            rec.setdefault(ds, {"volume": sel["pick"], "n_gt": sel["n_gt"], "median_rule_d_F1@0.7": round(sel["d"], 4),
                                "median": round(sel["median"], 4), "n_eligible": sel["n_eligible"]})[k] = {
                "F1@0.5": f05, "F1@0.7": f07, "classes": cnt[k], "downsample": round(1 / sc, 2)}
    hl = [Patch(fc=F.COL[k], ec="none", label=n) for k, n in (("match", "matched, IoU ≥ 0.7"), ("iou57", "IoU 0.5–0.7"),
                                                             ("boundary", "boundary error"), ("split", "split"), ("merge", "merge"),
                                                             ("other", "false positive"), ("miss", "missed (ghost)"))]
    hl.append(Line2D([], [], ls="", label="numbers: F1@0.5 / F1@0.7"))
    fig.legend(handles=hl, loc="lower center", bbox_to_anchor=((x + w / 2) / W, 1 - (y + h) / H), ncol=8, fontsize=5.0, frameon=False,
               handlelength=1.0, columnspacing=0.8, handletextpad=0.35, labelcolor="#e8e8ea")
    prov["panels"][lt] = {"rule": open(os.path.join(RD, "R6_FIG5_RULE.md")).read(), "volumes": rec, "tiles_px": tpx, "ppi": V.PPI,
                          "render": "render3d/r6_fig5_render.py (r6_render pipeline: vivid palette, depth peeling, SSAO, 2x+ supersampling, "
                                    "Lanczos downsample, embedded unresampled)", "extra_colour": {"iou57": F.IOU57},
                          "numbers": "per-volume F1 from l1s_vol.json (rule T), equal to a re-derivation from the exported masks"}


# ------------------------------------------------------------------ layouts
if LAY == "A":
    panel_a(2, 1, 114, 41, cw=20.0)
    panel_slope(119, 1, 63, 41, "b")
    panel_loss(1, 45, 63, 40, "c")
    panel_dist(66, 45, 116, 40, "d")
    panel_l2_curves(1, 88, 120, 41, "e")
    panel_l2_heat_A(122, 88, 60, 41, "f")
    panel_renders(2, 131.5, 179, 38.5, "g")
else:
    panel_a(2, 1, 179, 34, cw=24.0)
    panel_renders(2, 37.5, 179, 41, "b")
    panel_slope(1, 81, 52, 44, "c")
    panel_loss(54, 81, 61, 44, "d")
    panel_dist(116, 81, 66, 44, "e", ncol_leg=3)
    panel_l2_heat_B(1, 127.5, 181, 42, "f")

def _cvd(cols):
    import itertools
    from colorspacious import cspace_convert
    rgb = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255  # noqa: E731
    out = {}
    for cvd in ("normal", "deuteranomaly", "protanomaly", "tritanomaly"):
        lab = {}
        for k, h in cols.items():
            c = rgb(h)
            if cvd != "normal":
                c = np.clip(cspace_convert(c, {"name": "sRGB1+CVD", "cvd_type": cvd, "severity": 100}, "sRGB1"), 0, 1)
            lab[k] = cspace_convert(c, "sRGB1", "CAM02-UCS")
        d = sorted((float(np.linalg.norm(lab[a] - lab[b])), a, b) for a, b in itertools.combinations(lab, 2))
        dn = [x for x in d if "iou57" in x[1:]]
        out[cvd] = {"min_all": round(d[0][0], 1), "closest_all": d[0][1:], "min_iou57": round(dn[0][0], 1), "closest_iou57": dn[0][1:],
                    "pairs_below_15": [(round(x, 1), a, b) for x, a, b in d if x < 15]}
    return out


CVD = _cvd(dict(F.COL, background="#000000"))
assert min(v["min_iou57"] for v in CVD.values()) >= 15.0, CVD
prov["render_colour_cvd"] = {"iou57": F.IOU57, "method": "CAM02-UCS distance, colorspacious sRGB1+CVD severity 100, all render error colours "
                             "plus the black background (same test as r6_render.cvd_vivid)", "result": CVD,
                             "note": "pairs below 15 not involving iou57 are inherited from palette_vivid_dark.json (match vs miss)"}
prov["overlap_issues"] = figcheck.check(fig)
prov.update(layout={"A": "schematic + slope on top; loss heatmap + distributions; L2 curves + S1-2D heatmap; black render band at the bottom",
                    "B": "full-width schematic; black render band second; slope + loss heatmap + distributions; three L2 heatmaps"}[LAY],
            size_mm=[W, H], palette="figures/palette.json (statistics), figures/palette_vivid_dark.json (renders)",
            fonts="Arial 5.0-6.5 pt, panel letters 8 pt", lines="0.25-1.0 pt", script="paper/scripts/figs/fig5_r6.py")
fig.savefig(os.path.join(GA, f"{CID}.pdf"), dpi=600)
fig.savefig(os.path.join(GA, f"{CID}.png"), dpi=600)
json.dump(prov, open(os.path.join(GA, f"{CID}.provenance.json"), "w"), indent=1, ensure_ascii=False, default=str)
print("saved", CID, "overlap:", prov["overlap_issues"])
