"""Fig. 2 round 8 (gallery R8-Fig2): R7 layout B with the unified 3D-viewer scheme (figures/palette_viewer.json): the black image
band of e and f becomes a warm-grey container, renders composited on #EDECE9 with the viewer error colours and muted instance colours.

Fig. 2, round 7 (gallery R7-Fig2-A / -B; R6b = isotropic cube chosen; heatmap refined; layout A: dark right column d+f;
layout B: one continuous black image band at the bottom). Round-6 notes follow.
Round 6 (gallery R6-Fig2*; user picked R5 2-B, then asked for black image tiles, more vivid and sharper renders
and a dataset x metric heatmap). Panels:
a, b  per-dataset forest plots (F1@0.5, F1@0.7; volume-level mean S1 - 2D, +-1.96 s.e.; pooled RE/HK) - same numbers as R-fig2-v2,
      rows now ordered by z/xy (see ORDER below) so that a, b and c share one row order.
c     heatmap dataset x metric, cell = dataset-level S1 - 2D (five-seed mean) for
        F1@0.5, F1@0.7     ref/s5_test_gt_5seeds_basefixed.json per_ds[*].d2d; bold frame = per-dataset call "win" (call2d == 胜,
                           registered three-way call: bootstrap lower bound > 0), as counted in NUMBERS.md (gt: 8 / 7 wins)
        matched IoU        data_snapshot/boundary_results.json subsets.all.all_matched["13 套"].per_ds (median IoU of matched pairs)
        count error        data_snapshot/round5/cpux/results.json analysis_A.readouts.n.sets["13"].per_ds_bias: |bias 2D| - |bias S1|
      sign: positive (red) = S1 better in every column. Colour scaled per column (symmetric, full colour = largest |value| in
      that column). Bottom row: 13-set macro value; bold frame = confirmed by the five-seed rule (macro only).
      Hatched row: mesoSPIM, single test volume (no per-dataset call).
d     paired seed differences (as before).
e     3D grid (black tiles, vivid colours, natively placed, render3d/r6_render.py).  f  cube (r6_render.py).
Row order: ascending z/xy of the dataset's median-rule test volume (ortho_selection.json), ties alphabetical.
"""
import hashlib
import json
import os
import sys

os.environ["NC_PALETTE"] = "npg"
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
import shutil  # noqa: E402
shutil.copy(os.path.join(HERE, "..", "..", "figures", "palette.json"), os.path.join(HERE, "palette.json"))
import nc_style as S  # noqa: E402
from nc_style import plt  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "render3d"))
import figcheck  # noqa: E402
import r6_render as V  # noqa: E402

S.setup()
P = S.PALETTE
ST = os.path.join(HERE, "localdata")
SNAP = os.path.abspath(os.path.join(HERE, "..", "..", "data_snapshot"))
GA = os.path.abspath(os.path.join(HERE, "..", "..", "figures", "gallery", "assets"))
SRC = {"nc_stats": os.path.join(ST, "nc_stats_results.json"), "s5_gt": os.path.join(ST, "ref", "s5_test_gt_5seeds_basefixed.json"),
       "boundary": os.path.join(SNAP, "boundary_results.json"), "cpux": os.path.join(SNAP, "round5", "cpux", "results.json"),
       "selection": os.path.join(HERE, "ortho_selection.json")}
MD5 = {k: hashlib.md5(open(v, "rb").read()).hexdigest() for k, v in SRC.items()}
assert MD5["cpux"] == "86f699a2ac00cd7256e8b346fcfda214"
R = json.load(open(SRC["nc_stats"]))
REF = {(m, sub): json.load(open(f"{ST}/ref/s5_test_{m}_5seeds_basefixed{sfx}.json"))
       for m in ("gt", "fp") for sub, sfx in (("all", ""), ("C1", "_clean_test"))}
S5 = REF[("gt", "all")]
BND = json.load(open(SRC["boundary"]))["subsets"]["all"]["all_matched"]["13 套"]
CPX = json.load(open(SRC["cpux"]))["analysis_A"]["readouts"]["n"]["sets"]["13"]
SEL = json.load(open(SRC["selection"]))["volumes"]
CUBE = "tapenade"; LAY = os.environ.get("FIG2_LAYOUT", "A"); CID = f"R8-Fig2-{LAY}"

W = 183.0; AB = 78.0; DH = 58.5; CUBE_W = 48.0; DW = W - CUBE_W - 3
GAP2 = 12.0 if LAY == "A" else 15.0
H = AB + GAP2 + DH + (3.0 if LAY == "B" else 0.0)
fig = plt.figure(figsize=(W / 25.4, H / 25.4))
MUT, FG, BAND = P["muted"], P["text"], "#f4f4f2"
prov = {"figure": CID, "sources_md5": MD5, "panels": {}}


def ax_mm(x, y, w, h):
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def letter(x, y, s, title=None, color=None):
    fig.text(x / W, 1 - y / H, s, fontsize=8, fontweight="bold", va="top", ha="left", color=color or "black")
    if title:
        fig.text((x + 4) / W, 1 - (y + 0.3) / H, title, fontsize=6.3, fontweight="bold", va="top", ha="left", color=FG)


dss_all = sorted(R["results"]["gt"]["all"]["F1@0.5"]["vs2D"]["per_ds"])
ORDER = sorted(dss_all, key=lambda d: (SEL[d]["ratio"], d))
n = len(ORDER)
prov["row_order"] = {"rule": "ascending z/xy of the median-rule test volume (ortho_selection.json), ties alphabetical",
                     "order": ORDER, "z/xy": {d: round(SEL[d]["ratio"], 2) for d in ORDER}}
TOP, BOT = 9.0, 62.0          # mm, row area of a-c (rows + pooled row)
FOOT_Y = BOT + 14.6           # mm, common baseline of the a/b footnote and the c legend (bottom row of a-d aligned)
LBL_X, A_X, AW, B_X = 0.0, 23.0, 30.0, 59.0
HM_X, HM_CW = 96.0, 8.0
yrow = lambda i: n - i  # noqa: E731   data y of row i (top = n)

# ---------------- a, b forests
for j, lab in enumerate(("F1@0.5", "F1@0.7")):
    x0 = A_X if j == 0 else B_X
    ax = ax_mm(x0, TOP, AW, BOT - TOP)
    e = R["results"]["gt"]["all"][lab]["vs2D"]; eff, s2 = e["per_ds"], e["S2"]
    for i, ds in enumerate(ORDER):
        y = yrow(i)
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color=BAND, lw=0, zorder=0)
        d = eff[ds]
        if d["n"] >= 2:
            lo, hi = d["effect"] - 1.96 * d["se"], d["effect"] + 1.96 * d["se"]
            small = d["n"] < 5
            ax.plot([lo, hi], [y, y], color="#9a9a96" if small else "#4d4d52", lw=0.6, ls=(0, (1, 1)) if small else "-", zorder=2)
            ax.plot(d["effect"], y, "s", mfc="white" if small else S.C["S1"], mec=S.C["S1"], ms=2.2 + 6 * s2["weights"].get(ds, 0), mew=0.7, zorder=3)
        else:
            ax.plot(d["effect"], y, "o", mfc="white", mec=MUT, ms=3, mew=0.6, zorder=3)
        ax.text(1.0, y, f"{d['n']}", transform=ax.get_yaxis_transform(), fontsize=5.0, va="center", ha="left", color=MUT, clip_on=False)
    mu, (lo, hi) = s2["mu"], s2["ci_hk"]
    ax.fill([lo, mu, hi, mu], [-0.6, -0.15, -0.6, -1.05], color=S.C["S1"], lw=0, zorder=3)
    ax.axhline(0.0, color=P["rule"], lw=0.5)
    ax.axvline(0, color=MUT, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.text(mu, -1.35, f"{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]", fontsize=5.0, ha="center", va="top", color=S.C["S1"], fontweight="bold", zorder=5,
            bbox=dict(fc="white", ec="none", pad=0.4))
    ax.set_ylim(-2.3, n + 0.7)
    ax.set_yticks(list(range(n, 0, -1)) + [-0.6])
    if j == 0:
        ax.set_yticklabels([S.DS_NAME[d] + (" †" if 2 <= eff[d]["n"] < 5 else "") for d in ORDER] + ["Pooled (RE, HK)"], fontsize=5.5)
        ax.get_yticklabels()[-1].set_fontweight("bold")
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    for sp in ("left", "top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(1.0, n + 0.75, "n", transform=ax.get_yaxis_transform(), fontsize=5.0, ha="left", va="bottom", color=MUT, fontweight="bold")
    ax.set_xticks([-0.05, 0, 0.05, 0.10, 0.15] if j == 0 else [0, 0.1, 0.2])
    ax.tick_params(axis="x", labelsize=5.2)
    ax.set_xlabel(f"Δ{lab}, S1 − 2D\n(volume-level mean)", fontsize=5.5, labelpad=1.5)
    letter(x0 - (A_X - 1 if j == 0 else 4.5) + (0 if j else 0), 2.0, "ab"[j], None)
    fig.text((x0 + 1) / W, 1 - 2.3 / H, f"{lab}   I² = {100 * s2['I2']:.0f}%", fontsize=6.3, fontweight="bold", va="top", color=FG)
    prov["panels"]["ab"[j]] = {"pooled": [mu, lo, hi], "I2": s2["I2"], "per_ds": {d: [eff[d]["effect"], eff[d]["se"], eff[d]["n"]] for d in ORDER}}
fig.text(A_X / W, 1 - FOOT_Y / H, "† n < 5 volumes: open symbol, dotted ±1.96 s.e.; open circle: single volume, not pooled", fontsize=5.0, color=MUT, va="bottom")

# ---------------- c heatmap (round 7: shared colour scale for the two F1 columns, per-column scales for IoU and count,
# mini colour bars, dot = win / confirmed, z/xy strip on the left, marginal bar of n test volumes on the right)
from matplotlib.colors import LogNorm  # noqa: E402
cols = [("F1@0.5", "ΔF1\n@0.5"), ("F1@0.7", "ΔF1\n@0.7"), ("iou", "ΔIoU\nmatched"), ("count", "Δ|count\nerror|")]
vals = np.full((n, 4), np.nan); win = np.zeros((n, 4), bool)
for i, ds in enumerate(ORDER):
    for c, (key, _) in enumerate(cols):
        if key.startswith("F1"):
            v = S5["per_ds"][key][ds]; vals[i, c] = v["d2d"]; win[i, c] = v.get("call2d") == "胜"
        elif key == "iou":
            b = BND["per_ds"][ds]; vals[i, c] = b["S1"]["iou_median"] - b["2D"]["iou_median"]
        else:
            b = CPX["per_ds_bias"][ds]; vals[i, c] = abs(b["2D"]) - abs(b["S1"])
macro = [S5["macro"]["F1@0.5"]["13 套"], S5["macro"]["F1@0.7"]["13 套"], BND["S1_minus_2D"]["iou_median"], CPX["abs_bias"]]
mvals = [macro[0]["diff_mean"], macro[1]["diff_mean"], macro[2]["mean"], -macro[3]["diff_mean"]]
mconf = [bool(macro[0]["confirmed"]), bool(macro[1]["confirmed"]), bool(macro[2]["confirmed"]), bool(macro[3]["confirmed"])]
# round 8 style: softer diverging map, white gutters, values in percentage points, dot = win / confirmed,
# faded cells for the single-volume dataset; the n-volume margin is dropped (n is printed next to a and b)
_lo, _mid, _hi = P["diverging"]
cmap = LinearSegmentedColormap.from_list("npgdiv2", [_lo, "#93a5cb", "#f7f7f5", "#f3a898", _hi])
sF1 = float(np.nanmax(np.abs(vals[:, :2]))); scale = [sF1, sF1, float(np.nanmax(np.abs(vals[:, 2]))), float(np.nanmax(np.abs(vals[:, 3])))]
HM_X, HM_CW, STRIP = 97.0, 8.1, 1.3
x_strip = HM_X - STRIP - 0.7
from matplotlib.patches import FancyBboxPatch  # noqa: E402
axh = ax_mm(HM_X, TOP, HM_CW * 4, BOT - TOP)
axh.set_xlim(-0.5, 3.5); axh.set_ylim(-2.3, n + 0.7); axh.axis("off")
fmt = lambda v: f"{100 * v:+.1f}".replace("-", "\u2212")  # noqa: E731   percentage points
GW, GH = 0.90, 0.84
for c in range(4):
    for i in range(n + 1):
        if i < n:
            v, y, mark = vals[i, c], yrow(i), win[i, c]
        else:
            v, y, mark = mvals[c], -0.6, mconf[c]
        col = cmap(0.5 + 0.5 * np.clip(v / scale[c], -1, 1))
        single = i < n and ORDER[i] == "mesospim"
        axh.add_patch(FancyBboxPatch((c - GW / 2, y - GH / 2), GW, GH, boxstyle="round,pad=0,rounding_size=0.10",
                                     fc=col, ec="none", alpha=0.45 if single else 1.0, zorder=2, mutation_aspect=1 / 0.9))
        lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        tc = "#8a8a86" if single else ("white" if lum < 0.55 else FG)
        axh.text(c - 0.06, y, fmt(v), ha="center", va="center", fontsize=5.0, color=tc, zorder=4,
                 fontweight="bold" if i == n else "normal", fontstyle="italic" if single else "normal")
        if mark:
            axh.plot(c + GW / 2 - 0.075, y + GH / 2 - 0.15, "o", ms=1.5, mfc="white" if lum < 0.55 else FG, mec="none", zorder=5)
for c, lab in enumerate(("F1@0.5", "F1@0.7", "IoU", "|count\nerror|")):
    axh.text(c, n + 0.75, lab, ha="center", va="bottom", fontsize=5.0, color=FG, linespacing=1.05)
axh.plot([-0.45, 3.45], [0.0, 0.0], color=P["rule"], lw=0.5)
# z/xy strip (row annotation; log scale, white -> NPG navy)
axz = ax_mm(x_strip, TOP, STRIP, BOT - TOP); axz.set_xlim(0, 1); axz.set_ylim(-2.3, n + 0.7); axz.axis("off")
zr = np.array([SEL[d]["ratio"] for d in ORDER]); zn = LogNorm(vmin=zr.min(), vmax=zr.max())
zmap = LinearSegmentedColormap.from_list("zxy", ["#ffffff", P["models"]["stardist3d"]])
for i, d in enumerate(ORDER):
    axz.add_patch(FancyBboxPatch((0.05, yrow(i) - GH / 2), 0.9, GH, boxstyle="round,pad=0,rounding_size=0.08",
                                 fc=zmap(0.12 + 0.88 * zn(zr[i])), ec="#d6d6d2", lw=0.25, mutation_aspect=4))
axz.text(0.5, n + 0.75, "z/xy", ha="center", va="bottom", fontsize=5.0, color=FG, rotation=90)
# colour bars under the columns: one shared for the two F1 columns, one each for IoU and count (pp)
gy = BOT + 1.2
for (c0, c1, sc_) in ((0, 1, scale[0]), (2, 2, scale[2]), (3, 3, scale[3])):
    axc = ax_mm(HM_X + c0 * HM_CW + 0.5, gy, (c1 - c0 + 1) * HM_CW - 1.0, 1.0)
    axc.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmap, aspect="auto", extent=(-sc_, sc_, 0, 1))
    axc.set_yticks([]); axc.set_xticks([])
    for sp in axc.spines.values():
        sp.set_visible(False)
    t = f"{100 * sc_:.1f}" if c0 == 0 else f"{100 * sc_:.0f}" if 100 * sc_ >= 10 else f"{100 * sc_:.1f}"
    axc.text(0.5, -0.9, f"\u00b1{t}", transform=axc.transAxes, ha="center", va="top", fontsize=5.0, color=FG)
fig.text(x_strip / W, 1 - (FOOT_Y) / H, "Percentage points; red, S1 better (F1 columns\nshare one scale). Dot: per-dataset win (F1) or\nconfirmed macro. Faded: one volume. z/xy strip:\nlog scale, 0.48 (light) \u2013 9.6 (dark)",
         fontsize=5.0, color=MUT, va="bottom", linespacing=1.2)
letter(x_strip - 4.2, 2.0, "c", None)
fig.text((x_strip + 0.2) / W, 1 - 2.3 / H, "Dataset × metric, S1 − 2D", fontsize=6.3, fontweight="bold", va="top", color=FG)
prov["panels"]["c"] = {"cells": {d: {k: float(vals[i, c]) for c, (k, _) in enumerate(cols)} for i, d in enumerate(ORDER)},
                       "win_dots": {d: [k for c, (k, _) in enumerate(cols) if win[i, c]] for i, d in enumerate(ORDER)},
                       "macro": {k: [float(mvals[c]), mconf[c]] for c, (k, _) in enumerate(cols)},
                       "colour_scales": {"F1 (shared, both columns)": sF1, "iou": scale[2], "count": scale[3]},
                       "z/xy_strip": "log scale, ortho_selection.json median-rule volume ratio", "display": "values x100 (percentage points), 1 decimal",
                       "sources": {"F1": "s5 per_ds d2d / call2d; macro 13 套 diff_mean / confirmed", "iou": "boundary all_matched 13 套 per_ds iou_median S1 - 2D; macro S1_minus_2D.iou_median",
                                   "count": "cpux n sets 13 per_ds_bias |2D| - |S1|; macro -abs_bias.diff_mean (not confirmed)"}}

# ---------------- d seeds (layout A: dark theme, part of a dark right column with f)
DK = LAY == "A"
DTC, DMC, DBG, DBAND = ("#e8e8ea", "#a0a0a8", "#000000", "#1c1c1e") if DK else (FG, MUT, "white", BAND)
if DK:
    from matplotlib.patches import Rectangle as _R  # noqa: E402
    fig.add_artist(_R(((W - CUBE_W - 2.5) / W, 0), (CUBE_W + 2.5) / W, 1, transform=fig.transFigure, fc="black", ec="none", zorder=-10))
ax = ax_mm(145.5, TOP + 1, 36.5, BOT - TOP - 1)   # bottom spine at BOT, same as a-c
ax.set_facecolor(DBG)
for sp in ax.spines.values():
    sp.set_color(DMC)
ax.tick_params(colors=DTC)
groups = [("gt", "all13", "13"), ("gt", "all", "12"), ("gt", "C1", "C1"), ("fp", "all13", "13"), ("fp", "all", "12"), ("fp", "C1", "C1")]
rng = np.random.default_rng(0)
ax.axvspan(2.5, 5.6, color=DBAND, lw=0, zorder=0)
pdv = {}
for g, (m, sub, _) in enumerate(groups):
    mac = REF[(m, "all" if sub == "all13" else sub)]["macro"]
    for k, lab in enumerate(("F1@0.5", "F1@0.7")):
        tag = [t for t in mac[lab] if ("mesospim" in t) != (sub == "all13")][0]
        ent = mac[lab][tag]
        d = np.array(ent["S1_seeds"]) - np.array(ent["two_d_seeds"]); mean, sd = d.mean(), d.std(ddof=1)
        conf = (mean > 2 * sd) and (mean >= 0.01)
        x = g + (-0.17 if k == 0 else 0.17); col = P["errors"]["boundary"] if k == 0 else S.C["S1"]
        ax.scatter(x + rng.uniform(-0.05, 0.05, len(d)), d, s=3.5, color=col, alpha=0.5, lw=0, zorder=3)
        ax.errorbar(x + 0.08, mean, yerr=2 * sd, fmt="o", ms=3, color=col, mfc=col if conf else DBG, mew=0.7, elinewidth=0.6, capsize=1.2, zorder=4)
        pdv[f"{m}/{sub}/{lab}"] = [float(mean), float(sd), bool(conf)]
ax.axhline(0, color=DMC, lw=0.5, ls=(0, (3, 2)))
ax.axhline(0.01, color=P["models"]["nnunet"], lw=0.6, ls=(0, (1, 1.2)))
ax.text(5.55, 0.0105, "0.01", color=P["models"]["nnunet"], fontsize=5, va="bottom", ha="right")
ax.set_xticks(range(6), [g[2] for g in groups], fontsize=5.3); ax.set_xlim(-0.5, 5.6); ax.set_ylim(-0.02, 0.09); ax.set_yticks([-0.02, 0, 0.02, 0.04, 0.06, 0.08])
for xx, tt in ((1, "gt protocol"), (4, "full pipeline")):
    ax.text(xx, -0.09, tt, ha="center", va="top", fontsize=5.4, transform=ax.get_xaxis_transform(), color=DTC)
ax.set_ylabel("Paired difference S1 − 2D (macro F1)", fontsize=5.6, color=DTC)
ax.tick_params(axis="y", labelsize=5.2)
h = [Line2D([], [], marker="o", ls="", color=P["errors"]["boundary"], ms=3, label="F1@0.5"), Line2D([], [], marker="o", ls="", color=S.C["S1"], ms=3, label="F1@0.7"),
     Line2D([], [], marker="o", ls="", color=DMC, mfc=DBG, ms=3, label="open: not confirmed"),
     Line2D([], [], color=P["models"]["nnunet"], ls=(0, (1, 1.2)), lw=0.6, label="confirmation minimum")]
ax.legend(handles=h, loc="upper left", fontsize=5, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.25, labelcolor=DTC)
letter(W - CUBE_W - 1.5, 2.0, "d", None, color=DTC)
fig.text((W - CUBE_W + 2.5) / W, 1 - 2.3 / H, "Five seeds, paired", fontsize=6.3, fontweight="bold", va="top", color=DTC)
prov["panels"]["d"] = pdv
fig.text((W - CUBE_W + 2.5) / W, 1 - FOOT_Y / H,
         "13, all datasets; 12, excluding mesoSPIM;\nC1, eight datasets. Small dots, single seeds;\nbars, mean \u00b1 2 s.d.; filled, mean > 2 s.d.\nand \u2265 0.01 (registered confirmation rule)",
         fontsize=5.0, color=MUT if not DK else DMC, va="bottom", linespacing=1.2)

# ---------------- e grid, f cube
Y2 = AB + GAP2
_PVW = json.load(open(os.path.join(HERE, "..", "..", "figures", "palette_viewer.json")))
V.ERRV.update(_PVW["errors"]); V.QUALV[:] = _PVW["qualitative_instances"]; V.MATCH_OP = _PVW["matched_opacity"]
V.CACHE = os.path.join(HERE, "..", "render3d", "out", "r8f2_viewer_cache"); os.makedirs(V.CACHE, exist_ok=True)
from PIL import Image as _Im  # noqa: E402
from matplotlib.colors import to_rgb as _rgb  # noqa: E402


def _on_pane(im, size):
    a_ = np.asarray(im).copy(); a_[..., 3][a_[..., 3] < _PVW["dehalo_alpha_below"]] = 0; im = _Im.fromarray(a_)
    w, h = size
    sc = min(w * 0.96 / im.width, h * 0.96 / im.height)
    r = im.resize((max(1, int(round(im.width * sc))), max(1, int(round(im.height * sc)))), _Im.LANCZOS)
    t = _Im.new("RGBA", (w, h), tuple(int(round(255 * c)) for c in _rgb(_PVW["pane"])) + (255,))
    t.alpha_composite(r, ((w - r.width) // 2, (h - r.height) // 2))
    return t.convert("RGB"), sc


V.on_black = _on_pane
if LAY == "A":
    prov["panels"]["e"] = V.draw_twide(fig, W, H, 0.0, Y2, DW, DH, ["nis3d", "tapenade", "nisnet_zf", "mouse_embryo", "blastospim"])
    letter(0, Y2 - 6.5, "e", None)
    fig.text(4.0 / W, 1 - (Y2 - 6.2) / H, "Representative test volumes in 3D (median rule, seed 0, gt protocol): 2D control vs S1; numbers: per-volume F1@0.5",
             fontsize=6.3, fontweight="bold", va="top", color=FG)
    prov["panels"]["f"] = V.draw_cube(fig, W, H, W - CUBE_W, Y2 + 1, CUBE_W, DH - 1.0, CUBE, fs=5.2, cap_color=DMC)
    letter(W - CUBE_W - 1.5, Y2 - 6.5, "f", None, color=DTC)
    fig.text((W - CUBE_W + 2.5) / W, 1 - (Y2 - 6.2) / H, "Raw volume and S1 nuclei", fontsize=6.3, fontweight="bold", va="top", color=DTC)
else:
    from matplotlib.patches import Rectangle as _R  # noqa: E402
    yb = Y2 - 9.0
    from matplotlib.patches import FancyBboxPatch as _FB  # noqa: E402
    fig.add_artist(_FB((0.5 / W, 0.5 / H), (W - 1.0) / W, (H - yb - 1.0 - 4.5) / H, boxstyle="round,pad=0,rounding_size=0.012", transform=fig.transFigure,
                       fc=_PVW["pane"], ec="none", zorder=-10, mutation_aspect=W / H))
    LT, LM = FG, MUT
    prov["panels"]["e"] = V.draw_twide(fig, W, H, 1.0, Y2, DW - 1.0, DH, ["nis3d", "tapenade", "nisnet_zf", "mouse_embryo", "blastospim"], tc=LT)
    from matplotlib.text import Text as _T  # noqa: E402
    for _t in fig.findobj(_T):
        if str(_t.get_color()).lower() == "#d0d0d4":   # 2D-control F1 numbers were light grey for the black band
            _t.set_color("#4d4d52")
    letter(1.0, Y2 - 6.5, "e", None, color=LT)
    fig.text(5.0 / W, 1 - (Y2 - 6.2) / H, "Representative test volumes in 3D (median rule, seed 0, gt protocol): 2D control vs S1; numbers: per-volume F1@0.5",
             fontsize=6.3, fontweight="bold", va="top", color=LT)
    prov["panels"]["f"] = V.draw_cube(fig, W, H, W - CUBE_W - 1.0, Y2 + 1, CUBE_W, DH - 1.0, CUBE, fs=5.2, cap_color=LM)
    letter(W - CUBE_W - 2.5, Y2 - 6.5, "f", None, color=LT)
    fig.text((W - CUBE_W + 1.5) / W, 1 - (Y2 - 6.2) / H, "Raw volume and S1 nuclei", fontsize=6.3, fontweight="bold", va="top", color=LT)
prov["layout"] = {"A": "dark right column (d + f), data panels a-c on white", "B": "data panels a-d on white, one black image band (e + f) at the bottom"}[LAY]

fig.savefig(os.path.join(GA, f"{CID}.pdf"))
fig.savefig(os.path.join(GA, f"{CID}.png"), dpi=600)
rep = figcheck.check(fig)
prov["overlap_issues"] = rep; prov["size_mm"] = [W, round(H, 1)]
json.dump(prov, open(os.path.join(GA, f"{CID}.provenance.json"), "w"), indent=1, default=str)
print("saved", CID, round(H, 1), rep)
