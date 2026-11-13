"""Round 12 of Fig. 3 (user 2026-11-11: polish alignment, colour, chart form, background). Same data as R11. Changes: a in the Fig. 2c
heatmap style (white gutters, rounded cells, softer diverging map); b foreground-logit curve in neutral dark grey (teal is reserved
for nnU-Net); row-1 plot bottoms aligned (a, b, c end at R1 + 47 mm); c legend on one line, neutral dumbbell bars, one shared
"S1 - zeroed" header; row 2: d and f axes share top/bottom, f title on one line, f moved into the c column; e composite
near-white pixels (>= 250) set to pure white so no off-white box shows; titles of each row on one baseline."""
"""Fig. 3, round 7 (gallery R7-Fig3): R6 with a looser, aligned layout (user: "too dense in places"). Changes: rows at
R1 = 3, R2 = 61, R3 = 122 mm with all panel titles on those baselines; absolute macro parts of a moved to the caption /
source data; c labels moved to a right-hand column; e slot 62 x 56 mm for nc-viz ISO-c (text block moved to the caption);
g full width, its f-summary text moved to the caption. Round-6 notes:
Round 6 (gallery R6-Fig3; user picked R5 3-B = R5-Fig3-dark and asked for richer chart forms, all real data).
a  heatmap error type x dataset: per-dataset contribution to 1 - F1@0.5 (gt, five-seed mean), cell = 2D - S1 in percentage
   points (red = S1 fewer errors of that type), 12 datasets ordered by z/xy; macro row; absolute macro values for S1 / 2D /
   original underneath (the numbers of the former stacked bars).                errors/results.json gt/A1/all/<12 sets>/*/per_ds d_*
b  decoder ablation curves (unchanged)                                            gpu/a4/a4_interp.json v1
c  dumbbell: macro F1 of the 2D control, S1 and S1 with the decoder output zeroed, by view group, +-1 s.d. over five seeds;
   S1 - zeroed printed (seed mean)                                                m1_strat.json gt
d  isotropic flow following: on - off (val., test; 95 % interval) with the 12 per-dataset differences as dots; merged-nucleus
   rate off -> on as a paired slope (val., test)                                  g6_{val,test}.json T_12, T_err_on/off
e  nc-viz ISO-b composite (placed at its native 60 x 55 mm, 900 ppi, unresampled)  assets_viz/ISO-b_composite.png
f  FLOWERR per dataset (unchanged)                                                flowerr/results.json, per_volume.csv
g  exploded stack with pulled xz / xy planes, dark image panel (R5 3-B), rasters unresampled
No new statistics: every number is read from the files above (same as the FINAL Fig. 3 provenance).
"""
import csv
import hashlib
import json
import os
import sys

os.environ["NC_PALETTE"] = "npg"
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
import shutil  # noqa: E402
shutil.copy(os.path.join(HERE, "..", "..", "figures", "palette.json"), os.path.join(HERE, "palette.json"))
import nc_style as S  # noqa: E402
from nc_style import plt  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "render3d"))
import figcheck  # noqa: E402

S.setup()
P = S.PALETTE
SNAP = os.path.abspath(os.path.join(HERE, "..", "..", "data_snapshot"))
VIZ = os.path.abspath(os.path.join(HERE, "..", "..", "figures", "assets_viz"))
FL = os.path.abspath(os.path.join(HERE, "..", "render3d", "export", "flowerr"))
GA = os.path.abspath(os.path.join(HERE, "..", "..", "figures", "gallery", "assets"))
MD5 = {os.path.join(HERE, "localdata", "errors_results.json"): "01b398578a871f84d76efb5646f2ed58",
       os.path.join(SNAP, "a4_interp.json"): "bdfe76fbdff469a19582f9f831fd31ea",
       os.path.join(SNAP, "g6_val.json"): "d801aca33cfb982cde52121c5078a295",
       os.path.join(SNAP, "g6_test.json"): "f65cc969a323a06307d7320247b88450",
       os.path.join(FL, "results.json"): "ae201d58b66d80f08a20622de036619e",
       os.path.join(FL, "per_volume.csv"): "2ef93fb8dd9f19a7621ae24f036afbe4"}
for f, m in MD5.items():
    assert hashlib.md5(open(f, "rb").read()).hexdigest() == m, f
E = json.load(open(os.path.join(HERE, "localdata", "errors_results.json")))
A4 = json.load(open(os.path.join(SNAP, "a4_interp.json")))
G6 = {k: json.load(open(os.path.join(SNAP, f"g6_{k}.json"))) for k in ("val", "test")}
FR = json.load(open(os.path.join(FL, "results.json")))
M12 = [k for k in E["gt"]["A1"]["all"] if "mesospim" in k][0]
FG, MUT = P["text"], P["muted"]
ERRC = P["errors"]
W, H = 183.0, 170.0
CID = os.environ.get("FIG3_CID", "R12-Fig3")
fig = plt.figure(figsize=(W / 25.4, H / 25.4))
prov = {"figure": CID, "sources_md5": {os.path.relpath(k, HERE): v for k, v in MD5.items()}, "panels": {}}
M1 = json.load(open(os.path.join(SNAP, "m1_strat.json")))["gt"]
assert hashlib.md5(open(os.path.join(SNAP, "m1_strat.json"), "rb").read()).hexdigest() == "f4ba9ee63204fc169f58b4adadcd8e74"
prov["sources_md5"]["data_snapshot/m1_strat.json"] = "f4ba9ee63204fc169f58b4adadcd8e74"
plt.rcParams["hatch.linewidth"] = 0.4


def ax_mm(x, y, w, h):
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def letter(x, y, s, title=None):
    fig.text(x / W, 1 - y / H, s, fontsize=8, fontweight="bold", va="top", ha="left")
    if title:
        fig.text((x + 4) / W, 1 - (y + 0.3) / H, title, fontsize=6.3, fontweight="bold", va="top", ha="left", color=FG)



R1 = 3.0
sys.path.insert(0, os.path.join(HERE, "..", "render3d"))
SEL = json.load(open(os.path.join(HERE, "ortho_selection.json")))["volumes"]

# ---------------- a: error type x dataset heatmap
e = E["gt"]["A1"]["all"][M12]
COMP = [("miss", "miss"), ("merge", "merge"), ("split", "split"), ("boundary", "bound."), ("false", "FP")]
ORD = sorted(e["S1"]["per_ds"], key=lambda d: (SEL[d]["ratio"], d))
nd = len(ORD)
dv = np.array([[100 * (e["2D"]["per_ds"][d]["d_" + c] - e["S1"]["per_ds"][d]["d_" + c]) for c, _ in COMP] for d in ORD])
dv = np.hstack([dv, dv.sum(1, keepdims=True)])
mac = np.array([100 * (e["2D"]["decomp"][c] - e["S1"]["decomp"][c]) for c, _ in COMP] + [100 * (e["2D"]["one_minus_f1"] - e["S1"]["one_minus_f1"])])
absm = {m: [100 * e[m]["decomp"][c] for c, _ in COMP] + [100 * e[m]["one_minus_f1"]] for m in ("S1", "2D", "base")}   # caption / source data only
_lo, _mid, _hi = P["diverging"]
cmap = LinearSegmentedColormap.from_list("npgdiv2", [_lo, "#93a5cb", "#f7f7f5", "#f3a898", _hi])
from matplotlib.patches import FancyBboxPatch  # noqa: E402
sc = np.max(np.abs(dv))
LBX, CW = 18.5, 6.0
AH = 38.5
axa = ax_mm(LBX, R1 + 10.0, CW * 6, AH)
nr = nd + 1
axa.set_xlim(-0.5, 5.5); axa.set_ylim(nr - 0.3 + 0.2, -0.6); axa.axis("off")
rowy = list(range(nd)) + [nd + 0.3]
yy_mm = lambda r: R1 + 10.0 + AH * (r + 0.6) / (nr - 0.3 + 0.8)  # noqa: E731
for c in range(6):
    for i in range(nd + 1):
        v = dv[i, c] if i < nd else mac[c]
        col = cmap(0.5 + 0.5 * np.clip(v / sc, -1, 1)); y = rowy[i]
        axa.add_patch(FancyBboxPatch((c - 0.45, y - 0.42), 0.90, 0.84, boxstyle="round,pad=0,rounding_size=0.12", fc=col, ec="none",
                                     mutation_aspect=(nr + 0.5) / 6 * (CW * 6) / AH / 1.0))
        lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        axa.text(c, y, f"{v:+.1f}".replace("-", "\u2212"), ha="center", va="center", fontsize=5.0,
                 color="white" if lum < 0.55 else FG, fontweight="bold" if i == nd else "normal")
for c, (_, lab) in enumerate(COMP + [(None, "total")]):
    axa.text(c, -0.75, lab, ha="center", va="bottom", fontsize=5.2, color=FG)
for i, d in enumerate(ORD):
    fig.text((LBX - 0.8) / W, 1 - yy_mm(rowy[i]) / H, S.DS_NAME[d], ha="right", va="center", fontsize=5.2)
fig.text((LBX - 0.8) / W, 1 - yy_mm(rowy[nd]) / H, "macro", ha="right", va="center", fontsize=5.2, fontweight="bold")
letter(0, R1, "a", "Error types per dataset")
fig.text(4 / W, 1 - (R1 + 3.6) / H, "2D − S1, percentage points of 1 − F1@0.5\n(red: S1 makes fewer errors of that type)", fontsize=5.0, color=MUT, va="top", linespacing=1.15)
prov["panels"]["a"] = {"row_order": "ascending z/xy of the median-rule volume (ortho_selection.json)", "order": ORD,
                       "cells_pp_2D_minus_S1": {d: dict(zip([c for c, _ in COMP] + ["total"], map(float, dv[i]))) for i, d in enumerate(ORD)},
                       "macro_pp_2D_minus_S1": dict(zip([c for c, _ in COMP] + ["total"], map(float, mac))),
                       "absolute_macro_pp": absm, "colour_scale_pp": float(sc), "source": "errors_results.json gt/A1/all/12 套 per_ds d_* (five-seed mean)"}
# ---------------- b: decoder ablation curves (A4)
x = np.array(A4["bins"]); cv = A4["v1"]["curve"]
axb1 = ax_mm(72, R1 + 10, 41, 16); axb2 = ax_mm(72, R1 + 29, 41, 18)
for a_ in (axb1, axb2):
    a_.axvspan(-3, 3, color="#f0f0ee", lw=0); a_.axvline(0, color=MUT, lw=0.5)
axb1.plot(x, cv["dz"], color=S.C["S1"], lw=0.9, label="|Δ axial flow|")
axb1.plot(x, cv["dxy"], color=S.C["S1"], lw=0.9, ls=(0, (3, 1.5)), label="|Δ lateral flow|")
axb1.set_ylabel("flow change\n(unit vector = 5)", fontsize=5.6); axb1.set_yticks([0, 1, 2]); plt.setp(axb1.get_xticklabels(), visible=False)
axb1.legend(loc="upper right", fontsize=5.2, handlelength=1.6, borderaxespad=0.1)
axb2.plot(x, cv["dcp"], color="#4d4d52", lw=0.9)
axb2.set_ylabel("|Δ foreground\nlogit|", fontsize=5.6); axb2.set_yticks([0.5, 1.0])
axb2.set_xlabel("signed distance to GT boundary (voxels)", fontsize=5.6); axb2.set_xticks([-20, -10, 0, 10, 20])
axb1.text(0, 1.02, "boundary band", transform=axb1.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.2, color=MUT)
letter(60, R1, "b", f"Decoder output zeroed ({A4['v1']['n_vol']} volumes, {A4['v1']['n_ds']} datasets)")
prov["panels"]["b"] = {"region_means": A4["v1"]["region"], "n_vol": A4["v1"]["n_vol"], "n_ds": A4["v1"]["n_ds"]}


# ---------------- c: dumbbell, decoder zeroed by view group (M1)
GRP = [("单视图3套", "single view (3 sets)"), ("三视图10套", "three views (10 sets)")]
MK = [("2D", "2D control", dict(marker="s", color=S.C["2D"], mfc=S.C["2D"])), ("Z", "S1, decoder zeroed", dict(marker="o", color=S.C["S1"], mfc="white")),
      ("S1", "S1", dict(marker="o", color=S.C["S1"], mfc=S.C["S1"]))]
pcv = {}
for j, lab in enumerate(("F1@0.5", "F1@0.7")):
    ax = ax_mm(136, R1 + 13.5 + j * 20.0, 34, 13.5)
    for gi, (key, glab) in enumerate(GRP):
        em = M1[f"{lab}|{key}"]; y = 1 - gi
        ax.plot([em["Z"][0], em["S1"][0]], [y, y], color="#d9d9d6", lw=2.0, solid_capstyle="round", zorder=1)
        for q, (m, mlab, kw) in enumerate(MK):
            mu, sd = em[m]
            ax.errorbar(mu, y + (0.18 if m == "2D" else 0), xerr=sd, ms=3.2, mew=0.7, elinewidth=0.6, capsize=1.0, ls="", zorder=3,
                        label=mlab if (j == 0 and gi == 0) else None, **kw)
            pcv[f"{lab}/{glab}/{m}"] = [mu, sd]
        dz = em["S1-Z"]
        ax.text(1.04, y, f"{dz['mean']:+.3f}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.0, color=S.C["S1"])
        pcv[f"{lab}/{glab}/S1-Z"] = [dz["mean"], dz["sd"], dz["confirmed"]]
    ax.set_ylim(-0.55, 1.45); ax.set_yticks([1, 0], ["single\nview", "three\nviews"] if True else None, fontsize=5.2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.0 if j == 0 else 0.75); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0] if j == 0 else [0, 0.25, 0.5, 0.75])
    ax.tick_params(axis="x", labelsize=5.0)
    ax.set_xlabel(f"macro {lab}", fontsize=5.4, labelpad=1)
    for sp in ("left", "top", "right"):
        ax.spines[sp].set_visible(False)
    if j == 0:
        ax.text(1.04, 1.5, "S1 − zeroed", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=5.0, color=S.C["S1"])
    if j == 0:
        fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(124 / W, 1 - (R1 + 3.6) / H), ncol=3, fontsize=5.0,
                   handlelength=1.0, columnspacing=0.6, handletextpad=0.2, frameon=False)
letter(124 - 4, R1, "c", "Decoder zeroed, by view group (gt, 5 seeds)")
fig.text(124 / W, 1 - (R1 + 6.9) / H, "single view: Cellos, Parhyale, mesoSPIM; mean ± 1 s.d.", fontsize=5.0, color=MUT, va="top")
prov["panels"]["c"] = {"values": pcv, "groups": "single view: Cellos, Parhyale, mesoSPIM; three views: other 10 datasets", "error_bars": "s.d. over 5 seeds"}

# ---------------- d: isotropic flow following
R2 = 61.0
axd1 = ax_mm(10, R2 + 8, 19, 43); axd2 = ax_mm(39, R2 + 8, 12, 43)
pdd = {}
for i, split in enumerate(("val", "test")):
    T = G6[split]["T_12"]
    for k, (lab, key, dk) in enumerate((("F1@0.5", "ci05", "d05"), ("F1@0.7", "ci07", "d07"))):
        d = -T["diff"][k]; lo, hi = -T[key][1], -T[key][0]
        col = P["errors"]["boundary"] if k == 0 else S.C["S1"]
        x = i + (-0.17 if k == 0 else 0.17)
        pds = np.array([-T["per_ds"][q][dk] for q in sorted(T["per_ds"])])
        axd1.scatter(x - 0.08 + np.linspace(-0.04, 0.04, len(pds)), pds, s=2.2, color=col, alpha=0.45, lw=0, zorder=2)
        axd1.errorbar(x + 0.05, d, yerr=[[d - lo], [hi - d]], fmt="o", ms=2.6, color=col, elinewidth=0.7, capsize=1.2, zorder=3)
        pdd[f"{split}/{lab}"] = [d, lo, hi]; pdd[f"{split}/{lab}/per_ds_on_minus_off"] = dict(zip(sorted(T["per_ds"]), map(float, pds)))
    on, off = G6[split]["T_err_on"], G6[split]["T_err_off"]
    yy = [100 * off["merge"], 100 * on["merge"]]
    mk = "o" if split == "test" else "s"
    axd2.plot([0, 1], yy, color=S.C["S1"], lw=0.8, marker=mk, ms=3, mfc="white", zorder=2)
    axd2.plot([1], [yy[1]], marker=mk, ms=3, color=S.C["S1"], ls="", zorder=3)
    axd2.text(1.12, yy[1], f"{split}.\n{yy[0]:.1f}→{yy[1]:.1f}" if split == "val" else f"test\n{yy[0]:.1f}→{yy[1]:.1f}", fontsize=5.0, va="center", ha="left", color=FG, linespacing=1.05)
    pdd[f"{split}/merge_on_off"] = [on["merge"], off["merge"], on["n_gt"], G6[split]["n_aniso"]]
axd1.axhline(0, color=MUT, lw=0.5, ls=(0, (3, 2)))
axd1.set_xticks([0, 1], ["val.", "test"]); axd1.set_xlim(-0.5, 1.5); axd1.set_ylim(-0.03, 0.105); axd1.set_yticks([-0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.10])
axd1.set_ylabel("Δ F1, on − off (dots: 12 datasets)", fontsize=5.4)
axd1.tick_params(axis="y", labelsize=5.0)
axd1.legend(handles=[Line2D([], [], marker="o", ls="", color=P["errors"]["boundary"], ms=2.6, label="F1@0.5"),
                     Line2D([], [], marker="o", ls="", color=S.C["S1"], ms=2.6, label="F1@0.7")], loc="upper left", fontsize=5.0,
            handletextpad=0.1, borderaxespad=0.05)
axd2.set_xticks([0, 1], ["off", "on"]); axd2.set_xlim(-0.3, 1.1); axd2.set_ylim(0, 6)
axd2.set_ylabel("merged GT nuclei (%)", fontsize=5.4); axd2.tick_params(axis="y", labelsize=5.0)
letter(0, R2, "d", "Isotropic flow following")
prov["panels"]["d"] = {"values": pdd, "note": "json diff = off - on; plotted as on - off; 12 datasets, S1 seed 0, rule T; dots: per-dataset on - off (T_12.per_ds)"}

# ---------------- e: nc-viz ISO (ISO-c if delivered: 62 x 56 mm, text block moved to the caption; else ISO-b 60 x 55 placeholder)
ISOC = os.path.join(VIZ, os.environ.get("FIG3_ISO", "ISO-c_composite.png"))   # R8: ISO-d_composite_{light,dark}.png (same 62 x 56 layout)
if os.path.exists(ISOC):
    ISO, EW, EH = ISOC, 62.0, 56.0
else:
    ISO, EW, EH = os.path.join(VIZ, "ISO-b_composite.png"), 60.0, 55.0
ime = np.asarray(Image.open(ISO).convert("RGB")).copy()
ime[(ime >= 250).all(-1)] = 255   # pure white page (the composite carried a 254 off-white box)
axe = ax_mm(62 + (62 - EW) / 2, R2 + 4.8, EW, EH); axe.imshow(ime, interpolation="none"); axe.axis("off")
letter(60, R2, "e", "Stacked pairs: native vs isotropic grid (S1)")
ivp = json.load(open(os.path.join(VIZ, os.path.basename(ISO).split("_")[0] + ".provenance.json")))
prov["panels"]["e"] = {"asset": os.path.relpath(ISO, VIZ), "md5": hashlib.md5(open(ISO, "rb").read()).hexdigest(), "caption_required": ivp.get("caption_required")}
# ---------------- f: FLOWERR compact (horizontal dot plot, datasets as rows)
pr = FR["primary"]["set12_region_a"]; pdg = pr["per_dataset"]
rows = list(csv.DictReader(open(os.path.join(FL, "per_volume.csv"))))
SEEDS = ("s0", "s1000", "s2000", "s3000", "s4000")
dss = sorted(pdg, key=lambda d: -pdg[d]["mean"])
seedv = {d: [float(np.mean([float(r[f"delta_a_{s}"]) for r in rows if r["dataset"] == d])) for s in SEEDS] for d in dss}
ax = ax_mm(146, R2 + 8, 36, 43)
n = len(dss)
for i, d in enumerate(dss):
    y = n - i
    ax.scatter(seedv[d], np.full(5, y) + np.linspace(-0.2, 0.2, 5), s=2, color=S.C["S1"], alpha=0.45, lw=0, zorder=2)
    ax.scatter(pdg[d]["mean"], y, s=8, color=S.C["S1"], lw=0, zorder=3)
mu, (lo, hi) = pr["macro_delta"], pr["ci95"]
ax.errorbar(mu, -0.3, xerr=[[mu - lo], [hi - mu]], fmt="D", ms=3, color=P["models"]["S1_dark"], elinewidth=0.8, capsize=1.3, zorder=4)
ax.axvline(0, color=MUT, lw=0.5, ls=(0, (3, 2))); ax.axhline(0.4, color=P["rule"], lw=0.5)
ax.set_yticks(list(range(n, 0, -1)) + [-0.3], [S.DS_NAME[d] for d in dss] + ["macro"], fontsize=5.2)
ax.get_yticklabels()[-1].set_fontweight("bold"); ax.tick_params(axis="y", length=0)
ax.set_ylim(-1.0, n + 0.6); ax.set_xlim(-6.2, 0.6); ax.set_xticks([-6, -4, -2, 0])
ax.set_xlabel("Δ median angle, S1 − 2D (°)", fontsize=5.6)
ax.text(lo - 0.15, -0.3, f"{mu:.2f}°".replace("-", "−"), va="center", ha="right", fontsize=5.2, color=P["models"]["S1_dark"])
letter(120, R2, "f", "Angle to 3D flow target (S1 trained on it)")
D = FR["descriptive_macro_median_theta"]
regs = {r: FR["primary"][f"set12_region_{r}"]["macro_delta"] for r in "abc"}
prov["panels"]["f"] = {"per_dataset_mean": {d: pdg[d]["mean"] for d in dss}, "seed_values": seedv, "macro_a": [mu, lo, hi],
                       "regions": regs, "descriptive": {"S1": D["set12_region_a_S1"], "2D": D["set12_region_a_2D"], "base": D["set12_region_a_base"]},
                       "confirmation": FR["confirmation"], "note": "region a, 12 datasets, 5 seeds; macro 95 % interval"}

# ---------------- g: exploded stack (R5 3-B, dark), rasters unresampled
import r5_panels as P5  # noqa: E402
R3 = 124.0
summ = ("Summary of f, other regions:\n" + "macro \u0394 by region a / b / c:\n" + " / ".join(f"{v:.2f}\u00b0".replace("-", "\u2212") for v in regs.values())
        + "\n\nmacro median angle (descriptive):\n" + f"S1 {D['set12_region_a_S1']:.1f}\u00b0, 2D {D['set12_region_a_2D']:.1f}\u00b0,\noriginal {D['set12_region_a_base']:.1f}\u00b0")
import r10_g  # noqa: E402
prov["panels"]["g"] = r10_g.draw_g(fig, W, H, 8.0, R3 + 5.0, 167.0, H - R3 - 5.5)
prov["caption_text_moved"] = {"f_summary": summ, "a_absolute_macro_pp": absm}
letter(0, R3, "g", "Boundaries in one volume: xy and xz planes pulled from the stack, and three nuclei chosen by ΔIoU percentile")
fig.savefig(os.path.join(GA, f"{CID}.pdf")); fig.savefig(os.path.join(GA, f"{CID}.png"), dpi=600)
rep = figcheck.check(fig)
prov["overlap_issues"] = rep; prov["size_mm"] = [W, H]
json.dump(prov, open(os.path.join(GA, f"{CID}.provenance.json"), "w"), indent=1, default=str)
print("saved", CID, rep)
