"""Fig. 7, round 7 (gallery R7-Fig7): layout B of round 6 in the style of the installed Figs 2-6 (user 2026-11-11): heat-map cells
with white gutters and rounded corners, column best marked by a dot instead of a frame. Same data and checks.

Round 6 notes:
Fig. 7 redesign, round 6 (gallery R6-Fig7-A / R6-Fig7-B).  Tools comparison, speed and memory.

Only existing results are drawn; no new statistics.  Every number is read from the same files as figures/final/Fig7
(provenance figures/final/Fig7.provenance.json) or from the snapshots listed in NUMBERS.md:

  internal heatmap / radar   gt protocol, 227 test volumes, 13 datasets, pooled F1 per dataset
      S1, 2D (means of five seeds; rule T) and original Cellpose-SAM (fixed threshold):
          scripts/figs/localdata/ref/s5_test_gt_5seeds_basefixed.json  per_ds[F1@x][ds].{S1,two_d,base}, macro[F1@x][13 套/去 mesospim 12 套]
      nnU-Net, StarDist-3D (validation-selected setting): data_snapshot/compare_test_gt.json  F1@x.per_ds / macro13 / macro12
  published tools            G1, gt diameters, 44 full-pipeline test volumes, 12 datasets (no mesoSPIM), F1@0.5 only;
                             S1 / 2D = means of three seeds (as in final Fig7a): figures/final/Fig7.provenance.json panels.a.values
                             (= data_snapshot/g1_table2.md).  Hatched = tool trained on these test images (OV lines 35-36).
  u-Segment3D                figures/final/Fig7.provenance.json panels.b.values (= g3_compare.json; S1 five seeds, others seed 0)
  paired differences         data_snapshot/paired_stats_{gt,fp}.json  <comparator>|<13,12,C1>|F1@x : mean, boot_ci, confirm, per_ds.d
  speed / memory             data_snapshot/{s1,d2,base,nnunet,stardistB}.json and stardistB_x{64..256}.json
                             time = median (IQR) over the 45 full-pipeline volumes of t_ruleT (S1, 2D) or t_fixed (original, baselines),
                             i.e. the setting whose accuracy is plotted; memory = 256^3 cube peak, max(framework, nvidia-smi),
                             in GiB (the per-volume StarDist peak is a cumulative upper bound and is not used).

Ordering rules (also written to the provenance):
  columns  the 13 datasets in descending order of their mean F1@0.5 over the five internal models (S1, 2D, original,
           nnU-Net, StarDist-3D; gt protocol), then Macro 13, Macro 12.  The same order is used for the radar axes.
  rows     fixed model order: S1, 2D control, original Cellpose-SAM, nnU-Net (single fold) [, nnU-Net five-fold], StarDist-3D;
           published-tool block: S1, 2D, original, Tapenade StarDist, AnyStar, Cellos StarDist (order of final Fig7a).
  best     framed cell = highest value in that column within the block (ties all framed).

Five-fold nnU-Net: set NNUNET5F=<dir with compare_test_gt.json and paired_stats_{gt,fp}.json> to add the row/points
(key 'nnU-Net 5f' in those files).  Without it the figure uses the single fold only and draws no placeholder.

Usage: python fig7_r6.py A|B
"""
import hashlib
import json
import os
import sys

os.environ["NC_PALETTE"] = "npg"
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
import nc_style as S  # noqa: E402
from nc_style import plt  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "render3d"))
import figcheck  # noqa: E402

VER = sys.argv[1] if len(sys.argv) > 1 else "A"
assert VER in ("A", "B")
S.setup()
PAL = json.load(open(os.path.join(HERE, "..", "..", "figures", "palette.json")))
FG, MUT, RULE = PAL["text"], PAL["muted"], PAL["rule"]
COL = {"S1": "#E64B35", "2D": "#4DBBD5", "base": "#8491B4", "nnunet": "#00A087", "nnunet5f": "#00A087",
       "stardist3d": "#3C5488", "useg": "#91D1C2", "tools": "#7E6148"}
NAME = {"S1": "S1", "2D": "2D control", "base": "Original Cellpose-SAM", "nnunet": "nnU-Net", "nnunet5f": "nnU-Net, 5 folds",
        "stardist3d": "StarDist-3D", "tapenade_sd": "Tapenade StarDist", "anystar": "AnyStar", "cellos_sd": "Cellos StarDist"}
SHORT = {"S1": "S1", "2D": "2D", "base": "Original", "nnunet": "nnU-Net", "nnunet5f": "nnU-Net 5f", "stardist3d": "StarDist-3D"}
plt.rcParams.update({"hatch.linewidth": 0.4, "axes.linewidth": 0.5, "font.size": 5.5})

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SNAP = os.path.join(ROOT, "data_snapshot")
GA = os.path.join(ROOT, "figures", "gallery", "assets")
SRC = {"s5_gt": os.path.join(HERE, "localdata", "ref", "s5_test_gt_5seeds_basefixed.json"),
       "cmp_gt": os.path.join(SNAP, "compare_test_gt.json"),
       "ps_gt": os.path.join(SNAP, "paired_stats_gt.json"), "ps_fp": os.path.join(SNAP, "paired_stats_fp.json"),
       "fig7_prov": os.path.join(ROOT, "figures", "backups_before_r6fig7", "Fig7.provenance.json")}  # pre-R6 provenance, md5 51f70d3b as in R6
for f in ("s1", "d2", "base", "nnunet", "stardistB") + tuple(f"stardistB_x{s}" for s in (64, 96, 128, 160, 192, 256)):
    SRC[f"g5_{f}"] = os.path.join(SNAP, f"{f}.json")
MD5 = {k: hashlib.md5(open(v, "rb").read()).hexdigest() for k, v in SRC.items()}
S5 = json.load(open(SRC["s5_gt"]))
CMP = json.load(open(SRC["cmp_gt"]))
PS = {"gt": json.load(open(SRC["ps_gt"])), "fp": json.load(open(SRC["ps_fp"]))}
F7 = json.load(open(SRC["fig7_prov"]))["panels"]
G5 = {k[3:]: json.load(open(v)) for k, v in SRC.items() if k.startswith("g5_")}
FIVEF = os.environ.get("NNUNET5F")
if FIVEF:
    CMP5 = json.load(open(os.path.join(FIVEF, "compare_test_gt.json")))
    PS5 = {m: json.load(open(os.path.join(FIVEF, f"paired_stats_{m}.json"))) for m in ("gt", "fp")}
    for _f in ("compare_test_gt.json", "paired_stats_gt.json", "paired_stats_fp.json"):
        MD5[f"nnunet5f/{_f}"] = hashlib.md5(open(os.path.join(FIVEF, _f), "rb").read()).hexdigest()

MET = ("F1@0.5", "F1@0.7")
DSS = sorted(CMP["F1@0.5"]["per_ds"]["S1"])
assert len(DSS) == 13


def internal(m, met, ds):
    """Per-dataset (ds) or macro ('macro13' / 'macro12') pooled F1, gt protocol, 227 volumes."""
    if m in ("S1", "2D", "base"):
        key = {"S1": "S1", "2D": "two_d", "base": "base"}[m]
        if ds == "macro13":
            return S5["macro"][met]["13 套"][key]
        if ds == "macro12":
            return S5["macro"][met]["去 mesospim 12 套"][key]
        return S5["per_ds"][met][ds][key]
    src, key = (CMP5, "nnU-Net 5f") if m == "nnunet5f" else (CMP, {"nnunet": "nnU-Net", "stardist3d": "StarDist"}[m])
    if ds in ("macro13", "macro12"):
        return src[met][ds][key]
    return src[met]["per_ds"][key][ds]


# consistency with the final Fig7c values (13-set macro, gt) and with compare_test (S1 five seeds, rule T)
for m, k in (("S1", "S1"), ("2D", "2D"), ("base", "base"), ("nnunet", "nnunet"), ("stardist3d", "stardist3d")):
    for met in MET:
        assert abs(internal(m, met, "macro13") - F7["c"]["values"]["abs_gt13"][f"{k}/{met}"]) < 6e-5, (m, met)
for met in MET:
    for ds in DSS:
        assert abs(S5["per_ds"][met][ds]["S1"] - CMP[met]["per_ds"]["S1"][ds]) < 1e-9

MODELS = ["S1", "2D", "base", "nnunet"] + (["nnunet5f"] if FIVEF else []) + ["stardist3d"]
mean05 = {ds: np.mean([internal(m, "F1@0.5", ds) for m in ("S1", "2D", "base", "nnunet", "stardist3d")]) for ds in DSS}
ORDER = sorted(DSS, key=lambda d: (-mean05[d], d))
COLS = ORDER + ["macro13", "macro12"]
CLAB = {d: S.DS_NAME[d] for d in DSS}
CLAB.update({"mesospim": "mesoSPIM†", "macro13": "Macro 13", "macro12": "Macro 12", "nisnet_zf": "NISNet3D zf."})

# ---------------------------------------------------------------- figure frame
W = 183.0
H = 170.0 if VER == "A" else 168.0
fig = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor="white")
prov = {"figure": f"R6-Fig7-{VER}", "sources_md5": MD5, "five_fold_included": bool(FIVEF), "panels": {},
        "ordering": {"columns": "13 datasets by descending mean F1@0.5 over S1, 2D, original, nnU-Net, StarDist-3D (gt protocol); "
                                "then Macro 13, Macro 12; radar axes use the same order, clockwise from 12 o'clock",
                     "column_order": ORDER,
                     "rows": "fixed: S1, 2D control, original Cellpose-SAM, nnU-Net (single fold)[, nnU-Net five folds], StarDist-3D; "
                             "published-tool block in the order of final Fig7a",
                     "best": "framed cell = column maximum within its block (ties all framed)"}}


def ax_mm(x, y, w, h, **kw):
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)


def letter(x, y, s, title=None):
    fig.text(x / W, 1 - y / H, s, fontsize=7, fontweight="bold", ha="left", va="top", color=FG)
    if title:
        fig.text((x + 3.6) / W, 1 - (y + 0.35) / H, title, fontsize=6.3, fontweight="bold", ha="left", va="top", color=FG)


SEQ = LinearSegmentedColormap.from_list("f1seq", ["#ffffff", "#91D1C2", "#4DBBD5", "#3C5488"])   # NPG cool ramp (cells are not model-coded)
VMIN, VMAX = 0.0, 1.0


def cell_colour(v):
    c = SEQ((v - VMIN) / (VMAX - VMIN))
    lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    return c, ("white" if lum < 0.55 else FG)


def heat_block(ax, rows, rowlab, rowcol, cols, getv, hatch=None, header=True, fs=5.0, rot=50):
    """rows: keys; getv(row, col) -> value or None (not evaluated). Returns {row: {col: value}}."""
    n, k = len(rows), len(cols)
    ax.set_xlim(-0.5, k - 0.5); ax.set_ylim(n - 0.5, -0.5); ax.axis("off")
    pos = ax.get_position(); ASP = [(n / (pos.height * H)) / (k / (pos.width * W))]
    vals = {}
    for c, col in enumerate(cols):
        colv = [getv(r, col) for r in rows]
        good = [v for v in colv if v is not None]
        best = max(good) if good else None
        for i, (r, v) in enumerate(zip(rows, colv)):
            x0 = c + (0.18 if col.startswith("macro") else 0)    # small gap before the macro columns
            if v is None:
                ax.add_patch(FancyBboxPatch((x0 - 0.44, i - 0.42), 0.88, 0.84, boxstyle="round,pad=0,rounding_size=0.10", fc="#f4f4f2", ec="none", mutation_aspect=ASP[0]))
                ax.text(x0, i, "–", ha="center", va="center", fontsize=fs, color=MUT)
                continue
            vals.setdefault(r, {})[col] = round(float(v), 4)
            fc, tc = cell_colour(v)
            ax.add_patch(FancyBboxPatch((x0 - 0.44, i - 0.42), 0.88, 0.84, boxstyle="round,pad=0,rounding_size=0.10", fc=fc, ec="none", zorder=2, mutation_aspect=ASP[0]))
            if hatch and (r, col) in hatch:
                ax.add_patch(FancyBboxPatch((x0 - 0.44, i - 0.42), 0.88, 0.84, boxstyle="round,pad=0,rounding_size=0.10", fc="none", ec=COL["tools"], lw=0, hatch="////", zorder=3, mutation_aspect=ASP[0]))
            ax.text(x0 - 0.05, i, f"{v:.2f}"[1:] if v < 1 else "1.00", ha="center", va="center", fontsize=fs, color=tc, zorder=4,
                    fontweight="bold" if col.startswith("macro") else "normal",
                    bbox=dict(fc=fc, ec="none", pad=0.3) if hatch and (r, col) in hatch else None)
            if best is not None and abs(v - best) < 1e-12:
                ax.plot(x0 + 0.31, i - 0.25, "o", ms=1.4, mfc="white" if tc == "white" else FG, mec="none", zorder=5)
    for i, r in enumerate(rows):
        ax.text(-0.75, i, rowlab[r], ha="right", va="center", fontsize=5.5, color=FG)
        if rowcol.get(r):
            ax.add_patch(Rectangle((-0.62, i - 0.36), 0.12, 0.72, fc=rowcol[r], ec="none", clip_on=False))
    if header:
        for c, col in enumerate(cols):
            x0 = c + (0.18 if col.startswith("macro") else 0)
            ax.text(x0 - (0.15 if rot == 90 else 0), -0.75, CLAB[col], ha="left", va="bottom", rotation=rot, rotation_mode="anchor", fontsize=5.3, color=FG,
                    fontweight="bold" if col.startswith("macro") else "normal")
    return vals


def colourbar(x, y, w, label):
    axc = ax_mm(x, y, w, 1.3)
    axc.imshow(np.linspace(0, 1, 256)[None, :], cmap=SEQ, aspect="auto", extent=(VMIN, VMAX, 0, 1))
    axc.set_yticks([]); axc.set_xticks([0, 0.5, 1], ["0", "0.5", "1"]); axc.tick_params(axis="x", length=1.2, pad=0.8, labelsize=5)
    for sp in axc.spines.values():
        sp.set_visible(False)
    fig.text((x + w + 1.2) / W, 1 - (y + 0.65) / H, label, fontsize=5, color=FG, va="center")


# ---------------------------------------------------------------- panel: internal heatmap (both metrics)
def panel_internal_heat(x, y, cw, rh, metrics=MET, header=True, title_x=None, short=False, rot=50):
    out = {}
    yy = y
    rowlab = {m: (SHORT[m] if short else (NAME[m] if m != "base" else "Original")) for m in MODELS}
    for j, met in enumerate(metrics):
        h = rh * len(MODELS)
        ax = ax_mm(x, yy, cw * (len(COLS) + 0.18), h)
        out[met] = heat_block(ax, MODELS, rowlab, {m: COL[m] for m in MODELS}, COLS, lambda r, c: internal(r, met, c),
                              header=(header and j == 0), rot=rot)
        fig.text((x - 1.0 - (len(max(rowlab.values(), key=len)) * 1.05) - 4) / W, 1 - (yy + h / 2) / H, met, rotation=90,
                 fontsize=5.8, fontweight="bold", color=FG, ha="center", va="center")
        yy += h + 1.6
    return out, yy


def tools_values():
    a = F7["a"]["values"]
    rows = ["S1_mean", "D2_mean", "base", "tapenade_sd", "anystar", "cellos_sd"]
    lab = {"S1_mean": "S1 (3 seeds)", "D2_mean": "2D (3 seeds)", "base": "Original", "tapenade_sd": "Tapenade StarDist",
           "anystar": "AnyStar", "cellos_sd": "Cellos StarDist"}
    rc = {"S1_mean": COL["S1"], "D2_mean": COL["2D"], "base": COL["base"], "tapenade_sd": COL["tools"], "anystar": COL["tools"],
          "cellos_sd": COL["tools"]}

    def getv(r, c):
        if c == "mesospim" or c == "macro13":
            return None
        return a[f"{'macro' if c == 'macro12' else c}/{r}"]
    hatch = {("tapenade_sd", "tapenade"), ("cellos_sd", "cellos")}
    return rows, lab, rc, getv, hatch


def panel_tools_heat(x, y, cw, rh, header=False, short=False):
    rows, lab, rc, getv, hatch = tools_values()
    if short:
        lab = {**lab, "tapenade_sd": "Tapenade SD", "cellos_sd": "Cellos SD", "S1_mean": "S1 (3 s.)", "D2_mean": "2D (3 s.)"}
    ax = ax_mm(x, y, cw * (len(COLS) + 0.18), rh * len(rows))
    v = heat_block(ax, rows, lab, rc, COLS, getv, hatch=hatch, header=header)
    return v, y + rh * len(rows)


# ---------------------------------------------------------------- panel: radar (F1@0.7 per dataset, common 0-1 scale)
def panel_radar(x, y, size, met="F1@0.7"):
    ax = ax_mm(x, y, size, size, projection="polar")
    th = np.linspace(0, 2 * np.pi, len(ORDER), endpoint=False)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0], ["", "0.5", "", "1"])
    ax.set_rlabel_position(-360 / len(ORDER) / 2)
    ax.tick_params(axis="y", labelsize=5, colors=MUT, pad=0)
    for t in ax.get_yticklabels():
        t.set_bbox(dict(fc="white", ec="none", pad=0.4, alpha=0.85))
    ax.set_xticks(th, [CLAB[d] for d in ORDER])
    ax.tick_params(axis="x", labelsize=5.2, pad=1.2, colors=FG)
    ax.grid(color=RULE, lw=0.4); ax.spines["polar"].set_color("#b9b9b5"); ax.spines["polar"].set_linewidth(0.4)
    vals = {}
    for m in MODELS[::-1]:
        v = np.array([internal(m, met, d) for d in ORDER])
        vals[m] = dict(zip(ORDER, np.round(v, 4).tolist()))
        tt, vv = np.r_[th, th[:1]], np.r_[v, v[:1]]
        lw = 1.0 if m == "S1" else 0.7
        ls = (0, (2.2, 1.2)) if m == "nnunet5f" else "-"
        ax.plot(tt, vv, color=COL[m], lw=lw, ls=ls, zorder=4 if m == "S1" else 3)
        if m == "S1":
            ax.fill(tt, vv, color=COL[m], alpha=0.10, lw=0, zorder=2)
        ax.scatter(th, v, s=3.5, color=COL[m], lw=0, zorder=5)
    for lab in ax.get_xticklabels():
        lab.set_horizontalalignment("center")
    return ax, vals


# ---------------------------------------------------------------- panel: accuracy x time x memory (bubble / Pareto)
def gib(js, side):
    d = js["X"][side]
    return max(d.get("peak_torch_mib", d.get("peak_fw_mib", 0)), d.get("peak_smi_mib", 0)) / 1024


def speed_points():
    tkey = {"S1": ("s1", "t_ruleT"), "2D": ("d2", "t_ruleT"), "base": ("base", "t_fixed"), "nnunet": ("nnunet", "t_fixed"),
            "stardist3d": ("stardistB", "t_fixed")}
    mem = {"S1": gib(G5["s1"], "256_v3"), "2D": gib(G5["d2"], "256_vNone"), "base": gib(G5["base"], "256_vNone"),
           "nnunet": gib(G5["nnunet"], "256"), "stardist3d": gib(G5["stardistB_x256"], "256")}
    pts = {}
    for m in ("S1", "2D", "base", "nnunet", "stardist3d"):
        f, k = tkey[m]
        t = np.array([p[k] for p in G5[f]["P"].values()])
        q1, med, q3 = np.percentile(t, (25, 50, 75))
        pts[m] = dict(t=float(med), q1=float(q1), q3=float(q3), mem=float(mem[m]), f05=float(internal(m, "F1@0.5", "macro13")),
                      f07=float(internal(m, "F1@0.7", "macro13")), time_key=k, n=int(len(t)))
    # consistency with final Fig7d
    pv = F7["d"]["values"]["per_volume"]
    for m, lab in (("S1", "S1, rule T"), ("2D", "2D, rule T"), ("base", "Original"), ("nnunet", "nnU-Net"), ("stardist3d", "StarDist-3D")):
        assert abs(pts[m]["t"] - pv[lab]["median_s"]) < 0.01, m
    return pts


def pareto(pts, key):
    """non-dominated set: no other model is at least as fast and at least as accurate (strictly better in one)."""
    front = []
    for m, p in pts.items():
        dom = any((q["t"] <= p["t"] and q[key] >= p[key]) and (q["t"] < p["t"] or q[key] > p[key]) for o, q in pts.items() if o != m)
        if not dom:
            front.append(m)
    return sorted(front, key=lambda m: pts[m]["t"])


def panel_pareto(x, y, w, h):
    pts = speed_points()
    ax = ax_mm(x, y, w, h)
    ax.set_xscale("log")
    ax.set_xlim(11, 115); ax.set_ylim(0.2, 0.84); ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xticks([20, 50, 100], ["20", "50", "100"]); ax.minorticks_off()
    ax.set_xlabel("Median time per volume (s, log scale; 45 volumes)", fontsize=5.5)
    ax.set_ylabel("Macro F1, 13 datasets (gt protocol)", fontsize=5.5)
    ax.grid(lw=0.3, color=RULE); ax.tick_params(labelsize=5)
    fronts = {}
    for key, ls in (("f05", (0, (3, 1.5))), ("f07", (0, (1, 1.2)))):
        fr = pareto(pts, key)
        fronts[key] = fr
        xs = [pts[m]["t"] for m in fr]; ys = [pts[m][key] for m in fr]
        # staircase: from the fastest front member, accuracy can only be bought with time
        sx, sy = [xs[0]], [ys[0]]
        for xi, yi in zip(xs[1:], ys[1:]):
            sx += [xi, xi]; sy += [sy[-1], yi]
        sx.append(115); sy.append(sy[-1])
        ax.plot(sx, sy, color=MUT, lw=0.6, ls=ls, zorder=1)
    scale = 9.0                                                # marker area (pt^2) per GiB
    for m, p in pts.items():
        ax.plot([p["t"], p["t"]], [p["f07"], p["f05"]], color=COL[m], lw=0.5, zorder=2)
        ax.scatter(p["t"], p["f05"], s=scale * p["mem"], color=COL[m], ec="white", lw=0.4, zorder=4, alpha=0.95)
        ax.scatter(p["t"], p["f07"], s=scale * p["mem"], facecolor="white", ec=COL[m], lw=0.8, zorder=4)
    off = {"S1": (-5, 5), "2D": (6, -6), "base": (-6, 0), "nnunet": (0, 6.5), "stardist3d": (-6, 0)}
    ha = {"S1": "right", "base": "right", "stardist3d": "right", "nnunet": "center"}
    for m, p in pts.items():
        dx, dy = off[m]
        ax.annotate(SHORT[m], (p["t"], p["f05"]), xytext=(dx, dy), textcoords="offset points", fontsize=5.3, color=COL[m] if m != "useg" else FG,
                    fontweight="bold" if m == "S1" else "normal", ha=ha.get(m, "left"), va="center")
    # legends: metric (filled/open), front (line styles), memory (bubble sizes)
    h1 = [Line2D([], [], marker="o", ls="", color=MUT, ms=3.2, label="F1@0.5"),
          Line2D([], [], marker="o", ls="", mfc="white", mec=MUT, mew=0.8, ms=3.2, label="F1@0.7"),
          Line2D([], [], color=MUT, lw=0.6, ls=(0, (3, 1.5)), label="front, F1@0.5"),
          Line2D([], [], color=MUT, lw=0.6, ls=(0, (1, 1.2)), label="front, F1@0.7")]
    lg = ax.legend(handles=h1, loc="lower right", fontsize=5, handletextpad=0.3, borderaxespad=0.3, labelspacing=0.25,
                   handlelength=1.6, frameon=False)
    ax.add_artist(lg)
    h2 = [plt.scatter([], [], s=scale * g, facecolor="none", ec=MUT, lw=0.6, label=f"{g} GiB") for g in (2, 4, 8)]
    ax.legend(handles=h2, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=5, title="peak GPU memory\n(256³ cube)",
              title_fontsize=5, handletextpad=0.4, labelspacing=0.75, borderaxespad=0.2, frameon=False)
    return ax, {"points": {m: {k: round(v, 4) if isinstance(v, float) else v for k, v in p.items()} for m, p in pts.items()},
                "pareto_front": fronts,
                "note": "x: median of per-volume time (IQR in the points entry, not drawn), t_ruleT for S1/2D (rule T is what is scored), t_fixed for original, "
                        "nnU-Net, StarDist-3D; y: 13-set macro F1, gt protocol, 227 volumes (filled F1@0.5, open F1@0.7); "
                        "bubble area proportional to the 256^3-cube peak GPU memory (GiB, max of framework and nvidia-smi); "
                        "same GPU (RTX 5090), host load not controlled; front = models not dominated in (time, F1)"}


# ---------------------------------------------------------------- panel: forest plot of paired differences
def panel_forest(x, y, w, h, comps=(("nnU-Net", "nnunet"), ("StarDist", "stardist3d")), title=True):
    rows = [(p, s) for p in ("gt", "fp") for s in ("13", "12", "C1")]
    rlab = {("gt", "13"): "gt, 13", ("gt", "12"): "gt, 12", ("gt", "C1"): "gt, C1",
            ("fp", "13"): "full, 13", ("fp", "12"): "full, 12", ("fp", "C1"): "full, C1"}
    if FIVEF:
        comps = (("nnU-Net", "nnunet"), ("nnU-Net 5f", "nnunet5f")) + tuple(c for c in comps if c[0] == "StarDist")
    nC = len(comps)
    pw = (w - 2.0 * (nC - 1)) / nC
    out = {}
    for ci, (key, m) in enumerate(comps):
        for mi, met in enumerate(MET):
            ax = ax_mm(x + ci * (pw + 2.0) + mi * pw / 2, y, pw / 2 - 1.2, h)
            ax.set_ylim(len(rows) - 0.5, -0.5)
            for i in range(len(rows)):
                if i % 2 == 0:
                    ax.axhspan(i - 0.5, i + 0.5, color="#f4f4f2", lw=0, zorder=0)
            src = PS5 if m == "nnunet5f" else PS
            allpts = []
            for i, (p, s) in enumerate(rows):
                e = src[p][f"{key}|{s}|{met}"]
                d = np.array([v["d"] for v in e["per_ds"].values()])
                allpts.extend(d.tolist())
                jit = (np.arange(len(d)) - (len(d) - 1) / 2) / max(len(d) - 1, 1) * 0.42
                ax.scatter(d, i + jit, s=1.4, color=COL["S1"], alpha=0.35, lw=0, zorder=2)
                lo, hi = e["boot_ci"]
                conf = ("确认" in e["confirm"]) and ("未" not in e["confirm"])
                col = COL["S1"] if met == "F1@0.5" else PAL["models"]["S1_dark"]
                ax.plot([lo, hi], [i, i], color=col, lw=0.9, zorder=3, solid_capstyle="butt")
                ax.scatter(e["mean"], i, s=11, marker="s", color=col if conf else "white", ec=col, lw=0.7, zorder=4)
                out[f"{key}|{p}|{s}|{met}"] = {"mean": round(e["mean"], 4), "boot_ci": [round(lo, 4), round(hi, 4)],
                                               "confirm": e["confirm"], "per_ds_d": {k: round(v["d"], 4) for k, v in e["per_ds"].items()}}
            ax.axvline(0, color=MUT, lw=0.5, ls=(0, (3, 2)), zorder=1)
            lim = max(abs(min(allpts)), abs(max(allpts))) * 1.05
            if m == "stardist3d":
                ax.set_xlim(-0.12, 0.55); ax.set_xticks([0, 0.25, 0.5], ["0", ".25", ".5"])
            else:
                ax.set_xlim(-0.2, 0.2)
                if FIVEF:   # three comparators -> narrower panels; two ticks avoid touching labels
                    ax.set_xticks([-0.1, 0.1], ["−.1", ".1"])   # dashed line marks 0
                else:
                    ax.set_xticks([-0.15, 0, 0.15], ["−.15", "0", ".15"])
            ax.tick_params(labelsize=5, pad=1); ax.spines["left"].set_visible(False)
            ax.set_yticks(range(len(rows)), [rlab[r] for r in rows] if (ci == 0 and mi == 0) else [""] * len(rows))
            ax.tick_params(axis="y", length=0)
            ax.set_title(met, fontsize=5.3, pad=1.5, color=FG)
            if mi == 0:
                fig.text((x + ci * (pw + 2.0) + pw / 2 - 0.6) / W, 1 - (y - 4.3) / H, f"S1 − {SHORT[m]}", ha="center",
                         va="bottom", fontsize=5.8, fontweight="bold", color=FG)
            _ = lim
    fig.text((x + w / 2) / W, 1 - (y + h + 5.2) / H, "Paired difference in macro F1 (S1 minus baseline)", ha="center", va="top",
             fontsize=5.5, color=FG)
    return out


# ---------------------------------------------------------------- panel: u-Segment3D dumbbell
def panel_useg(x, y, w, h):
    """Dumbbell: direct inference (filled) -> u-Segment3D reconstruction (open), per protocol x metric; S1 = red tick."""
    v = F7["b"]["values"]
    groups = [(p, met) for p in ("gt", "fp") for met in ("tp5", "tp7")]
    ys, yl = {}, []
    yy = 0.0
    for g in groups:
        yy += 0.9                                  # room for the group header
        for m in ("d2", "base"):
            ys[(g, m)] = yy; yl.append((yy, "2D" if m == "d2" else "Original")); yy += 1.0
        yy += 0.15
    ax = ax_mm(x, y, w, h)
    ax.set_ylim(yy - 0.4, -0.2)
    ax.set_xlim(0.25, 0.92); ax.set_xticks([0.3, 0.5, 0.7, 0.9], ["0.3", "0.5", "0.7", "0.9"])
    ax.grid(axis="x", lw=0.3, color=RULE); ax.tick_params(labelsize=5)
    for gi, (p, met) in enumerate(groups):
        y0 = ys[((p, met), "d2")]
        if gi % 2 == 0:
            ax.axhspan(y0 - 1.15, y0 + 1.55, color="#f4f4f2", lw=0, zorder=0)
        ax.text(0.27, y0 - 0.72, f"{'gt protocol' if p == 'gt' else 'full pipeline'}, {'F1@0.5' if met == 'tp5' else 'F1@0.7'}",
                fontsize=5, color=MUT, va="center", ha="left")
        s1 = v[f"{p}/{met}/S1"]
        ax.plot([s1, s1], [y0 - 0.4, y0 + 1.4], color=COL["S1"], lw=0.9, zorder=2)
        for m in ("d2", "base"):
            yv = ys[((p, met), m)]
            a, b = v[f"{p}/{met}/{m}"], v[f"{p}/{met}/U_{m}"]
            c = COL["2D" if m == "d2" else "base"]
            ax.annotate("", xy=(b, yv), xytext=(a, yv), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=c, shrinkA=2.0, shrinkB=2.0,
                                                                         mutation_scale=4.5), zorder=3)
            ax.scatter(a, yv, s=10, color=c, lw=0, zorder=4)
            ax.scatter(b, yv, s=10, facecolor="white", ec=c, lw=0.8, zorder=4)
            xt = max(a, b, s1 if m == "d2" else 0) + 0.02
            ax.text(xt, yv, f"{b - a:+.3f}".replace("-", "\u2212"), fontsize=5, va="center", color=c)
    ax.set_yticks([t for t, _ in yl], [l for _, l in yl]); ax.tick_params(axis="y", length=0, labelsize=5.2)
    ax.set_xlabel("Macro F1, 13 datasets", fontsize=5.5)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=MUT, ms=2.8, label="direct"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=MUT, mew=0.8, ms=2.8, label="+ u-Segment3D"),
                       Line2D([], [], color=COL["S1"], lw=0.9, label="S1")],
              loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, fontsize=5, handletextpad=0.25, columnspacing=0.6,
              borderaxespad=0.3, handlelength=1.0)
    return {k: v[k] for k in v}


# ================================================================= layouts
if VER == "A":
    # a  heatmaps (internal both metrics + published tools), full width
    CW, RH = 9.4, 3.15
    HX = 31.0
    letter(1.0, 1.5, "a", "Model × dataset (F1, higher is better; frame = best in column)")
    pa, yy = panel_internal_heat(HX, 21.5, CW, RH)
    fig.text(1.5 / W, 1 - (yy - 0.6) / H, "", fontsize=5)
    rows_t = 6
    fig.text((HX - 27.5) / W, 1 - (yy + 1.4) / H, "Published tools  ·  44 full-pipeline volumes, gt diameters, F1@0.5 (S1, 2D: 3 seeds)",
             fontsize=5.5, fontweight="bold", color=FG, va="top")
    pt, yy2 = panel_tools_heat(HX, yy + 4.4, CW, RH)
    colourbar(HX + 0.5, yy2 + 1.6, 26, "F1")
    fig.text((HX + 45) / W, 1 - (yy2 + 2.3) / H,
             "Top: gt protocol, 227 test volumes; S1, 2D = mean of 5 seeds (rule T); original: fixed threshold.\n"
             "Hatched: tool trained on these test images.  † mesoSPIM: one test volume.  –: not evaluated.",
             fontsize=5, color=MUT, va="center", linespacing=1.25)
    prov["panels"]["a"] = {"internal": pa, "published_tools": pt, "hatched": [["tapenade_sd", "tapenade"], ["cellos_sd", "cellos"]],
                           "colour_scale": "fixed 0-1, white -> #91D1C2 -> #4DBBD5 -> #3C5488 (cells are not model-coded; model identity = row swatch)"}
    YB = yy2 + 13.0
    # b  Pareto bubble, c forest, d dumbbell
    letter(1.0, YB, "b", "Accuracy, time and memory")
    pb_ax, pb = panel_pareto(13.0, YB + 6.0, 52.0, 170 - YB - 18.5)
    prov["panels"]["b"] = pb
    letter(73.0, YB, "c", "Paired differences (95% CI)")
    prov["panels"]["c"] = panel_forest(84.0, YB + 10.5, 55.0, 170 - YB - 27.0)
    fig.text(84.0 / W, 1 - (170 - 5.0) / H, ("square: macro difference (filled = confirmed by the five-seed rule, either sign);\ndots: datasets; nnU-Net 5f not timed in c" if FIVEF else "square: macro difference (filled = confirmed by the five-seed rule); dots: datasets"),
             fontsize=5, color=MUT, va="top")
    letter(141.5, YB, "d", "u-Segment3D")
    prov["panels"]["d"] = panel_useg(152.5, YB + 9.5, 29.5, 170 - YB - 20.5)
else:
    # a  radar (F1@0.7) beside the F1@0.7 heatmap with the same column order; b published tools; c Pareto; d forest; e dumbbell
    letter(1.0, 1.5, "a", "Model × dataset (gt protocol, 227 volumes)")
    rax, rv = panel_radar(9.5, 15.0, 45.0)
    hl = [Line2D([], [], color=COL[m], lw=1.0 if m == "S1" else 0.7, ls=(0, (2.2, 1.2)) if m == "nnunet5f" else "-", label=SHORT[m])
          for m in MODELS]
    fig.legend(handles=hl, loc="upper left", bbox_to_anchor=(3.0 / W, 1 - 69.0 / H), ncol=len(MODELS), fontsize=5,
               handlelength=1.3, columnspacing=0.7, handletextpad=0.3, frameon=False)
    CW, RH = 6.42, 3.0
    HX = 84.0
    pa, yy = panel_internal_heat(HX, 21.0, CW, RH, short=True, rot=90)
    letter(HX - 19.0, yy + 0.6, "b", "Published tools (44 volumes, gt diameters, F1@0.5)")
    pt, yy2 = panel_tools_heat(HX, yy + 5.0, CW, RH, short=True)
    colourbar(HX + 0.5, yy2 + 1.5, 22, "F1  (dot: best in column; hatched: tool trained on these test images)")
    fig.text(3.0 / W, 1 - 74.0 / H, "Radar = F1@0.7 rows of the heatmap (same axis order, common 0\u20131 scale).\n"
             "Heatmap: gt protocol, 227 test volumes; S1, 2D: 5 seeds; tools block: S1, 2D 3 seeds.\n\u2020 mesoSPIM: one test volume.",
             fontsize=5, color=MUT, va="top", linespacing=1.25)
    prov["panels"]["a"] = {"radar_F1@0.7": rv, "heatmap": pa, "radar_scale": "0-1 radial, rings at 0.25 steps, same for all axes"}
    prov["panels"]["b"] = {"published_tools": pt, "hatched": [["tapenade_sd", "tapenade"], ["cellos_sd", "cellos"]]}
    YB = max(yy2 + 9.0, 75.0)
    letter(1.0, YB, "c", "Accuracy, time and memory")
    pc_ax, pc = panel_pareto(13.0, YB + 6.0, 52.0, H - YB - 18.5)
    prov["panels"]["c"] = pc
    letter(73.0, YB, "d", "Paired differences (95% CI)")
    prov["panels"]["d"] = panel_forest(84.0, YB + 10.5, 55.0, H - YB - 27.0)
    fig.text(84.0 / W, 1 - (H - 5.0) / H, ("square: macro difference (filled = confirmed by the five-seed rule, either sign);\ndots: datasets; nnU-Net 5f not timed in c" if FIVEF else "square: macro difference (filled = confirmed by the five-seed rule); dots: datasets"),
             fontsize=5, color=MUT, va="top")
    letter(141.5, YB, "e", "u-Segment3D")
    prov["panels"]["e"] = panel_useg(152.5, YB + 9.5, 29.5, H - YB - 20.5)

# ---------------------------------------------------------------- checks and export
issues = figcheck.check(fig)
big = []
for t in fig.findobj(lambda o: hasattr(o, "get_fontsize") and hasattr(o, "get_text")):
    if t.get_text().strip() and t.get_visible() and t.get_fontsize() > 7.0 + 1e-6:
        big.append(f"FONT {t.get_fontsize():.1f} pt > 7: '{t.get_text()[:30]}'")
lw_bad = [f"{type(a).__name__} lw {a.get_linewidth():.2f}" for a in fig.findobj(Line2D)
          if a.get_visible() and a.get_linewidth() and not (0.25 - 1e-6 <= a.get_linewidth() <= 1.0 + 1e-6) and a.get_xydata().size]
prov["checks"] = {"overlap_and_min_font": issues, "font_above_7pt": sorted(set(big)), "line_width_outside_0.25_1pt": sorted(set(lw_bad)),
                  "size_mm": [W, H], "background": "white", "generative_ai": "none (matplotlib only, data-driven)"}
print("overlap/min-font issues:", issues)
print("font > 7 pt:", sorted(set(big)))
print("line widths outside 0.25-1:", sorted(set(lw_bad))[:10])
name = "R7-Fig7"
fig.savefig(os.path.join(GA, f"{name}.pdf"), facecolor="white")
fig.savefig(os.path.join(GA, f"{name}.png"), dpi=300, facecolor="white")
prov["script"] = {"path": "paper/scripts/figs/fig7_r7.py", "md5": hashlib.md5(open(__file__, "rb").read()).hexdigest()}
json.dump(prov, open(os.path.join(GA, f"{name}.provenance.json"), "w"), indent=1, ensure_ascii=False)
print("saved", name, f"{W:.0f} x {H:.0f} mm")
