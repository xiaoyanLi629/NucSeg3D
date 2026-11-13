"""R9 Fig. 4 (gallery R9-Fig4): R8 plus (user 2026-11-11, reference figures in 顶刊美学配图: no black/white patchwork)
3D renders composited on the panel tint instead of black tiles, and the two image panels (a, e) set on a soft rounded tint
(TINT) so that dark raw-image tiles sit in a light container rather than directly on white.

Based on R8 Fig. 4 (gallery R8-Fig4): the installed R6-Fig4-B layout kept (user 2026-11-11: "原版就很好 ... 对里面每个子图的形式做一些调整");
only the form of each panel changes. Same data, numbers and selection rule (FIG4R6_RULE.md); builders of fig4r6 / fig4r7 imported.
  a  degradation grid on the white page (dark only inside the image tiles; fig4r7.panel_a), dark labels
  b  k x noise heat maps in the Fig. 2c / Fig. 3a style: white gutters, rounded cells, softer diverging map; the black frame for
     "all three seed differences share the sign" becomes a small dot (bold value kept)
  c  dumbbell unchanged (fig4r6.dumbbell)
  d  dataset x condition map in the same cell style (no marks, as in R6-B)
  e  white band (fig4r7.panel_e tiles, dark only inside each tile) with the error-class legend in two columns at the right
Run: python fig4r8.py"""
import json
import os

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

import fig4r6 as F
import fig4r7 as F7
from fig4r6 import COND, CS, G, KS, MET, NUMS, P, cell_text, dumbbell, letter, mm_ax, plt, text
import rstyle as R

F.H = F7.H = 181          # R9: 11 mm taller than R6-B so that e can fill the full width at a legible tile height
W, H = F.W, F.H
_lo, _mid, _hi = P["diverging"]
CMAP2 = LinearSegmentedColormap.from_list("npgdiv2", [_lo, "#93a5cb", "#f7f7f5", "#f3a898", _hi])


def cell(ax, xc, yc, v, vmax, w=0.9, h=0.84, asp=1.0, **kw):
    col = CMAP2(0.5 + 0.5 * np.clip(v / vmax, -1, 1))
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.10", fc=col, ec="none",
                                mutation_aspect=asp, zorder=2))
    lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
    return lum


def heat_kn2(fig, x, y, w, h, met, lab, vmax):
    ax = mm_ax(fig, x, y, w, h)
    M = np.array([[G["conditions"][f"z{k}_n{c}"][f"all_{met}"]["diff"][0] for c in CS] for k in KS])
    Rg = [[G["conditions"][f"z{k}_n{c}"][f"all_{met}"]["diff"][1:] for c in CS] for k in KS]
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(3.5, -0.5)
    asp = (4 / h) / (3 / w)
    for i in range(4):
        for j in range(3):
            v = M[i, j]; lo, hi = Rg[i][j]; ex = lo > 0 or hi < 0
            lum = cell(ax, j, i, v, vmax, 0.92, 0.88, asp)
            dark = lum < 0.55
            ax.text(j - 0.03, i, cell_text(v), ha="center", va="center", fontsize=6, color="white" if dark else R.FG,
                    fontweight="bold" if ex else "normal", zorder=3)
            if ex:
                ax.plot(j + 0.36, i - 0.30, "o", ms=1.7, mfc="white" if dark else R.FG, mec="none", zorder=4)
            NUMS[f"b/{met}/k{KS[i]}/c{CS[j]}"] = dict(mean=float(v), seed_range=[float(lo), float(hi)], framed=bool(ex))
    ax.set_xticks(range(3), ["0", "0.25", "0.5"]); ax.set_yticks(range(4), [str(k) for k in KS])
    ax.set_xlabel("Noise c (× nuclear contrast)", labelpad=1.5); ax.set_ylabel("Axial subsampling k", labelpad=1.5)
    ax.tick_params(length=0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Δ macro {lab} (S1 − 2D)", fontsize=6.5, pad=2.5, fontweight="bold")
    return ax


def cbar2(fig, x, y, h, vmax):
    cax = mm_ax(fig, x, y, 1.3, h)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP2, norm=plt.Normalize(-vmax, vmax)), cax=cax)
    cb.set_ticks([-vmax, 0, vmax], labels=[cell_text(-vmax, 2), "0", cell_text(vmax, 2)])
    cb.ax.tick_params(labelsize=5, length=1.2, width=0.4, pad=1); cb.outline.set_visible(False)


TINT = "#eef2f7"


def tint_box(fig, x, y, w, h, col=TINT):
    fig.add_artist(FancyBboxPatch((x / W, 1 - (y + h) / H), w / W, h / H, boxstyle="round,pad=0,rounding_size=0.012",
                                  transform=fig.transFigure, fc=col, ec="none", zorder=-10, mutation_aspect=W / H))


def on_bg(im, size, col):
    from PIL import Image
    w, h = size
    sc = min(w * 0.96 / im.width, h * 0.96 / im.height)
    r = im.resize((max(1, int(round(im.width * sc))), max(1, int(round(im.height * sc)))), Image.LANCZOS)
    t = Image.new("RGBA", (w, h), tuple(int(round(255 * c)) for c in __import__("matplotlib.colors", fromlist=["x"]).to_rgb(col)) + (255,))
    t.alpha_composite(r, ((w - r.width) // 2, (h - r.height) // 2))
    return t.convert("RGB")


def panel_e9(fig, x, y, h, wmax, gap0=2.4):
    """2D 3D | S1 3D (renders on the tint, no tile) | 2D xz | S1 xz (image data, thin frame), sized to fill wmax (mm)."""
    import r6_render as V
    cnt = {}
    TL = 4.2
    xzs = {}
    for m in ("2D", "S1"):
        fx = F.fx_for((h - TL) * 1.35)
        xzs[m] = F.xz_overlay(m, 8, fx)
    r = xzs["2D"][0].shape[1] / xzs["2D"][0].shape[0]
    th = min(h - TL, (wmax - 3 * gap0 - 2.5) / (2 * 1.12 + 2 * r))
    used = th * (2 * 1.12 + 2 * r)
    gap = gap0; gmid = gap0 + max(0.0, wmax - used - 3 * gap0)        # extra space goes between the 3D pair and the xz pair
    yt = y + TL + (h - TL - th) / 2
    xx = x
    for m, lab, col in (("2D", R.LAB["2D"], F.D2C), ("S1", R.LAB["S1"], F.S1C)):
        rw = th * 1.12
        ax = mm_ax(fig, xx, yt, rw, th)
        px = (V.mm_px(rw, F.DPI), V.mm_px(th, F.DPI))
        ax.imshow(np.asarray(on_bg(F.render3d(m, 8, 2 * max(px)), px, TINT)), interpolation="none", aspect="auto"); ax.axis("off")
        text(fig, xx + rw / 2, yt - 0.8, f"{lab}, 3D", ha="center", va="bottom", fontweight="bold", color=col)
        xx += rw + (gap if m == "2D" else gmid)
    for m, lab, col in (("2D", R.LAB["2D"], F.D2C), ("S1", R.LAB["S1"], F.S1C)):
        if (h - TL) * 1.35 != th * 1.35:
            im, c_ = F.xz_overlay(m, 8, F.fx_for(th * 1.35))
        else:
            im, c_ = xzs[m]
        cnt[f"{m}_k8"] = c_
        tw = th * r
        ax = mm_ax(fig, xx, yt, tw, th)
        ax.imshow(im, interpolation="none", aspect="auto"); F.tile_frame(ax, col, 0.7)
        text(fig, xx + tw / 2, yt - 0.8, f"{lab}, xz", ha="center", va="bottom", fontweight="bold", color=col)
        xx += tw + gap
    return xx - gap, cnt


def build():
    NUMS.clear()
    fig = plt.figure(figsize=(W * F.MM, H * F.MM), facecolor="white")
    # ---- a
    letter(fig, 0, 2, "a")
    text(fig, 4.5, 2.2, "One test volume, degraded as in G4 (xz)", fontsize=6.5, fontweight="bold")
    ya = F7.panel_a(fig, 11, 11, 24.0, 0.7)
    tint_box(fig, 3.0, 5.5, 85.0, ya + 8.0 - 5.5)
    text(fig, 11, ya + 1.2, "C. elegans test crop (median rule); every k-th plane kept;\nnoise σ = c × (nucleus − background median); white: GT outline",
         fontsize=5.2, color=R.MUTED, ha="left")
    # ---- b
    letter(fig, 91, 2, "b")
    for i, (met, lab) in enumerate(MET):
        heat_kn2(fig, 99 + i * 43, 9, 33, 30, met, lab, F.vmax_macro(met)); cbar2(fig, 99 + i * 43 + 34.0, 9, 30, F.vmax_macro(met))
    text(fig, 91, 48.5, "6 datasets, 20 volumes, 3 paired seeds; red: S1 better; dot and bold: all three seed\n"
         "differences share the sign (descriptive, no test)", fontsize=5.2, color=R.MUTED, ha="left")
    # ---- c
    letter(fig, 91, 55, "c")
    text(fig, 95.5, 55.2, "Absolute macro F1 in the 12 conditions (S1 vs 2D)", fontsize=6.5, fontweight="bold")
    for i, (met, lab) in enumerate(MET):
        dumbbell(fig, 111 + i * 37, 64, 33, 25, met, lab, ylabels=i == 0)
    hs = [Line2D([], [], color=F.S1C, marker="o", ms=2.4, lw=0, label="S1"), Line2D([], [], color=F.D2C, marker="s", ms=2.2, lw=0, label="2D control"),
          Line2D([], [], color=P["models"]["base"], marker="^", ms=2.2, lw=0, label="original")]
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(110 / W, 1 - 59.2 / H), ncol=3, fontsize=5.3, handletextpad=0.2, borderaxespad=0,
               columnspacing=1.2, frameon=False)
    # ---- d
    letter(fig, 0, 96, "d")
    text(fig, 4.5, 96.2, "Dataset × condition: Δ (S1 − 2D) per dataset, both metrics; (n) = volumes", fontsize=6.5, fontweight="bold")
    v = max(abs(d[0]) for met, _ in MET for _, vals, _ in F.big_rows(met) for d in vals)
    vmax = float(np.ceil(v * 20) / 20)
    allr = []
    for ds_i in range(7):
        for mi, (met, lab) in enumerate(MET):
            br = F.big_rows(met)[ds_i]
            allr.append((br[0] if mi == 0 else "", lab, br[1], br[2], met))
    n = len(allr)
    AW, AHh = 132, 31
    ax = mm_ax(fig, 40, 107, AW, AHh)
    ax.set_xlim(-0.5, 11.86); ax.set_ylim(n - 0.5 + 0.35, -0.5); ax.axis("off")
    asp = ((n + 0.35) / AHh) / (12.36 / AW)
    for i, (dsl, ml, vals, macro, met) in enumerate(allr):
        yi = i + (0.35 if macro else 0) + 0.12 * (i // 2)
        for j, d in enumerate(vals):
            v_, lo_, hi_ = d; ex = lo_ > 0 or hi_ < 0; xj = j + 0.18 * (j // 4)
            lum = cell(ax, xj, yi, v_, vmax, 0.93, 0.86, asp)
            ax.text(xj, yi, cell_text(v_, 2), ha="center", va="center", fontsize=5.0, color="white" if lum < 0.55 else R.FG,
                    fontweight="bold" if macro else "normal", zorder=3)
            k, c = COND[j]
            NUMS[f"d/{met}/{'macro' if macro else dsl or allr[i - 1][0]}/k{k}/c{c}"] = dict(mean=float(v_), seed_range=[float(lo_), float(hi_)], framed=bool(ex))
        ax.text(-0.62, yi, ml, ha="right", va="center", fontsize=5, color=R.MUTED)
        if dsl:
            ax.text(-2.15, yi + 0.56, dsl, ha="right", va="center", fontsize=5.5, fontweight="bold" if macro else "normal")
    ax.set_ylim(n - 0.5 + 0.35 + 0.12 * (n // 2 - 1), -0.5)
    for j, (k, c) in enumerate(COND):
        ax.text(j + 0.18 * (j // 4), -0.62, f"k = {k}", ha="center", va="bottom", fontsize=5.2, color=R.MUTED)
    for g, c in enumerate(CS):
        xc = g * 4 + 1.5 + 0.18 * g
        ax.text(xc, -1.7, "no noise" if c == "0" else f"noise {c}", ha="center", va="bottom", fontsize=5.8)
        ax.plot([xc - 1.9, xc + 1.9], [-1.6, -1.6], color=R.MUTED, lw=0.4, clip_on=False)
    cbar2(fig, 175, 107, AHh, vmax)
    # ---- e (white band; dark only inside each tile)
    y0 = 141.0
    letter(fig, 0, y0, "e")
    text(fig, 4.5, y0 + 0.2, "Same crop at k = 8, no noise: S1 vs 2D control (seed 0)", fontsize=6.5, fontweight="bold")
    tint_box(fig, 0, y0 + 3.6, W, H - y0 - 3.6)
    LGW = 25.0
    xr, cnt = panel_e9(fig, 4.5, y0 + 4.2, H - y0 - 5.2, W - 4.5 - LGW - 4.0)
    hs = [Patch(fc=F.ERRV[k], ec="#9a9a96", lw=0.3, label=P["error_names"][k]) for k in ("match", "boundary", "split", "merge", "other", "miss")]
    hs.append(Patch(fc="white", ec=R.FG, lw=0.6, label="GT outline (xz)"))
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=((W - LGW - 1.0) / W, 1 - (y0 + 6.5) / H), ncol=1, fontsize=5.5, handlelength=1.2,
               handleheight=0.8, borderaxespad=0, labelspacing=0.35, columnspacing=0.9, frameon=False)
    text(fig, W - LGW - 1.0, y0 + 26.0, "colour: error class of each\npredicted nucleus vs. the\nsubsampled GT (one crop,\nno metric)",
         fontsize=5.2, color=R.MUTED, ha="left", va="top")
    return fig, cnt, xr


if __name__ == "__main__":
    fig, cnt, xr = build()
    F.save(fig, "R9-Fig4", dict(version="R8: R6-B layout, panel forms polished (white page, rounded heat-map cells, dot = same-sign seeds, "
                                         "white band e with dark tiles only)", e_error_counts_on_crop=cnt))
    pj = os.path.join(F.GA, "R9-Fig4.provenance.json")
    d = json.load(open(pj)); d["script"] = "scripts/render3d/fig4r9.py"; d["script_md5"] = F.md5(__file__)
    d["frame_rule"] = "dot (and bold value) when all three paired seed differences S1 - 2D have the same sign; descriptive, no test"
    json.dump(d, open(pj, "w"), indent=1, ensure_ascii=False, default=str)
