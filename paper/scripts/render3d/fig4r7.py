"""R7 Fig. 4 candidate (gallery R7-Fig4): re-layout of R6-Fig4-B (= final Fig. 4) after user feedback.
Same data, same numbers, same selection (FIG4R6_RULE.md); the panel builders of fig4r6.py are imported, not modified.
Changes: no full-panel black backgrounds (page white; raw-image tiles and 3D renders keep a dark background only inside
their own tile frames); a on white with dark labels; the k = 8 comparison (e) enlarged to one row of four tiles across
the full width with a compact legend at the right; c (dumbbell) kept under b so that row 1 has no empty space.
Panel letters: a degradation grid, b k x noise heat maps, c dumbbell (absolute macro F1), d dataset x condition heat map,
e S1 vs 2D at k = 8 (3D renders + xz sections). Run: .venv/bin/python fig4r7.py"""
import json
import os

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import to_rgb

import fig4r6 as F  # noqa: E402  (imports only; its __main__ is not run)
from fig4r6 import (CMAP, COND, CS, D2C, DPI, E, E4, G, KS, MET, MM, NUMS, P, S1C, Z, cbar, cell_text, dumbbell, fx_for,  # noqa: F401
                    gray, heat_kn, letter, md5, mm_ax, outline, phys, plt, render3d, text, tile_frame, vmax_macro, xz_overlay)
import figcheck  # noqa: E402
import r6_render as V  # noqa: E402
import rstyle as R  # noqa: E402

W, H = F.W, F.H
HERE = os.path.dirname(os.path.abspath(__file__))
GA = F.GA
ERRV = F.ERRV


def panel_a(fig, x, y, tw, gap):
    """Degradation grid on the white page: dark image tiles with thin gaps, dark row/column labels."""
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
                text(fig, x + j * (tw + gap) + tw / 2, y - 0.8, "no noise" if c == "0" else f"noise c = {c}", ha="center", va="bottom")
            if j == 0:
                text(fig, x - 1.2, y + i * (th + gap) + th / 2, f"k = {k}", ha="right", va="center")
    sb = 5 / float(Z["vox0"][1]) / 96 * tw
    xb = x + 2 * (tw + gap) + tw - sb - 1.5; yb = y + 3 * (th + gap) + th - 1.5
    fig.add_artist(Line2D([xb / W, (xb + sb) / W], [1 - yb / H] * 2, color="white", lw=1.0, transform=fig.transFigure))
    text(fig, xb + sb / 2, yb - 0.6, "5 µm", ha="center", va="bottom", fontsize=5.5, color="white", bbox=dict(fc="black", ec="none", pad=0.6))
    return y + 4 * th + 3 * gap


def panel_e(fig, x, y, h, gap):
    """One row: 2D 3D | S1 3D | 2D xz | S1 xz; dark only inside the framed tiles (renders 2x supersampled, lossless)."""
    cnt = {}
    th = h - 4.2
    xx = x
    for m, lab, col in (("2D", R.LAB["2D"], D2C), ("S1", R.LAB["S1"], S1C)):
        rw = th * 1.12
        ax = mm_ax(fig, xx, y + 4.2, rw, th)
        px = (V.mm_px(rw, DPI), V.mm_px(th, DPI))
        tile, _ = V.on_black(render3d(m, 8, 2 * max(px)), px)
        ax.imshow(np.asarray(tile), interpolation="none", aspect="auto"); tile_frame(ax, col, 0.8)
        text(fig, xx + rw / 2, y + 3.4, f"{lab}, 3D", ha="center", va="bottom", fontweight="bold", color=col)
        xx += rw + gap
    for m, lab, col in (("2D", R.LAB["2D"], D2C), ("S1", R.LAB["S1"], S1C)):
        fx = fx_for(th * 1.35)
        im, c_ = xz_overlay(m, 8, fx); cnt[f"{m}_k8"] = c_
        tw = th * im.shape[1] / im.shape[0]
        ax = mm_ax(fig, xx, y + 4.2, tw, th)
        ax.imshow(im, interpolation="none", aspect="auto"); tile_frame(ax, col, 0.8)
        text(fig, xx + tw / 2, y + 3.4, f"{lab}, xz", ha="center", va="bottom", fontweight="bold", color=col)
        xx += tw + gap
    return xx, cnt


def legend_e(fig, x, y):
    hs = [Patch(fc=ERRV[k], ec="#9a9a96", lw=0.3, label=P["error_names"][k]) for k in ("match", "boundary", "split", "merge", "other", "miss")]
    hs.append(Patch(fc="white", ec=R.FG, lw=0.6, label="GT outline (xz)"))
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(x / W, 1 - y / H), ncol=1, fontsize=5.5, handlelength=1.2, handleheight=0.8,
               borderaxespad=0, labelspacing=0.35)


def build():
    NUMS.clear()
    fig = plt.figure(figsize=(W * MM, H * MM), facecolor="white")
    # ---- row 1: a (left) | b + c (right)
    letter(fig, 0, 1.5, "a")
    text(fig, 4.5, 1.7, "One test volume, degraded as in G4 (xz)", fontsize=6.5, fontweight="bold")
    ya = panel_a(fig, 9.5, 10.5, 25.0, 0.6)
    text(fig, 9.5, ya + 1.0, "C. elegans test crop (median rule); every k-th plane kept; noise σ = c × (nucleus −\nbackground median); white: GT outline",
         fontsize=5.2, color=R.MUTED, ha="left")
    letter(fig, 93, 1.5, "b")
    for i, (met, lab) in enumerate(MET):
        heat_kn(fig, 101 + i * 42, 9, 32, 27, met, lab, F.vmax_macro(met)); cbar(fig, 101 + i * 42 + 33.2, 9, 27, F.vmax_macro(met))
    text(fig, 93, 44.0, "6 datasets, 20 volumes, 3 paired seeds; red: S1 better; framed / bold: all three seed\n"
         "differences share the sign (descriptive, no test)", fontsize=5.2, color=R.MUTED, ha="left")
    letter(fig, 93, 52.0, "c")
    text(fig, 97.5, 52.2, "Absolute macro F1 in the 12 conditions", fontsize=6.5, fontweight="bold")
    for i, (met, lab) in enumerate(MET):
        dumbbell(fig, 116 + i * 34, 61.5, 30, 22.5, met, lab, ylabels=i == 0)
    hs = [Line2D([], [], color=S1C, marker="o", ms=2.4, lw=0, label="S1"), Line2D([], [], color=D2C, marker="s", ms=2.2, lw=0, label="2D control"),
          Line2D([], [], color=P["models"]["base"], marker="^", ms=2.2, lw=0, label="original")]
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(115 / W, 1 - 56.3 / H), ncol=3, fontsize=5.3, handletextpad=0.2, borderaxespad=0,
               columnspacing=1.2)
    # ---- row 2: d dataset x condition heat map (both metrics interleaved), as R6-B
    y_d = 92.5
    letter(fig, 0, y_d, "d")
    text(fig, 4.5, y_d + 0.2, "Dataset × condition: Δ (S1 − 2D) per dataset, both metrics; (n) = volumes", fontsize=6.5, fontweight="bold")
    v = max(abs(d[0]) for met, _ in MET for _, vals, _ in F.big_rows(met) for d in vals)
    vmax = float(np.ceil(v * 20) / 20)
    allr = []
    for ds_i in range(7):
        for mi, (met, lab) in enumerate(MET):
            br = F.big_rows(met)[ds_i]
            allr.append((br[0] if mi == 0 else "", lab, br[1], br[2], met))
    n = len(allr)
    ax = mm_ax(fig, 40, y_d + 10.5, 132, 30)
    ax.set_xlim(-0.5, 11.86); ax.set_ylim(n - 0.5, -0.5); ax.axis("off")
    for i, (dsl, ml, vals, macro, met) in enumerate(allr):
        for j, d in enumerate(vals):
            v_, lo_, hi_ = d; ex = lo_ > 0 or hi_ < 0; xj = j + 0.18 * (j // 4)
            ax.add_patch(Rectangle((xj - 0.47, i - 0.44), 0.94, 0.88, fc=CMAP((v_ + vmax) / (2 * vmax)), ec="none"))
            ax.text(xj, i, cell_text(v_, 2), ha="center", va="center", fontsize=5.0, color="white" if abs(v_) > 0.62 * vmax else R.FG,
                    fontweight="bold" if macro else "normal")
            k, c = COND[j]
            NUMS[f"d/{met}/{'macro' if macro else dsl or allr[i - 1][0]}/k{k}/c{c}"] = dict(mean=float(v_), seed_range=[float(lo_), float(hi_)], framed=bool(ex))
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
        ax.text(xc, -1.75, "no noise" if c == "0" else f"noise {c}", ha="center", va="bottom", fontsize=5.8)
        ax.plot([xc - 1.9, xc + 1.9], [-1.65, -1.65], color=R.MUTED, lw=0.4, clip_on=False)
    cbar(fig, 175, y_d + 10.5, 30, vmax)
    # ---- row 3: e enlarged, white page, dark only inside tiles
    y_e = 134.5
    letter(fig, 0, y_e, "e")
    text(fig, 4.5, y_e + 0.2, "Same crop at k = 8, no noise: S1 vs 2D control (seed 0)", fontsize=6.5, fontweight="bold")
    xr, cnt = panel_e(fig, 4.5, y_e + 2.0, H - y_e - 3.0, 2.2)
    legend_e(fig, xr + 2.0, y_e + 5.0)
    text(fig, xr + 2.0, y_e + 26.3, "colour: error class of each\npredicted nucleus vs. the sub-\nsampled GT (one crop, no metric)", fontsize=5.2, color=R.MUTED, ha="left")
    # NUMS keys: b/... from heat_kn, e/<met>/k/c from dumbbell -> c/..., d/... from the map above
    remap = dict(NUMS); NUMS.clear()
    NUMS.update({("c/" + k[2:] if k.startswith("e/") else k): v for k, v in remap.items()})
    return fig, cnt, xr


def save(fig, cnt, xr):
    rep = figcheck.check(fig)
    cid = "R7-Fig4"
    fn = os.path.join(GA, f"{cid}.png")
    fig.savefig(fn, dpi=DPI, facecolor="white"); fig.savefig(fn.replace(".png", ".pdf"), dpi=DPI, facecolor="white")
    src = {p: md5(os.path.join(E4, p)) for p in ("g4_summary.json", "fig4r6_perds.json", "fig4r6_degrade.npz", "fig4r6_export_log.json")}
    src.update({f"f5_{m}_k{k}.npz": md5(os.path.join(E, f"f5_{m}_k{k}.npz")) for m in ("S1", "2D") for k in (1, 8)})
    prov = {"figure": cid, "based_on": "R6-Fig4-B (final/Fig4), re-layout only", "rule": "scripts/render3d/FIG4R6_RULE.md",
            "rule_md5": md5(os.path.join(HERE, "FIG4R6_RULE.md")), "script": "scripts/render3d/fig4r7.py", "script_md5": md5(__file__),
            "imports_from": {"fig4r6.py": md5(F.__file__)}, "inputs_md5": src,
            "palettes": {"palette.json": md5(os.path.join(HERE, "..", "..", "figures", "palette.json")),
                         "palette_vivid_dark.json": md5(os.path.join(HERE, "..", "..", "figures", "palette_vivid_dark.json"))},
            "panel_letters": {"a": "degradation grid (xz, k x noise)", "b": "k x noise heat maps of macro S1 - 2D (F1@0.5, F1@0.7)",
                              "c": "dumbbell: absolute macro F1 of S1, 2D control, original in the 12 conditions",
                              "d": "dataset x condition heat map, both metrics interleaved", "e": "same crop at k = 8, no noise: 2D / S1 3D renders and xz sections"},
            "layout": "page white everywhere; dark background only inside image / render tiles (framed); e one row of four tiles across the width",
            "render": "3D tiles rendered at 2x the final pixel size (r6_render plotter, depth peeling, SSAO), Lanczos downsample (SSAA), embedded unresampled",
            "frame_rule": "b: cell framed when all three paired seed differences share the sign (descriptive, no test); d: no frames",
            "numbers_drawn": dict(NUMS), "e_error_counts_on_crop": cnt, "size_mm": [W, H], "dpi": DPI, "overlap_and_font_check": rep,
            "generative_ai": "none"}
    json.dump(prov, open(fn.replace(".png", ".provenance.json"), "w"), indent=1, ensure_ascii=False, default=str)
    plt.close(fig)
    print("saved", cid, "check:", rep, "e right edge", round(xr, 1))


if __name__ == "__main__":
    fig, cnt, xr = build()
    save(fig, cnt, xr)
