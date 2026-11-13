"""Mechanism panel v2 (gallery M-v2-a, M-v2-b), centred on nc-viz's V2 flow-angle detail; no 2D-vs-S1 streamline comparison
(team-lead decision 2026-11-10, following nc-viz's finding that S1 cleanly separates 0/17 2D-control z-merges on mouse embryo).
a-row: V2-h dense xz angle + arrows (2D fused | S1); b-row: V2-i zoom on nucleus 62 (GT target | 2D | S1) and, in M-v2-b only,
V2-k z-profile of the median angle (single-volume illustration). M-v2-a leaves a labelled slot for the FLOWERR summary."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image  # noqa: E402
import figcheck  # noqa: E402
import rstyle as R  # noqa: E402
for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6.5})
HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", "..", "figures", "assets_viz"))
GA = os.path.abspath(os.path.join(HERE, "..", "..", "figures", "gallery", "assets"))


def crop(fn, pad=20):
    """Content bbox, ignoring near-full-width dark rules (>95 % of a row/column dark, e.g. a frame edge); nothing is blanked."""
    im = Image.open(os.path.join(V, fn)).convert("RGB"); a = np.asarray(im).min(-1) < 245
    a[a.mean(1) > 0.95] = False
    a[:, a.mean(0) > 0.95] = False
    ys, xs = np.nonzero(a)
    y1 = ys.max()
    full = np.nonzero(np.asarray(im).min(-1).mean(1) < 60)[0]          # dark rules below the content: stop above them
    below = full[full > ys.min() + 10]
    if len(below):
        y1 = min(y1, below.min() - 3)
    return im.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0), xs.max() + pad, min(y1 + pad, im.height)))


def build(cid, with_k):
    h, i, k = crop("V2-h_angle_quiver_xz_dense.png"), crop("V2-i_zoom_nucleus62_angle_quiver.png"), crop("V2-k_zprofile_angle.png")
    W = 183
    ha = W * h.height / h.width
    wi = W * (0.64 if True else 1.0); hi_ = wi * i.height / i.width
    H = ha + hi_ + 14
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=450, facecolor="white")
    ax = fig.add_axes([0, 1 - (ha + 5) / H, 1, ha / H]); ax.imshow(h); ax.axis("off")
    fig.text(0.003, 1 - 0.5 / H, "a", fontsize=8, fontweight="bold", va="top")
    fig.text(0.025, 1 - 0.8 / H, "Angle between each model's flow and the true 3D flow (central xz section, mouse embryo; arrows: model flow)",
             fontsize=6.3, fontweight="bold", va="top", color=R.FG)
    yb = 1 - (ha + 9) / H
    ax = fig.add_axes([0, 0.005, 0.64, hi_ / H]); ax.imshow(i); ax.axis("off")
    fig.text(0.003, yb + 3 / H, "b", fontsize=8, fontweight="bold", va="top")
    fig.text(0.025, yb + 2.7 / H, "One nucleus: true 3D target (arrows coloured by u_z) and each model's angle", fontsize=6.3, fontweight="bold", va="top", color=R.FG)
    if with_k:
        kh = 0.33 * W * k.height / k.width
        ax = fig.add_axes([0.66, 0.02, 0.33, kh / H]); ax.imshow(k); ax.axis("off")
        fig.text(0.655, yb + 3 / H, "c", fontsize=8, fontweight="bold", va="top")
        fig.text(0.68, yb + 2.7 / H, "Where along z the error sits", fontsize=6.3, fontweight="bold", va="top", color=R.FG)
    else:
        ax = fig.add_axes([0.665, 0.03, 0.325, hi_ / H * 0.9]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#f7f7f5")
        for sp in ax.spines.values():
            sp.set_linestyle((0, (3, 2))); sp.set_color("#c9c9c6"); sp.set_linewidth(0.5)
        ax.text(0.5, 0.5, "reserved: FLOWERR summary\n(test-set flow error, all datasets)", ha="center", va="center", fontsize=5.8, color=R.MUTED, transform=ax.transAxes)
        fig.text(0.655, yb + 3 / H, "c", fontsize=8, fontweight="bold", va="top")
    fig.savefig(os.path.join(GA, f"{cid}.png"), facecolor="white")
    rep = figcheck.check(fig); plt.close(fig); print("saved", cid, rep)


if __name__ == "__main__":
    build("M-v2-a", False)
    build("M-v2-b", True)
