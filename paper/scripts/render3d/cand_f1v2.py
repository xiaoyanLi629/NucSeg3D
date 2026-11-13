"""F1 exploded stack, v2 layout (gallery F1-d light, F1-e dark).
Stack of 7 raw xy planes (mouse-embryo Fig. 2d volume, 220-px crop around the GT-foreground centroid) spread along z in the
centre; the central xy plane is pulled out to the right and the central xz plane (through the crop's y centre) to the left,
both flat and enlarged with GT (magenta), 2D control (orange) and S1 (blue) outlines; leader lines start at the projected
corners of the exact plane positions in the 3D render. Masks are the evaluated final masks.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

import rstyle as R  # noqa: E402

for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6.5, "pdf.fonttype": 42})
HERE = os.path.dirname(os.path.abspath(__file__))
GA = os.path.abspath(os.path.join(HERE, "..", "..", "figures", "gallery", "assets"))
Z = np.load(os.path.join(HERE, "export", "mouse_embryo.npz"))
GT, IMG, RATIO = Z["gt"], Z["img"].astype(np.float32) / 255.0, float(Z["ratio"])
s = 220
zc = int(np.argmax([(GT[k] > 0).sum() for k in range(GT.shape[0])]))
nz = int(round(s / RATIO))
z0 = int(np.clip(zc - nz // 2, 0, GT.shape[0] - nz))
cy, cx = [int(v) for v in np.argwhere(GT > 0)[:, 1:].mean(0)]
y0 = int(np.clip(cy - s // 2, 0, GT.shape[1] - s)); x0 = int(np.clip(cx - s // 2, 0, GT.shape[2] - s))
crop = (slice(z0, z0 + nz), slice(y0, y0 + s), slice(x0, x0 + s))
img, gt, s1, d2 = IMG[crop], GT[crop], Z["S1"][crop], Z["2D"][crop]
lo, hi = np.percentile(img, (1, 99.8))
GTC = R.PALETTE["gt_outline_on_image"]
N, GAP = 7, 2.2
ks = np.round(np.linspace(1, nz - 2, N)).astype(int)
kmid = int(ks[N // 2]); ymid = s // 2


def rgb_of(im, labs=None):
    rgb = np.repeat(np.clip((im - lo) / (hi - lo), 0, 1)[..., None], 3, -1)
    if labs:
        for L, c in labs:
            rgb[find_boundaries(L, mode="inner")] = to_rgb(c)
    return rgb


def quad(rgb, o, iv, jv):
    pts = np.array([o, o + iv, o + iv + jv, o + jv], float)
    q = pv.PolyData(pts, faces=[4, 0, 1, 2, 3])
    q.active_texture_coordinates = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    return q, pv.Texture((np.flipud(rgb) * 255).astype(np.uint8))


def build(cid, dark):
    zpos = lambda k: k * RATIO * GAP  # noqa: E731
    p = R.plotter((2600, 2600), dark)
    edge = "#ffffff" if not dark else "#5b606b"
    for k in ks:
        q, t = quad(rgb_of(img[k]), np.array([0, 0, zpos(k)]), np.array([s, 0, 0]), np.array([0, s, 0]))
        if k == kmid:
            p.add_mesh(q, texture=t, opacity=1.0, lighting=False)
            p.add_mesh(q.extract_feature_edges(), color=R.S1, line_width=6)
        else:
            dist = abs(int(np.where(ks == k)[0][0]) - N // 2) / (N // 2)       # depth falloff away from the pulled plane
            p.add_mesh(q, texture=t, opacity=1.0 - 0.35 * dist, lighting=False)
            p.add_mesh(q.extract_feature_edges(), color=edge, line_width=1.6, opacity=0.9 - 0.5 * dist)
    # xz plane position: a vertical outline through the stack at y = ymid
    zt = zpos(ks[-1]); zb = zpos(ks[0])
    xz = pv.PolyData(np.array([[0, ymid, zb], [s, ymid, zb], [s, ymid, zt], [0, ymid, zt]], float), lines=[5, 0, 1, 2, 3, 0])
    p.add_mesh(xz, color=R.D2, line_width=5)
    ctr = np.array([s / 2, s / 2, (zb + zt) / 2])
    p.camera_position = [tuple(ctr + np.array([-s * 1.5, -s * 2.6, s * 2.0])), tuple(ctr), (0, 0, 1)]
    p.enable_parallel_projection(); p.reset_camera(); p.camera.zoom(1.0)
    p.render()

    def proj(pt):
        r = p.renderer
        r.SetWorldPoint(*pt, 1.0); r.WorldToDisplay()
        x, y, _ = r.GetDisplayPoint()
        return x, p.window_size[1] - y
    cz4 = [proj(c) for c in ((0, 0, zpos(kmid)), (s, 0, zpos(kmid)), (s, s, zpos(kmid)), (0, s, zpos(kmid)))]
    # visible corners of the highlighted plane: the right-most and the lowest (the upper corners are hidden by the planes above)
    rgt = [max(cz4, key=lambda q: q[0]), max(cz4, key=lambda q: q[1])]
    corners = {"xy_r_top": rgt[0], "xy_r_bot": rgt[1],
               "xz_l_top": proj((0, ymid, zt)), "xz_l_bot": proj((0, ymid, zb))}
    raw = R.shot(p, ssao=False)
    a = np.asarray(raw)[..., 3]; ys_, xs_ = np.nonzero(a > 8); m = 20
    bx0, by0 = max(xs_.min() - m, 0), max(ys_.min() - m, 0)
    stack = raw.crop((bx0, by0, min(xs_.max() + m, raw.width), min(ys_.max() + m, raw.height)))
    corners = {k: (v[0] - bx0, v[1] - by0) for k, v in corners.items()}
    # ---- compose
    bg = R.DARK_BG[0] if dark else "white"; tc = "#e8e8ea" if dark else R.FG; mc = "#a0a0a8" if dark else R.MUTED
    fig = plt.figure(figsize=(183 / 25.4, 74 / 25.4), dpi=500, facecolor=bg)
    axs = fig.add_axes([0.30, 0.04, 0.36, 0.92]); axs.imshow(R.tile(stack, dark) if False else stack); axs.axis("off")
    labs = [(gt[kmid], GTC), (d2[kmid], R.D2), (s1[kmid], R.S1)]
    axr = fig.add_axes([0.685, 0.12, 0.30, 0.78]); axr.imshow(rgb_of(img[kmid], labs), interpolation="nearest")
    labs_xz = [(gt[:, ymid], GTC), (d2[:, ymid], R.D2), (s1[:, ymid], R.S1)]
    axl = fig.add_axes([0.01, 0.30, 0.27, 0.42]); axl.imshow(rgb_of(img[:, ymid], labs_xz), interpolation="nearest", aspect=RATIO)
    for a_, col, ttl in ((axr, R.S1, f"xy plane, z = {kmid + z0}"), (axl, R.D2, f"xz plane, y = {ymid + y0}")):
        a_.set_xticks([]); a_.set_yticks([])
        for sp in a_.spines.values():
            sp.set_color(col); sp.set_linewidth(1.2)
        a_.set_title(ttl, fontsize=6.5, fontweight="bold", color=tc, pad=3)
    # leader lines from exact projected plane corners
    H, W = np.asarray(stack).shape[:2]
    for (cname, (tx, ty)) in (("xy_r_top", (0, 0)), ("xy_r_bot", (0, 1))):
        px, py = corners[cname]
        fig.add_artist(ConnectionPatch(xyA=(px, py), coordsA=axs.transData, xyB=(tx, ty), coordsB=axr.transAxes, color=R.S1, lw=0.7))
    for (cname, (tx, ty)) in (("xz_l_top", (1, 1)), ("xz_l_bot", (1, 0))):
        px, py = corners[cname]
        fig.add_artist(ConnectionPatch(xyA=(px, py), coordsA=axs.transData, xyB=(tx, ty), coordsB=axl.transAxes, color=R.D2, lw=0.7))
    leg = [Line2D([], [], color=GTC if dark else R.GT, lw=1.4, label="ground truth"), Line2D([], [], color=R.D2, lw=1.4, label="2D control"),
           Line2D([], [], color=R.S1, lw=1.4, label="S1")]
    lg = fig.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.01, 0.06), ncol=1, fontsize=6, frameon=True, labelcolor=tc,
                    title="outlines", title_fontsize=6, handlelength=1.6, borderpad=0.6)
    lg.get_frame().set_facecolor(R.DARK_BG[1] if dark else "#f7f7f5"); lg.get_frame().set_edgecolor("none"); lg.get_title().set_color(tc)
    fig.text(0.31, 0.97, f"Raw planes z = {ks[0] + z0}–{ks[-1] + z0} (every {int(ks[1] - ks[0])}th shown, spread along z)", fontsize=5.8,
             color=mc, va="top")
    fn = os.path.join(GA, f"{cid}.png"); fig.savefig(fn, facecolor=bg); plt.close(fig)
    json.dump({"desc": "exploded stack v2", "crop_zyx": [z0, y0, x0, nz, s, s], "planes": [int(k + z0) for k in ks],
               "xy_pulled": int(kmid + z0), "xz_pulled_y": int(ymid + y0), "LUT": "linear 1-99.8 percentile of the crop",
               "outlines": {"GT": GTC, "2D": R.D2, "S1": R.S1}}, open(os.path.join(GA, f"{cid}.provenance.json"), "w"), indent=1)
    print("saved", cid)


build("F1-f", True)
build("F1-g", False)
