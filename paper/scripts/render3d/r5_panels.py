"""Round-5 panels (team-lead 2026-11-10): F1-f exploded stack and F2-e semi-transparent cube, re-composed at their final
print sizes so that text stays 5-7 pt, with scale bars and data source. Same volume, crop and LUT rules as the gallery
originals (cand_f1v2.py, cand_f2v2.py): mouse-embryo median-rule test volume E15_Ab2 (ortho_selection.json; S1 seed 0 run
07992b55, 2D control seed 0 run 81116913), 220-px crop around the GT-foreground centroid, evaluated final masks.
Voxel size from the exported mask file: z 1.0 um, xy 0.215 um (z/xy 4.65).

Functions draw into an existing matplotlib figure at a rectangle given in mm (x, y from the top-left of the page).
dark=True: the image panel sits on a dark slate background inside the white page.
"""
import json
import os

import numpy as np
import pyvista as pv
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from PIL import Image
from skimage.segmentation import find_boundaries

import rstyle as R

pv.OFF_SCREEN = True
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "out", "r5_cache"); os.makedirs(CACHE, exist_ok=True)
SEL = json.load(open(os.path.join(HERE, "..", "figs", "ortho_selection.json")))
DS = "mouse_embryo"
Z = np.load(os.path.join(HERE, "export", f"{DS}.npz"))
GT, IMG, RAT, VOX = Z["gt"], Z["img"].astype(np.float32) / 255.0, float(Z["ratio"]), np.asarray(Z["vox"], float)
UM_PX = float(VOX[2])                       # um per lateral pixel
s = 220
zc = int(np.argmax([(GT[k] > 0).sum() for k in range(GT.shape[0])]))
nz = int(round(s / RAT))
z0 = int(np.clip(zc - nz // 2, 0, GT.shape[0] - nz))
cy, cx = [int(v) for v in np.argwhere(GT > 0)[:, 1:].mean(0)]
y0 = int(np.clip(cy - s // 2, 0, GT.shape[1] - s)); x0 = int(np.clip(cx - s // 2, 0, GT.shape[2] - s))
CROP = (slice(z0, z0 + nz), slice(y0, y0 + s), slice(x0, x0 + s))
img, gt, s1, d2 = IMG[CROP], GT[CROP], Z["S1"][CROP], Z["2D"][CROP]
GTC = R.PALETTE["gt_outline_on_image"]
N, GAP = 7, 2.2
ks = np.round(np.linspace(1, nz - 2, N)).astype(int)
kmid = int(ks[N // 2]); ymid = s // 2
VOL = SEL["volumes"][DS]
SOURCE = {"dataset": R.DS_NAME[DS], "volume": VOL["id"], "selection": "median rule (ortho_selection.json; S1 - 2D F1@0.5 closest to the dataset median)",
          "runs_seed0": {"S1": SEL["runs"]["S1"], "2D": SEL["runs"]["2D"]}, "seed": 0, "masks": "evaluated final masks (rule T, fragments < 0.15 V removed)",
          "crop_zyx": [z0, y0, x0, nz, s, s], "voxel_um_zyx": VOX.tolist(), "F1@0.5": {k: VOL["f1"][k] for k in ("S1", "2D")}}
DARK_TC, DARK_MC = "#e8e8ea", "#a0a0a8"
XZC = {True: "#c8c8cc", False: R.MUTED}   # neutral marker colour for the xz plane (model colours are reserved)


def _mm(fig, W, H, x, y, w, h):
    return [x / W, 1 - (y + h) / H, w / W, h / H]


def _lut(im, lo, hi):
    return np.clip((im - lo) / (hi - lo), 0, 1)


LO, HI = np.percentile(img, (1, 99.8))


def _rgb(im, labs=None):
    rgb = np.repeat(_lut(im, LO, HI)[..., None], 3, -1)
    for L, c in (labs or []):
        rgb[find_boundaries(L, mode="inner")] = to_rgb(c)
    return rgb


def _quad(rgb, o, iv, jv):
    pts = np.array([o, o + iv, o + iv + jv, o + jv], float)
    q = pv.PolyData(pts, faces=[4, 0, 1, 2, 3])
    q.active_texture_coordinates = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    return q, pv.Texture((np.flipud(rgb) * 255).astype(np.uint8))


def stack_render(dark):
    """Exploded stack (7 raw xy planes spread along z), exact geometry of cand_f1v2.build; RGBA + projected plane corners."""
    fn = os.path.join(CACHE, f"stack_{'d' if dark else 'l'}")
    if os.path.exists(fn + ".png"):
        return Image.open(fn + ".png"), json.load(open(fn + ".json"))
    zpos = lambda k: k * RAT * GAP  # noqa: E731
    p = R.plotter((2600, 2600), dark)
    edge = "#5b606b" if dark else "#b9b9b5"
    for k in ks:
        q, t = _quad(_rgb(img[k]), np.array([0, 0, zpos(k)]), np.array([s, 0, 0]), np.array([0, s, 0]))
        if k == kmid:
            p.add_mesh(q, texture=t, opacity=1.0, lighting=False)
            p.add_mesh(q.extract_feature_edges(), color=R.FG if not dark else "#ffffff", line_width=6)
        else:
            dist = abs(int(np.where(ks == k)[0][0]) - N // 2) / (N // 2)
            p.add_mesh(q, texture=t, opacity=1.0 - 0.35 * dist, lighting=False)
            p.add_mesh(q.extract_feature_edges(), color=edge, line_width=1.6, opacity=0.9 - 0.5 * dist)
    zt, zb = zpos(ks[-1]), zpos(ks[0])
    xz = pv.PolyData(np.array([[0, ymid, zb], [s, ymid, zb], [s, ymid, zt], [0, ymid, zt]], float), lines=[5, 0, 1, 2, 3, 0])
    p.add_mesh(xz, color=XZC[dark], line_width=5)
    ctr = np.array([s / 2, s / 2, (zb + zt) / 2])
    p.camera_position = [tuple(ctr + np.array([-s * 1.5, -s * 2.6, s * 2.0])), tuple(ctr), (0, 0, 1)]
    p.enable_parallel_projection(); p.reset_camera(); p.render()

    def proj(pt):
        r = p.renderer
        r.SetWorldPoint(*pt, 1.0); r.WorldToDisplay()
        x, y, _ = r.GetDisplayPoint()
        return x, p.window_size[1] - y
    cz4 = [proj(c) for c in ((0, 0, zpos(kmid)), (s, 0, zpos(kmid)), (s, s, zpos(kmid)), (0, s, zpos(kmid)))]
    corners = {"xy_a": max(cz4, key=lambda q: q[0]), "xy_b": max(cz4, key=lambda q: q[1]),
               "xz_top": proj((0, ymid, zt)), "xz_bot": proj((0, ymid, zb))}
    raw = R.shot(p, ssao=False)
    a = np.asarray(raw)[..., 3]; ys_, xs_ = np.nonzero(a > 8); m = 20
    bx0, by0 = max(xs_.min() - m, 0), max(ys_.min() - m, 0)
    stack = raw.crop((bx0, by0, min(xs_.max() + m, raw.width), min(ys_.max() + m, raw.height)))
    corners = {k: [float(v[0] - bx0), float(v[1] - by0)] for k, v in corners.items()}
    stack.save(fn + ".png"); json.dump(corners, open(fn + ".json", "w"))
    return stack, corners


def _scalebar(ax, px_len, label, color, h_px, w_px, yfrac=0.06):
    x1 = w_px * 0.06; x2 = x1 + px_len; y = h_px * (1 - yfrac)
    ax.plot([x1, x2], [y, y], color=color, lw=1.0, solid_capstyle="butt")
    import matplotlib.patheffects as pe
    ax.text((x1 + x2) / 2, y - h_px * 0.035, label, color=color, fontsize=5.2, ha="center", va="bottom",
            path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])


def draw_f1(fig, W, H, x, y, w, h, dark, sharp=False):
    """sharp (round 6): planes upsampled 4x nearest-neighbour and all rasters embedded unresampled (interpolation none)."""
    up = (lambda a: np.repeat(np.repeat(a, 4, 0), 4, 1)) if sharp else (lambda a: a)
    ip = "none" if sharp else "nearest"
    """Exploded stack in the centre, xz plane (left) and xy plane (right) pulled out with GT / 2D control / S1 outlines."""
    tc, mc = (DARK_TC, DARK_MC) if dark else (R.FG, R.MUTED)
    if dark:
        fig.add_artist(Rectangle((x / W, 1 - (y + h) / H), w / W, h / H, transform=fig.transFigure, fc=R.DARK_BG[0], ec="none", zorder=-5))
    stack, cn = stack_render(dark)
    pad = 2.0; side = h - 2 * pad - 6.0          # pulled planes: square (xz has physical aspect 220 x 47*4.65 = 219 px)
    leg_h = 0
    xz_ext = (nz * RAT)
    # left: xz, right: xy, centre: stack scaled to fit the remaining width / height
    axl = fig.add_axes(_mm(fig, W, H, x + pad, y + pad + 4.0, side * s / max(s, xz_ext), side))
    # xz at physical aspect: raw plane linearly interpolated along z, label planes nearest-neighbour, outlines traced at that grid
    from scipy.ndimage import zoom
    zf = (RAT, 1)
    labs_xz = [(zoom(gt[:, ymid], zf, order=0), GTC), (zoom(d2[:, ymid], zf, order=0), R.D2), (zoom(s1[:, ymid], zf, order=0), R.S1)]
    xz_img = zoom(img[:, ymid], zf, order=1)
    axl.imshow(up(_rgb(xz_img, labs_xz)), interpolation=ip, extent=(-0.5, s - 0.5, xz_img.shape[0] - 0.5, -0.5))
    axr = fig.add_axes(_mm(fig, W, H, x + w - pad - side, y + pad + 4.0, side, side))
    labs = [(gt[kmid], GTC), (d2[kmid], R.D2), (s1[kmid], R.S1)]
    axr.imshow(up(_rgb(img[kmid], labs)), interpolation=ip, extent=(-0.5, s - 0.5, s - 0.5, -0.5))
    cw = w - 2 * pad - 2 * side - 2 * 6.0
    sw, sh = stack.size; ch = min(h - 2 * pad, cw * sh / sw); cw2 = ch * sw / sh
    axs = fig.add_axes(_mm(fig, W, H, x + pad + side + 6.0 + (cw - cw2) / 2, y + (h - ch) / 2, cw2, ch))
    axs.imshow(stack, interpolation="none" if sharp else "antialiased"); axs.axis("off")
    for a_, ttl in ((axl, f"xz plane (y = {ymid + y0})"), (axr, f"xy plane (z = {kmid + z0})")):
        a_.set_xticks([]); a_.set_yticks([])
        for sp in a_.spines.values():
            sp.set_color(tc if dark else R.FG); sp.set_linewidth(0.6)
        a_.set_title(ttl, fontsize=6.0, fontweight="bold", color=tc, pad=2)
    _scalebar(axr, 10 / UM_PX, "10 µm", "#ffffff", s, s)
    _scalebar(axl, 10 / UM_PX, "10 µm", "#ffffff", xz_img.shape[0], s)
    lk = "#ffffff" if dark else R.FG
    for cname, (tx, ty) in (("xy_a", (0, 1)), ("xy_b", (0, 0))):
        fig.add_artist(ConnectionPatch(xyA=cn[cname], coordsA=axs.transData, xyB=(tx, ty), coordsB=axr.transAxes, color=lk, lw=0.5))
    xzc = XZC[dark]
    for cname, (tx, ty) in (("xz_top", (1, 1)), ("xz_bot", (1, 0))):
        fig.add_artist(ConnectionPatch(xyA=cn[cname], coordsA=axs.transData, xyB=(tx, ty), coordsB=axl.transAxes, color=xzc, lw=0.5))
    leg = [Line2D([], [], color=GTC, lw=1.2, label=R.LAB["gt"], path_effects=[] if dark else [__import__("matplotlib.patheffects", fromlist=["x"]).withStroke(linewidth=2.2, foreground=R.FG)]), Line2D([], [], color=R.D2, lw=1.2, label=R.LAB["2D"]),
           Line2D([], [], color=R.S1, lw=1.2, label=R.LAB["S1"])]
    lg = fig.legend(handles=leg, loc="upper center", bbox_to_anchor=((x + w / 2) / W, 1 - (y + 0.6) / H), ncol=3, fontsize=5.6, frameon=False,
                    labelcolor=tc, handlelength=1.4, columnspacing=1.0, borderaxespad=0)
    fig.text((x + w / 2) / W, 1 - (y + h - 0.8) / H,
             f"{SOURCE['dataset']} test volume {SOURCE['volume']} (median rule), seed 0; raw planes z = {ks[0] + z0}–{ks[-1] + z0}, every {int(ks[1] - ks[0])}th",
             fontsize=5.2, color=mc, ha="center", va="bottom")
    return {"panel": "exploded stack with pulled xz / xy planes (F1-f layout)", "dark": dark, "planes": [int(k + z0) for k in ks],
            "xy_pulled_z": int(kmid + z0), "xz_pulled_y": int(ymid + y0), "LUT": "linear 1-99.8 percentile of the crop (as F1-f)",
            "outlines": {"GT": GTC, "2D": R.D2, "S1": R.S1}, "xz_display": "raw plane linearly interpolated along z to the physical aspect (whole-image), labels nearest-neighbour; outlines traced on that grid", "scale_bar": "10 um = %.1f lateral px" % (10 / UM_PX), **SOURCE}


# ---------------------------------------------------------------- cube (F2-e)
def cube_render(dark, alpha=0.5, size=2400):
    """Semi-transparent cube of the same crop: raw faces (1-99 % LUT, gamma 0.8, side faces linearly interpolated along z),
    S1 nuclei inside (one colour each); a 10-um scale bar drawn along the front-bottom x edge in world units."""
    fn = os.path.join(CACHE, f"cube_{'d' if dark else 'l'}")
    if os.path.exists(fn + ".png"):
        return Image.open(fn + ".png"), json.load(open(fn + ".json"))
    from scipy.ndimage import zoom
    ic, sc = img, s1
    lo, hi2 = np.percentile(ic, 1), np.percentile(ic, 99.0)
    Hc = nz * RAT

    def face(a2, pts, zrows):
        q = pv.PolyData(np.array(pts, float), faces=[4, 0, 1, 2, 3])
        q.active_texture_coordinates = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        a = zoom(a2, (RAT, 1), order=1) if zrows else a2
        g = np.clip((a - lo) / (hi2 - lo), 0, 1) ** 0.8
        t = pv.Texture((np.flipud(np.repeat(g[..., None], 3, -1)) * 255).astype(np.uint8)); t.interpolate = True
        return q, t
    p = R.plotter((size, size), dark)
    ns, bl = R.label_surface(sc, RAT)
    R.add_rgb_mesh(p, ns, [pv.Color(R.QUAL[b % len(R.QUAL)]).float_rgb for b in bl])
    faces = [(ic[-1], [(0, 0, Hc), (s, 0, Hc), (s, s, Hc), (0, s, Hc)]), (ic[:, 0, :], [(0, 0, 0), (s, 0, 0), (s, 0, Hc), (0, 0, Hc)]),
             (ic[:, :, -1], [(s, 0, 0), (s, s, 0), (s, s, Hc), (s, 0, Hc)])]
    for fi, (arr, pts) in enumerate(faces):
        q, t = face(arr.copy(), pts, fi > 0)
        p.add_mesh(q, texture=t, opacity=alpha, lighting=False)
    p.add_mesh(pv.Box(bounds=(0, s, 0, s, 0, Hc)).extract_feature_edges(), color="#c8c8c4" if dark else "#7a7a76", line_width=2)
    L = 10 / UM_PX; off = 0.06 * s
    bar = pv.Line((0, -off, 0), (L, -off, 0))
    p.add_mesh(bar, color="#ffffff" if dark else R.FG, line_width=7)
    ctr = np.array([s / 2, s / 2, Hc / 2]); rad = np.linalg.norm(ctr)
    p.camera_position = [tuple(ctr + rad * np.array([1.6, -2.0, 1.6])), tuple(ctr), (0, 0, 1)]
    p.enable_parallel_projection(); p.reset_camera(); p.camera.zoom(1.05); p.render()

    def proj(pt):
        r = p.renderer
        r.SetWorldPoint(*pt, 1.0); r.WorldToDisplay()
        x, y, _ = r.GetDisplayPoint()
        return x, p.window_size[1] - y
    b0, b1 = proj((0, -off, 0)), proj((L, -off, 0))
    raw = R.shot(p, ssao=False)
    a = np.asarray(raw)[..., 3]; ys_, xs_ = np.nonzero(a > 8); m = int(0.03 * size)
    bx0, by0 = max(xs_.min() - m, 0), max(ys_.min() - m, 0)
    im = raw.crop((bx0, by0, min(xs_.max() + m, raw.width), min(ys_.max() + m, raw.height)))
    meta = {"bar": [[b0[0] - bx0, b0[1] - by0], [b1[0] - bx0, b1[1] - by0]]}
    im.save(fn + ".png"); json.dump(meta, open(fn + ".json", "w"))
    return im, meta


def draw_cube(fig, W, H, x, y, w, h, dark, caption=True, fs=5.2):
    """Cube render in a (dark or light) image tile; scale bar label placed at the projected bar; source line underneath."""
    tc, mc = (DARK_TC, DARK_MC) if dark else (R.FG, R.MUTED)
    im, meta = cube_render(dark)
    cap_h = 4 * fs * 0.3528 * 1.25 + 1.0 if caption else 0.0   # four text lines in mm
    th = h - cap_h
    ax = fig.add_axes(_mm(fig, W, H, x, y, w, th))
    t = R.tile(im, dark, size=(int(im.height * w / th), im.height), pad=0.0)
    ax.imshow(t); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    offx = (t.width - im.width) / 2; offy = (t.height - im.height) / 2
    (ax0, ay0), (ax1, ay1) = meta["bar"]
    ax.text(max(offx + min(ax0, ax1), 0.04 * t.width), offy + max(ay0, ay1) + 0.035 * t.height, "10 µm", color="#ffffff" if dark else R.FG,
            fontsize=fs, ha="left", va="top")
    if caption:
        fig.text((x + w / 2) / W, 1 - (y + th + 0.6) / H,
                 f"{SOURCE['dataset']} test volume {SOURCE['volume']}\n(median rule); S1, seed 0; z/xy {RAT:.2f}\n"
                 f"raw planes on the faces (50 % opaque),\nS1 nuclei inside, one colour each",
                 fontsize=fs, color=R.MUTED, ha="center", va="top", linespacing=1.25)
    return {"panel": "semi-transparent cube (F2-e)", "dark": dark, "face_opacity": 0.5,
            "LUT": "1-99 percentile of the crop, gamma 0.8 (whole-image); side faces linearly interpolated along z",
            "scale_bar": "10 um along the x edge, drawn in world units (%.1f px)" % (10 / UM_PX), **SOURCE}
