"""Shared v2 render style (pyvista offscreen, CPU): palette, materials, lighting, ortho camera, autocrop, tiles.

Palette (team-lead decision 2026-11-09): S1 #2f6fde, 2D #E69F00 (ink #b26a00), original #6b6b70, text #1d1d1f,
muted #6b6b70, rules #e2e2e0, shell #9AA3B5 @ 0.16. Error classes: Paul Tol muted as in Fig. 3a
(miss #88CCEE, merge #CC6677, split #DDCC77, boundary #117733, false positive #AA4499); matched nuclei desaturated.
"""
import numpy as np
import pyvista as pv
from PIL import Image
from scipy import ndimage as ndi

pv.OFF_SCREEN = True
import json as _json
import os as _os
PALETTE = _json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "figures", "palette.json")))
S1, D2, ORIG = PALETTE["models"]["S1"], PALETTE["models"]["2D"], PALETTE["models"]["base"]
FG, MUTED, RULE, SHELL = PALETTE["text"], PALETTE["muted"], PALETTE["rule"], PALETTE["shell"]
GT = PALETTE["gt_outline"]
ERR = dict(PALETTE["errors"])
ERR_NAME = dict(PALETTE["error_names"])
QUAL = list(PALETTE["qualitative_instances"])
DARK_BG = tuple(PALETTE["tile_dark"])
LIGHT_BG = tuple(PALETTE["tile_light"])
# manuscript names: single source is the label table in ../figs/nc_style.py (LAB: models, DS_NAME: datasets)
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "figs"))
from nc_style import DS_NAME, LAB  # noqa: E402
LAB = dict(LAB, gt="Ground truth")
RENDER = dict(smoothing_iterations=24, smoothing_relaxation=0.5, ssao=dict(radius=6.0, bias=0.3, kernel_size=128, blur=True),
              material=dict(ambient=0.18, diffuse=0.75, specular=0.28, specular_power=28), camera="orthographic (parallel projection)")


def label_surface(lab, ratio, keep=None, f=1):
    if keep is not None:
        lab = np.where(np.isin(lab, keep), lab, 0)
    lab = lab[:, ::f, ::f]
    Z, Y, X = lab.shape
    img = pv.ImageData(dimensions=(X, Y, Z), spacing=(float(f), float(f), ratio))
    img.point_data["lab"] = lab.ravel(order="C").astype(np.int32)
    s = img.contour_labels(boundary_style="external", scalars="lab", smoothing=True, output_mesh_type="triangles",
                           smoothing_iterations=RENDER["smoothing_iterations"], smoothing_relaxation=RENDER["smoothing_relaxation"])
    bl = np.asarray(s.cell_data["boundary_labels"]); bl = (bl[:, 0] if bl.ndim == 2 else bl).astype(int)
    return s, bl


def shell_surface(fg, ratio, sigma):
    sm = ndi.gaussian_filter(fg.astype(np.float32), sigma=(sigma / ratio, sigma, sigma))
    sm = sm / max(sm.max(), 1e-6)
    Z, Y, X = sm.shape
    img = pv.ImageData(dimensions=(X, Y, Z), spacing=(1.0, 1.0, ratio))
    img.point_data["v"] = sm.ravel(order="C")
    return img.contour([0.18], scalars="v").smooth_taubin(n_iter=40)


def plotter(size=(1600, 1600), dark=False):
    p = pv.Plotter(off_screen=True, window_size=size, lighting="none")
    p.set_background("black" if dark else "white")
    key = pv.Light(position=(-1, 1, 1.4), focal_point=(0, 0, 0), intensity=0.75 if not dark else 0.85, light_type="camera light")
    fill = pv.Light(position=(1, 0.2, 0.6), focal_point=(0, 0, 0), intensity=0.3, light_type="camera light")
    rim = pv.Light(position=(0.2, -1, -0.8), focal_point=(0, 0, 0), intensity=0.45 if dark else 0.25, light_type="camera light")
    head = pv.Light(light_type="headlight", intensity=0.25)
    for l_ in (key, fill, rim, head):
        p.add_light(l_)
    return p


def add_rgb_mesh(p, surf, rgb, opacity=1.0):
    surf = surf.copy()
    surf.cell_data["rgb"] = (np.asarray(rgb) * 255).astype(np.uint8)
    return p.add_mesh(surf, scalars="rgb", rgb=True, smooth_shading=True, opacity=opacity, **RENDER["material"])


def ortho_camera(p, azim=35.0, elev=25.0, zoom=1.0, view="xy"):
    p.camera_position = view
    p.camera.azimuth = azim
    p.camera.elevation = -elev
    p.enable_parallel_projection()
    p.reset_camera()
    p.camera.zoom(zoom)


def shot(p, ssao=True):
    if ssao:
        p.enable_ssao(**RENDER["ssao"])
    img = p.screenshot(transparent_background=True, return_img=True)
    p.close()
    return Image.fromarray(img)


def autocrop(im, margin=0.02):
    a = np.asarray(im)[..., 3]
    ys, xs = np.nonzero(a > 8)
    if len(ys) == 0:
        return im
    m = int(margin * max(im.size))
    return im.crop((max(xs.min() - m, 0), max(ys.min() - m, 0), min(xs.max() + m, im.width), min(ys.max() + m, im.height)))


def tile(im, dark=False, size=None, pad=0.04):
    """Composite an RGBA render onto a subtle vertical gradient tile (light #f7f7f5->white or dark slate)."""
    top, bot = DARK_BG if dark else LIGHT_BG
    if size is None:
        w, h = int(im.width * (1 + 2 * pad)), int(im.height * (1 + 2 * pad))
    else:
        w, h = size
    t = np.linspace(0, 1, h)[:, None, None]
    c0 = np.array(Image.new("RGB", (1, 1), top).getpixel((0, 0)), float)
    c1 = np.array(Image.new("RGB", (1, 1), bot).getpixel((0, 0)), float)
    g = (c0 * (1 - t) + c1 * t) * np.ones((1, w, 1))
    bg = Image.fromarray(g.astype(np.uint8)).convert("RGBA")
    if size is not None:
        sc = min(w * (1 - 2 * pad) / im.width, h * (1 - 2 * pad) / im.height)
        im = im.resize((int(im.width * sc), int(im.height * sc)), Image.LANCZOS)
    bg.alpha_composite(im, ((w - im.width) // 2, (h - im.height) // 2))
    return bg.convert("RGB")


def classify_preds(gt, pm):
    """Per predicted instance: match / merge / split / boundary / other (false positive); missed GT ids.
    Same colouring rule as render_vol.classify_preds (colouring only; no metric is reported from it)."""
    from collections import defaultdict
    m = (gt > 0) & (pm > 0)
    gsz = np.bincount(gt.ravel()); psz = np.bincount(pm.ravel())
    K = int(pm.max()) + 1
    key = gt[m].astype(np.int64) * K + pm[m]
    u, c = np.unique(key, return_counts=True)
    g, p_ = u // K, u % K
    iou = c / (gsz[g] + psz[p_] - c); covg = c / gsz[g]
    pids = np.unique(pm); pids = pids[pids > 0]
    gids = np.unique(gt); gids = gids[gids > 0]
    cls = {int(j): "other" for j in pids}
    pbest = defaultdict(float); gbest = defaultdict(float)
    for gi, pj, v in zip(g, p_, iou):
        pbest[int(pj)] = max(pbest[int(pj)], v); gbest[int(gi)] = max(gbest[int(gi)], v)
        if v >= 0.5:
            cls[int(pj)] = "match"
    big = defaultdict(int)
    for gi, pj, cv in zip(g, p_, covg):
        if cv > 0.5:
            big[int(pj)] += 1
    for pj, k in big.items():
        if k >= 2:
            cls[pj] = "merge"
    matched_g = {int(gi) for gi, v in zip(g, iou) if v >= 0.5}
    parts = defaultdict(list)
    for gi, pj, cv in zip(g, p_, covg):
        if cv >= 0.15:
            parts[int(gi)].append(int(pj))
    for gi, ps in parts.items():
        if gi not in matched_g and len(ps) >= 2:
            for pj in ps:
                if cls[pj] != "merge":
                    cls[pj] = "split"
    for pj in list(cls):
        if cls[pj] == "other" and pbest[pj] >= 0.1:
            cls[pj] = "boundary"
    missed = [int(x) for x in gids if gbest[int(x)] < 0.1]
    return cls, missed, dict(gbest)


def gt_best_iou(gt, pm):
    m = (gt > 0) & (pm > 0)
    gsz = np.bincount(gt.ravel()); psz = np.bincount(pm.ravel())
    K = int(pm.max()) + 1
    u, c = np.unique(gt[m].astype(np.int64) * K + pm[m], return_counts=True)
    g, p_ = u // K, u % K
    iou = c / (gsz[g] + psz[p_] - c)
    best = np.zeros(int(gt.max()) + 1)
    np.maximum.at(best, g, iou)
    return best
