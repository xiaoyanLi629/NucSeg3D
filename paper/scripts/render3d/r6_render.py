"""Round-6 render pipeline (team-lead 2026-11-10, user feedback on R5-Fig2-cube): black image tiles, more vivid colours, sharp output.

Blur diagnosis of R5 (see r6_compare.py for the zoom crops):
  1. the d grid was rendered (1300 px), cropped, resampled into 800-px tiles, composited into a 500-dpi PNG (text rasterised too)
     and that PNG was embedded in Fig. 2 and resampled again by matplotlib to the 300-dpi figure resolution;
  2. translucent nuclei were blended without depth peeling and without supersampling (jagged / smeared edges);
  3. the cube side faces of E15_Ab2 are 47 native z planes stretched 4.65x (intrinsically soft).
Fixes here: every tile is rendered at 2x its final pixel size (>= 1000 ppi at print size) and downsampled with a Lanczos filter
(supersampling anti-aliasing), depth peeling for the translucent layers, tiles are embedded unresampled
(imshow interpolation="none") with vector text drawn by matplotlib, PNG previews only (lossless).

Vivid colours: hue of every palette colour is kept (HLS hue unchanged); only lightness / saturation are adjusted for a black
background (derived values written to figures/palette_vivid_dark.json together with a CVD check, cvd_vivid()).
"""
import colorsys
import itertools
import json
import os

import numpy as np
import pyvista as pv
from PIL import Image

import rstyle as R

pv.OFF_SCREEN = True
HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.abspath(os.path.join(HERE, "..", "..", "figures"))
CACHE = os.path.join(HERE, "out", "r6_cache"); os.makedirs(CACHE, exist_ok=True)
SEL = json.load(open(os.path.join(HERE, "..", "figs", "ortho_selection.json")))["volumes"]


def _hex(rgb):
    return "#%02X%02X%02X" % tuple(int(round(255 * max(0, min(1, c)))) for c in rgb)


def adj(h, l=None, s=None, s_mul=None):
    """Keep the HLS hue of palette colour h; set lightness l and/or saturation s (or scale saturation)."""
    r, g, b = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    if s_mul is not None:
        ss = min(1.0, ss * s_mul)
    if s is not None:
        ss = s
    if l is not None:
        ll = l
    return _hex(colorsys.hls_to_rgb(hh, ll, ss))


E0 = R.PALETTE["errors"]
# matched: bright grey with a cool tint taken from the NPG blue hue (merge colour hue at low saturation)
# lightness values chosen by a small grid search that maximises the minimum CAM02-UCS distance under the four CVD
# simulations (including the black background); result 14.4 vs 9.5 for the palette.json error colours (see cvd_vivid)
ERRV = {"match": adj(E0["merge"], l=0.86, s=0.16),
        "boundary": adj(E0["boundary"], l=0.62, s_mul=1.15),
        "split": adj(E0["split"], l=0.38, s_mul=1.6),
        "merge": adj(E0["merge"], l=0.56, s_mul=1.55),
        "other": adj(E0["other"], l=0.52),
        "miss": adj(E0["miss"], l=0.40)}
QUALV = [adj(c, l=0.62, s_mul=1.9) for c in R.PALETTE["qualitative_instances"]]
MAT = dict(ambient=0.12, diffuse=0.92, specular=0.45, specular_power=40)
MATCH_OP, GHOST_OP, SHELL_OP = 0.62, 0.7, 0.05
PPI = 1000


def cvd_vivid(bg="#000000"):
    """Same test as cvd_check.py (CAM02-UCS distance, severity 100) for the vivid error colours, plus their distance to black."""
    from colorspacious import cspace_convert
    rgb = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255  # noqa: E731
    out = {}
    cols = dict(ERRV, background=bg)
    for cvd in ("normal", "deuteranomaly", "protanomaly", "tritanomaly"):
        lab = {}
        for k, h in cols.items():
            c = rgb(h)
            if cvd != "normal":
                c = np.clip(cspace_convert(c, {"name": "sRGB1+CVD", "cvd_type": cvd, "severity": 100}, "sRGB1"), 0, 1)
            lab[k] = cspace_convert(c, "sRGB1", "CAM02-UCS")
        d = sorted((float(np.linalg.norm(lab[a] - lab[b])), a, b) for a, b in itertools.combinations(lab, 2))
        out[cvd] = {"min": round(d[0][0], 1), "closest": d[0][1:], "pairs_below_15": [(round(x, 1), a, b) for x, a, b in d if x < 15]}
    ref = json.load(open(os.path.join(FIGD, "palette_cvd.json")))["errors"]
    rec = {"rule": "hue of each palette.json colour kept (HLS); lightness/saturation adjusted for a black background",
           "errors_vivid_dark": ERRV, "qualitative_instances_vivid": QUALV, "material": MAT,
           "opacity": {"matched": MATCH_OP, "missed_ghost": GHOST_OP, "specimen_shell": SHELL_OP},
           "cvd_vivid_incl_black_bg": out, "cvd_palette_json_errors_for_reference": ref}
    json.dump(rec, open(os.path.join(FIGD, "palette_vivid_dark.json"), "w"), indent=1)
    return rec


def plotter(px):
    p = pv.Plotter(off_screen=True, window_size=(px, px), lighting="none")
    p.set_background("black")
    for l_ in (pv.Light(position=(-1, 1, 1.4), focal_point=(0, 0, 0), intensity=0.95, light_type="camera light"),
               pv.Light(position=(1, 0.2, 0.6), focal_point=(0, 0, 0), intensity=0.35, light_type="camera light"),
               pv.Light(position=(0.2, -1, -0.8), focal_point=(0, 0, 0), intensity=0.45, light_type="camera light"),
               pv.Light(light_type="headlight", intensity=0.2)):
        p.add_light(l_)
    p.enable_depth_peeling(number_of_peels=10, occlusion_ratio=0.0)
    return p


def mesh(p, surf, rgb, opacity=1.0):
    surf = surf.copy()
    surf.cell_data["rgb"] = (np.asarray(rgb) * 255).astype(np.uint8)
    p.add_mesh(surf, scalars="rgb", rgb=True, smooth_shading=True, opacity=opacity, **MAT)


def shoot(p):
    p.enable_ssao(**R.RENDER["ssao"])
    im = p.screenshot(transparent_background=True, return_img=True)
    p.close()
    return Image.fromarray(im)


def render_err_set(ds, kinds, px):
    """kinds rendered with one camera at px x px (2x supersampled), cropped to their common bbox. RGBA, black transparent."""
    key = os.path.join(CACHE, f"{ds}_{'-'.join(kinds)}_{px}")
    if all(os.path.exists(f"{key}_{k}.png") for k in kinds):
        return [Image.open(f"{key}_{k}.png") for k in kinds]
    Z = np.load(os.path.join(HERE, "export", f"{ds}.npz"))
    gt, ratio = Z["gt"], float(Z["ratio"])
    shell = R.shell_surface(gt > 0, ratio, 0.6 * float(Z["d"]))
    ims = []
    for k in kinds:
        p = plotter(px)
        if k == "gt":
            s, bl = R.label_surface(gt, ratio)
            mesh(p, s, [pv.Color(QUALV[b % len(QUALV)]).float_rgb for b in bl])
        else:
            lab = Z[k]
            s, bl = R.label_surface(lab, ratio)
            cls, missed, _ = R.classify_preds(gt, lab)
            kk = np.array([cls.get(int(b), "other") for b in bl])
            for c in ("match", "boundary", "split", "merge", "other"):
                sel = np.nonzero(kk == c)[0]
                if len(sel):
                    sub = s.extract_cells(sel).extract_surface(algorithm="dataset_surface")
                    mesh(p, sub, np.tile(pv.Color(ERRV[c]).float_rgb, (sub.n_cells, 1)), MATCH_OP if c == "match" else 1.0)
            if missed:
                gs, _ = R.label_surface(gt, ratio, missed)
                p.add_mesh(gs, color=ERRV["miss"], opacity=GHOST_OP, smooth_shading=True, **MAT)
        p.add_mesh(shell, color=R.SHELL, opacity=SHELL_OP, smooth_shading=True)
        R.ortho_camera(p, 35, 25)
        ims.append(shoot(p))
    a = np.max([np.asarray(i)[..., 3] for i in ims], 0)
    ys, xs = np.nonzero(a > 8); m = int(0.012 * px)
    box = (max(xs.min() - m, 0), max(ys.min() - m, 0), min(xs.max() + m, a.shape[1]), min(ys.max() + m, a.shape[0]))
    ims = [i.crop(box) for i in ims]
    for k, i in zip(kinds, ims):
        i.save(f"{key}_{k}.png")
    return ims


def on_black(im, size):
    """Fit RGBA render into a black tile of size (w, h) px: downsample by Lanczos (supersampling AA), centre, composite."""
    w, h = size
    sc = min(w * 0.96 / im.width, h * 0.96 / im.height)
    r = im.resize((max(1, int(round(im.width * sc))), max(1, int(round(im.height * sc)))), Image.LANCZOS)
    t = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    t.alpha_composite(r, ((w - r.width) // 2, (h - r.height) // 2))
    return t.convert("RGB"), sc


def mm_px(mm, ppi=PPI):
    return int(round(mm / 25.4 * ppi))


def cube_render(ds, px, face_op=0.42):
    """Semi-transparent cube (as F2-e) for any exported volume, black background, vivid instance colours, 2x supersampled."""
    fn = os.path.join(CACHE, f"cube_{ds}_{px}")
    if os.path.exists(fn + ".png"):
        return Image.open(fn + ".png"), json.load(open(fn + ".json"))
    from scipy.ndimage import zoom
    Z = np.load(os.path.join(HERE, "export", f"{ds}.npz"))
    GT, IMG, RAT, VOX = Z["gt"], Z["img"].astype(np.float32) / 255.0, float(Z["ratio"]), np.asarray(Z["vox"], float)
    s = int(min(220, GT.shape[1], GT.shape[2]))
    zc = int(np.argmax([(GT[k] > 0).sum() for k in range(GT.shape[0])]))
    nz = int(min(round(s / RAT), GT.shape[0]))
    z0 = int(np.clip(zc - nz // 2, 0, GT.shape[0] - nz))
    cy, cx = [int(v) for v in np.argwhere(GT > 0)[:, 1:].mean(0)]
    y0 = int(np.clip(cy - s // 2, 0, GT.shape[1] - s)); x0 = int(np.clip(cx - s // 2, 0, GT.shape[2] - s))
    sl = (slice(z0, z0 + nz), slice(y0, y0 + s), slice(x0, x0 + s))
    ic, sc = IMG[sl], Z["S1"][sl]
    lo, hi = np.percentile(ic, 1), np.percentile(ic, 99.0)
    Hc = nz * RAT

    def face(a2, pts, zrows):
        q = pv.PolyData(np.array(pts, float), faces=[4, 0, 1, 2, 3])
        q.active_texture_coordinates = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        a = zoom(a2, (RAT, 1), order=1) if (zrows and RAT > 1.01) else a2
        g = np.clip((a - lo) / (hi - lo), 0, 1) ** 0.8
        t = pv.Texture((np.flipud(np.repeat(g[..., None], 3, -1)) * 255).astype(np.uint8)); t.interpolate = True
        return q, t
    p = plotter(px)
    ns, bl = R.label_surface(sc, RAT)
    mesh(p, ns, [pv.Color(QUALV[b % len(QUALV)]).float_rgb for b in bl])
    faces = [(ic[-1], [(0, 0, Hc), (s, 0, Hc), (s, s, Hc), (0, s, Hc)]), (ic[:, 0, :], [(0, 0, 0), (s, 0, 0), (s, 0, Hc), (0, 0, Hc)]),
             (ic[:, :, -1], [(s, 0, 0), (s, s, 0), (s, s, Hc), (s, 0, Hc)])]
    for fi, (arr, pts) in enumerate(faces):
        q, t = face(arr.copy(), pts, fi > 0)
        p.add_mesh(q, texture=t, opacity=face_op, lighting=False)
    p.add_mesh(pv.Box(bounds=(0, s, 0, s, 0, Hc)).extract_feature_edges(), color="#d0d0cc", line_width=max(2, px // 900))
    L = 10 / float(VOX[2]); off = 0.06 * s
    p.add_mesh(pv.Line((0, -off, 0), (L, -off, 0)), color="#ffffff", line_width=max(6, px // 300))
    ctr = np.array([s / 2, s / 2, Hc / 2]); rad = np.linalg.norm(ctr)
    p.camera_position = [tuple(ctr + rad * np.array([1.6, -2.0, 1.6])), tuple(ctr), (0, 0, 1)]
    p.enable_parallel_projection(); p.reset_camera(); p.camera.zoom(1.05); p.render()

    def proj(pt):
        r = p.renderer
        r.SetWorldPoint(*pt, 1.0); r.WorldToDisplay()
        x, y, _ = r.GetDisplayPoint()
        return x, p.window_size[1] - y
    b0, b1 = proj((0, -off, 0)), proj((L, -off, 0))
    raw = shoot(p)
    a = np.asarray(raw)[..., 3]; ys_, xs_ = np.nonzero(a > 8); m = int(0.02 * px)
    bx0, by0 = max(xs_.min() - m, 0), max(ys_.min() - m, 0)
    im = raw.crop((bx0, by0, min(xs_.max() + m, raw.width), min(ys_.max() + m, raw.height)))
    vol = SEL[ds]
    meta = {"bar": [[b0[0] - bx0, b0[1] - by0], [b1[0] - bx0, b1[1] - by0]], "dataset": R.DS_NAME[ds], "volume": vol["id"],
            "ratio": RAT, "voxel_um_zyx": VOX.tolist(), "crop_zyx": [z0, y0, x0, nz, s, s], "face_opacity": face_op,
            "LUT": "1-99 percentile of the crop, gamma 0.8 (whole-image)" + ("; side faces linearly interpolated along z" if RAT > 1.01 else ""),
            "render_px": px, "masks": "S1 seed 0 (07992b55), evaluated final masks"}
    im.save(fn + ".png"); json.dump(meta, open(fn + ".json", "w"))
    return im, meta


# ---------------------------------------------------------------- figure drawing (matplotlib, vector text, unresampled tiles)
def _ax(fig, W, H, x, y, w, h):
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def draw_twide(fig, W, H, x, y, w, h, dss, kinds=("2D", "S1"), render_px=2600, tc=None):
    """Rows = models, columns = datasets; black tiles; vivid error colours; per-volume F1@0.5 in the tile corner."""
    from matplotlib.patches import Patch
    FG, MUT = (tc or R.FG), R.MUTED
    lab_w, head_h, leg_h, g = 4.5, 6.5, 5.0, 0.6
    n = len(dss); tw = (w - lab_w - g * (n - 1)) / n; th = (h - head_h - leg_h - g * (len(kinds) - 1)) / len(kinds)
    tpx = (mm_px(tw), mm_px(th))
    rec = {"tiles_px": tpx, "tile_mm": [round(tw, 2), round(th, 2)], "ppi": PPI, "render_px": render_px, "volumes": {}}
    for c, ds in enumerate(dss):
        ims = render_err_set(ds, list(kinds), render_px)
        x0 = x + lab_w + c * (tw + g)
        fig.text((x0 + tw / 2) / W, 1 - (y + head_h - 0.8) / H, f"{R.DS_NAME[ds]}\nz/xy {SEL[ds]['ratio']:.2f}", ha="center", va="bottom",
                 fontsize=6.0, fontweight="bold", color=FG, linespacing=1.15)
        rec["volumes"][ds] = {"id": SEL[ds]["id"], "f1": {k: SEL[ds]["f1"][k] for k in kinds}}
        for r_, (k, im) in enumerate(zip(kinds, ims)):
            ax = _ax(fig, W, H, x0, y + head_h + r_ * (th + g), tw, th)
            t, sc = on_black(im, tpx)
            ax.imshow(np.asarray(t), interpolation="none"); ax.set_xticks([]); ax.set_yticks([])
            rec["volumes"][ds][f"downsample_{k}"] = round(1 / sc, 2)
            for sp in ax.spines.values():
                sp.set_visible(k == "S1"); sp.set_color(R.S1); sp.set_linewidth(0.8)
            ax.text(0.96, 0.03, f"{SEL[ds]['f1'][k]:.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6,
                    color=R.S1 if k == "S1" else "#d0d0d4", fontweight="bold" if k == "S1" else "normal")
            if c == 0:
                fig.text((x + lab_w - 1.2) / W, 1 - (y + head_h + r_ * (th + g) + th / 2) / H, R.LAB[k], rotation=90, ha="right", va="center",
                         fontsize=6.3, fontweight="bold", color=R.S1 if k == "S1" else R.D2)
    leg = [Patch(fc=ERRV[k], ec="none", label=R.ERR_NAME[k]) for k in ("match", "boundary", "split", "merge", "other", "miss")]
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=((x + lab_w + (w - lab_w) / 2) / W, 1 - (y + h) / H), ncol=6, fontsize=5.6,
               frameon=False, handlelength=1.1, columnspacing=1.0, borderaxespad=0.1, labelcolor=FG)
    rec.update(errors_vivid_dark=ERRV, material=MAT, opacity={"matched": MATCH_OP, "missed_ghost": GHOST_OP, "shell": SHELL_OP},
               aa="2x+ supersampling (render_px window, Lanczos downsample to the tile), depth peeling, SSAO", embed="imshow interpolation none (unresampled)")
    return rec


def draw_cube(fig, W, H, x, y, w, h, ds, render_px=4000, fs=5.2, cap_color=None):
    import matplotlib.patheffects as pe
    im, meta = cube_render(ds, render_px)
    cap_h = 4 * fs * 0.3528 * 1.25 + 1.0
    th = h - cap_h
    ax = _ax(fig, W, H, x, y, w, th)
    tpx = (mm_px(w), mm_px(th))
    t, sc = on_black(im, tpx)
    ax.imshow(np.asarray(t), interpolation="none"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    offx = (tpx[0] - im.width * sc) / 2; offy = (tpx[1] - im.height * sc) / 2
    (a0, b0), (a1, b1) = meta["bar"]
    ax.text(max(offx + min(a0, a1) * sc, 0.04 * tpx[0]), offy + max(b0, b1) * sc + 0.03 * tpx[1], "10 µm", color="#ffffff", fontsize=fs,
            ha="left", va="top", path_effects=[pe.withStroke(linewidth=1.0, foreground="black")])
    fig.text((x + w / 2) / W, 1 - (y + th + 0.6) / H,
             f"{meta['dataset']} test volume {meta['volume']}\n(median rule); S1, seed 0; z/xy {meta['ratio']:.2f}\n"
             f"raw planes on the faces ({int(round(100 * meta['face_opacity']))} % opaque),\nS1 nuclei inside, one colour each",
             fontsize=fs, color=cap_color or R.MUTED, ha="center", va="top", linespacing=1.25)
    meta = dict(meta, tile_px=tpx, ppi=PPI, downsample=round(1 / sc, 2), instance_colours="qualitative_instances_vivid (palette_vivid_dark.json)",
                embed="imshow interpolation none (unresampled)")
    return meta
