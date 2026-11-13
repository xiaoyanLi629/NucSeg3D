"""R10 Fig. 4 (gallery R10-Fig4): installed R9 recoloured with the unified 3D-viewer scheme (figures/palette_viewer.json,
user 2026-11-11): warm-grey containers #EDECE9, error colours of the viewer (renders and xz overlays), matched opacity 0.9,
near-black low-alpha SSAO shading dropped before compositing. Layout, data and numbers identical to R9 (fig4r9.py, imported)."""
import json
import os

import numpy as np
import pyvista as pv
from PIL import Image

import fig4r6 as F
import fig4r9 as B
import r6_render as V
import rstyle as R

PVW = json.load(open(os.path.join(F.HERE, "..", "..", "figures", "palette_viewer.json")))
F.ERRV.update(PVW["errors"])
V.MATCH_OP = PVW["matched_opacity"]
B.TINT = PVW["pane"]
B.tint_box.__defaults__ = (PVW["pane"],)


def render3d_v(m, k, px):
    cache = os.path.join(F.HERE, "out", f"fig4r10_viewer_{m}_k{k}_{px}.png")
    if os.path.exists(cache):
        return Image.open(cache)
    gt, pm, ratio, cls, missed = F.classify(m, k)
    zz = np.nonzero((gt > 0).any((1, 2)))[0]; sl = slice(max(zz.min() - 1, 0), zz.max() + 2)
    gt, pm = gt[sl], pm[sl]
    p = V.plotter(px)
    s, bl = R.label_surface(pm, ratio)
    kk = np.array([cls.get(int(b), "other") for b in bl])
    for c_ in ("match", "boundary", "split", "merge", "other"):
        sel = np.nonzero(kk == c_)[0]
        if len(sel):
            sub = s.extract_cells(sel).extract_surface(algorithm="dataset_surface")
            V.mesh(p, sub, np.tile(pv.Color(F.ERRV[c_]).float_rgb, (sub.n_cells, 1)), V.MATCH_OP if c_ == "match" else 1.0)
    if missed:
        gs, _ = R.label_surface(gt, ratio, missed)
        p.add_mesh(gs, color=F.ERRV["miss"], opacity=V.GHOST_OP, smooth_shading=True, **V.MAT)
    R.ortho_camera(p, 35, 25)
    im = V.shoot(p)
    a = np.asarray(im).copy(); a[..., 3][a[..., 3] < PVW["dehalo_alpha_below"]] = 0; im = Image.fromarray(a)
    al = a[..., 3]; ys, xs = np.nonzero(al > 8); mg = int(0.01 * px)
    im = im.crop((max(xs.min() - mg, 0), max(ys.min() - mg, 0), min(xs.max() + mg, al.shape[1]), min(ys.max() + mg, al.shape[0])))
    im.save(cache)
    return im


F.render3d = render3d_v

if __name__ == "__main__":
    fig, cnt, xr = B.build()
    F.save(fig, "R10-Fig4", dict(version="R10: R9 with the unified 3D-viewer colour scheme (palette_viewer.json)", e_error_counts_on_crop=cnt))
    pj = os.path.join(F.GA, "R10-Fig4.provenance.json")
    d = json.load(open(pj)); d["script"] = "scripts/render3d/fig4r10.py (layout fig4r9.py)"; d["script_md5"] = F.md5(__file__)
    d["palette"] = "figures/palette_viewer.json"
    d["frame_rule"] = "dot (and bold value) when all three paired seed differences S1 - 2D have the same sign; descriptive, no test"
    json.dump(d, open(pj, "w"), indent=1, ensure_ascii=False, default=str)
