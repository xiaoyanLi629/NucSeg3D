"""Re-render the network-output glyph thumbnail (flow_glyphs) in the palette.json d_z ramp, and check its direction.

Same data, crop, sampling, glyph geometry and camera as paper/scripts/render3d/fig1_assets.py:82-113 (nc-figures);
the only change is the colour map (palette.json "diverging" instead of PRGn) and the nucleus shell colour (palette "shell").
Direction check: cosine between each drawn arrow vector and the vector from the arrow base to the nucleus centroid, both
in render coordinates (x, y, z * z/xy), over glyphs inside the nucleus. Cellpose flows must point inwards (cos > 0).
Run with paper/scripts/render3d/.venv/bin/python.
"""
import json
import os

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage as ndi

pv.OFF_SCREEN = True
HERE = os.path.dirname(os.path.abspath(__file__))
R3 = os.path.abspath(os.path.join(HERE, "..", "..", "scripts", "render3d"))
PJ = json.load(open(os.path.join(HERE, "..", "palette.json")))
ratio = float(np.load(os.path.join(R3, "export", "mouse_embryo.npz"))["ratio"])

F = np.load(os.path.join(R3, "export", "flow_mouse_embryo.npz"))
g, dP, nid = F["gt"], F["dP"].astype(np.float32), int(F["nucleus"])
zz, yy, xx = np.nonzero(g == nid)
cz, cy, cx = int(zz.mean()), int(yy.mean()), int(xx.mean())
ry = max(int((yy.max() - yy.min()) * 0.75), 8)
sub = (slice(None), slice(max(cy - ry, 0), cy + ry), slice(max(cx - ry, 0), cx + ry))
gs, dPs = g[sub], dP[(slice(None),) + sub]
Zs, Ys, Xs = gs.shape
box = pv.ImageData(dimensions=(Xs, Ys, Zs), spacing=(1, 1, ratio))
box.point_data["n"] = (gs == nid).astype(np.float32).ravel(order="C")
nuc = box.contour([0.5], scalars="n").smooth_taubin(n_iter=40)
step = 5
near = ndi.binary_dilation(gs == nid, iterations=2)
Zi, Yi, Xi = np.meshgrid(np.arange(0, Zs, 2), np.arange(0, Ys, step), np.arange(0, Xs, step), indexing="ij")
inside = near[Zi, Yi, Xi]
pts = np.c_[Xi[inside], Yi[inside], Zi[inside] * ratio].astype(float)
vec = np.c_[dPs[2][Zi, Yi, Xi][inside], dPs[1][Zi, Yi, Xi][inside], dPs[0][Zi, Yi, Xi][inside]]

# ---- direction check in render coordinates
m = gs == nid
mz, my, mx = np.nonzero(m)
cen = np.array([mx.mean(), my.mean(), mz.mean() * ratio])
inn = m[Zi, Yi, Xi][inside]
to_c = cen[None] - pts
cos = (vec * to_c).sum(1) / (np.linalg.norm(vec, axis=1) * np.linalg.norm(to_c, axis=1) + 1e-9)
chk = dict(nucleus=nid, n_glyphs=int(len(pts)), n_inside=int(inn.sum()), mean_cos_inside=float(cos[inn].mean()),
           frac_inward_inside=float((cos[inn] > 0).mean()))
print(chk)

cloud = pv.PolyData(pts)
cloud["v"] = vec
cloud["axial"] = vec[:, 2]
glyph = cloud.glyph(orient="v", scale=False, factor=4.0, geom=pv.Arrow(tip_length=0.38, tip_radius=0.17, shaft_radius=0.07))
p = pv.Plotter(off_screen=True, window_size=(1000, 1000), lighting="none")
p.set_background("white")
p.add_light(pv.Light(light_type="headlight", intensity=0.35))
p.add_light(pv.Light(position=(-1, -1, 2), focal_point=(0, 0, 0), intensity=0.6, light_type="scene light"))
p.add_light(pv.Light(position=(2, 1, 1), focal_point=(0, 0, 0), intensity=0.3, light_type="scene light"))
p.add_mesh(nuc, color=PJ["shell"], opacity=0.22, smooth_shading=True, specular=0.2)
lim = float(np.percentile(np.abs(vec[:, 2]), 90))
cmap = LinearSegmentedColormap.from_list("div", PJ["diverging"])
p.add_mesh(glyph, scalars="axial", cmap=cmap, clim=(-lim, lim), show_scalar_bar=False, smooth_shading=True)
p.camera_position = [(Xs * 2.2, -Ys * 1.2, Zs * ratio * 2.4), (Xs / 2, Ys / 2, Zs * ratio / 2), (0, 0, 1)]
p.camera.zoom(1.3)
out = os.path.join(HERE, "assets", "flow_glyphs_pal.png")
p.screenshot(out, transparent_background=True)
p.close()
json.dump(dict(check=chk, source="export/flow_mouse_embryo.npz (S1 seed-0 dP, E15_Ab2, flow_export.py)",
               recipe="fig1_assets.py:82-113 with palette diverging cmap and shell colour", clim=lim),
          open(os.path.join(HERE, "assets", "flow_glyphs_pal.json"), "w"), indent=1)
print("wrote", out)
