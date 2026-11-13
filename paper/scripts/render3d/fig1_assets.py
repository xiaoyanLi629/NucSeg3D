"""Raster assets for the Fig. 1 schematic, rendered from real data (pyvista offscreen).

input_volume.png : mouse-embryo test volume (E15_Ab2, eval grid) as orthogonal image planes plus a translucent
                   isosurface of the ground-truth nuclei inside a wire box (z scaled to physical aspect).
flow_glyphs.png  : S1 seed-0 flow field (dP: axial, lateral) as arrows on and around one real nucleus,
                   arrows coloured by the axial component; nucleus surface translucent.
prob_map.png     : S1 foreground probability sigmoid(cp) on the central plane of that crop.
slices/*.png     : five raw xy planes of the volume for the per-slice encoder drawing.
Sources: export/mouse_embryo.npz (render_export.py), export/flow_mouse_embryo.npz (flow_export.py).
"""
import os

import numpy as np
import pyvista as pv
from PIL import Image

pv.OFF_SCREEN = True
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "fig1")
os.makedirs(os.path.join(OUT, "slices"), exist_ok=True)
ACC = "#2f6fde"
SHELL = "#9aa3b5"

V = np.load(os.path.join(HERE, "export", "mouse_embryo.npz"))
img, gt, ratio = V["img"].astype(np.float32) / 255.0, V["gt"], float(V["ratio"])
Z, Y, X = img.shape
# crop a cube-ish region for a compact icon (physical units)
y0, x0, s = 90, 90, 200
zc = int(np.argmax([(gt[k, y0:y0 + s, x0:x0 + s] > 0).sum() for k in range(Z)]))
nz = int(round(s / ratio))
z0 = int(np.clip(zc - nz // 2, 0, Z - nz))
img_c, gt_c = img[z0:z0 + nz, y0:y0 + s, x0:x0 + s], gt[z0:z0 + nz, y0:y0 + s, x0:x0 + s]
Z = img_c.shape[0]


def lights(p):
    p.add_light(pv.Light(light_type="headlight", intensity=0.35))
    p.add_light(pv.Light(position=(-1, -1, 2), focal_point=(0, 0, 0), intensity=0.6, light_type="scene light"))
    p.add_light(pv.Light(position=(2, 1, 1), focal_point=(0, 0, 0), intensity=0.3, light_type="scene light"))


def iso_cam(p, zoom=1.15, az=-40, el=28):
    p.camera_position = "xy"
    p.camera.elevation = -60
    p.camera.azimuth = az
    p.camera.elevation = el - 60 + 60
    p.reset_camera()
    p.camera.zoom(zoom)


# ---------------- input volume
grid = pv.ImageData(dimensions=(s, s, Z), spacing=(1, 1, ratio))
grid.point_data["I"] = img_c.ravel(order="C")
grid.point_data["L"] = (gt_c > 0).astype(np.float32).ravel(order="C")
p = pv.Plotter(off_screen=True, window_size=(1000, 1000), lighting="none")
p.set_background("white")
lights(p)
lo, hi = np.percentile(img_c, (1, 99.7))
sl = grid.slice(normal="z", origin=(s / 2, s / 2, ratio * (Z // 2)))
p.add_mesh(sl, scalars="I", cmap="gray", clim=(lo, hi), show_scalar_bar=False, lighting=False)
lab = pv.ImageData(dimensions=(s, s, Z), spacing=(1, 1, ratio))
gl = gt_c.copy()
gl[:Z // 2] = 0  # nuclei above the displayed plane only, so the plane reads as a cut through the volume
lab.point_data["lab"] = gl.ravel(order="C").astype(np.int32)
surf = lab.contour_labels(boundary_style="external", scalars="lab", smoothing=True, smoothing_iterations=24, output_mesh_type="triangles")
QUAL = ["#88a9c9", "#c9a27e", "#9cc0a4", "#c99aa8", "#b7b0d6", "#d6c98e", "#8fbfc0", "#c7a6c6"]
bl = np.asarray(surf.cell_data["boundary_labels"]); bl = (bl[:, 0] if bl.ndim == 2 else bl).astype(int)
surf.cell_data["rgb"] = (np.array([pv.Color(QUAL[b % len(QUAL)]).float_rgb for b in bl]) * 255).astype(np.uint8)
p.add_mesh(surf, scalars="rgb", rgb=True, smooth_shading=True, specular=0.15)
p.add_mesh(grid.outline(), color="#1d1d1f", line_width=1.5)
p.camera_position = [(s * 2.6, -s * 1.6, Z * ratio * 3.2), (s / 2, s / 2, Z * ratio / 2), (0, 0, 1)]
p.camera.zoom(1.05)
p.enable_ssao(radius=8, kernel_size=64)
p.screenshot(os.path.join(OUT, "input_volume.png"), transparent_background=True)
p.close()

# ---------------- raw slices for the encoder drawing
for k, zz in enumerate(np.linspace(Z * 0.2, Z * 0.8, 5).astype(int)):
    a = (np.clip(img_c[zz], 0, 1) * 255).astype(np.uint8)
    Image.fromarray(a).resize((240, 240)).save(os.path.join(OUT, "slices", f"s{k}.png"))

# ---------------- flow glyphs on a real nucleus
F = np.load(os.path.join(HERE, "export", "flow_mouse_embryo.npz"))
g, dP, cp, nid = F["gt"], F["dP"].astype(np.float32), F["cp"].astype(np.float32), int(F["nucleus"])
zz, yy, xx = np.nonzero(g == nid)
cz, cy, cx = int(zz.mean()), int(yy.mean()), int(xx.mean())
ry = max(int((yy.max() - yy.min()) * 0.75), 8)
sub = (slice(None), slice(max(cy - ry, 0), cy + ry), slice(max(cx - ry, 0), cx + ry))
gs, dPs, cps = g[sub], dP[(slice(None),) + sub], cp[sub]
Zs, Ys, Xs = gs.shape
box = pv.ImageData(dimensions=(Xs, Ys, Zs), spacing=(1, 1, ratio))
box.point_data["n"] = (gs == nid).astype(np.float32).ravel(order="C")
nuc = box.contour([0.5], scalars="n").smooth_taubin(n_iter=40)
step = 5
from scipy import ndimage as ndi  # noqa: E402
near = ndi.binary_dilation(gs == nid, iterations=2)
Zi, Yi, Xi = np.meshgrid(np.arange(0, Zs, 2), np.arange(0, Ys, step), np.arange(0, Xs, step), indexing="ij")
inside = near[Zi, Yi, Xi]
pts = np.c_[Xi[inside], Yi[inside], Zi[inside] * ratio].astype(float)
vec = np.c_[dPs[2][Zi, Yi, Xi][inside], dPs[1][Zi, Yi, Xi][inside], dPs[0][Zi, Yi, Xi][inside]]
cloud = pv.PolyData(pts)
cloud["v"] = vec
cloud["axial"] = vec[:, 2]
glyph = cloud.glyph(orient="v", scale=False, factor=4.0, geom=pv.Arrow(tip_length=0.38, tip_radius=0.17, shaft_radius=0.07))
p = pv.Plotter(off_screen=True, window_size=(1000, 1000), lighting="none")
p.set_background("white")
lights(p)
p.add_mesh(nuc, color="#dfe7f4", opacity=0.25, smooth_shading=True, specular=0.2)
lim = float(np.percentile(np.abs(vec[:, 2]), 90))
p.add_mesh(glyph, scalars="axial", cmap="PRGn", clim=(-lim, lim), show_scalar_bar=False, smooth_shading=True)
p.camera_position = [(Xs * 2.2, -Ys * 1.2, Zs * ratio * 2.4), (Xs / 2, Ys / 2, Zs * ratio / 2), (0, 0, 1)]
p.camera.zoom(1.3)
p.screenshot(os.path.join(OUT, "flow_glyphs.png"), transparent_background=True)
p.close()

# ---------------- probability map (central plane of the crop, sigmoid of the foreground logit)
pr = 1 / (1 + np.exp(-F["cp"][F["cp"].shape[0] // 2].astype(np.float32)))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
from matplotlib import cm  # noqa: E402
rgb = (matplotlib.colormaps["magma"](pr)[..., :3] * 255).astype(np.uint8)
Image.fromarray(rgb).resize((320, 320), Image.BILINEAR).save(os.path.join(OUT, "prob_map.png"))
print("assets in", OUT, "nucleus", nid, "crop", gs.shape)

# ---------------- transparent cut-away input render for Fig. 1a (same construction as gallery F2-a, no background tile)
import rstyle as R  # noqa: E402
from scipy import ndimage as _ndi  # noqa: E402
G2, I2, RAT = V["gt"], V["img"].astype(np.float32) / 255.0, float(V["ratio"])
sc_ = 220
zc_ = int(np.argmax([(G2[k] > 0).sum() for k in range(G2.shape[0])])); nz_ = int(round(sc_ / RAT))
z0_ = int(np.clip(zc_ - nz_ // 2, 0, G2.shape[0] - nz_))
cy_, cx_ = [int(v) for v in np.argwhere(G2 > 0)[:, 1:].mean(0)]
y0_ = int(np.clip(cy_ - sc_ // 2, 0, G2.shape[1] - sc_)); x0_ = int(np.clip(cx_ - sc_ // 2, 0, G2.shape[2] - sc_))
ic = I2[z0_:z0_ + nz_, y0_:y0_ + sc_, x0_:x0_ + sc_]; s1c = V["S1"][z0_:z0_ + nz_, y0_:y0_ + sc_, x0_:x0_ + sc_]
lo_, hi_ = np.percentile(ic, (1, 99.8))
grid2 = pv.ImageData(dimensions=(sc_, sc_, nz_), spacing=(1, 1, RAT))
grid2.point_data["rgb"] = (np.repeat(np.clip((ic - lo_) / (hi_ - lo_), 0, 1)[..., None], 3, -1).reshape(-1, 3) * 255).astype(np.uint8)
hx_, hy_, hz_ = sc_ / 2, sc_ / 2, nz_ * RAT / 2
surf_ = grid2.clip_box((hx_, sc_, 0, hy_, hz_, nz_ * RAT), invert=True).extract_surface(algorithm="dataset_surface")
p = R.plotter((1600, 1600))
p.add_mesh(surf_, scalars="rgb", rgb=True, lighting=False)
u_ = np.unique(s1c)[1:]
cen_ = dict(zip(u_.tolist(), _ndi.center_of_mass(np.ones_like(s1c, np.uint8), s1c, u_)))
keep_ = [i for i, (a_, b_, c_) in cen_.items() if c_ >= hx_ and b_ < hy_ and a_ * RAT >= hz_]
ns_, bl_ = R.label_surface(s1c, RAT, keep_)
R.add_rgb_mesh(p, ns_, [pv.Color(R.QUAL[b % 10]).float_rgb for b in bl_])
p.add_mesh(grid2.outline(), color="#9a9a96", line_width=1.2)
R.ortho_camera(p, azim=-35, elev=28)
R.autocrop(R.shot(p)).save(os.path.join(OUT, "input_cutaway.png"))
print("input_cutaway.png")
