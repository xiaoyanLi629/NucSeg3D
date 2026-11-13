"""Render the 3D heat-diffusion target panel from assets3/diffusion.npz (pyvista; run with render3d/.venv python).
Three images with the same camera: diff_source.png (nucleus 62 surface + heat source at the label centre),
diff_heat.png (heat T after 160 iterations of the flows3d_gpu update, shown as nested iso-surfaces at 0.3, 3 and 30 % of its maximum),
diff_flow.png (streamlines of the final unit target flow mu inside nucleus 62, coloured by u_z with the palette ramp)."""
import json
import os

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap

pv.OFF_SCREEN = True
HERE = os.path.dirname(os.path.abspath(__file__))
PJ = json.load(open(os.path.join(HERE, "..", "palette.json")))
M = PJ["models"]
D = np.load(os.path.join(HERE, "assets3", "diffusion.npz"))
L, mu, k = D["L"], D["mu"], int(D["k62"])
cz, cy, cx = (int(v) for v in D["centre"])
Z, Y, X = L.shape


def grid(arr):
    g = pv.ImageData(dimensions=(X, Y, Z), spacing=(1, 1, 1))
    g.point_data["v"] = np.asarray(arr, np.float32).ravel(order="C")
    return g


m = (L == k).astype(np.float32)
surf = grid(m).contour([0.5], scalars="v").smooth_taubin(n_iter=30)
others = []
for j in np.unique(L):
    if j in (0, k):
        continue
    if (L == j).sum() > 400:
        others.append(grid((L == j).astype(np.float32)).contour([0.5], scalars="v").smooth_taubin(n_iter=30))


def plotter():
    p = pv.Plotter(off_screen=True, window_size=(900, 900), lighting="none")
    p.set_background("white")
    p.add_light(pv.Light(light_type="headlight", intensity=0.45))
    p.add_light(pv.Light(position=(-1, -1, 2), focal_point=(0, 0, 0), intensity=0.55, light_type="scene light"))
    return p


def cam(p):
    p.camera_position = [(X * 2.3, -Y * 1.5, Z * 2.6), (cx, cy, cz), (0, 0, 1)]
    p.camera.parallel_projection = True
    p.camera.parallel_scale = 0.62 * max(X, Y, Z) * 0.62


INST = PJ["qualitative_instances"]
# ---- 1 source
p = plotter()
p.add_mesh(surf, color=INST[0], opacity=0.35, smooth_shading=True, specular=0.2)
p.add_mesh(pv.Sphere(radius=1.6, center=(cx, cy, cz)), color=M["S1"], smooth_shading=True)
cam(p); p.screenshot(os.path.join(HERE, "assets3", "diff_source.png"), transparent_background=True); p.close()
# ---- 2 heat after 40 iterations, orthogonal slices
HEAT = LinearSegmentedColormap.from_list("heat", ["#ffffff", M["cellos"], M["S1"], M["S1_dark"]])
T = D["T160"]
gT = grid(T / T.max())
p = plotter()
p.add_mesh(surf, color=INST[0], opacity=0.14, smooth_shading=True)
for lev, op in ((0.003, 0.22), (0.03, 0.35), (0.3, 0.9)):
    iso = gT.contour([lev], scalars="v")
    if iso.n_points:
        p.add_mesh(iso.smooth_taubin(n_iter=20), color=HEAT(0.25 + 0.75 * (np.log10(lev) + 2.6) / 2.2), opacity=op, smooth_shading=True)
p.add_mesh(pv.Sphere(radius=1.0, center=(cx, cy, cz)), color=M["S1_dark"])
cam(p); p.screenshot(os.path.join(HERE, "assets3", "diff_heat.png"), transparent_background=True); p.close()
# ---- 3 target flow streamlines inside nucleus 62
g = pv.ImageData(dimensions=(X, Y, Z), spacing=(1, 1, 1))
vec = np.stack([mu[2], mu[1], mu[0]], -1) * m[..., None]           # (x, y, z) components, zero outside nucleus 62
g.point_data["u"] = vec.reshape(-1, 3, order="C") if False else np.stack([vec[..., 0].ravel(order="C"), vec[..., 1].ravel(order="C"), vec[..., 2].ravel(order="C")], 1)
g.point_data["uz"] = (mu[0] * m).ravel(order="C")
seeds_idx = np.argwhere(m > 0)
rng = np.random.default_rng(0)
pick = seeds_idx[rng.choice(len(seeds_idx), 220, replace=False)]
src = pv.PolyData(np.c_[pick[:, 2], pick[:, 1], pick[:, 0]].astype(float))
st = g.streamlines_from_source(src, vectors="u", integration_direction="forward", max_steps=400, initial_step_length=0.3,
                               terminal_speed=1e-3, max_length=200)
DIV = LinearSegmentedColormap.from_list("div", PJ["diverging"])
p = plotter()
p.add_mesh(surf, color=INST[0], opacity=0.16, smooth_shading=True)
p.add_mesh(st.tube(radius=0.12), scalars="uz", cmap=DIV, clim=(-1, 1), show_scalar_bar=False, smooth_shading=True)
p.add_mesh(pv.Sphere(radius=1.0, center=(cx, cy, cz)), color=PJ["text"])
cam(p); p.screenshot(os.path.join(HERE, "assets3", "diff_flow.png"), transparent_background=True); p.close()
print("ok", st.n_points)
