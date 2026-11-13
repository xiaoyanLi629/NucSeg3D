"""F2 v2: semi-transparent ortho-slicer cube (gallery F2-d..g). Mouse-embryo Fig. 2d volume, 220-px crop around the GT-foreground
centroid. The three visible faces (raw image, linear 1-99.8 % LUT) are drawn at ~50 % opacity so the S1 nucleus meshes inside
(one colour per nucleus) show through; variants with and without a cut-away octant, light and dark."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

import figcheck  # noqa: E402
import rstyle as R  # noqa: E402

for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6.5})
HERE = os.path.dirname(os.path.abspath(__file__))
GA = os.path.abspath(os.path.join(HERE, "..", "..", "figures", "gallery", "assets"))
Z = np.load(os.path.join(HERE, "export", "mouse_embryo.npz"))
GT, IMG, RAT = Z["gt"], Z["img"].astype(np.float32) / 255.0, float(Z["ratio"])
s = 220
zc = int(np.argmax([(GT[k] > 0).sum() for k in range(GT.shape[0])])); nz = int(round(s / RAT))
z0 = int(np.clip(zc - nz // 2, 0, GT.shape[0] - nz))
cy, cx = [int(v) for v in np.argwhere(GT > 0)[:, 1:].mean(0)]
y0 = int(np.clip(cy - s // 2, 0, GT.shape[1] - s)); x0 = int(np.clip(cx - s // 2, 0, GT.shape[2] - s))
sl = (slice(z0, z0 + nz), slice(y0, y0 + s), slice(x0, x0 + s))
ic, sc = IMG[sl], Z["S1"][sl]
lo, hi = np.percentile(ic, (1, 99.8)); hi2 = np.percentile(ic, 99.0)
H = nz * RAT


def face(arr2d, pts, zrows=False):
    """zrows: rows of arr2d are native z planes -> linear interpolation along z to the physical aspect (no blocky texels)."""
    from scipy.ndimage import zoom
    q = pv.PolyData(np.array(pts, float), faces=[4, 0, 1, 2, 3])
    q.active_texture_coordinates = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    a = zoom(arr2d, (RAT, 1), order=1) if zrows else arr2d
    g = np.clip((a - lo) / (hi2 - lo), 0, 1) ** 0.8          # slightly higher contrast than the 1-99.8 % linear LUT
    rgb = np.repeat(g[..., None], 3, -1)
    t = pv.Texture((np.flipud(rgb) * 255).astype(np.uint8)); t.interpolate = True
    return q, t


def build(cid, cut, dark, alpha=0.5):
    p = R.plotter((2400, 2400), dark)
    # nuclei (all, or only those outside the removed octant plus those inside it, shown whole)
    u = np.unique(sc)[1:]
    ns, bl = R.label_surface(sc, RAT)
    R.add_rgb_mesh(p, ns, [pv.Color(R.QUAL[b % len(R.QUAL)]).float_rgb for b in bl])
    # three visible faces: top (z = H), front (y = 0), right (x = s); camera from (+x, -y, +z)
    faces = [(ic[-1], [(0, 0, H), (s, 0, H), (s, s, H), (0, s, H)]),            # top: rows = y, cols = x
             (ic[:, 0, :], [(0, 0, 0), (s, 0, 0), (s, 0, H), (0, 0, H)]),       # front y = 0: rows = z, cols = x
             (ic[:, :, -1], [(s, 0, 0), (s, s, 0), (s, s, H), (s, 0, H)])]      # right x = s: rows = z, cols = y
    if cut:
        # remove the octant x > s/2, y < s/2, z > H/2 from the faces by masking it (transparent) on each face
        pass
    box = pv.Box(bounds=(0, s, 0, s, 0, H))
    if not cut:
        for fi, (arr, pts) in enumerate(faces):
            q, t = face(arr.copy(), pts, zrows=fi > 0)
            p.add_mesh(q, texture=t, opacity=alpha, lighting=False)
    else:
        # cut-away: the volume surface with the octant (x > s/2, y < s/2, z > H/2) removed, all faces semi-transparent,
        # so the L-shaped inner walls of the cut and the nuclei behind them both show
        grid = pv.ImageData(dimensions=(s, s, nz), spacing=(1, 1, RAT))
        grid.point_data["rgb"] = (np.repeat(np.clip((ic - lo) / (hi - lo), 0, 1)[..., None], 3, -1).reshape(-1, 3) * 255).astype(np.uint8)
        surf = grid.clip_box((s / 2, s, 0, s / 2, H / 2, H), invert=True).extract_surface(algorithm="dataset_surface")
        p.add_mesh(surf, scalars="rgb", rgb=True, lighting=False, opacity=alpha)
    p.add_mesh(box.extract_feature_edges(), color="#c8c8c4" if dark else "#7a7a76", line_width=2)
    ctr = np.array([s / 2, s / 2, H / 2]); rad = np.linalg.norm(ctr)
    p.camera_position = [tuple(ctr + rad * np.array([1.6, -2.0, 1.6])), tuple(ctr), (0, 0, 1)]
    p.enable_parallel_projection(); p.reset_camera(); p.camera.zoom(1.05)
    im = R.autocrop(R.shot(p, ssao=False))
    bg = R.DARK_BG[0] if dark else "white"; tc = "#e8e8ea" if dark else R.FG
    fig = plt.figure(figsize=(89 / 25.4, 89 / 25.4), dpi=700, facecolor=bg)
    ax = fig.add_axes([0, 0.07, 1, 0.93]); ax.imshow(R.tile(im, dark)); ax.axis("off")
    fig.text(0.5, 0.025, "raw planes on the cube faces at %d %% opacity; S1 nuclei inside (one colour each)" % int(alpha * 100),
             ha="center", fontsize=5.5, color=tc)
    fig.savefig(os.path.join(GA, f"{cid}.png"), facecolor=bg)
    rep = figcheck.check(fig)
    json.dump({"desc": "semi-transparent cube", "cut_away": cut, "dark": dark, "face_opacity": alpha, "crop_zyx": [z0, y0, x0, nz, s, s],
               "LUT": "1-99 percentile of the crop, gamma 0.8 (whole-image adjustment); side faces linearly interpolated along z to physical aspect", "overlap_issues": rep}, open(os.path.join(GA, f"{cid}.provenance.json"), "w"), indent=1)
    plt.close(fig); print("saved", cid, rep)


build("F2-d", cut=False, dark=False, alpha=0.6)
build("F2-e", cut=False, dark=True)
build("F2-f", cut=True, dark=False, alpha=0.6)
build("F2-g", cut=True, dark=True)
