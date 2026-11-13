"""README turntables: orbit frames of the Fig. 2e volumes with the paper's R8 render pipeline (viewer palette).

Meshes, materials, lights, error classification and colours are exactly those of r6_render.render_err_set as patched by
r8_render2 (palette_viewer.json); only the camera differs: it starts at the figure view (ortho_camera 35/25) and is
rotated about the view-up axis in N equal steps (as turntable.py does for the Supplementary Movie).
Frames are RGBA (transparent background), dehalo'ed exactly as r8_render2.on_pane, and saved per (dataset, model).
Usage (render3d environment, requirements/figures_render3d.txt): python docs/scripts/orbit.py <ds> <kinds comma> <N> <px>
"""
import os
import sys

_PAPER = os.environ.get("NUCSEG3D_PAPER", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper"))
R3 = os.path.join(_PAPER, "scripts", "render3d")
sys.path.insert(0, R3)
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from PIL import Image  # noqa: E402

import r8_render2 as P  # noqa: E402  (patches r6_render colours to the viewer palette)
import r6_render as V  # noqa: E402
import rstyle as R  # noqa: E402

OUT = os.path.join(os.environ.get("NUCSEG3D_FIG_SCRATCH", "."), "readme_frames")


def build(p, Z, k):
    gt, ratio = Z["gt"], float(Z["ratio"])
    if k == "gt":
        s, bl = R.label_surface(gt, ratio)
        V.mesh(p, s, [pv.Color(V.QUALV[b % len(V.QUALV)]).float_rgb for b in bl])
    else:
        lab = Z[k]
        s, bl = R.label_surface(lab, ratio)
        cls, missed, _ = R.classify_preds(gt, lab)
        kk = np.array([cls.get(int(b), "other") for b in bl])
        for c in ("match", "boundary", "split", "merge", "other"):
            sel = np.nonzero(kk == c)[0]
            if len(sel):
                sub = s.extract_cells(sel).extract_surface(algorithm="dataset_surface")
                V.mesh(p, sub, np.tile(pv.Color(V.ERRV[c]).float_rgb, (sub.n_cells, 1)), V.MATCH_OP if c == "match" else 1.0)
        if missed:
            gs, _ = R.label_surface(gt, ratio, missed)
            p.add_mesh(gs, color=V.ERRV["miss"], opacity=V.GHOST_OP, smooth_shading=True, **V.MAT)


def main(ds, kinds, N, px):
    Z = np.load(os.path.join(R3, "export", f"{ds}.npz"))
    gt, ratio = Z["gt"], float(Z["ratio"])
    shell = R.shell_surface(gt > 0, ratio, 0.6 * float(Z["d"]))
    for k in kinds:
        od = os.path.join(OUT, f"{ds}_{k}"); os.makedirs(od, exist_ok=True)
        p = V.plotter(px)
        build(p, Z, k)
        p.add_mesh(shell, color=R.SHELL, opacity=V.SHELL_OP, smooth_shading=True)
        R.ortho_camera(p, 35, 25)
        b = np.array(shell.bounds).reshape(3, 2)
        rad = 0.5 * np.linalg.norm(b[:, 1] - b[:, 0])
        p.camera.parallel_scale = rad * 1.02
        p.enable_ssao(**R.RENDER["ssao"])
        for i in range(N):
            p.render()
            im = p.screenshot(transparent_background=True, return_img=True)
            P.dehalo(Image.fromarray(im)).save(os.path.join(od, f"{i:03d}.png"))
            p.camera.Azimuth(360.0 / N)
        p.close()
        print(ds, k, "done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]), int(sys.argv[4]))
