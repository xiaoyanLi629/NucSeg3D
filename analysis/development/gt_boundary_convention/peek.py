"""Exploratory peek (read-only): intensity percentiles, GT extents, and raw centroid profiles for a few nuclei."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import sys

import numpy as np
import tifffile
from scipy import ndimage as ndi

STD = f"{_NSP_ROOT}/data/std"
ds, vid = sys.argv[1], sys.argv[2]
img = tifffile.imread(f"{STD}/{ds}/images/{vid}.tif").astype(np.float32)
lab = tifffile.imread(f"{STD}/{ds}/masks/{vid}.tif")
print(ds, vid, img.shape, "labels", int(lab.max()), "fg frac %.3f" % (lab > 0).mean())
p = np.percentile(img, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.8, 99.99])
print("pct", np.round(p, 1))
lo, hi = np.percentile(img, [1, 99.8])
n = (img - lo) / (hi - lo)
print("fg median %.3f  bg median %.3f  bg p95 %.3f" % (np.median(n[lab > 0]), np.median(n[lab == 0]), np.percentile(n[lab == 0], 95)))
objs = ndi.find_objects(lab)
ext = np.array([[s.stop - s.start for s in sl] for sl in objs if sl is not None])
print("bbox extent median (z,y,x)", np.median(ext, 0), "IQR z", np.percentile(ext[:, 0], [25, 75]), "IQR x", np.percentile(ext[:, 2], [25, 75]))
rng = np.random.default_rng(0)
ids = [i + 1 for i, sl in enumerate(objs) if sl is not None]
for i in rng.choice(ids, size=min(4, len(ids)), replace=False):
    sl = objs[i - 1]
    m = lab[sl] == i
    cz, cy, cx = [int(round(np.mean(a))) + s.start for a, s in zip(np.nonzero(m), sl)]
    ez, ey, ex = [s.stop - s.start for s in sl]
    z0, z1 = max(0, cz - ez), min(img.shape[0], cz + ez + 1)
    x0, x1 = max(0, cx - 2 * ex), min(img.shape[2], cx + 2 * ex + 1)
    zp = n[z0:z1, cy, cx]; zl = lab[z0:z1, cy, cx]
    xp = n[cz, cy, x0:x1]; xl = lab[cz, cy, x0:x1]
    print(f"-- nucleus {i}: bbox ext z{ez} y{ey} x{ex} vol {m.sum()} p95in {np.percentile(n[sl][m], 95):.2f}")
    print("  z:", " ".join(f"{v:.2f}{'*' if l == i else ('o' if l else ' ')}" for v, l in zip(zp, zl)))
    print("  x:", " ".join(f"{v:.2f}{'*' if l == i else ('o' if l else ' ')}" for v, l in zip(xp, xl)))
