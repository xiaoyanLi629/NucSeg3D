"""Data for the 3D heat-diffusion panel (Fig. 1 v4): GT nucleus 62 of mouse_embryo E15_Ab2 (+ touching neighbours) on the
training grid (isotropic, median diameter 30 px; labels nearest-exact as data3d.to_patch), the final 3D target from the project's own
data3d.flows3d_gpu (S1 snapshot, CPU), and snapshots of the heat field T of nucleus 62 during the same diffusion loop
(identical update: T[centre] += 1; T = mean of T over the 6-neighbourhood + self restricted to the same label; niter = 6 x sum of
bounding-box extents, data3d.py:58-74). Writes assets3/diffusion.npz. Run with the system python (torch)."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SRC = _nsp_os.environ.get('NUCSEG3D_SRC', _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", "..", "src")))
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, "..", ".."))
SNAP = f"{_NSP_SRC}"
sys.path.insert(0, SNAP)
from data3d import flows3d_gpu  # noqa: E402

E = np.load(os.path.join(PAPER, "scripts", "render3d", "export", "mouse_embryo.npz"))
gt, vox, d = E["gt"], E["vox"], float(E["d"])
ratio = float(vox[0] / vox[1]); s = 30.0 / d
nid = 62
zz, yy, xx = np.nonzero(gt == nid)
cz, cy, cx = int(zz.mean()), int(yy.mean()), int(xx.mean())
hz, h = 6, 34
sl = (slice(max(cz - hz, 0), cz + hz + 1), slice(cy - h, cy + h), slice(cx - h, cx + h))
lab = gt[sl]
u, inv = np.unique(lab, return_inverse=True)
shape_iso = (int(round(lab.shape[0] * ratio * s)), int(round(lab.shape[1] * s)), int(round(lab.shape[2] * s)))
lt = F.interpolate(torch.from_numpy(inv.reshape(lab.shape).astype(np.float32))[None, None], size=shape_iso, mode="nearest-exact")[0, 0].round().long()
mu = flows3d_gpu(lt).numpy()
L = lt.numpy()
k62 = int(np.searchsorted(u, nid))
# ---- heat snapshots for nucleus 62, same update as flows3d_gpu (single label, float64)
m = L == k62
Z, Y, X = m.shape
co = np.argwhere(m).astype(np.float64)
mean = co.mean(0)
d2 = ((co - mean) ** 2).sum(1)
cen = tuple(np.argwhere(m)[np.argmin(d2)])
lo, hi = np.argwhere(m).min(0), np.argwhere(m).max(0)
niter = int(min(6 * int(((hi - lo) + 2).sum()), 4000))
mp = np.pad(m, 1)
T = np.zeros(mp.shape)
c1 = tuple(np.array(cen) + 1)
snaps, keep = {}, sorted({10, 40, 160, niter})
for it in range(1, niter + 1):
    T[c1] += 1
    acc = T.copy()
    cnt = np.ones_like(T)
    for ax in range(3):
        for sh in (1, -1):
            acc += np.roll(T, sh, ax)
    T = np.where(mp, acc / 7.0, 0.0)          # neighbours outside the label contribute 0 (isn = 0), as in flows3d_gpu
    if it in keep:
        snaps[it] = T[1:-1, 1:-1, 1:-1].astype(np.float32).copy()
np.savez_compressed(os.path.join(HERE, "assets3", "diffusion.npz"), L=L.astype(np.int16), mu=mu.astype(np.float32), k62=k62,
                    centre=np.array(cen), niter=niter, **{f"T{k}": v for k, v in snaps.items()}, keep=np.array(keep))
json.dump(dict(nucleus=nid, crop_native=[[a.start, a.stop] for a in sl], iso_shape=shape_iso, niter=niter, snapshots=keep,
               centre_iso=[int(c) for c in cen], n_labels_in_crop=int(len(u) - 1)),
          open(os.path.join(HERE, "assets3", "diffusion.json"), "w"), indent=1)
print("iso", shape_iso, "niter", niter, "labels", len(u) - 1)
