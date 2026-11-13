"""Extract a small real example of S1 outputs (flows dP, foreground logit cp) around nuclei of the mouse-embryo
showcase volume for the Fig. 1 schematic (server, CPU, read-only). Writes export/flow_mouse_embryo.npz."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import numpy as np
from scipy import ndimage as ndi
SEL = json.load(open(f"{_NSP_SCRATCH}/nc/figs/ortho/selection.json"))["volumes"]["mouse_embryo"]
z = np.load(SEL["npz"]["S1"])
gt, dP, cp, vox = z["gt"], z["dP"], z["cp"], z["vox"]
ids, cnt = np.unique(gt[gt > 0], return_counts=True)
cen = np.array(ndi.center_of_mass(np.ones_like(gt, np.uint8), gt, ids))
# nucleus closest to the volume centre among the larger half (deterministic, not chosen by result)
big = cnt >= np.median(cnt)
c0 = np.array(gt.shape) / 2
i = np.argmin(np.where(big, np.linalg.norm((cen - c0) * [vox[0] / vox[1], 1, 1], axis=1), np.inf))
cz, cy, cx = np.round(cen[i]).astype(int)
hz, h = 10, 40
sl = (slice(max(cz - hz, 0), cz + hz), slice(max(cy - h, 0), cy + h), slice(max(cx - h, 0), cx + h))
np.savez_compressed(f"{_NSP_SCRATCH}/nc/figs/render3d/export/flow_mouse_embryo.npz", gt=gt[sl], dP=dP[(slice(None),) + sl].astype(np.float16),
                    cp=cp[sl].astype(np.float16), cp_mid=cp[cz].astype(np.float16), gt_mid=gt[cz], vox=vox, centre=[cz, cy, cx],
                    nucleus=int(ids[i]), box=[[s.start, s.stop] for s in sl], npz=SEL["npz"]["S1"])
print("nucleus", ids[i], "centre", cz, cy, cx, "dP shape", dP.shape)
