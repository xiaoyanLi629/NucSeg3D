"""F5 illustration: CPU re-inference on one crop at k = 1, 2, 4, 8 (rule: F5_RULE.md). Report-only; no metrics written.
Run on hostA from the eval worktree src (for eval_gt / infer3d imports):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 nice -n 19 python f5_infer.py
Writes $NUCSEG3D_SCRATCH/nc/figs/render3d/export/f5_<model>_k<k>.npz (img, gt, pred) and f5_log.json.
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import glob
import json
import os
import sys
import time

import numpy as np

SRC = _nsp_os.path.join(_NSP_REPO, 'src')
sys.path.insert(0, SRC)
os.chdir(SRC)
import torch  # noqa: E402

torch.set_num_threads(8)
import eval_gt as E  # noqa: E402
from eval_cpsam import load_case  # noqa: E402
from eval_val import zinterp  # noqa: E402
from stddata import load_manifest  # noqa: E402

KEY = ("celegans", "pha4I2L_0408072")
O = f"{_NSP_SCRATCH}/nc/figs/render3d/export"
MD = f"{_NSP_ROOT}/models/models"
MODELS = {"S1": (f"{MD}/orx_t3d_r5s1_s0", dict(views=3)), "2D": ("orx_v5_2d_s0", {})}
THR = {"S1": -2.0, "2D": -2.0}   # rule-T thresholds the G4 runs selected for this volume at k = 8 (f5_select.py); k < 8 checked below
SH = f"{_NSP_SHARED}/nc_gpu"
DMAP = {f"{x['ds']}__{x['id']}": x["d_used"] for x in json.load(open(f"{SH}/g4_diam/rows.json"))}
S = json.load(open(f"{_NSP_SCRATCH}/r6/sched_state.json"))
RES = f"{_NSP_ROOT}/results"


def g4_thr(model, k):
    key = f"G4_{model}_s0_z{k}"
    d = glob.glob(f"{RES}/orx_{S[key]['run']}*")[0] + "/noise_0"
    return {x["ds"] + "__" + x["id"]: x["t"] for x in json.load(open(f"{d}/ruleT.json"))}[f"{KEY[0]}__{KEY[1]}"]


r = [x for x in load_manifest(split="test", dim="3d") if (x["ds"], x["id"]) == KEY][0]
img0, gt0, box, ign = load_case(r, 384)
vox0 = tuple(r["voxel"] or (1.0, 1.0, 1.0))
# crop: 96 x 96 around the GT nucleus nearest the volume centre
from scipy import ndimage as ndi  # noqa: E402
ids = np.unique(gt0)[1:]
cen = np.array(ndi.center_of_mass(np.ones_like(gt0, np.uint8), gt0, ids))
c0 = (np.array(gt0.shape) - 1) / 2
i = int(np.argmin(np.linalg.norm((cen - c0) * [vox0[0] / vox0[1], 1, 1], axis=1)))
cy, cx = [int(round(v)) for v in cen[i, 1:]]
H = 48
y0 = int(np.clip(cy - H, 0, gt0.shape[1] - 2 * H)); x0 = int(np.clip(cx - H, 0, gt0.shape[2] - 2 * H))
img_c, gt_c = img0[:, y0:y0 + 2 * H, x0:x0 + 2 * H], gt0[:, y0:y0 + 2 * H, x0:x0 + 2 * H]
d = DMAP[f"{KEY[0]}__{KEY[1]}"]
log = {"key": KEY, "crop_yx": [y0, x0, 2 * H], "centre_nucleus": int(ids[i]), "d": d, "vox": vox0, "runs": {}}
for name, (spec, kw) in MODELS.items():
    M = E.load_model(spec, kw or None)
    for k in (1, 2, 4, 8):
        t0 = time.time()
        img, gt = img_c[::k], gt_c[::k]
        vox = (vox0[0] * k, vox0[1], vox0[2]); ratio = vox[0] / vox[1]
        if ratio >= 2:
            Zu = int(round(img.shape[0] * ratio)); zi = np.round(np.linspace(0, Zu - 1, img.shape[0])).astype(int)
            im = img.astype(np.float32); lo, hi = np.percentile(im, [1, 99.8]); im = (im - lo) / max(hi - lo, 1e-6)
            img_in, an = zinterp(im, Zu), 1.0
        else:
            img_in, an, zi = img, ratio, None
        out = M.eval(img_in, do_3D=True, z_axis=0, diameter=d, anisotropy=an, flow3D_smooth=1.0, batch_size=8, compute_masks=False)
        fl = out[1] if isinstance(out[1], (list, tuple)) else out
        dP, cp = fl[1], fl[2]
        t = float(g4_thr(name, k))
        m = M._compute_masks((*img_in.shape, 1), dP, cp, flow_threshold=0.4, cellprob_threshold=t, min_size=15,
                             max_size_fraction=0.4, niter=int(200 / (30.0 / d)), do_3D=True).squeeze()
        m = (m if zi is None else m[zi]).astype(np.int32)
        ev = np.pi / 6 * d ** 3 * vox[1] / vox[0]
        cnt = np.bincount(m.ravel()); small = np.where(cnt < 0.15 * ev)[0]; small = small[small > 0]
        m[np.isin(m, small)] = 0
        np.savez_compressed(f"{O}/f5_{name}_k{k}.npz", img=img, gt=gt, pred=m, ratio=ratio, thr=t)
        log["runs"][f"{name}_k{k}"] = dict(thr=t, sec=round(time.time() - t0, 1), shape=list(gt.shape))
        print(name, k, "thr", t, "sec", round(time.time() - t0, 1), flush=True)
        json.dump(log, open(f"{O}/f5_log.json", "w"), indent=1, default=str)
    del M
