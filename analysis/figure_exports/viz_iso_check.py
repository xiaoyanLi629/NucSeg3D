"""Check (team lead): for the G6 'off' volumes with rule-T threshold != -2, do the masks recomputed on the native grid by viz_iso_ext.py
(off_mask) reproduce G6's recorded per-volume counts (rows.json at thr t, frac 0.15: tp5, tp7, n_pred, n_gt)? Uses errors/common.analyse
(same rules as eval_gt.taxonomy). Writes $NUCSEG3D_SCRATCH/nc/figs/viz/iso_off/iso_check.json."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/figure_exports')); sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/errors'))
import viz_iso_ext as X  # noqa: E402
import common as C  # noqa: E402

rows = json.load(open(f"{X.OFF}/rows.json"))
out = {}
for key, t in X.RT.items():
    if t == -2.0:
        continue
    z = np.load(f"{X.OFF}/masks/{key}.npz"); gt = z["gt"].astype(np.int32); vox = np.asarray(z["vox"], float)
    m, _ = X.off_mask(key, z)
    res, _, _ = C.analyse(gt, m, vox)
    ds, vid = key.split("__", 1)
    ref = [r for r in rows if r["ds"] == ds and r["id"] == vid and r["thr"] == t and abs(r["frac"] - 0.15) < 1e-9][0]
    f = ("tp5", "tp7", "n_pred", "n_gt")
    out[key] = dict(t=t, recomputed={k: int(res[k]) for k in f}, g6={k: int(ref[k]) for k in f},
                    match=all(int(res[k]) == int(ref[k]) for k in f))
    print(key, out[key], flush=True)
json.dump(out, open(f"{_NSP_SCRATCH}/nc/figs/viz/iso_off/iso_check.json", "w"), indent=1)
