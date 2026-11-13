import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import os, json, glob
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
runs = json.load(open(f"{_NSP_SCRATCH}/nc/errors/runs.json"))
sizes = []
for f in sorted(glob.glob(f"{runs['gt_S1_s0']}/masks/mouse_embryo__*.npz")):
    g = np.load(f)["gt"]; c = np.bincount(g.ravel())[1:]; sizes += list(c[c > 0])
med = float(np.median(sizes))
cand = json.load(open(f"{_NSP_SCRATCH}/nc/figs/viz/v3_candidates.json"))
for c in cand: c["dist_to_median"] = abs(c["nvox"] - med)
a = min(cand, key=lambda c: (c["dist_to_median"], c["key"], c["gt_id"]))
zs = [c for c in cand if c["zsplit2D"]]; zm = [c for c in cand if c["zmerge2D"]]
b = min(zm, key=lambda c: (c["dist_to_median"], c["key"], c["gt_id"])) if zm else None
out = dict(median_gt_nvox_dataset=med, n_gt=len(sizes), n_candidates=len(cand), n_zsplit=len(zs), n_zmerge=len(zm),
           zmerge_candidates=zm, median_pick=a, median_zmerge_pick=b)
json.dump(out, open(f"{_NSP_SCRATCH}/nc/figs/viz/v3_median.json", "w"), indent=1)
print(json.dumps(out, indent=1))
