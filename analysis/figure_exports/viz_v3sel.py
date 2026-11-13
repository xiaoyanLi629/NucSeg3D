"""nc-viz V3 selection, step 2 (pre-specified fallback in paper/figures/assets_viz/selection_rules.json).
GT nuclei of all mouse_embryo test volumes whose 2D-control class is split or merge and whose S1 class is hit
(errors/common.analyse on the seed-0 final masks in errors/cache), sorted by voxel count. Read-only; CPU."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import os, sys, json, glob
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/errors'))
import common as C
runs = json.load(open(f"{_NSP_SCRATCH}/nc/errors/runs.json"))
gtsrc = runs["gt_S1_s0"]
out = []
for f in sorted(glob.glob(f"{_NSP_SCRATCH}/nc/errors/cache/gt_2D_s0/mouse_embryo__*.npz")):
    k = os.path.basename(f)[:-4]
    z = np.load(f"{gtsrc}/masks/{k}.npz"); gt = z["gt"].astype(np.int32); vox = np.asarray(z["vox"], float)
    m2 = np.load(f)["mask"]; m1 = np.load(f"{_NSP_SCRATCH}/nc/errors/cache/gt_S1_s0/{k}.npz")["mask"]
    _, G1, _ = C.analyse(gt, m1, vox); _, G2, _ = C.analyse(gt, m2, vox)
    assert np.array_equal(G1["id"], G2["id"])
    cnt = np.bincount(gt.ravel())
    for i, g in enumerate(G1["id"]):
        c1, c2 = C.CLS[G1["cls"][i]], C.CLS[G2["cls"][i]]
        if c2 in ("split", "merge") and c1 == "hit":
            out.append(dict(key=k, gt_id=int(g), cls2D=c2, clsS1=c1, nvox=int(cnt[g]), zsplit2D=bool(G2["zsplit"][i]),
                            zmerge2D=bool(G2["zmerge"][i]), xymerge2D=bool(G2["xymerge"][i])))
    print(k, "done", flush=True)
out.sort(key=lambda r: -r["nvox"])
json.dump(out, open(f"{_NSP_SCRATCH}/nc/figs/viz/v3_candidates.json", "w"), indent=1)
print(json.dumps(out, indent=1))
