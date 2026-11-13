"""nc-viz V3 revision 3 + base rate (descriptive; mouse_embryo test volumes, seed 0 final masks from errors/cache, gt protocol).

z-merge pair of model M: two GT nuclei g1, g2 both covered > 50 % by the same prediction of M, with the centroid offset mostly along z
(errors/common.is_z on um centroids) -- exactly the merge_z pairs counted by errors/common.analyse.
Outcome of the other model O on such a pair:
  separated : O matches both g1 and g2 at IoU >= 0.5 (per-GT best IoU >= 0.5; IoU > 0.5 matches are unique)
  merged    : some prediction of O covers > 50 % of both g1 and g2
  other     : neither (e.g. one missed, split or boundary error)
Revision-3 candidates: 2D z-merge pairs that S1 separates; pick the pair whose mean GT voxel count is closest to the dataset median
GT voxel count (4625.5, from viz/v3_median.json); ties by key then ids. Writes viz/v3_zpairs.json and viz/v3_candidates_rev3.json."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import os, sys, json, glob
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/errors'))
import common as C

runs = json.load(open(f"{_NSP_SCRATCH}/nc/errors/runs.json"))
med = json.load(open(f"{_NSP_SCRATCH}/nc/figs/viz/v3_median.json"))["median_gt_nvox_dataset"]


def stats(gt, pr, vox):
    gids, pids, gsz, psz, pg, pp, inter = C.overlaps(gt, pr)
    iou = inter / (gsz[pg] + psz[pp] - inter); cov = inter / gsz[pg]
    best = {}
    for g, v in zip(pg, iou):
        best[int(g)] = max(best.get(int(g), 0.0), float(v))
    big = {}
    for g, p, c in zip(pg, pp, cov):
        if c > 0.5:
            big.setdefault(int(p), []).append(int(g))
    return best, big


out = dict(per_volume={}, pairs_2D_zmerge=[], pairs_S1_zmerge=[])
for f in sorted(glob.glob(f"{runs['gt_S1_s0']}/masks/mouse_embryo__*.npz")):
    k = os.path.basename(f)[:-4]
    z = np.load(f); gt = z["gt"].astype(np.int32); vox = np.asarray(z["vox"], float)
    M = {m: np.load(f"{_NSP_SCRATCH}/nc/errors/cache/gt_{m}_s0/{k}.npz")["mask"] for m in ("S1", "2D")}
    ids = np.unique(gt); ids = ids[ids > 0]
    cen = dict(zip(ids.tolist(), (np.array(ndi.center_of_mass(np.ones_like(gt, np.uint8), gt, ids)).reshape(-1, 3) * vox).tolist()))
    cnt = np.bincount(gt.ravel())
    S = {m: stats(gt, M[m], vox) for m in M}
    for m, o in (("2D", "S1"), ("S1", "2D")):
        best_o, big_o = S[o]
        merged_o = {tuple(sorted((a, b))) for gs in big_o.values() for i, a in enumerate(gs) for b in gs[i + 1:]}
        for p, gs in S[m][1].items():
            for i in range(len(gs)):
                for j in range(i + 1, len(gs)):
                    a, b = sorted((gs[i], gs[j]))
                    if not C.is_z(np.array(cen[b]) - np.array(cen[a])):
                        continue
                    if best_o.get(a, 0) >= 0.5 and best_o.get(b, 0) >= 0.5:
                        oc = "separated"
                    elif (a, b) in merged_o:
                        oc = "merged"
                    else:
                        oc = "other"
                    out[f"pairs_{m}_zmerge"].append(dict(key=k, a=a, b=b, nvox_a=int(cnt[a]), nvox_b=int(cnt[b]), other_model=o, outcome=oc))
    print(k, flush=True)
for m in ("2D", "S1"):
    L = out[f"pairs_{m}_zmerge"]
    out[f"base_rate_{m}_zmerge"] = {oc: sum(1 for x in L if x["outcome"] == oc) for oc in ("separated", "merged", "other")} | {"total": len(L)}
cand = [x for x in out["pairs_2D_zmerge"] if x["outcome"] == "separated"]
for x in cand:
    x["dist_to_median"] = abs((x["nvox_a"] + x["nvox_b"]) / 2 - med)
cand.sort(key=lambda x: (x["dist_to_median"], x["key"], x["a"], x["b"]))
out["rev3_candidates"] = cand
out["label"] = "descriptive; seed 0 only; mouse_embryo test volumes only (14); final evaluated masks (errors/cache)"
json.dump(out, open(f"{_NSP_SCRATCH}/nc/figs/viz/v3_zpairs.json", "w"), indent=1)
# candidates file in the format viz_v3_export.py expects (target = the larger nucleus of the pair, cls2D merge)
c3 = [dict(key=x["key"], gt_id=(x["a"] if x["nvox_a"] >= x["nvox_b"] else x["b"]), partner=(x["b"] if x["nvox_a"] >= x["nvox_b"] else x["a"]),
           cls2D="merge", clsS1="pair separated", nvox=max(x["nvox_a"], x["nvox_b"]), dist_to_median=x["dist_to_median"]) for x in cand]
json.dump(c3, open(f"{_NSP_SCRATCH}/nc/figs/viz/v3_candidates_rev3.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k.startswith("base_rate")}, indent=1)); print(json.dumps(cand[:5], indent=1))
