"""统计 eval_gt（提交 7d9c666）的 384 裁窗影响了哪些体积。只读数据，不改仓库。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'src'))
from eval_gt import pick_cases
from eval_cpsam import best_box
from stddata import load
SPLIT = f"{_NSP_SCRATCH}/nc/methods/src7d/val_split.json"
def one(r):
    _, lab = load(r); lab = np.asarray(lab).astype(np.int64)
    b = best_box(lab, 384)
    ids = np.unique(lab[lab > 0])
    if b is None:
        return dict(ds=r["ds"], id=r["id"], shape=list(lab.shape), crop=False, n_gt=len(ids))
    y, x = b; w = lab[:, y:y+384, x:x+384]
    iw = np.unique(w[w > 0])
    return dict(ds=r["ds"], id=r["id"], shape=list(lab.shape), crop=True, box=[y, x], n_gt=len(ids), n_in=len(iw),
                fg_frac=float((w > 0).sum() / max((lab > 0).sum(), 1)))
if __name__ == "__main__":
    sets = {"test_gt": pick_cases("test", SPLIT), "test_fp": pick_cases("test", SPLIT, k=4), "val": pick_cases("val", SPLIT)}
    allr = {(r["ds"], r["id"]): r for v in sets.values() for r in v}
    with Pool(4) as p:
        res = dict(zip(allr, p.map(one, list(allr.values()))))
    out = {}
    for k, v in sets.items():
        rows = [res[(r["ds"], r["id"])] for r in v]
        by = defaultdict(lambda: dict(n=0, cropped=0, n_gt=0, n_in=0))
        for q in rows:
            d = by[q["ds"]]; d["n"] += 1; d["n_gt"] += q["n_gt"]
            if q["crop"]: d["cropped"] += 1; d["n_in"] += q["n_in"]
            else: d["n_in"] += q["n_gt"]
        out[k] = dict(n=len(rows), cropped=sum(q["crop"] for q in rows), per_ds=by, rows=rows)
        print(k, len(rows), "cropped", out[k]["cropped"])
        for ds, d in sorted(by.items()):
            if d["cropped"]:
                cr = [q for q in rows if q["ds"] == ds and q["crop"]]
                print(f"  {ds}: {d['cropped']}/{d['n']} cropped; GT nuclei kept in cropped vols {sum(q['n_in'] for q in cr)}/{sum(q['n_gt'] for q in cr)}"
                      f" ({sum(q['n_in'] for q in cr)/sum(q['n_gt'] for q in cr):.3f}); fg voxel frac median {np.median([q['fg_frac'] for q in cr]):.3f}; shapes {sorted(set(tuple(q['shape'][1:]) for q in cr))[:4]}")
    json.dump(out, open(f"{_NSP_SCRATCH}/nc/methods/crop_count.json", "w"), indent=1)
