"""Select representative volumes for the orthogonal-view panels (rule: $NUCSEG3D_SCRATCH/nc/figs/ortho/RULE.md).

Writes $NUCSEG3D_SCRATCH/nc/figs/ortho/selection.json.
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import sys

import numpy as np

sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/five_seed'))
from test_r6 import recs  # noqa: E402

RES = f"{_NSP_ROOT}/results"
RUNS = {"S1": "07992b55-0982-4116-b065-b34859c9a80b", "2D": "81116913-06c5-47a3-9dcc-ab7d64615e68",
        "base": "b5973303-f479-41fe-be14-f80d7c2fbb1a"}
DIRS = {m: f"{RES}/orx_{r}_thr_eval" for m, r in RUNS.items()}
R = {m: recs(DIRS[m], fixed=(m == "base")) for m in RUNS}
keys = sorted(set.intersection(*[set(v) for v in R.values()]))
rows = {}
for d in DIRS.values():
    for x in json.load(open(d + "/rows.json")):
        rows.setdefault((x["ds"], x["id"]), x)
f1 = lambda m, k: 2 * R[m][k]["tp5"] / max(R[m][k]["n"], 1)  # noqa: E731
sel = {}
for ds in sorted({k[0] for k in keys}):
    ks = sorted(k for k in keys if k[0] == ds)
    d = np.array([f1("S1", k) - f1("2D", k) for k in ks])
    med = float(np.median(d))
    dist = np.abs(d - med)
    # tie-break amended 2026-11-07 (RULE.md addendum): among volumes equally close to the median, take the one with the
    # most ground-truth nuclei, then the smallest id
    i = min(range(len(ks)), key=lambda j: (round(float(dist[j]), 12), -R["S1"][ks[j]]["n_gt"], ks[j][1]))
    n_tied = int(sum(abs(float(dist[j]) - float(dist[i])) < 1e-12 for j in range(len(ks))))
    k = ks[i]
    sel[ds] = dict(id=k[1], n_vol=len(ks), median_diff=med, diff=float(d[i]), ratio=rows[k]["ratio"], shape=rows[k]["shape"],
                   f1={m: f1(m, k) for m in RUNS}, thr={m: R[m][k]["t"] for m in RUNS}, npz={m: R[m][k]["npz"] for m in RUNS},
                   n_pred_rec={m: R[m][k]["n_pred"] for m in RUNS}, n_gt=R["S1"][k]["n_gt"], n_tied=n_tied)
pool = sorted((ds for ds in sel if ds != "mesospim"), key=lambda ds: (sel[ds]["ratio"], ds))
pos = [int(p) for p in np.round(np.linspace(0, len(pool) - 1, 5))]
main = [pool[p] for p in pos]
out = {"rule": f"{_NSP_SCRATCH}/nc/figs/ortho/RULE.md", "runs": RUNS, "volumes": sel, "pool_sorted_by_ratio": pool,
       "positions": pos, "main": main}
json.dump(out, open(f"{_NSP_SCRATCH}/nc/figs/ortho/selection.json", "w"), indent=1)
for ds in pool + ["mesospim"]:
    v = sel[ds]
    print(f"{ds:13s} {v['id'][:40]:40s} n={v['n_vol']:3d} ratio={v['ratio']:.2f} med={v['median_diff']:+.3f} d={v['diff']:+.3f} "
          f"ngt={v['n_gt']} tied={v['n_tied']} F1 {v['f1']['S1']:.3f}/{v['f1']['2D']:.3f}/{v['f1']['base']:.3f} t={v['thr']}" + ("  MAIN" if ds in main else ""))
