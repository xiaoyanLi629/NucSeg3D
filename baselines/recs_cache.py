"""test_r6.recs 的带缓存、并行版本：逐体积的 m(t)（规则 T 的判据，需要读 npz 里的 cp）只算一次，存到 results/eval/cache/<目录名>.json；
之后规则 T 与固定 cp−2 两种口径都从缓存派生，结果与 test_r6.recs 逐项相同（选阈值规则逐字照抄）。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import glob
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/five_seed'))
import test_r6 as T  # noqa: E402

CACHE = f"{_NSP_SHARED}/baselines/results/eval/cache"


def _m(args):
    f, thrs, npred, V = args
    cp = np.load(f)["cp"].astype(np.float32)
    return {f"{t:g}": float((cp > t).sum()) / max(npred[f"{t:g}"], 1) / V for t in thrs}


def _build(d):
    rows = json.load(open(d + "/rows.json"))
    by = {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in rows}
    thrs = sorted({x["thr"] for x in rows}, reverse=True)
    jobs, meta = [], []
    for f in sorted(glob.glob(d + "/masks/*.npz")):
        ds, vid = os.path.basename(f)[:-4].split("__", 1)
        if ds in T.DROP:
            continue
        r0 = by[(ds, vid, -2.0, 0.0)]
        dd = r0.get("d_used", r0["d_gt"]); V = math.pi / 6 * dd ** 3 / r0["ratio"]
        jobs.append((f, thrs, {f"{t:g}": by[(ds, vid, t, 0.0)]["n_pred"] for t in thrs}, V))
        meta.append((ds, vid, f))
    with Pool(16) as p:
        ms = p.map(_m, jobs, chunksize=1)
    out = {}
    for (ds, vid, f), m in zip(meta, ms):
        out[f"{ds}__{vid}"] = dict(m=m, npz=f, rows={f"{t:g}": {k: by[(ds, vid, t, 0.15)][k] for k in ("tp5", "tp7", "n_gt", "n_pred")} for t in thrs})
    return dict(thrs=thrs, vols=out)


def crecs(d, fixed=False):
    os.makedirs(CACHE, exist_ok=True)
    cf = f"{CACHE}/{os.path.basename(d.rstrip('/'))}.json"
    if os.path.exists(cf):
        C = json.load(open(cf))
    else:
        C = _build(d); json.dump(C, open(cf + ".part", "w")); os.replace(cf + ".part", cf)
    thrs = C["thrs"]; out = {}
    for k, v in C["vols"].items():
        ds, vid = k.split("__", 1); m = {float(t): x for t, x in v["m"].items()}
        t2 = -2.0 if (fixed or m[-2.0] >= 0.7) else next((t for t in thrs if t < -2 and m[t] >= 0.85), min(thrs))
        r = v["rows"][f"{t2:g}"]
        out[(ds, vid)] = dict(t=t2, m2=m[-2.0], tp5=r["tp5"], tp7=r["tp7"], n=r["n_gt"] + r["n_pred"], n_gt=r["n_gt"], n_pred=r["n_pred"], npz=v["npz"])
    return out
