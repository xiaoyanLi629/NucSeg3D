"""NC 统计分析的公共读数（只读复用 $NUCSEG3D_SCRATCH/r6/test_r6.py 与 s5_stats.py，不修改它们）。
逐体积记录 = test_r6.recs()：规则 T（R2）逐体积选阈值，体积下限 0.15；原版用固定 cp−2（--base_fixed）。
体积级 F1 = 2·tp / (n_gt + n_pred)，与 test_r6.score 的 fp 口径相同；gt 协议的宏平均在 test_r6 里是每套 pooled，这里两种都能算。
recs() 要读每个 npz 的 cp，较慢，所以结果按 (目录, fixed, rows.json 修改时间) 缓存成 JSON。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import os
import sys

import numpy as np

sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/five_seed'))
from test_r6 import recs, score  # noqa: E402
import s5_stats  # noqa: E402

HERE = f"{_NSP_SCRATCH}/nc/stats"
CACHE = f"{HERE}/cache"
SUBSETS = {"all": None, "C1": f"{_NSP_SCRATCH}/r9/clean_test.json", "C2": f"{_NSP_SCRATCH}/r9/clean_test_plus_b.json"}
MODES = ("gt", "fp")
METRICS = (("tp5", "F1@0.5"), ("tp7", "F1@0.7"))


def subset_keys(name):
    p = SUBSETS[name]
    if p is None:
        return None
    k = json.load(open(p, encoding="utf-8"))
    k = (k["keep"] if "keep" in k else k["clean"]) if isinstance(k, dict) else k
    return {tuple(x) for x in k}


def cached_recs(d, fixed=False):
    os.makedirs(CACHE, exist_ok=True)
    tag = f"{CACHE}/recs_{os.path.basename(d)}{'_fixed' if fixed else ''}.json"
    mt = os.path.getmtime(f"{d}/rows.json")
    if os.path.exists(tag):
        c = json.load(open(tag))
        if c["mtime"] == mt:
            return {tuple(k.split("||")): v for k, v in c["recs"].items()}
    R = recs(d, fixed=fixed)
    json.dump({"dir": d, "fixed": fixed, "mtime": mt, "recs": {"||".join(k): v for k, v in R.items()}}, open(tag, "w"))
    return R


def load_mode(mode, seeds):
    """返回 M = {"S1": [每种子 recs], "2D": [...], "base": [recs]} 和目录表；口径与 s5_stats.py --base_fixed 相同。"""
    D = s5_stats.dirs("test", mode, seeds)
    M = {"S1": [cached_recs(d) for d in D["S1"]], "2D": [cached_recs(d) for d in D["2D"]],
         "base": [cached_recs(D["base"][0], fixed=True)]}
    return M, D


def common_keys(M, sub=None):
    common = set.intersection(*[set(R) for L in M.values() for R in L])
    if sub is not None:
        common &= sub
    return sorted(common)


def vol_f1(r, k):
    return 2 * r[k] / max(r["n"], 1)


def volume_table(M, keys):
    """逐体积：每个模型的种子平均体积级 F1，以及逐种子值。"""
    out = []
    for x in keys:
        e = {"ds": x[0], "id": x[1]}
        for k, _ in METRICS:
            for nm in ("S1", "2D", "base"):
                vals = [vol_f1(R[x], k) for R in M[nm]]
                e[f"{nm}_{k}"] = float(np.mean(vals))
                e[f"{nm}_{k}_seeds"] = vals
        e["t"] = {nm: [R[x]["t"] for R in M[nm]] for nm in M}
        out.append(e)
    return out


def macro_seeds(M, keys, mode, k, keep=lambda ds: True):
    """与 s5_stats.py 相同的逐种子宏平均（gt：每套 pooled；fp：每套逐体积平均）。"""
    dss = sorted({ds for ds, _ in keys if keep(ds)})
    return {nm: [float(np.mean([score(R, [x for x in keys if x[0] == ds], mode, k) for ds in dss])) for R in L]
            for nm, L in M.items()}, dss
