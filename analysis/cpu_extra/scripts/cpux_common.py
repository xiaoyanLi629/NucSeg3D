"""CPUX（设计 $NUCSEG3D_SCRATCH/nc/cpu_extra/DESIGN_CPUX.md，md5 27bd2326…，登记 b8932d9）公用读数。只读 inputs/ 与共享盘上的评测结果。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np  # noqa: E402

BASE = f"{_NSP_SHARED}/nc_cpu_extra"
INP = f"{BASE}/inputs"
OUT = f"{BASE}/out"
SRC = f"{INP}/src_k1"
RUNMAP = json.load(open(f"{INP}/runmap.json"))
SEEDS = RUNMAP["seeds"]
NPROC = 40                      # 登记要求 ≤ 48 个进程
B = 10000


def recs(d, fixed=False):
    b = os.path.basename(d)
    c = json.load(open(f"{INP}/recs/recs_{b}{'_fixed' if fixed else ''}.json"))
    return {tuple(k.split("||")): v for k, v in c["recs"].items()}


def models():
    D = RUNMAP["test_gt"]
    return {"S1": [recs(d) for d in D["S1"]], "2D": [recs(d) for d in D["2D"]], "base": [recs(D["base"][0], fixed=True)]}, D


def ro(npz, t):
    run = os.path.basename(os.path.dirname(os.path.dirname(npz)))
    return json.load(open(f"{INP}/ro/{run}/{os.path.basename(npz)[:-4]}__t{float(t):g}.json"))


def ro_gt(ds, vid):
    return json.load(open(f"{INP}/ro/gt/{ds}__{vid}.json"))


def boot_idx(dss_keys, seed=0):
    """套内分层体积自助索引：{ds: [B, n]}，所有模型种子共用。"""
    rng = np.random.default_rng(seed)
    return {ds: rng.integers(0, len(ks), (B, len(ks))) for ds, ks in dss_keys.items()}
