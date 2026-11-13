"""第三轮 CPU 分析的公共读数（只读复用 ncio / nc_stats / compare_test）。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/stats'))
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'baselines'))
import numpy as np  # noqa: E402
import ncio  # noqa: E402
import nc_stats  # noqa: E402

SEEDS = ["s0", "s1000", "s2000", "s3000", "s4000"]
R3 = f"{_NSP_SCRATCH}/nc/stats/round3"
CAT = {tuple(x["key"]): x["cat"] for x in json.load(open(f"{_NSP_SCRATCH}/r9/clean_test.json", encoding="utf-8"))["all_227"]}


def vf1(r, k):
    return 2 * r[k] / max(r["n"], 1)


def nnunet(mode):
    from compare_test import BL, R, bl_recs
    tag, s = BL["nnU-Net"]
    thr = float(s["key"].split("_vol")[0][2:])
    return bl_recs(f"{R}/{tag}_test_{mode}", thr, s["frac"]), f"{R}/{tag}_test_{mode}", s


def dump(obj, path):
    json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
