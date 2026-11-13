"""nc G 系列名单（只看清单里的体素与编号，不看任何结果分数）：
  g4_keys.txt：G4 的“接近各向同性”测试体积 = 清单体素 0.67 ≤ z/xy ≤ 1.5（体素缺失按 1/1/1）的测试体积，每套按编号排序后均匀取至多 4 个
              （与 eval_gt.pick_cases 的 linspace 取法相同）。
  g4_diam/rows.json：全部 227 个测试体积的真值直径（取已有 gt 协议运行 07992b55 的 d_gt，并核对与二维、原版运行的 d_gt 相同），
              作为 --diam_from 传入，使 z 降采样后仍用原体积的真值直径（避免降采样改变直径估计）。
  g6_test_keys.txt：G6 测试集会受开关影响的体积（z/xy ≥ 2），只作记录（作业用 --gx_min_ratio 2）。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json, os, sys, glob
from collections import defaultdict
import numpy as np
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'src'))
from stddata import load_manifest
O = f"{_NSP_SCRATCH}/nc/gpu/lists"; os.makedirs(O + "/g4_diam", exist_ok=True)
R = f"{_NSP_ROOT}/results/"
rs = sorted((r for r in load_manifest(split="test", dim="3d") if r["ds"] != "bbbc032"), key=lambda r: (r["ds"], r["id"]))
rat = lambda r: (r["voxel"] or (1, 1, 1))[0] / (r["voxel"] or (1, 1, 1))[1]
by = defaultdict(list)
for r in rs:
    if 0.67 <= rat(r) <= 1.5:
        by[r["ds"]].append(r)
keys = []
for ds, v in sorted(by.items()):
    if len(v) > 4:
        v = [v[int(j)] for j in np.linspace(0, len(v) - 1, 4).round()]
    keys += [f"{r['ds']}__{r['id']}" for r in v]
open(O + "/g4_keys.txt", "w").write("\n".join(keys) + "\n")
print("G4", len(keys), {ds: sum(k.startswith(ds + "__") for k in keys) for ds in by})
dm = {}
for run in ("07992b55", "81116913", "b5973303"):
    d = glob.glob(R + "orx_" + run + "*")[0]
    for x in json.load(open(d + "/rows.json")):
        k = (x["ds"], x["id"])
        if run == "07992b55":
            dm.setdefault(k, x["d_gt"])
        else:
            assert abs(dm[k] - x["d_gt"]) < 1e-6, (run, k, dm[k], x["d_gt"])
assert len(dm) == len(rs), (len(dm), len(rs))
json.dump([dict(ds=k[0], id=k[1], d_used=v) for k, v in sorted(dm.items())], open(O + "/g4_diam/rows.json", "w"))
g6 = [f"{r['ds']}__{r['id']}" for r in rs if rat(r) >= 2]
open(O + "/g6_test_keys.txt", "w").write("\n".join(g6) + "\n")
print("G6 test aniso", len(g6), "val:", end=" ")
vs = json.load(open(f"{_NSP_SCRATCH}/nc/gpu/wt/val_split.json"))
man = {(r["ds"], r["id"]): r for r in load_manifest(split="train", dim="3d")}
va = [(ds, i) for ds, L in vs.items() if ds != "bbbc032" for i in L]
print(len(va), "aniso", sum(rat(man[k]) >= 2 for k in va))
