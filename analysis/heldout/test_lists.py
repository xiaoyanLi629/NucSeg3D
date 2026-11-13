"""CPU 自检：nc_L.diff 打补丁后的 train3d/data3d 在 L1、L2 名单下选出的训练体积，以及 L2 的父清单检查。不加载任何模型。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json, os, sys, types
from collections import Counter
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'src'))
import data3d, train3d
H = f"{_NSP_SCRATCH}/nc/heldout"
store = f"{_NSP_SCRATCH}/cpft_iso30"
exd = ("mesospim",)
off = [e for e in data3d.load_index(store) if e["split"] == "train" and e["ds"] not in exd]


def sel(path):
    ex, m = train3d.load_exclude_ids(path, store, exd)
    idx = [e for e in data3d.load_index(store) if e["split"] in ("train", "val") and e["ds"] not in exd and (e["ds"], e["id"]) not in ex]
    return ex, m, idx


ex1, m1, idx1 = sel(f"{H}/L1_exclude.json")
print("L1 md5", m1, "训练体积", len(idx1), "其中 split=val", sum(e["split"] == "val" for e in idx1))
assert {(e["ds"], e["id"]) for e in idx1} == {(e["ds"], e["id"]) for e in off if e["ds"] not in ("bbbc050", "blastospim", "nucverse3d")}
print("L1 = 官方训练体积去掉三套，逐项相同；数据集份额", data3d.dataset_probs(idx1, {"cellos": 2, "tapenade": 2}, {"blastospim": 0.25})[1])
for d in ("bbbc050", "blastospim", "nucverse3d"):
    for k in (1, 2, 5, 10):
        p = f"{H}/L2_{d}_k{k}_exclude.json"
        ex, m, idx = sel(p)
        back = Counter(e["ds"] for e in idx if e["ds"] == d)
        a = types.SimpleNamespace(guard_parent=f"{H}/L1_exclude.json", guard_max_back=10, init="__none__", init3d=None)
        try:
            train3d.fold_guard(a, m, ex)      # 二维 config 不存在时 ex2 为 None，会在 md5 比较处报错，这里只看父清单检查是否通过
        except AssertionError as err:
            assert "二维初始权重" in str(err), err
        assert len(idx) == len(idx1) + min(k, 10) and back[d] == k, (d, k, len(idx), back)
        print(d, k, "训练体积", len(idx), "放回", back[d])
# 反例：父清单检查应拦住放回两套数据
bad = json.load(open(f"{H}/L1_exclude.json")); bad["bbbc050"] = bad["bbbc050"][1:]; bad["nucverse3d"] = bad["nucverse3d"][1:]
json.dump(bad, open("/tmp/nc_bad.json", "w"))
exb, mb = train3d.load_exclude_ids("/tmp/nc_bad.json", store, exd)
a = types.SimpleNamespace(guard_parent=f"{H}/L1_exclude.json", guard_max_back=10, init="__none__", init3d=None)
try:
    train3d.fold_guard(a, mb, exb); print("反例没有被拦住！")
except AssertionError as err:
    print("反例被拦住：", str(err)[:60])
os.remove("/tmp/nc_bad.json")
print("NC_TEST_DONE")
