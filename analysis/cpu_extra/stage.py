"""把 CPUX 计算所需的只读输入原样复制到共享盘（hostC 读不到 hostA 本地盘），并写 md5 清单。不做任何统计计算。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import hashlib
import json
import os
import shutil
import sys
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/five_seed'))
import s5_stats  # noqa: E402

DST = f"{_NSP_SHARED}/nc_cpu_extra/inputs"
SEEDS = ["s0", "s1000", "s2000", "s3000", "s4000"]
os.makedirs(DST, exist_ok=True)
D = s5_stats.dirs("test", "gt", SEEDS)
V = s5_stats.dirs("val", "gt", ["s0"])
runmap = {"test_gt": D, "val_gt_S1_s0": V["S1"][0], "seeds": SEEDS}
json.dump(runmap, open(f"{DST}/runmap.json", "w"), indent=1)
pairs = []
for d in D["S1"] + D["2D"]:
    b = os.path.basename(d); pairs.append((f"{_NSP_SCRATCH}/nc/stats/cache/recs_{b}.json", f"{DST}/recs/recs_{b}.json"))
b = os.path.basename(D["base"][0]); pairs.append((f"{_NSP_SCRATCH}/nc/stats/cache/recs_{b}_fixed.json", f"{DST}/recs/recs_{b}_fixed.json"))
for root, sub in ((f"{_NSP_SCRATCH}/nc/stats/cache/ro", "ro"), (_nsp_os.path.join(_NSP_REPO, 'src'), "src_k1")):
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".pyc"):
                continue
            s = os.path.join(dp, f); pairs.append((s, os.path.join(DST, sub, os.path.relpath(s, root))))
for nm in ("S1", "2D"):
    for s in SEEDS:
        root = f"{_NSP_SCRATCH}/nc/errors/cache/gt_{nm}_{s}"
        for f in os.listdir(root):
            pairs.append((f"{root}/{f}", f"{DST}/errcache/gt_{nm}_{s}/{f}"))
pairs.append((f"{_NSP_SCRATCH}/nc/errors/runs.json", f"{DST}/errors_runs.json"))
man = {}
for s, d in pairs:
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.copy2(s, d)
    h1 = hashlib.md5(open(s, "rb").read()).hexdigest(); h2 = hashlib.md5(open(d, "rb").read()).hexdigest()
    assert h1 == h2, s
    man[os.path.relpath(d, DST)] = dict(src=s, md5=h1)
json.dump(man, open(f"{DST}/MANIFEST_md5.json", "w"), indent=1)
print(len(man), "个文件，md5 逐一一致")
