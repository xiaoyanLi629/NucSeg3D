"""CPU 自检，不发任何作业。读 orx/nc-l1-l3-r7-t 的代码副本。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_ORX_ENV = _nsp_os.environ.get('NUCSEG3D_ORX_ENV', '/path/to/orx_env.sh')
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import glob, hashlib, json, os, re, subprocess, sys
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'src'))
import data3d
H = f"{_NSP_SCRATCH}/nc/heldout"; SH = f"{_NSP_SHARED}/nc_heldout"; R = f"{_NSP_ROOT}"
J = json.load(open(f"{H}/jobs_L1seeds.json")); live = json.load(open(f"{_NSP_SCRATCH}/r6/jobs.json")); st = json.load(open(f"{_NSP_SCRATCH}/r6/sched_state.json"))
L = {j["key"]: j for j in live}; D = {j["key"]: j for j in J}
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
# 1. 排除清单
EX = f"{SH}/L1_exclude.json"
assert md5(EX) == md5(f"{H}/L1_exclude.json") == "eee3dda52aef78038e1851e5fe4df783"
ex = {(d, i) for d, v in json.load(open(EX)).items() for i in v}; print("1 清单 md5 一致，体积", len(ex))
# 2. 三维训练体积（分支的 make_loader，workers=0；种子只进抽样，不改体积列表）
store = f"{_NSP_SCRATCH}/cpft_iso30"
sets = []
for s in (0, 1000, 2000):
    _, idx, share = data3d.make_loader(store, "train", {"cellos": 2.0, "tapenade": 2.0}, {"blastospim": 0.25}, (32, 256, 256), workers=0, seed=s,
                                       exclude_ds=("mesospim",), include_val=True, exclude_ids=ex)
    sets.append([(e["ds"], e["id"]) for e in idx]); assert len(idx) == 265, len(idx); assert not any(x in ex for x in sets[-1])
assert sets[0] == sets[1] == sets[2]
off = {(e["ds"], e["id"]) for e in data3d.load_index(store) if e["split"] == "train" and e["ds"] != "mesospim"}
assert set(sets[0]) == {x for x in off if x[0] not in ("bbbc050", "blastospim", "nucverse3d")}
print("2 三维训练体积 265，三个种子列表逐项相同，等于正式版 910 个去掉三套；split=val 0 个")
# 3. 二维切片
files = [f for f in glob.glob(f"{_NSP_SCRATCH}/cpft_v5/v5_2d/*.tif") if not f.endswith(("_masks.tif", "_flows.tif"))]
n = sum(tuple(os.path.basename(f).split("__")[:2]) not in ex for f in files); assert n == 3823, n; print("3 二维切片 3823（与种子 0 日志相同）")
# 4. 还原比较：把草稿的种子改回 0，配置必须与 jobs.json 的种子 0 逐字段相同
def back(x, s):
    if isinstance(x, dict): return {k: back(v, s) for k, v in x.items()}
    if isinstance(x, list):
        o = [back(v, s) for v in x]
        for i in range(1, len(x)):
            if x[i - 1] in ("--seed", "--seed_offset"):
                assert x[i] == str(s), x; o[i] = "0"
        return o
    return re.sub(rf"_s{s}(?![0-9])", "_s0", x) if isinstance(x, str) else x
nc = 0
for j in J[1:]:
    s = int(re.match(r"L1s(\d+)_", j["key"]).group(1)); k0 = re.sub(r"L1s\d+_", "L1_", j["key"]); o = L[k0]
    if j["kind"] == "cmd":
        c = back(j["cmd"], s).replace(f"L1s{s}_2D", "L1_2D"); assert c == o["cmd"], (c, o["cmd"])
    else:
        assert back(j["config"], s) == o["config"], j["key"]; assert j["parent"] == o["parent"] and j["mem"] == o["mem"] and j["block"] == o["block"], j["key"]
        assert not re.search(r"_s0(?![0-9])", json.dumps(j["config"])), j["key"]
        assert "hostC:2" not in j["gpus"], j["key"]
    nc += 1
print("4 还原比较", nc, "个作业，配置、父节点、显存预留、block 都与种子 0 相同")
# 5. 依赖、重名、已存在
keys = [j["key"] for j in J]; assert len(keys) == len(set(keys))
for j in J:
    for d in j["deps"]: assert d in D, (j["key"], d)
    assert j["key"] not in L and j["key"] not in st, j["key"]
titles = [j["title"] for j in J if j["kind"] != "cmd"]; assert len(titles) == len(set(titles))
txt = subprocess.run(["bash", "-c", f"source {_NSP_ORX_ENV} && orx project view e7a8a409-e4c6-4509-82a1-102de563dcb8"], capture_output=True, text=True, check=True).stdout
ID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
nodes = re.findall(rf"^\s*({ID})\s+(.*?)\s+\((orx/[^)]*)\)\s*$", txt, re.M); have = {t for _, t, _ in nodes} | {j.get("title") for j in live}
assert len(nodes) > 100, len(nodes); dup = [t for t in titles if t in have]; assert not dup, dup
par = [b for i, _, b in nodes if i.startswith("fcb392eb")]; print("5 标题", len(titles), "个，与现有", len(nodes), "个节点不重名；父节点分支", par)
assert par == ["orx/nc-l1-l3-r7-t"]
for s in (1000, 2000):
    for n in ("2d", "a1", "c", "x1", "s1"):
        assert not glob.glob(f"{R}/models/models/orx_nc_l1_{n}_s{s}*"), n
    assert not os.path.lexists(f"{R}/results/orx_nc_l1_2d_s{s}_test")
print("6 权重名与链接名都未被占用")
assert not os.path.exists(f"{SH}/GO_seeds"); print("7 GO_seeds 尚未创建，草稿发不出去")
print("NC_SEEDS_TEST_DONE")
