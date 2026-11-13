"""L1/L2/L3 名单生成（只读数据与清单，不读任何模型输出或分数）。
输出都写在本脚本所在目录（$NUCSEG3D_SCRATCH/nc/heldout/）。
用法：$NUCSEG3D_SHARED/env_torch/bin/python make_lists.py
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import csv, glob, hashlib, json, os
from collections import Counter, defaultdict
import numpy as np
import tifffile

OUT = os.path.dirname(os.path.abspath(__file__))
STD = f"{_NSP_ROOT}/data/std"
VAL = f"{STD}/val_split.json"
CLEAN = f"{_NSP_SCRATCH}/r9/clean_test.json"
ISO = f"{_NSP_SCRATCH}/cpft_iso30/index.json"
SL = f"{_NSP_SCRATCH}/cpft_v5/v5_2d"
SEED = 20261105                     # L2 抽样种子，事先写定
KS = (1, 2, 5, 10)
L2_EVAL_CAP = 44                    # L2 评测：每套至多 44 个测试体积（按编号均匀取；只影响 blastospim 98→44），只因成本，事先写定

# 成像方式（粗分三类：共聚焦 = 点扫描与转盘；光片；多光子）。依据见 DESIGN.md 第 2 节。
# 混合数据集按训练体积多数的方式归类；“未核实”的不影响本次结果（见 DESIGN.md 2.3）。
MOD = {
    "bbbc050": ("共聚焦", "转盘共聚焦（Tokuoka 2020；训练 IX71、测试 CV1000），待一手核实 IX71 的共聚焦单元"),
    "blastospim": ("光片", "光片（Nunley 2024）"),
    "celegans": ("共聚焦", "Leica 共聚焦 63×（数据 readme）"),
    "cellos": ("共聚焦", "共聚焦（Mukashyaka 2023）"),
    "gonuclear": ("共聚焦", "共聚焦（Vijayan 2024）"),
    "mouse_embryo": ("共聚焦", "LSM780/880 共聚焦模式（数据 README）"),
    "nisnet_zf": ("未核实", "斑马鱼胚胎荧光（Li 等 2023 EI），显微方式未找到一手说明"),
    "nucverse3d": ("多光子", "肝脏 14/17 个训练体积为双光子，果蝇胶质 3/17 为共聚焦（NucVerse3D）"),
}


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def dump(name, obj):
    p = os.path.join(OUT, name)
    json.dump(obj, open(p, "w"), ensure_ascii=False, indent=1)
    return p, md5(p)


man = list(csv.DictReader(open(f"{STD}/manifest.csv")))
val = json.load(open(VAL))
valset = {(d, i) for d, ids in val.items() for i in ids}
clean = json.load(open(CLEAN))["clean"]
c1_ds = Counter(d for d, _ in clean)
tr = defaultdict(list); te = defaultdict(list)
for r in man:
    if r["dim"] != "3d":
        continue
    (tr if r["split"] == "train" else te)[r["ds"]].append(r["id"])

# ---------- 1. 候选表与确定性选择
rows = []
for ds in sorted(set(tr) | set(te)):
    rows.append(dict(ds=ds, in_C1=ds in c1_ds, n_train=len(tr[ds]), n_val=sum((ds, i) in valset for i in tr[ds]),
                     n_test=len(te[ds]), n_test_C1=c1_ds.get(ds, 0), modality=MOD.get(ds, ("—", ""))[0],
                     modality_src=MOD.get(ds, ("—", ""))[1]))
cand = [r for r in rows if r["in_C1"] and r["n_train"] >= 3]
# 规则：每类取训练体积最多者（并列按名字），再在各类胜者中取训练体积最多的 3 个（并列按名字）；“未核实”不单独成类、不参选
winners = {}
for r in sorted(cand, key=lambda r: (-r["n_train"], r["ds"])):
    if r["modality"] in ("共聚焦", "光片", "多光子") and r["modality"] not in winners:
        winners[r["modality"]] = r
sel = sorted(winners.values(), key=lambda r: (-r["n_train"], r["ds"]))[:3]
SEL = sorted(r["ds"] for r in sel)
# 稳健性：nisnet_zf 无论归哪一类都不改变结果
robust = {}
for alt in ("共聚焦", "光片", "多光子"):
    w = {}
    for r in sorted(cand, key=lambda r: (-r["n_train"], r["ds"])):
        m = alt if r["ds"] == "nisnet_zf" else r["modality"]
        if m in ("共聚焦", "光片", "多光子") and m not in w:
            w[m] = r["ds"]
    robust[alt] = sorted(sorted(w.values(), key=lambda d: (-len(tr[d]), d))[:3])

# ---------- 2. L1 排除清单：val_split ∪ 三套全部训练体积（含其验证体积）
ex = {d: sorted(v) for d, v in val.items()}
for d in SEL:
    ex[d] = sorted(set(ex.get(d, [])) | set(tr[d]))
p_l1, m_l1 = dump("L1_exclude.json", ex)
exset = {(d, i) for d, ids in ex.items() for i in ids}

# 二维切片数（L1 前后）
files = sorted(f for f in glob.glob(f"{SL}/*.tif") if not f.endswith(("_masks.tif", "_flows.tif")))
key = lambda f: tuple(os.path.basename(f).split("__")[:2])
n_off = sum(key(f) not in valset for f in files)
n_l1 = sum(key(f) not in exset for f in files)
sl_by_ds = Counter(key(f)[0] for f in files if key(f) not in valset)
# iso30
iso = json.load(open(ISO))
iso_off = [e for e in iso if e["split"] == "train" and e["ds"] != "mesospim"]
iso_l1 = [e for e in iso_off if (e["ds"], e["id"]) not in exset]
iso_by_ds = Counter(e["ds"] for e in iso_off)

# ---------- 3. L2 抽样池与嵌套抽样
iso_train = {(e["ds"], e["id"]) for e in iso_off}
sl_vol = defaultdict(list)
for f in files:
    sl_vol[key(f)].append(f)


def n_masks(f):
    return int(len(np.unique(tifffile.imread(f[:-4] + "_masks.tif"))) - 1)


l2 = {}
rng = np.random.default_rng(SEED)
for d in SEL:                                   # 按名字顺序依次抽，种子共用一个生成器（顺序固定）
    pool = []
    for i in sorted(tr[d]):
        if (d, i) in valset or (d, i) not in iso_train:
            continue
        good = [f for f in sl_vol[(d, i)] if n_masks(f) >= 3]   # Cellpose min_train_masks=3 之后仍有切片
        if good:
            pool.append(i)
    perm = [pool[j] for j in rng.permutation(len(pool))]
    l2[d] = dict(pool_size=len(pool), n_train_nonval=sum((d, i) not in valset for i in tr[d]), order=perm[:max(KS)], subsets={})
    for k in KS:
        chosen = perm[:min(k, len(perm))]
        e2 = {dd: list(v) for dd, v in ex.items()}
        e2[d] = sorted(set(e2[d]) - set(chosen))
        p, m = dump(f"L2_{d}_k{k}_exclude.json", e2)
        nsl = sum(len(sl_vol[(d, i)]) for i in chosen)
        nsl3 = sum(sum(n_masks(f) >= 3 for f in sl_vol[(d, i)]) for i in chosen)
        nn = sum(int(next(e for e in iso if e["ds"] == d and e["id"] == i)["n_nuclei"]) for i in chosen)
        l2[d]["subsets"][k] = dict(ids=chosen, exclude_json=p, md5=m, n_slices=nsl, n_slices_ge3=nsl3, n_nuclei=nn)

# ---------- 4. 评测名单
ev_l1 = {d: sorted(te[d]) for d in SEL}
dump("L1_eval_test.json", ev_l1)
ev_l2 = {}
for d in SEL:
    v = sorted(te[d])
    if len(v) > L2_EVAL_CAP:
        v = [v[int(j)] for j in np.linspace(0, len(v) - 1, L2_EVAL_CAP).round()]
    ev_l2[d] = v
dump("L2_eval_test.json", ev_l2)

# ---------- 5. L3 固定直径的 d_used 目录（eval_gt --diam_from 读 rows.json 的 d_used）
for d0 in (17, 30, 52):
    od = os.path.join(OUT, f"L3_fixed_d{d0}")
    os.makedirs(od, exist_ok=True)
    rr = [dict(ds=d, id=i, d_used=float(d0)) for d, ids in val.items() for i in ids]
    rr += [dict(ds=r["ds"], id=r["id"], d_used=float(d0)) for r in man if r["split"] == "test" and r["dim"] == "3d"]
    json.dump(rr, open(os.path.join(od, "rows.json"), "w"))

rep = dict(candidates=rows, eligible=[r["ds"] for r in cand], winners={k: v["ds"] for k, v in winners.items()},
           selected=SEL, robust_if_nisnet_zf_is=robust, L1_exclude=dict(path=p_l1, md5=m_l1, n_volumes=len(exset)),
           slices=dict(official=n_off, L1=n_l1, by_ds_official=dict(sl_by_ds)),
           iso30=dict(official=len(iso_off), L1=len(iso_l1), by_ds_official=dict(iso_by_ds)),
           L2=l2, L2_seed=SEED, L1_eval={d: len(v) for d, v in ev_l1.items()}, L2_eval={d: len(v) for d, v in ev_l2.items()})
json.dump(rep, open(os.path.join(OUT, "lists_report.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: rep[k] for k in ("eligible", "winners", "selected", "robust_if_nisnet_zf_is", "L1_exclude", "slices", "iso30", "L1_eval", "L2_eval")},
                 ensure_ascii=False, indent=1))
for r in rows:
    print(r)
for d in SEL:
    print(d, l2[d]["pool_size"], l2[d]["n_train_nonval"], {k: (v["ids"], v["n_slices"], v["n_slices_ge3"], v["n_nuclei"]) for k, v in l2[d]["subsets"].items()})
