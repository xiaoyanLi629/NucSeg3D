"""nc G6 比较（验证集：python g6_val.py；测试集：python g6_val.py test，只报告）。原验证集说明：（CPU）：各向同性跟流 开（a69ee4fa）对 关（G6v_off 运行 24cfefb5），S1 种子 0，gt 协议，视图规则，规则 T（体积下限 0.15）。
口径见 $NUCSEG3D_SCRATCH/nc/gpu/DESIGN.md 第 2 节。规则 T 按 sizeThr_dir 的 R2 从两边各自的 masks 重算。
报告：12 套（去 mesospim）/ 13 套宏平均 pooled F1@0.5、F1@0.7 与差（关 − 开），按体积在各套内分层配对自助 10000 次（种子 0）的 95% 区间；
受影响（z/xy ≥ 2）体积上的合并比例（真值核被归为 merge 的比例）、找对比例（hit）、z 向合并占比（merge_z/(merge_z+merge_xy)）；
z/xy < 2 的体积两边逐行比较（除耗时、gx、gpu_peak_mib 外逐字段相同）作复现核对。另给固定 cp−2 的同样结果作参照。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import glob, json, math, os
import numpy as np
R = f"{_NSP_ROOT}/results/"
import sys
TEST = len(sys.argv) > 1 and sys.argv[1] == "test"
ON = glob.glob(R + ("orx_07992b55*" if TEST else "orx_a69ee4fa*"))[0]; OFF = glob.glob(R + ("orx_d10cd4b9*" if TEST else "orx_24cfefb5*"))[0]
# 测试集：“关”只跑了 z/xy ≥ 2 的 144 个体积，其余体积两种做法代码相同，直接用“开”的结果（宏平均与验证集同口径，覆盖全部 227 个体积）
O = f"{_NSP_SCRATCH}/nc/gpu/g6"


def load(d):
    rows = json.load(open(d + "/rows.json")); by = {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in rows}
    thrs = sorted({x["thr"] for x in rows}, reverse=True); sel = {}
    for f in sorted(glob.glob(d + "/masks/*.npz")):
        ds, vid = os.path.basename(f)[:-4].split("__", 1)
        cp = np.load(f)["cp"].astype(np.float32); r0 = by[(ds, vid, -2.0, 0.0)]
        V = math.pi / 6 * r0.get("d_used", r0["d_gt"]) ** 3 / r0["ratio"]
        m = {t: float((cp > t).sum()) / max(by[(ds, vid, t, 0.0)]["n_pred"], 1) / V for t in thrs}
        sel[(ds, vid)] = -2.0 if m[-2.0] >= 0.7 else next((t for t in thrs if t < -2 and m[t] >= 0.85), min(thrs))
    return by, sel


bon, son = load(ON); boff, soff = load(OFF)
vols = sorted(son)
if TEST:
    assert len(vols) == 227 and len(soff) == 144
    for k in vols:
        if k not in soff:
            soff[k] = son[k]
            for t in (0.0, -1.0, -2.0, -3.0, -4.0, -6.0):
                for f in (0.0, 0.15, 0.25):
                    boff[(*k, t, f)] = bon[(*k, t, f)]
else:
    assert sorted(soff) == vols and len(vols) == 88
ratio = {k: bon[(*k, -2.0, 0.0)]["ratio"] for k in vols}
# 复现核对
skip = {"sec_net", "sec_total", "gx", "gpu_peak_mib"}
iso_lt2 = [k for k in vols if ratio[k] < 2]; same = 0; diffs = []
for k in iso_lt2:
    ok = True
    for t in (0.0, -1.0, -2.0, -3.0, -4.0, -6.0):
        for f in (0.0, 0.15, 0.25):
            a, b = bon[(*k, t, f)], boff[(*k, t, f)]
            dd = [c for c in a if c not in skip and a[c] != b.get(c)]
            if dd: ok = False; diffs.append((k, t, f, dd[:3]))
    same += ok


def pick(by, sel, k, rule):
    return by[(*k, sel[k] if rule == "T" else -2.0, 0.15)]


def macro(rows_by_vol, idx_by_ds, dss):
    out5, out7 = [], []
    for ds in dss:
        I = idx_by_ds[ds]; tp5 = sum(rows_by_vol[i]["tp5"] for i in I); tp7 = sum(rows_by_vol[i]["tp7"] for i in I)
        den = max(sum(rows_by_vol[i]["n_gt"] + rows_by_vol[i]["n_pred"] for i in I), 1)
        out5.append(2 * tp5 / den); out7.append(2 * tp7 / den)
    return np.mean(out5), np.mean(out7), out5, out7


C1 = {tuple(x) for x in json.load(open(f"{_NSP_SHARED}/cellposeft_orx/audit/clean_test.json"))["clean"]}
res = dict(n=len(vols), n_aniso=sum(ratio[k] >= 2 for k in vols), repro=dict(n_lt2=len(iso_lt2), identical=same, diffs=diffs[:5]))
rng = np.random.default_rng(0)
dsl = sorted({k[0] for k in vols}); pos = {ds: [i for i, k in enumerate(vols) if k[0] == ds] for ds in dsl}
for rule in ("T", "fixed"):
    A = [pick(bon, son, k, rule) for k in vols]; B = [pick(boff, soff, k, rule) for k in vols]
    subsets = [("12", [d for d in dsl if d != "mesospim"], pos), ("13", dsl, pos)]
    if TEST:
        pc = {ds: [i for i in pos[ds] if vols[i] in C1] for ds in dsl}
        subsets.append(("C1", [d for d in dsl if pc[d] and d != "mesospim"], pc))
    for tag, dss, pos_ in subsets:
        a5, a7, pa5, pa7 = macro(A, pos_, dss); b5, b7, pb5, pb7 = macro(B, pos_, dss)
        bs = np.zeros((10000, 2))
        for r in range(10000):
            P = {ds: [pos_[ds][j] for j in rng.integers(0, len(pos_[ds]), len(pos_[ds]))] for ds in dss}
            x5, x7, _, _ = macro(A, P, dss); y5, y7, _, _ = macro(B, P, dss); bs[r] = (y5 - x5, y7 - x7)
        lo, hi = np.percentile(bs, [2.5, 97.5], 0)
        res[f"{rule}_{tag}"] = dict(on=(a5, a7), off=(b5, b7), diff=(b5 - a5, b7 - a7), ci05=(lo[0], hi[0]), ci07=(lo[1], hi[1]),
                                    per_ds={d: dict(on=pa5[i], off=pb5[i], d05=pb5[i] - pa5[i], d07=pb7[i] - pa7[i]) for i, d in enumerate(dss)})
    an = [i for i, k in enumerate(vols) if ratio[k] >= 2]
    for nm, L in (("on", A), ("off", B)):
        g = sum(L[i]["n_gt"] for i in an)
        res[f"{rule}_err_{nm}"] = dict(merge=sum(L[i]["merge"] for i in an) / g, hit=sum(L[i]["hit"] for i in an) / g,
                                       split=sum(L[i]["split"] for i in an) / g, miss=sum(L[i]["miss"] for i in an) / g, n_gt=g,
                                       zmerge=sum(L[i]["merge_z"] for i in an) / max(sum(L[i]["merge_z"] + L[i]["merge_xy"] for i in an), 1))
    res[f"{rule}_thr_changed"] = sum(son[k] != soff[k] for k in vols) if rule == "T" else 0
json.dump(res, open(f"{O}/g6_{'test' if TEST else 'val'}.json", "w"), indent=1, default=float)
print(f"复现：z/xy<2 {len(iso_lt2)} 个体积中 {same} 个逐字段相同；z/xy≥2 {res['n_aniso']} 个")
for rule in ("T", "fixed"):
    for tag in ("12", "13") + (("C1",) if TEST else ()):
        x = res[f"{rule}_{tag}"]
        print(f"{rule} {tag}套 开 {x['on'][0]:.4f}/{x['on'][1]:.4f} 关 {x['off'][0]:.4f}/{x['off'][1]:.4f} 差 {x['diff'][0]:+.4f} [{x['ci05'][0]:+.4f},{x['ci05'][1]:+.4f}] / {x['diff'][1]:+.4f} [{x['ci07'][0]:+.4f},{x['ci07'][1]:+.4f}]")
    for nm in ("on", "off"):
        e = res[f"{rule}_err_{nm}"]; print(f"  {rule} {nm}: 合并 {e['merge']:.4f} 找对 {e['hit']:.4f} 切碎 {e['split']:.4f} 漏 {e['miss']:.4f} z向合并占比 {e['zmerge']:.3f} (n_gt {e['n_gt']})")
print("规则T阈值不同的体积", res["T_thr_changed"])
print("逐套（规则T，13套）F1@0.5 关−开：" + "  ".join(f"{d} {v['d05']:+.3f}" for d, v in res["T_13"]["per_ds"].items()))
print("逐套（规则T）F1@0.7 关−开：" + "  ".join(f"{d} {v['d07']:+.3f}" for d, v in res["T_13"]["per_ds"].items()))
