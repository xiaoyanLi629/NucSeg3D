import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
# 完整流程比较：A、B 各为一个结果目录（eval_gt 的 fp 结果或 eval_val 的结果，都读 rows.json），只比两边都有的体积（去掉 bbbc032）。
# 口径：cp-2_vol0.15；每套先对体积求平均再对数据集求平均（eval_val 口径，记为 vm），另给 pooled（每套合并对象再算 F1，记为 po）。
# 用法：python3 fp_cmp.py <A 目录> <B 目录> [标签A 标签B]
import json, sys
from collections import defaultdict
ONE = {"cellos", "parhyale", "mesospim"}                               # 视图规则下的单视图数据集
R = f"{_NSP_ROOT}/results/"


def load(d):
    d = d if d.startswith("/") else R + d
    rows = [x for x in json.load(open(d + "/rows.json")) if x["thr"] == -2.0 and abs(x["frac"] - 0.15) < 1e-9 and x["ds"] != "bbbc032"]
    return {(x["ds"], x["id"]): x for x in rows}


def agg(V, keys):
    by = defaultdict(list)
    for k in keys:
        by[k[0]].append(V[k])
    out = {}
    for ds, xs in by.items():
        vm5 = sum(x["f1"] for x in xs) / len(xs); vm7 = sum(x["f1_7"] for x in xs) / len(xs)
        den = sum(x["n_gt"] + x["n_pred"] for x in xs)
        po5 = sum(x["f1"] * (x["n_gt"] + x["n_pred"]) for x in xs) / max(den, 1)       # f1 = 2tp/(n_gt+n_pred)，按分母加权即 pooled
        po7 = sum(x["f1_7"] * (x["n_gt"] + x["n_pred"]) for x in xs) / max(den, 1)
        out[ds] = dict(vm5=vm5, vm7=vm7, po5=po5, po7=po7, n=len(xs))
    return out


def macros(P):
    sets = {"全13套": list(P), "去mesospim": [d for d in P if d != "mesospim"], "十个三视图": [d for d in P if d not in ONE]}
    return {nm: {k: sum(P[d][k] for d in ds) / len(ds) for k in ("vm5", "vm7", "po5", "po7")} for nm, ds in sets.items()}


if __name__ == "__main__":
    A, B = load(sys.argv[1]), load(sys.argv[2])
    la, lb = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("A", "B")
    ks = sorted(set(A) & set(B))
    onlyA, onlyB = len(set(A) - set(B)), len(set(B) - set(A))
    PA, PB = agg(A, ks), agg(B, ks)
    print(f"共同体积 {len(ks)}（只在 {la} {onlyA}，只在 {lb} {onlyB}），{len(PA)} 套；cp-2_vol0.15，逐体积平均口径")
    for d in sorted(PA):
        a, b = PA[d], PB[d]
        print(f"  {d:13s} n={a['n']:2d}  {la} {a['vm5']:.3f}/{a['vm7']:.3f}  {lb} {b['vm5']:.3f}/{b['vm7']:.3f}  差 {a['vm5'] - b['vm5']:+.3f}/{a['vm7'] - b['vm7']:+.3f}")
    MA, MB = macros(PA), macros(PB)
    for nm in MA:
        a, b = MA[nm], MB[nm]
        print(f"{nm:9s} 逐体积 {la} {a['vm5']:.4f}/{a['vm7']:.4f} {lb} {b['vm5']:.4f}/{b['vm7']:.4f} 差 {a['vm5'] - b['vm5']:+.4f}/{a['vm7'] - b['vm7']:+.4f}"
              f" | pooled 差 {a['po5'] - b['po5']:+.4f}/{a['po7'] - b['po7']:+.4f}")
    w5 = sum(PA[d]["vm5"] > PB[d]["vm5"] for d in PA); w7 = sum(PA[d]["vm7"] > PB[d]["vm7"] for d in PA)
    print(f"{la} 胜 {lb} 的数据集数：F1@0.5 {w5}/{len(PA)}，F1@0.7 {w7}/{len(PA)}")
    print("JSON " + json.dumps({nm: {k: round(MA[nm][k] - MB[nm][k], 4) for k in MA[nm]} for nm in MA}, ensure_ascii=False))
