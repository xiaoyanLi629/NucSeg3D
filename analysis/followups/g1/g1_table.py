"""nc G1：同一批测试体积上的横评表（CPU，只用已有结果；口径见 $NUCSEG3D_SCRATCH/nc/gpu/DESIGN_G2.md 第 1 节与 AGENTS.md 71977cd）。
体积：完整流程的 45 个测试体积（eval_gt --split test --max_per_ds 4），去 mesospim 后 44 个、12 套；不含任何外部数据集。
直径：全部用真值直径（gt 协议 / 旧横评同口径），384 裁块。
我们的模型 = S1 + 视图规则 + 规则 T（体积下限 0.15），种子 0/1000/2000 = tGT_S1 运行 07992b55、6981b29d、25e92303；
二维 + 规则 T = 81116913、f10f8a63、8bf74241；原版固定 cp−2（体积下限 0.15）= b5973303。规则 T 按 sizeThr_dir 的 R2 从 masks 重算。
对方方法 = 旧横评 bench_others（orx_1020_004144_others_tapenade、orx_1020_004452_others_anystar、orx_1020_004452_others_cellos），
取各自在旧验证集上选出的倍数与体积下限（summary.json），逐体积 TP = f1·(n_gt+n_pred)/2（F1@0.7 同理）还原后按 pooled 合并。
指标：每套 pooled F1@0.5、F1@0.7，12 套宏平均；另给干净子集 C1（audit/clean_test.json）体积上的宏平均。
注：Tapenade StarDist 训练时见过 tapenade 测试图像，Cellos StarDist 见过 cellos 测试图像（各自主场），表中标注。
用法：python g1_table.py  → $NUCSEG3D_SCRATCH/nc/gpu/g1/g1_table.json 与 g1_table.md"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import glob, json, math, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'src'))
from eval_gt import pick_cases
R = f"{_NSP_ROOT}/results/"
O = f"{_NSP_SCRATCH}/nc/gpu/g1"
OURS = dict(S1_s0="07992b55", S1_s1000="6981b29d", S1_s2000="25e92303", D2_s0="81116913", D2_s1000="f10f8a63", D2_s2000="8bf74241", base="b5973303")
OTH = dict(tapenade_sd="orx_1020_004144_others_tapenade", anystar="orx_1020_004452_others_anystar", cellos_sd="orx_1020_004452_others_cellos")
keys = [(r["ds"], r["id"]) for r in pick_cases("test", f"{_NSP_SCRATCH}/nc/gpu/wt/val_split.json", None, None, 4) if r["ds"] != "mesospim"]
K = set(keys)
clean = json.load(open(f"{_NSP_SHARED}/cellposeft_orx/audit/clean_test.json"))["clean"]
C1 = {tuple(k.split("__", 1)) if isinstance(k, str) else tuple(k) for k in (clean if isinstance(clean, list) else [f"{d}__{i}" for d, L in clean.items() for i in L])}


def ours(run, ruleT):
    d = glob.glob(R + "orx_" + run + "*")[0]
    rows = json.load(open(d + "/rows.json")); by = {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in rows}
    thrs = sorted({x["thr"] for x in rows}, reverse=True); out = {}
    for ds, vid in keys:
        t2 = -2.0
        if ruleT:
            cp = np.load(f"{d}/masks/{ds}__{vid}.npz")["cp"].astype(np.float32)
            r0 = by[(ds, vid, -2.0, 0.0)]; V = math.pi / 6 * r0.get("d_used", r0["d_gt"]) ** 3 / r0["ratio"]
            m = {t: float((cp > t).sum()) / max(by[(ds, vid, t, 0.0)]["n_pred"], 1) / V for t in thrs}
            t2 = -2.0 if m[-2.0] >= 0.7 else next((t for t in thrs if t < -2 and m[t] >= 0.85), min(thrs))
        x = by[(ds, vid, t2, 0.15)]
        out[(ds, vid)] = (x["tp5"], x["tp7"], x["n_gt"] + x["n_pred"])
    return out


def other(dn):
    d = R + dn; S = json.load(open(d + "/summary.json")); out = {}
    for x in json.load(open(d + "/rows.json")):
        k = (x["ds"], x["id"])
        if x["split"] == "test" and x["mult"] == S["mult"] and x["frac"] == S["frac"] and k in K:
            den = x["n_gt"] + x["n_pred"]; out[k] = (x["f1"] * den / 2, x["f1_7"] * den / 2, den)
    assert set(out) == K, (dn, len(out), sorted(K - set(out))[:3])
    return out, S["mult"], S["frac"]


def table(v, sub=None):
    per = {}
    for ds in sorted({k[0] for k in keys}):
        L = [v[k] for k in keys if k[0] == ds and (sub is None or k in sub)]
        if L:
            den = max(sum(x[2] for x in L), 1); per[ds] = (2 * sum(x[0] for x in L) / den, 2 * sum(x[1] for x in L) / den, len(L))
    return per, float(np.mean([p[0] for p in per.values()])), float(np.mean([p[1] for p in per.values()]))


M, meta = {}, {}
for k, run in OURS.items():
    M[k] = ours(run, ruleT=not k.startswith("base"))
for k, dn in OTH.items():
    M[k], mu, fr = other(dn); meta[k] = dict(mult=mu, frac=fr, run=dn)
res = {}
for k, v in M.items():
    per, a5, a7 = table(v); _, c5, c7 = table(v, C1)
    res[k] = dict(per_ds={d: dict(f1_05=p[0], f1_07=p[1], nvol=p[2]) for d, p in per.items()}, macro12_f1_05=a5, macro12_f1_07=a7,
                  c1_f1_05=c5, c1_f1_07=c7, n_vol=len(keys))
s1 = [res[f"S1_s{s}"] for s in (0, 1000, 2000)]; d2 = [res[f"D2_s{s}"] for s in (0, 1000, 2000)]
for tag, L in (("S1_mean", s1), ("D2_mean", d2)):
    res[tag] = {q: float(np.mean([x[q] for x in L])) for q in ("macro12_f1_05", "macro12_f1_07", "c1_f1_05", "c1_f1_07")}
    res[tag]["per_ds"] = {d: dict(f1_05=float(np.mean([x["per_ds"][d]["f1_05"] for x in L])), f1_07=float(np.mean([x["per_ds"][d]["f1_07"] for x in L]))) for d in L[0]["per_ds"]}
json.dump(dict(volumes=len(keys), datasets=len(res["S1_s0"]["per_ds"]), c1_volumes=len(K & C1), others=meta, results=res), open(f"{O}/g1_table.json", "w"), indent=1)
cols = ["S1_mean", "D2_mean", "base", "tapenade_sd", "anystar", "cellos_sd"]
name = dict(S1_mean="我们的模型（S1，3 种子均值）", D2_mean="二维（3 种子均值）", base="原版 Cellpose-SAM", tapenade_sd="Tapenade StarDist", anystar="AnyStar", cellos_sd="Cellos StarDist")
lines = ["| 数据集 | " + " | ".join(name[c] for c in cols) + " |", "|---|" + "---|" * len(cols)]
for d in res["S1_s0"]["per_ds"]:
    lines.append(f"| {d} | " + " | ".join(f"{res[c]['per_ds'][d]['f1_05']:.3f}" for c in cols) + " |")
lines.append("| 宏平均 F1@0.5（12 套） | " + " | ".join(f"{res[c]['macro12_f1_05']:.3f}" for c in cols) + " |")
lines.append("| 宏平均 F1@0.7（12 套） | " + " | ".join(f"{res[c]['macro12_f1_07']:.3f}" for c in cols) + " |")
lines.append("| C1 宏平均 F1@0.5 | " + " | ".join(f"{res[c]['c1_f1_05']:.3f}" for c in cols) + " |")
lines.append("")
lines.append("S1 三个种子 12 套 F1@0.5：" + " / ".join(f"{x['macro12_f1_05']:.4f}" for x in s1) + "；F1@0.7：" + " / ".join(f"{x['macro12_f1_07']:.4f}" for x in s1))
lines.append("二维三个种子 12 套 F1@0.5：" + " / ".join(f"{x['macro12_f1_05']:.4f}" for x in d2) + "；F1@0.7：" + " / ".join(f"{x['macro12_f1_07']:.4f}" for x in d2))
lines.append("对方方法所选（倍数, 体积下限）：" + "，".join(f"{k} ({v['mult']:g}, {v['frac']:g})" for k, v in meta.items()))
open(f"{O}/g1_table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines)); print("C1 体积", len(K & C1))
