"""nc G4 汇总（CPU，只报告）。口径见 $NUCSEG3D_SCRATCH/nc/gpu/DESIGN.md 第 3 节。
20 个接近各向同性的测试体积（lists/g4_keys.txt），6 套；每个条件 = z 降采样 k ∈ {1,2,4,8} × 噪声 c ∈ {0,0.25,0.5}。
S1、二维：规则 T（各运行 noise_<c>/ruleT.json 的逐体积阈值），体积下限 0.15；原版：固定 cp−2、体积下限 0.15。每套 pooled，再 6 套宏平均。
S1、二维各三个种子（0/1000/2000）给均值 ± SD；S1−二维 按同种子配对，给三个配对差的均值与范围。另给去掉 tapenade、purdue、nis3d 后的 3 套（C1 套）。
复现核对：k=1、c=0 与已有 gt 协议运行（07992b55 等）逐体积比较。
G4m：mesoSPIM 测试体积在旧体素 0.16/1/1 下的 S1 三种子（规则 T），对照源数据体素 0.475 的已有 gt 协议结果。
输出 $NUCSEG3D_SCRATCH/nc/gpu/g4/g4_summary.json 与 g4_fig.png。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import glob, json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
R = f"{_NSP_ROOT}/results/"
O = f"{_NSP_SCRATCH}/nc/gpu/g4"
S = json.load(open(f"{_NSP_SCRATCH}/r6/sched_state.json"))
KEYS = [tuple(l.strip().split("__", 1)) for l in open(f"{_NSP_SCRATCH}/nc/gpu/lists/g4_keys.txt") if l.strip()]
DS = sorted({k[0] for k in KEYS}); C1 = [d for d in DS if d not in ("tapenade", "purdue", "nis3d")]
KS, CS = (1, 2, 4, 8), ("0", "0.25", "0.5")
INST = [("S1", s) for s in (0, 1000, 2000)] + [("2D", s) for s in (0, 1000, 2000)] + [("base", None)]


def rdir(key):
    return glob.glob(R + "orx_" + S[key]["run"] + "*")[0]


def pick(d, ruleT):
    rows = {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in json.load(open(d + "/rows.json"))}
    t = {(r["ds"], r["id"]): r["t"] for r in json.load(open(d + "/ruleT.json"))} if ruleT else {}
    out = {}
    for k in KEYS:
        x = rows[(*k, t.get(k, -2.0) if ruleT else -2.0, 0.15)]
        out[k] = (x["tp5"], x["tp7"], x["n_gt"] + x["n_pred"])
    return out


def macro(v, dss):
    a5, a7 = [], []
    for ds in dss:
        L = [v[k] for k in KEYS if k[0] == ds]; den = max(sum(x[2] for x in L), 1)
        a5.append(2 * sum(x[0] for x in L) / den); a7.append(2 * sum(x[1] for x in L) / den)
    return float(np.mean(a5)), float(np.mean(a7))


res = {}
for k in KS:
    for c in CS:
        cell = {}
        for m, s in INST:
            key = f"G4_{m}_s{s}_z{k}" if s is not None else f"G4_base_z{k}"
            v = pick(f"{rdir(key)}/noise_{c}", ruleT=m != "base")
            cell[f"{m}_{s}"] = dict(all=macro(v, DS), c1=macro(v, C1))
        for sub in ("all", "c1"):
            for i, met in enumerate(("f05", "f07")):
                s1 = [cell[f"S1_{s}"][sub][i] for s in (0, 1000, 2000)]; d2 = [cell[f"2D_{s}"][sub][i] for s in (0, 1000, 2000)]
                dd = [a - b for a, b in zip(s1, d2)]
                cell[f"{sub}_{met}"] = dict(S1=(np.mean(s1), np.std(s1, ddof=1)), D2=(np.mean(d2), np.std(d2, ddof=1)), base=cell["base_None"][sub][i],
                                            diff=(np.mean(dd), min(dd), max(dd)), diff_seeds=dd)
        res[f"z{k}_n{c}"] = cell
# 复现核对
ref = dict(S1_0="07992b55", S1_1000="6981b29d", S1_2000="25e92303", D2_0="81116913", D2_1000="f10f8a63", D2_2000="8bf74241", base="b5973303")
rep = {}
for nm, run in ref.items():
    key = {"base": "G4_base_z1"}.get(nm) or ("G4_S1_s%s_z1" % nm.split("_")[1] if nm.startswith("S1") else "G4_2D_s%s_z1" % nm.split("_")[1])
    a = {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in json.load(open(rdir(key) + "/noise_0/rows.json"))}
    b = {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in json.load(open(glob.glob(R + "orx_" + run + "*")[0] + "/rows.json")) if (x["ds"], x["id"]) in set(KEYS)}
    same = sum(all(a[q][f] == b[q][f] for f in ("tp5", "tp7", "n_gt", "n_pred")) for q in b)
    rep[nm] = f"{same}/{len(b)}"
# G4m
def rule_from_masks(d, k):
    rows = {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in json.load(open(d + "/rows.json"))}
    thrs = sorted({q[2] for q in rows}, reverse=True)
    cp = np.load(f"{d}/masks/{k[0]}__{k[1]}.npz")["cp"].astype(np.float32); r0 = rows[(*k, -2.0, 0.0)]
    V = math.pi / 6 * r0.get("d_used", r0["d_gt"]) ** 3 / r0["ratio"]
    m = {t: float((cp > t).sum()) / max(rows[(*k, t, 0.0)]["n_pred"], 1) / V for t in thrs}
    t2 = -2.0 if m[-2.0] >= 0.7 else next((t for t in thrs if t < -2 and m[t] >= 0.85), min(thrs))
    x = rows[(*k, t2, 0.15)]; y = rows[(*k, -2.0, 0.15)]
    return dict(ruleT=(x["f1"], x["f1_7"], t2), fixed=(y["f1"], y["f1_7"]), d=r0["d_gt"], ratio=r0["ratio"])
mk = ("mesospim", "visual"); g4m = {}
for s, run in ((0, "07992b55"), (1000, "6981b29d"), (2000, "25e92303")):
    g4m[s] = dict(v016=rule_from_masks(rdir(f"G4m_S1_s{s}"), mk), v0475=rule_from_masks(glob.glob(R + "orx_" + run + "*")[0], mk))
out = dict(keys=len(KEYS), datasets=DS, c1_sets=C1, repro_k1c0=rep, conditions=res, g4m=g4m)
json.dump(out, open(f"{O}/g4_summary.json", "w"), indent=1, default=float)
print("复现 k=1 c=0：", rep)
for sub in ("all", "c1"):
    print(f"== {sub}（{len(DS) if sub == 'all' else len(C1)} 套）  S1 均值±SD | 二维 均值±SD | 原版 | S1−二维 均值 [最小, 最大]")
    for met in ("f05", "f07"):
        for k in KS:
            print(f"  {met} z×{k}: " + "   ".join(
                f"c{c}: {res[f'z{k}_n{c}'][f'{sub}_{met}']['S1'][0]:.3f}±{res[f'z{k}_n{c}'][f'{sub}_{met}']['S1'][1]:.3f} | {res[f'z{k}_n{c}'][f'{sub}_{met}']['D2'][0]:.3f}±{res[f'z{k}_n{c}'][f'{sub}_{met}']['D2'][1]:.3f} | {res[f'z{k}_n{c}'][f'{sub}_{met}']['base']:.3f} | {res[f'z{k}_n{c}'][f'{sub}_{met}']['diff'][0]:+.3f} [{res[f'z{k}_n{c}'][f'{sub}_{met}']['diff'][1]:+.3f},{res[f'z{k}_n{c}'][f'{sub}_{met}']['diff'][2]:+.3f}]"
                for c in CS))
for s, v in g4m.items():
    print(f"G4m S1 s{s}: 0.16 规则T {v['v016']['ruleT'][0]:.3f}/{v['v016']['ruleT'][1]:.3f} (t={v['v016']['ruleT'][2]:g}, d={v['v016']['d']:.1f}) 固定 {v['v016']['fixed'][0]:.3f}/{v['v016']['fixed'][1]:.3f} | 0.475 规则T {v['v0475']['ruleT'][0]:.3f}/{v['v0475']['ruleT'][1]:.3f} (t={v['v0475']['ruleT'][2]:g}, d={v['v0475']['d']:.1f}) 固定 {v['v0475']['fixed'][0]:.3f}/{v['v0475']['fixed'][1]:.3f}")
# 图
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
col = dict(S1="C3", D2="C0", base="0.4"); lab = dict(S1="我们的模型 S1（3 种子均值±SD）", D2="二维微调（3 种子）", base="原版 Cellpose-SAM（固定 cp−2）")
for i, met in enumerate(("f05", "f07")):
    for j, c in enumerate(CS):
        ax = axes[i, j]
        for m in ("S1", "D2"):
            y = np.array([res[f"z{k}_n{c}"][f"all_{met}"][m][0] for k in KS]); e = np.array([res[f"z{k}_n{c}"][f"all_{met}"][m][1] for k in KS])
            ax.errorbar(KS, y, yerr=e, marker="o", color=col[m], capsize=3, label=lab[m])
        ax.plot(KS, [res[f"z{k}_n{c}"][f"all_{met}"]["base"] for k in KS], marker="s", ls="--", color=col["base"], label=lab["base"])
        ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels([f"×{k}" for k in KS])
        ax.set_title(f"{'F1@0.5' if met == 'f05' else 'F1@0.7'}，噪声 c={c}", fontsize=10); ax.grid(alpha=0.3)
        if j == 0: ax.set_ylabel("6 套宏平均（20 个测试体积）")
        if i == 1: ax.set_xlabel("z 向降采样倍数")
axes[0, 0].legend(fontsize=8, loc="lower left")
fig.suptitle("G4：接近各向同性的测试体积在 z 降采样与加噪下的稳健性（S1、二维为规则 T；只报告）", fontsize=11)
fig.tight_layout(); fig.savefig(f"{O}/g4_fig.png", dpi=150); print("PNG", f"{O}/g4_fig.png")
