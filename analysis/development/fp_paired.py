import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
# 完整流程三种子配对：每个种子一对 (三维结果目录, 二维结果目录)，用 fp_cmp 的口径（cp-2_vol0.15，只比两边都有的体积，去 bbbc032）。
# 用法：python3 fp_paired.py 三维s0 二维s0 三维s1000 二维s1000 三维s2000 二维s2000 [--base 原版目录]
# 打印每个种子的宏平均差、均值、SD、是否过确认规则（均值 > 2×SD 且 ≥ 0.01），以及逐数据集三种子平均。
import sys, statistics as st
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/development'))
from fp_cmp import load, agg, macros, ONE

args = sys.argv[1:]; base = None
if "--base" in args:
    i = args.index("--base"); base = args[i + 1]; args = args[:i] + args[i + 2:]
pairs = [(args[i], args[i + 1]) for i in range(0, len(args), 2)]
diffs, perA, perB, perC = [], [], [], []
C = load(base) if base else None
for a, b in pairs:
    A, B = load(a), load(b); ks = sorted(set(A) & set(B))
    PA, PB = agg(A, ks), agg(B, ks); MA, MB = macros(PA), macros(PB)
    diffs.append({nm: {k: MA[nm][k] - MB[nm][k] for k in MA[nm]} for nm in MA}); perA.append(PA); perB.append(PB)
    if C:
        kc = sorted(set(A) & set(C)); perC.append(agg(C, kc))
    print(f"种子对 {a[-40:]} vs {b[-40:]}：{len(ks)} 个体积；" + "  ".join(f"{nm} {MA[nm]['vm5'] - MB[nm]['vm5']:+.4f}/{MA[nm]['vm7'] - MB[nm]['vm7']:+.4f}" for nm in MA))
for nm in diffs[0]:
    for k, lab in (("vm5", "F1@0.5 逐体积"), ("vm7", "F1@0.7 逐体积"), ("po5", "F1@0.5 pooled"), ("po7", "F1@0.7 pooled")):
        v = [d[nm][k] for d in diffs]; m = st.mean(v); sd = st.stdev(v) if len(v) > 1 else float("nan")
        ok = len(v) > 1 and m > 2 * sd and m >= 0.01
        print(f"  {nm:9s} {lab:14s} 差 {' '.join(f'{x:+.4f}' for x in v)}  均值 {m:+.4f} SD {sd:.4f}  {'通过' if ok else '未过'}")
dss = sorted(perA[0])
print("逐数据集三种子平均（逐体积口径）：三维 | 二维" + (" | 原版" if C else "") + " | 三维−二维")
for d in dss:
    a5 = st.mean(p[d]["vm5"] for p in perA); a7 = st.mean(p[d]["vm7"] for p in perA)
    b5 = st.mean(p[d]["vm5"] for p in perB); b7 = st.mean(p[d]["vm7"] for p in perB)
    c = f" | {st.mean(p[d]['vm5'] for p in perC):.3f}/{st.mean(p[d]['vm7'] for p in perC):.3f}" if C else ""
    print(f"  {d:13s}{'*' if d in ONE else ' '} {a5:.3f}/{a7:.3f} | {b5:.3f}/{b7:.3f}{c} | {a5 - b5:+.3f}/{a7 - b7:+.3f}")
w5 = sum(st.mean(p[d]["vm5"] for p in perA) > st.mean(p[d]["vm5"] for p in perB) for d in dss)
w7 = sum(st.mean(p[d]["vm7"] for p in perA) > st.mean(p[d]["vm7"] for p in perB) for d in dss)
print(f"三维三种子平均胜二维的数据集：F1@0.5 {w5}/{len(dss)}，F1@0.7 {w7}/{len(dss)}（* 为单视图数据集）")
if C:
    w5 = sum(st.mean(p[d]["vm5"] for p in perA) > st.mean(p[d]["vm5"] for p in perC) for d in dss)
    w7 = sum(st.mean(p[d]["vm7"] for p in perA) > st.mean(p[d]["vm7"] for p in perC) for d in dss)
    print(f"三维三种子平均胜原版的数据集：F1@0.5 {w5}/{len(dss)}，F1@0.7 {w7}/{len(dss)}")
