import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
# 第四轮预定选择规则：R3-P1 与 A1 三种子配对（gt 协议、验证集 88 个体积、13 套、视图规则、cp-2_vol0.15）。
# 规则（AGENTS.md）：P1 至少在一个阈值上过确认规则（三种子差值均值 > 2×SD 且 ≥ 0.01），另一个阈值均值不为负，才替换 A1；否则用 A1。
# 用法：python paired_p1.py <P1 种子 2000 单视图 gtval 目录名> <P1 种子 2000 三视图 gtval 目录名>
import json, sys, collections, statistics as st
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/development'))
R = f"{_NSP_ROOT}/results/"; cell = "cp-2_vol0.15"; THR = 5.0
idx = json.load(open(f"{_NSP_SCRATCH}/cpft_iso30/index.json"))
up = collections.defaultdict(list)
for e in idx: up[e["ds"]].append(max(e["factors"]))
up = {k: st.median(v) for k, v in up.items()}
def per(d): return json.load(open(R + d + "/summary.json"))["cells"][cell]["per_ds"]
def rule(v1, v3):
    V1, V3 = per(v1), per(v3); return {d: (V1 if up[d] > THR else V3)[d] for d in V1}
def mac(D, k, ex=()):
    ks = [d for d in D if d not in ex]; return sum(D[d][k] for d in ks) / len(ks)
A1 = [rule("orx_f9a4da66-77b6-440c-adcf-f62ac279a144_gtval", "orx_c7a77c61-bb5f-436c-942c-678787765d71_gtval"),
      rule("orx_b8dbb6aa-201f-4ea1-ac91-4e645e793e47_gtval", "orx_15078456-6f8c-442b-a2f9-22e7c655313a_gtval"),
      rule("orx_f76e7145-a114-40b9-b8e3-2c234cdc44a0_gtval", "orx_34192d05-fc09-4d1a-ad04-46bf9863e5ba_gtval")]
P1 = [rule("orx_fbaafe76-8a3f-4a3b-85cb-6fdfbce12d39_gtval", "orx_11d1c33f-a5d3-42a6-b266-4099149dea02_gtval"),
      rule("orx_8574f6d0-461e-401f-952f-1fc602dc6f42_gtval", "orx_cbd85f25-8e04-439f-a02c-5ac6dc314f70_gtval"),
      rule(sys.argv[1], sys.argv[2])]
D2 = [per("orx_2e0880e4-2e2c-408b-9d7e-397c60cd882e_gtval_orx_v5_2d_s" + s) for s in ("0", "1000", "2000")]
res = {}
for ex, tag in (((), "全13套"), (("mesospim",), "去mesospim")):
    for k in ("f1_05", "f1_07"):
        print(f"== {tag} {k}")
        for nm, L in (("2D", D2), ("A1", A1), ("P1", P1)):
            print(f"  {nm:4s} " + " ".join(f"{mac(x, k, ex):.4f}" for x in L) + f"  均值 {st.mean(mac(x, k, ex) for x in L):.4f}")
        for nm, X, Y in (("P1-A1", P1, A1), ("A1-2D", A1, D2), ("P1-2D", P1, D2)):
            d = [mac(x, k, ex) - mac(y, k, ex) for x, y in zip(X, Y)]; m, s = st.mean(d), st.stdev(d)
            ok = m > 2 * s and m >= 0.01; res[(tag, k, nm)] = (m, ok)
            print(f"  {nm:6s} " + " ".join(f"{x:+.4f}" for x in d) + f"  均值 {m:+.4f} SD {s:.4f}  确认规则{'通过' if ok else '未过'}")
a5, a7 = res[("全13套", "f1_05", "P1-A1")], res[("全13套", "f1_07", "P1-A1")]
choose = (a5[1] and a7[0] >= 0) or (a7[1] and a5[0] >= 0)
print(f"\n预定规则（全 13 套）：P1−A1 F1@0.5 均值 {a5[0]:+.4f}（{'过' if a5[1] else '未过'}），F1@0.7 均值 {a7[0]:+.4f}（{'过' if a7[1] else '未过'}）"
      f" → 选 {'R3-P1' if choose else 'A1'}")
dss = sorted(A1[0])
print("逐数据集三种子平均 F1@0.5/F1@0.7：A1 | P1 | 2D")
for d in dss:
    f = lambda L, k: st.mean(x[d][k] for x in L)
    print(f"  {d:13s} {f(A1,'f1_05'):.3f}/{f(A1,'f1_07'):.3f} | {f(P1,'f1_05'):.3f}/{f(P1,'f1_07'):.3f} | {f(D2,'f1_05'):.3f}/{f(D2,'f1_07'):.3f}")
