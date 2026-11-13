"""第五轮 b 确认（AGENTS.md 预定规则）：S1 与 C 三种子配对。
用法：python r5_confirm.py s0:C验证run:S验证run:C_meso_run:S_meso_run s1000:... s2000:...
mesoSPIM：c1–c5 pooled F1@0.5（源数据体素、cp-2_vol0.15）；验证集：去掉 mesospim 的 12 套宏平均。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import sys, statistics as st
src = open(f"{_NSP_SCRATCH}/r5/r5_screen.py").read().split("if __name__")[0]
exec(src)
rows = []
for arg in sys.argv[1:]:
    seed, cv, sv, cm, sm = arg.split(":")
    C, S = per(full(cv)), per(full(sv))
    mc, ms = meso(cm)[0], meso(sm)[0]
    rows.append((seed, ms - mc, mac(S, "f1_05") - mac(C, "f1_05"), mac(S, "f1_07") - mac(C, "f1_07"), ms, mc, mac(S, "f1_05"), mac(C, "f1_05")))
    print(f"种子 {seed}：c1–c5 S {ms:.3f} C {mc:.3f} 差 {ms - mc:+.3f}；12 套 S {mac(S, 'f1_05'):.4f}/{mac(S, 'f1_07'):.4f} C {mac(C, 'f1_05'):.4f}/{mac(C, 'f1_07'):.4f}"
          f" 差 {mac(S, 'f1_05') - mac(C, 'f1_05'):+.4f}/{mac(S, 'f1_07') - mac(C, 'f1_07'):+.4f}")
def summ(i, nm):
    v = [r[i] for r in rows]; m = st.mean(v); s = st.stdev(v) if len(v) > 1 else float("nan")
    return m, s, v
m1, s1, v1 = summ(1, "meso")
m2, s2, v2 = summ(2, "f05"); m3, s3, v3 = summ(3, "f07")
ok_meso = m1 > 2 * s1 and m1 >= 0.05
ok_dom = m2 >= -0.005 and m3 >= -0.005
print(f"c1–c5 配对差 {' '.join(f'{x:+.3f}' for x in v1)} 均值 {m1:+.3f} SD {s1:.3f} → {'通过' if ok_meso else '未过'}（均值 > 2×SD 且 ≥ 0.05）")
print(f"12 套配对差 F1@0.5 {' '.join(f'{x:+.4f}' for x in v2)} 均值 {m2:+.4f}；F1@0.7 {' '.join(f'{x:+.4f}' for x in v3)} 均值 {m3:+.4f} → {'通过' if ok_dom else '未过'}（两个均值都不低于 −0.005）")
print("确认结论：" + ("S1 通过确认，作为第五轮模型" if ok_meso and ok_dom else "S1 未通过确认"))
