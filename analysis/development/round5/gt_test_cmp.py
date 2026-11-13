"""测试集补充结果（gt 协议、全部 227 个测试体积、去 bbbc032、cp-2_vol0.15、pooled）：三维（A1、S1）对二维与原版，按种子配对。
二维与原版的 mesospim 一行换成按源数据体素重评的结果（meso_fp 的 *_gt_test），三维的测试集评测本身就在新体素下跑。
用法：python gt_test_cmp.py 名字=种子0目录,种子1000目录,种子2000目录 ...（目录为 results/ 下的 gt 协议测试结果目录名）"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import json, sys, statistics as st
R = f"{_NSP_ROOT}/results/"; CELL = "cp-2_vol0.15"
MESO = R + "orx_4349ddbc-f2c2-4119-8162-fa0dd9f75fae_meso_fp/"
TWO = ["orx_26044a2f-1198-47ed-86e5-5d514a05d00f_gttest", "orx_440f0cf2-ee6b-436f-bf09-fb634a50db53_gttest", "orx_0021daf0-5778-4632-a567-4cec8bff207f_gttest"]
BASE = "orx_acbcea5a-0ec3-49c6-8123-d84813624333_gttest"


def per(d, meso_from=None):
    P = json.load(open(R + d + "/summary.json"))["cells"][CELL]["per_ds"]
    P = {k: dict(v) for k, v in P.items() if k != "bbbc032"}
    if meso_from:
        m = json.load(open(meso_from + "/summary.json"))["cells"][CELL]["per_ds"]["mesospim"]
        P["mesospim"] = dict(m)
    return P


def mac(P, k, ex=()):
    ks = [d for d in P if d not in ex]; return sum(P[d][k] for d in ks) / len(ks)


D2 = [per(TWO[i], MESO + f"orx_v5_2d_s{s}_gt_test") for i, s in enumerate((0, 1000, 2000))]
B = per(BASE, MESO + "base_gt_test")
print(f"数据集 {len(B)} 套：{sorted(B)}")
print("二维三种子 13 套 F1@0.5 " + " ".join("%.4f" % mac(p, "f1_05") for p in D2))
for arg in sys.argv[1:]:
    nm, ds = arg.split("="); X = [per(d) for d in ds.split(",")]
    print(f"\n== {nm}")
    for ex, tag in (((), "13 套"), (("mesospim",), "去 mesospim")):
        for k in ("f1_05", "f1_07"):
            dx = [mac(x, k, ex) - mac(y, k, ex) for x, y in zip(X, D2)]
            m, s = st.mean(dx), st.stdev(dx)
            print(f"  {tag:10s} {k}：{nm} {' '.join(f'{mac(x, k, ex):.4f}' for x in X)} 二维 {' '.join(f'{mac(y, k, ex):.4f}' for y in D2)} 原版 {mac(B, k, ex):.4f}"
                  f" | {nm}−二维 {' '.join(f'{v:+.4f}' for v in dx)} 均值 {m:+.4f} SD {s:.4f} {'过' if m > 2 * s and m >= 0.01 else '未过'}确认规则"
                  f" | {nm}−原版 {st.mean(mac(x, k, ex) for x in X) - mac(B, k, ex):+.4f}")
    print("  逐数据集三种子平均 F1@0.5/F1@0.7：" + nm + " | 二维 | 原版")
    w = [0, 0, 0, 0]
    for d in sorted(B):
        a5 = st.mean(x[d]["f1_05"] for x in X); a7 = st.mean(x[d]["f1_07"] for x in X)
        b5 = st.mean(y[d]["f1_05"] for y in D2); b7 = st.mean(y[d]["f1_07"] for y in D2)
        w[0] += a5 > b5; w[1] += a7 > b7; w[2] += a5 > B[d]["f1_05"]; w[3] += a7 > B[d]["f1_07"]
        print(f"    {d:13s} {a5:.3f}/{a7:.3f} | {b5:.3f}/{b7:.3f} | {B[d]['f1_05']:.3f}/{B[d]['f1_07']:.3f} | 对二维 {a5 - b5:+.3f}/{a7 - b7:+.3f}")
    print(f"  胜二维 F1@0.5 {w[0]}/{len(B)}，F1@0.7 {w[1]}/{len(B)}；胜原版 {w[2]}/{len(B)}，{w[3]}/{len(B)}")
