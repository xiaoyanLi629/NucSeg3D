"""第五轮筛选/确认的汇总（AGENTS.md 预定规则）。
用法：python r5_screen.py 种子 C验证run X验证run C_meso_run X_meso_run [--a1 A1单视图gtval目录 A1三视图gtval目录]
mesoSPIM：c1–c5 pooled F1（gt 协议、单视图、源数据体素、cp-2_vol0.15），读 results/orx_<run>_meso_diag/meso_diag.json。
验证集：88 个体积视图规则，去掉 mesospim 的 12 套宏平均（per_ds 的 pooled F1），读 results/orx_<run>_gtval/summary.json。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json, sys, collections, statistics as st
R = f"{_NSP_ROOT}/results/"; CELL = "cp-2_vol0.15"


def full(run):
    import glob
    m = glob.glob(R + f"orx_{run}*_gtval") or glob.glob(R + f"orx_{run}*_meso_diag")
    assert len(m) == 1, (run, m)
    return m[0]


def per(d):
    return json.load(open(d + "/summary.json"))["cells"][CELL]["per_ds"]


def meso(run, name=None):
    J = json.load(open(full(run) + "/meso_diag.json"))
    k = name or next(iter(J))
    return J[k]["cp-2_vol0.15"]


def mac(D, k):
    ks = [d for d in D if d != "mesospim"]; return sum(D[d][k] for d in ks) / len(ks)


def rule(v1, v3):
    idx = json.load(open(f"{_NSP_SCRATCH}/cpft_iso30/index.json"))
    up = collections.defaultdict(list)
    for e in idx: up[e["ds"]].append(max(e["factors"]))
    up = {k: st.median(v) for k, v in up.items()}
    V1, V3 = per(R + v1), per(R + v3); return {d: (V1 if up[d] > 5 else V3)[d] for d in V1}


if __name__ == "__main__":
    a = sys.argv[1:]; seed = a[0]
    Cv, Xv = per(full(a[1])), per(full(a[2]))
    Cm, Xm = meso(a[3], a[5] if len(a) > 5 and not a[5].startswith("--") else None), meso(a[4])
    print(f"== 种子 {seed}")
    print(f"mesoSPIM c1–c5（源数据体素）F1@0.5/0.7：C {Cm[0]:.3f}/{Cm[1]:.3f}  X {Xm[0]:.3f}/{Xm[1]:.3f}  X−C {Xm[0] - Cm[0]:+.3f}/{Xm[1] - Cm[1]:+.3f}")
    print(f"验证集 12 套（去 mesospim）宏平均：C {mac(Cv, 'f1_05'):.4f}/{mac(Cv, 'f1_07'):.4f}  X {mac(Xv, 'f1_05'):.4f}/{mac(Xv, 'f1_07'):.4f}"
          f"  X−C {mac(Xv, 'f1_05') - mac(Cv, 'f1_05'):+.4f}/{mac(Xv, 'f1_07') - mac(Cv, 'f1_07'):+.4f}")
    if "--a1" in a:
        i = a.index("--a1"); A = rule(a[i + 1], a[i + 2])
        print(f"  A1 同种子（视图规则）12 套：{mac(A, 'f1_05'):.4f}/{mac(A, 'f1_07'):.4f}  X−A1 {mac(Xv, 'f1_05') - mac(A, 'f1_05'):+.4f}/{mac(Xv, 'f1_07') - mac(A, 'f1_07'):+.4f}")
    print("逐数据集 F1@0.5（X−C）：" + "  ".join(f"{d} {Xv[d]['f1_05'] - Cv[d]['f1_05']:+.3f}" for d in sorted(Cv) if d != "mesospim"))
