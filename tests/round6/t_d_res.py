"""测试 d：打印各数据集训练块与推理时的分辨率条件 f = (f_z, f_y, f_x)（一个源采样间隔 = 多少网络体素）。
训练：PatchSource(res_cond=True) 用真实的训练体积与抽样份额，不读盘（只看增强参数，同 r5/rngcheck.py），每种配置 N 块，
      f 由 data3d.block_res 给出（含置换、缩放、z 拉伸、xy 旋转、各向异性模拟）。
推理：88 个验证体积用真值直径（形状、z/xy、直径取自 A1 种子 0 的 gt 协议结果 f9a4da66 的 rows.json，这些量与模型无关），
      按 eval_gt + Model3D.eval 的取整规则算 f（r6common.expected_res，测试 f 里与 eval_gt 实际记录的值核对）；
      mesoSPIM c1–c5 按源数据体素 5/10.52/10.52 现读标签算真值直径；完整流程（--diam_from，二维 r2a2 的 d_used）用第四轮
      验证集检查 5fc51f6c 的 rows.json（36 个体积）。
用法：python t_d_res.py [N]（CPU）"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import os
import sys
from collections import defaultdict

os.getpid = lambda: 4242
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import data3d  # noqa: E402
from r6common import CONFIGS, train_source, expected_res  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
data3d._read = lambda path, sl: 0
# 不读盘时块内容无关紧要：让 data3d 里的 np.zeros 只建 1×1×1 的数组（否则每块要清零约 70 MB，36000 块要半小时）
import types  # noqa: E402
_np = types.SimpleNamespace(**{k: getattr(np, k) for k in dir(np) if not k.startswith("__")})
_np.zeros = lambda shape, dtype=float: np.zeros((1, 1, 1), dtype)
data3d.np = _np


def fmt(a):
    a = np.asarray(a)
    return f"{np.median(a):6.2f} [{np.percentile(a, 5):5.2f},{np.percentile(a, 95):5.2f}]"


cfgs = dict(CONFIGS)
cfgs["X1abs"] = dict(aniso=dict(CONFIGS["X1"]["aniso"], abs=1))
train_med = {}
for name in ("A1", "X1", "X1abs", "X3", "perm", "ALL"):
    ps, idx = train_source(data3d, cfgs[name], res_cond=True)
    ps._vol = lambda i: ((64, 512, 512), np.zeros((0, 3), int))     # 不读盘
    it = iter(ps)
    by = defaultdict(list); an = defaultdict(list); nan = defaultdict(int)
    for _ in range(N):
        s = next(it); a = s["aug"]; r = data3d.block_res(a)
        by[s["ds"]].append(r)
        if "aniso" in a:
            nan[s["ds"]] += 1
            ax = (0,) if a["aniso"][2] == "z" else (1, 2)
            base = list(a["res"])
            if a["rot"] % 2:
                base[1], base[2] = base[2], base[1]
            an[s["ds"]].append((max(r[k] for k in ax), max(base[k] for k in ax), a["aniso"][0]))
    print(f"\n== 训练块 {name}（{N} 块，{cfgs[name] or '无增强选项'}）：各数据集 f 的中位数 [5%, 95%]")
    print(f"  {'数据集':12s} {'块数':>5s}   {'f_z':^20s} {'f_y':^20s} {'f_x':^20s}" + ("   各向异性块：比例、被模拟轴 f 中位/最大（原有 g 中位）" if "aniso" in cfgs[name] else ""))
    for ds in sorted(by):
        R = np.array(by[ds])
        line = f"  {ds:12s} {len(R):5d}   {fmt(R[:, 0])} {fmt(R[:, 1])} {fmt(R[:, 2])}"
        if "aniso" in cfgs[name] and an[ds]:
            A = np.array(an[ds])
            line += f"   {nan[ds] / len(R):.2f}、{np.median(A[:, 0]):5.2f}/{A[:, 0].max():5.2f}（{np.median(A[:, 1]):4.2f}）"
        print(line)
        if name == "A1":
            train_med[ds] = np.median(R, 0)

# ---- 推理：验证集 gt 协议
R0 = f"{_NSP_ROOT}/results"
rows = json.load(open(f"{R0}/orx_f9a4da66-77b6-440c-adcf-f62ac279a144_gtval/rows.json"))
vol = {(r["ds"], r["id"]): r for r in rows}
by = defaultdict(list); nets = {}
for (ds, id_), r in sorted(vol.items()):
    if ds == "mesospim":
        continue                                                    # 这次结果里是旧体素，下面按源数据体素重算
    f, net = expected_res(r["shape"], r["ratio"], r["d_gt"])
    by[ds].append(f); nets[(ds, id_)] = net
print("\n== 推理（验证集 gt 协议，真值直径）：各数据集体积 f 的中位数 [最小, 最大]；最后一列是训练块 A1 的中位数（z,y,x）")
for ds in sorted(by):
    F = np.array(by[ds])
    t = train_med.get(ds)
    print(f"  {ds:12s} n={len(F):3d}  " + "  ".join(f"{np.median(F[:, k]):6.2f} [{F[:, k].min():5.2f},{F[:, k].max():5.2f}]" for k in range(3))
          + (f"   训练 {t[0]:5.2f} {t[1]:5.2f} {t[2]:5.2f}" if t is not None else ""))

# mesoSPIM 开发集 c1–c5：源数据体素
from stddata import load_manifest, load  # noqa: E402
from eval_cpsam import eq_diam_xy_px  # noqa: E402
man = {r["id"]: r for r in load_manifest(ds="mesospim", split="train", dim="3d")}
print("\n== 推理（mesoSPIM c1–c5，源数据体素 5/10.52/10.52，真值直径）")
for c in ("c1", "c2", "c3", "c4", "c5"):
    r = man[c]
    vox = (5.0, 10.52, 10.52)
    _, gt = load(r, image=False)
    d = eq_diam_xy_px(gt, vox)
    f, net = expected_res(gt.shape, vox[0] / vox[1], d)
    print(f"  mesospim {c}  源 {gt.shape} 真值直径 {d:5.2f} px  z/xy {vox[0] / vox[1]:.3f}  网络网格 {net}  f = ({f[0]:.2f}, {f[1]:.2f}, {f[2]:.2f})"
          f"  清单体素 {r['voxel']}")

# 完整流程（--diam_from，二维 r2a2 的 d_used）
rows = json.load(open(f"{R0}/orx_5fc51f6c-2c54-4acc-9338-4f28257a5b9b_fpval/rows.json"))
vol = {(r["ds"], r["id"]): r for r in rows}
by = defaultdict(list); rel = defaultdict(list)
for (ds, id_), r in sorted(vol.items()):
    f, _ = expected_res(r["shape"], r["ratio"], r["d_used"])
    fg, _ = expected_res(r["shape"], r["ratio"], r["d_gt"])
    by[ds].append(f); rel[ds].append(f[1] / fg[1])
print("\n== 推理（完整流程，--diam_from 二维 r2a2 的 d_used，验证集每套至多 4 个；mesospim 这一行是 11-02 改清单前的旧体素 0.16）")
for ds in sorted(by):
    F = np.array(by[ds])
    print(f"  {ds:12s} n={len(F):2d}  " + "  ".join(f"{np.median(F[:, k]):6.2f} [{F[:, k].min():5.2f},{F[:, k].max():5.2f}]" for k in range(3))
          + f"   f_xy(d_used)/f_xy(真值直径) 中位 {np.median(rel[ds]):.3f}")
print("T_D_DONE")
