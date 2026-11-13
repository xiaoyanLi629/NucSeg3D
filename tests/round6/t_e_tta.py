"""测试 e：翻转 TTA。
1) 模拟网络（严格翻转等变：流 = 输入的中心差分梯度、两端复制填充，cellprob = 输入），随机输入：
   单窗口体积 (32,256,256) 上，对 z、y、x 各单独翻转：翻转输入 → 滑窗 → 输出翻回、该轴流分量变号，应与直接推理逐位相同；
   不变号时该分量应正好差一个负号。tta_flips=yx / zyx 的平均也应与直接推理相同（只差浮点舍入）。
   多窗口体积 (70,300,290)：滑窗起点与填充在翻转后不完全对称，报告差值大小。
2) 真实网络（S1 种子 0）在验证体积 bbbc050 train_Emb11_t501 的网络网格输入上：单独翻 y / x / z 再翻回，与直接推理按通道
   求相关；变号正确时各流分量应强正相关，不变号时被翻轴的分量应强负相关。再用 Model3D(tta_flips=...) 走完整 eval。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import os
import sys

sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from infer3d import Model3D, flip_sets  # noqa: E402
from r6common import S1  # noqa: E402


class MockNet(torch.nn.Module):
    """严格翻转等变的“网络”：(dz,dy,dx) = 输入沿三个轴的中心差分（两端复制填充），cellprob = 输入。"""
    def forward(self, x, res=None):
        v = x[:, 0]
        outs = []
        for ax in (1, 2, 3):
            n = v.shape[ax]
            p = torch.cat([v.narrow(ax, 0, 1), v, v.narrow(ax, n - 1, 1)], ax)
            outs.append((p.narrow(ax, 2, n) - p.narrow(ax, 0, n)) / 2)
        return torch.stack(outs + [v], 1)


class RecNet(MockNet):
    """记下每次前向收到的 f（条件模型的推理接口）。"""
    def __init__(self):
        super().__init__(); self.seen = []

    def forward(self, x, res=None):
        self.seen.append(None if res is None else tuple(round(float(v), 4) for v in res[0]))
        return super().forward(x)


def mock_model(tta=None, cond=None, **kw):
    m = Model3D.__new__(Model3D)
    m.device = torch.device("cuda"); m.variant = "dec3d"; m.patch_z = 32; m.overlap_z = 0.25; m.overlap_xy = 0.25
    m.net = RecNet() if cond else MockNet(); m.tta_flips = tta; m.flip_sets = flip_sets(tta) if tta else None
    m.cond = cond; m.cond_null = False; m.last_res = None
    m.views = 1; m.view_perm = None; m.view_fuse = None; m.last_fuse = None; m.m2d = None; m.ens_alpha = 0.5
    for k, v in kw.items():
        setattr(m, k, v)
    return m


def unflip(y, axes, sign=True):
    y = np.flip(y, tuple(a + 1 for a in axes)).copy()
    if sign:
        y[list(axes)] *= -1.0
    return y


rng = np.random.default_rng(0)
m0 = mock_model()
for shape in ((32, 256, 256), (70, 300, 290)):
    v = rng.random(shape, dtype=np.float32)
    y0 = m0.run_volume(v)
    for a in (0, 1, 2):
        yf = m0._run_volume(np.ascontiguousarray(np.flip(v, a)))
        d_ok = float(np.abs(unflip(yf, (a,)) - y0).max())
        d_bad = float(np.abs(unflip(yf, (a,), sign=False)[a] + y0[a]).max())
        print(f"T_e 模拟网络 {shape} 翻转轴 {'zyx'[a]}：翻回+变号 与直接推理最大差 {d_ok:.3e}；不变号时该分量 = −直接推理（最大差 {d_bad:.3e}），"
              f"|分量| 均值 {float(np.abs(y0[a]).mean()):.3f}", flush=True)
    for t in ("yx", "zyx"):
        yt = mock_model(t).run_volume(v)
        print(f"T_e 模拟网络 {shape} tta_flips={t}（{len(flip_sets(t))} 次）：与直接推理最大差 {float(np.abs(yt - y0).max()):.3e}", flush=True)

# ---- 条件模型的推理接口：三视图 / view_perm 时交给网络的 f 跟着轴置换，翻转不改 f
# 体积 (20,64,64)，直径 15（rescale 2）、anisotropy 2、src_zoom x 1.5 -> 体积 f = (4, 2, 3)
vol = rng.random((20, 64, 64), dtype=np.float32)
for kw in (dict(views=1), dict(views=3), dict(view_perm=(1, 0, 2)), dict(views=3, tta="yx")):
    tta = kw.pop("tta", None)
    mr = mock_model(tta=tta, cond="res", **kw)
    mr.eval(vol, diameter=15.0, anisotropy=2.0, compute_masks=False, src_zoom=(1.0, 1.0, 1.5))
    seen = list(dict.fromkeys(mr.net.seen))
    print(f"T_e 条件接口 {kw}{' tta=' + tta if tta else ''}：体积 f = {mr.last_res}，网络收到的 f（按出现顺序去重）= {seen}", flush=True)
print("    期望：单视图 [(4,2,3)]；三视图 VIEW_PERMS (0,1,2)/(1,0,2)/(2,0,1) -> [(4,2,3), (2,4,3), (3,4,2)]；view_perm 102 -> [(2,4,3)]")

# ---- 真实网络
from cellpose import models as cpm, transforms  # noqa: E402
from t_a_infer_lib import prep, run_cases  # noqa: E402
img_in, an, d, shp, ratio = prep("bbbc050", "train_Emb11_t501")
x4 = img_in.astype(np.float32)[..., None]
prm = dict(cpm.normalize_default); prm["normalize"] = True; prm["norm3D"] = True
x4 = transforms.normalize_img(x4, **prm)
r = 30.0 / d
xi = transforms.resize_image_3d(x4, (int(x4.shape[0] * an * r), int(x4.shape[1] * r), int(x4.shape[2] * r)), no_channels=False)[..., 0]
m = Model3D(S1, device="cuda")
y0 = m._run_volume(np.ascontiguousarray(xi))
print(f"真实网络 S1 种子 0，bbbc050 train_Emb11_t501 网络网格输入 {xi.shape}", flush=True)


def corr(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1])


for a in (1, 2, 0):
    yf = m._run_volume(np.ascontiguousarray(np.flip(xi, a)))
    ok = unflip(yf, (a,)); bad = unflip(yf, (a,), sign=False)
    print(f"T_e 真实网络 翻 {'zyx'[a]} 再翻回：与直接推理的相关（dz, dy, dx, cp）= " +
          ", ".join(f"{corr(ok[c], y0[c]):.3f}" for c in range(4)) + f"；不变号时 {'zyx'[a]} 分量的相关 = {corr(bad[a], y0[a]):.3f}", flush=True)
for opts in (dict(tta_flips="yx"), dict(tta_flips="zyx"), dict(tta_flips="yx", views=3)):
    (k, r1), = run_cases(S1, [("bbbc050", "train_Emb11_t501", opts)], keep=True)
    dP, cp = r1["out"]
    print(f"T_e Model3D.eval {opts}：dP {dP.shape} 有限 = {bool(np.isfinite(dP).all() and np.isfinite(cp).all())}，|dP| 均值 {r1['mean_abs_dP']:.4f}", flush=True)
print("T_E_DONE")
