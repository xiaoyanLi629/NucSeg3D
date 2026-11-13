"""round 6 测试的公共部分：路径、训练配置、按 eval_gt + Model3D.eval 的规则算推理时的 f。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import hashlib

import numpy as np

SRC_OLD = f"{_NSP_SCRATCH}/r6/src_before_cond"
SRC_NEW = _nsp_os.path.join(_NSP_REPO, 'src')
ISO = f"{_NSP_SCRATCH}/cpft_iso30"
MODELS = f"{_NSP_ROOT}/models/models"
A1 = f"{MODELS}/orx_t3d_a1_s0"
S1 = f"{MODELS}/orx_t3d_r5s1_s0"
T = f"{_NSP_SCRATCH}/r6/tests"

# 训练块的增强配置（与 train3d 的命令行一一对应；A1 = 默认、X1/X3 = 第五轮筛选、ALL = 全开）
CONFIGS = {
    "A1": {},
    "X1": dict(aniso=dict(p=0.33, lo=2.0, hi=10.0, xy=1.0, blur=0.3, blur_max=0.6)),
    "X3": dict(zstretch=dict(p=0.33, a=3.0)),
    "perm": dict(perm_aug=0.5),
    "ALL": dict(aniso=dict(p=0.33, lo=2.0, hi=13.0, xy=0.5, blur=0.3, blur_max=0.6), zstretch=dict(p=0.33, a=3.0), perm_aug=0.3),
}


def train_source(data3d, cfg, seed=0, patch=(32, 256, 256), **kw):
    """与 train3d.make_loader 相同的体积、权重、份额（--weights cellos=2,tapenade=2 --caps blastospim=0.25，去掉 mesospim）。"""
    idx = [e for e in data3d.load_index(ISO) if e["split"] == "train" and e["ds"] not in ("mesospim",)]
    probs, share = data3d.dataset_probs(idx, {"cellos": 2.0, "tapenade": 2.0}, {"blastospim": 0.25})
    return data3d.PatchSource(idx, probs, patch, 32, seed=seed, **cfg, **kw), idx


def aug_key(a):
    """增强参数的精确表示（不含 round 6 的 res）。"""
    out = []
    for k in sorted(a):
        if k == "res":
            continue
        v = a[k]
        if k == "flip":
            v = tuple(bool(x) for x in v)
        elif isinstance(v, (tuple, list)):
            v = tuple(float(x) if isinstance(x, (float, np.floating)) else x for x in v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        out.append((k, v))
    return repr(out)


def h_arrays(*arrs):
    h = hashlib.sha256()
    for x in arrs:
        x = np.ascontiguousarray(x)
        h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()


def expected_res(shape, ratio, d):
    """推理时体积 (z,y,x) 的 f：照抄 eval_gt.run_volume（z/xy ≥ 2 先插值到 round(Z·ratio) 层、an=1，否则 an=ratio）与
    Model3D.eval（rescale = 30/d，网格 (int(Lz·an·r), int(Ly·r), int(Lx·r))）的取整规则。"""
    Z, Y, X = shape
    if ratio >= 2:
        Lz, an = int(round(Z * ratio)), 1.0
        zoom = Lz / Z
    else:
        Lz, an, zoom = Z, ratio, 1.0
    r = 30.0 / d
    if r != 1.0 or an != 1.0:
        net = (int(Lz * an * r), int(Y * r), int(X * r))
    else:
        net = (Lz, Y, X)
    return [net[0] / Lz * zoom, net[1] / Y, net[2] / X], net
