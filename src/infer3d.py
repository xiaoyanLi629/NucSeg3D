"""真三维模型的推理接口（true3d_spec §2）。

    m = Model3D("$NUCSEG3D_ROOT/models/models3d/<name>")
    masks, (_, dP, cellprob), _ = m.eval(img, do_3D=True, z_axis=0, diameter=d, anisotropy=an, flow3D_smooth=1.0,
                                         compute_masks=False)
    masks = m._compute_masks((*img.shape, 1), dP, cellprob, cellprob_threshold=-2, niter=..., do_3D=True)

与 CellposeModel.eval(do_3D=True) 相同的约定：先整体积按 1–99 百分位归一化（cellpose normalize_img，norm3D），
再按 30/diameter 缩放、z 再乘 anisotropy（cellpose transforms.resize_image_3d，同样的取整）；网络在缩放后的
各向同性空间里滑窗（z 窗 = 训练块厚度，xy 256，带重叠与渐变权重）；输出缩回输入网格。网络输出 5×单位流与单个 logit，
为与 run_3D“三个方向相加”的量纲一致，dP ×2、cellprob ×3。flow3D_smooth>0 时在输入网格上对 dP 做高斯平滑（同 Cellpose）。
_compute_masks 直接调用 Cellpose 的 CellposeModel._compute_masks（不改动）。
eval 返回 (masks, [None, dP, cellprob], None)，与 CellposeModel.eval 的 (masks, flows, styles) 同位置，
所以 eval_val 里 `model.eval(...)[1]` 取到的 fl[1]、fl[2] 就是 dP、cellprob。

round 2 选项（默认关闭，行为与旧版相同）：
  views=3   三视图：体积的三个轴轮流充当模型的 z（VIEW_PERMS），各自滑窗后转回 (Z,Y,X)、流分量按轴对应重排，
            三份原始输出（5×单位流、logit）取平均，再走同样的缩回、×2/×3、平滑。
  view_fuse=名（round 3，只配合 views=3，默认 None = 等权平均、旧行为）：各向异性感知的视图融合，无手工阈值。
            每个体积轴的插值放大量 b_a = max(f_a, 1)，f_a = 原始采样 -> 网络网格的放大倍数（含 eval_gt 事先做的 z 插值，
            由 eval 的 src_zoom 传入）。模型的二维编码器逐层看“视图平面”（模型 y、x 两轴），训练时这个平面内两轴模糊程度
            总相同；视图 p 的平面内模糊各向异性 ρ_p = max(b)/min(b)（取 p[1]、p[2] 两轴），恒等视图 ρ=1。
              aniso1 / aniso2：整视图权重 w_p = ρ_p^-1 / ρ_p^-2，四个通道共用，归一化后加权平均；
              comp1 / comp2：按分量：流分量 j 在把轴 j 当模型 z（穿平面、由三维解码器负责）的视图里权重 1，
                             其余视图用 ρ_p^-γ；cellprob 用整视图权重。各通道分别归一化。
  ens2d=名  与二维模型集成：二维 CellposeModel.eval(do_3D=True) 在同一输入、同一参数下的 (dP, cellprob)
            （run_3D：dP≈两个方向之和≈10·u，cp=三个 logit 之和）与本模型 (dP×2≈10·u, cp×3) 加权平均：
            ens_alpha·三维 + (1−ens_alpha)·二维（与 $NUCSEG3D_SCRATCH/ana_err/ens.py 的 "mean" 相同，alpha=0.5）。
round 5 选项（默认关闭）：
  view_perm=(p0,p1,p2)  单视图换切片方向（只配合 views=1）：体积按 perm 转置后滑窗，模型 z = 体积轴 p0，编码器逐层看
            体积 (p1, p2) 平面；输出按 from_view 转回 (Z,Y,X)。(0,1,2) 与默认相同；(1,0,2) 看 zx 平面，(2,0,1) 看 zy 平面。
round 6：
  分辨率条件（model3d.json 的 cfg 里 cond="res" 时自动启用，旧模型没有这个键、行为不变）：体积各轴
            f_a = 网络网格尺寸 / 输入尺寸 × src_zoom_a（调用方事先做过的插值放大，eval_gt 的 z 插值；与 view_fuse 的 factors
            同一个量），即“一个源采样间隔 = 多少网络体素”；每个视图把 f 按视图的轴置换后交给网络（模型轴 k = 体积轴 perm[k]）。
            cond_null=True 时改用网络的“未知”嵌入（诊断用）。最近一次的体积 f 记在 last_res。
  tta_flips="yx"|"zyx"（默认 None = 关闭）：每个视图里对模型 y、x（及 z）翻转的所有组合各滑窗一次，翻回后对应流分量变号，
            取平均；与三视图、view_perm、view_fuse 可以叠加。
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import itertools
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net3d import build_from_dir  # noqa: E402

MODELS2D = f"{_NSP_ROOT}/models/models"


def load_model(path_or_name, device=None, **kw):
    """评测用：二维权重名 -> CellposeModel；含 model3d.json 的目录 -> Model3D（kw: views/ens2d/ens_alpha）。"""
    if os.path.isdir(path_or_name) and os.path.exists(os.path.join(path_or_name, "model3d.json")):
        return Model3D(path_or_name, device=device, **kw)
    from cellpose import models
    p = path_or_name if os.path.exists(path_or_name) else f"{MODELS2D}/{path_or_name}"
    return models.CellposeModel(gpu=True, pretrained_model=p, device=device)


def _starts(L, w, ov):
    if L <= w:
        return [0]
    n = int(math.ceil((L - w) / (w * (1 - ov)))) + 1
    return sorted({int(round(s)) for s in np.linspace(0, L - w, n)})


# 三视图：perm[k] = 模型第 k 轴对应的体积轴；每个体积轴恰好当一次模型的 z
VIEW_PERMS = [(0, 1, 2), (1, 0, 2), (2, 0, 1)]


def to_view(y, perm):
    """体积布局的 (C,Z,Y,X) 输出（前 3 通道为 (dz,dy,dx)）-> 视图布局。只用于自检（真值流来回变换）。"""
    perm = tuple(perm)
    o = np.transpose(y, (0, *(p + 1 for p in perm)))
    if o.shape[0] >= 3:
        o = np.concatenate([o[list(perm)], o[3:]], 0)                 # 视图第 k 分量 = 体积第 perm[k] 分量
    return o


def from_view(y, perm):
    """视图布局的网络输出 (C,·,·,·)（前 3 通道为模型的 (dz,dy,dx)）-> 体积布局 (C,Z,Y,X)。"""
    perm = tuple(perm)
    inv = tuple(int(i) for i in np.argsort(perm))
    o = np.transpose(y, (0, *(i + 1 for i in inv)))
    if o.shape[0] >= 3:
        o = np.concatenate([o[list(inv)], o[3:]], 0)                  # 体积第 j 分量 = 视图第 inv[j] 分量
    return o


def run_views(fn, v, views=1, pass_perm=False):
    """fn: (Z,Y,X) -> (C,Z,Y,X)。views=1 直接调用；views=3 按 VIEW_PERMS 三个视图各跑一次、转回后取平均。
    pass_perm（round 6）：调用 fn(视图体积, perm)，让 fn 知道视图的轴置换（条件 f 跟着置换）；views=1 时 perm = (0,1,2)。"""
    if views == 1:
        return fn(v, (0, 1, 2)) if pass_perm else fn(v)
    assert views == 3, views
    acc = None
    for p in VIEW_PERMS:
        y = from_view(fn(np.ascontiguousarray(np.transpose(v, p)), *((p,) if pass_perm else ())), p)
        acc = y if acc is None else acc + y
        del y
    return acc / 3.0


VIEW_FUSE = ("aniso1", "aniso2", "comp1", "comp2")


def view_fuse_weights(fuse, factors):
    """round 3：各向异性感知融合的权重。factors = 体积三个轴 (z,y,x) 从原始采样到网络网格的放大倍数。
    返回 (3 视图, 4 通道) 的权重（每个通道在三个视图上和为 1），通道顺序同体积布局 (dz,dy,dx,cp)。"""
    assert fuse in VIEW_FUSE, fuse
    gamma = float(fuse[-1])
    b = np.maximum(np.asarray(factors, np.float64), 1.0)
    W = np.zeros((len(VIEW_PERMS), 4))
    for i, p in enumerate(VIEW_PERMS):
        rho = max(b[p[1]], b[p[2]]) / min(b[p[1]], b[p[2]])
        w = rho ** -gamma
        W[i] = w
        if fuse.startswith("comp"):
            W[i, p[0]] = 1.0                                          # 体积轴 p[0] 在这个视图里是模型 z（穿平面）
    return W / W.sum(0, keepdims=True)


def run_views_weighted(fn, v, W, pass_perm=False):
    """views=3 的加权版：W 为 view_fuse_weights 的 (3,4) 权重。pass_perm 同 run_views。"""
    acc = None
    for i, p in enumerate(VIEW_PERMS):
        y = from_view(fn(np.ascontiguousarray(np.transpose(v, p)), *((p,) if pass_perm else ())), p)
        y *= W[i].astype(np.float32)[:, None, None, None]
        acc = y if acc is None else acc + y
        del y
    return acc


def res_factors(net_shape, in_shape, src_zoom=(1.0, 1.0, 1.0)):
    """round 6：体积 (z,y,x) 各轴“一个源采样间隔 = 多少网络体素”：网络网格尺寸 / 输入尺寸 × 调用方事先做过的插值放大 src_zoom
    （与 view_fuse 的 factors 相同的算法）。"""
    return [net_shape[k] / in_shape[k] * float(src_zoom[k]) for k in range(3)]


TTA_FLIPS = {"yx": (1, 2), "zyx": (0, 1, 2)}


def flip_sets(mode):
    """round 6：翻转 TTA 的轴组合（含不翻转），如 "yx" -> [(), (1,), (2,), (1, 2)]。"""
    ax = TTA_FLIPS[mode]
    return [c for n in range(len(ax) + 1) for c in itertools.combinations(ax, n)]


def _taper(w, multi):
    if not multi:
        return np.ones(w, np.float32)
    return (np.sin(np.pi * (np.arange(w) + 0.5) / w) ** 2 + 1e-3).astype(np.float32)


class Model3D:
    def __init__(self, path, device=None, gpu=True, views=1, ens2d=None, ens_alpha=0.5, view_fuse=None, view_perm=None,
                 tta_flips=None, cond_null=False):
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.views = int(views); self.ens_alpha = float(ens_alpha); self.ens2d = ens2d or None
        self.view_fuse = view_fuse or None; self.last_fuse = None
        self.view_perm = tuple(int(i) for i in view_perm) if view_perm is not None else None
        if self.view_perm is not None and (sorted(self.view_perm) != [0, 1, 2] or self.views != 1 or self.view_fuse is not None):
            raise ValueError(f"view_perm={view_perm} 需要是 (0,1,2) 的排列，且只配合 views=1、不开 view_fuse")
        if self.view_fuse is not None and (self.view_fuse not in VIEW_FUSE or self.views != 3):
            raise ValueError(f"view_fuse={view_fuse} 需要 views=3，且取值于 {VIEW_FUSE}")
        self.tta_flips = tta_flips or None                             # round 6：翻转 TTA
        if self.tta_flips is not None and self.tta_flips not in TTA_FLIPS:
            raise ValueError(f"tta_flips={tta_flips} 取值于 {tuple(TTA_FLIPS)}")
        self.flip_sets = flip_sets(self.tta_flips) if self.tta_flips else None
        self.cond_null = bool(cond_null); self.last_res = None
        self.m2d = None
        if self.ens2d:
            from cellpose import models
            p2 = self.ens2d if os.path.exists(self.ens2d) else f"{MODELS2D}/{self.ens2d}"
            self.m2d = models.CellposeModel(gpu=self.device.type == "cuda", pretrained_model=p2, device=self.device)
        self.net, self.meta = build_from_dir(path, self.device)
        inf = self.meta.get("infer", {})
        self.patch_z = int(inf.get("patch_z", self.meta.get("patch_z", 32)))
        self.overlap_z = float(inf.get("overlap_z", 0.25)); self.overlap_xy = float(inf.get("overlap_xy", 0.25))
        self.variant = self.meta["variant"]
        self.cond = self.net.cfg.get("cond")                           # round 6：None（旧模型）或 "res"
        if self.cond_null and not self.cond:
            raise ValueError("cond_null 只用于分辨率条件模型（model3d.json 的 cfg 里 cond=res）")
        self.backbone = "sam_vitl"
        self.diam_mean = 30.0
        self.pretrained_model = path

    # ------------------------------------------------------------ 网络滑窗
    @torch.no_grad()
    def run_volume(self, v, res=None):
        """v: (Z,Y,X) 归一化、已缩放的体积 -> (4,Z,Y,X) float32 原始网络输出（5×单位流、单个 logit）。
        res（round 6）：条件模型的 (f_z, f_y, f_x)，按 v 的轴（None = 不加条件 / “未知”嵌入）。
        tta_flips（round 6）：对每个翻转组合各滑窗一次，输出翻回、被翻转轴的流分量变号，再平均。"""
        if not self.tta_flips:
            return self._run_volume(v, res)
        acc = None
        for sub in self.flip_sets:
            y = self._run_volume(np.ascontiguousarray(np.flip(v, sub)) if sub else v, res)
            if sub:
                y = np.flip(y, tuple(a + 1 for a in sub)).copy()
                y[list(sub)] *= -1.0                                  # 通道 a = 沿 v 第 a 轴的流分量
            acc = y if acc is None else acc + y
            del y
        return acc / float(len(self.flip_sets))

    @torch.no_grad()
    def _run_volume(self, v, res=None):
        """单次滑窗（旧版 run_volume 的主体）。"""
        Z, Y, X = v.shape
        wz = Z if self.variant == "p25d" else self.patch_z           # 2.5D 逐层计算，z 不用分窗
        win = (wz, 256, 256)
        pads = [max(0, w - L) for w, L in zip(win, (Z, Y, X))]
        pw = [(p // 2, p - p // 2) for p in pads]
        vp = np.pad(v, pw, mode="constant")
        Zp, Yp, Xp = vp.shape
        sz, sy, sx = _starts(Zp, wz, self.overlap_z), _starts(Yp, 256, self.overlap_xy), _starts(Xp, 256, self.overlap_xy)
        wt = (_taper(wz, len(sz) > 1)[:, None, None] * _taper(256, len(sy) > 1)[None, :, None]
              * _taper(256, len(sx) > 1)[None, None, :])
        wt_t = torch.from_numpy(wt).to(self.device)
        acc = torch.zeros((4, Zp, Yp, Xp), dtype=torch.float32)
        ws = torch.zeros((Zp, Yp, Xp), dtype=torch.float32)
        vt = torch.from_numpy(np.ascontiguousarray(vp, dtype=np.float32))
        ck = {} if res is None else dict(res=torch.tensor([list(res)], dtype=torch.float32, device=self.device))   # round 6
        for z0 in sz:
            for y0 in sy:
                for x0 in sx:
                    blk = vt[z0:z0 + wz, y0:y0 + 256, x0:x0 + 256].to(self.device)[None, None]
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                        y = self.net(blk, **ck)[0].float()
                    acc[:, z0:z0 + wz, y0:y0 + 256, x0:x0 + 256] += (y * wt_t).cpu()
                    ws[z0:z0 + wz, y0:y0 + 256, x0:x0 + 256] += wt
        out = (acc / ws.clamp(min=1e-6)).numpy()
        return out[:, pw[0][0]:pw[0][0] + Z, pw[1][0]:pw[1][0] + Y, pw[2][0]:pw[2][0] + X]

    # ------------------------------------------------------------ Cellpose 风格的 eval
    def eval(self, x, batch_size=8, resample=True, channel_axis=None, z_axis=None, normalize=True, rescale=None,
             diameter=None, flow_threshold=0.4, cellprob_threshold=0.0, do_3D=True, anisotropy=None, flow3D_smooth=0,
             stitch_threshold=0.0, min_size=15, max_size_fraction=0.4, niter=None, augment=False, tile_overlap=0.1,
             bsize=None, compute_masks=True, progress=None, **kw):
        from cellpose import models, transforms
        from scipy.ndimage import gaussian_filter
        if not do_3D:
            raise ValueError("Model3D 只支持 do_3D=True")
        x = np.asarray(x)
        if channel_axis is not None and x.ndim == 4:
            x = np.take(np.moveaxis(x, channel_axis, -1), 0, axis=-1)
        if z_axis not in (None, 0):
            x = np.moveaxis(x, z_axis, 0)
        if x.ndim != 3:
            raise ValueError(f"需要 (Z,Y,X) 单通道体积，得到 {x.shape}")
        x4 = x.astype(np.float32)[..., None]
        if normalize:
            params = dict(models.normalize_default)
            if isinstance(normalize, dict):
                params.update(normalize)
            params["normalize"] = True; params["norm3D"] = True
            x4 = transforms.normalize_img(x4, **params)
        rescale = 30.0 / diameter if (diameter is not None and diameter > 0) else 1.0
        Lz, Ly, Lx = x4.shape[:3]
        an = 1.0 if anisotropy is None else anisotropy
        xi = x4
        if rescale != 1.0 or an != 1.0:
            new_shape = (int(Lz * an * rescale), int(Ly * rescale), int(Lx * rescale))
            xi = transforms.resize_image_3d(x4, new_shape, no_channels=False)
        res_vol = None
        if self.cond:                                                  # round 6：体积各轴的源采样间隔（网络体素）
            res_vol = res_factors(xi.shape[:3], x4.shape[:3], kw.get("src_zoom") or (1.0, 1.0, 1.0))
            self.last_res = [round(f, 4) for f in res_vol]
        fn = self._view_fn(res_vol)
        if self.view_perm is not None:                                 # round 5：单视图换切片方向
            p = self.view_perm
            y = from_view(fn(np.ascontiguousarray(np.transpose(xi[..., 0], p)), p), p)
        elif self.view_fuse is None:
            y = run_views(fn, xi[..., 0], self.views, pass_perm=True)  # (4, Z', Y', X')
        else:
            sz = kw.get("src_zoom") or (1.0, 1.0, 1.0)                  # 调用方事先做过的插值放大（eval_gt 的 z 插值）
            fac = [xi.shape[k] / x4.shape[k] * float(sz[k]) for k in range(3)]
            W = view_fuse_weights(self.view_fuse, fac)
            self.last_fuse = dict(factors=[round(f, 3) for f in fac], weights=W.round(4).tolist())
            y = run_views_weighted(fn, xi[..., 0], W, pass_perm=True)
        yf = np.moveaxis(y, 0, -1)
        if resample and (rescale != 1.0 or Lz != yf.shape[0]):
            yf = transforms.resize_image_3d(yf, (Lz, Ly, Lx), no_channels=False)
        dP = np.ascontiguousarray(np.moveaxis(yf[..., :3], -1, 0)) * 2.0
        cellprob = np.ascontiguousarray(yf[..., 3]) * 3.0
        if flow3D_smooth:
            s = flow3D_smooth if isinstance(flow3D_smooth, (list, tuple)) else [flow3D_smooth] * 3
            if len(s) == 1:
                s = list(s) * 3
            if any(v > 0 for v in s):
                dP = gaussian_filter(dP, [0, *s])
        if self.m2d is not None:
            # 二维模型在同一输入（同样的归一化、直径、各向异性、平滑）上的 run_3D 输出，量纲已与上面一致
            o2 = self.m2d.eval(x, batch_size=batch_size, resample=resample, z_axis=0, normalize=normalize, diameter=diameter,
                               do_3D=True, anisotropy=anisotropy, flow3D_smooth=flow3D_smooth, compute_masks=False,
                               augment=augment, tile_overlap=tile_overlap, bsize=bsize)
            al = self.ens_alpha
            dP = al * dP + (1 - al) * o2[1][1]
            cellprob = al * cellprob + (1 - al) * o2[1][2]
            del o2
        masks = np.zeros(0)
        if compute_masks:
            niter_scale = 1 if not resample else rescale
            niter = int(200 / niter_scale) if not niter else niter
            masks = self._compute_masks(x4.shape, dP, cellprob, flow_threshold=flow_threshold,
                                        cellprob_threshold=cellprob_threshold, min_size=min_size,
                                        max_size_fraction=max_size_fraction, niter=niter, do_3D=True).squeeze()
        return masks, [None, dP.astype(np.float32), cellprob.astype(np.float32)], None

    def _view_fn(self, res_vol):
        """round 6：视图函数 fn(视图体积, perm)；模型第 k 轴 = 体积第 perm[k] 轴，条件 f 跟着置换。
        res_vol 为 None（非条件模型）或 cond_null 时不传条件。"""
        def fn(vv, perm=(0, 1, 2)):
            r = None if (res_vol is None or self.cond_null) else tuple(float(res_vol[k]) for k in perm)
            return self.run_volume(vv, res=r)
        return fn

    def _compute_masks(self, shape, dP, cellprob, **kw):
        from cellpose import models
        return models.CellposeModel._compute_masks(self, shape, dP, cellprob, **kw)
