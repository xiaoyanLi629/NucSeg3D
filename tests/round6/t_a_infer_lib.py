"""测试用的推理小库：预处理照抄 eval_gt.run_volume（load_case 裁块 384、真值直径、z/xy ≥ 2 先沿 z 插值到 round(Z·ratio) 层），
再调 Model3D.eval(compute_masks=False)，返回 float32 dP / cellprob 的哈希。调用方先把要测的 src 目录放进 sys.path。"""
import hashlib

import numpy as np

_MAN = None


def manifest():
    global _MAN
    if _MAN is None:
        from stddata import load_manifest
        _MAN = {(r["ds"], r["id"]): r for r in load_manifest(split="train", dim="3d")}
    return _MAN


def prep(ds, id_, vox=None):
    from eval_cpsam import load_case, eq_diam_xy_px
    from eval_val import zinterp
    r = manifest()[(ds, id_)]
    img, gt, box, ign = load_case(r, 384)
    if ign is not None:
        gt = gt.copy(); gt[ign] = 0
    vox = vox or r["voxel"] or (1.0, 1.0, 1.0); ratio = vox[0] / vox[1]; d = eq_diam_xy_px(gt, vox)
    if ratio >= 2:
        Zu = int(round(img.shape[0] * ratio))
        im = img.astype(np.float32); lo, hi = np.percentile(im, [1, 99.8]); im = (im - lo) / max(hi - lo, 1e-6)
        img_in, an = zinterp(im, Zu), 1.0
    else:
        img_in, an = img, ratio
    return img_in, an, d, img.shape, ratio


def h(x):
    x = np.ascontiguousarray(np.asarray(x, np.float32))
    return hashlib.sha256(x.tobytes()).hexdigest()[:16]


def run_cases(mdir, cases, device="cuda", keep=False):
    """cases: [(ds, id, Model3D 选项)]。返回 [(描述, dict(dP, cp, res, mean_abs_dP[, out]))]。"""
    from infer3d import Model3D
    models, res = {}, []
    for ds, id_, opts in cases:
        key = tuple(sorted(opts.items()))
        if key not in models:
            models[key] = Model3D(mdir, device=device, **opts)
        m = models[key]
        img_in, an, d, shp, ratio = prep(ds, id_)
        sz = (img_in.shape[0] / shp[0], 1.0, 1.0)
        kw = dict(src_zoom=sz) if getattr(m, "cond", None) else {}      # 与 eval_gt 一样：条件模型才传 src_zoom
        out = m.eval(img_in, do_3D=True, z_axis=0, diameter=d, anisotropy=an, flow3D_smooth=1.0, batch_size=32,
                     compute_masks=False, **kw)
        dP, cp = out[1][1], out[1][2]
        r = dict(dP=h(dP), cp=h(cp), res=getattr(m, "last_res", None), mean_abs_dP=float(np.abs(dP).mean()),
                 shape_in=tuple(img_in.shape), d=d, an=an, ratio=ratio, shape_src=tuple(shp))
        if keep:
            r["out"] = (dP, cp)
        res.append((f"{ds}/{id_} {opts}", r))
    return res
