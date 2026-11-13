"""nc G 系列评测入口：在 eval_gt.py（不改动）外面加几个开关，其余（选样、裁块、网格、跟流、错误分类、汇总、输出）全部照 eval_gt。
设计见 $NUCSEG3D_SCRATCH/nc/gpu/DESIGN.md。用法与 eval_gt.py 相同，另加：
  --gx_iso on|off        off = 关掉“各向同性跟流”（z/xy ≥ 2 的体积不再事先沿 z 插值到各向同性，而是把 anisotropy=z/xy 交给模型，
                         网络输出缩回原网格后在原网格上跟流；即 Cellpose 原生做法）。默认 on = eval_gt 原行为。（G6）
  --gx_zdec k            z 向降采样：图像、真值、未定义区域都取每 k 层的第 1 层（从第 0 层起），体素 z 乘 k。（G4）
  --gx_noise c           在（降采样后的）原始强度上加高斯噪声，σ = c ×（真值核内中位强度 − 核外中位强度）；
                         随机数种子 = crc32("ds/id/k/c")，所以同一体积同一条件下各模型拿到的噪声逐体素相同。（G4）
  --gx_keys 文件         只跑文件里列出的体积（每行 ds__id），其余体积直接跳过（不载入、不计入汇总）。
  --gx_min_ratio r       只跑（原始体素）z/xy ≥ r 的体积，其余跳过。（G6 测试集只跑会受开关影响的体积）
  --gx_zero_dec          （M1 消融，2026-11-08 加）真三维模型载入后把三维残差解码器的输出层 dec.out 的权重与偏置置零：
                         残差 r ≡ 0，dz 恒为 0，dy、dx、cellprob 只剩（S1 微调后的）编码器 + 二维头。与 A4 的“关”相同。
每个体积的结果另记 gx（本次开关）与 gpu_peak_mib（该进程本体积的 torch 峰值显存）。
不改变 eval_gt 的任何默认行为：不加 --gx_* 时与 eval_gt.py 逐元素相同（冒烟测试核对过）。"""
import json
import os
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_gt as E  # noqa: E402

_M = None
GX = {}


def _gx():
    g = json.loads(os.environ.get("GX_OPTS", "{}"))
    if g.get("keys"):
        g["keyset"] = {l.strip() for l in open(g["keys"]) if l.strip()}
    return g


def _init(spec, opts=None):
    global _M, GX
    import logging
    logging.disable(logging.WARNING)
    GX = _gx()
    _M = E.load_model(spec, opts)
    if GX.get("zero_dec"):                                                         # [gx] M1：三维解码器输出层置零
        import torch
        assert type(_M).__name__ == "Model3D" and hasattr(_M.net, "dec"), "--gx_zero_dec 只用于含三维解码器的真三维模型"
        with torch.no_grad():
            _M.net.dec.out.weight.zero_(); _M.net.dec.out.bias.zero_()
    E._M = _M


def degrade(img, gt, ign, r, k, c):
    """G4：z 向每 k 层取 1 层，再按核内外中位强度差加高斯噪声。返回新的 img、gt、ign，并就地改 r["voxel"]。"""
    vox = tuple(r["voxel"] or (1.0, 1.0, 1.0))
    if k > 1:
        img, gt = img[::k], gt[::k]
        ign = ign[::k] if ign is not None else None
        r["voxel"] = (vox[0] * k, vox[1], vox[2])
    if c > 0:
        im = img.astype(np.float32)
        valid = np.ones(gt.shape, bool) if ign is None else ~ign
        fg = gt > 0
        mf = float(np.median(im[fg & valid])) if (fg & valid).any() else float(im.max())
        mb = float(np.median(im[~fg & valid])) if (~fg & valid).any() else float(im.min())
        sig = c * max(mf - mb, 1e-6)
        rng = np.random.default_rng(zlib.crc32(f"{r['ds']}/{r['id']}/{k}/{c:g}".encode()))
        img = im + rng.normal(0.0, sig, im.shape).astype(np.float32)
        r["gx_noise_sigma"] = sig
        r["gx_contrast"] = mf - mb
    return img, gt, ign


def run_volume(job):
    """eval_gt.run_volume 的副本，只在标了 [gx] 的地方不同。"""
    import torch
    r, a = job
    from eval_cpsam import load_case, eq_diam_xy_px, drop_ignored
    from eval_val import zinterp
    key = f"{r['ds']}__{r['id']}"
    if GX.get("keyset") is not None and key not in GX["keyset"]:                 # [gx] 名单外的体积跳过
        return []
    v0 = r["voxel"] or (1.0, 1.0, 1.0)
    if v0[0] / v0[1] < GX.get("min_ratio", 0):                                     # [gx] 只跑 z/xy ≥ min_ratio 的体积
        return []
    vp = f"{a['out']}/vols/{key}.json"
    if os.path.exists(vp):
        return json.load(open(vp))
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    img, gt, box, ign = load_case(r, a["box"] or 10**9)
    k, c = int(GX.get("zdec", 1)), float(GX.get("noise", 0.0))
    if k > 1 or c > 0:                                                             # [gx] G4 退化
        img, gt, ign = degrade(img, gt, ign, r, k, c)
    if ign is not None:
        gt = gt.copy(); gt[ign] = 0
    if gt.max() == 0:
        return []
    vox = r["voxel"] or (1.0, 1.0, 1.0); ratio = vox[0] / vox[1]; d = d_gt = eq_diam_xy_px(gt, vox)
    r4 = bool(a.get("dmap")) or a.get("views1_ds") is not None
    if a.get("dmap"):
        d = a["dmap"][key]
    if a.get("views1_ds") is not None and type(_M).__name__ == "Model3D":
        if not hasattr(_M, "views0"):
            _M.views0 = _M.views
        _M.views = 1 if r["ds"] in a["views1_ds"] else _M.views0
    iso_on = GX.get("iso", "on") == "on"
    if ratio >= 2 and iso_on:                                                      # [gx] off 时不事先插值
        Zu = int(round(img.shape[0] * ratio)); zi = np.round(np.linspace(0, Zu - 1, img.shape[0])).astype(int)
        im = img.astype(np.float32); lo, hi = np.percentile(im, [1, 99.8]); im = (im - lo) / max(hi - lo, 1e-6)
        img_in, an = zinterp(im, Zu), 1.0
    else:
        img_in, an, zi = img, ratio, None
    fuse = getattr(_M, "view_fuse", None)
    cond = getattr(_M, "cond", None)
    if fuse is None and not cond:
        out = _M.eval(img_in, do_3D=True, z_axis=0, diameter=d, anisotropy=an, flow3D_smooth=1.0, batch_size=a["batch"], compute_masks=False)
    else:
        sz = (img_in.shape[0] / img.shape[0], 1.0, 1.0)
        out = _M.eval(img_in, do_3D=True, z_axis=0, diameter=d, anisotropy=an, flow3D_smooth=1.0, batch_size=a["batch"], compute_masks=False,
                      src_zoom=sz)
    fl = out[1] if isinstance(out[1], (list, tuple)) else out
    dP, cp = fl[1], fl[2]
    t_net = time.time() - t0
    G = E.GTInfo(gt, vox)
    exp_vol = np.pi / 6 * d ** 3 * vox[1] / vox[0]
    rows = []
    for t in (a.get("thrs") or E.THRS):
        m = _M._compute_masks((*img_in.shape, 1), dP, cp, flow_threshold=0.4, cellprob_threshold=t, min_size=15,
                              max_size_fraction=0.4, niter=int(200 / (30.0 / d)), do_3D=True).squeeze()
        m = m if zi is None else m[zi]
        m = drop_ignored(m.astype(np.int32), ign)
        if a.get("save_masks") and t == -2.0:
            os.makedirs(f"{a['out']}/masks", exist_ok=True)
            sel = (lambda v: v) if zi is None else (lambda v: v[..., zi, :, :])
            np.savez_compressed(f"{a['out']}/masks/{key}.npz", pred=m, gt=gt.astype(np.int32), vox=np.asarray(vox, np.float64),
                                d=np.float64(d), dP=sel(np.asarray(dP)).astype(np.float16), cp=sel(np.asarray(cp)).astype(np.float16))
        cnt = np.bincount(m.ravel())
        for f in E.FRACS:
            p = m
            if f > 0:
                small = np.where(cnt < f * exp_vol)[0]; small = small[small > 0]
                p = m.copy(); p[np.isin(p, small)] = 0
            tx = E.taxonomy(G, p)
            den = max(tx["n_gt"] + tx["n_pred"], 1)
            rows.append(dict(ds=r["ds"], id=r["id"], thr=t, frac=f, d_gt=d_gt, ratio=ratio, iso=zi is not None, shape=list(gt.shape),
                             f1=2 * tx["tp5"] / den, f1_7=2 * tx["tp7"] / den, **tx,
                             **(dict(d_used=d, views=getattr(_M, "views", None)) if r4 else {})))
    os.makedirs(os.path.dirname(vp), exist_ok=True)
    peak = int(torch.cuda.max_memory_reserved() // 2**20) if torch.cuda.is_available() else 0
    gxrec = {kk: vv for kk, vv in GX.items() if kk != "keyset"}
    for kk in ("gx_noise_sigma", "gx_contrast"):
        if kk in r:
            gxrec[kk] = r[kk]
    for x in rows:
        x["sec_net"] = round(t_net, 1); x["sec_total"] = round(time.time() - t0, 1)
        x["gx"] = gxrec; x["gpu_peak_mib"] = peak
        if fuse is not None:
            x["fuse"] = _M.last_fuse
        if cond:
            x["res"] = _M.last_res
    json.dump(rows, open(vp + ".part", "w")); os.replace(vp + ".part", vp)
    return rows


def main():
    argv, g, i = sys.argv[1:], {}, 0
    rest = []
    while i < len(argv):
        t = argv[i]
        if t == "--gx_iso":
            g["iso"] = argv[i + 1]; assert g["iso"] in ("on", "off"); i += 2
        elif t == "--gx_zdec":
            g["zdec"] = int(argv[i + 1]); assert g["zdec"] >= 1; i += 2
        elif t == "--gx_noise":
            g["noise"] = float(argv[i + 1]); assert g["noise"] >= 0; i += 2
        elif t == "--gx_keys":
            g["keys"] = os.path.abspath(argv[i + 1]); assert os.path.exists(g["keys"]), g["keys"]; i += 2
        elif t == "--gx_zero_dec":
            g["zero_dec"] = True; i += 1
        elif t == "--gx_min_ratio":
            g["min_ratio"] = float(argv[i + 1]); i += 2
        else:
            rest.append(t); i += 1
    os.environ["GX_OPTS"] = json.dumps(g)                                          # spawn 出来的工作进程继承环境变量
    print("GX", json.dumps(g, ensure_ascii=False), flush=True)
    E.run_volume = run_volume                                                      # main() 里按模块全局名取，换成本文件的版本
    E._init = _init
    sys.argv = [sys.argv[0]] + rest
    E.main()


if __name__ == "__main__":
    main()
