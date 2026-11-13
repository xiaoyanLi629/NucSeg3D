"""nc A4：三维解码器（三维残差）到底改了什么。设计见 $NUCSEG3D_SCRATCH/nc/gpu/DESIGN.md 第 A4 节。
同一输入、同一预处理（gt 协议：真值直径、z/xy ≥ 2 先沿 z 插值到各向同性、按 30/d 缩放），S1 网络各跑两次：
  开 = 原权重；关 = 把三维解码器最后一层 dec.out 的权重与偏置置零（等价于残差 r ≡ 0：dz 恒为 0，dy、dx、cellprob 只剩二维头）。
在网络网格（各向同性、核径约 30 体素）上比较原始输出（5×单位流、logit），两种视图设置都算：
  v1 = 单视图（模型 z = 体积 z，看机制）；vr = 视图规则（cellos、parhyale、mesospim 单视图，其余三视图平均，看实际流程的输出）。
分箱：到最近真值边界（核与背景、核与核的交界）的有符号距离（网络体素，核内为负），取整后夹到 [−20, 20]；未定义区域不计。
每箱累加：体素数、|Δdz|、|Δ(dy,dx)|、|Δcp|，以及开、关两种输出的 |dz|、|(dy,dx)|（求均值用）。
输出 <out>/vols/<ds>__<id>.json（逐体积直方图）、<out>/fig/<ds>__<id>.npz（中间 xz、xy 切面，float16，画示意图用）、<out>/summary.json。
用法：python a4_resid.py --out <目录> --model <S1 目录> [--split test] [--per_ds 2] [--box 384] [--only ds ...] [--ids id ...]"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NB = 20
VIEWS1 = ("cellos", "parhyale", "mesospim")


def nn_resize(a, shape):
    """最近邻缩放（每轴 src = floor((i+0.5)·L_src/L_dst)）。"""
    idx = [np.minimum(((np.arange(n) + 0.5) * a.shape[k] / n).astype(int), a.shape[k] - 1) for k, n in enumerate(shape)]
    return a[np.ix_(*idx)]


def signed_dist(lab):
    from scipy import ndimage as ndi
    b = np.zeros(lab.shape, bool)
    for ax in range(3):
        d = np.diff(lab, axis=ax) != 0
        sl0 = [slice(None)] * 3; sl1 = [slice(None)] * 3
        sl0[ax] = slice(0, -1); sl1[ax] = slice(1, None)
        b[tuple(sl0)] |= d; b[tuple(sl1)] |= d
    dist = ndi.distance_transform_edt(~b).astype(np.float32)
    return np.where(lab > 0, -dist, dist)


def hist(sd, valid, on, off):
    k = (np.clip(np.rint(sd), -NB, NB).astype(np.int64) + NB)[valid]
    n = 2 * NB + 1
    acc = lambda w: np.bincount(k, weights=w[valid].astype(np.float64), minlength=n).tolist()
    dz = on[0] - off[0]; dxy = np.sqrt((on[1] - off[1]) ** 2 + (on[2] - off[2]) ** 2); dcp = np.abs(on[3] - off[3])
    return dict(n=np.bincount(k, minlength=n).tolist(), dz=acc(np.abs(dz)), dxy=acc(dxy), dcp=acc(dcp),
                on_z=acc(np.abs(on[0])), on_xy=acc(np.sqrt(on[1] ** 2 + on[2] ** 2)),
                off_z=acc(np.abs(off[0])), off_xy=acc(np.sqrt(off[1] ** 2 + off[2] ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"]); ap.add_argument("--per_ds", type=int, default=2)
    ap.add_argument("--box", type=int, default=384); ap.add_argument("--only", nargs="*"); ap.add_argument("--ids", nargs="*")
    ap.add_argument("--split_json", default=os.path.join(HERE, "..", "val_split.json"))
    a = ap.parse_args()
    import torch
    from cellpose import models, transforms
    from eval_cpsam import load_case, eq_diam_xy_px
    from eval_gt import pick_cases
    from eval_val import zinterp
    from infer3d import Model3D, run_views
    os.makedirs(f"{a.out}/vols", exist_ok=True); os.makedirs(f"{a.out}/fig", exist_ok=True)
    cases = pick_cases(a.split, a.split_json, a.only, a.ids, a.per_ds)
    print(f"A4：{len(cases)} 个体积，{len({r['ds'] for r in cases})} 套，模型 {a.model}", flush=True)
    M = Model3D(a.model)
    W0 = M.net.dec.out.weight.detach().clone(); B0 = M.net.dec.out.bias.detach().clone()

    def resid(on):
        with torch.no_grad():
            M.net.dec.out.weight.copy_(W0 if on else torch.zeros_like(W0)); M.net.dec.out.bias.copy_(B0 if on else torch.zeros_like(B0))

    t00 = time.time()
    for ci, r in enumerate(cases):
        key = f"{r['ds']}__{r['id']}"; vp = f"{a.out}/vols/{key}.json"
        if os.path.exists(vp):
            continue
        t0 = time.time()
        img, gt, box, ign = load_case(r, a.box)
        if ign is not None:
            gt = gt.copy(); gt[ign] = 0
        vox = r["voxel"] or (1.0, 1.0, 1.0); ratio = vox[0] / vox[1]; d = eq_diam_xy_px(gt, vox)
        if ratio >= 2:                                                   # 同 eval_gt：先插值到各向同性
            Zu = int(round(img.shape[0] * ratio))
            im = img.astype(np.float32); lo, hi = np.percentile(im, [1, 99.8]); im = (im - lo) / max(hi - lo, 1e-6)
            img_in, an = zinterp(im, Zu), 1.0
        else:
            img_in, an = img, ratio
        x4 = transforms.normalize_img(img_in.astype(np.float32)[..., None], **dict(models.normalize_default, normalize=True, norm3D=True))
        rs = 30.0 / d; Lz, Ly, Lx = x4.shape[:3]
        shp = (int(Lz * an * rs), int(Ly * rs), int(Lx * rs))
        xi = transforms.resize_image_3d(x4, shp, no_channels=False)[..., 0] if (rs != 1.0 or an != 1.0) else x4[..., 0]
        lab = nn_resize(gt, xi.shape).astype(np.int32)
        valid = np.ones(xi.shape, bool) if ign is None else ~nn_resize(ign, xi.shape)
        sd = signed_dist(lab)
        rec = dict(ds=r["ds"], id=r["id"], ratio=ratio, d=d, net_shape=list(xi.shape), n_gt=int(len(np.unique(lab)) - 1), nb=NB)
        fn = M._view_fn(None)
        fig = {}
        for tag, nv in (("v1", 1), ("vr", 1 if r["ds"] in VIEWS1 else 3)):
            if tag == "vr" and nv == 1:
                rec["vr"] = rec["v1"]; rec["vr_views"] = 1
                continue
            outs = {}
            for on in (True, False):
                resid(on)
                outs[on] = run_views(fn, xi, nv, pass_perm=True)        # (4, Z', Y', X') 原始输出
            resid(True)
            rec[tag] = hist(sd, valid, outs[True], outs[False])
            if tag == "vr":
                rec["vr_views"] = nv
            if tag == "v1":
                zc, yc = xi.shape[0] // 2, xi.shape[1] // 2
                for nm, arr in (("img", xi[None]), ("lab", lab[None].astype(np.float32)), ("on", outs[True]), ("off", outs[False])):
                    fig[f"{nm}_xz"] = arr[:, :, yc, :].astype(np.float16); fig[f"{nm}_xy"] = arr[:, zc].astype(np.float16)
            del outs
        np.savez_compressed(f"{a.out}/fig/{key}.npz", **fig)
        rec["sec"] = round(time.time() - t0, 1)
        json.dump(rec, open(vp + ".part", "w")); os.replace(vp + ".part", vp)
        h = rec["v1"]; n = np.asarray(h["n"], float)
        inner = slice(NB - 3, NB + 4)
        print(f"  {ci + 1:3d}/{len(cases)} {key[:50]:50s} net {tuple(xi.shape)}  |Δdz| 边界±3 {np.sum(h['dz'][inner]) / max(n[inner].sum(), 1):.3f} "
              f"|Δxy| {np.sum(h['dxy'][inner]) / max(n[inner].sum(), 1):.3f}  {rec['sec']:.0f}s [{(time.time() - t00) / 60:.1f} min]", flush=True)
    # 汇总：逐套把直方图相加，再对数据集取平均（每箱均值）
    import glob
    V = [json.load(open(f)) for f in sorted(glob.glob(f"{a.out}/vols/*.json"))]
    S = {}
    for tag in ("v1", "vr"):
        per = {}
        for ds in sorted({v["ds"] for v in V}):
            L = [v[tag] for v in V if v["ds"] == ds]
            tot = {k: np.sum([np.asarray(x[k], float) for x in L], 0) for k in L[0]}
            per[ds] = {k: (tot[k] / np.maximum(tot["n"], 1)).round(4).tolist() for k in tot if k != "n"}
            per[ds]["n"] = tot["n"].astype(int).tolist()
        mac = {k: np.nanmean([np.where(np.asarray(per[d]["n"]) > 0, per[d][k], np.nan) for d in per], 0).round(4).tolist()
               for k in ("dz", "dxy", "dcp", "on_z", "on_xy", "off_z", "off_xy")}
        S[tag] = dict(per_ds=per, macro=mac)
    S.update(bins=list(range(-NB, NB + 1)), n_volumes=len(V), model=a.model, minutes=round((time.time() - t00) / 60, 1))
    json.dump(S, open(f"{a.out}/summary.json", "w"), indent=1)
    m = S["v1"]["macro"]
    print("SUMMARY A4 v1 宏平均（距离 −6..6）：|Δdz| " + " ".join(f"{x:.2f}" for x in m["dz"][NB - 6:NB + 7]), flush=True)
    print("SUMMARY A4 v1 宏平均（距离 −6..6）：|Δxy| " + " ".join(f"{x:.2f}" for x in m["dxy"][NB - 6:NB + 7]), flush=True)


if __name__ == "__main__":
    main()
