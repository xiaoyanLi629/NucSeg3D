"""round 5：逐核形状分析（只读 eval_gt --save_masks 存下的 npz，CPU）。

对每个真值核 g：取重叠最多的预测 p，算三维 IoU、z 向层数比（p/g）、xy 等效直径比（各层最大截面）、体积比、
z 与 xy 质心偏移（单位：真值核的 z 层数 / xy 直径），以及把 IoU 拆成“xy 投影 IoU”和“z 区间 IoU”两部分。
另外沿真值核质心所在的 (y, x) 竖线、质心所在层的 x 横线，量网络 cellprob > −2 的连续段长度与真值长度之比，
看网络输出本身（跟流之前）够不够长。
用法：python shape_diag.py <eval_gt 结果目录> [...]   打印每个目录的汇总，并写 <目录>/shape_diag.json
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi


def run_len(v, c, thr=-2.0):
    """一维数组 v 在下标 c 处、> thr 的连续段长度（c 处不满足则为 0）。"""
    if v[c] <= thr:
        return 0
    lo = c
    while lo > 0 and v[lo - 1] > thr:
        lo -= 1
    hi = c
    while hi < len(v) - 1 and v[hi + 1] > thr:
        hi += 1
    return hi - lo + 1


def analyze(npz):
    D = np.load(npz)
    P, G = D["pred"].astype(np.int64), D["gt"].astype(np.int64)
    cp = D["cp"].astype(np.float32) if "cp" in D else None
    out = []
    gs, ps_all = ndi.find_objects(G), ndi.find_objects(P)
    for gi, sl in enumerate(gs, start=1):
        if sl is None:
            continue
        gm = G[sl] == gi
        pv = P[sl][gm]
        pv = pv[pv > 0]
        z_g = np.where(gm.any((1, 2)))[0]; nz_g = len(z_g)
        a_g = gm.sum((1, 2)).max(); dxy_g = 2 * np.sqrt(a_g / np.pi); V_g = gm.sum()
        cz, cy, cx = [float(np.mean(i)) + s.start for i, s in zip(np.nonzero(gm), sl)]
        r = dict(g=gi, nz_g=nz_g, dxy_g=float(dxy_g), V_g=int(V_g))
        if cp is not None:                                              # 网络输出本身沿 z / x 的前景长度
            iz, iy, ix = int(round(cz)), int(round(cy)), int(round(cx))
            col = G[:, iy, ix] == gi; row = G[iz, iy, :] == gi
            r["cpz_ratio"] = run_len(cp[:, iy, ix], iz) / max(col.sum(), 1) if col[iz] else np.nan
            r["cpx_ratio"] = run_len(cp[iz, iy, :], ix) / max(row.sum(), 1) if row[ix] else np.nan
        if pv.size == 0:
            r.update(iou=0.0, matched=False); out.append(r); continue
        p = int(np.bincount(pv).argmax())
        # 预测核的外接框可能超出真值框：在两者并集框里算
        ps = ps_all[p - 1]
        u = tuple(slice(min(a.start, b.start), max(a.stop, b.stop)) for a, b in zip(sl, ps))
        gm2, pm2 = G[u] == gi, P[u] == p
        inter = (gm2 & pm2).sum(); iou = inter / (gm2 | pm2).sum()
        z_p = np.where(pm2.any((1, 2)))[0]; nz_p = len(z_p)
        a_p = pm2.sum((1, 2)).max(); dxy_p = 2 * np.sqrt(a_p / np.pi)
        pz, py, px = [float(np.mean(i)) + s.start for i, s in zip(np.nonzero(pm2), u)]
        gproj, pproj = gm2.any(0), pm2.any(0)
        iou_xy = (gproj & pproj).sum() / (gproj | pproj).sum()
        zg = set(z_g + sl[0].start); zp = set(z_p + u[0].start)
        iou_z = len(zg & zp) / len(zg | zp)
        r.update(matched=True, iou=float(iou), iou_xy=float(iou_xy), iou_z=float(iou_z), nz_ratio=nz_p / nz_g, dxy_ratio=float(dxy_p / dxy_g),
                 V_ratio=float(pm2.sum() / V_g), dz_off=(pz - cz) / nz_g, dxy_off=float(np.hypot(py - cy, px - cx) / dxy_g))
        out.append(r)
    return out


def summarize(rs):
    m = [r for r in rs if r["matched"]]
    bad = [r for r in m if r["iou"] < 0.5]
    med = lambda k, L: float(np.nanmedian([r[k] for r in L])) if L else float("nan")
    frac = lambda f, L: float(np.mean([f(r) for r in L])) if L else float("nan")
    S = dict(n_gt=len(rs), matched=len(m), iou_lt05=len(bad),
             med_iou=med("iou", m), med_iou_xy=med("iou_xy", m), med_iou_z=med("iou_z", m),
             med_nz_ratio=med("nz_ratio", m), med_dxy_ratio=med("dxy_ratio", m), med_V_ratio=med("V_ratio", m),
             med_dz_off=med("dz_off", m), med_dxy_off=med("dxy_off", m),
             bad_zshort=frac(lambda r: r["nz_ratio"] < 0.75, bad), bad_zlong=frac(lambda r: r["nz_ratio"] > 1.33, bad),
             bad_xysmall=frac(lambda r: r["dxy_ratio"] < 0.8, bad), bad_xylarge=frac(lambda r: r["dxy_ratio"] > 1.25, bad),
             bad_xy_worse=frac(lambda r: r["iou_xy"] < r["iou_z"], bad))
    if rs and "cpz_ratio" in rs[0]:
        S.update(med_cpz_ratio=med("cpz_ratio", rs), med_cpx_ratio=med("cpx_ratio", rs))
    return S


if __name__ == "__main__":
    for d in sys.argv[1:]:
        per, allr = {}, []
        for f in sorted(glob.glob(f"{d}/masks/*.npz")):
            rs = analyze(f); allr += rs; per[os.path.basename(f)[:-4]] = summarize(rs)
        S = dict(all=summarize(allr), per_vol=per)
        json.dump(S, open(f"{d}/shape_diag.json", "w"), indent=1)
        a = S["all"]
        print(f"SHAPE {os.path.basename(d.rstrip('/')):28s} n={a['n_gt']} IoU<0.5 {a['iou_lt05']}/{a['matched']}  中位 IoU {a['med_iou']:.2f}"
              f"（xy 投影 {a['med_iou_xy']:.2f}，z 区间 {a['med_iou_z']:.2f}）  层数比 {a['med_nz_ratio']:.2f}  xy 直径比 {a['med_dxy_ratio']:.2f}"
              f"  体积比 {a['med_V_ratio']:.2f}  z 偏移 {a['med_dz_off']:+.2f}  xy 偏移 {a['med_dxy_off']:.2f}"
              + (f"  cellprob 段长/真值 z {a['med_cpz_ratio']:.2f} x {a['med_cpx_ratio']:.2f}" if 'med_cpz_ratio' in a else "")
              + f" | 坏核里 z 太短 {a['bad_zshort']:.2f} z 太长 {a['bad_zlong']:.2f} xy 太小 {a['bad_xysmall']:.2f} xy 太大 {a['bad_xylarge']:.2f} xy 更差 {a['bad_xy_worse']:.2f}",
              flush=True)
