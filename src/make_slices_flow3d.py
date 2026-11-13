"""生成 v5 训练切片与两套训练目标（三维流目标实验用）。

切片本身与 make_slices_std.py 完全相同：同一 manifest、同一随机数序列、同样的缩放与裁剪，只是顺便记下裁剪窗口。
每张切片写两份 4 通道目标文件 [标签, 是否细胞核, 行方向流, 列方向流]（与 cellpose labels_to_flows 的格式一致）：
  <out>/v5_2d/<name>_flows.tif  Cellpose 原做法：在二维切片标注上做扩散，得到指向“切面中心”的平面内单位向量；
  <out>/v5_3d/<name>_flows.tif  三维流目标：把整个三维标注按切片同样的比例缩放成各向同性，做三维扩散，得到指向
                                细胞核三维中心的单位向量，再取切片平面内的两个分量（不再归一化）。
Cellpose 三维推理把三个方向切片的同名分量相加，所以三个方向都学到三维流的投影时，相加正好还原三维流场。
两个目录里图像与掩码是同一份文件的硬链接，只有 _flows.tif 不同。二维数据（crypt）两套目标相同。

三维流场只在与所选切片相交的细胞核上计算（扩散在每个细胞核内部独立进行，所以只取这些核是精确的），
各向同性体积按 scipy ndi.zoom(order=0) 的取样规则直接用下标取出，网格与切片的二维缩放逐像素对齐。
切片法向上取最近的各向同性层；该层同一位置不是同一个细胞核时，再试另一侧的邻层；仍对不上的像素退回二维流，
并统计退回比例。

用法: python make_slices_flow3d.py --out <统计结果目录> [--dest $NUCSEG3D_SCRATCH/cpft_v5] [--workers 8] [--only ds ...] [--tar]
  --tar  生成完后把 v5_2d、v5_3d 分别打包到共享盘 CellposeFT/slices/（子节点训练时从这里解包）
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import tifffile
from scipy import ndimage as ndi

from make_slices_std import (BUDGET, CROP, MIN_INST, MIN_SIDE, PAD, TARGET_D, eq_diam_um, pick_planes,
                             relabel_clean)
from stddata import load, load_ignore, load_manifest

MAX_VOX = 1.5e8   # 一次三维扩散的最大体素数，超过就逐张切片分别算


def crop_fg_win(H, W, lab, rng):
    """与 make_slices_std.crop_fg 相同（同样的随机数调用），返回窗口。"""
    if H <= CROP and W <= CROP:
        return slice(0, H), slice(0, W)
    fg = np.argwhere(lab > 0)
    cy, cx = fg[rng.integers(len(fg))] if len(fg) else (H // 2, W // 2)
    y0 = int(np.clip(cy - CROP // 2, 0, max(H - CROP, 0))); x0 = int(np.clip(cx - CROP // 2, 0, max(W - CROP, 0)))
    return slice(y0, y0 + CROP), slice(x0, x0 + CROP)


def tight_win(lab):
    """与 make_slices_std.tight_crop 相同，返回窗口。"""
    H, W = lab.shape
    ys, xs = np.where(lab.any(1))[0], np.where(lab.any(0))[0]
    win = []
    for lo, hi, L in ((ys[0], ys[-1] + 1, H), (xs[0], xs[-1] + 1, W)):
        lo, hi = max(lo - PAD, 0), min(hi + PAD, L)
        if hi - lo < MIN_SIDE:
            c = (lo + hi) // 2; lo = int(np.clip(c - MIN_SIDE // 2, 0, max(L - MIN_SIDE, 0))); hi = min(lo + MIN_SIDE, L)
        win.append(slice(lo, hi))
    return tuple(win)


def zoom_index(n_in, f):
    """ndi.zoom(order=0) 沿一个轴按比例 f 取样时用到的输入下标（直接让 scipy 缩放一个下标序列，舍入规则完全一致）。"""
    return ndi.zoom(np.arange(n_in, dtype=np.float64), f, order=0, mode="nearest").round().astype(int)


def flows3d(lab_iso, device):
    """lab_iso: 各向同性标签（原编号）。返回 (3, Z, Y, X) float32 单位向量 (dz, dy, dx)。

    逐个细胞核在它自己的外接框里做扩散（每个核内部的扩散互不影响，所以与整块一起算等价）。
    整块一起算时 Cellpose 对大体积改用单精度，且迭代步数按最大的物体定，远离中心处会下溢成 0、归一化得到无穷大；
    逐核计算时小块自动用双精度，步数按该核自身大小定。
    同一编号若由几块互不相连的部分组成（个别标注如此），按三维 6 连通拆开分别扩散，
    与二维切片里 relabel_clean 按连通域重新编号的规则一致。"""
    from cellpose.dynamics import masks_to_flows_gpu_3d
    out = np.zeros((3,) + lab_iso.shape, np.float32)
    for i, sl in enumerate(ndi.find_objects(lab_iso)):
        if sl is None:
            continue
        sl = tuple(slice(max(s.start - 1, 0), min(s.stop + 1, n)) for s, n in zip(sl, lab_iso.shape))
        comp, nc = ndi.label(lab_iso[sl] == i + 1)
        for j, csl in enumerate(ndi.find_objects(comp)):
            csl = tuple(slice(max(s.start - 1, 0), min(s.stop + 1, n)) for s, n in zip(csl, comp.shape))
            m = comp[csl] == j + 1
            mu = masks_to_flows_gpu_3d(m.astype(np.int32), device=device)
            gsl = tuple(slice(a.start + b.start, a.start + b.stop) for a, b in zip(sl, csl))
            out[(slice(None),) + gsl][:, m] = mu[:, m]
    if not np.isfinite(out).all():
        raise FloatingPointError("三维流场出现非有限值")
    return out


def flows2d(lb, device):
    from cellpose.dynamics import masks_to_flows_gpu
    if lb.max() == 0:
        return np.zeros((2,) + lb.shape, np.float32)
    return masks_to_flows_gpu(lb.astype(np.int64), device=device)[0].astype(np.float32)


def write_pair(out, name, im, lb, v2, v3):
    d2, d3 = f"{out}/v5_2d", f"{out}/v5_3d"
    tifffile.imwrite(f"{d2}/{name}.tif", im.astype(np.float32)); tifffile.imwrite(f"{d2}/{name}_masks.tif", lb)
    for a, b in ((f"{d2}/{name}.tif", f"{d3}/{name}.tif"), (f"{d2}/{name}_masks.tif", f"{d3}/{name}_masks.tif")):
        if os.path.exists(b):
            os.remove(b)
        os.link(a, b)
    lbf = lb.astype(np.float32)[None]
    for d, v in ((d2, v2), (d3, v3)):
        tifffile.imwrite(f"{d}/{name}_flows.tif", np.concatenate([lbf, lbf > 0.5, v], 0).astype(np.float32))


def stats_of(lb, v2, v3):
    fg = lb > 0
    if not fg.any():
        return None
    a, b = v2[:, fg], v3[:, fg]
    na, nb = np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0)
    ok = (na > 1e-6) & (nb > 1e-6)
    cos = float(((a[:, ok] * b[:, ok]).sum(0) / (na[ok] * nb[ok])).mean()) if ok.any() else float("nan")
    return dict(npx=int(fg.sum()), cos=cos, mag3=float(nb.mean()))


def work(args):
    r, out, seed, gpu = args
    import torch
    device = torch.device(f"cuda:{gpu}")
    rng = np.random.default_rng(seed)
    rows = []
    try:
        img, lab = load(r)
        lab = lab.astype(np.int32)
        ign = load_ignore(r)
        if ign is not None:
            img[ign] = np.percentile(img[~ign], 1)
        tag = re.sub("[^A-Za-z0-9_.-]", "_", r["id"])
        if r["dim"] == "2d":                                  # 与 make_slices_std 相同；两套目标都是二维流
            ids, cnt = np.unique(lab[lab > 0], return_counts=True)
            if len(ids) < MIN_INST:
                return rows
            f = TARGET_D / (2 * np.sqrt(np.median(cnt) / np.pi))
            im2 = ndi.zoom(img, f, order=1); lb2 = ndi.zoom(lab, f, order=0)
            wy, wx = crop_fg_win(*lb2.shape, lb2, rng); im2, lb2 = im2[wy, wx], lb2[wy, wx]
            lb2, n = relabel_clean(lb2)
            if n:
                w = tight_win(lb2); im2, lb2 = im2[w], lb2[w]
            if n >= MIN_INST:
                name = f"{r['ds']}__{tag}__2d"
                v2 = flows2d(lb2, device)
                write_pair(out, name, im2, lb2, v2, v2)
                rows.append(dict(ds=r["ds"], name=name, ori="2d", n=n, fallback=0.0, **(stats_of(lb2, v2, v2) or {})))
            return rows

        vz, vy, vx = r["voxel"] or (1.0, 1.0, 1.0)
        aniso = vz / vy
        d = eq_diam_um(lab, (vz, vy, vx))
        scale = TARGET_D / (d / vy) if d else 1.0
        if r["ds"] not in BUDGET:                            # 与 make_slices_std 一致：不在 BUDGET 里的数据集（mesospim）不出训练切片
            return rows
        nxy, nxz, nyz = BUDGET[r["ds"]]
        fg = lab > 0
        Z, Y, X = lab.shape
        zf3 = (aniso * scale, scale, scale)
        Zi, Yi, Xi = (int(round(n * f)) for n, f in zip((Z, Y, X), zf3))
        iz, iy, ix = zoom_index(Z, zf3[0]), zoom_index(Y, zf3[1]), zoom_index(X, zf3[2])
        assert (len(iz), len(iy), len(ix)) == (Zi, Yi, Xi)
        planes = {"xy": (pick_planes(fg.sum((1, 2)), nxy, 2, rng), lambda k: (img[k], lab[k]), (scale, scale), 0),
                  "xz": (pick_planes(fg.sum((0, 2)), nxz, 5, rng), lambda k: (img[:, k], lab[:, k]), (aniso * scale, scale), 1),
                  "yz": (pick_planes(fg.sum((0, 1)), nyz, 5, rng), lambda k: (img[:, :, k], lab[:, :, k]), (aniso * scale, scale), 2)}
        # 先按原脚本的顺序生成全部切片（随机数用法与原脚本一致），三维流稍后统一补上
        todo = []
        for ori, (ks, get, zf, ax) in planes.items():
            for k in ks:
                im2d, lb2d = get(k)
                im2 = ndi.zoom(im2d.astype(np.float32), zf, order=1); lbz = ndi.zoom(lb2d, zf, order=0)
                wy, wx = crop_fg_win(*lbz.shape, lbz, rng)
                im2, lb2 = im2[wy, wx], lbz[wy, wx]
                lb2, n = relabel_clean(lb2)
                tw = (slice(None), slice(None))
                if n:
                    tw = tight_win(lb2); im2, lb2 = im2[tw], lb2[tw]
                if n < MIN_INST:
                    continue
                todo.append(dict(ori=ori, k=k, ax=ax, im=im2, lb=lb2, lbz=lbz, win=(wy, wx), tw=tw))
        if not todo:
            return rows
        # 与这些切片相交的细胞核（原编号）
        need = set()
        for t in todo:
            need.update(np.unique(t["lbz"][t["lbz"] > 0]).tolist())

        def iso_block(ids):
            """取包含 ids 的各向同性子块：返回 (标签块, 各轴起点)。"""
            sub = np.where(np.isin(lab, list(ids)), lab, 0)
            objs = ndi.find_objects(sub)
            lo = [min(o[a].start for o in objs if o is not None) for a in range(3)]
            hi = [max(o[a].stop for o in objs if o is not None) for a in range(3)]
            rngs = []
            for a, idx in enumerate((iz, iy, ix)):
                j = np.where((idx >= lo[a]) & (idx < hi[a]))[0]
                j0, j1 = max(j[0] - 1, 0), min(j[-1] + 2, len(idx))
                rngs.append(np.arange(j0, j1))
            blk = sub[np.ix_(iz[rngs[0]], iy[rngs[1]], ix[rngs[2]])]
            return blk, [g[0] for g in rngs]

        def nvox(ids):
            sub = np.isin(lab, list(ids))
            objs = ndi.find_objects(sub.astype(np.uint8))[0]
            return np.prod([(s.stop - s.start) * f for s, f in zip(objs, zf3)])

        groups = [(todo, need)] if nvox(need) <= MAX_VOX else \
                 [([t], set(np.unique(t["lbz"][t["lbz"] > 0]).tolist())) for t in todo]
        for ts, ids in groups:
            blk, off = iso_block(ids)
            mu = flows3d(blk, device)
            for t in ts:
                ax, k = t["ax"], t["k"]
                n_in, n_iso = (Z, Y, X)[ax], (Zi, Yi, Xi)[ax]
                fpos = k * (n_iso - 1) / max(n_in - 1, 1)
                cand = sorted({int(np.floor(fpos)), int(np.ceil(fpos))}, key=lambda j: abs(j - fpos))
                lbz = t["lbz"]
                v3 = np.zeros((2,) + lbz.shape, np.float32); filled = np.zeros(lbz.shape, bool)
                inplane = [c for c in range(3) if c != ax]           # 平面内两个分量：(行轴, 列轴)
                for j in cand:
                    jj = j - off[ax]
                    if jj < 0 or jj >= blk.shape[ax]:
                        continue
                    # 该层在块内的二维截面，再放回切片的完整网格
                    full_l = np.zeros(lbz.shape, np.int64); full_v = np.zeros((2,) + lbz.shape, np.float32)
                    sl_blk = [slice(None)] * 3; sl_blk[ax] = jj
                    bl = blk[tuple(sl_blk)]; bv = mu[(slice(None),) + tuple(sl_blk)]
                    r0, c0 = (off[a] for a in inplane)
                    hh, ww = bl.shape
                    full_l[r0:r0 + hh, c0:c0 + ww] = bl
                    full_v[:, r0:r0 + hh, c0:c0 + ww] = bv[inplane]
                    m = (full_l == lbz) & (lbz > 0) & ~filled
                    v3[:, m] = full_v[:, m]; filled |= m
                wy, wx = t["win"]; tw = t["tw"]
                v3c = v3[:, wy, wx][(slice(None),) + tw]; fc = filled[wy, wx][tw]
                lb2 = t["lb"]
                v2 = flows2d(lb2, device)
                miss = (lb2 > 0) & ~fc
                v3c = np.where(miss[None], v2, v3c); v3c[:, lb2 == 0] = 0
                name = f"{r['ds']}__{tag}__{t['ori']}{k:04d}"
                write_pair(out, name, t["im"], lb2, v2, v3c)
                st = stats_of(lb2, v2, v3c) or {}
                rows.append(dict(ds=r["ds"], name=name, ori=t["ori"], n=int(lb2.max()),
                                 fallback=float(miss.sum() / max((lb2 > 0).sum(), 1)), **st))
    except Exception as e:
        import traceback
        rows.append(dict(ds=r["ds"], name=r["id"], error=repr(e)[:300], tb=traceback.format_exc()[-800:]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="统计结果目录")
    ap.add_argument("--dest", default=f"{_NSP_SCRATCH}/cpft_v5", help="切片输出目录")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tar", action="store_true")
    ap.add_argument("--gpus", default="", help="逗号分隔的 GPU 序号；默认用所有可见的卡")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for d in ("v5_2d", "v5_3d"):
        os.makedirs(f"{a.dest}/{d}", exist_ok=True)
    rows = load_manifest(split="train")
    import torch
    gpus = [int(g) for g in a.gpus.split(",")] if a.gpus else list(range(torch.cuda.device_count()))
    # 种子 i 是样本在完整训练 manifest 里的序号（与 make_slices_std 全量运行相同），--only 不改变种子
    jobs = [(r, a.dest, i, gpus[i % len(gpus)]) for i, r in enumerate(rows) if not a.only or r["ds"] in a.only]
    import multiprocessing as mp
    out = []
    with mp.get_context("spawn").Pool(a.workers) as p:
        for k, x in enumerate(p.imap_unordered(work, jobs, chunksize=1)):
            out.extend(x)
            if k % 100 == 0:
                print(f"  {k}/{len(jobs)} 个体积", flush=True)
    errs = [x for x in out if "error" in x]; ok = [x for x in out if "error" not in x]
    for e in errs[:10]:
        print("  ERR", e["ds"], e["name"], e["error"], "\n", e.get("tb", ""))
    agg = defaultdict(lambda: defaultdict(list))
    for x in ok:
        g = agg[(x["ds"], x["ori"])]
        g["n"].append(1); g["px"].append(x.get("npx", 0)); g["fb"].append(x["fallback"] * x.get("npx", 0))
        g["cos"].append(x.get("cos", np.nan) * x.get("npx", 0)); g["mag"].append(x.get("mag3", np.nan) * x.get("npx", 0))
    print(f"{'数据集':12s} {'方向':4s} {'切片':>6s} {'退回二维比例':>10s} {'二三维流夹角余弦':>14s} {'三维流平面内模长':>14s}")
    summ = {}
    for (ds, ori), g in sorted(agg.items()):
        px = max(sum(g["px"]), 1)
        s = dict(slices=len(g["n"]), fallback=sum(g["fb"]) / px, cos=np.nansum(g["cos"]) / px, mag3=np.nansum(g["mag"]) / px)
        summ[f"{ds}/{ori}"] = s
        print(f"{ds:12s} {ori:4s} {s['slices']:6d} {s['fallback']:10.4f} {s['cos']:14.3f} {s['mag3']:14.3f}")
    json.dump(dict(slices=ok, errors=errs, summary=summ), open(f"{a.out}/v5_manifest.json", "w"), indent=1)
    print(f"SLICES_DONE 切片 {len(ok)} 张，出错体积 {len(errs)} 个", flush=True)
    if errs:
        raise SystemExit("有体积出错，不打包")
    if a.tar:
        import subprocess
        for d in ("v5_2d", "v5_3d"):
            dst = f"{_NSP_ROOT}/slices/{d}.tar"
            subprocess.run(["tar", "cf", dst + ".part", "-C", a.dest, d], check=True)
            os.replace(dst + ".part", dst)
            print("打包", dst, os.path.getsize(dst) // 2**20, "MB", flush=True)


if __name__ == "__main__":
    main()
