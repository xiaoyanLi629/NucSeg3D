"""从标准化数据（data/std，经 stddata.py 读取）生成 Cellpose-SAM 微调用的 2D 训练切片。

与 make_slices.py 的切片规则相同（每个样本按自身细胞核大小缩放到约 30 像素，XY 直接取层、XZ/YZ 按体素比例拉成各向同性，
挑前景多的层、裁到 ≤768、重新编号并去碎片，再裁到前景外接框外扩 64 像素），区别是只读标准数据与 manifest 里 split=train 的样本，
不再处理原始文件的格式差异。
用法: python make_slices_std.py [--out $NUCSEG3D_SCRATCH/cpft_slices_v3/train] [--workers 16] [--only ds ...]
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import argparse
import json
import os
import re
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.measure import label as cc_label

from stddata import load_manifest, load, load_ignore

TARGET_D = 30.0      # Cellpose 默认细胞核直径（像素）
CROP = 768           # 切片最大边长
MIN_INST = 3         # 每张切片至少的细胞核数
MIN_AREA = 60        # 缩放后小于此面积的碎片去掉
PAD = 64             # 前景外接框外扩的像素数（保留一圈背景作负样本）
MIN_SIDE = 256       # 切片最小边长（Cellpose 训练图块大小）

# 每个体积在三个方向各取多少张（xy, xz, yz）
BUDGET = {"tapenade": (2, 1, 1), "cellos": (4, 3, 3), "celegans": (6, 6, 6), "nis3d": (40, 30, 30),
          "blastospim": (2, 1, 1), "nucverse3d": (8, 6, 6), "purdue": (3, 2, 2), "gonuclear": (20, 15, 15),
          "nisnet_zf": (4, 3, 3), "mouse_embryo": (8, 6, 6), "bbbc050": (2, 1, 1), "parhyale": (30, 25, 25), "ratbrain": (20, 15, 15)}


def eq_diam_um(lab, voxel):
    ids, cnt = np.unique(lab[lab > 0], return_counts=True)
    if not len(cnt):
        return None
    vol = cnt * np.prod(voxel)
    return float(np.median((6 * vol / np.pi) ** (1 / 3)))


def relabel_clean(lab2d):
    """按“同值连通域”重新编号，去掉过小碎片。"""
    out = cc_label(lab2d, background=0, connectivity=1)
    if out.max() == 0:
        return out.astype(np.uint16), 0
    area = np.bincount(out.ravel())
    small = np.where(area < MIN_AREA)[0]
    out[np.isin(out, small[small > 0])] = 0
    ids = np.unique(out); ids = ids[ids > 0]
    lut = np.zeros(int(out.max()) + 1, np.uint16); lut[ids] = np.arange(1, len(ids) + 1)
    return lut[out], len(ids)


def crop_fg(img, lab, rng):
    """大切片裁到 ≤ CROP，裁剪窗口以前景为中心随机选。"""
    H, W = lab.shape
    if H <= CROP and W <= CROP:
        return img, lab
    fg = np.argwhere(lab > 0)
    cy, cx = fg[rng.integers(len(fg))] if len(fg) else (H // 2, W // 2)
    y0 = int(np.clip(cy - CROP // 2, 0, max(H - CROP, 0))); x0 = int(np.clip(cx - CROP // 2, 0, max(W - CROP, 0)))
    return img[y0:y0 + CROP, x0:x0 + CROP], lab[y0:y0 + CROP, x0:x0 + CROP]


def tight_crop(img, lab):
    """裁到前景外接框外扩 PAD 像素，每边至少 MIN_SIDE（不足时在图内向两侧扩），去掉大片空背景。"""
    H, W = lab.shape
    ys, xs = np.where(lab.any(1))[0], np.where(lab.any(0))[0]
    win = []
    for lo, hi, L in ((ys[0], ys[-1] + 1, H), (xs[0], xs[-1] + 1, W)):
        lo, hi = max(lo - PAD, 0), min(hi + PAD, L)
        if hi - lo < MIN_SIDE:
            c = (lo + hi) // 2; lo = int(np.clip(c - MIN_SIDE // 2, 0, max(L - MIN_SIDE, 0))); hi = min(lo + MIN_SIDE, L)
        win.append(slice(lo, hi))
    return img[tuple(win)], lab[tuple(win)]


def pick_planes(fg_per_plane, n, margin, rng):
    """在前景面积排前 50% 的层里随机挑 n 层，避开两端 margin 层。"""
    L = len(fg_per_plane)
    cand = np.arange(margin, L - margin) if L > 2 * margin + n else np.arange(L)
    cand = cand[fg_per_plane[cand] > 0]
    if not len(cand):
        return []
    top = cand[np.argsort(fg_per_plane[cand])[::-1][:max(n, len(cand) // 2)]]
    return sorted(rng.choice(top, size=min(n, len(top)), replace=False).tolist())


def work(args):
    r, out_dir, seed = args
    rng = np.random.default_rng(seed)
    rows = []
    try:
        img, lab = load(r)
        lab = lab.astype(np.int32)
        ign = load_ignore(r)
        if ign is not None:                      # 未定义区域没有标注，把图像抹成背景，免得模型把真实细胞核当背景学
            img[ign] = np.percentile(img[~ign], 1)
        tag = re.sub("[^A-Za-z0-9_.-]", "_", r["id"])
        if r["dim"] == "2d":
            ids, cnt = np.unique(lab[lab > 0], return_counts=True)
            if len(ids) < MIN_INST:
                return rows
            f = TARGET_D / (2 * np.sqrt(np.median(cnt) / np.pi))
            im2 = ndi.zoom(img, f, order=1); lb2 = ndi.zoom(lab, f, order=0)
            im2, lb2 = crop_fg(im2, lb2, rng); lb2, n = relabel_clean(lb2)
            if n: im2, lb2 = tight_crop(im2, lb2)
            if n >= MIN_INST:
                name = f"{r['ds']}__{tag}__2d"
                tifffile.imwrite(f"{out_dir}/{name}.tif", im2.astype(np.float32)); tifffile.imwrite(f"{out_dir}/{name}_masks.tif", lb2)
                rows.append(dict(ds=r["ds"], name=name, ori="2d", n=n))
            return rows
        vz, vy, vx = r["voxel"] or (1.0, 1.0, 1.0)
        aniso = vz / vy
        d = eq_diam_um(lab, (vz, vy, vx))
        scale = TARGET_D / (d / vy) if d else 1.0
        nxy, nxz, nyz = BUDGET[r["ds"]]
        fg = lab > 0
        planes = {"xy": (pick_planes(fg.sum((1, 2)), nxy, 2, rng), lambda k: (img[k], lab[k]), (scale, scale)),
                  "xz": (pick_planes(fg.sum((0, 2)), nxz, 5, rng), lambda k: (img[:, k], lab[:, k]), (aniso * scale, scale)),
                  "yz": (pick_planes(fg.sum((0, 1)), nyz, 5, rng), lambda k: (img[:, :, k], lab[:, :, k]), (aniso * scale, scale))}
        for ori, (ks, get, zf) in planes.items():
            for k in ks:
                im2d, lb2d = get(k)
                im2 = ndi.zoom(im2d.astype(np.float32), zf, order=1); lb2 = ndi.zoom(lb2d, zf, order=0)
                im2, lb2 = crop_fg(im2, lb2, rng); lb2, n = relabel_clean(lb2)
                if n: im2, lb2 = tight_crop(im2, lb2)
                if n < MIN_INST:
                    continue
                name = f"{r['ds']}__{tag}__{ori}{k:04d}"
                tifffile.imwrite(f"{out_dir}/{name}.tif", im2); tifffile.imwrite(f"{out_dir}/{name}_masks.tif", lb2)
                rows.append(dict(ds=r["ds"], name=name, ori=ori, n=n))
    except Exception as e:
        rows.append(dict(ds=r["ds"], name=r["id"], error=repr(e)[:200]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{_NSP_SCRATCH}/cpft_slices_v3/train")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load_manifest(split="train", ds=a.only or None)
    out = []
    with Pool(a.workers) as p:
        for x in p.imap_unordered(work, [(r, a.out, i) for i, r in enumerate(rows)], chunksize=1):
            out.extend(x)
    errs = [x for x in out if "error" in x]; ok = [x for x in out if "error" not in x]
    for e in errs[:10]:
        print("  ERR", e)
    agg = defaultdict(lambda: [0, 0])
    for x in ok:
        agg[x["ds"]][0] += 1; agg[x["ds"]][1] += x["n"]
    for k, (n, inst) in sorted(agg.items()):
        print(f"  {k:11s} 训练切片 {n:5d}  细胞核实例 {inst:7d}")
    json.dump(dict(slices=ok, errors=errs), open(os.path.join(os.path.dirname(a.out), "train_manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
