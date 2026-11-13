"""把 3D 标注数据切成 Cellpose-SAM 微调用的 2D 训练切片，并生成 3D 测试集清单。

做法参照 Cellpose-SAM 论文作者脚本（paper/cpsam/create_plant_train_wolny.py），但不对整个体积做三维重采样，
而是取出切片后对每张 2D 切片重采样，效果等价、内存占用小得多：
  1. 每个样本由自身真值估计细胞核中位等效直径（µm），求出缩放倍数 s，使细胞核约为 30 像素
     （BlastoSPIM 早期胚胎核约 23 µm、后期约 10 µm，同一数据集内差异很大，所以不用数据集统一值）；
  2. XY 切片：取原始 z 层，按 (s, s) 缩放；XZ / YZ 切片：取 y / x 固定的平面，按 (vz/vxy·s, s) 缩放，即拉成各向同性；
  3. 每个方向按“前景多”的原则挑若干层，避开两端；大切片随机裁到 ≤ CROP；
  4. 每张切片重新编号（3D 核切出来可能断成几块），去掉过小碎片，少于 MIN_INST 个核的切片丢弃；
  5. 图像存 float32 tif，标签存 *_masks.tif（uint16），文件名 <数据集>__<样本>__<方向><层号>。
测试集：保留完整 3D 体积（大体积裁出前景最多的子块），写入 test_manifest.json，供 3D 评测使用。
用法: python make_slices.py [--out $NUCSEG3D_SCRATCH/cpft_slices] [--workers 12] [--only ds1 ds2]
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

from cpft_data import samples, read, voxel_of

TARGET_D = 30.0      # Cellpose 默认细胞核直径（像素）
CROP = 768           # 切片最大边长
MIN_INST = 3         # 每张切片至少的细胞核数
MIN_AREA = 60        # 缩放后小于此面积的碎片去掉

# 每个体积在三个方向各取多少张（xy, xz, yz）
BUDGET = {"tapenade": (2, 1, 1), "cellos": (4, 3, 3), "celegans": (6, 6, 6), "nis3d": (40, 30, 30),
          "blastospim": (2, 1, 1), "nucverse3d": (8, 6, 6), "purdue": (3, 2, 2), "gonuclear": (20, 15, 15)}


def assign_split(s, order):
    """原数据没有划分的，按样本固定规则留出测试集；val 并入训练。order: {数据集: 排序后的样本 id 列表}"""
    ds, sid = s["ds"], s["id"]
    if s["split"] in ("train", "val"):
        return "train"
    if s["split"] == "test":
        return "test"
    if ds == "cellos":
        return "test" if order["cellos"].index(sid) % 6 == 0 else "train"
    if ds == "gonuclear":
        return "test" if sid.startswith("1170") else "train"
    if ds == "purdue":
        return "test" if order["purdue"].index(sid) % 5 == 0 else "train"
    if ds == "nis3d":
        return "split_x"      # 同一体积按 x 方向切开：左 75% 训练，右 25% 测试
    return "train"            # crypt（已是 2D 切片）


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
    s, scale, out_dir, seed = args
    rng = np.random.default_rng(seed)
    rows = []
    try:
        if s["kind"] == "2d":
            img = np.squeeze(read(s["img"])).astype(np.float32); lab = np.squeeze(read(s["lab"]))
            if img.ndim == 3:
                img = img[..., 0] if img.shape[-1] <= 4 else img[0]
            ids, cnt = np.unique(lab[lab > 0], return_counts=True)
            if len(ids) < MIN_INST:
                return rows
            f = TARGET_D / (2 * np.sqrt(np.median(cnt) / np.pi))      # 由面积估计直径
            im2 = ndi.zoom(img, f, order=1); lb2 = ndi.zoom(lab, f, order=0)
            im2, lb2 = crop_fg(im2, lb2, rng); lb2, n = relabel_clean(lb2)
            if n >= MIN_INST:
                name = f"{s['ds']}__{re.sub('[^A-Za-z0-9_.-]', '_', s['id'])}__2d"
                tifffile.imwrite(f"{out_dir}/{name}.tif", im2.astype(np.float32)); tifffile.imwrite(f"{out_dir}/{name}_masks.tif", lb2)
                rows.append(dict(ds=s["ds"], name=name, ori="2d", n=n))
            return rows
        img = np.squeeze(read(s["img"])); lab = np.squeeze(read(s["lab"])).astype(np.int32)
        if s.get("x_range"):
            a, b = s["x_range"]; img, lab = img[..., a:b], lab[..., a:b]
        vz, vy, vx = voxel_of(s) or (1.0, 1.0, 1.0)
        aniso = vz / vy
        d = eq_diam_um(lab, (vz, vy, vx))          # 每个样本按自己的细胞核大小定缩放（同一数据集里大小可能差很多）
        if d:
            scale = TARGET_D / (d / vy)
        nxy, nxz, nyz = BUDGET[s["ds"]]
        tag = re.sub("[^A-Za-z0-9_.-]", "_", s["id"])
        fg = lab > 0
        planes = {"xy": (pick_planes(fg.sum((1, 2)), nxy, 2, rng), lambda k: (img[k], lab[k]), (scale, scale)),
                  "xz": (pick_planes(fg.sum((0, 2)), nxz, 5, rng), lambda k: (img[:, k], lab[:, k]), (aniso * scale, scale)),
                  "yz": (pick_planes(fg.sum((0, 1)), nyz, 5, rng), lambda k: (img[:, :, k], lab[:, :, k]), (aniso * scale, scale))}
        for ori, (ks, get, zf) in planes.items():
            for k in ks:
                im2d, lb2d = get(k)
                im2 = ndi.zoom(im2d.astype(np.float32), zf, order=1); lb2 = ndi.zoom(lb2d, zf, order=0)
                im2, lb2 = crop_fg(im2, lb2, rng); lb2, n = relabel_clean(lb2)
                if n < MIN_INST:
                    continue
                name = f"{s['ds']}__{tag}__{ori}{k:04d}"
                tifffile.imwrite(f"{out_dir}/{name}.tif", im2); tifffile.imwrite(f"{out_dir}/{name}_masks.tif", lb2)
                rows.append(dict(ds=s["ds"], name=name, ori=ori, n=n))
    except Exception as e:
        rows.append(dict(ds=s["ds"], name=s["id"], error=repr(e)[:200]))
    return rows


def best_box(lab, size):
    """测试体积过大时，在 xy 上裁出前景最多的 size×size 子块。"""
    H, W = lab.shape[-2:]
    if H <= size and W <= size:
        return None
    proj = (lab > 0).sum(0).astype(np.float32)
    k = np.ones((min(size, H), min(size, W)), np.float32)
    integ = proj.cumsum(0).cumsum(1)
    best, by, bx = -1, 0, 0
    for y in range(0, max(H - size, 0) + 1, 32):
        for x in range(0, max(W - size, 0) + 1, 32):
            y1, x1 = min(y + size, H) - 1, min(x + size, W) - 1
            v = integ[y1, x1] - (integ[y - 1, x1] if y else 0) - (integ[y1, x - 1] if x else 0) + (integ[y - 1, x - 1] if y and x else 0)
            if v > best:
                best, by, bx = v, y, x
    return [by, bx, size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{_NSP_SCRATCH}/cpft_slices")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--test_box", type=int, default=384, help="测试体积 xy 最大边长（3D 推理较慢）")
    a = ap.parse_args()
    ALL = samples()
    order = defaultdict(list)
    for x in ALL:
        order[x["ds"]].append(x["id"])
    order = {k: sorted(v) for k, v in order.items()}
    S = [s for s in ALL if not a.only or s["ds"] in a.only]
    tr_dir = f"{a.out}/train"; os.makedirs(tr_dir, exist_ok=True)

    # 1) 划分
    train, test = [], []
    for s in S:
        sp = assign_split(s, order)
        if sp == "split_x":
            W = tifffile.TiffFile(s["img"]).series[0].shape[-1]
            cut = int(W * 0.75)
            train.append({**s, "x_range": (0, cut)}); test.append({**s, "x_range": (cut, W)})
        elif sp == "test":
            test.append(s)
        else:
            train.append(s)

    # 2) 每套数据的缩放倍数：用最多 8 个训练样本估计细胞核中位等效直径
    scale = {}
    for ds in sorted({s["ds"] for s in train if s["kind"] == "3d"}):
        ds_s = [s for s in train if s["ds"] == ds][:8]
        dd = []
        for s in ds_s:
            lab = np.squeeze(read(s["lab"]))
            v = voxel_of(s) or (1.0, 1.0, 1.0)
            d = eq_diam_um(lab, v)
            if d:
                dd.append(d / v[1])            # 换成 xy 像素
        scale[ds] = TARGET_D / float(np.median(dd))
        print(f"  {ds:11s} 细胞核中位直径 {np.median(dd):6.1f} xy 像素（数据集参考值；切片时按样本单独计算）", flush=True)

    # 3) 切片
    jobs = [(s, scale.get(s["ds"], 1.0), tr_dir, i) for i, s in enumerate(train)]
    rows = []
    with Pool(a.workers) as p:
        for r in p.imap_unordered(work, jobs, chunksize=1):
            rows.extend(r)
    errs = [r for r in rows if "error" in r]
    for r in errs[:10]:
        print("  ERR", r)
    ok = [r for r in rows if "error" not in r]
    agg = defaultdict(lambda: [0, 0])
    for r in ok:
        agg[r["ds"]][0] += 1; agg[r["ds"]][1] += r["n"]
    for k, (n, inst) in sorted(agg.items()):
        print(f"  {k:11s} 训练切片 {n:5d}  细胞核实例 {inst:7d}")
    json.dump(dict(scale=scale, slices=ok, errors=errs), open(f"{a.out}/train_manifest.json", "w"), indent=1)

    # 4) 测试集清单（不切片，保留 3D；大体积记录裁剪框）
    man = []
    for s in test:
        lab = np.squeeze(read(s["lab"]))
        if s.get("x_range"):
            lab = lab[..., s["x_range"][0]:s["x_range"][1]]
        box = best_box(lab, a.test_box)
        man.append({k: v for k, v in s.items() if k in ("ds", "id", "img", "lab", "x_range")} |
                   {"voxel": voxel_of(s), "box": box, "scale": scale.get(s["ds"])})
    json.dump(man, open(f"{a.out}/test_manifest.json", "w"), indent=1, ensure_ascii=False)
    tc = defaultdict(int)
    for m in man:
        tc[m["ds"]] += 1
    print("  测试体积:", dict(tc))


if __name__ == "__main__":
    main()
