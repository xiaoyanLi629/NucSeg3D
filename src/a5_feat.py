"""nc A5 第一步（GPU）：每个体积的 S1 编码器特征。设计见 $NUCSEG3D_SCRATCH/nc/gpu/DESIGN_G2.md 第 A5 节。
体积：参照分布 = S1 三维训练集，即 iso30 index.json 里 split=train 的 910 个体积（逐体积与 S1 的训练体积相同）；
      另算 88 个验证体积（作参照）与 227 个测试体积（去 bbbc032）。
每个体积只读中间 8 层（TIFF 每层一页，tifffile key=range(z0, z0+8)，z0 = Z//2 − 4；Z < 8 时取全部层并循环补足），
同样读这 8 层的标签，用 eval_cpsam.best_box 在这 8 层上找前景最多的 384×384 窗口（与评测的裁块规则同一函数）；
按这 8 层的 1–99 百分位归一化到 0..1（Cellpose 默认），xy 按 30/d 线性缩放（d = 真值 xy 等效直径：训练与验证取 iso30 的
diam_xy_px，测试取 07992b55 的 d_gt，两者都是 eq_diam_xy_px 的结果），中心裁到 256×256（不足补零）；
送进 S1 的二维编码器（含 neck），得到 (8, 256, 32, 32)，在 token 与 8 层上平均，得到 256 维特征；另存 8 层各自的 256 维（方差分析用）。
输出 <out>/feats.npz（keys、split、ds、feat、feat_slices、d）与 <out>/meta.json。
用法：python a5_feat.py --out <目录> --model <S1 目录> [--iso_index ...] [--test_diam .../g4_diam] [--limit n]"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def read_mid(path, Z, n=8):
    import tifffile
    if Z >= n:
        z0 = Z // 2 - n // 2
        idx = list(range(z0, z0 + n))
    else:
        idx = [i % Z for i in range(n)]
    u = sorted(set(idx))
    a = tifffile.imread(path, key=u)
    a = a[None] if a.ndim == 2 else a
    m = {z: a[k] for k, z in enumerate(u)}
    return np.stack([m[z] for z in idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--iso_index", default=f"{_NSP_SHARED}/nc_gpu/iso30_index.json")
    ap.add_argument("--test_diam", default=f"{_NSP_SHARED}/nc_gpu/g4_diam")
    ap.add_argument("--limit", type=int, default=0, help="冒烟测试：每个划分只取前 n 个")
    a = ap.parse_args()
    import cv2
    import torch
    from eval_cpsam import best_box
    from infer3d import Model3D
    from stddata import STD, load_manifest
    os.makedirs(a.out, exist_ok=True)
    man = {(r["ds"], r["id"]): r for s in ("train", "test") for r in load_manifest(split=s, dim="3d")}
    jobs = [(x["split"], man[(x["ds"], x["name"])], float(x["diam_xy_px"])) for x in json.load(open(a.iso_index))]
    td = {(x["ds"], x["id"]): x["d_used"] for x in json.load(open(os.path.join(a.test_diam, "rows.json")))}
    jobs += [("test", man[k], float(v)) for k, v in sorted(td.items())]
    if a.limit:
        jobs = [j for s in ("train", "val", "test") for j in [j for j in jobs if j[0] == s][:a.limit]]
    print(f"A5：{len(jobs)} 个体积（" + "，".join(f"{s} {sum(j[0] == s for j in jobs)}" for s in ("train", "val", "test")) + "）", flush=True)
    M = Model3D(a.model); net = M.net
    dev = M.device
    keys, splits, dss, F, FS, D = [], [], [], [], [], []
    t0 = time.time()
    for i, (sp, r, d) in enumerate(jobs):
        img = read_mid(os.path.join(STD, r["image"]), r["z"]).astype(np.float32)
        lab = read_mid(os.path.join(STD, r["mask"]), r["z"])
        b = best_box(lab, 384)
        if b:
            y, x = b; img = img[:, y:y + 384, x:x + 384]
        lo, hi = np.percentile(img, [1, 99]); img = (img - lo) / max(hi - lo, 1e-6)
        s = 30.0 / d
        H, W = img.shape[1:]
        nh, nw = max(1, int(round(H * s))), max(1, int(round(W * s)))
        xs = np.zeros((img.shape[0], 256, 256), np.float32)
        for k in range(img.shape[0]):
            z = cv2.resize(img[k], (nw, nh), interpolation=cv2.INTER_LINEAR)
            y0, x0 = max(0, (nh - 256) // 2), max(0, (nw - 256) // 2)
            z = z[y0:y0 + 256, x0:x0 + 256]
            py, px = (256 - z.shape[0]) // 2, (256 - z.shape[1]) // 2
            xs[k, py:py + z.shape[0], px:px + z.shape[1]] = z
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            f = net.encode(torch.from_numpy(xs)[:, None].to(dev), 1, xs.shape[0]).float()     # (8, 256, 32, 32)
        fs = f.mean((2, 3)).cpu().numpy()
        keys.append(f"{r['ds']}__{r['id']}"); splits.append(sp); dss.append(r["ds"]); F.append(fs.mean(0)); FS.append(fs); D.append(d)
        if (i + 1) % 50 == 0 or i + 1 == len(jobs):
            print(f"  {i + 1}/{len(jobs)} {keys[-1][:50]}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    np.savez_compressed(f"{a.out}/feats.npz", keys=np.array(keys), split=np.array(splits), ds=np.array(dss), feat=np.array(F, np.float32),
                        feat_slices=np.array(FS, np.float32), d=np.array(D))
    json.dump(dict(model=a.model, n=len(keys), minutes=round((time.time() - t0) / 60, 1)), open(f"{a.out}/meta.json", "w"))
    print(f"SUMMARY A5 特征 {len(keys)} 个体积，{(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
