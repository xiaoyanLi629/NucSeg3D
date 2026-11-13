"""把 iso30 体积库（共享盘 iso30.tar，零拷贝读取）转成 nnU-Net v2 原始数据集 Dataset801_NucIso30。

图像：iso30 已归一化的 float16 体积，原样写成 zlib 压缩的 float32 TIFF（nnU-Net 的 Tiff3DIO 读；体素间距写 1,1,1，
      iso30 本身就是各向同性网格，核径约 30 像素）。nnU-Net 再按它自己的 ZScore（逐图像）规范化。
标签：三类语义标签 0 = 背景，1 = 核内部，2 = 核边界（每个核的内侧边界，厚 BW 个体素：与背景或别的核相邻、
      6 邻接菱形结构元迭代 BW 次的范围内）。实例由推理后处理恢复（见 nnunet_infer.py）。
病例：iso30 index 里 split=train 的 910 个体积（与我们三维模型完全相同的训练体积，mesospim 不在其中）
      加上 split=val 的 88 个体积（只放进 nnU-Net 第 0 折的验证列表，不参与训练）。
另写 splits_final.json（第 0 折：train = 910 个训练体积，val = 88 个验证体积），规划完成后拷进 nnUNet_preprocessed。
用法：python make_nnunet_raw.py --out <nnUNet_raw> [--workers 16] [--bw 2] [--limit N]（--limit 只用于冒烟测试）"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import tifffile
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iso30tar import Iso30  # noqa: E402

DS_NAME = "Dataset801_NucIso30"


def case_id(e):
    return e["ds"] + "__" + os.path.basename(e["img"])[: -len("_img.npy")]


def three_class(lab, bw):
    st = ndi.generate_binary_structure(3, 1)
    fp = ndi.iterate_structure(st, bw)
    mx = ndi.grey_dilation(lab, footprint=fp, mode="nearest")
    mn = ndi.grey_erosion(lab, footprint=fp, mode="constant", cval=0)
    fg = lab > 0
    bd = fg & ((mx != lab) | (mn != lab))
    out = fg.astype(np.uint8)
    out[bd] = 2
    return out


def work(args):
    e, out, bw = args
    cid = case_id(e)
    fi = f"{out}/imagesTr/{cid}_0000.tif"; fl = f"{out}/labelsTr/{cid}.tif"
    if os.path.exists(fl) and os.path.exists(fi):
        return cid, 0.0, None
    t0 = time.time()
    S = Iso30()
    img = S.load(e["img"]).astype(np.float32)
    lab = S.load(e["lab"])
    lab = lab.astype(np.int32 if lab.dtype == np.uint32 else lab.dtype)
    sem = three_class(lab, bw)
    spc = {"spacing": [1.0, 1.0, 1.0]}
    tifffile.imwrite(fi + ".part.tif", img, compression="zlib", compressionargs={"level": 1}, predictor=True, bigtiff=True)
    tifffile.imwrite(fl + ".part.tif", sem, compression="zlib", compressionargs={"level": 6}, bigtiff=True)
    os.replace(fi + ".part.tif", fi); os.replace(fl + ".part.tif", fl)
    json.dump(spc, open(fi[: -len("_0000.tif")] + ".json", "w"))
    json.dump(spc, open(fl[: -len(".tif")] + ".json", "w"))
    frac = np.bincount(sem.ravel(), minlength=3) / sem.size
    return cid, time.time() - t0, frac.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="nnUNet_raw 目录")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--bw", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = f"{a.out}/{DS_NAME}"
    os.makedirs(f"{out}/imagesTr", exist_ok=True); os.makedirs(f"{out}/labelsTr", exist_ok=True)
    idx = [e for e in Iso30().index() if e["split"] in ("train", "val")]
    tr = [e for e in idx if e["split"] == "train" and e["ds"] != "mesospim"]
    va = [e for e in idx if e["split"] == "val"]
    if a.limit:
        rng = np.random.default_rng(0)
        tr = [tr[i] for i in sorted(rng.choice(len(tr), a.limit, replace=False))]
        va = va[:: max(1, len(va) // 4)][:4]
    todo = sorted(tr + va, key=lambda e: -np.prod(e["shape"]))       # 大的先做
    print(f"train {len(tr)}  val {len(va)}", flush=True)
    stats = {}
    t0 = time.time()
    with Pool(a.workers) as p:
        for k, (cid, sec, frac) in enumerate(p.imap_unordered(work, [(e, out, a.bw) for e in todo], chunksize=1)):
            stats[cid] = frac
            if k % 20 == 0 or k == len(todo) - 1:
                print(f"[{k + 1}/{len(todo)}] {cid} {sec:.0f}s frac={frac}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    dj = {"channel_names": {"0": "fluorescence"}, "labels": {"background": 0, "interior": 1, "boundary": 2},
          "numTraining": len(todo), "file_ending": ".tif", "overwrite_image_reader_writer": "Tiff3DIO",
          "description": "NucSeg3D iso30 (isotropic, nucleus diameter ~30 px); 3-class interior/boundary; train = iso30 train (910), val = iso30 val (88)",
          "boundary_width": a.bw}
    json.dump(dj, open(f"{out}/dataset.json", "w"), indent=1)
    splits = [{"train": sorted(case_id(e) for e in tr), "val": sorted(case_id(e) for e in va)}]
    json.dump(splits, open(f"{out}/splits_final.json", "w"), indent=1)
    json.dump({"ds": {case_id(e): e["ds"] for e in todo}, "class_frac": stats}, open(f"{out}/case_meta.json", "w"))
    print("DONE", len(todo), f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
