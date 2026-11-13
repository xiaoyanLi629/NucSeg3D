"""Cellpose-SAM 微调数据登记表：枚举每套原始数据的 (图像, 标签, 体素) 样本，并提供统一读取接口。

每个样本是一个 dict：
  ds       数据集名
  id       样本名
  kind     "3d" 或 "2d"
  img/lab  文件路径；h5 用 "路径::数据集键" 表示
  voxel    (z, y, x) µm；未知时为 None（切片时按细胞核像素直径来缩放）
  split    原数据自带的划分（train/val/test），没有则为 None
用法: python cpft_data.py   打印清单并逐个检查（尺寸、实例数、标签类型）
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import glob
import json
import os
import re
import sys

import numpy as np
import tifffile

RAW = f"{_NSP_ROOT}/data/raw"


def _nis3d_voxel(info):
    """Info.txt 的 Resolution 行形如 “0.43 um x 0.43 um x 2.5 um”（x, y, z 顺序）。"""
    if not os.path.exists(info):
        return None
    t = open(info, errors="ignore").read()
    m = re.search(r"Resolution:\s*\n?\s*([\d.]+)\s*um\s*x\s*([\d.]+)\s*um\s*x\s*([\d.]+)\s*um", t)
    return (float(m.group(3)), float(m.group(2)), float(m.group(1))) if m else None


def samples():
    S = []
    # 线虫 L1：train/val/test 已分好，0.116×0.116×0.122 µm
    for sp in ("train", "val", "test"):
        for f in sorted(glob.glob(f"{RAW}/celegans/c_elegans_nuclei/{sp}/images/*.tif")):
            S.append(dict(ds="celegans", id=os.path.basename(f)[:-4], kind="3d", img=f,
                          lab=f.replace("/images/", "/masks/"), voxel=(0.122, 0.116, 0.116), split=sp))
    # Cellos 肿瘤类器官：Opera Phenix，xy 0.598 µm，z 5 µm（3DCellScope models_info 中 Cellos 的设定）
    cdir = f"{RAW}/cellos/Cellos-1.0.0/docs/cellos_annotated_images"
    cmask = {os.path.basename(x).lower(): x for x in glob.glob(f"{cdir}/mask/*.tif")}   # Hoechst 的掩码文件名是小写
    for f in sorted(glob.glob(f"{cdir}/images/*.tif")):
        m = cmask.get("mask_" + os.path.basename(f).lower(), "")
        S.append(dict(ds="cellos", id=os.path.basename(f)[:-4], kind="3d", img=f, lab=m, voxel=(5.0, 0.598, 0.598), split=None))
    # 隐窝支架 DAPI：已是 2D Cellpose 切片，只取有 _masks 的
    for m in sorted(glob.glob(f"{RAW}/crypt/*/*_masks.tif")):
        S.append(dict(ds="crypt", id=os.path.basename(m)[:-10], kind="2d", img=m.replace("_masks.tif", ".tif"), lab=m, voxel=None, split=None))
    # Go-Nuclear：h5，raw/clear 与 raw/noisy 两个版本都用，标签 label/gold
    for f in sorted(glob.glob(f"{RAW}/gonuclear/*/*/*/3d_all_in_one/*.h5")):
        for ver in ("clear", "noisy"):
            S.append(dict(ds="gonuclear", id=f"{os.path.basename(f)[:-3]}_{ver}", kind="3d", img=f"{f}::raw/{ver}",
                          lab=f"{f}::label/gold", voxel="h5", split=None))
    # NIS3D：每个样本 data / GroundTruth / ConfidenceScore
    for d in sorted(glob.glob(f"{RAW}/nis3d/NIS3D/NIS3D/*/")):
        lab = d + "GroundTruth.tif" if os.path.exists(d + "GroundTruth.tif") else d + "gt.tif"           # MusMusculus_2 命名不同
        conf = d + "ConfidenceScore.tif" if os.path.exists(d + "ConfidenceScore.tif") else d + "scoreOfConfidence.tif"
        S.append(dict(ds="nis3d", id=os.path.basename(d.rstrip("/")), kind="3d", img=d + "data.tif", lab=lab,
                      conf=conf, voxel=_nis3d_voxel(d + "Info.txt"), split=None))
    # NucVerse3D：三个子集，标签命名各不相同
    for sub, vox in (("2_livernuclei", (0.3, 0.3, 0.3)), ("7_liver_hcc_dataset", (0.3, 0.3, 0.3)), ("6_Drosophila_denoised", (0.3, 0.15, 0.15))):
        for sp in ("train", "test"):
            masks = glob.glob(f"{RAW}/nucverse3d/raw/{sub}/{sp}/masks/*.tif")
            for f in sorted(glob.glob(f"{RAW}/nucverse3d/raw/{sub}/{sp}/images/*.tif")):
                stem = os.path.basename(f)[:-4].replace("Denoised-", "")
                cand = [m for m in masks if os.path.basename(m).startswith(stem)]
                if cand:
                    S.append(dict(ds="nucverse3d", id=f"{sub}/{os.path.basename(f)[:-4]}", kind="3d", img=f, lab=sorted(cand)[0], voxel=vox, split=sp))
    # Purdue：子体积 xxx.tif + xxx_gt.tif（跳过 synthetic）
    for g in sorted(glob.glob(f"{RAW}/purdue/ground_truth_and_synthetic/*/*/*_gt.tif")):
        if "/synthetic/" in g:
            continue
        S.append(dict(ds="purdue", id=os.path.basename(g)[:-7], kind="3d", img=g.replace("_gt.tif", ".tif"), lab=g, voxel=None, split=None))
    # Tapenade：trainval 64³ 小块 + 两位标注者的大块（已重采样为 0.62 µm 各向同性）
    for sp in ("train", "val"):
        for f in sorted(glob.glob(f"{RAW}/tapenade/stardist_training_datasets/trainval_dataset/{sp}/imgs/*.tif")):
            S.append(dict(ds="tapenade", id=f"{sp}_{os.path.basename(f)[:-4]}", kind="3d", img=f, lab=f.replace("/imgs/", "/masks/"),
                          voxel=(0.62, 0.62, 0.62), split=sp))
    A = f"{RAW}/tapenade/stardist_training_datasets"
    for i in (1, 2, 3):
        S.append(dict(ds="tapenade", id=f"annotator1_image{i}", kind="3d", img=f"{A}/annotator1/image{i}.tif", lab=f"{A}/annotator1/label{i}.tif",
                      voxel=(0.62, 0.62, 0.62), split="test"))
    S.append(dict(ds="tapenade", id="annotator2_block", kind="3d", img=f"{A}/annotator2/data_block.tif", lab=f"{A}/annotator2/annotated_block.tif",
                  voxel=(0.62, 0.62, 0.62), split="test"))
    for k in range(17):   # square.tif 是 17 个 90×192×192 的块叠在一起
        S.append(dict(ds="tapenade", id=f"annotator2_square{k:02d}", kind="3d", img=f"{A}/annotator2/square.tif#{k}",
                      lab=f"{A}/annotator2/square_annotations_gt.tif#{k}", voxel=(0.62, 0.62, 0.62), split="test"))
    # BlastoSPIM：images/X_image_0001.npy 与 masks/X_masks_0001.npy 成对，0.208×0.208×2.0 µm
    for f in sorted(glob.glob(f"{RAW}/blastospim/*/*/images/*_image_*.npy")):
        m = f.replace("/images/", "/masks/").replace("_image_", "_masks_")
        if os.path.exists(m):
            part = f.split("/blastospim/")[1].split("/")[0]          # 如 BlastoSPIM1_test_lowSNR
            sp = "test" if "_test_" in part else ("val" if part.endswith("_val") else "train")
            S.append(dict(ds="blastospim", id=f"{part}/{os.path.basename(f)[:-4]}", kind="3d", img=f, lab=m,
                          voxel=(2.0, 0.208, 0.208), split=sp))
    # ---- 第三批 ----
    # NISNet3D 斑马鱼胚胎：image_cropped_A_B.tif 与 labelled_A_B.nrrd 成对；nrrd 为 (x,y,z) 轴序，
    # 标签是 1–6 的“着色”（相邻核颜色不同），每种颜色内的连通域才是一个核；Embryo4 没有标签。体素未知，之后按真值形状估计。
    for e in ("Embryo1", "Embryo2", "Embryo3"):
        pairs = {}
        for lb in sorted(glob.glob(f"{RAW}/nisnet_zf/{e}/*abel*.nrrd")):
            m = re.findall(r"(\d+)_(\d+)", os.path.basename(lb))
            if not m:
                continue
            key = (int(m[0][0]), int(m[0][1]))
            if key not in pairs or "_gt" in os.path.basename(lb):      # 同一块有 _gt 版本时优先用它
                pairs[key] = lb
        for (a, b), lb in sorted(pairs.items()):
            im = f"{RAW}/nisnet_zf/{e}/image_cropped_{a:04d}_{b:04d}.tif"
            if os.path.exists(im):
                S.append(dict(ds="nisnet_zf", id=f"{e}_{a:04d}_{b:04d}", kind="3d", img=im, lab=lb + "@color",
                              voxel=None, split="test" if e == "Embryo3" else "train"))
    # 小鼠围着床期胚胎（PlantSeg 团队）：h5 的 raw / label；说明写体素 0.207×0.207×1 或 0.23×0.23×1 µm，取 0.215
    for sp in ("train", "test"):
        for f in sorted(glob.glob(f"{RAW}/mouse_embryo/Nuclei/{sp}/*.h5")):
            S.append(dict(ds="mouse_embryo", id=os.path.basename(f)[:-3], kind="3d", img=f"{f}::raw", lab=f"{f}::label",
                          voxel=(1.0, 0.215, 0.215), split=sp))
    # BBBC050（QCANet）：小鼠早期胚胎，GroundTruth_QCANet 为实例标签（NSN/NDN 是前景与核中心，不用）；体素未知，按真值估计
    for sp in ("train", "test"):
        for g in sorted(glob.glob(f"{RAW}/bbbc050/GroundTruth/{sp}/GroundTruth_QCANet/*.tif")):
            im = f"{RAW}/bbbc050/Images/{sp}/Images/{os.path.basename(g)}"
            if os.path.exists(im):
                S.append(dict(ds="bbbc050", id=f"{sp}_{os.path.basename(g)[:-4]}", kind="3d", img=im, lab=g, voxel=None, split=sp))
    # BBBC032：小鼠囊胚一个大体积，C3 为 Hoechst；体素 0.101×0.101×0.5 µm；只有一个样本，作测试
    S.append(dict(ds="bbbc032", id="BMP4blastocyst", kind="3d", img=f"{RAW}/bbbc032/BMP4blastocystC3.tif",
                  lab=f"{RAW}/bbbc032/BBBC032_v1_DatasetGroundTruth.tif", voxel=(0.5, 0.101, 0.101), split="test"))
    # Parhyale 钩虾胚胎：50 个时间点叠成 (t,z,y,x)，标注 6 个时间点；体素 0.277×0.277×2.16 µm；后两个时间点作测试
    for tp in (1, 11, 21, 31, 41, 50):
        S.append(dict(ds="parhyale", id=f"tp{tp:02d}", kind="3d", img=f"{RAW}/parhyale/Parhyale_H2B-EGFP_images_tp01-50.tif#{tp - 1}",
                      lab=f"{RAW}/parhyale/Parhyale_H2B-EGFP_instance-segmentation-labels_tp{tp:02d}.tif",
                      voxel=(2.1605, 0.2768, 0.2768), split="test" if tp >= 41 else "train"))
    # 大鼠脑（EPFL BIOP）：4 个人工标注的原始块；另外 16 个 64³ 小块是它们切出来的，重复，不用。体素未知
    for f in sorted(glob.glob(f"{RAW}/ratbrain/images/_CCK*.tif")):
        if re.search(r"\.tif_\d_\d\.tif$", f):
            continue
        S.append(dict(ds="ratbrain", id=re.sub(r"[^A-Za-z0-9]+", "_", os.path.basename(f)[:-4]).strip("_"), kind="3d", img=f,
                      lab=f.replace("/images/", "/masks/"), voxel=None, split="train"))
    # CellSeg3D mesoSPIM 透明化小鼠脑：c1–c5 与 visual；visual 作测试。体素未知
    M = f"{RAW}/mesospim/DATASET_WITH_GT"
    for n in ("c1", "c2", "c3", "c4", "c5", "visual"):
        lab = f"{M}/labels/{n}labels_new_label.tif" if n != "visual" else f"{M}/labels/visual_gt_new_label.tif"
        img = f"{M}/{n}image.tif" if n != "visual" else f"{M}/visual.tif"
        S.append(dict(ds="mesospim", id=n, kind="3d", img=img, lab=lab, voxel=None, split="test" if n == "visual" else "train"))
    return S


def _color_to_instances(a):
    """“着色”标签（相邻核颜色不同）→ 实例：每种颜色内按 3D 连通域拆开。"""
    from scipy import ndimage as ndi
    out = np.zeros(a.shape, np.uint32); nxt = 0
    for v in np.unique(a):
        if v == 0:
            continue
        cc, n = ndi.label(a == v)
        out[cc > 0] = cc[cc > 0] + nxt; nxt += n
    return out


def read(path):
    """读取 tif / h5::key / npy(.gz)，返回 numpy 数组。“路径#k”表示取第 0 轴第 k 个块。"""
    if "#" in path and path.rsplit("#", 1)[1].isdigit():   # 文件名本身可能含 #（大鼠脑），只认结尾的 #数字
        p, k = path.rsplit("#", 1)
        return read(p)[int(k)]
    if path.endswith(".nrrd@color") or path.endswith(".nrrd"):
        import nrrd
        a, _ = nrrd.read(path.replace("@color", ""))
        a = np.ascontiguousarray(a.transpose(2, 1, 0))               # (x,y,z) → (z,y,x)
        return _color_to_instances(a) if path.endswith("@color") else a
    if "::" in path:
        import h5py
        f, k = path.split("::")
        with h5py.File(f, "r") as h:
            return h[k][()]
    if path.endswith(".npy.gz"):
        import gzip
        with gzip.open(path, "rb") as fh:
            return np.load(fh)
    if path.endswith(".npy"):
        return np.load(path)
    with tifffile.TiffFile(path) as t:
        a = t.asarray()
        if a.ndim == 2 and len(t.pages) > 1:          # Cellos：逐层分开存的多页 tif
            a = np.stack([p.asarray() for p in t.pages])
    return a


def voxel_of(s):
    if s["voxel"] == "h5":
        import h5py
        f, k = s["img"].split("::")
        with h5py.File(f, "r") as h:
            return tuple(float(v) for v in h[k].attrs["element_size_um"])
    return s["voxel"]


if __name__ == "__main__":
    S = samples()
    only = sys.argv[1:]
    rows = []
    for s in S:
        if only and s["ds"] not in only:
            continue
        try:
            img, lab = np.squeeze(read(s["img"])), np.squeeze(read(s["lab"]))
            ids = np.unique(lab)
            r = dict(ds=s["ds"], id=s["id"], split=s["split"], img_shape=list(img.shape), lab_shape=list(lab.shape),
                     img_dtype=str(img.dtype), lab_dtype=str(lab.dtype), n_inst=int((ids > 0).sum()),
                     binary=bool(len(ids) <= 2), match=bool(img.shape == lab.shape), voxel=voxel_of(s))
        except Exception as e:
            r = dict(ds=s["ds"], id=s["id"], error=repr(e)[:150])
        rows.append(r)
    os.makedirs(f"{_NSP_ROOT}/data", exist_ok=True)
    json.dump(rows, open(f"{_NSP_ROOT}/data/inventory{'_' + '_'.join(only) if only else ''}.json", "w"), indent=1)
    from collections import defaultdict
    agg = defaultdict(lambda: dict(n=0, inst=0, bad=0, binary=0, err=0, shapes=set(), vox=set()))
    for r in rows:
        a = agg[r["ds"]]; a["n"] += 1
        if "error" in r:
            a["err"] += 1; print("ERR", r["ds"], r["id"], r["error"]); continue
        a["inst"] += r["n_inst"]; a["bad"] += (not r["match"]); a["binary"] += r["binary"]
        a["shapes"].add(tuple(r["img_shape"])); a["vox"].add(str(r["voxel"]))
    for k, a in agg.items():
        sh = sorted(a["shapes"])
        print(f"{k:11s} 样本 {a['n']:4d}  实例 {a['inst']:7d}  尺寸不符 {a['bad']}  二值标签 {a['binary']}  出错 {a['err']}  "
              f"示例尺寸 {sh[0] if sh else '-'}{' …' if len(sh) > 1 else ''}  体素 {', '.join(sorted(a['vox']))[:60]}")
