"""类器官三层分析流程：细胞核 → 细胞 → 整个类器官，并计算形态与拓扑指标。

照 Ong et al. 2025（Nature Methods，3DCellScope）方法部分实现，细胞核分割可以换成任意模型的结果：
  1. 预处理（可选）：逐层直方图匹配到 99 分位最亮的一层，校正深层信号衰减（论文 “Image preprocessing”）；
     只用于类器官轮廓、细胞分割和亮度指标，细胞核分割始终在原图上做。
  2. 类器官轮廓：各通道归一化后取平均，逐层 CLAHE，三维高斯平滑，Otsu 阈值 × 0.8，只保留最大连通块，
     再逐层做膨胀 → 补洞 → 腐蚀得到光滑掩码（论文 “Organoid segmentation by the Otsu method”）。
  3. 细胞：以细胞核为种子，在细胞骨架（actin）通道上做灰度分水岭；扩张限制在类器官掩码内、
     且离细胞核边界不超过 14 µm；最后用球形核做三维中值滤波平滑（论文 “Cell segmentation by seeded grayscale watershed”）。
  4. 指标：每个细胞核、细胞、类器官的体积、表面积、球形度、等效椭球三轴、伸长度、质心、到类器官边界的距离、
     各通道亮度统计；细胞核的拓扑指标——半径 r 邻域内细胞核坐标拟合椭球（轴长、长形比/扁形比、盘状/棒状/球状），
     最近邻距离、局部密度与 Ripley K 比值（论文 “Extraction of morphological and topological descriptors”）。
输出：<out>/nuclei_mask.tif、cell_mask.tif、organoid_mask.tif、nuclei.csv、cells.csv、organoid.csv、params.json

用法:
  python organoid_pipeline.py --nuc-img C1.tif --cell-img C2.tif --nuc-mask nuclei.tif --voxel 1.0 0.345 0.345 --out 目录
  （或用 --model 模型路径 现场跑 Cellpose 分割细胞核，此时 --diameter 给细胞核直径像素）
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import exposure, filters, measure, morphology, segmentation


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def norm01(a):
    lo, hi = np.percentile(a, (0.5, 99.8))
    return np.clip((a.astype(np.float32) - lo) / (hi - lo + 1e-8), 0, 1)


def correct_bleaching(img):
    """逐层直方图匹配到 99 分位最亮的那一层。"""
    p99 = np.percentile(img.reshape(img.shape[0], -1), 99, axis=1)
    ref = img[int(np.argmax(p99))]
    return np.stack([exposure.match_histograms(s, ref) for s in img]).astype(np.float32)


def ball(rz, ry, rx):
    z, y, x = np.ogrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
    return (z / max(rz, 1e-6)) ** 2 + (y / ry) ** 2 + (x / rx) ** 2 <= 1


def organoid_mask(chans, voxel, scale=0.8, disk_um=3.0):
    mean = np.mean([norm01(c) for c in chans], 0)
    eq = np.stack([exposure.equalize_adapthist(s) for s in mean]).astype(np.float32)
    sm = filters.gaussian(eq, sigma=(1.0 * voxel[1] / voxel[0], 1.0, 1.0) if voxel[0] > voxel[1] else 1.0)
    thr = filters.threshold_otsu(sm) * scale
    m = sm > thr
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    r = max(1, int(round(disk_um / voxel[1])))
    d = morphology.disk(r)
    out = np.zeros_like(m)
    for z in range(m.shape[0]):           # 论文：逐层 膨胀 → 补洞 → 腐蚀
        s = ndi.binary_dilation(m[z], d)
        s = ndi.binary_fill_holes(s)
        out[z] = ndi.binary_erosion(s, d)
    return out, float(thr)


def segment_cells(cell_img, nuclei, org, voxel, dist_max=14.0, median_um=1.0):
    elev = filters.gaussian(norm01(cell_img), sigma=(0.5 * voxel[1] / voxel[0], 1.0, 1.0))
    dist = ndi.distance_transform_edt(nuclei == 0, sampling=voxel)
    mask = (org | (nuclei > 0)) & (dist <= dist_max)
    cells = segmentation.watershed(elev, markers=nuclei, mask=mask)
    rz = max(1, int(round(median_um / voxel[0]))); rxy = max(1, int(round(median_um / voxel[1])))
    cells = ndi.median_filter(cells, footprint=ball(rz, rxy, rxy))
    cells[nuclei > 0] = nuclei[nuclei > 0]          # 细胞一定包含自己的核
    return cells


def surface_area(binary, voxel):
    if binary.sum() < 8:
        return np.nan
    try:
        v, f, _, _ = measure.marching_cubes(np.pad(binary.astype(np.float32), 1), 0.5, spacing=voxel)
        return float(measure.mesh_surface_area(v, f))
    except (RuntimeError, ValueError):
        return np.nan


def ellipsoid_axes(eigvals):
    """实心椭球的惯性张量特征值 → 三个半轴长（从大到小）。"""
    l = np.sort(np.asarray(eigvals, float))      # λ1 ≤ λ2 ≤ λ3
    a2 = 2.5 * np.array([l[1] + l[2] - l[0], l[0] + l[2] - l[1], l[0] + l[1] - l[2]])
    return np.sort(np.sqrt(np.clip(a2, 0, None)))[::-1]


def morph_table(lab, voxel, chans, chan_names, org_dist, prefix):
    props = measure.regionprops(lab, intensity_image=np.stack(chans, -1), spacing=voxel)
    rows = []
    for p in props:
        sl = p.slice
        V = float(p.area)
        A = surface_area(p.image, voxel)
        ax = ellipsoid_axes(p.inertia_tensor_eigvals)
        c = np.array(p.centroid)                       # µm（spacing 已计入）
        ci = tuple(np.clip(np.round(c / np.array(voxel)).astype(int), 0, np.array(lab.shape) - 1))
        r = {f"{prefix}_id": p.label, "volume_um3": V, "surface_um2": A,
             "sphericity": float(np.pi ** (1 / 3) * (6 * V) ** (2 / 3) / A) if A and A > 0 else np.nan,
             "equiv_diameter_um": float((6 * V / np.pi) ** (1 / 3)),
             "axis_major_um": 2 * ax[0], "axis_mid_um": 2 * ax[1], "axis_minor_um": 2 * ax[2],
             "elongation": float(ax[0] / ax[2]) if ax[2] > 0 else np.nan,
             "flatness": float(ax[1] / ax[2]) if ax[2] > 0 else np.nan,
             "centroid_z_um": c[0], "centroid_y_um": c[1], "centroid_x_um": c[2],
             "bbox_z": sl[0].stop - sl[0].start, "bbox_y": sl[1].stop - sl[1].start, "bbox_x": sl[2].stop - sl[2].start,
             "dist_to_border_um": float(org_dist[ci])}
        vals = p.image_intensity[p.image]
        for k, n in enumerate(chan_names):
            v = vals[:, k]
            r.update({f"{n}_mean": float(v.mean()), f"{n}_std": float(v.std()), f"{n}_min": float(v.min()), f"{n}_max": float(v.max())})
        rows.append(r)
    return pd.DataFrame(rows)


def topology(df, r_um, org_volume):
    """细胞核拓扑：r 邻域内细胞核坐标拟合椭球 + 密度指标。"""
    P = df[["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy()
    tree = cKDTree(P)
    n = len(P)
    lam = n / org_volume                                   # 全类器官平均细胞核密度（个/µm³）
    vsph = 4 / 3 * np.pi * r_um ** 3
    k = min(7, n)
    dnn, _ = tree.query(P, k=k)
    out = {k_: [] for k_ in ("n_neighbors_r", "local_density_per_1000um3", "ripley_k_ratio", "nn_dist_um", "mean_dist_6nn_um",
                             "nbhd_axis1_um", "nbhd_axis2_um", "nbhd_axis3_um", "nbhd_prolate", "nbhd_oblate", "nbhd_shape")}
    for i, idx in enumerate(tree.query_ball_point(P, r_um)):
        m = len(idx) - 1
        out["n_neighbors_r"].append(m)
        out["local_density_per_1000um3"].append(1000 * m / vsph)
        out["ripley_k_ratio"].append(m / (lam * vsph) if lam > 0 else np.nan)
        out["nn_dist_um"].append(float(dnn[i, 1]) if k > 1 else np.nan)
        out["mean_dist_6nn_um"].append(float(dnn[i, 1:].mean()) if k > 1 else np.nan)
        if len(idx) >= 4:
            Q = P[idx] - P[idx].mean(0)
            ev = np.sort(np.linalg.eigvalsh(Q.T @ Q / len(idx)))[::-1]
            a = np.sqrt(5 * np.clip(ev, 0, None))           # 均匀实心椭球：方差 = 半轴² / 5
            pro = a[0] / a[1] if a[1] > 0 else np.nan
            obl = a[1] / a[2] if a[2] > 0 else np.nan
            shape = "rod" if pro >= 1.5 and pro > obl else "disk" if obl >= 1.5 else "sphere"
        else:
            a = [np.nan] * 3; pro = obl = np.nan; shape = "too_few"
        out["nbhd_axis1_um"].append(a[0]); out["nbhd_axis2_um"].append(a[1]); out["nbhd_axis3_um"].append(a[2])
        out["nbhd_prolate"].append(pro); out["nbhd_oblate"].append(obl); out["nbhd_shape"].append(shape)
    for k_, v in out.items():
        df[k_] = v
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuc-img", required=True)
    ap.add_argument("--cell-img", required=True)
    ap.add_argument("--nuc-mask")
    ap.add_argument("--model", help="不给 --nuc-mask 时用此 Cellpose 模型分割细胞核")
    ap.add_argument("--diameter", type=float, default=30.0)
    ap.add_argument("--voxel", type=float, nargs=3, required=True, metavar=("DZ", "DY", "DX"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dist-max", type=float, default=14.0, help="细胞边界离细胞核边界的最大距离 µm（论文默认 14）")
    ap.add_argument("--otsu-scale", type=float, default=0.8, help="Otsu 阈值系数（论文默认 0.8）")
    ap.add_argument("--r-topo", type=float, default=25.0, help="拓扑指标的邻域半径 µm")
    ap.add_argument("--no-bleach", action="store_true", help="不做逐层直方图匹配")
    ap.add_argument("--min-nuc-um3", type=float, default=0.0, help="小于此体积的细胞核视为碎屑去掉")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    vox = tuple(a.voxel)
    c1 = tifffile.imread(a.nuc_img).astype(np.float32); c2 = tifffile.imread(a.cell_img).astype(np.float32)
    raw1 = c1                                       # 细胞核分割用原图（与 DeepStar3D 原版输出的条件一致）
    if not a.no_bleach:
        log("逐层直方图匹配"); c1, c2 = correct_bleaching(c1), correct_bleaching(c2)
    if a.nuc_mask:
        nuc = tifffile.imread(a.nuc_mask).astype(np.int32)
    else:
        from cellpose import models
        log("Cellpose 分割细胞核", a.model)
        nuc, _, _ = models.CellposeModel(gpu=True, pretrained_model=a.model).eval(
            raw1, do_3D=True, z_axis=0, diameter=a.diameter, anisotropy=vox[0] / vox[1], flow3D_smooth=1, batch_size=32)
        nuc = nuc.astype(np.int32)
    if a.min_nuc_um3 > 0:
        vol = np.bincount(nuc.ravel()) * np.prod(vox)
        nuc[np.isin(nuc, np.where(vol < a.min_nuc_um3)[0][1:])] = 0
    nuc = segmentation.relabel_sequential(nuc)[0]
    log("细胞核", int(nuc.max()))

    log("类器官轮廓")
    org, thr = organoid_mask([c1, c2], vox, a.otsu_scale)
    org_dist = ndi.distance_transform_edt(org, sampling=vox)
    out_nuc = (nuc > 0) & ~org
    log(f"类器官体积 {org.sum() * np.prod(vox):.0f} µm³；落在类器官外的细胞核体素占 {out_nuc.sum() / max((nuc > 0).sum(), 1):.1%}")

    log("细胞分割（种子分水岭）")
    cells = segment_cells(c2, nuc, org, vox, a.dist_max)
    log("细胞", len(np.unique(cells)) - 1)

    log("指标")
    names = ["nuc_ch", "cell_ch"]
    dn = morph_table(nuc, vox, [c1, c2], names, org_dist, "nucleus")
    dc = morph_table(cells, vox, [c1, c2], names, org_dist, "cell")
    dn = topology(dn, a.r_topo, org.sum() * np.prod(vox))
    cv = dict(zip(dc["cell_id"], dc["volume_um3"]))
    dn["cell_volume_um3"] = dn["nucleus_id"].map(cv)
    dn["nc_ratio"] = dn["volume_um3"] / dn["cell_volume_um3"]
    dc = dc.merge(dn[["nucleus_id", "volume_um3"]].rename(columns={"nucleus_id": "cell_id", "volume_um3": "nucleus_volume_um3"}), on="cell_id", how="left")
    Vo = float(org.sum() * np.prod(vox)); Ao = surface_area(org, vox)
    do = pd.DataFrame([{"organoid_volume_um3": Vo, "organoid_surface_um2": Ao,
                        "organoid_sphericity": float(np.pi ** (1 / 3) * (6 * Vo) ** (2 / 3) / Ao) if Ao else np.nan,
                        "organoid_equiv_diameter_um": float((6 * Vo / np.pi) ** (1 / 3)),
                        "n_nuclei": len(dn), "n_cells": len(dc), "nuclei_per_1e5um3": 1e5 * len(dn) / Vo,
                        "nuclei_volume_median_um3": float(dn["volume_um3"].median()), "cell_volume_median_um3": float(dc["volume_um3"].median()),
                        "nc_ratio_median": float(dn["nc_ratio"].median()),
                        **{f"frac_{s}": float((dn["nbhd_shape"] == s).mean()) for s in ("disk", "rod", "sphere")}}])
    tifffile.imwrite(f"{a.out}/nuclei_mask.tif", nuc.astype(np.uint16), compression="zlib")
    tifffile.imwrite(f"{a.out}/cell_mask.tif", cells.astype(np.uint16), compression="zlib")
    tifffile.imwrite(f"{a.out}/organoid_mask.tif", org.astype(np.uint8), compression="zlib")
    dn.to_csv(f"{a.out}/nuclei.csv", index=False); dc.to_csv(f"{a.out}/cells.csv", index=False); do.to_csv(f"{a.out}/organoid.csv", index=False)
    json.dump({**vars(a), "otsu_threshold": thr}, open(f"{a.out}/params.json", "w"), indent=1, ensure_ascii=False)
    log("完成", f"细胞核 {len(dn)} 个指标 {dn.shape[1]} 列；细胞 {len(dc)} 个指标 {dc.shape[1]} 列")


if __name__ == "__main__":
    main()
