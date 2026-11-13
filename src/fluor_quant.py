"""在原图上做逐细胞、分区的荧光定量（相对量）。

区域：细胞核 nuc、细胞膜 mem（细胞边界向内 1 µm 的薄壳，不含核）、细胞质 cyto（细胞减去核与膜）、整个细胞 cell。
步骤：逐层背景（类器官外、留 5 µm 余量的空白区中位数）扣除 → 深度衰减校正（用核通道拟合）→ 体积分 → 相对量。
用法：python fluor_quant.py --seg results/pdac_pipeline/ft_v4_a_f300 --out results/pdac_fluor/ft_v4_a_f300
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import argparse, json, os
import numpy as np, pandas as pd, tifffile
from scipy import ndimage as ndi

C1 = f"{_NSP_SHARED}/data/samples_3dcellscope/PDAC-C1.tif"
C2 = f"{_NSP_SHARED}/data/samples_3dcellscope/PDAC-C2.tif"
VOX = (1.0, 0.345, 0.345)
SAT = {"C1": 4095, "C2": 4095}


def background(img, org, vox, margin_um=5.0):
    """逐层背景：类器官外再向外留 margin 的区域的中位数与标准差。"""
    r = int(round(margin_um / vox[1]))
    far = ~np.stack([ndi.binary_dilation(s, iterations=r) for s in org])
    med = np.array([np.median(img[z][far[z]]) for z in range(img.shape[0])])
    sd = np.array([np.std(img[z][far[z]]) for z in range(img.shape[0])])
    return med.astype(np.float32), sd.astype(np.float32)


def compartments(nuc, cell, vox, shell_um=1.0):
    """返回区域标签图：每个体素属于哪个细胞 + 哪一区（1 核 2 膜 3 质）。"""
    bnd = np.zeros(cell.shape, bool)
    for ax in range(3):
        a = np.swapaxes(cell, 0, ax)
        d = a[1:] != a[:-1]
        b = np.zeros(a.shape, bool); b[1:] |= d; b[:-1] |= d
        bnd |= np.swapaxes(b, 0, ax)
    bnd &= cell > 0
    dist = ndi.distance_transform_edt(~bnd, sampling=vox)
    comp = np.zeros(cell.shape, np.uint8)
    comp[cell > 0] = 3
    comp[(cell > 0) & (dist <= shell_um)] = 2
    comp[nuc > 0] = 1
    return comp


def path_length(org, vox, from_top):
    """每个体素上方（光进来一侧）穿过的类器官组织厚度，µm。"""
    o = org.astype(np.float32)
    if not from_top:
        o = o[::-1]
    L = (np.cumsum(o, 0) - o) * vox[0]
    return L if from_top else L[::-1]


def fit_attenuation(df, L_col, y_col="nuc_C1_bg"):
    """log(亮度) ~ a + b·x，稳健（两轮去掉 3 倍 MAD 外点）。返回 a、b、R²、外点数。"""
    x = df[L_col].values; y = np.log(np.clip(df[y_col].values, 1, None))
    keep = np.ones(len(x), bool)
    for _ in range(2):
        b, a = np.polyfit(x[keep], y[keep], 1)
        r = y - (a + b * x); mad = np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r) < 3 * 1.4826 * mad
    res = y[keep] - (a + b * x[keep])
    r2 = 1 - res.var() / y[keep].var()
    return float(a), float(b), float(r2), int((~keep).sum())


def per_cell(img, comp, cell, ids, vvox):
    out = {}
    for k, name in [(1, "nuc"), (2, "mem"), (3, "cyto")]:
        m = comp == k
        lab = np.where(m, cell, 0)
        s = ndi.sum(img, lab, ids); n = ndi.sum(m, lab, ids)
        out[name] = (np.asarray(s), np.asarray(n))
    s = sum(v[0] for v in out.values()); n = sum(v[1] for v in out.values())
    out["cell"] = (s, n)
    return {k: (s * vvox, n * vvox, s / np.maximum(n, 1)) for k, (s, n) in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shell", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=5.0, help="背景区离类器官的留距 µm")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    raw = {"C1": tifffile.imread(C1).astype(np.float32), "C2": tifffile.imread(C2).astype(np.float32)}
    nuc = tifffile.imread(f"{a.seg}/nuclei_mask.tif").astype(np.int32)
    cell = tifffile.imread(f"{a.seg}/cell_mask.tif").astype(np.int32)
    org = tifffile.imread(f"{a.seg}/organoid_mask.tif") > 0
    vvox = float(np.prod(VOX))
    ids = np.unique(cell); ids = ids[ids > 0]
    comp = compartments(nuc, cell, VOX, a.shell)

    info = {"shell_um": a.shell, "margin_um": a.margin, "n_cells": int(len(ids))}
    bg = {}
    for c, im in raw.items():
        med, sd = background(im, org, VOX, a.margin)
        bg[c] = (med, sd)
        info[f"bg_{c}_median"] = float(np.median(med)); info[f"bg_{c}_sd"] = float(np.median(sd))
        info[f"sat_frac_{c}_in_cells"] = float((im[cell > 0] >= SAT[c]).mean())
    sub = {c: raw[c] - bg[c][0][:, None, None] for c in raw}

    df = pd.DataFrame({"cell_id": ids})
    cz = ndi.mean(np.broadcast_to(np.arange(nuc.shape[0])[:, None, None], nuc.shape).astype(np.float32), nuc, ids)
    df["z_um"] = np.asarray(cz) * VOX[0]
    for c in raw:
        r = per_cell(sub[c], comp, cell, ids, vvox)
        for k, (integ, vol, mean) in r.items():
            df[f"{k}_vol_um3"] = vol
            df[f"{k}_{c}_bg"] = mean
    # 光路上的组织厚度：两种方向都试，取衰减更明显（斜率更负）且拟合更好的那个
    best = None
    for top in (True, False):
        L = path_length(org, VOX, top)
        df[f"L_{'top' if top else 'bot'}"] = np.asarray(ndi.mean(L, nuc, ids))
    df["z_rel"] = df["z_um"] - df["z_um"].min()
    fits = {col: fit_attenuation(df, col) for col in ["L_top", "L_bot", "z_rel"]}
    info["fits"] = {k: dict(zip(["a", "b_per_um", "r2", "n_outliers"], v)) for k, v in fits.items()}
    best = max(fits, key=lambda k: fits[k][2] if fits[k][1] < 0 else -1)
    info["attenuation_model"] = best
    if best == "z_rel":
        Lvox = np.broadcast_to((np.arange(nuc.shape[0]) * VOX[0] - df["z_um"].min())[:, None, None], nuc.shape)
    else:
        Lvox = path_length(org, VOX, best == "L_top")
    # 每个通道各用自己的衰减斜率：C1 用细胞核亮度拟合；C2 用整个细胞的亮度拟合。
    # C2 的拟合假定 actin 总量与光路厚度无关；光路厚度与“到类器官边界的距离”基本不相关时这一假定才说得过去，见 stats。
    ref = {"C1": "nuc_C1_bg", "C2": "cell_C2_bg"}
    info["channel_fits"] = {}
    for c in raw:
        a0, b0, r2, no = fit_attenuation(df, best, ref[c])
        info["channel_fits"][c] = {"a": a0, "b_per_um": b0, "r2": r2, "n_outliers": no}
        gain = np.exp(-min(b0, 0.0) * Lvox).astype(np.float32)          # 乘回去的增益，L=0 处为 1
        df[f"gain_{c}"] = np.exp(-min(b0, 0.0) * df[best])
        info[f"gain_{c}_p50_p99_in_cells"] = [float(np.percentile(gain[cell > 0], 50)), float(np.percentile(gain[cell > 0], 99))]
        r = per_cell(sub[c] * gain, comp, cell, ids, vvox)
        for k, (integ, vol, mean) in r.items():
            df[f"{k}_{c}_corr"] = mean
            df[f"{k}_{c}_int"] = integ
    df["deep_low_conf"] = df["gain_C1"] > 4          # 核通道增益超过 4 倍：信号已衰减到四分之一以下，可信度低
    info["frac_deep_low_conf"] = float(df["deep_low_conf"].mean())
    # 衰减校正前后，核通道均值的离散程度（变异系数）与和深度的相关
    from scipy.stats import spearmanr
    for tag in ("bg", "corr"):
        v = df[f"nuc_C1_{tag}"]
        info[f"nucC1_{tag}_cv"] = float(v.std() / v.mean())
        info[f"nucC1_{tag}_rho_depth"] = float(spearmanr(v, df[best]).correlation)
        v2 = df[f"cell_C2_{tag}"]
        info[f"cellC2_{tag}_rho_depth"] = float(spearmanr(v2, df[best]).correlation)
    # 派生指标（相对量）
    df["ratio_C2cell_C1nuc"] = df["cell_C2_int"] / df["nuc_C1_int"]
    df["mem_over_cyto_C2"] = df["mem_C2_corr"] / df["cyto_C2_corr"]
    for col in ["nuc_C1_int", "cell_C2_int", "mem_C2_corr", "cyto_C2_corr", "ratio_C2cell_C1nuc", "mem_over_cyto_C2"]:
        df[f"{col}_pct"] = 100 * df[col] / df[col].median()
    # 到类器官边界的距离（从三层分析结果里取）
    cells_csv = pd.read_csv(f"{a.seg}/cells.csv").set_index("cell_id")
    df["dist_to_border_um"] = df["cell_id"].map(cells_csv["dist_to_border_um"])
    df["old_cell_ch_mean"] = df["cell_id"].map(cells_csv["cell_ch_mean"])   # 三层分析里在“匹配过直方图”的图上测的
    df["old_nuc_ch_mean"] = df["cell_id"].map(cells_csv["nuc_ch_mean"])
    info["bg_profile"] = {c: bg[c][0].tolist() for c in raw}
    df.to_csv(f"{a.out}/cells_fluor.csv", index=False)
    json.dump(info, open(f"{a.out}/fluor_info.json", "w"), indent=1)
    tifffile.imwrite(f"{a.out}/compartments.tif", comp, compression="zlib")
    print(json.dumps({k: v for k, v in info.items() if k != "bg_profile"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
