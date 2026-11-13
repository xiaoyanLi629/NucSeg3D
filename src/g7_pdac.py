"""nc G7：用 S1（我们的模型）重跑 PDAC 类器官三层分析与荧光定量，替换旧模型（ft_v4_a）的结果。设计见 DESIGN_G2.md 第 G7 节。
不改 pdac_run.py / organoid_pipeline.py / fluor_quant.py，只在外面接一层：
  1. 直径：完整流程的免直径推理——二维种子 0 的 r2a2（eval_val --diam_mode joint --joint_rule r2a2 --iso --iso_final 的同一算法，
     尺度 {6,10,17,30,52,90,150} × 阈值 {0,−1,−2}，在原始 z 分辨率上扫尺度），本文件 r2a2_diam() 逐行照抄；
     --check_r2a2 模式在已有测试体积上与 orx_r5merged_2d_s0_test 的 d_used 核对。敏感性：旧分析用的 31.25 像素。
  2. 细胞核分割：z/xy = 1.0/0.345 = 2.90 ≥ 2，走各向同性跟流（先 1–99.8 百分位归一化、沿 z 插值到各向同性，推理与跟流，再取回原层）；
     S1 三视图（PDAC 不在单视图名单里）；阈值网格 {0,−1,−2,−3,−4,−6}，规则 T（sizeThr_dir 的 R2）选阈值，体积下限 0.15。
     对照：二维种子 0 + 规则 T、原版固定 cp−2（体积下限 0.15）。
  3. 三层分析：organoid_pipeline.py --nuc-mask（其余参数与旧的 ft_v4_a_f300 相同：dist_max 14、otsu 0.8、r_topo 25、做直方图匹配、
     min_nuc_um3 300），再 fluor_quant.py --seg；另存不加 300 µm³ 过滤的版本。
  4. 统计：各分割的核数、核体积分位数、两两一致性（metrics.evaluate；含旧的 ft_v4_a 与原版 DeepStar3D 输出），无人工真值，一致性不等于准确率。
用法：python g7_pdac.py --out <目录> [--s1 ...] [--d2 orx_v5_2d_s0] [--skip_pipeline]
      python g7_pdac.py --out <目录> --check_r2a2 bbbc050__test_Emb1_t001 cellos__EGFP_d0_flt32_108"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
C1 = f"{_NSP_SHARED}/data/samples_3dcellscope/PDAC-C1.tif"
C2 = f"{_NSP_SHARED}/data/samples_3dcellscope/PDAC-C2.tif"
VOX = (1.0, 0.345, 0.345)
OLD = {"ft_v4_a": f"{_NSP_ROOT}/results/pdac_pipeline/ft_v4_a/nuclei_mask.tif",
       "deepstar3d": f"{_NSP_ROOT}/results/pdac/deepstar3d.tif"}
SCALES = (6, 10, 17, 30, 52, 90, 150)
THRS_S = (0.0, -1.0, -2.0)
THRS = (0.0, -1.0, -2.0, -3.0, -4.0, -6.0)


def r2a2_diam(model, img, vox, scales=SCALES, thrs=THRS_S, min_obj=2, max_vox=8e8, batch=32):
    """eval_val.py joint + r2a2 + iso_final 的选直径部分（原始 z 分辨率），返回 (d_used, 选中的 (尺度, 阈值), info)。"""
    from eval_cpsam import eq_diam_xy_px
    from metrics import evaluate
    ratio = vox[0] / vox[1]
    cmask_s = lambda dP, cp, **kw: model._compute_masks((*img.shape, 1), dP, cp, **kw).squeeze()
    run_s = lambda diam: model.eval(img, do_3D=True, z_axis=0, diameter=diam, anisotropy=ratio, flow3D_smooth=1.0,
                                    batch_size=batch, compute_masks=False)[1]
    ms, info = {}, {}
    for sc in sorted(scales, reverse=True):
        nvox = img.size * (30.0 / sc) ** 3 * vox[0] / vox[1]
        if nvox > max_vox and ms:
            continue
        fl = run_s(sc)
        for t in thrs:
            m = cmask_s(fl[1], fl[2], flow_threshold=0.4, cellprob_threshold=t, min_size=15,
                        max_size_fraction=0.4, niter=int(200 / (30.0 / sc)), do_3D=True).squeeze().astype(np.int32)
            n = int(len(np.unique(m)) - 1)
            d_est = eq_diam_xy_px(m, vox) if n > 0 else float("nan")
            ms[(sc, t)] = m
            info[(sc, t)] = dict(n=n, d_est=d_est, cons=abs(np.log(d_est / sc)) if n > 0 else float("inf"))
        del fl
    scs = sorted({k[0] for k in ms})
    for t in thrs:
        agree = {(s0, s1): (evaluate(ms[(s0, t)], ms[(s1, t)])["f1@0.5"] if info[(s0, t)]["n"] and info[(s1, t)]["n"] else 0.0)
                 for s0, s1 in zip(scs[:-1], scs[1:])}
        for i, sc in enumerate(scs):
            nb = ([agree[(scs[i - 1], sc)]] if i > 0 else []) + ([agree[(sc, scs[i + 1])]] if i + 1 < len(scs) else [])
            info[(sc, t)]["stab"] = float(np.mean(nb)) if nb else 0.0
    del ms
    ok = [k for k in info if info[k]["n"] >= min_obj] or list(info)
    ok0 = [k for k in ok if k[1] == thrs[0]] or ok
    s_best = max(ok0, key=lambda k: info[k]["stab"] - info[k]["cons"])[0]
    sel = max([k for k in ok if k[0] == s_best] or [k for k in info if k[0] == s_best], key=lambda k: info[k]["stab"])
    d = info[sel]["d_est"] if np.isfinite(info[sel]["d_est"]) else sel[0]
    if img.size * (30.0 / d) ** 3 * vox[0] / vox[1] > max_vox:
        d = sel[0]
    return float(d), sel, {f"{k[0]:g}/{k[1]:g}": v for k, v in info.items()}


def segment(model, img, vox, d, rule="T", batch=32):
    """各向同性跟流 + 阈值网格；rule="T" 用规则 T 选阈值，"fixed" 用 cp−2；体积下限 0.15。返回 (标签, 记录)。"""
    from eval_val import zinterp
    ratio = vox[0] / vox[1]
    if ratio >= 2:
        Zu = int(round(img.shape[0] * ratio)); zi = np.round(np.linspace(0, Zu - 1, img.shape[0])).astype(int)
        im = img.astype(np.float32); lo, hi = np.percentile(im, [1, 99.8]); im = (im - lo) / max(hi - lo, 1e-6)
        img_in, an = zinterp(im, Zu), 1.0
    else:
        img_in, an, zi = img, ratio, None
    t0 = time.time()
    out = model.eval(img_in, do_3D=True, z_axis=0, diameter=d, anisotropy=an, flow3D_smooth=1.0, batch_size=batch, compute_masks=False)
    fl = out[1] if isinstance(out[1], (list, tuple)) else out
    dP, cp = fl[1], fl[2]
    cps = cp if zi is None else cp[zi]
    V = math.pi / 6 * d ** 3 / ratio
    masks, m = {}, {}
    for t in (THRS if rule == "T" else (-2.0,)):
        mk = model._compute_masks((*img_in.shape, 1), dP, cp, flow_threshold=0.4, cellprob_threshold=t, min_size=15,
                                  max_size_fraction=0.4, niter=int(200 / (30.0 / d)), do_3D=True).squeeze()
        mk = (mk if zi is None else mk[zi]).astype(np.int32)
        masks[t] = mk
        m[t] = float((cps > t).sum()) / max(len(np.unique(mk)) - 1, 1) / V
    if rule == "T":
        ts = sorted(m, reverse=True)
        t2 = -2.0 if m[-2.0] >= 0.7 else next((t for t in ts if t < -2 and m[t] >= 0.85), min(ts))
    else:
        t2 = -2.0
    lab = masks[t2]
    exp_vol = math.pi / 6 * d ** 3 * vox[1] / vox[0]
    cnt = np.bincount(lab.ravel()); small = np.where(cnt < 0.15 * exp_vol)[0]; small = small[small > 0]
    lab = lab.copy(); lab[np.isin(lab, small)] = 0
    from skimage.segmentation import relabel_sequential
    lab = relabel_sequential(lab)[0].astype(np.int32)
    return lab, dict(d=d, thr=t2, m={f"{k:g}": v for k, v in m.items()}, n=int(lab.max()), sec=round(time.time() - t0, 1),
                     iso=zi is not None)


def stats(labs):
    from metrics import evaluate
    out = {}
    ks = list(labs)
    for k, l in labs.items():
        c = np.bincount(l.ravel())[1:]; c = c[c > 0] * float(np.prod(VOX))
        out[k] = dict(n=int(len(c)), vol_um3_med=float(np.median(c)), vol_um3_p10=float(np.percentile(c, 10)),
                      vol_um3_p90=float(np.percentile(c, 90)), fg=float((l > 0).mean()))
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            e = evaluate(labs[ks[i]], labs[ks[j]])
            out[f"{ks[i]}|{ks[j]}"] = {k: round(float(e[k]), 4) for k in ("f1@0.5", "f1@0.7", "mean_iou_matched", "n_gt", "n_pred")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--s1", default=f"{_NSP_ROOT}/models/models/orx_t3d_r5s1_s0")
    ap.add_argument("--d2", default="orx_v5_2d_s0")
    ap.add_argument("--old_diam", type=float, default=31.251275489877024)
    ap.add_argument("--skip_pipeline", action="store_true", help="只分割、不跑三层分析与荧光定量（冒烟用）")
    ap.add_argument("--crop", type=int, default=0, help="冒烟用：只取中心 crop×crop 的 xy 区域")
    ap.add_argument("--check_r2a2", nargs="*", default=None, help="在这些测试体积上核对 r2a2_diam 与 orx_r5merged_2d_s0_test 的 d_used")
    a = ap.parse_args()
    import tifffile
    from eval_gt import load_model
    os.makedirs(a.out, exist_ok=True)
    M2 = load_model(a.d2)
    if a.check_r2a2 is not None:
        from eval_cpsam import load_case
        from stddata import load_manifest
        ref = {f"{x['ds']}__{x['id']}": x for x in json.load(open(f"{_NSP_ROOT}/results/orx_r5merged_2d_s0_test/rows.json"))}
        man = {f"{r['ds']}__{r['id']}": r for r in load_manifest(split="test", dim="3d")}
        res = {}
        for k in a.check_r2a2:
            r = man[k]; img, gt, box, ign = load_case(r, 384)
            d, sel, _ = r2a2_diam(M2, img, r["voxel"] or (1.0, 1.0, 1.0))
            res[k] = dict(d=d, sel=list(sel), ref_d=ref[k]["d_used"], ref_sel=[ref[k]["scale"], ref[k]["sel_thr"]])
            print(f"CHECK r2a2 {k}: 本实现 d={d:.4f} 选 {sel}；已有 d={ref[k]['d_used']:.4f} 选 ({ref[k]['scale']}, {ref[k]['sel_thr']})", flush=True)
        json.dump(res, open(f"{a.out}/check_r2a2.json", "w"), indent=1)
        return
    img = tifffile.imread(C1).astype(np.float32)
    if a.crop:
        H, W = img.shape[1:]; y0, x0 = (H - a.crop) // 2, (W - a.crop) // 2
        img = img[:, y0:y0 + a.crop, x0:x0 + a.crop]
    rec = {}
    t0 = time.time()
    d, sel, info = r2a2_diam(M2, img, VOX)
    rec["r2a2"] = dict(d=d, sel=list(sel), info=info, sec=round(time.time() - t0, 1))
    print(f"r2a2 直径 {d:.2f} 像素（选 {sel}），{time.time() - t0:.0f}s", flush=True)
    labs = {}
    runs = [("s1", a.s1, "T", d), ("d2", a.d2, "T", d), ("base", "base", "fixed", d), ("s1_d31", a.s1, "T", a.old_diam)]
    for tag, name, rule, dd in runs:
        M = M2 if name == a.d2 else load_model(name, dict(views=3) if name == a.s1 else None)
        lab, x = segment(M, img, VOX, dd, rule)
        rec[tag] = dict(model=name, rule=rule, **x)
        labs[tag] = lab
        os.makedirs(f"{a.out}/nuclei", exist_ok=True)
        tifffile.imwrite(f"{a.out}/nuclei/{tag}.tif", lab.astype(np.uint16 if lab.max() < 65535 else np.uint32), compression="zlib")
        print(f"  {tag}: 直径 {dd:.2f}，阈值 {x['thr']:g}，{x['n']} 个核，{x['sec']:.0f}s", flush=True)
        if M is not M2:
            del M
    if not a.crop:
        for k, p in OLD.items():
            labs[k] = tifffile.imread(p).astype(np.int32)
    rec["stats"] = stats(labs)
    json.dump(rec, open(f"{a.out}/g7.json", "w"), indent=1)
    if not a.skip_pipeline and not a.crop:                             # 裁块只用于冒烟，三层分析要整幅图
        for tag in ("s1", "d2", "base", "s1_d31"):
            for sfx, mn in (("_f300", 300.0), ("", 0.0)):
                po = f"{a.out}/pipeline/{tag}{sfx}"
                subprocess.run([sys.executable, "organoid_pipeline.py", "--nuc-img", C1, "--cell-img", C2, "--nuc-mask", f"{a.out}/nuclei/{tag}.tif",
                                "--voxel", *map(str, VOX), "--out", po, "--min-nuc-um3", str(mn)], check=True)
                if sfx == "_f300":
                    subprocess.run([sys.executable, "fluor_quant.py", "--seg", po, "--out", f"{a.out}/fluor/{tag}{sfx}"], check=True)
    print("SUMMARY G7 " + json.dumps({k: rec[k]["n"] for k in ("s1", "d2", "base", "s1_d31")} | {"d_r2a2": round(d, 2)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
