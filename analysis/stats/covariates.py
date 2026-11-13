"""S6 元回归的逐体积协变量（只读图像与真值，不读任何模型输出的分数）。
图像与真值用评测时的同一套读入与裁块（$NUCSEG3D_SCRATCH/orx_wt_orx_gt_227_0/src 的 eval_cpsam.load_case，xy 裁块 384，未定义区域置 0），
并与评测存下的 npz 里的 gt 逐体素核对。
协变量：
  zxy      = 清单体素 z / xy（与 rows.json 的 ratio 相同）；
  diam_px  = 真值核等效直径（xy 像素，eval_cpsam.eq_diam_xy_px，即评测用的 d_gt；部分数据集体素未标定，故用像素而非 µm）；
  density  = 核密度，定义为真值前景体素占（非未定义区域）体素的比例；
  snr      = （核内强度中位数 − 核外强度中位数）/ 核外强度标准差（SD，ddof=0；核外不含未定义区域）。
             偏离预先登记：原定分母为核外 MAD，因 5 套数据集部分体积核外 MAD = 0 无法计算，按 AGENTS.md 提交 a099d3a 改为 SD，分子不变。
             旧的 MAD 版本另存为 snr_mad 以备查。
输出 cache/covariates.json：{"<ds>__<id>": {...}}。用法：python covariates.py [--workers 32]"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

SRC = _nsp_os.path.join(_NSP_REPO, 'src')
OUT = f"{_NSP_SCRATCH}/nc/stats/cache/covariates.json"
REF_RUN = f"{_NSP_ROOT}/results/orx_07992b55-0982-4116-b065-b34859c9a80b_thr_eval"  # tGT_S1_s0，只用其中的 gt 做核对


def one(r):
    sys.path.insert(0, SRC)
    from eval_cpsam import eq_diam_xy_px, load_case
    img, gt, box, ign = load_case(r, 384)
    if ign is not None:
        gt = gt.copy(); gt[ign] = 0
    key = f"{r['ds']}__{r['id']}"
    ref = f"{REF_RUN}/masks/{key}.npz"
    same = None
    if os.path.exists(ref):
        g2 = np.load(ref)["gt"]
        same = bool(g2.shape == gt.shape and np.array_equal(g2 > 0, gt > 0))
    vox = r["voxel"] or (1.0, 1.0, 1.0)
    im = np.asarray(img, np.float32)
    valid = np.ones(gt.shape, bool) if ign is None else ~ign
    fg, bg = gt > 0, (gt == 0) & valid
    mi, mo = float(np.median(im[fg])), float(np.median(im[bg]))
    mad = float(np.median(np.abs(im[bg] - mo)))
    sd = float(np.std(im[bg]))
    return key, dict(ds=r["ds"], id=r["id"], zxy=float(vox[0] / vox[1]), diam_px=float(eq_diam_xy_px(gt, vox)),
                     density=float(fg.sum() / max(valid.sum(), 1)), snr=float((mi - mo) / sd) if sd > 0 else None, snr_mad=float((mi - mo) / mad) if mad > 0 else None,
                     n_gt=int(len(np.unique(gt[gt > 0]))), shape=list(gt.shape), gt_matches_npz=same)


def main():
    sys.path.insert(0, SRC)
    from stddata import load_manifest
    rs = sorted((r for r in load_manifest(split="test", dim="3d") if r["ds"] != "bbbc032"), key=lambda r: (r["ds"], r["id"]))
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 32
    from multiprocessing import Pool
    with Pool(w) as p:
        res = dict(p.imap_unordered(one, rs))
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
    bad = [k for k, v in res.items() if v["gt_matches_npz"] is False]
    print(f"{len(res)} 个体积；真值与 npz 不一致 {len(bad)} 个：{bad[:10]}；无 npz 可核对 {sum(v['gt_matches_npz'] is None for v in res.values())} 个")


if __name__ == "__main__":
    main()
