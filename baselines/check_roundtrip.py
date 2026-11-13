"""检查 bl_eval 的缩放往返：把真值按最近邻缩到 iso 网格，再按 back_index 取回，F1 应接近 1。"""
import sys, numpy as np
from scipy import ndimage as ndi
import bl_eval as B
from eval_cpsam import load_case, eq_diam_xy_px
cases = B.EG.pick_cases("val", B.os.path.join(B.HERE, "val_split.json"))
cases = cases[::max(1, len(cases)//8)][:8]
for r in cases:
    img, gt, box, ign = load_case(r, 384)
    if ign is not None: gt = gt.copy(); gt[ign] = 0
    vox = r["voxel"] or (1,1,1); d = eq_diam_xy_px(gt, vox); f3 = B.iso_factors(d, vox)
    x = B.to_iso(img, f3)
    li = ndi.zoom(gt, f3, order=0)
    assert li.shape == x.shape, (li.shape, x.shape)
    ii = [B.back_index(n, m) for n, m in zip(img.shape, x.shape)]
    m = li[np.ix_(*ii)]
    tx = B.EG.taxonomy(B.EG.GTInfo(gt, vox), m); den = tx["n_gt"] + tx["n_pred"]
    print(r["ds"], r["id"], gt.shape, "->", x.shape, "F1@0.5 %.3f F1@0.7 %.3f" % (2*tx["tp5"]/den, 2*tx["tp7"]/den))
