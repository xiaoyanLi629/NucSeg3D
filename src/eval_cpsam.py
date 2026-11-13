"""在 3D 测试集上评测 Cellpose-SAM（原版或微调后），输出各数据集的 F1@IoU0.5/0.7 等指标。

测试集来自标准化数据 data/std/manifest.csv 中 split=test 的 3D 体积（通过 stddata.py 读取）；
xy 超过 --box 的大体积，裁出前景最多的 box×box 子块（3D 推理较慢）。
推理：do_3D=True；diameter 取该体积真值细胞核的 xy 中位直径（让模型把核缩放到 30 像素，
与训练切片的尺度一致）；anisotropy = z 体素 / xy 体素。
用法: python eval_cpsam.py --model cpsam_v2 --tag base [--gpu 0] [--max_per_ds 12] [--only ds ...]
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import tifffile

from stddata import load_manifest, load, load_ignore
from metrics import evaluate

ROOT = f"{_NSP_ROOT}"


def eq_diam_xy_px(lab, voxel):
    ids, cnt = np.unique(lab[lab > 0], return_counts=True)
    d_um = np.median((6 * cnt * np.prod(voxel) / np.pi) ** (1 / 3))
    return float(d_um / voxel[1])


def best_box(lab, size):
    """在 xy 上找前景最多的 size×size 窗口，返回 (y, x)；不需要裁剪时返回 None。"""
    H, W = lab.shape[-2:]
    if H <= size and W <= size:
        return None
    integ = np.pad((lab > 0).sum(0).astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sy, sx = min(size, H), min(size, W)
    v = integ[sy:, sx:] - integ[:-sy, sx:] - integ[sy:, :-sx] + integ[:-sy, :-sx]
    y, x = np.unravel_index(np.argmax(v), v.shape)
    return int(y), int(x)


def load_case(r, size):
    img, lab = load(r)
    lab = lab.astype(np.int32)
    ign = load_ignore(r)
    b = best_box(lab, size)
    if b:
        y, x = b
        img, lab = img[:, y:y + size, x:x + size], lab[:, y:y + size, x:x + size]
        ign = ign[:, y:y + size, x:x + size] if ign is not None else None
    return img, lab, b, ign


def drop_ignored(pred, ign):
    """去掉一半以上体素落在未定义区域里的预测（该区域没有标注，不计对错）。"""
    if ign is None or not ign.any():
        return pred
    tot = np.bincount(pred.ravel()); inn = np.bincount(pred[ign].ravel(), minlength=len(tot))
    bad = np.where(inn > 0.5 * np.maximum(tot, 1))[0]; bad = bad[bad > 0]
    pred = pred.copy(); pred[np.isin(pred, bad)] = 0
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cpsam_v2")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--box", type=int, default=384)
    ap.add_argument("--max_per_ds", type=int, default=12)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--flow3D_smooth", type=float, default=1.0)
    ap.add_argument("--resume", action="store_true", help="与 --only 合用：保留其他数据集已有的逐例结果")
    a = ap.parse_args()
    from cellpose import models
    model = models.CellposeModel(gpu=True, pretrained_model=a.model)

    import logging
    logging.basicConfig(level=logging.WARNING)
    by = defaultdict(list)
    for r in load_manifest(split="test", dim="3d", ds=a.only or None):
        by[r["ds"]].append(r)
    out = f"{ROOT}/results/{a.tag}"
    os.makedirs(out, exist_ok=True)
    rows = []
    if a.resume and os.path.exists(f"{out}/per_case.json"):     # 只重跑 --only 指定的数据集，其余沿用已有结果
        rows = [r for r in json.load(open(f"{out}/per_case.json")) if r["ds"] not in by]
    for ds, ms in sorted(by.items()):
        ms = sorted(ms, key=lambda m: m["id"])
        if len(ms) > a.max_per_ds:            # 均匀抽取，覆盖不同子集（如 BlastoSPIM 的 lowSNR/modSNR）
            ms = [ms[int(i)] for i in np.linspace(0, len(ms) - 1, a.max_per_ds)]
        for m in ms:
            img, gt, box, ign = load_case(m, a.box)
            if gt.max() == 0:
                continue
            vox = m["voxel"] or (1.0, 1.0, 1.0)
            d = eq_diam_xy_px(gt, vox)
            t0 = time.time()
            pred, _, _ = model.eval(img, do_3D=True, z_axis=0, diameter=d, anisotropy=vox[0] / vox[1],
                                    flow3D_smooth=a.flow3D_smooth, batch_size=32)
            pred = drop_ignored(pred.astype(np.int32), ign)
            r = {"ds": ds, "id": m["id"], "shape": list(img.shape), "box_yx": box, "box": a.box, "diam_px": d,
                 "seconds": time.time() - t0, **evaluate(gt, pred.astype(np.int32))}
            os.makedirs(f"{out}/pred", exist_ok=True)       # 保存预测（裁剪后的坐标系），供报告画图
            tifffile.imwrite(f"{out}/pred/{ds}__{m['id']}.tif", pred.astype(np.uint16 if pred.max() < 65535 else np.uint32),
                             compression="zlib", compressionargs={"level": 1})
            rows.append(r)
            print(f"  {ds:11s} {m['id'][-40:]:40s} F1@0.5={r['f1@0.5']:.3f} F1@0.7={r['f1@0.7']:.3f} "
                  f"n={r['n_pred']}/{r['n_gt']}  {r['seconds']:.0f}s", flush=True)
            json.dump(rows, open(f"{out}/per_case.json", "w"), indent=1)
    summ = {}
    for ds in sorted({r["ds"] for r in rows}):
        rr = [r for r in rows if r["ds"] == ds]
        summ[ds] = {k: float(np.mean([r[k] for r in rr])) for k in ("f1@0.5", "f1@0.7", "count_err")} | {"n_cases": len(rr)}
    summ["_mean_over_ds"] = {k: float(np.mean([v[k] for kk, v in summ.items()])) for k in ("f1@0.5", "f1@0.7")}
    json.dump(dict(model=a.model, summary=summ, cases=rows), open(f"{out}/summary.json", "w"), indent=1)
    print(f"[{a.tag}] " + "  ".join(f"{k}: {v['f1@0.5']:.3f}/{v['f1@0.7']:.3f}" for k, v in summ.items()))


if __name__ == "__main__":
    main()
