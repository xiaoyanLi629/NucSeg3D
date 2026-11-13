"""“真值直径”评测协议（gt-diam，只用于开发阶段；共享约定 true3d_spec.md 第 2、3 节）。

样本：val_split.json 里的全部验证体积（88 个，去掉 bbbc032），裁块规则同 eval_cpsam.load_case（xy 超过 --box 时取前景最多的窗口）。
推理：直径取该体积真值细胞核 xy 等效球直径的中位数，anisotropy = vz/vxy；z/xy ≥ 2 的体积先沿 z 线性插值到各向同性，
     在各向同性网格上推理和跟流，再按原层位置取回标签（与 eval_val --iso、diag3d/infer_save.py 相同）。每个体积只跑一次网络。
网格：cellprob 阈值 {0, −1, −2} × 体积下限比例 {0, 0.15, 0.25}（删掉体积 < 比例 × π/6·d³·vxy/vz 的预测）。
指标：每套数据把所有体积的 TP/FP/FN 合在一起算 F1@0.5、F1@0.7（pooled-object），再对数据集求平均（宏平均）；
     另报只算验证体积 ≥ 3 个的数据集的宏平均，以及逐体积 F1 先平均的旧口径。
错误分类（与 diag3d/diag_errs.py 相同的规则）：每个真值核归为 hit/miss/merge/split/boundary，每个未匹配的预测归为
     false/merge/split/boundary；1−F1@0.5 按 (FN+FP)/(n_gt+n_pred) 分解成 miss/merge/split/boundary/false 五项；
     合并的方向：被同一个预测覆盖 >50% 的真值核两两之间，质心连线以 z 为主（|Δz| > xy 距离，单位 µm）记为 z 向合并。
模型：load_model(名字或目录)
     - 目录且含 model3d.json：src/infer3d.py 的 Model3D(目录)（真三维模型，eval/_compute_masks 接口与 CellposeModel 相同）；
     - 否则按 Cellpose 权重处理：绝对路径直接用；"base" 为原版 cpsam_v2；其他名字在共享盘 models/models/ 下找。
并行：同一张卡上起 --workers 个进程（每个进程各载一份模型），体积从大到小分发；每个体积的结果先写 vols/<ds>__<id>.json，
     中断后重跑会跳过已完成的体积。
输出：<out>/rows.json（逐体积 × 设置）与 <out>/summary.json；最后打印一行以 SUMMARY 开头的结果。
用法: python eval_gt.py --model orx_v5_2d_s0 --out $NUCSEG3D_ROOT/results/orx_x_gtval [--workers 3] [--only ds ...] [--ids id ...]
round 2（只对真三维模型，默认关闭）：--views 3（三视图平均）、--ens2d <二维权重名> --ens_alpha 0.5（与二维 run_3D 输出加权平均）。
round 3（默认关闭）：--view_fuse aniso1|aniso2|comp1|comp2（配合 --views 3，各向异性感知的视图融合，见 infer3d.py 文档）；
     开启时把事先 z 插值的倍数作为 src_zoom 传给模型，并在逐体积结果里记 fuse（各轴放大倍数与视图权重）。
round 4（默认关闭）：完整流程的最后一步。--diam_from <eval_val 结果目录>：每个体积改用该次运行 rows.json 里 r2a2 选出的
     直径 d_used（代替真值直径，网格、跟流步数、体积下限都按它算）；--split test：清单里的三维测试体积（去掉 bbbc032）；
     --max_per_ds 4：每套按编号均匀取至多 4 个（与 eval_val.pick_cases 的取法相同，先取样再按 --only/--ids 过滤）；
     --views1_ds ds ...：配合 --views 3，列出的数据集改用单视图（视图规则）。开启任一项时逐体积结果多记 d_used 与 views。
round 5（默认关闭，mesoSPIM 诊断）：--view_perm 102|201|012（真三维模型单视图换切片方向，见 infer3d.py）；
     --thrs t ...（代替 cellprob 阈值网格 {0, −1, −2}）；--save_masks：把 cellprob −2 的预测（未删小块）、真值和
     网络输出（dP、cellprob，float16）存到 <out>/masks/<ds>__<id>.npz，供 shape_diag.py 做逐核形状分析；
     --vox_override ds=vz/vy/vx（覆盖清单体素，如 mesospim 的源数据体素 5/10.52/10.52）。
round 6（默认关闭）：--tta_flips yx|zyx（真三维模型的翻转测试时增强，见 infer3d.py；可与 --views 3 叠加）；
     分辨率条件模型（model3d.json 的 cfg 里 cond=res）自动启用：把事先 z 插值的倍数作为 src_zoom 传给模型，由清单体素
     （含 --vox_override）与本次所用直径（真值或 --diam_from 的 d_used）得到的各轴 f 记进逐体积结果的 res 字段；
     --cond_null（诊断用）让条件模型改用“未知”嵌入。
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = f"{_NSP_ROOT}"
THRS = (0.0, -1.0, -2.0)
FRACS = (0.0, 0.15, 0.25)
MAIN = "cp-2_vol0.15"
ERR_CLASSES = ("hit", "miss", "merge", "split", "boundary")


def resolve_model(spec):
    """返回 ("3d", 目录) 或 ("2d", 权重路径)。"""
    cands = [spec] if os.path.isabs(spec) else [spec, f"{ROOT}/models/models/{spec}"]
    for p in cands:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "model3d.json")):
            return "3d", os.path.abspath(p)
    if spec == "base":
        return "2d", f"{ROOT}/models/base/cpsam_v2"
    for p in cands:
        if os.path.isfile(p):
            return "2d", os.path.abspath(p)
    raise FileNotFoundError(f"找不到模型 {spec}（既不是含 model3d.json 的目录，也不是 Cellpose 权重文件）")


def load_model(spec, opts=None):
    """opts（round 2，可选）：dict(views=1|3, ens2d=名或 None, ens_alpha=0.5)，只用于真三维模型。"""
    opts = {k: v for k, v in (opts or {}).items() if v is not None}
    kind, path = resolve_model(spec)
    if kind == "3d":
        sys.path.insert(0, HERE)
        from infer3d import Model3D                                      # 真三维模型（由 3D 节点提供），按需导入
        return Model3D(path, **opts)
    if (opts.get("views", 1) != 1 or opts.get("ens2d") or opts.get("view_fuse") or opts.get("view_perm") or opts.get("tta_flips")
            or opts.get("cond_null")):
        raise ValueError("views/ens2d/view_fuse/view_perm/tta_flips/cond_null 只用于真三维模型（含 model3d.json 的目录）")
    from cellpose import models
    return models.CellposeModel(gpu=True, pretrained_model=path)


def pick_val(split_json, only=None, ids=None):
    from stddata import load_manifest
    s = json.load(open(split_json))
    man = {(r["ds"], r["id"]): r for r in load_manifest(split="train", dim="3d")}
    out = []
    for ds, v in sorted(s.items()):
        if ds == "bbbc032" or (only and ds not in only):
            continue
        out += [man[(ds, i)] for i in sorted(v) if not ids or i in ids]
    return out


def pick_cases(split, split_json, only=None, ids=None, k=0):
    """round 4：split=val 同 pick_val；split=test 取清单里全部三维测试体积（去掉 bbbc032）。k>0 时每套按编号均匀取至多 k 个。"""
    from stddata import load_manifest
    if split == "val":
        rs = pick_val(split_json)
    else:
        rs = sorted((r for r in load_manifest(split="test", dim="3d") if r["ds"] != "bbbc032"), key=lambda r: (r["ds"], r["id"]))
    by = defaultdict(list)
    for r in rs:
        by[r["ds"]].append(r)
    out = []
    for ds, v in sorted(by.items()):
        if k and len(v) > k:
            v = [v[int(j)] for j in np.linspace(0, len(v) - 1, k).round()]
        out += [r for r in v if (not only or ds in only) and (not ids or r["id"] in ids)]
    return out


# ---------------------------------------------------------------- 错误分类（diag_errs.py 的规则）
class GTInfo:
    """一个体积的真值：编号、体素数、质心（µm）。对 9 个设置复用。"""

    def __init__(self, gt, vox):
        from scipy import ndimage as ndi
        self.gt = gt
        self.ids = np.unique(gt); self.ids = self.ids[self.ids > 0]
        self.n = len(self.ids)
        self.size = np.bincount(gt.ravel())[self.ids] if self.n else np.zeros(0)
        self.idx = np.zeros(int(gt.max()) + 1, np.int64); self.idx[self.ids] = np.arange(self.n)
        self.cen = (np.array(ndi.center_of_mass(np.ones_like(gt, np.uint8), gt, self.ids)).reshape(-1, 3) * np.asarray(vox)) if self.n else np.zeros((0, 3))


def is_z(dv):
    return abs(dv[0]) > np.hypot(dv[1], dv[2])


def taxonomy(G, pred):
    from scipy.optimize import linear_sum_assignment
    gt = G.gt; ng = G.n
    pids = np.unique(pred); pids = pids[pids > 0]; npr = len(pids)
    res = dict(n_gt=ng, n_pred=npr, tp5=0, tp7=0, **{k: 0 for k in ERR_CLASSES},
               fpc_false=0, fpc_merge=0, fpc_split=0, fpc_boundary=0, merge_z=0, merge_xy=0)
    if ng == 0:
        res["fpc_false"] = npr
        return res
    if npr == 0:
        res["miss"] = ng
        return res
    psz = np.bincount(pred.ravel())[pids]
    pi = np.zeros(int(pred.max()) + 1, np.int64); pi[pids] = np.arange(npr)
    inter = np.zeros((ng, npr))
    m = (gt > 0) & (pred > 0)
    if m.any():
        key = G.idx[gt[m]] * npr + pi[pred[m]]
        u, c = np.unique(key, return_counts=True)
        inter[u // npr, u % npr] = c
    iou = inter / np.maximum(G.size[:, None] + psz[None, :] - inter, 1)
    cov = inter / np.maximum(G.size[:, None], 1)
    rr, cc = linear_sum_assignment(-iou)
    mt = iou[rr, cc]
    res["tp5"] = int((mt >= 0.5).sum()); res["tp7"] = int((mt >= 0.7).sum())
    tp_p = {j for i, j in zip(rr, cc) if iou[i, j] >= 0.5}
    best = iou.max(1); bcov = cov.max(1)
    cls = []
    for i in range(ng):
        if best[i] >= 0.5:
            c_ = "hit"
        elif best[i] < 0.1 and bcov[i] < 0.15:
            c_ = "miss"
        else:
            j = int(np.argmax(cov[i])); others = np.delete(cov[:, j], i)
            if others.size and others.max() > 0.5 and cov[i, j] > 0.3:
                c_ = "merge"
            elif (cov[i] >= 0.15).sum() >= 2:
                c_ = "split"
            else:
                c_ = "boundary"
        cls.append(c_); res[c_] += 1
    for j in range(npr):
        g = np.where(cov[:, j] > 0.5)[0]
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                res["merge_z" if is_z(G.cen[g[b]] - G.cen[g[a]]) else "merge_xy"] += 1
        if j in tp_p:
            continue
        o = inter[:, j]
        if o.sum() < 0.1 * psz[j]:
            res["fpc_false"] += 1
        elif len(g) >= 2:
            res["fpc_merge"] += 1
        else:
            cb = cls[int(np.argmax(o))]
            res["fpc_split" if cb in ("split", "hit") else ("fpc_merge" if cb == "merge" else "fpc_boundary")] += 1
    return res


# ---------------------------------------------------------------- 逐体积推理（子进程）
_M = None


def _init(spec, opts=None):
    global _M
    import logging
    logging.disable(logging.WARNING)
    _M = load_model(spec, opts)


def run_volume(job):
    r, a = job
    from eval_cpsam import load_case, eq_diam_xy_px, drop_ignored
    from eval_val import zinterp
    key = f"{r['ds']}__{r['id']}"
    vp = f"{a['out']}/vols/{key}.json"
    if os.path.exists(vp):
        return json.load(open(vp))
    t0 = time.time()
    img, gt, box, ign = load_case(r, a["box"] or 10**9)
    if ign is not None:
        gt = gt.copy(); gt[ign] = 0
    if gt.max() == 0:
        return []
    vox = r["voxel"] or (1.0, 1.0, 1.0); ratio = vox[0] / vox[1]; d = d_gt = eq_diam_xy_px(gt, vox)
    r4 = bool(a.get("dmap")) or a.get("views1_ds") is not None
    if a.get("dmap"):                                                    # round 4：r2a2 选出的直径代替真值直径
        d = a["dmap"][key]
    if a.get("views1_ds") is not None and type(_M).__name__ == "Model3D":   # round 4：视图规则，列出的数据集用单视图
        if not hasattr(_M, "views0"):
            _M.views0 = _M.views
        _M.views = 1 if r["ds"] in a["views1_ds"] else _M.views0
    if ratio >= 2:
        Zu = int(round(img.shape[0] * ratio)); zi = np.round(np.linspace(0, Zu - 1, img.shape[0])).astype(int)
        im = img.astype(np.float32); lo, hi = np.percentile(im, [1, 99.8]); im = (im - lo) / max(hi - lo, 1e-6)
        img_in, an = zinterp(im, Zu), 1.0
    else:
        img_in, an, zi = img, ratio, None
    fuse = getattr(_M, "view_fuse", None)
    cond = getattr(_M, "cond", None)                                     # round 6：分辨率条件模型也要知道事先的 z 插值
    if fuse is None and not cond:
        out = _M.eval(img_in, do_3D=True, z_axis=0, diameter=d, anisotropy=an, flow3D_smooth=1.0, batch_size=a["batch"], compute_masks=False)
    else:
        sz = (img_in.shape[0] / img.shape[0], 1.0, 1.0)
        out = _M.eval(img_in, do_3D=True, z_axis=0, diameter=d, anisotropy=an, flow3D_smooth=1.0, batch_size=a["batch"], compute_masks=False,
                      src_zoom=sz)
    # CellposeModel.eval 返回 (masks, [hsv, dP, cellprob], styles)；Model3D.eval 也可以直接返回 [None, dP, cellprob]
    fl = out[1] if isinstance(out[1], (list, tuple)) else out
    dP, cp = fl[1], fl[2]
    t_net = time.time() - t0
    G = GTInfo(gt, vox)
    exp_vol = np.pi / 6 * d ** 3 * vox[1] / vox[0]
    rows = []
    for t in (a.get("thrs") or THRS):
        m = _M._compute_masks((*img_in.shape, 1), dP, cp, flow_threshold=0.4, cellprob_threshold=t, min_size=15,
                              max_size_fraction=0.4, niter=int(200 / (30.0 / d)), do_3D=True).squeeze()
        m = m if zi is None else m[zi]
        m = drop_ignored(m.astype(np.int32), ign)
        if a.get("save_masks") and t == -2.0:                            # round 5：存预测、真值与网络输出（只在诊断时开）
            os.makedirs(f"{a['out']}/masks", exist_ok=True)
            sel = (lambda v: v) if zi is None else (lambda v: v[..., zi, :, :])
            np.savez_compressed(f"{a['out']}/masks/{key}.npz", pred=m, gt=gt.astype(np.int32), vox=np.asarray(vox, np.float64),
                                d=np.float64(d), dP=sel(np.asarray(dP)).astype(np.float16), cp=sel(np.asarray(cp)).astype(np.float16))
        cnt = np.bincount(m.ravel())
        for f in FRACS:
            p = m
            if f > 0:
                small = np.where(cnt < f * exp_vol)[0]; small = small[small > 0]
                p = m.copy(); p[np.isin(p, small)] = 0
            tx = taxonomy(G, p)
            den = max(tx["n_gt"] + tx["n_pred"], 1)
            rows.append(dict(ds=r["ds"], id=r["id"], thr=t, frac=f, d_gt=d_gt, ratio=ratio, iso=zi is not None, shape=list(gt.shape),
                             f1=2 * tx["tp5"] / den, f1_7=2 * tx["tp7"] / den, **tx,
                             **(dict(d_used=d, views=getattr(_M, "views", None)) if r4 else {})))
    os.makedirs(os.path.dirname(vp), exist_ok=True)
    for x in rows:
        x["sec_net"] = round(t_net, 1); x["sec_total"] = round(time.time() - t0, 1)
        if fuse is not None:
            x["fuse"] = _M.last_fuse
        if cond:
            x["res"] = _M.last_res                                       # round 6：体积 (z,y,x) 各轴的 f
    json.dump(rows, open(vp + ".part", "w")); os.replace(vp + ".part", vp)
    return rows


# ---------------------------------------------------------------- 汇总
def summarize(rows):
    nvol = defaultdict(set)
    for x in rows:
        nvol[x["ds"]].add(x["id"])
    big = {d for d, s in nvol.items() if len(s) >= 3}
    cells = {}
    for t in sorted({x["thr"] for x in rows}, reverse=True):         # 默认就是 THRS（0、−1、−2）；--thrs 时按实际阈值
        for f in FRACS:
            k = f"cp{t:g}_vol{f:g}"
            sub = [x for x in rows if x["thr"] == t and x["frac"] == f]
            agg = defaultdict(lambda: defaultdict(float)); vm = defaultdict(list)
            for x in sub:
                for c in ("tp5", "tp7", "n_gt", "n_pred", *ERR_CLASSES, "fpc_false", "fpc_merge", "fpc_split", "fpc_boundary", "merge_z", "merge_xy"):
                    agg[x["ds"]][c] += x[c]
                vm[x["ds"]].append(x["f1"])
            per = {}
            for ds, g in sorted(agg.items()):
                den = max(g["n_gt"] + g["n_pred"], 1)
                per[ds] = dict(f1_05=2 * g["tp5"] / den, f1_07=2 * g["tp7"] / den, nvol=len(nvol[ds]), n_gt=int(g["n_gt"]), n_pred=int(g["n_pred"]),
                               f1_05_volmean=float(np.mean(vm[ds])),
                               err=dict(miss=g["miss"] / den, merge=(g["merge"] + g["fpc_merge"]) / den, split=(g["split"] + g["fpc_split"]) / den,
                                        boundary=(g["boundary"] + g["fpc_boundary"]) / den, false=g["fpc_false"] / den),
                               zmerge_share=g["merge_z"] / max(g["merge_z"] + g["merge_xy"], 1), merge_pairs=int(g["merge_z"] + g["merge_xy"]))
            mac = lambda key, dss: float(np.mean([per[d][key] for d in dss])) if dss else float("nan")
            mz = sum(agg[d]["merge_z"] for d in agg); mx = sum(agg[d]["merge_xy"] for d in agg)
            cells[k] = dict(thr=t, frac=f, macro_f1_05=mac("f1_05", list(per)), macro_f1_07=mac("f1_07", list(per)),
                            macro3_f1_05=mac("f1_05", [d for d in per if d in big]), macro3_f1_07=mac("f1_07", [d for d in per if d in big]),
                            macro_volmean_f1_05=mac("f1_05_volmean", list(per)),
                            err_macro={e: float(np.mean([per[d]["err"][e] for d in per])) for e in ("miss", "merge", "split", "boundary", "false")},
                            zmerge_share_pooled=mz / max(mz + mx, 1), per_ds=per)
    best = max(cells, key=lambda k: cells[k]["macro_f1_05"])
    return dict(n_volumes=sum(len(s) for s in nvol.values()), n_datasets=len(nvol), datasets_ge3=sorted(big), main=MAIN, best=best, cells=cells)


def summary_line(model, S, tag="gtval"):
    c, b = S["cells"][MAIN], S["cells"][S["best"]]
    e = c["err_macro"]
    return (f"SUMMARY model={model} {tag} n={S['n_volumes']}/{S['n_datasets']}ds {MAIN}: F1@0.5={c['macro_f1_05']:.4f} F1@0.7={c['macro_f1_07']:.4f} "
            f"(>=3vol {c['macro3_f1_05']:.4f}/{c['macro3_f1_07']:.4f}) | best {S['best']}: F1@0.5={b['macro_f1_05']:.4f} F1@0.7={b['macro_f1_07']:.4f} | "
            f"1-F1 miss={e['miss']:.3f} merge={e['merge']:.3f} split={e['split']:.3f} bnd={e['boundary']:.3f} fp={e['false']:.3f} zmerge={c['zmerge_share_pooled']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Cellpose 权重名/路径，或含 model3d.json 的目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split_json", default=os.path.join(HERE, "..", "val_split.json"))
    ap.add_argument("--workers", type=int, default=3, help="同一张卡上并行的进程数")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--box", type=int, default=384, help="xy 裁块边长（0 表示整幅）")
    ap.add_argument("--only", nargs="*"); ap.add_argument("--ids", nargs="*")
    ap.add_argument("--views", type=int, default=1, choices=[1, 3], help="真三维模型：1 = 单视图（旧行为），3 = 三视图平均")
    ap.add_argument("--ens2d", default=None, help="真三维模型：与此二维模型（models/models 下的名字或路径）的输出平均")
    ap.add_argument("--ens_alpha", type=float, default=0.5, help="集成时三维模型的权重")
    ap.add_argument("--view_fuse", default=None, choices=["aniso1", "aniso2", "comp1", "comp2"],
                    help="真三维模型、--views 3：各向异性感知的视图融合（默认不开 = 等权平均）")
    ap.add_argument("--split", default="val", choices=["val", "test"], help="round 4：test = 清单里的三维测试体积（去掉 bbbc032）")
    ap.add_argument("--max_per_ds", type=int, default=0, help="round 4：每套按编号均匀取至多这么多个体积（0 = 全部；完整流程用 4，同 eval_val）")
    ap.add_argument("--diam_from", default=None, help="round 4：eval_val 结果目录（含 rows.json），每个体积用其中的 d_used 代替真值直径")
    ap.add_argument("--views1_ds", nargs="*", default=None, help="round 4：配合 --views 3，这些数据集用单视图（视图规则：cellos parhyale mesospim）")
    ap.add_argument("--view_perm", default=None, choices=["012", "102", "201"], help="round 5：真三维模型单视图换切片方向")
    ap.add_argument("--thrs", type=float, nargs="+", default=None, help="round 5：cellprob 阈值网格（须含 −2，主格不变）")
    ap.add_argument("--save_masks", action="store_true", help="round 5：保存 cellprob −2 的预测、真值与网络输出")
    ap.add_argument("--vox_override", nargs="*", default=None, help="round 5：ds=vz/vy/vx 覆盖清单体素（µm，诊断用）")
    ap.add_argument("--tta_flips", default=None, choices=["yx", "zyx"],
                    help="round 6：真三维模型的翻转测试时增强（模型 y、x 或 z、y、x 翻转的全部组合取平均，流分量相应变号）；默认关闭")
    ap.add_argument("--cond_null", action="store_true", help="round 6：分辨率条件模型改用“未知”嵌入（诊断用）")
    a = ap.parse_args()
    opts = dict(views=a.views, ens2d=a.ens2d, ens_alpha=a.ens_alpha) if (a.views != 1 or a.ens2d) else None
    if a.view_fuse:
        opts = dict(opts or {}, view_fuse=a.view_fuse)
    if a.view_perm:
        opts = dict(opts or {}, view_perm=tuple(int(c) for c in a.view_perm))
    if a.tta_flips:
        opts = dict(opts or {}, tta_flips=a.tta_flips)
    if a.cond_null:
        opts = dict(opts or {}, cond_null=True)
    assert a.thrs is None or -2.0 in a.thrs, "--thrs 须含 −2（主格 cp-2_vol0.15）"
    if opts and resolve_model(a.model)[0] != "3d" and (opts.get("views", 1) != 1 or opts.get("ens2d") or opts.get("view_fuse") or opts.get("view_perm")
                                                        or opts.get("tta_flips") or opts.get("cond_null")):
        raise SystemExit(f"{opts} 只用于真三维模型（含 model3d.json 的目录），{a.model} 不是")
    os.makedirs(f"{a.out}/vols", exist_ok=True)
    fp = bool(a.diam_from or a.split != "val" or a.max_per_ds)
    cases = pick_cases(a.split, a.split_json, a.only, a.ids, a.max_per_ds) if fp else pick_val(a.split_json, a.only, a.ids)
    if a.vox_override:                                                 # round 5：覆盖清单体素（只在诊断时用）
        ov = {k: tuple(float(x) for x in v.split("/")) for k, v in (t.split("=") for t in a.vox_override)}
        assert all(len(v) == 3 for v in ov.values()), ov
        for r in cases:
            if r["ds"] in ov:
                r["voxel"] = ov[r["ds"]]
        print(f"round 5：体素覆盖 {ov}", flush=True)
    box = a.box or 10**9
    cases.sort(key=lambda r: -r["z"] * min(r["y"], box) * min(r["x"], box) * max(r["voxel"][0] / r["voxel"][1] if r["voxel"] else 1, 1))
    print(f"gt-diam 验证：{len(cases)} 个体积，{len({r['ds'] for r in cases})} 套数据，模型 {a.model}（{resolve_model(a.model)[0]}），"
          f"{a.workers} 个进程" + (f"，选项 {opts}" if opts else ""), flush=True)
    t0 = time.time(); rows = []
    cfg = dict(out=a.out, box=a.box, batch=a.batch)
    if a.thrs:
        cfg["thrs"] = tuple(float(t) for t in a.thrs)
    if a.save_masks:
        cfg["save_masks"] = True
    if a.diam_from:
        dmap = {}
        for x in json.load(open(os.path.join(a.diam_from, "rows.json"))):
            k = f"{x['ds']}__{x['id']}"
            assert abs(dmap.setdefault(k, x["d_used"]) - x["d_used"]) < 1e-9, k
        miss = [f"{r['ds']}__{r['id']}" for r in cases if f"{r['ds']}__{r['id']}" not in dmap]
        assert not miss, f"{a.diam_from} 里没有这些体积的直径：{miss[:5]}（共 {len(miss)} 个）"
        cfg["dmap"] = dmap
    if a.views1_ds is not None:
        cfg["views1_ds"] = list(a.views1_ds)
    tag = (f"{'fp' if a.diam_from else 'gt'}{a.split}" + (f"_k{a.max_per_ds}" if a.max_per_ds else "")) if fp else "gtval"
    if fp or a.views1_ds is not None:
        print(f"round 4 协议：{tag}，直径来自 {a.diam_from or '真值'}，单视图数据集 {a.views1_ds}", flush=True)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.model, opts)) as p:
        for k, rs in enumerate(p.imap_unordered(run_volume, [(r, cfg) for r in cases], chunksize=1)):
            rows += rs
            if rs:
                x = {(y["thr"], y["frac"]): y for y in rs}
                m, z = x[(-2.0, 0.15)], x[(0.0, 0.0)]
                print(f"  {k + 1:3d}/{len(cases)} {m['ds']:12s} {m['id'][-36:]:36s} d={m.get('d_used', m['d_gt']):5.1f} z/xy={m['ratio']:.2f} n={m['n_gt']:4d} "
                      f"cp-2/v.15 F1={m['f1']:.3f}/{m['f1_7']:.3f} (cp0/v0 {z['f1']:.3f})  {m['sec_total']:.0f}s  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    json.dump(rows, open(f"{a.out}/rows.json", "w"))
    S = summarize(rows)
    S.update(model=a.model, minutes=round((time.time() - t0) / 60, 1))
    if opts:
        S.update(eval_opts=opts)
    if fp or a.views1_ds is not None:
        S.update(protocol=dict(tag=tag, split=a.split, max_per_ds=a.max_per_ds, diam_from=a.diam_from, views1_ds=a.views1_ds))
    json.dump(S, open(f"{a.out}/summary.json", "w"), indent=1)
    c = S["cells"]
    print("\nGRID pooled-object 宏平均 F1@0.5 / F1@0.7（行：cellprob 阈值；列：体积下限比例）")
    for t in sorted({x["thr"] for x in rows}, reverse=True):
        print(f"  cp{t:<4g} " + "  ".join(f"{c[f'cp{t:g}_vol{f:g}']['macro_f1_05']:.4f}/{c[f'cp{t:g}_vol{f:g}']['macro_f1_07']:.4f}" for f in FRACS))
    print(f"PER_DATASET（{MAIN}）：F1@0.5 F1@0.7 体积数 真值核数 | 1−F1 分解 miss merge split boundary false | z 向合并占比")
    for ds, v in c[MAIN]["per_ds"].items():
        e = v["err"]
        print(f"  {ds:12s} {v['f1_05']:.3f} {v['f1_07']:.3f} {v['nvol']:3d} {v['n_gt']:6d} | " + " ".join(f"{e[k]:.3f}" for k in ("miss", "merge", "split", "boundary", "false"))
              + f" | {v['zmerge_share']:.2f} ({v['merge_pairs']})")
    print(summary_line(a.model, S, tag), flush=True)


if __name__ == "__main__":
    main()
