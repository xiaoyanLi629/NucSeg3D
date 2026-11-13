"""第八轮候选 E3 / E3-2D 的精确计算（orx script 模式入口；离线脚本 ens_offline.py 也复用这里的核心函数）。

背景：eval_gt 对 z/xy ≥ 2 的体积先沿 z 插值到各向同性网格，在插值网格上推理和跟流，再取回原层位；而 --save_masks 只存了
原层位上的 dP、cellprob（sel = v[..., zi, :, :]），所以这些体积不能用存档离线重做跟流，必须重跑网络。

做法（每个体积，与 eval_gt.run_volume 逐步对应）：load_case → 忽略区置零 → 真值直径 d → z/xy ≥ 2 时 z 插值（zinterp）；
依次用 --models 的每个模型推理（与原 thr_eval 运行相同：--views、--views1_ds 的视图规则、--batch、flow3D_smooth=1.0），
float32 的 dP、cellprob 按模型顺序累加再除以模型数（逐体素平均）；然后与 eval_gt 完全相同地跟流（cellpose
CellposeModel._compute_masks：flow_threshold 0.4、min_size 15、max_size_fraction 0.4、niter = int(200/(30/d))、do_3D），
取回原层位、drop_ignored、按体积下限比例 FRACS 删小块、taxonomy，写出与 rows.json 同格式的行；cellprob −2 时把预测、
真值与平均后原层位的 dP、cellprob（float16）存到 masks/<ds>__<id>.npz（与 thr_eval 的存法相同，供规则 T 算 m(t)）。
逐位核对（--ref_dirs 与 --models 一一对应，为原 thr_eval 结果目录）：每个模型重跑的输出在原层位上转 float16 后，与该目录
存档的 dP、cp 逐元素比较，记进 verify；--repro 列出的模型序号再用它自己的 float32 输出单独跟流，逐阈值、逐体积下限与该目录
rows.json 的全部计数字段（tp5、tp7、n_gt、n_pred 与错误分类）比较，并比较 cp−2 的预测标签图，记进 repro。
GPU 由 CUDA_VISIBLE_DEVICES 决定（脚本不指定卡号），没有 GPU 时用 CPU；单进程，不用的模型暂存在 CPU 内存，轮到时再搬上 GPU，
每个体积结束清显存缓存。每个体积的结果先写 vols/<key>.json，中断后重跑跳过已完成的体积。
用法（orx：工作目录为节点的 src/，run.sh 以 python ens_exact.py --out <结果目录> <args> 调用）：
  python ens_exact.py --out <目录> --models m0 m1 m2 --ref_dirs d0 d1 d2 --iso_only [--views 3 --views1_ds cellos parhyale mesospim]
                      [--batch 32] [--box 384] [--repro 0]
冒烟测试（CPU、裁小块，只验证流程能跑通；核对照样执行但必然不一致）：--smoke_crop 4,48,48 --max_vols 1 --threads 8
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import eval_gt as EG                                                   # noqa: E402  开发版 eval_gt（同目录或由 ens_offline.py 给出路径）
from eval_cpsam import drop_ignored, eq_diam_xy_px, load_case          # noqa: E402
from eval_val import zinterp                                           # noqa: E402

THRS = (0.0, -1.0, -2.0, -3.0, -4.0, -6.0)                             # 与 thr_eval.py 的 --thrs 相同
INT_FIELDS = ("n_gt", "n_pred", "tp5", "tp7", "hit", "miss", "merge", "split", "boundary",
              "fpc_false", "fpc_merge", "fpc_split", "fpc_boundary", "merge_z", "merge_xy")


# ---------------------------------------------------------------- 与 eval_gt.run_volume 对应的各步
class DevOnly:
    """只给 CellposeModel._compute_masks 提供 device（该方法只用到 self.device）。"""

    def __init__(self, device):
        self.device = device


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prep_case(r, box, crop=None):
    """eval_gt.run_volume 推理之前的部分：读体积、忽略区置零、真值直径、z 插值。crop=(z,y,x) 只用于冒烟测试。"""
    img, gt, _, ign = load_case(r, box or 10**9)
    if crop:                                                           # 冒烟测试：以前景质心为中心裁小块
        c = np.argwhere(gt > 0).mean(0).round().astype(int) if gt.max() > 0 else np.zeros(3, int)
        s = [max(0, min(int(c[k]) - crop[k] // 2, gt.shape[k] - crop[k])) for k in range(3)]
        sl = tuple(slice(s[k], s[k] + crop[k]) for k in range(3))
        img, gt = img[sl], gt[sl]
        ign = ign[sl] if ign is not None else None
    if ign is not None:
        gt = gt.copy(); gt[ign] = 0
    if gt.max() == 0:
        return None
    vox = r["voxel"] or (1.0, 1.0, 1.0); ratio = vox[0] / vox[1]; d = d_gt = eq_diam_xy_px(gt, vox)
    if ratio >= 2:
        Zu = int(round(img.shape[0] * ratio)); zi = np.round(np.linspace(0, Zu - 1, img.shape[0])).astype(int)
        im = img.astype(np.float32); lo, hi = np.percentile(im, [1, 99.8]); im = (im - lo) / max(hi - lo, 1e-6)
        img_in, an = zinterp(im, Zu), 1.0
    else:
        img_in, an, zi = img, ratio, None
    return dict(r=r, key=f"{r['ds']}__{r['id']}", img_in=img_in, z0=img.shape[0], an=an, zi=zi, gt=gt, ign=ign, vox=vox,
                ratio=ratio, d=d, d_gt=d_gt)


def sel_fn(zi):
    return (lambda v: v) if zi is None else (lambda v: v[..., zi, :, :])


def net_out(M, C, batch, views1_ds):
    """一个模型在这个体积上的 (dP, cellprob)（float32，插值网格），与 eval_gt.run_volume 的推理调用相同。"""
    if views1_ds is not None and type(M).__name__ == "Model3D":           # 视图规则：列出的数据集用单视图
        if not hasattr(M, "views0"):
            M.views0 = M.views
        M.views = 1 if C["r"]["ds"] in views1_ds else M.views0
    fuse = getattr(M, "view_fuse", None)
    cond = getattr(M, "cond", None)
    kw = dict(do_3D=True, z_axis=0, diameter=C["d"], anisotropy=C["an"], flow3D_smooth=1.0, batch_size=batch, compute_masks=False)
    if fuse is None and not cond:
        out = M.eval(C["img_in"], **kw)
    else:
        out = M.eval(C["img_in"], src_zoom=(C["img_in"].shape[0] / C["z0"], 1.0, 1.0), **kw)
    fl = out[1] if isinstance(out[1], (list, tuple)) else out
    return np.asarray(fl[1]), np.asarray(fl[2])


def compute_masks(dev, C, dP, cp, t):
    from cellpose import models
    m = models.CellposeModel._compute_masks(dev, (*C["img_in"].shape, 1), dP, cp, flow_threshold=0.4, cellprob_threshold=t, min_size=15,
                                            max_size_fraction=0.4, niter=int(200 / (30.0 / C["d"])), do_3D=True).squeeze()
    m = m if C["zi"] is None else m[C["zi"]]
    return drop_ignored(m.astype(np.int32), C["ign"])


def score_rows(C, dP, cp, dev, thrs=THRS, views=None, r4=False, save=None, keep_pred=False):
    """跟流 + 评分（eval_gt.run_volume 的后半段）。save：npz 路径（cellprob −2 时存预测、真值与原层位 dP、cp）。
    keep_pred：返回 cellprob −2 的预测标签图（用于与存档 pred 比较）。"""
    gt, vox, d = C["gt"], C["vox"], C["d"]
    G = EG.GTInfo(gt, vox)
    exp_vol = np.pi / 6 * d ** 3 * vox[1] / vox[0]
    rows, pred2 = [], None
    for t in thrs:
        m = compute_masks(dev, C, dP, cp, t)
        if t == -2.0:
            if save:
                sel = sel_fn(C["zi"])
                os.makedirs(os.path.dirname(save), exist_ok=True)
                np.savez_compressed(save + ".part.npz", pred=m, gt=gt.astype(np.int32), vox=np.asarray(vox, np.float64), d=np.float64(d),
                                    dP=sel(np.asarray(dP)).astype(np.float16), cp=sel(np.asarray(cp)).astype(np.float16))
                os.replace(save + ".part.npz", save)
            if keep_pred:
                pred2 = m
        cnt = np.bincount(m.ravel())
        for f in EG.FRACS:
            p = m
            if f > 0:
                small = np.where(cnt < f * exp_vol)[0]; small = small[small > 0]
                p = m.copy(); p[np.isin(p, small)] = 0
            tx = EG.taxonomy(G, p)
            den = max(tx["n_gt"] + tx["n_pred"], 1)
            rows.append(dict(ds=C["r"]["ds"], id=C["r"]["id"], thr=t, frac=f, d_gt=C["d_gt"], ratio=C["ratio"], iso=C["zi"] is not None,
                             shape=list(gt.shape), f1=2 * tx["tp5"] / den, f1_7=2 * tx["tp7"] / den, **tx,
                             **(dict(d_used=d, views=views) if r4 else {})))
    return (rows, pred2) if keep_pred else rows


# ---------------------------------------------------------------- 核对
def load_ref_rows(d):
    rows = json.load(open(os.path.join(d, "rows.json")))
    by = {}
    for x in rows:
        by.setdefault(f"{x['ds']}__{x['id']}", {})[(float(x["thr"]), float(x["frac"]))] = x
    return by


def cmp_rows(mine, ref):
    """逐阈值、逐体积下限比较全部计数字段；另比较 d_gt、ratio、iso、shape。返回 dict(equal, n_cells, diffs)。"""
    diffs = []
    mm = {(float(x["thr"]), float(x["frac"])): x for x in mine}
    for k, x in sorted(mm.items()):
        y = ref.get(k)
        if y is None:
            diffs.append(dict(thr=k[0], frac=k[1], field="missing_in_ref"))
            continue
        for f in INT_FIELDS:
            if int(x[f]) != int(y[f]):
                diffs.append(dict(thr=k[0], frac=k[1], field=f, mine=int(x[f]), ref=int(y[f])))
        for f in ("d_gt", "ratio"):
            if abs(float(x[f]) - float(y[f])) > 1e-9:
                diffs.append(dict(thr=k[0], frac=k[1], field=f, mine=x[f], ref=y[f]))
        if bool(x["iso"]) != bool(y["iso"]) or list(x["shape"]) != list(y["shape"]):
            diffs.append(dict(thr=k[0], frac=k[1], field="iso/shape", mine=[x["iso"], x["shape"]], ref=[y["iso"], y["shape"]]))
    missing = [k for k in ref if k not in mm]
    for k in missing:
        diffs.append(dict(thr=k[0], frac=k[1], field="missing_in_mine"))
    return dict(equal=not diffs, n_cells=len(mm), diffs=diffs)


def cmp_arr(a16, b16):
    """两个 float16 数组逐元素比较：形状、不同元素数、最大绝对差。"""
    if a16.shape != b16.shape:
        return dict(equal=False, shape=[list(a16.shape), list(b16.shape)])
    ne = a16 != b16
    n = int(ne.sum())
    mx = float(np.abs(a16.astype(np.float32) - b16.astype(np.float32)).max()) if n else 0.0
    return dict(equal=n == 0, n_diff=n, n=int(a16.size), max_abs=mx)


def partition_diff(A, B):
    """两张标签图在“标签一一对应”意义下不同的体素数：A、B 任一为前景的体素里，(a, b) 不是互为最大重叠的一对
    （或一侧为背景）的体素个数；只差标签编号时为 0。另返回逐体素标签值不同的个数。"""
    A = np.asarray(A).ravel().astype(np.int64); B = np.asarray(B).ravel().astype(np.int64)
    if A.shape != B.shape:
        return dict(shape_mismatch=True)
    sel = (A > 0) | (B > 0)
    a, b = A[sel], B[sel]
    nb = int(b.max()) + 1 if b.size else 1
    u, c = np.unique(a * nb + b, return_counts=True)
    pa, pb = u // nb, u % nb
    fg = (pa > 0) & (pb > 0)
    o = np.lexsort((-c, pa)); best_b = {}
    for i in o:
        if fg[i] and pa[i] not in best_b:
            best_b[pa[i]] = pb[i]
    o = np.lexsort((-c, pb)); best_a = {}
    for i in o:
        if fg[i] and pb[i] not in best_a:
            best_a[pb[i]] = pa[i]
    ok = np.array([bool(fg[i] and best_b.get(pa[i]) == pb[i] and best_a.get(pb[i]) == pa[i]) for i in range(len(u))])
    return dict(partition_diff=int(c[~ok].sum()), label_diff=int((A != B).sum()), n_fg=int(sel.sum()))


def check_case(C, z):
    """存档 npz 与本次读取的真值、体素、直径是否相同。"""
    return dict(gt_equal=bool(z["gt"].shape == C["gt"].shape and np.array_equal(z["gt"], C["gt"].astype(np.int32))),
                vox_equal=bool(np.allclose(z["vox"], np.asarray(C["vox"], np.float64), rtol=0, atol=0)),
                d_absdiff=float(abs(float(z["d"]) - C["d"])))


# ---------------------------------------------------------------- 模型（暂存在 CPU，轮到时搬上 GPU）
def park(M, dev):
    import torch
    M.net.to(dev)
    M.device = dev
    if dev.type == "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_mem():
    import torch
    if not torch.cuda.is_available():
        return None
    return dict(max_alloc_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), max_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2))


def pick_cases(a):
    cases = EG.pick_val(a.split_json)
    if a.vols:
        want = set(json.load(open(a.vols)))
        cases = [r for r in cases if f"{r['ds']}__{r['id']}" in want]
        assert len(cases) == len(want), f"{a.vols} 里有 {len(want) - len(cases)} 个体积不在验证集清单里"
    if a.iso_only:
        cases = [r for r in cases if (r["voxel"] or (1.0, 1.0, 1.0))[0] / (r["voxel"] or (1.0, 1.0, 1.0))[1] >= 2]
    if a.only:
        cases = [r for r in cases if r["ds"] in a.only]
    box = a.box or 10**9
    cases.sort(key=lambda r: -r["z"] * min(r["y"], box) * min(r["x"], box) * max(r["voxel"][0] / r["voxel"][1] if r["voxel"] else 1, 1))
    return cases[:a.max_vols] if a.max_vols else cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="要平均的模型（eval_gt.load_model 能认的名字或路径），顺序与 --ref_dirs 对应")
    ap.add_argument("--ref_dirs", nargs="*", default=None, help="与 --models 一一对应的原 thr_eval 结果目录（逐位核对与复现核对用）")
    ap.add_argument("--repro", type=int, nargs="*", default=[], help="另外单独跟流、与 rows.json 比较的模型序号（如 0）")
    ap.add_argument("--split_json", default=os.path.join(os.path.dirname(os.path.abspath(EG.__file__)), "..", "val_split.json"))
    ap.add_argument("--vols", default=None, help="json 列表，元素为 <ds>__<id>")
    ap.add_argument("--iso_only", action="store_true", help="只算 z/xy ≥ 2（事先 z 插值）的体积")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--max_vols", type=int, default=0)
    ap.add_argument("--box", type=int, default=384)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--views", type=int, default=1, choices=[1, 3])
    ap.add_argument("--views1_ds", nargs="*", default=None)
    ap.add_argument("--thrs", type=float, nargs="+", default=list(THRS))
    ap.add_argument("--smoke_crop", default=None, help="冒烟测试：z,y,x 裁块（不做核对）")
    ap.add_argument("--threads", type=int, default=0, help="CPU 线程数（0 = 默认）")
    ap.add_argument("--gpu_cap_gb", type=float, default=0.0, help="本进程可分配显存上限（GiB，0 = 不设；超出前 PyTorch 会先释放缓存再重试）")
    a = ap.parse_args()
    import logging
    import torch
    logging.disable(logging.WARNING)
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.gpu_cap_gb > 0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_cap_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    crop = tuple(int(v) for v in a.smoke_crop.split(",")) if a.smoke_crop else None
    refs = a.ref_dirs if a.ref_dirs else None                          # 冒烟测试（裁块）时核对照样走一遍，结果必然不一致
    if refs is not None:
        assert len(refs) == len(a.models), "--ref_dirs 须与 --models 一一对应"
    ref_rows = [load_ref_rows(d) for d in refs] if refs else None
    dev = device(); cpu = torch.device("cpu")
    os.makedirs(f"{a.out}/vols", exist_ok=True)
    opts = dict(views=a.views, ens2d=None, ens_alpha=0.5) if a.views != 1 else None   # 与 eval_gt.main 构造 opts 的方式相同
    r4 = a.views1_ds is not None                                        # eval_gt：给了 --views1_ds 时逐体积结果多记 d_used、views
    cases = pick_cases(a)
    cfg = dict(models=a.models, ref_dirs=refs, repro=a.repro, views=a.views, views1_ds=a.views1_ds, batch=a.batch, box=a.box, thrs=a.thrs,
               iso_only=a.iso_only, vols=a.vols, smoke_crop=a.smoke_crop, device=str(dev), gpu_cap_gb=a.gpu_cap_gb,
               cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), split_json=os.path.abspath(a.split_json))
    json.dump(cfg, open(f"{a.out}/config.json", "w"), ensure_ascii=False, indent=1)
    print(f"ens_exact：{len(cases)} 个体积，{len(a.models)} 个模型 {a.models}，设备 {dev}（CUDA_VISIBLE_DEVICES={cfg['cuda_visible']}），"
          f"opts {opts}，views1_ds {a.views1_ds}，复现核对 {a.repro}", flush=True)
    t_load = time.time()
    Ms = []
    for spec in a.models:
        M = EG.load_model(spec, opts)
        park(M, cpu)
        Ms.append(M)
    print(f"  载入 {len(Ms)} 个模型 {time.time() - t_load:.0f}s（暂存在 CPU）", flush=True)
    t0 = time.time(); allrows = []; done = 0
    for i, r in enumerate(cases):
        key = f"{r['ds']}__{r['id']}"
        vp = f"{a.out}/vols/{key}.json"
        if os.path.exists(vp):
            rec = json.load(open(vp)); allrows += rec["rows"]; done += 1
            continue
        tv = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        C = prep_case(r, a.box, crop)
        if C is None:
            json.dump(dict(rows=[], skipped="gt 为空"), open(vp, "w")); continue
        sel = sel_fn(C["zi"])
        acc_dP = acc_cp = None; verify = []; repro = {}; views = None; sec = {}
        for k, M in enumerate(Ms):
            tk = time.time()
            park(M, dev)
            dP, cp = net_out(M, C, a.batch, a.views1_ds)
            park(M, cpu)
            views = getattr(M, "views", None) if r4 else None
            sec[f"net{k}"] = round(time.time() - tk, 1)
            assert dP.dtype == np.float32 and cp.dtype == np.float32, (dP.dtype, cp.dtype)
            if refs:
                zp = f"{refs[k]}/masks/{key}.npz"
                if os.path.exists(zp):
                    z = np.load(zp)
                    verify.append(dict(model=k, **check_case(C, z), dP=cmp_arr(sel(dP).astype(np.float16), z["dP"]),
                                       cp=cmp_arr(sel(cp).astype(np.float16), z["cp"])))
                else:
                    z = None
                    verify.append(dict(model=k, missing=zp))
            if k in a.repro:
                tr = time.time()
                rows_k, pred_k = score_rows(C, dP, cp, DevOnly(dev), a.thrs, views, r4, keep_pred=True)
                ent = dict(rows=rows_k, sec=round(time.time() - tr, 1))
                if refs:
                    ent["cmp"] = cmp_rows(rows_k, ref_rows[k].get(key, {}))
                    if z is not None:
                        ent["pred_equal"] = bool(z["pred"].shape == pred_k.shape and np.array_equal(z["pred"], pred_k))
                        ent["pred_cmp"] = partition_diff(z["pred"], pred_k)
                repro[str(k)] = ent
                del pred_k
            if acc_dP is None:
                acc_dP, acc_cp = dP.copy(), cp.copy()
            else:
                acc_dP += dP; acc_cp += cp
            del dP, cp
            gc.collect()
        acc_dP /= np.float32(len(Ms)); acc_cp /= np.float32(len(Ms))
        te = time.time()
        rows = score_rows(C, acc_dP, acc_cp, DevOnly(dev), a.thrs, views, r4, save=f"{a.out}/masks/{key}.npz")
        sec["masks_ens"] = round(time.time() - te, 1); sec["total"] = round(time.time() - tv, 1)
        for x in rows:
            x["sec_net"] = round(sum(v for k_, v in sec.items() if k_.startswith("net")), 1); x["sec_total"] = sec["total"]
        rec = dict(rows=rows, verify=verify, repro=repro, sec=sec, mem=gpu_mem())
        json.dump(rec, open(vp + ".part", "w")); os.replace(vp + ".part", vp)
        allrows += rows; done += 1
        del acc_dP, acc_cp, C
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        m2 = {(x["thr"], x["frac"]): x for x in rows}[(-2.0, 0.15)]
        vtxt = " ".join(f"m{v['model']}:" + ("缺存档" if "missing" in v else
                                             f"dP{'=' if v['dP'].get('equal') else '≠'}{v['dP'].get('n_diff', '?')} cp{'=' if v['cp'].get('equal') else '≠'}{v['cp'].get('n_diff', '?')}")
                        for v in verify)
        rtxt = " ".join(f"r{k_}:{'一致' if e.get('cmp', {}).get('equal') else ('无参照' if 'cmp' not in e else str(len(e['cmp']['diffs'])) + '处不同')}"
                        for k_, e in repro.items())
        print(f"  {i + 1:3d}/{len(cases)} {key[-48:]:48s} E3 cp-2/v.15 F1={m2['f1']:.3f}/{m2['f1_7']:.3f} n={m2['n_gt']} | {vtxt} | {rtxt} | "
              f"{sec['total']:.0f}s mem {rec['mem']} [{(time.time() - t0) / 60:.1f} min]", flush=True)
    json.dump(allrows, open(f"{a.out}/rows.json", "w"))
    # 汇总核对结果
    V, R = [], []
    for r in cases:
        vp = f"{a.out}/vols/{r['ds']}__{r['id']}.json"
        if os.path.exists(vp):
            rec = json.load(open(vp))
            V += [dict(key=f"{r['ds']}__{r['id']}", **v) for v in rec.get("verify", [])]
            R += [dict(key=f"{r['ds']}__{r['id']}", model=int(k), **{kk: vv for kk, vv in e.items() if kk != "rows"})
                  for k, e in rec.get("repro", {}).items()]
    json.dump(dict(verify=V, repro=R), open(f"{a.out}/verify.json", "w"), ensure_ascii=False, indent=1)
    nv = len(V); nv_ok = sum(1 for v in V if "missing" not in v and v["dP"].get("equal") and v["cp"].get("equal") and v["gt_equal"] and v["vox_equal"]
                             and v["d_absdiff"] < 1e-9)
    nr = len(R); nr_ok = sum(1 for e in R if e.get("cmp", {}).get("equal")); np_ok = sum(1 for e in R if e.get("pred_equal"))
    S = EG.summarize(allrows) if allrows else {}
    S.update(ens_exact=cfg, verify_ok=f"{nv_ok}/{nv}", repro_rows_ok=f"{nr_ok}/{nr}", repro_pred_ok=f"{np_ok}/{nr}",
             minutes=round((time.time() - t0) / 60, 1))
    json.dump(S, open(f"{a.out}/summary.json", "w"), indent=1, ensure_ascii=False)
    print(f"SUMMARY ens_exact {done}/{len(cases)} 个体积；逐位核对（重跑输出原层位 float16 = 存档，且真值/体素/直径相同）{nv_ok}/{nv}；"
          f"单模型复现 rows 一致 {nr_ok}/{nr}、cp−2 预测图一致 {np_ok}/{nr}；{(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
