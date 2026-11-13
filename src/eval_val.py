"""在三维验证集（val_split.json 里留出的训练体积）或测试集上评测，一次网络推理出一整张“阈值 × 体积下限”网格。

每个样本只跑一次网络（compute_masks=False），再对每个 cellprob 阈值出掩码，对每个体积下限比例去掉小物体，算 F1。
裁块与直径规则同 eval_cpsam.py：xy 超过 384 的取前景最多的 384×384 窗口；直径取真值细胞核 xy 中位直径。
每套数据按 id 排序后均匀取至多 max_per_ds 个样本。
输出：results/<prefix>_<split>/rows.json（逐样本逐设置的 F1）和 grid.json（按数据集宏平均）。
用法：python eval_val.py --model <path> --split val --split_json ../val_split.json --thrs 0 -1 -2 --fracs 0 0.15 0.25 0.35 --prefix orx_x
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import argparse, json, os, time
from collections import defaultdict
import numpy as np
from stddata import load_manifest
from metrics import evaluate
from eval_cpsam import load_case, eq_diam_xy_px, drop_ignored

ROOT = f"{_NSP_ROOT}"


def pick_cases(split, split_json, k):
    if split == "val":
        s = json.load(open(split_json))
        man = {(r["ds"], r["id"]): r for r in load_manifest(split="train", dim="3d")}
        by = {ds: [man[(ds, i)] for i in sorted(ids)] for ds, ids in s.items()}
    else:
        by = defaultdict(list)
        for r in load_manifest(split="test", dim="3d"):
            by[r["ds"]].append(r)
        by = {ds: sorted(v, key=lambda r: r["id"]) for ds, v in by.items()}
    out = []
    for ds, rs in sorted(by.items()):
        if k and len(rs) > k:
            rs = [rs[int(j)] for j in np.linspace(0, len(rs) - 1, k).round()]
        out += rs
    return out


def zinterp(img, Zu):
    Z = img.shape[0]; z = np.linspace(0, Z - 1, Zu); i = np.clip(np.floor(z).astype(int), 0, Z - 2); t = (z - i)[:, None, None]
    return ((1 - t) * img[i] + t * img[i + 1]).astype(np.float32)


def log_diam(img, vox, diams=None):
    """经典尺度空间（Lindeberg）的细胞核直径估计，不依赖网络：对每个候选直径 D 用 σ = D/(2√3) 的高斯拉普拉斯
    （z 方向按体素比例缩小 σ；D 大时先降采样），取尺度归一化响应 −σ²·LoG 的 99.9 百分位，响应最强的 D 即估计直径（xy 像素）。"""
    from scipy import ndimage as ndi
    diams = diams or list(np.geomspace(4, 160, 21))
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, (1, 99.8)); x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    rz = vox[1] / vox[0]
    resp = []
    for D in diams:
        f = max(1.0, D / 16.0); fz = max(1.0, f * rz)
        w = ndi.zoom(x, (1 / fz, 1 / f, 1 / f), order=1) if f > 1 else x
        sxy = D / (2 * np.sqrt(3)) / f; sz = max(D / (2 * np.sqrt(3)) * rz / fz, 0.5)
        sig = (sz, sxy, sxy)  # 各轴分别做尺度归一化：Σ σ_i² ∂²/∂x_i²（各向异性时不能统一乘 σ_xy²）
        r = -sum(sig[i] ** 2 * ndi.gaussian_filter(w, sig, order=[2 if j == i else 0 for j in range(3)]) for i in range(3))
        resp.append(float(np.percentile(r, 99.9)))
    k = int(np.argmax(resp))
    return float(diams[k]), [round(v, 4) for v in resp]


def macro(rows, key):
    by = defaultdict(list)
    for r in rows:
        by[r["ds"]].append(r[key])
    per = {d: float(np.mean(v)) for d, v in sorted(by.items())}
    return float(np.mean(list(per.values()))), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--split_json", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "val_split.json"))
    ap.add_argument("--thrs", type=float, nargs="+", default=[0, -1, -2]); ap.add_argument("--fracs", type=float, nargs="+", default=[0, 0.15, 0.25, 0.35])
    ap.add_argument("--max_per_ds", type=int, default=4); ap.add_argument("--prefix", required=True); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--diam_mode", default="gt", choices=["gt", "none", "twopass", "scan", "scanrefine", "stab", "joint"],
                    help="gt：用真值量出的直径；none：不给直径（按 30 像素）；twopass：先不给直径推理，从预测掩码量出直径再推理一遍；scan：在 --scales 每个直径上推理，选“量出的直径与输入直径最接近”的尺度")
    ap.add_argument("--max_vox", type=float, default=8e8, help="scan 时放大后体素数超过它的尺度跳过（防止内存超限）")
    ap.add_argument("--scales", type=float, nargs="+", default=[6, 10, 17, 30, 52, 90])
    ap.add_argument("--save_raw", action="store_true", help="保存每个样本的 cellprob（float16）和各阈值下未加体积下限的掩码，供离线做后处理（纠错器）实验")
    ap.add_argument("--joint_rule", default="stab", choices=["stab", "r2a2"], help="joint：stab=尺度×阈值联合稳定性最高；r2a2=cp0 下稳定性−自洽选尺度，再按稳定性选阈值")
    ap.add_argument("--min_obj", type=int, default=2, help="stab：分出的物体少于它的尺度不参选")
    ap.add_argument("--iso_final", action="store_true", help="与 --iso 合用：扫尺度选直径仍在原始 z 空间做，只在最终分割时各向同性跟流")
    ap.add_argument("--iso", action="store_true", help="z/xy ≥ 2 的样本先线性插值到各向同性，在各向同性分辨率下预测、跟流、出掩码，再按原层位置取回标签")
    a = ap.parse_args()
    from cellpose import models
    model = models.CellposeModel(gpu=True, pretrained_model=a.model)
    cases = pick_cases(a.split, a.split_json, a.max_per_ds)
    out_dir = f"{ROOT}/results/{a.prefix}_{a.split}"; os.makedirs(out_dir, exist_ok=True)
    print(f"{a.split}: {len(cases)} 个样本，{len({c['ds'] for c in cases})} 套数据", flush=True)
    rows, all_rows = [], []
    for r in cases:
        img, gt, box, ign = load_case(r, 384)
        if ign is not None:
            gt = gt.copy(); gt[ign] = 0
        if gt.max() == 0:
            continue
        vox = r["voxel"] or (1.0, 1.0, 1.0); d_gt = eq_diam_xy_px(gt, vox)
        t0 = time.time()
        ratio = vox[0] / vox[1]
        if a.iso and ratio >= 2:                                         # 各向同性跟流：Cellpose 内部会把流缩回粗 z 再跟流，叠核易被合并
            Zu = int(round(img.shape[0] * ratio)); zi = np.round(np.linspace(0, Zu - 1, img.shape[0])).astype(int)
            im = img.astype(np.float32); lo, hi = np.percentile(im, [1, 99.8]); im = (im - lo) / max(hi - lo, 1e-6)
            img_in, an = zinterp(im, Zu), 1.0
        else:
            img_in, an, zi = img, ratio, None

        def cmask(dP, cp, **kw):
            m = model._compute_masks((*img_in.shape, 1), dP, cp, **kw).squeeze()
            return m if zi is None else m[zi]
        run = lambda diam: model.eval(img_in, do_3D=True, z_axis=0, diameter=diam, anisotropy=an, flow3D_smooth=1.0,
                                      batch_size=a.batch, compute_masks=False)[1]
        # 选尺度用的推理与掩码：--iso_final 时留在原始 z 空间（各向同性空间里扫尺度会选偏），否则与最终分割相同
        if a.iso_final and zi is not None:
            cmask_s = lambda dP, cp, **kw: model._compute_masks((*img.shape, 1), dP, cp, **kw).squeeze()
            run_s = lambda diam: model.eval(img, do_3D=True, z_axis=0, diameter=diam, anisotropy=ratio, flow3D_smooth=1.0,
                                            batch_size=a.batch, compute_masks=False)[1]
        else:
            cmask_s, run_s = cmask, run

        def est_diam(fl, dd):
            m = cmask(fl[1], fl[2], flow_threshold=0.4, cellprob_threshold=0, min_size=15,
                                     max_size_fraction=0.4, niter=int(200 / (30.0 / dd)), do_3D=True).squeeze()
            return eq_diam_xy_px(m, vox) if m.max() > 0 else 30.0

        def score(fl, d_net, d_vol, extra):
            dP, cp = fl[1], fl[2]
            exp_vol = np.pi / 6 * d_vol ** 3 * vox[1] / vox[0]
            out, msg = [], []
            for t in a.thrs:
                pred = cmask(dP, cp, flow_threshold=0.4, cellprob_threshold=t, min_size=15,
                                            max_size_fraction=0.4, niter=int(200 / (30.0 / d_net)), do_3D=True).squeeze().astype(np.int32)
                pred = drop_ignored(pred, ign)
                if a.save_raw:
                    import tifffile
                    os.makedirs(f"{out_dir}/raw", exist_ok=True)
                    tifffile.imwrite(f"{out_dir}/raw/{r['ds']}__{r['id']}__cp{t:g}.tif", pred.astype(np.uint16 if pred.max() < 65535 else np.uint32), compression="zlib")
                    if t == a.thrs[0]:
                        np.save(f"{out_dir}/raw/{r['ds']}__{r['id']}__cellprob.npy", cp.astype(np.float16))
                        json.dump(dict(d_net=d_net, d_vol=d_vol, vox=list(vox), box=box), open(f"{out_dir}/raw/{r['ds']}__{r['id']}__meta.json", "w"))
                cnt = np.bincount(pred.ravel())
                for f in a.fracs:
                    p = pred
                    if f > 0:
                        small = np.where(cnt < f * exp_vol)[0]; small = small[small > 0]
                        p = pred.copy(); p[np.isin(p, small)] = 0
                    ev = evaluate(gt, p)
                    out.append(dict(ds=r["ds"], id=r["id"], thr=t, frac=f, d_gt=d_gt, d_used=d_vol, f1=ev["f1@0.5"], f1_7=ev["f1@0.7"],
                                    prec=ev["prec@0.5"], rec=ev["rec@0.5"], n_gt=ev["n_gt"], n_pred=ev["n_pred"], **extra))
                msg.append(f"cp{t:g}={out[-1]['f1']:.3f}")
            return out, " ".join(msg)

        if a.diam_mode == "joint":
            # 尺度 × 阈值联合稳定性：每个尺度推理一次，在每个 cellprob 阈值下各出一张掩码；
            # 同一阈值下相邻尺度的掩码互评 F1@0.5，(尺度, 阈值) 的稳定性 = 与左右相邻尺度一致度的平均。
            # 另记经典 LoG 尺度估计 d_log。全部写进 rows_all（含每个组合的真实 F1），规则在验证集上离线比较；
            # 本次选 (尺度, 阈值) 中稳定性最高者（物体数 ≥ min_obj），按量出的直径细化并完整评分。
            d_log, log_resp = log_diam(img, vox)
            ms, info = {}, {}
            for sc in sorted(a.scales, reverse=True):
                nvox = img.size * (30.0 / sc) ** 3 * vox[0] / vox[1]
                if nvox > a.max_vox and ms:
                    print(f"    scale {sc:5.1f}: 放大后约 {nvox:.1e} 体素，超过上限，跳过", flush=True)
                    continue
                fl = run_s(sc)
                for t in a.thrs:
                    m = cmask_s(fl[1], fl[2], flow_threshold=0.4, cellprob_threshold=t, min_size=15,
                                             max_size_fraction=0.4, niter=int(200 / (30.0 / sc)), do_3D=True).squeeze().astype(np.int32)
                    n = int(len(np.unique(m)) - 1)
                    d_est = eq_diam_xy_px(m, vox) if n > 0 else float("nan")
                    ms[(sc, t)] = m
                    info[(sc, t)] = dict(n=n, d_est=d_est, cons=abs(np.log(d_est / sc)) if n > 0 else float("inf"),
                                         f1=evaluate(gt, drop_ignored(m, ign))["f1@0.5"])
                del fl
            scs = sorted({k[0] for k in ms})
            for t in a.thrs:
                agree = {(s0, s1): (evaluate(ms[(s0, t)], ms[(s1, t)])["f1@0.5"] if info[(s0, t)]["n"] and info[(s1, t)]["n"] else 0.0)
                         for s0, s1 in zip(scs[:-1], scs[1:])}
                for i, sc in enumerate(scs):
                    nb = ([agree[(scs[i - 1], sc)]] if i > 0 else []) + ([agree[(sc, scs[i + 1])]] if i + 1 < len(scs) else [])
                    q = info[(sc, t)]; q["stab"] = float(np.mean(nb)) if nb else 0.0
                    all_rows.append(dict(ds=r["ds"], id=r["id"], scale=sc, thr=t, d_gt=d_gt, d_log=d_log, d_est=q["d_est"], n_obj=q["n"],
                                         cons=q["cons"], stab=q["stab"], f1=q["f1"]))
            del ms
            for sc in scs:
                print(f"    scale {sc:5.1f}: " + "  ".join(f"cp{t:g} n={info[(sc, t)]['n']:4d} stab={info[(sc, t)]['stab']:.2f} F1={info[(sc, t)]['f1']:.2f}"
                                                   for t in a.thrs), flush=True)
            ok = [k for k in info if info[k]["n"] >= a.min_obj] or list(info)
            if a.joint_rule == "r2a2":
                # 先在 cp0 下按 稳定性 − 自洽偏差 选尺度，再在该尺度的各阈值里取稳定性最高者
                ok0 = [k for k in ok if k[1] == a.thrs[0]] or ok
                s_best = max(ok0, key=lambda k: info[k]["stab"] - info[k]["cons"])[0]
                sel = max([k for k in ok if k[0] == s_best] or [k for k in info if k[0] == s_best], key=lambda k: info[k]["stab"])
            else:
                sel = max(ok, key=lambda k: info[k]["stab"])
            d = info[sel]["d_est"] if np.isfinite(info[sel]["d_est"]) else sel[0]
            if img.size * (30.0 / d) ** 3 * vox[0] / vox[1] > a.max_vox:
                d = sel[0]
            out, msg = score(run(d), d, d, dict(scale=sel[0], sel_thr=sel[1], d_net=d, d_log=d_log))
            msg = f"选中 {sel[0]:g}/cp{sel[1]:g} LoG={d_log:.1f}  " + msg
            rows += out
            json.dump(all_rows, open(f"{out_dir}/rows_all.json", "w"))
        elif a.diam_mode == "stab":
            # 跨尺度稳定性：每个尺度都跑（不提前停），出 cp0 掩码；相邻两个尺度的掩码互相做 F1@0.5（“一致度”），
            # 每个尺度的稳定性 = 与左右相邻尺度一致度的平均。真实尺度附近分出的核在相邻尺度下也能被复现，
            # 而小尺度下把大核切碎的结果碎法随尺度变化，一致度低。所有量写进 rows_all，供在验证集上比较选尺度规则；
            # 本次选中尺度用 stab − cons 最大（物体数 ≥ min_obj），再按量出的直径细化一遍并完整评分。
            ms, info = {}, {}
            for sc in sorted(a.scales, reverse=True):
                nvox = img.size * (30.0 / sc) ** 3 * vox[0] / vox[1]
                if nvox > a.max_vox and ms:
                    print(f"    scale {sc:5.1f}: 放大后约 {nvox:.1e} 体素，超过上限，跳过", flush=True)
                    continue
                fl = run(sc)
                m = cmask(fl[1], fl[2], flow_threshold=0.4, cellprob_threshold=0, min_size=15,
                                         max_size_fraction=0.4, niter=int(200 / (30.0 / sc)), do_3D=True).squeeze().astype(np.int32)
                del fl
                n = int(len(np.unique(m)) - 1)
                d_est = eq_diam_xy_px(m, vox) if n > 0 else float("nan")
                ms[sc] = m
                info[sc] = dict(n=n, d_est=d_est, cons=abs(np.log(d_est / sc)) if n > 0 else float("inf"),
                                f1=evaluate(gt, drop_ignored(m, ign))["f1@0.5"])
            scs = sorted(ms)
            agree = {}
            for s0, s1 in zip(scs[:-1], scs[1:]):
                agree[(s0, s1)] = evaluate(ms[s0], ms[s1])["f1@0.5"] if info[s0]["n"] and info[s1]["n"] else 0.0
            for i, sc in enumerate(scs):
                nb = [agree[(scs[i - 1], sc)]] if i > 0 else []
                nb += [agree[(sc, scs[i + 1])]] if i + 1 < len(scs) else []
                info[sc].update(stab=float(np.mean(nb)) if nb else 0.0, agr_lo=nb[0] if i > 0 else float("nan"),
                                agr_hi=agree[(sc, scs[i + 1])] if i + 1 < len(scs) else float("nan"))
                all_rows.append(dict(ds=r["ds"], id=r["id"], scale=sc, d_gt=d_gt, d_est=info[sc]["d_est"], n_obj=info[sc]["n"],
                                     cons=info[sc]["cons"], stab=info[sc]["stab"], agr_lo=info[sc]["agr_lo"], agr_hi=info[sc]["agr_hi"],
                                     f1_cp0=info[sc]["f1"]))
                print(f"    scale {sc:5.1f}: d_est={info[sc]['d_est']:6.1f} n={info[sc]['n']:5d} cons={info[sc]['cons']:.3f} "
                      f"stab={info[sc]['stab']:.3f}  cp0 F1={info[sc]['f1']:.3f}", flush=True)
            del ms
            ok = [sc for sc in scs if info[sc]["n"] >= a.min_obj] or scs
            sel = max(ok, key=lambda sc: info[sc]["stab"] - (info[sc]["cons"] if np.isfinite(info[sc]["cons"]) else 9))
            d = info[sel]["d_est"] if np.isfinite(info[sel]["d_est"]) else sel
            if img.size * (30.0 / d) ** 3 * vox[0] / vox[1] > a.max_vox:
                d = sel
            out, msg = score(run(d), d, d, dict(scale=sel, d_net=d))
            rows += out
            json.dump(all_rows, open(f"{out_dir}/rows_all.json", "w"))
        elif a.diam_mode in ("scan", "scanrefine"):
            # 每个尺度只出一次 cp0 掩码：估直径、算自洽度，并记下该尺度的 F1（rows_all，供在验证集上另定规则）；
            # 只对选中的尺度按全部阈值与体积下限完整评分
            best = None
            for sc in sorted(a.scales, reverse=True):
                if best is not None and best["cons"] < 0.35 and last_cons > best["cons"] + 0.1:
                    break
                nvox = img.size * (30.0 / sc) ** 3 * vox[0] / vox[1]
                if nvox > a.max_vox and best is not None:
                    print(f"    scale {sc:5.1f}: 放大后约 {nvox:.1e} 体素，超过上限，跳过", flush=True)
                    continue
                fl = run(sc)
                m = cmask(fl[1], fl[2], flow_threshold=0.4, cellprob_threshold=0, min_size=15,
                                         max_size_fraction=0.4, niter=int(200 / (30.0 / sc)), do_3D=True).squeeze().astype(np.int32)
                n = int(len(np.unique(m)) - 1)
                d_est = eq_diam_xy_px(m, vox) if n > 0 else float("nan")
                cons = abs(np.log(d_est / sc)) if n > 0 else float("inf")
                f1 = evaluate(gt, drop_ignored(m, ign))["f1@0.5"]
                all_rows.append(dict(ds=r["ds"], id=r["id"], scale=sc, d_gt=d_gt, d_est=d_est, n_obj=n, cons=cons, f1_cp0=f1))
                print(f"    scale {sc:5.1f}: d_est={d_est:6.1f} n={n:5d} cons={cons:.3f}  cp0 F1={f1:.3f}", flush=True)
                last_cons = cons
                if best is None or cons < best["cons"]:
                    best = dict(cons=cons, sc=sc, fl=fl, d_est=d_est)
                del fl
            d = best["d_est"] if np.isfinite(best["d_est"]) else best["sc"]
            d_net = best["sc"]
            # scanrefine：网格尺度相邻约 1.7 倍，选中尺度后再按量出的直径精确推理一遍（放大后超过 max_vox 就不细化）
            if a.diam_mode == "scanrefine" and np.isfinite(best["d_est"]) and img.size * (30.0 / d) ** 3 * vox[0] / vox[1] <= a.max_vox:
                best["fl"] = run(d); d_net = d
            out, msg = score(best["fl"], d_net, d, dict(scale=best["sc"], d_net=d_net))
            rows += out; best = None
            json.dump(all_rows, open(f"{out_dir}/rows_all.json", "w"))
        else:
            if a.diam_mode == "gt":
                d = d_gt; fl = run(d)
            elif a.diam_mode == "none":
                fl = run(None); d = est_diam(fl, 30.0)
            else:
                d = est_diam(run(None), 30.0); fl = run(d)
            d_net = 30.0 if a.diam_mode == "none" else d
            out, msg = score(fl, d_net, d, {})
            rows += out
        json.dump(rows, open(f"{out_dir}/rows.json", "w"))
        print(f"  {r['ds']:12s} {r['id'][-34:]:34s} d={d:.1f}px (真值 {d_gt:.1f})  (vol {a.fracs[-1]}) " + msg + f"  {time.time()-t0:.0f}s", flush=True)
    grid = {}
    for t in a.thrs:
        for f in a.fracs:
            sub = [x for x in rows if x["thr"] == t and x["frac"] == f]
            m5, per = macro(sub, "f1"); m7, _ = macro(sub, "f1_7")
            grid[f"cp{t:g}_vol{f:g}"] = dict(thr=t, frac=f, macro_f1_05=m5, macro_f1_07=m7, per_ds=per)
    if rows and "sel_thr" in rows[0]:
        # 逐样本自适应阈值：每个样本取规则选出的阈值
        for f in a.fracs:
            sub = [x for x in rows if x["thr"] == x["sel_thr"] and x["frac"] == f]
            m5, per = macro(sub, "f1"); m7, _ = macro(sub, "f1_7")
            grid[f"cpSel_vol{f:g}"] = dict(thr="sel", frac=f, macro_f1_05=m5, macro_f1_07=m7, per_ds=per)
    json.dump(grid, open(f"{out_dir}/grid.json", "w"), indent=1)
    print(f"\nGRID {a.split} macro F1@0.5（行：cellprob 阈值；列：体积下限比例）")
    print("        " + "".join(f"{f:>8g}" for f in a.fracs))
    for t in a.thrs:
        print(f"cp{t:<5g} " + "".join(f"{grid[f'cp{t:g}_vol{f:g}']['macro_f1_05']:8.3f}" for f in a.fracs))
    if "cpSel_vol0" in grid:
        print("cpSel  " + "".join(f"{grid[f'cpSel_vol{f:g}']['macro_f1_05']:8.3f}" for f in a.fracs))
    print("GRID_JSON " + json.dumps({k: round(v["macro_f1_05"], 4) for k, v in grid.items()}), flush=True)


if __name__ == "__main__":
    main()
