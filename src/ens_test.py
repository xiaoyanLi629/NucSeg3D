"""第八轮最终系统 E3（及对照 E3-2D）在测试集上的完整流程（orx script 模式入口；按预定计划只跑一次、只报告）。
与 ens_exact.py 的关系：预处理、推理、逐体素平均、跟流与评分直接调用 ens_exact.py 里已在验证集上逐位核对过的函数
（prep_case、net_out、score_rows、park 等，须与本文件放在同一 src/ 目录），本文件只增加完整流程需要的三处：
  1. 体积选择与 eval_gt --split test --max_per_ds 4 完全相同（eval_gt.pick_cases，去掉 bbbc032，共 45 个），再套 --vox_override；
  2. 直径：每个体积取 --diam_dirs（三个二维种子完整流程 orx_r5merged_2d_s*_test）rows.json 里 r2a2 d_used 的算术平均，
     与 eval_gt --diam_from 一样代替真值直径（网格缩放、跟流步数 niter、体积下限都按它算），行里照常记 d_gt 与 d_used、views；
  3. 复现核对（--repro_ref/--repro_diam/--repro_vols）：在主循环之前，用模型 0 和它自己种子的 d_used 对列出的几个小体积单独
     推理、跟流，与该种子原测试运行（thr_eval 结果目录）比较网络输出（原层位 float16 逐元素）、全部计数字段与 cp−2 预测图；
     只记录是否相同与不同的格子名，不记录、不打印任何计数或分数。
全部体积都重跑网络：三个模型依次推理（视图规则 --views1_ds），float32 输出按模型顺序累加再除以模型数，跟流阈值 0 −1 −2 −3 −4 −6，
cp−2 时存 masks/<ds>__<id>.npz（pred、gt、vox、d、原层位 dP、cp 的 float16），rows.json 与 thr_eval 同格式（test_r6.py 的规则 T 可直接用）。
日志与 summary.json 只写进度、耗时、显存与核对结果，不写分数。GPU 由 CUDA_VISIBLE_DEVICES 决定；单进程，不用的模型暂存在 CPU 内存。
用法：python ens_test.py --out <目录> --models m0 m1 m2 --diam_dirs d0 d1 d2 --vox_override mesospim=5/10.52/10.52
        [--views 3 --views1_ds cellos parhyale mesospim] [--batch 32] [--box 384] [--gpu_cap_gb 9.0]
        [--repro_ref <模型 0 原测试运行目录> --repro_diam <模型 0 自己的直径目录> --repro_vols <ds>__<id> ...]
检查：--dry_run 只列出体积、直径与复现体积后退出（不载入模型）；冒烟：--smoke_crop z,y,x --max_vols 1 --threads 8（CPU）。"""
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

import ens_exact as E  # noqa: E402  同目录的 ens_exact.py（验证集上已逐位核对）

EG = E.EG


def load_dmap(d):
    """与 eval_gt --diam_from 相同：rows.json 里每个体积的 d_used（同一体积各行必须相同）。"""
    m = {}
    for x in json.load(open(os.path.join(d, "rows.json"))):
        k = f"{x['ds']}__{x['id']}"
        assert abs(m.setdefault(k, x["d_used"]) - x["d_used"]) < 1e-9, k
    return m


def prep_case_fp(r, box, d_used, crop=None):
    """eval_gt.run_volume 在 --diam_from 时的预处理：与 gt 协议相同地读取、置零忽略区、z 插值，d_gt 照算，直径换成 d_used。"""
    C = E.prep_case(r, box, crop)
    if C is not None:
        C["d"] = float(d_used)
    return C


def cmp_names_only(mine, ref):
    """只比较计数是否相同：返回 (是否全部相同, 不同的格子与字段名列表)，不带任何数值。"""
    res = E.cmp_rows(mine, ref)
    return res["equal"], sorted({f"t{d['thr']:g}/v{d['frac']:g}:{d['field']}" for d in res["diffs"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--diam_dirs", nargs="+", required=True, help="三个二维种子完整流程的 r5merged 目录；E3 直径 = 各自 d_used 的算术平均")
    ap.add_argument("--vox_override", nargs="*", default=None, help="ds=vz/vy/vx，同 eval_gt")
    ap.add_argument("--split_json", default=os.path.join(os.path.dirname(os.path.abspath(EG.__file__)), "..", "val_split.json"))
    ap.add_argument("--max_per_ds", type=int, default=4)
    ap.add_argument("--box", type=int, default=384)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--views", type=int, default=1, choices=[1, 3])
    ap.add_argument("--views1_ds", nargs="*", default=None)
    ap.add_argument("--thrs", type=float, nargs="+", default=list(E.THRS))
    ap.add_argument("--repro_ref", default=None, help="模型 0 自己种子的原测试运行目录（orx_<运行>_thr_eval）")
    ap.add_argument("--repro_diam", default=None, help="模型 0 自己种子的直径目录（该原运行的 --diam_from）")
    ap.add_argument("--repro_vols", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--vols_file", default=None, help="分片用：json 列表（<ds>__<id>），只处理其中的体积（须都在测试选择里）")
    ap.add_argument("--max_vols", type=int, default=0)
    ap.add_argument("--smoke_crop", default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--gpu_cap_gb", type=float, default=0.0)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    import logging
    import torch
    logging.disable(logging.WARNING)
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.gpu_cap_gb > 0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_cap_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    crop = tuple(int(v) for v in a.smoke_crop.split(",")) if a.smoke_crop else None
    # ---- 体积选择：与 eval_gt --split test --max_per_ds 4 相同（先取样，再按 --only 过滤），再覆盖体素
    cases = EG.pick_cases("test", a.split_json, a.only, None, a.max_per_ds)
    if a.vox_override:
        ov = {k: tuple(float(x) for x in v.split("/")) for k, v in (t.split("=") for t in a.vox_override)}
        assert all(len(v) == 3 for v in ov.values()), ov
        for r in cases:
            if r["ds"] in ov:
                r["voxel"] = ov[r["ds"]]
    keys = [f"{r['ds']}__{r['id']}" for r in cases]
    # ---- 直径：三个二维种子 d_used 的算术平均
    dms = [load_dmap(d) for d in a.diam_dirs]
    miss = [k for k in keys if any(k not in m for m in dms)]
    assert not miss, f"直径目录缺这些体积：{miss[:5]}（共 {len(miss)} 个）"
    dE = {k: float(sum(m[k] for m in dms) / len(dms)) for k in keys}
    d_own = load_dmap(a.repro_diam) if a.repro_diam else None
    bad = [k for k in a.repro_vols if k not in keys]
    assert not bad, f"复现体积不在测试选择里：{bad}"
    box = a.box or 10**9
    cases.sort(key=lambda r: -r["z"] * min(r["y"], box) * min(r["x"], box) * max(r["voxel"][0] / r["voxel"][1] if r["voxel"] else 1, 1))
    byk = {f"{r['ds']}__{r['id']}": r for r in cases}                    # 复现核对按全体选择查体积（不受分片影响）
    if a.vols_file:                                                    # 分片：只处理列出的体积（选择、直径、体素覆盖都已按全体算好）
        want = set(json.load(open(a.vols_file)))
        assert want <= set(keys), f"{a.vols_file} 里有不在测试选择里的体积：{sorted(want - set(keys))[:3]}"
        cases = [r for r in cases if f"{r['ds']}__{r['id']}" in want]
    if a.max_vols:
        cases = cases[:a.max_vols]
    cfg = dict(models=a.models, diam_dirs=a.diam_dirs, vox_override=a.vox_override, max_per_ds=a.max_per_ds, views=a.views, views1_ds=a.views1_ds,
               batch=a.batch, box=a.box, thrs=a.thrs, repro_ref=a.repro_ref, repro_diam=a.repro_diam, repro_vols=a.repro_vols,
               smoke_crop=a.smoke_crop, gpu_cap_gb=a.gpu_cap_gb, n_cases=len(keys), vols_file=a.vols_file, split_json=os.path.abspath(a.split_json),
               device=str(E.device()), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
    print(f"ens_test：测试集 {len(keys)} 个体积、{len({k.split('__')[0] for k in keys})} 套（本次处理 {len(cases)} 个），模型 {a.models}，"
          f"直径 = {len(dms)} 个目录 d_used 的平均，体素覆盖 {a.vox_override}，views {a.views}，views1_ds {a.views1_ds}，复现体积 {a.repro_vols}，"
          f"设备 {cfg['device']}（CUDA_VISIBLE_DEVICES={cfg['cuda_visible']}）", flush=True)
    if a.dry_run:
        for k in keys:
            print(f"  {k[-60:]:60s} d_E3={dE[k]:.3f}" + (f"  d_own={d_own[k]:.3f}" if d_own and k in a.repro_vols else ""))
        return
    os.makedirs(f"{a.out}/vols", exist_ok=True)
    json.dump(cfg, open(f"{a.out}/config.json", "w"), ensure_ascii=False, indent=1)
    json.dump(dict(d_E3=dE, d_dirs=[{k: m[k] for k in keys} for m in dms]), open(f"{a.out}/diameters.json", "w"), indent=0)
    dev = E.device(); cpu = torch.device("cpu")
    opts = dict(views=a.views, ens2d=None, ens_alpha=0.5) if a.views != 1 else None   # 与 eval_gt.main 相同
    t_load = time.time(); Ms = []
    for spec in a.models:
        M = EG.load_model(spec, opts); E.park(M, cpu); Ms.append(M)
    print(f"  载入 {len(Ms)} 个模型 {time.time() - t_load:.0f}s（暂存在 CPU）", flush=True)
    # ---- 复现核对：模型 0、它自己种子的直径
    rep = []
    if a.repro_vols and a.repro_ref:
        ref_rows = E.load_ref_rows(a.repro_ref)
        for k in a.repro_vols:
            if k not in byk:
                continue
            tv = time.time()
            C0 = prep_case_fp(byk[k], a.box, d_own[k], crop)
            M = Ms[0]; E.park(M, dev); dP, cp = E.net_out(M, C0, a.batch, a.views1_ds); E.park(M, cpu)
            views = getattr(M, "views", None)
            z = np.load(f"{a.repro_ref}/masks/{k}.npz")
            sel = E.sel_fn(C0["zi"])
            chk = E.check_case(C0, z)
            vd, vc = E.cmp_arr(sel(dP).astype(np.float16), z["dP"]), E.cmp_arr(sel(cp).astype(np.float16), z["cp"])
            rows0, pred0 = E.score_rows(C0, dP, cp, E.DevOnly(dev), a.thrs, views, True, keep_pred=True)
            eq, names = cmp_names_only(rows0, ref_rows.get(k, {}))
            pc = E.partition_diff(z["pred"], pred0)
            ent = dict(key=k, gt_equal=chk["gt_equal"], vox_equal=chk["vox_equal"], d_equal=chk["d_absdiff"] < 1e-9,
                       out_equal=bool(vd.get("equal") and vc.get("equal")), dP_ndiff=vd.get("n_diff"), cp_ndiff=vc.get("n_diff"),
                       counts_equal=eq, diff_cells=names, pred_equal=bool(z["pred"].shape == pred0.shape and np.array_equal(z["pred"], pred0)),
                       pred_partition_diff=pc.get("partition_diff"), sec=round(time.time() - tv, 1))
            rep.append(ent)
            print(f"  REPRO {k[-48:]:48s} 输入一致 {chk['gt_equal'] and chk['vox_equal'] and ent['d_equal']} | 网络输出逐位相同 {ent['out_equal']}"
                  f"（dP 不同元素 {vd.get('n_diff', '形状不同')}、cp {vc.get('n_diff', '形状不同')}） | 计数全部相同 {eq}（不同格子 {len(names)} 个） | "
                  f"cp−2 预测图相同 {ent['pred_equal']} | {ent['sec']:.0f}s", flush=True)
            del dP, cp, pred0; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        json.dump(rep, open(f"{a.out}/repro.json", "w"), ensure_ascii=False, indent=1)
    # ---- 主循环：三个模型逐体素平均后跟流
    t0 = time.time(); allrows = []; done = 0
    for i, r in enumerate(cases):
        key = f"{r['ds']}__{r['id']}"
        vp = f"{a.out}/vols/{key}.json"
        if os.path.exists(vp):
            allrows += json.load(open(vp))["rows"]; done += 1
            continue
        tv = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        C = prep_case_fp(r, a.box, dE[key], crop)
        if C is None:
            json.dump(dict(rows=[], skipped="gt 为空"), open(vp, "w")); continue
        acc_dP = acc_cp = None; sec = {}; views = None
        for k, M in enumerate(Ms):
            tk = time.time()
            E.park(M, dev); dP, cp = E.net_out(M, C, a.batch, a.views1_ds); E.park(M, cpu)
            views = getattr(M, "views", None)
            sec[f"net{k}"] = round(time.time() - tk, 1)
            assert dP.dtype == np.float32 and cp.dtype == np.float32, (dP.dtype, cp.dtype)
            if acc_dP is None:
                acc_dP, acc_cp = dP.copy(), cp.copy()
            else:
                acc_dP += dP; acc_cp += cp
            del dP, cp; gc.collect()
        acc_dP /= np.float32(len(Ms)); acc_cp /= np.float32(len(Ms))
        te = time.time()
        rows = E.score_rows(C, acc_dP, acc_cp, E.DevOnly(dev), a.thrs, views, True, save=f"{a.out}/masks/{key}.npz")
        sec["masks"] = round(time.time() - te, 1); sec["total"] = round(time.time() - tv, 1)
        for x in rows:
            x["sec_net"] = round(sum(v for k_, v in sec.items() if k_.startswith("net")), 1); x["sec_total"] = sec["total"]
        rec = dict(rows=rows, sec=sec, mem=E.gpu_mem())
        json.dump(rec, open(vp + ".part", "w")); os.replace(vp + ".part", vp)
        allrows += rows; done += 1
        del acc_dP, acc_cp, C; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  {i + 1:3d}/{len(cases)} {key[-48:]:48s} d={dE[key]:6.2f} views={views} | {sec['total']:.0f}s（网络 {rows[0]['sec_net']:.0f}s）"
              f" mem {rec['mem']} [{(time.time() - t0) / 60:.1f} min]", flush=True)
    json.dump(allrows, open(f"{a.out}/rows.json", "w"))
    S = dict(ens_test=cfg, n_done=done, n_cases=len(cases), repro=[{k: v for k, v in e.items() if k != "diff_cells"} for e in rep],
             minutes=round((time.time() - t0) / 60, 1))
    json.dump(S, open(f"{a.out}/summary.json", "w"), ensure_ascii=False, indent=1)
    nr = len(rep)
    print(f"SUMMARY ens_test {done}/{len(cases)} 个体积；复现核对（模型 0、自身直径）{nr} 个体积：输入一致 "
          f"{sum(e['gt_equal'] and e['vox_equal'] and e['d_equal'] for e in rep)}/{nr}，网络输出逐位相同 {sum(e['out_equal'] for e in rep)}/{nr}，"
          f"计数全部相同 {sum(e['counts_equal'] for e in rep)}/{nr}，cp−2 预测图相同 {sum(e['pred_equal'] for e in rep)}/{nr}；"
          f"{(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
