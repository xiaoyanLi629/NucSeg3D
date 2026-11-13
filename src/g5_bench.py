"""nc G5：推理时间、峰值显存、随体积大小的扩展。设计见 $NUCSEG3D_SCRATCH/nc/gpu/DESIGN.md 第 G5 节。
同一张卡、单进程、一次只跑一个模型（排队器预留 30000 MiB，保证本项目不会有别的作业同卡）。
两部分：
  P（逐体积）：完整流程同一批测试体积（--split test --max_per_ds 4，45 个），gt 协议预处理（真值直径、z/xy ≥ 2 先插值到各向同性），
     每个体积记 t_net（归一化、缩放、网络、缩回、平滑，即 model.eval(compute_masks=False)）、t_mask(t)（阈值 0、−1、−2、−3、−4、−6
     各跟流一次；规则 T 只需 −2、−3、−4、−6），以及峰值显存（torch max_memory_reserved 与 nvidia-smi 轮询本进程，取大者）。
     开始前用第一个体积热身一次（不计）。
  X（扩展）：nucverse3d 的各向同性测试体积 2_livernuclei_20221025_30_Control_12w_Central（0.3 µm，280×320×320）取中心立方块，边长 64、96、128、160、192、256，
     直径固定 30（网络网格 = 输入网格），每个尺寸跑两遍取较小值；S1 同时记三视图与单视图。
模型：S1（视图规则；X 部分两种视图都测）、二维（run_3D）、原版（run_3D）。nnU-Net 与 StarDist-3D 的环境只在 hostC，
在那里用同一脚本的 --only_x/--anchor 口径另测（见 DESIGN.md）。
输出 <out>/<模型标签>.json 与 <out>/summary.json。
用法：python g5_bench.py --out <目录> --models s1=<S1 目录> d2=orx_v5_2d_s0 base=base [--parts P X]"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
THRS = (0.0, -1.0, -2.0, -3.0, -4.0, -6.0)
VIEWS1 = ("cellos", "parhyale", "mesospim")


class Peak:
    """后台线程每 0.2 s 用 nvidia-smi 查本进程显存，记最大值（含 CUDA 上下文）。"""
    def __init__(self):
        self.pid = os.getpid(); self.mx = 0; self.on = True
        self.th = threading.Thread(target=self.loop, daemon=True); self.th.start()

    def loop(self):
        while self.on:
            try:
                o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=10).stdout
                for l in o.strip().splitlines():
                    p, m = (x.strip() for x in l.split(","))
                    if int(p) == self.pid:
                        self.mx = max(self.mx, int(m))
            except Exception:
                pass
            time.sleep(0.2)

    def reset(self):
        self.mx = 0


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_one(M, img, d, ratio, nviews, peak, thrs=THRS):
    import torch
    from eval_val import zinterp
    if hasattr(M, "views"):
        M.views = nviews
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); peak.reset()
    sync(); t0 = time.time()
    if ratio >= 2:
        Zu = int(round(img.shape[0] * ratio))
        im = img.astype(np.float32); lo, hi = np.percentile(im, [1, 99.8]); im = (im - lo) / max(hi - lo, 1e-6)
        img_in, an = zinterp(im, Zu), 1.0
    else:
        img_in, an = img, ratio
    out = M.eval(img_in, do_3D=True, z_axis=0, diameter=d, anisotropy=an, flow3D_smooth=1.0, batch_size=32, compute_masks=False)
    fl = out[1] if isinstance(out[1], (list, tuple)) else out
    dP, cp = fl[1], fl[2]
    sync(); t_net = time.time() - t0
    tm = {}
    for t in thrs:
        sync(); t1 = time.time()
        m = M._compute_masks((*img_in.shape, 1), dP, cp, flow_threshold=0.4, cellprob_threshold=t, min_size=15,
                             max_size_fraction=0.4, niter=int(200 / (30.0 / d)), do_3D=True)
        sync(); tm[f"{t:g}"] = round(time.time() - t1, 2)
        del m
    time.sleep(0.5)
    return dict(t_net=round(t_net, 2), t_mask=tm, t_ruleT=round(t_net + sum(tm[k] for k in ("-2", "-3", "-4", "-6")), 2),
                t_fixed=round(t_net + tm["-2"], 2), peak_torch_mib=int(torch.cuda.max_memory_reserved() // 2**20), peak_smi_mib=peak.mx,
                in_shape=list(img.shape), net_in_shape=list(img_in.shape), loadavg=round(os.getloadavg()[0], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--parts", nargs="+", default=["P", "X"]); ap.add_argument("--box", type=int, default=384)
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 96, 128, 160, 192, 256])
    ap.add_argument("--x_vol", default="nucverse3d__2_livernuclei_20221025_30_Control_12w_Central",
                    help="扩展测试用的体积（ds__id 前缀匹配，须为各向同性测试体积）")
    ap.add_argument("--max_vols", type=int, default=0, help="冒烟测试用：P 部分只跑前 n 个体积")
    ap.add_argument("--split_json", default=os.path.join(HERE, "..", "val_split.json"))
    a = ap.parse_args()
    import torch
    from eval_cpsam import load_case, eq_diam_xy_px, load
    from eval_gt import pick_cases, load_model
    from stddata import load_manifest
    os.makedirs(a.out, exist_ok=True)
    peak = Peak()
    host = os.uname().nodename; gpu = torch.cuda.get_device_name(0)
    cases = pick_cases("test", a.split_json, None, None, 4)
    if a.max_vols:
        cases = cases[:a.max_vols]
    xr = [r for r in load_manifest(split="test", dim="3d") if f"{r['ds']}__{r['id']}".startswith(a.x_vol)]
    assert len(xr) >= 1, a.x_vol
    xr = xr[0]
    print(f"G5：{host} {gpu}，P {len(cases)} 个体积，X 体积 {xr['ds']}__{xr['id']}", flush=True)
    summ = {}
    for spec in a.models:
        tag, name = spec.split("=", 1)
        p = f"{a.out}/{tag}.json"
        res = json.load(open(p)) if os.path.exists(p) else dict(model=name, host=host, gpu=gpu, P={}, X={})
        M = load_model(name, dict(views=3) if tag.startswith("s1") else None)
        is3d = type(M).__name__ == "Model3D"
        if "P" in a.parts:
            data = []
            for r in cases:
                img, gt, box, ign = load_case(r, a.box)
                if ign is not None:
                    gt = gt.copy(); gt[ign] = 0
                vox = r["voxel"] or (1.0, 1.0, 1.0)
                data.append((r, img, eq_diam_xy_px(gt, vox), vox[0] / vox[1]))
            r0, img0, d0, ra0 = data[0]
            run_one(M, img0, d0, ra0, 1 if r0["ds"] in VIEWS1 else 3, peak)                       # 热身
            for r, img, d, ra in data:
                k = f"{r['ds']}__{r['id']}"
                nv = (1 if r["ds"] in VIEWS1 else 3) if is3d else None
                x = run_one(M, img, d, ra, nv, peak); x.update(d=round(d, 2), ratio=ra, views=nv)
                res["P"][k] = x
                print(f"  {tag} {k[:48]:48s} net {x['t_net']:7.1f}s  mask(−2) {x['t_mask']['-2']:6.1f}s  ruleT {x['t_ruleT']:7.1f}s  "
                      f"peak {max(x['peak_torch_mib'], x['peak_smi_mib'])} MiB", flush=True)
                json.dump(res, open(p, "w"), indent=1)
        if "X" in a.parts:
            img, lab = load(xr)
            Z, Y, X = img.shape
            for s in a.sizes:
                s = min(s, Z, Y, X)
                z0, y0, x0 = (Z - s) // 2, (Y - s) // 2, (X - s) // 2
                cube = np.ascontiguousarray(img[z0:z0 + s, y0:y0 + s, x0:x0 + s])
                for nv in ((3, 1) if is3d else (None,)):
                    best = None
                    for rep in range(2):
                        x = run_one(M, cube, 30.0, 1.0, nv, peak)
                        best = x if best is None or x["t_fixed"] < best["t_fixed"] else best
                    best.update(side=s, voxels=int(s ** 3), views=nv)
                    res["X"][f"{s}_v{nv}"] = best
                    print(f"  {tag} 立方块 {s}³ 视图 {nv}: net {best['t_net']:.1f}s  mask(−2) {best['t_mask']['-2']:.1f}s  "
                          f"peak {max(best['peak_torch_mib'], best['peak_smi_mib'])} MiB", flush=True)
                    json.dump(res, open(p, "w"), indent=1)
        P = res["P"]
        if P:
            summ[tag] = dict(n=len(P), t_fixed_median=float(np.median([v["t_fixed"] for v in P.values()])),
                             t_ruleT_median=float(np.median([v["t_ruleT"] for v in P.values()])),
                             t_ruleT_sum_h=round(sum(v["t_ruleT"] for v in P.values()) / 3600, 3),
                             peak_max_mib=int(max(max(v["peak_torch_mib"], v["peak_smi_mib"]) for v in P.values())))
        del M; torch.cuda.empty_cache()
    json.dump(dict(host=host, gpu=gpu, models=summ), open(f"{a.out}/summary.json", "w"), indent=1)
    print("SUMMARY G5 " + json.dumps(summ, ensure_ascii=False), flush=True)
    peak.on = False


if __name__ == "__main__":
    main()
