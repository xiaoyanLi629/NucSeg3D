"""G5（DESIGN.md 第 4 节）中 nnU-Net 与 StarDist-3D 的计时，口径对齐 g5_bench.py（同一张卡、单进程、串行，先热身一个体积）。
P：完整流程同一批 45 个测试体积，gt 协议（真值直径）。t_net = 归一化 + 缩放到 iso30 网格 + 网络（nnU-Net 含它自己的预处理、
   滑窗与镜像 TTA；StarDist 含分块预测），对应 Cellpose 的 model.eval(compute_masks=False)；t_mask = 验证集选定变体的实例化
   （nnU-Net：P(内部) > 0.9 种子 + 标记分水岭；StarDist：默认阈值的 NMS）+ 取回原网格，对应 Cellpose 的跟流 t_mask(−2)；
   t_fixed = t_net + t_mask。基线没有规则 T，t_ruleT 不适用。
X：nucverse3d 各向同性体积的中心立方块 64…256，直径固定 30（缩放倍数 1，网络网格 = 输入网格），每个尺寸两遍取较小值。
峰值显存：框架计数（torch max_memory_reserved / TF get_memory_info 的 peak）与 nvidia-smi 每 0.2 s 轮询本进程，取大者。
计时时独占一张卡、不限显存（nnU-Net perform_everything_on_device=True，StarDist 显存按需增长），与评测时的 8 GB 限制不同。
用法：python bl_bench.py --method nnunet|stardist --model <目录> --out <目录> --tag <标签> [--parts P X] [--max_vols n]"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import bl_eval as B  # noqa: E402
from g5_bench import Peak  # noqa: E402

SEL = {"nnunet": 4, "stardist": 2}                                       # 验证集选定的变体（results/eval/selection_val.json）


class Fw:
    def __init__(self, method):
        self.m = method
        if method == "nnunet":
            import torch
            self.t = torch
        else:
            import tensorflow as tf
            self.tf = tf

    def sync(self):
        if self.m == "nnunet":
            self.t.cuda.synchronize()

    def reset(self):
        if self.m == "nnunet":
            self.t.cuda.empty_cache(); self.t.cuda.reset_peak_memory_stats()
        else:
            self.tf.config.experimental.reset_memory_stats("GPU:0")

    def peak(self):
        if self.m == "nnunet":
            return int(self.t.cuda.max_memory_reserved() // 2**20)
        return int(self.tf.config.experimental.get_memory_info("GPU:0")["peak"] // 2**20)


def run_one(P, fw, peak, method, img, d, vox):
    fw.reset(); peak.reset(); fw.sync(); t0 = time.time()
    f3 = B.iso_factors(d, vox)
    x = B.to_iso(img, f3)
    raw = P.net(x)
    if method == "stardist":
        raw = tuple(np.asarray(r) for r in raw)                            # 确保 GPU 计算已结束
    fw.sync(); t_net = time.time() - t0
    t1 = time.time()
    lab = P.post(raw, SEL[method]) if method == "nnunet" else P.post(raw, SEL[method], x.shape)
    ii = [B.back_index(n, m) for n, m in zip(img.shape, x.shape)]
    lab = lab[np.ix_(*ii)]
    t_mask = time.time() - t1
    time.sleep(0.5)
    return dict(t_net=round(t_net, 2), t_mask={"sel": round(t_mask, 2)}, t_fixed=round(t_net + t_mask, 2), peak_fw_mib=fw.peak(),
                peak_smi_mib=peak.mx, in_shape=list(img.shape), net_in_shape=list(x.shape), n_obj=int(len(np.unique(lab)) - 1),
                loadavg=round(os.getloadavg()[0], 1), **({"n_tiles": list(P.last_tiles)} if hasattr(P, "last_tiles") else {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["nnunet", "stardist"]); ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--parts", nargs="+", default=["P", "X"]); ap.add_argument("--box", type=int, default=384)
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 96, 128, 160, 192, 256])
    ap.add_argument("--x_vol", default="nucverse3d__2_livernuclei_20221025_30_Control_12w_Central")
    ap.add_argument("--max_vols", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="PREREV PR2：nnU-Net 用这些折做集成（默认单折 0）")
    a = ap.parse_args()
    from eval_cpsam import load_case, eq_diam_xy_px, load
    from stddata import load_manifest
    if a.method == "nnunet":
        from nnunet_pred import NNUNetPredictor
        P = NNUNetPredictor(a.model, on_device=True, **({"fold": a.folds} if a.folds else {}))
        import torch
        gpu = torch.cuda.get_device_name(0)
    else:
        from stardist_pred import StarDistPredictor
        P = StarDistPredictor(a.model)
        gpu = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader -i " + os.environ.get("CUDA_VISIBLE_DEVICES", "0")).read().strip()
    fw = Fw(a.method); peak = Peak(); host = os.uname().nodename
    os.makedirs(a.out, exist_ok=True)
    p = f"{a.out}/{a.tag}.json"
    res = json.load(open(p)) if os.path.exists(p) else dict(model=a.model, method=a.method, variant=SEL[a.method], host=host, gpu=gpu, P={}, X={})
    cases = B.EG.pick_cases("test", os.path.join(HERE, "val_split.json"), None, None, 4)
    if a.max_vols:
        cases = cases[:a.max_vols]
    print(f"G5 {a.tag}：{host} {gpu}，P {len(cases)} 个体积", flush=True)
    if "P" in a.parts:
        data = []
        for r in cases:
            img, gt, box, ign = load_case(r, a.box)
            if ign is not None:
                gt = gt.copy(); gt[ign] = 0
            vox = r["voxel"] or (1.0, 1.0, 1.0)
            if r["ds"] == "mesospim":                                       # 与评测相同：源数据体素
                vox = (5.0, 10.52, 10.52)
            data.append((r, img, eq_diam_xy_px(gt, vox), vox))
        run_one(P, fw, peak, a.method, data[0][1], data[0][2], data[0][3])  # 热身
        for r, img, d, vox in data:
            k = f"{r['ds']}__{r['id']}"
            x = run_one(P, fw, peak, a.method, img, d, vox); x.update(d=round(d, 2), ratio=vox[0] / vox[1])
            res["P"][k] = x
            print(f"  {a.tag} {k[:48]:48s} net {x['t_net']:7.1f}s  mask {x['t_mask']['sel']:6.1f}s  peak {max(x['peak_fw_mib'], x['peak_smi_mib'])} MiB",
                  flush=True)
            json.dump(res, open(p, "w"), indent=1)
    if "X" in a.parts:
        xr = [r for r in load_manifest(split="test", dim="3d") if f"{r['ds']}__{r['id']}".startswith(a.x_vol)][0]
        img, lab = load(xr)
        Z, Y, X = img.shape
        for s in a.sizes:
            s = min(s, Z, Y, X)
            z0, y0, x0 = (Z - s) // 2, (Y - s) // 2, (X - s) // 2
            cube = np.ascontiguousarray(img[z0:z0 + s, y0:y0 + s, x0:x0 + s])
            best = None
            for rep in range(2):
                x = run_one(P, fw, peak, a.method, cube, 30.0, (1.0, 1.0, 1.0))
                best = x if best is None or x["t_fixed"] < best["t_fixed"] else best
            best.update(side=s, voxels=int(s ** 3))
            res["X"][str(s)] = best
            print(f"  {a.tag} 立方块 {s}³: net {best['t_net']:.1f}s  mask {best['t_mask']['sel']:.1f}s  peak {max(best['peak_fw_mib'], best['peak_smi_mib'])} MiB", flush=True)
            json.dump(res, open(p, "w"), indent=1)
    Pv = res["P"]
    if Pv:
        s = dict(n=len(Pv), t_fixed_median=float(np.median([v["t_fixed"] for v in Pv.values()])),
                 t_fixed_sum_h=round(sum(v["t_fixed"] for v in Pv.values()) / 3600, 3),
                 peak_max_mib=int(max(max(v["peak_fw_mib"], v["peak_smi_mib"]) for v in Pv.values())))
        print("SUMMARY G5 " + a.tag + " " + json.dumps(s, ensure_ascii=False), flush=True)
    peak.on = False


if __name__ == "__main__":
    main()
