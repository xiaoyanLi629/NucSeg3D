"""nc G3 第二步（u-Segment3D 环境，CPU）：把一个体积三个方向的二维输出做共识重建。设计见 $NUCSEG3D_SCRATCH/nc/gpu/DESIGN_G3.md 第 2 节。
输入 --in <目录>：probs.npy (3,Z,Y,X，logit)，flows.npy (3,2,Z,Y,X，每个方向两个面内分量，顺序 xy=(dy,dx)、xz=(dz,dx)、yz=(dz,dy)，
与 segment3D.usegment3d.Cellpose2D_model_auto 的输出相同)。网格是网络网格（各向同性，核径约 30 体素）。
调用官方 aggregate_2D_to_3D_segmentation_direct_method 与 postprocess_3D_cell_segmentation，除下面三个超参数外全用官方默认：
  thr：auto = 官方默认多级 Otsu（3 级取第 1 级，偏移 −0.05）；fix = 固定概率 0.339（sigmoid(−2/3)，即三方向 cp 之和 −2 的同一处）
  (decay, sigma)：(0, 1) 官方默认、(0.1, 1)、(0, 2)。
变体 v0–v5 = thr × (decay, sigma)，只做官方的大小过滤、不做流一致性过滤；v(k+6) = 同一设置再加官方默认的流一致性过滤（阈值 0.85，每个变体约贵 5 倍）。
v6 是完整的官方默认。验证集先跑 v0–v6，再把 v0–v5 里最好的一个加流一致性过滤跑一次，一共至多 8 组。
输出 <in>/v<k>.npy（标签，网络网格）与 stats.json。
用法：env_useg/bin/python useg_cons.py --in <目录> --variants 0 1 ..."""
import argparse, json, os, time
import numpy as np

BASE = [(t, dc, s) for t in ("auto", "fix") for dc, s in ((0.0, 1.0), (0.1, 1.0), (0.0, 2.0))]
GRID = [(*b, False) for b in BASE] + [(*b, True) for b in BASE]


def variant_params(k, P):
    import copy
    t, dc, s, _ = GRID[k]
    a = copy.deepcopy(P)
    a["combine_cell_probs"]["cellpose_prob_mask"] = True
    a["combine_cell_probs"]["prob_thresh"] = None if t == "auto" else 0.339
    a["gradient_descent"]["gradient_decay"] = dc
    a["connected_component"]["smooth_sigma"] = s
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--variants", type=int, nargs="+", default=list(range(7)))
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import segment3D.parameters as UP
    import segment3D.usegment3d as US
    probs = np.load(f"{a.inp}/probs.npy", mmap_mode="r"); flows = np.load(f"{a.inp}/flows.npy", mmap_mode="r")
    P = UP.get_2D_to_3D_aggregation_params(); Q = UP.get_postprocess_segmentation_params()
    Q["flow_consistency"]["n_cpu"] = 8
    pl = [np.asarray(probs[i]) for i in range(3)]; gl = [np.asarray(flows[i]) for i in range(3)]
    grad = None; st = {}
    for k in a.variants:
        t0 = time.time(); p = variant_params(k, P); Q["flow_consistency"]["do_flow_remove"] = GRID[k][3]
        err = None
        try:
            seg, (prob3, grad3) = US.aggregate_2D_to_3D_segmentation_direct_method(probs=pl, gradients=gl, params=p, precombined_gradients=grad)
            if grad is None:
                grad = grad3
            t1 = time.time()
            seg = seg[0] if isinstance(seg, (list, tuple)) else seg
            n0 = int(seg.max())
            seg2 = seg
            if n0 > 0:
                out = US.postprocess_3D_cell_segmentation(seg, aggregation_params=p, postprocess_params=Q, cell_gradients=grad3, savefolder=None, basename=None)
                seg2 = out[0] if isinstance(out, (list, tuple)) else out
        except Exception as e:                                                     # 个别体积失败时记空预测并写明，不让整个作业中断
            err = repr(e)[:300]; t1 = time.time(); n0 = 0; seg2 = np.zeros(pl[0].shape, np.int32)
        seg2 = np.asarray(seg2); np.save(f"{a.inp}/v{k}.npy", seg2.astype(np.uint16 if seg2.max() < 65535 else np.int32))   # 省一半临时空间，读回时转 int32
        st[k] = dict(n_raw=n0, n=int(len(np.unique(seg2)) - 1), sec_agg=round(t1 - t0, 1), sec_post=round(time.time() - t1, 1), err=err)
        print("CONS", k, GRID[k], st[k], flush=True)
    json.dump(st, open(f"{a.inp}/stats.json", "w"))


if __name__ == "__main__":
    main()
