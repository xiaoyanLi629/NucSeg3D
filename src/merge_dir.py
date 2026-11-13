"""合并错误的方向：被同一个预测合并的两个真值核，是上下（z 向）叠着还是左右（xy 向）挨着。

对每个“合并”预测（覆盖 ≥2 个真值核各一半以上），取被合并的真值核两两之间的质心位移（换成 µm），
按 |dz| 与 |dxy| 的比较记为 z 向（|dz| > |dxy|）或 xy 向；并与同数据集所有相邻真值核对（最近邻）的方向比例对照，
看 z 向合并是否明显多于“随机碰到的相邻核”。
用法：python merge_dir.py --raw <results 前缀> --out <目录> [--thr -2 --frac 0.25]
"""
import argparse, json, os
from collections import defaultdict
import numpy as np, tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from metrics import iou_matrix
from eval_cpsam import load_case, drop_ignored
from eval_val import pick_cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--thr", default="-2"); ap.add_argument("--frac", type=float, default=0.25)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    sj = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "val_split.json")
    acc = defaultdict(lambda: dict(merge_z=0, merge_xy=0, nn_z=0, nn_xy=0))
    for sp in ("val", "test"):
        raw = f"{a.raw}_{sp}/raw"
        for r in pick_cases(sp, sj, 4):
            key = f"{r['ds']}__{r['id']}"
            if not os.path.exists(f"{raw}/{key}__meta.json"):
                continue
            meta = json.load(open(f"{raw}/{key}__meta.json")); vox = np.array(meta["vox"]); d = meta["d_vol"]
            img, gt, box, ign = load_case(r, 384)
            if ign is not None:
                gt = gt.copy(); gt[ign] = 0
            pred = drop_ignored(tifffile.imread(f"{raw}/{key}__cp{a.thr}.tif").astype(np.int32), ign)
            exp_vol = np.pi / 6 * d ** 3 * vox[1] / vox[0]; cnt = np.bincount(pred.ravel())
            small = np.where(cnt < a.frac * exp_vol)[0]; small = small[small > 0]; pred[np.isin(pred, small)] = 0
            iou, gids, pids = iou_matrix(gt, pred)
            if not len(gids) or not len(pids):
                continue
            gsz = np.bincount(gt.ravel())[gids]; psz = np.bincount(pred.ravel())[pids]
            inter = iou * (gsz[:, None] + psz[None, :]) / (1 + iou); cov = inter / gsz[:, None]
            cen = np.array(ndi.center_of_mass(gt > 0, gt, gids)) * vox
            s = acc[r["ds"]]
            for j in range(len(pids)):
                g = np.where(cov[:, j] > 0.5)[0]
                for i1 in range(len(g)):
                    for i2 in range(i1 + 1, len(g)):
                        dv = np.abs(cen[g[i1]] - cen[g[i2]])
                        s["merge_z" if dv[0] > np.hypot(dv[1], dv[2]) else "merge_xy"] += 1
            if len(cen) > 1:
                _, nn = cKDTree(cen).query(cen, k=2)
                for i, k in enumerate(nn[:, 1]):
                    dv = np.abs(cen[i] - cen[k])
                    s["nn_z" if dv[0] > np.hypot(dv[1], dv[2]) else "nn_xy"] += 1
            print(f"  {sp} {r['ds']:13s} {r['id'][:28]:28s} " + " ".join(f"{k}={v}" for k, v in s.items()), flush=True)
    print("\n数据集          合并对数  其中 z 向   相邻核对中 z 向（对照）")
    out = {}
    for ds, s in sorted(acc.items()):
        m = s["merge_z"] + s["merge_xy"]; n = s["nn_z"] + s["nn_xy"]
        out[ds] = dict(**s, merge_z_frac=s["merge_z"] / max(m, 1), nn_z_frac=s["nn_z"] / max(n, 1))
        print(f"  {ds:13s} {m:7d}   {s['merge_z'] / max(m, 1):6.2f}      {s['nn_z'] / max(n, 1):6.2f}")
    json.dump(out, open(f"{a.out}/merge_dir.json", "w"), indent=1)
    print("FINAL_METRICS " + json.dumps({ds: round(v["merge_z_frac"], 3) for ds, v in out.items()}), flush=True)
    print("SUMMARY 合并方向：" + "；".join(f"{ds} z 向 {out[ds]['merge_z_frac']:.2f}（对照 {out[ds]['nn_z_frac']:.2f}）" for ds in ("cellos", "parhyale", "nis3d") if ds in out), flush=True)


if __name__ == "__main__":
    main()
