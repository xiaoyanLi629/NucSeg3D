"""事后分析（只报告，不用于任何选择或设计）：真值把相邻核合成一个对象时，模型把它们分开会被记成错误。
逐体积比较标准 F1@0.5 与“容忍拆分”的 F1@0.5：预测对象有一半以上体素落在某个真值对象里，就归到这个真值对象；
一个真值对象的全部归属预测合起来与它的 IoU ≥ 0.5，就算命中，这些预测都不计假阳性。标准命中一定也是容忍拆分下的命中。
同时数：靠多个预测合起来才命中的真值对象（“被拆开的真值”），以及其中真值体积 ≥ 1.5 倍预期单核体积的个数（可能是多核团）。
掩膜与 test_r6.py 的 hits() 相同：阈值 −2 的预测（npz 的 pred）删去体积 < 0.15 倍预期体积的碎块；所有体积一律用固定 −2
（规则 T 只改极少数体积，这里不跟随）。
用法：python merge_tol.py <输出.json> [数据集 ...]（不给数据集时只算 purdue）"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import glob
import json
import math
import os
import sys

import numpy as np

RES = f"{_NSP_ROOT}/results"
SETS = {
    "val": {"S1": ["a69ee4fa-3a93-4916-97e1-dd853aeca060", "23e42a91-b7bf-4d12-ad51-62453abfabcd", "d9c72498-0c02-406a-8ed9-28466edeec61"],
            "2D": ["6afbb879-93cd-4d3b-a4a3-cff5ebc6b30a", "41e1a0ec-67f2-4a47-a599-602c9b3b33cc", "be20444a-22b7-402d-af70-86ca56121ee6"],
            "base": ["7095d247-8ddf-4ed7-9af5-7374438a4905"]},
    "test_fp": {"S1": ["36d5325c-8dac-4521-80df-9e204515239c", "68379c3b-2d19-47f4-8edc-f19734144ff6", "943ab4c5-9b91-44ec-baaf-a7ec4756769a"],
                "2D": ["b10fa739-bc32-45ea-8c3c-4f6975f90cbe", "3e8817e0-68dd-4877-a60c-90a07ce66242", "e90a3427-04ab-45ca-a447-2fdb43e71840"],
                "base": ["99b47f45-48a3-4306-8e99-b0056c66c143"]},
}
DROP = {"bbbc032", "mesospim"}


def one(npz):
    z = np.load(npz)
    gt, pm, d, vox = z["gt"].astype(np.int64), z["pred"].astype(np.int64), float(z["d"]), z["vox"]
    ev = math.pi / 6 * d ** 3 * vox[1] / vox[0]
    cnt = np.bincount(pm.ravel())
    small = np.nonzero(cnt < 0.15 * ev)[0]
    pm = np.where(np.isin(pm, small[small > 0]), 0, pm)
    g, p = gt.ravel(), pm.ravel()
    sel = (g > 0) & (p > 0)
    K = int(p.max()) + 1
    key, c = np.unique(g[sel] * K + p[sel], return_counts=True)
    gs, ps = np.bincount(g), np.bincount(p)
    n_gt, n_pr = int((np.unique(g) > 0).sum()), int((np.unique(p) > 0).sum())
    gi, pi = key // K, key % K
    iou = c / (gs[gi] + ps[pi] - c)
    tp = int((iou >= 0.5).sum())
    own = c >= 0.5 * ps[pi]                                  # 预测的多数体素在这个真值里（每个预测至多一个）
    og, op, oc = gi[own], pi[own], c[own]
    I = np.bincount(og, weights=oc, minlength=len(gs))        # 每个真值：归属预测与它的交
    U = np.bincount(og, weights=ps[op], minlength=len(gs))    # 归属预测的总体积
    k = np.bincount(og, minlength=len(gs))                    # 归属预测个数
    ids = np.nonzero(k)[0]
    iou_g = I[ids] / (gs[ids] + U[ids] - I[ids])
    hit = ids[iou_g >= 0.5]
    tp_t = len(hit)
    used = int(k[hit].sum())                                  # 命中组里的预测都不算假阳性
    fp_t, fn_t = n_pr - used, n_gt - tp_t
    split = hit[k[hit] >= 2]
    big = int((gs[split] >= 1.5 * ev).sum())
    gt_big = int((gs[np.unique(g[g > 0])] >= 1.5 * ev).sum())
    return dict(f1=2 * tp / max(n_gt + n_pr, 1), f1_tol=2 * tp_t / max(2 * tp_t + fp_t + fn_t, 1),
                n_gt=n_gt, n_pred=n_pr, split_hits=len(split), split_big=big, gt_big=gt_big)


def main():
    global WANT
    WANT = set(sys.argv[2:]) or {"purdue"}
    out = {}
    for split, M in SETS.items():
        for nm, runs in M.items():
            for r in runs:
                d = f"{RES}/orx_{r}_thr_eval"
                for f in sorted(glob.glob(d + "/masks/*.npz")):
                    ds, vid = os.path.basename(f)[:-4].split("__", 1)
                    if ds in DROP or ds not in WANT:
                        continue
                    out.setdefault(split, {}).setdefault(nm, {}).setdefault(r[:8], {})[f"{ds}__{vid}"] = one(f)
                print(split, nm, r[:8], "完成", flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for split in out:
        print(f"\n== {split}：每套逐体积平均（三种子平均）  标准 F1@0.5 | 容忍拆分 F1@0.5 | 靠拆分命中的真值数（其中 ≥1.5 倍预期体积）/ 真值数")
        dss = sorted({k.split("__")[0] for nm in out[split] for r in out[split][nm] for k in out[split][nm][r]})
        for ds in dss:
            line = f"  {ds:12s}"
            for nm in ("S1", "2D", "base"):
                vals = [[v for k, v in out[split][nm][r].items() if k.startswith(ds + "__")] for r in out[split][nm]]
                f1 = np.mean([np.mean([v["f1"] for v in V]) for V in vals])
                ft = np.mean([np.mean([v["f1_tol"] for v in V]) for V in vals])
                sh = np.mean([sum(v["split_hits"] for v in V) for V in vals])
                sb = np.mean([sum(v["split_big"] for v in V) for V in vals])
                ng = np.mean([sum(v["n_gt"] for v in V) for V in vals])
                line += f"   {nm} {f1:.3f} | {ft:.3f} | {sh:.0f}（{sb:.0f}）/{ng:.0f}"
            print(line)


if __name__ == "__main__":
    main()
