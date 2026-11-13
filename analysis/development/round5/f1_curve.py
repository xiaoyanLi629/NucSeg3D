"""IoU 阈值 0.1–0.9 上的实例 F1 曲线（pooled，读 eval_gt --save_masks 的 npz；预测先按主格删掉体积 < 0.15·π/6·d³·vxy/vz 的小块）。
一对一匹配：按 IoU 从大到小贪心配对（IoU ≥ 0.5 时与唯一匹配相同）。用法：python f1_curve.py <结果目录> [...]"""
import glob, os, sys
import numpy as np

TS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def match_counts(P, G):
    P = P.astype(np.int64).ravel(); G = G.astype(np.int64).ravel()
    npd, ng = int(P.max()), int(G.max())
    ap = np.bincount(P, minlength=npd + 1); ag = np.bincount(G, minlength=ng + 1)
    m = (P > 0) & (G > 0)
    key = P[m] * (ng + 1) + G[m]
    u, c = np.unique(key, return_counts=True)
    pi, gi = u // (ng + 1), u % (ng + 1)
    iou = c / (ap[pi] + ag[gi] - c)
    o = np.argsort(-iou); pi, gi, iou = pi[o], gi[o], iou[o]
    n_pred = int((ap[1:] > 0).sum()); n_gt = int((ag[1:] > 0).sum())
    tp = {}
    for t in TS:
        usedp, usedg, k = set(), set(), 0
        for a, b, v in zip(pi, gi, iou):
            if v < t:
                break
            if a in usedp or b in usedg:
                continue
            usedp.add(a); usedg.add(b); k += 1
        tp[t] = k
    return tp, n_gt, n_pred


for d in sys.argv[1:]:
    TP = {t: 0 for t in TS}; NG = NP = 0
    for f in sorted(glob.glob(f"{d}/masks/*.npz")):
        D = np.load(f); P = D["pred"].astype(np.int64)
        vox = D["vox"]; dd = float(D["d"])
        exp_vol = np.pi / 6 * dd ** 3 * vox[1] / vox[0]
        cnt = np.bincount(P.ravel()); small = np.where(cnt < 0.15 * exp_vol)[0]; small = small[small > 0]
        if len(small):
            P[np.isin(P, small)] = 0
        tp, ng, npd = match_counts(P, D["gt"])
        for t in TS:
            TP[t] += tp[t]
        NG += ng; NP += npd
    print(f"F1CURVE {os.path.basename(d.rstrip('/')):22s} n_gt={NG} n_pred={NP}  " + "  ".join(f"@{t:g} {2 * TP[t] / max(NG + NP, 1):.3f}" for t in TS), flush=True)
