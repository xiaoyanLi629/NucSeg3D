"""nc A5 第二步（CPU）：测试体积到训练分布的距离与 F1 的关系。设计见 DESIGN_G2.md 第 A5 节。
读 a5_feat.py 的 feats.npz。参照 = 910 个训练体积的 256 维特征。
距离：（1）马氏距离，协方差用 Ledoit–Wolf 收缩估计（sklearn），均值与协方差只用训练体积；
     （2）最近邻距离：按训练体积的均值、标准差逐维标准化后，到最近训练体积的欧氏距离（k=1，另报 k=5 的平均）。
F1：我们的模型 = S1 种子 0、gt 协议、规则 T（运行 07992b55；规则 T 按 sizeThr_dir 的 R2 从 masks/*.npz 重算，体积下限 0.15）的逐体积 F1@0.5，
    另报 F1@0.7。
检验：斯皮尔曼相关（227 个测试体积）；因为数据集是最大的混杂，另报（a）先减去各套均值后的体积级斯皮尔曼，（b）13 套数据集
    平均距离与 pooled F1 的斯皮尔曼；去 mesospim 的结果也给。UMAP（umap-learn，n_neighbors 15、min_dist 0.1、random_state 0，
    在标准化特征上拟合训练 + 验证 + 测试）坐标存盘供作图；没装 umap 时跳过并说明。
用法：python a5_dist.py --feats <feats.npz> --out <目录> [--run 07992b55 结果目录]"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import argparse
import glob
import json
import math
import os

import numpy as np

R = f"{_NSP_ROOT}/results/orx_07992b55-0982-4116-b065-b34859c9a80b_thr_eval"


def rule_t_f1(d):
    rows = json.load(open(d + "/rows.json"))
    by = {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in rows}
    thrs = sorted({x["thr"] for x in rows}, reverse=True)
    out = {}
    for f in sorted(glob.glob(d + "/masks/*.npz")):
        ds, vid = os.path.basename(f)[:-4].split("__", 1)
        cp = np.load(f)["cp"].astype(np.float32)
        r0 = by[(ds, vid, -2.0, 0.0)]
        dd = r0.get("d_used", r0["d_gt"]); V = math.pi / 6 * dd ** 3 / r0["ratio"]
        m = {t: float((cp > t).sum()) / max(by[(ds, vid, t, 0.0)]["n_pred"], 1) / V for t in thrs}
        t2 = -2.0 if m[-2.0] >= 0.7 else next((t for t in thrs if t < -2 and m[t] >= 0.85), min(thrs))
        x = by[(ds, vid, t2, 0.15)]
        out[f"{ds}__{vid}"] = dict(ds=ds, f1=x["f1"], f1_7=x["f1_7"], tp5=x["tp5"], tp7=x["tp7"], den=x["n_gt"] + x["n_pred"], t=t2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True); ap.add_argument("--out", required=True); ap.add_argument("--run", default=R)
    a = ap.parse_args()
    from scipy.stats import spearmanr
    from sklearn.covariance import LedoitWolf
    os.makedirs(a.out, exist_ok=True)
    z = np.load(a.feats)
    keys, sp, ds, X = z["keys"], z["split"], z["ds"], z["feat"].astype(np.float64)
    tr = sp == "train"; te = sp == "test"
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xs = (X - mu) / sd
    lw = LedoitWolf().fit(X[tr]); P = lw.precision_
    dm = np.sqrt(np.einsum("ij,jk,ik->i", X - lw.location_, P, X - lw.location_))
    D2 = ((Xs[:, None, :] - Xs[None, tr, :]) ** 2).sum(-1) ** 0.5
    D2s = np.sort(D2, 1); nn1, nn5 = D2s[:, 0], D2s[:, :5].mean(1)
    F = rule_t_f1(a.run)
    idx = [i for i in np.where(te)[0] if keys[i] in F]
    res = dict(n_train=int(tr.sum()), n_test=len(idx), shrinkage=float(lw.shrinkage_), tests={})
    for nm, dist in (("mahal", dm), ("nn1", nn1), ("nn5", nn5)):
        for met in ("f1", "f1_7"):
            for tag, keep in (("all", lambda i: True), ("no_meso", lambda i: ds[i] != "mesospim")):
                ii = [i for i in idx if keep(i)]
                x = np.array([dist[i] for i in ii]); y = np.array([F[keys[i]][met] for i in ii]); g = np.array([ds[i] for i in ii])
                rho, p = spearmanr(x, y)
                xc, yc = x.copy(), y.copy()
                for s in set(g):
                    xc[g == s] -= x[g == s].mean(); yc[g == s] -= y[g == s].mean()
                rw, pw = spearmanr(xc, yc)
                S = sorted(set(g))
                xd = [x[g == s].mean() for s in S]
                yd = [2 * sum(F[keys[i]][("tp5" if met == "f1" else "tp7")] for i in ii if ds[i] == s) / max(sum(F[keys[i]]["den"] for i in ii if ds[i] == s), 1) for s in S]
                rd, pd = spearmanr(xd, yd)
                res["tests"][f"{nm}|{met}|{tag}"] = dict(rho=float(rho), p=float(p), rho_within=float(rw), p_within=float(pw),
                                                       rho_dataset=float(rd), p_dataset=float(pd), n=len(ii), n_ds=len(S))
    per = {keys[i]: dict(ds=str(ds[i]), split=str(sp[i]), mahal=float(dm[i]), nn1=float(nn1[i]), nn5=float(nn5[i]),
                         **({k: F[keys[i]][k] for k in ("f1", "f1_7", "t")} if keys[i] in F else {})) for i in range(len(keys))}
    try:
        import umap
        U = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(Xs)
        for i, k in enumerate(keys):
            per[k]["umap"] = [float(U[i, 0]), float(U[i, 1])]
        res["umap"] = "done"
    except ImportError:
        res["umap"] = "umap-learn 未安装，跳过"
    json.dump(dict(summary=res, per_volume=per), open(f"{a.out}/a5_dist.json", "w"), indent=1)
    for k, v in res["tests"].items():
        print(f"SUMMARY A5 {k}: ρ={v['rho']:+.3f} (p={v['p']:.2g})  套内 ρ={v['rho_within']:+.3f}  套级 ρ={v['rho_dataset']:+.3f} (n_ds={v['n_ds']})", flush=True)


if __name__ == "__main__":
    main()
