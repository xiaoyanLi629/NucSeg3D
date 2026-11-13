"""第六轮测试集（只报告）：按 AGENTS.md 预定的逐套判定规则（0fecb3f）比较最终流程 S1 + 规则 T 与二维 + 规则 T、原版 + 规则 T。
口径：完整流程（fp）逐体积 F1 → 每套平均 → 宏平均；gt 协议（gt）每套 pooled → 宏平均。13 套，去掉已退役的 bbbc032。
规则 T（R2）逐体积选阈值，与 sizeThr_dir.py 相同；体积下限 0.15。
逐套：按体积配对自助 10000 次，三个种子共用同一组体积索引（同种子配对），原版与三种子平均比较；
区间下界 > 0 为“胜”，≥ −0.02 为“不劣”，否则为“劣”；只有 1 个体积的套不做区间，给点估计与逐核命中的 McNemar（IoU ≥ 0.5 / 0.7）。
用法：python test_r6.py fp|gt [--base_fixed] [--out 结果.json]（--base_fixed：原版用固定 cp−2，不套规则 T）"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import glob
import json
import math
import os
import sys

import numpy as np

RES = f"{_NSP_ROOT}/results"
STATE = f"{_NSP_SCRATCH}/r6/sched_state.json"
SEEDS = ("s0", "s1000", "s2000")
B = 10000
DROP = {"bbbc032"}


def run_dir(key):
    s = json.load(open(STATE, encoding="utf-8"))[key]
    d = f"{RES}/orx_{s['run']}_thr_eval"
    assert os.path.exists(f"{d}/rows.json"), d
    return d


def recs(d, fixed=False):
    """fixed=True 时一律用固定 cp−2（不套规则 T，用于“原版不加规则 T”的补充对照）。否则与 sizeThr_dir.py 相同的 R2：m(t) = #(cp > t) / 预测核数(t, 不删小块) / V，V = π/6·d³/ratio。返回 {(ds, id): 记录}。"""
    rows = json.load(open(d + "/rows.json"))
    by = {(x["ds"], x["id"], x["thr"], x["frac"]): x for x in rows}
    thrs = sorted({x["thr"] for x in rows}, reverse=True)
    out = {}
    for f in sorted(glob.glob(d + "/masks/*.npz")):
        ds, vid = os.path.basename(f)[:-4].split("__", 1)
        if ds in DROP:
            continue
        cp = np.load(f)["cp"].astype(np.float32)
        r0 = by[(ds, vid, -2.0, 0.0)]
        dd = r0.get("d_used", r0["d_gt"]); V = math.pi / 6 * dd ** 3 / r0["ratio"]
        m = {t: float((cp > t).sum()) / max(by[(ds, vid, t, 0.0)]["n_pred"], 1) / V for t in thrs}
        t2 = -2.0 if (fixed or m[-2.0] >= 0.7) else next((t for t in thrs if t < -2 and m[t] >= 0.85), min(thrs))
        r = by[(ds, vid, t2, 0.15)]
        out[(ds, vid)] = dict(t=t2, m2=m[-2.0], tp5=r["tp5"], tp7=r["tp7"], n=r["n_gt"] + r["n_pred"], n_gt=r["n_gt"], n_pred=r["n_pred"], npz=f)
    return out


def score(R, keys, mode, k):
    if mode == "fp":
        return float(np.mean([2 * R[x][k] / max(R[x]["n"], 1) for x in keys]))
    return 2 * sum(R[x][k] for x in keys) / max(sum(R[x]["n"] for x in keys), 1)


def call(lo):
    return "胜" if lo > 0 else ("不劣" if lo >= -0.02 else "劣")


def hits(npz, t, frac=0.15):
    """逐个真值核是否被命中（IoU ≥ 0.5、≥ 0.7）；最终掩膜 = 阈值 t 的预测再删去体积 < frac·预期体积的小块（t ≠ −2 时由存下的流场重算）。"""
    z = np.load(npz)
    gt = z["gt"]; d = float(z["d"]); vox = z["vox"]
    if t == -2.0:
        pm = z["pred"]
    else:
        from cellpose import dynamics
        dP, cp = z["dP"].astype(np.float32), z["cp"].astype(np.float32)
        pm = dynamics.compute_masks(dP, cp, cellprob_threshold=t, flow_threshold=0.4, min_size=15, max_size_fraction=0.4,
                                    niter=int(200 / (30.0 / d)), do_3D=True)
        pm = np.asarray(pm).squeeze()
        assert pm.shape == gt.shape, (pm.shape, gt.shape)
    ev = np.pi / 6 * d ** 3 * vox[1] / vox[0]
    cnt = np.bincount(pm.ravel())
    small = np.nonzero(cnt < frac * ev)[0]
    pm = np.where(np.isin(pm, small[small > 0]), 0, pm)
    g, p = gt.ravel().astype(np.int64), pm.ravel().astype(np.int64)
    sel = (g > 0) | (p > 0)
    pairs, inter = np.unique(np.stack([g[sel], p[sel]]), axis=1, return_counts=True)
    gs, ps = np.bincount(g), np.bincount(p)
    best = {}
    for (gi, pi), c in zip(pairs.T, inter):
        if gi and pi:
            iou = c / (gs[gi] + ps[pi] - c)
            best[gi] = max(best.get(gi, 0.0), iou)
    ids = np.unique(g[g > 0])
    h5 = np.array([best.get(i, 0.0) >= 0.5 for i in ids]); h7 = np.array([best.get(i, 0.0) >= 0.7 for i in ids])
    return h5, h7, int((np.unique(p) > 0).sum())


def mcnemar(a, b):
    from math import comb
    x, y = int((a & ~b).sum()), int((~a & b).sum())
    n = x + y
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(0, min(x, y) + 1)) / 2 ** n)
    return x, y, p


def main():
    mode = sys.argv[1]; assert mode in ("fp", "gt")
    pre = "tFP" if mode == "fp" else "tGT"
    M = {"S1": [recs(run_dir(f"{pre}_S1_{s}")) for s in SEEDS], "2D": [recs(run_dir(f"{pre}_2D_{s}")) for s in SEEDS],
         "base": [recs(run_dir(f"{pre}_base"), fixed="--base_fixed" in sys.argv)]}
    common = set.intersection(*[set(R) for L in M.values() for R in L])
    dss = sorted({ds for ds, _ in common})
    rng = np.random.default_rng(0)
    res = {"mode": mode, "per_ds": {}, "macro": {}}
    print(f"== 第六轮测试集（{'完整流程，逐体积再平均' if mode == 'fp' else 'gt 协议，pooled'}），{len(common)} 个体积，{len(dss)} 套；规则 T 改阈值的体积："
          + "，".join(f"{nm} {sum(sum(R[x]['t'] != -2.0 for x in common) for R in L)}" for nm, L in M.items()))
    for k, lab in (("tp5", "F1@0.5"), ("tp7", "F1@0.7")):
        print(f"\n-- {lab}：每套三种子均值（S1 | 二维 | 原版），S1−二维 与 S1−原版 的三种子平均差及 95% 配对自助区间与判定")
        macro = {nm: [] for nm in M}
        for ds in dss:
            keys = sorted(x for x in common if x[0] == ds)
            v = {nm: float(np.mean([score(R, keys, mode, k) for R in L])) for nm, L in M.items()}
            for nm in M:
                macro[nm].append(v[nm])
            d2 = float(np.mean([score(a, keys, mode, k) - score(b, keys, mode, k) for a, b in zip(M["S1"], M["2D"])]))
            db = v["S1"] - v["base"]
            ent = dict(n=len(keys), S1=v["S1"], two_d=v["2D"], base=v["base"], d2d=d2, dbase=db)
            if len(keys) >= 2:
                bs2, bsb = np.empty(B), np.empty(B)
                for i in range(B):
                    idx = [keys[j] for j in rng.integers(0, len(keys), len(keys))]
                    s1 = [score(R, idx, mode, k) for R in M["S1"]]
                    bs2[i] = np.mean([a - score(b, idx, mode, k) for a, b in zip(s1, M["2D"])])
                    bsb[i] = np.mean(s1) - score(M["base"][0], idx, mode, k)
                ent.update(ci2d=[float(np.percentile(bs2, 2.5)), float(np.percentile(bs2, 97.5))],
                           cibase=[float(np.percentile(bsb, 2.5)), float(np.percentile(bsb, 97.5))])
                ent.update(call2d=call(ent["ci2d"][0]), callbase=call(ent["cibase"][0]))
                txt = (f"二维 {d2:+.3f} [{ent['ci2d'][0]:+.3f}, {ent['ci2d'][1]:+.3f}] {ent['call2d']}  "
                       f"原版 {db:+.3f} [{ent['cibase'][0]:+.3f}, {ent['cibase'][1]:+.3f}] {ent['callbase']}")
            else:
                hk = 0 if k == "tp5" else 1
                mc = []
                for a, b in zip(M["S1"], M["2D"]):
                    ha, hb = hits(a[keys[0]]["npz"], a[keys[0]]["t"]), hits(b[keys[0]]["npz"], b[keys[0]]["t"])
                    mc.append(mcnemar(ha[hk], hb[hk]) + (ha[2], hb[2]))
                hb0 = hits(M["base"][0][keys[0]]["npz"], M["base"][0][keys[0]]["t"])
                mcb = [mcnemar(hits(a[keys[0]]["npz"], a[keys[0]]["t"])[hk], hb0[hk]) for a in M["S1"]]
                ent.update(mcnemar2d=mc, mcnemarbase=mcb, call2d="单体积不判定", callbase="单体积不判定")
                txt = ("单体积，逐核 McNemar（S1 独中/二维独中/p；预测核数）："
                       + "；".join(f"{x}/{y}/p={p:.3g}（{na} 对 {nb}）" for x, y, p, na, nb in mc)
                       + "  对原版：" + "；".join(f"{x}/{y}/p={p:.3g}" for x, y, p in mcb))
            res["per_ds"].setdefault(lab, {})[ds] = ent
            print(f"  {ds:12s} n={len(keys)}  {v['S1']:.3f} | {v['2D']:.3f} | {v['base']:.3f}   {txt}")
        for tag, keep in (("13 套", lambda s: True), ("去 mesospim 12 套", lambda s: s != "mesospim")):
            ix = [i for i, s in enumerate(dss) if keep(s)]
            mm = {nm: float(np.mean([macro[nm][i] for i in ix])) for nm in M}
            per_seed = [float(np.mean([score(a, sorted(x for x in common if x[0] == dss[i]), mode, k) for i in ix])) for a in M["S1"]]
            per_seed2 = [float(np.mean([score(b, sorted(x for x in common if x[0] == dss[i]), mode, k) for i in ix])) for b in M["2D"]]
            res["macro"].setdefault(lab, {})[tag] = dict(mm, S1_seeds=per_seed, two_d_seeds=per_seed2)
            print(f"  宏平均（{tag}）S1 {mm['S1']:.4f}（{'、'.join(f'{x:.4f}' for x in per_seed)}） 二维 {mm['2D']:.4f}（{'、'.join(f'{x:.4f}' for x in per_seed2)}） 原版 {mm['base']:.4f}")
        calls = [e.get("call2d") for e in res["per_ds"][lab].values()]
        print(f"  逐套判定（对二维）：胜 {calls.count('胜')}、不劣 {calls.count('不劣')}、劣 {calls.count('劣')}、单体积 {sum(c == '单体积不判定' for c in calls)}")
        callsb = [e.get("callbase") for e in res["per_ds"][lab].values()]
        print(f"  逐套判定（对原版）：胜 {callsb.count('胜')}、不劣 {callsb.count('不劣')}、劣 {callsb.count('劣')}、单体积 {sum(c == '单体积不判定' for c in callsb)}")
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"{_NSP_SCRATCH}/r6/test_r6_{mode}{'_basefixed' if '--base_fixed' in sys.argv else ''}.json"
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("写入", out)


if __name__ == "__main__":
    main()
