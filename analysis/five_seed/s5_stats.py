"""五种子报告（AGENTS.md f20d0e1 写定的口径，只报告）：S1 + 规则 T 对二维 + 规则 T 的同种子配对差，以及逐套判定。
口径与 test_r6.py 完全相同（直接调用它的 recs / score / call / hits / mcnemar）：
  验证集 gt 协议（视图规则、规则 T）；测试集完整流程（主）与 gt 协议（补充），规则 T；原版只有一个模型。
宏平均：每个种子的 12 套（去 mesospim）与 13 套宏平均，配对差的均值与 SD（ddof=1），确认规则：均值 > 2×SD 且 ≥ 0.01。
逐套：按体积配对自助 10000 次，所有种子共用同一组体积索引；区间下界 > 0 为“胜”，≥ −0.02 为“不劣”，否则为“劣”；单体积套给逐核 McNemar。
只给 --seeds s0 s1000 s2000 时，测试集的逐套结果应与 test_r6.py 逐位相同（自检）。
--twod_fixed：二维对照用固定 cp−2（不套规则 T），只作敏感性分析。
--subset 名单.json：只统计名单里的体积（[[数据集, 体积名], ...]，或 {"keep": [...]}，或审计的 clean_test.json 即 {"clean": [...]}），没有体积剩下的数据集不进宏平均；用于泄漏审计后的“干净测试子集”敏感性分析。
用法：python s5_stats.py val|test fp|gt [--seeds s0 s1000 ...] [--base_fixed] [--twod_fixed] [--subset 名单.json] [--out 结果.json]"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import os
import sys

import numpy as np

sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/five_seed'))
from test_r6 import B, RES, STATE, call, hits, mcnemar, recs, score  # noqa: E402

# 第六轮验证集 thr_eval（gt 协议、视图规则、阈值网格）运行编号；种子 3000、4000 由五种子扩充的作业给出
VAL = {"S1": {"s0": "a69ee4fa-3a93-4916-97e1-dd853aeca060", "s1000": "23e42a91-b7bf-4d12-ad51-62453abfabcd",
              "s2000": "d9c72498-0c02-406a-8ed9-28466edeec61"},
       "2D": {"s0": "6afbb879-93cd-4d3b-a4a3-cff5ebc6b30a", "s1000": "41e1a0ec-67f2-4a47-a599-602c9b3b33cc",
              "s2000": "be20444a-22b7-402d-af70-86ca56121ee6"},
       "base": "7095d247-8ddf-4ed7-9af5-7374438a4905"}


def d_of(run):
    d = f"{RES}/orx_{run}_thr_eval"
    assert os.path.exists(f"{d}/rows.json"), d
    return d


def dirs(split, mode, seeds):
    st = json.load(open(STATE, encoding="utf-8"))
    run = lambda k: d_of(st[k]["run"])
    out = {"S1": [], "2D": []}
    for s in seeds:
        new = s in ("s3000", "s4000")
        for nm in ("S1", "2D"):
            if split == "val":
                out[nm].append(run(f"S5_v{nm}_{s}") if new else d_of(VAL[nm][s]))
            else:
                pre = "tFP" if mode == "fp" else "tGT"
                out[nm].append(run(f"S5_{pre}_{nm}_{s}") if new else run(f"{pre}_{nm}_{s}"))
    out["base"] = [d_of(VAL["base"]) if split == "val" else run("tFP_base" if mode == "fp" else "tGT_base")]
    return out


def main():
    split, mode = sys.argv[1], sys.argv[2]
    assert split in ("val", "test") and mode in ("fp", "gt") and not (split == "val" and mode == "fp")
    seeds = sys.argv[sys.argv.index("--seeds") + 1:] if "--seeds" in sys.argv else ["s0", "s1000", "s2000", "s3000", "s4000"]
    seeds = [s for s in seeds if s.startswith("s") and s[1:].isdigit()]
    D = dirs(split, mode, seeds)
    M = {"S1": [recs(d) for d in D["S1"]], "2D": [recs(d, fixed="--twod_fixed" in sys.argv) for d in D["2D"]],
         "base": [recs(D["base"][0], fixed="--base_fixed" in sys.argv)]}
    common = set.intersection(*[set(R) for L in M.values() for R in L])
    sub = sys.argv[sys.argv.index("--subset") + 1] if "--subset" in sys.argv else None
    if sub:
        keep = json.load(open(sub, encoding="utf-8"))
        keep = {tuple(x) for x in ((keep["keep"] if "keep" in keep else keep["clean"]) if isinstance(keep, dict) else keep)}
        assert keep & common, "名单与评测记录没有交集（键的格式不对？）"
        print(f"子集名单 {sub}：{len(keep)} 个体积，其中 {len(keep & common)} 个在本口径的 {len(common)} 个评测体积里")
        common &= keep
    dss = sorted({ds for ds, _ in common})
    rng = np.random.default_rng(0)
    res = {"split": split, "mode": mode, "seeds": seeds, "dirs": D, "subset": sub, "per_ds": {}, "macro": {}}
    print(f"== {split} {'完整流程，逐体积再平均' if mode == 'fp' else 'gt 协议，pooled'}，种子 {'、'.join(seeds)}；{len(common)} 个体积，{len(dss)} 套；规则 T 改阈值的体积："
          + "，".join(f"{nm} {sum(sum(R[x]['t'] != -2.0 for x in common) for R in L)}" for nm, L in M.items()))
    for k, lab in (("tp5", "F1@0.5"), ("tp7", "F1@0.7")):
        print(f"\n-- {lab}：每套种子均值（S1 | 二维 | 原版），S1−二维 与 S1−原版 的种子平均差、95% 配对自助区间与判定")
        for ds in dss:
            keys = sorted(x for x in common if x[0] == ds)
            v = {nm: float(np.mean([score(R, keys, mode, k) for R in L])) for nm, L in M.items()}
            d2 = float(np.mean([score(a, keys, mode, k) - score(b, keys, mode, k) for a, b in zip(M["S1"], M["2D"])]))
            db = v["S1"] - v["base"]
            ent = dict(n=len(keys), S1=v["S1"], two_d=v["2D"], base=v["base"], d2d=d2, dbase=db,
                       d2d_seeds=[score(a, keys, mode, k) - score(b, keys, mode, k) for a, b in zip(M["S1"], M["2D"])])
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
            print(f"  {ds:12s} n={len(keys):3d}  {v['S1']:.3f} | {v['2D']:.3f} | {v['base']:.3f}   {txt}")
        tags = (("13 套", lambda s: True), ("去 mesospim 12 套", lambda s: s != "mesospim")) if not sub else (
            (f"子集 {len(dss)} 套", lambda s: True), (f"子集去 mesospim {sum(ds != 'mesospim' for ds in dss)} 套", lambda s: s != "mesospim"))
        for tag, keep in tags:
            ix = [ds for ds in dss if keep(ds)]
            per = {nm: [float(np.mean([score(R, sorted(x for x in common if x[0] == ds), mode, k) for ds in ix])) for R in L] for nm, L in M.items()}
            diff = [a - b for a, b in zip(per["S1"], per["2D"])]
            mu, sd = float(np.mean(diff)), float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan")
            ok = bool(mu > 2 * sd and mu >= 0.01)
            res["macro"].setdefault(lab, {})[tag] = dict(S1_seeds=per["S1"], two_d_seeds=per["2D"], base=per["base"][0],
                                                         diff_seeds=diff, diff_mean=mu, diff_sd=sd, confirmed=ok,
                                                         S1=float(np.mean(per["S1"])), two_d=float(np.mean(per["2D"])))
            print(f"  宏平均（{tag}）S1 {np.mean(per['S1']):.4f}（{'、'.join(f'{x:.4f}' for x in per['S1'])}）"
                  f" 二维 {np.mean(per['2D']):.4f}（{'、'.join(f'{x:.4f}' for x in per['2D'])}） 原版 {per['base'][0]:.4f}")
            print(f"    S1−二维 逐种子 {'、'.join(f'{x:+.4f}' for x in diff)}；均值 {mu:+.4f}，SD {sd:.4f}，"
                  f"{'通过' if ok else '未通过'}确认（均值 > 2×SD 且 ≥ 0.01）")
        for who, key in (("对二维", "call2d"), ("对原版", "callbase")):
            c = [e.get(key) for e in res["per_ds"][lab].values()]
            print(f"  逐套判定（{who}）：胜 {c.count('胜')}、不劣 {c.count('不劣')}、劣 {c.count('劣')}、单体积 {c.count('单体积不判定')}")
    out = (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else
           f"{_NSP_SCRATCH}/r6/s5_{split}_{mode}_{len(seeds)}seeds{'_basefixed' if '--base_fixed' in sys.argv else ''}{'_2dfixed' if '--twod_fixed' in sys.argv else ''}"
           f"{'_' + os.path.basename(sub)[:-5] if sub else ''}.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("写入", out)


if __name__ == "__main__":
    main()
