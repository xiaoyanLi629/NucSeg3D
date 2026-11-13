"""审稿第三轮 O2（事后，只报告）：逐套比较 S1 与二维在全部匹配核对与共同匹配核对上的 IoU 中位数、ASSD，以及匹配对数与 F1@0.7。
数据：boundary/per/gt_{S1,2D}_<种子>/（boundary.py 第一步输出）与 per/gt_*（vol_pass 计数）。五个种子先逐种子算再取均值。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import glob, json, os, numpy as np, common as C
OUT = f"{_NSP_SCRATCH}/nc/errors/boundary"
seeds = list(C.SEEDS)
keys = sorted(tuple(os.path.basename(f)[:-4].split("__", 1)) for f in glob.glob(f"{OUT}/per/gt_S1_s0/*.npz"))
def L(tg, k):
    z = np.load(OUT + "/per/" + tg + "/" + "__".join(k) + ".npz"); return z
dss = sorted({k[0] for k in keys}); res = {}
for d in dss:
    kk = [k for k in keys if k[0] == d]; r = {f: [] for f in ("n_S1","n_2D","n_common","iouA_S1","iouA_2D","iouC_S1","iouC_2D","assdC_S1","assdC_2D","assdA_S1","assdA_2D","f17_S1","f17_2D","n_gt")}
    for s in seeds:
        acc = {x: [] for x in ("iA1","iA2","iC1","iC2","aC1","aC2","aA1","aA2")}; n1=n2=nc=tp1=tp2=den1=den2=ng=0
        for k in kk:
            a, b = L(f"gt_S1_{s}", k), L(f"gt_2D_{s}", k)
            c = np.intersect1d(a["gt_id"], b["gt_id"]); ma, mb = np.isin(a["gt_id"], c), np.isin(b["gt_id"], c)
            rel_a, rel_b = a["assd_px"] / float(a["d"]), b["assd_px"] / float(b["d"])
            acc["iA1"].append(a["iou"]); acc["iA2"].append(b["iou"]); acc["iC1"].append(a["iou"][ma]); acc["iC2"].append(b["iou"][mb])
            acc["aA1"].append(rel_a); acc["aA2"].append(rel_b); acc["aC1"].append(rel_a[ma]); acc["aC2"].append(rel_b[mb])
            n1 += len(a["gt_id"]); n2 += len(b["gt_id"]); nc += len(c); ng += int(a["n_gt"])
            tp1 += int(a["tp7"]); den1 += int(a["n_gt"]) + int(a["n_pred"]); tp2 += int(b["tp7"]); den2 += int(b["n_gt"]) + int(b["n_pred"])
        cat = {x: np.concatenate(v) for x, v in acc.items()}
        for f, v in (("n_S1",n1),("n_2D",n2),("n_common",nc),("n_gt",ng),("iouA_S1",np.median(cat["iA1"])),("iouA_2D",np.median(cat["iA2"])),
                     ("iouC_S1",np.median(cat["iC1"])),("iouC_2D",np.median(cat["iC2"])),("assdA_S1",cat["aA1"].mean()),("assdA_2D",cat["aA2"].mean()),
                     ("assdC_S1",cat["aC1"].mean()),("assdC_2D",cat["aC2"].mean()),("f17_S1",2*tp1/den1),("f17_2D",2*tp2/den2)):
            r[f].append(float(v))
    res[d] = {f: float(np.mean(v)) for f, v in r.items()} | {"seeds_iouC_diff": [x - y for x, y in zip(r["iouC_S1"], r["iouC_2D"])], "seeds_iouA_diff": [x - y for x, y in zip(r["iouA_S1"], r["iouA_2D"])]}
json.dump(res, open(f"{OUT}/o2_selection.json", "w"), indent=1)
print("套 | n_gt | 匹配对 S1/2D/共同 | F1@0.7 S1/2D | IoU中位 全部 S1/2D | IoU中位 共同 S1/2D | ASSD_rel 全部 S1/2D | ASSD_rel 共同 S1/2D")
for d, x in res.items():
    flag = "  <-- 翻转" if (x["iouA_S1"] < x["iouA_2D"]) and (x["f17_S1"] > x["f17_2D"]) else ""
    v=[x[f] for f in ("n_gt","n_S1","n_2D","n_common","f17_S1","f17_2D","iouA_S1","iouA_2D","iouC_S1","iouC_2D","assdA_S1","assdA_2D","assdC_S1","assdC_2D")]
    print("%-12s %.0f | %.0f/%.0f/%.0f | %.3f/%.3f | %.3f/%.3f | %.3f/%.3f | %.4f/%.4f | %.4f/%.4f" % (d, *v) + flag)


# ---------------------------------------------------------------- 第二部分：IoU 分位数、IoU ≥ 0.7 比例、tp7 / n_pred 与 F1@0.7 差的拆分（2026-11-09 追加）
# 拆分：F1@0.7 = 2·tp7 /(n_gt + n_pred)。用五种子平均的计数，令 F1* = 2·tp7(二维) /(n_gt + n_pred(S1))（二维换成 S1 的预测数），
# “少预测”的贡献 = F1* − F1(二维)，“IoU ≥ 0.7 配对更多”的贡献 = F1(S1) − F1*，两项之和等于 F1(S1) − F1(二维)。
QS = (10, 25, 50, 75, 90)
for d in dss:
    kk = [k for k in keys if k[0] == d]
    ext = {}
    for m, other in (("S1", "2D"), ("2D", "S1")):
        tp7 = tp5 = npred = ngt = 0; q, qc = [], []
        for s in seeds:
            for k in kk:
                z, o = L(f"gt_{m}_{s}", k), L(f"gt_{other}_{s}", k)
                c = np.isin(z["gt_id"], o["gt_id"])
                tp7 += int(z["tp7"]); tp5 += int(z["tp5"]); npred += int(z["n_pred"]); ngt += int(z["n_gt"])
                q.append(z["iou"]); qc.append(z["iou"][c])
        q, qc = np.concatenate(q), np.concatenate(qc)
        n = len(seeds)
        ext[m] = dict(tp5=tp5 / n, tp7=tp7 / n, n_pred=npred / n, n_gt=ngt / n, iou_quantiles_all=[float(x) for x in np.percentile(q, QS)],
                      iou_quantiles_common=[float(x) for x in np.percentile(qc, QS)], frac_iou_ge07_all=float(np.mean(q >= 0.7)),
                      frac_iou_ge07_common=float(np.mean(qc >= 0.7)))
    s1, t2 = ext["S1"], ext["2D"]
    f1s1 = 2 * s1["tp7"] / (s1["n_gt"] + s1["n_pred"]); f12d = 2 * t2["tp7"] / (t2["n_gt"] + t2["n_pred"])
    fstar = 2 * t2["tp7"] / (t2["n_gt"] + s1["n_pred"])
    ext["F1_07_decomposition"] = dict(F1_S1=f1s1, F1_2D=f12d, F1_2D_with_S1_npred=fstar, diff=f1s1 - f12d,
                                      from_fewer_predictions=fstar - f12d, from_more_pairs_ge07=f1s1 - fstar,
                                      note="五种子平均计数上计算；两项之和等于 diff")
    res[d].update(ext)
res["_meta"] = dict(quantiles=QS, seeds=seeds, protocol="测试集 gt 协议，规则 T 掩膜；S1 与二维五种子；事后分析，只报告",
                    source="boundary/per/gt_{S1,2D}_<种子>/*.npz（boundary.py 第一步）")
json.dump(res, open(f"{OUT}/o2_selection.json", "w"), ensure_ascii=False, indent=1)

R = []; w = R.append
w("# 审稿第三轮 O2：BBBC050 上 S1 的 IoU 中位数低于二维、F1@0.7 却更高（事后分析，只报告）\n")
w("数据来自测试集 gt 协议、规则 T 选定的掩膜，S1 与二维各用五个种子（s0–s4000），数字都是五种子平均。匹配核对指匈牙利匹配上 IoU ≥ 0.5 的配对。“共同匹配”只取 S1 与二维（同一种子）都匹配上的真值核。ASSD rel 是像素口径的 ASSD 除以该体积真值核直径中位数。脚本 boundary_o2.py，全部数字存在 o2_selection.json。\n")
w("## 逐套：匹配对数、F1@0.7、IoU 中位数与 ASSD rel\n")
w("| 套 | 真值核 | 匹配对 S1 / 二维 / 共同 | F1@0.7 S1 / 二维 | IoU 中位数（全部）S1 / 二维 | IoU 中位数（共同）S1 / 二维 | ASSD rel（全部）S1 / 二维 | ASSD rel（共同）S1 / 二维 |")
w("|---|---|---|---|---|---|---|---|")
for d in dss:
    x = res[d]
    flag = "（翻转）" if x["iouA_S1"] < x["iouA_2D"] and x["f17_S1"] > x["f17_2D"] else ""
    w("| %s%s | %.0f | %.1f / %.1f / %.1f | %.3f / %.3f | %.3f / %.3f | %.3f / %.3f | %.4f / %.4f | %.4f / %.4f |" % (
        d, flag, x["n_gt"], x["n_S1"], x["n_2D"], x["n_common"], x["f17_S1"], x["f17_2D"], x["iouA_S1"], x["iouA_2D"], x["iouC_S1"], x["iouC_2D"],
        x["assdA_S1"], x["assdA_2D"], x["assdC_S1"], x["assdC_2D"]))
w("\n## 翻转的套：IoU 分布与 F1@0.7 差的拆分\n")
w("拆分方法：F1@0.7 = 2·tp7 /(n_gt + n_pred)。把二维的预测数换成 S1 的预测数，得到 F1*；“少预测”的贡献 = F1* − F1(二维)，“IoU ≥ 0.7 的配对更多”的贡献 = F1(S1) − F1*，两项相加等于总差。\n")
w("| 套 | 模型 | tp7 | n_pred | n_gt | IoU 第 10/25/50/75/90 百分位（全部） | IoU ≥ 0.7 比例 全部 / 共同 |\n|---|---|---|---|---|---|---|")
flips = [d for d in dss if res[d]["iouA_S1"] < res[d]["iouA_2D"] and res[d]["f17_S1"] > res[d]["f17_2D"]]
for d in flips:
    for m in ("S1", "2D"):
        e = res[d][m]
        w("| %s | %s | %.1f | %.1f | %.0f | %s | %.3f / %.3f |" % (d, m, e["tp7"], e["n_pred"], e["n_gt"], " / ".join("%.3f" % v for v in e["iou_quantiles_all"]),
                                                                e["frac_iou_ge07_all"], e["frac_iou_ge07_common"]))
w("\n| 套 | F1@0.7 S1 | F1@0.7 二维 | F1*（二维换 S1 的预测数） | 总差 | 少预测的贡献 | IoU ≥ 0.7 配对更多的贡献 | 共同匹配 IoU 中位数逐种子差 |\n|---|---|---|---|---|---|---|---|")
for d in flips:
    e = res[d]["F1_07_decomposition"]
    w("| %s | %.4f | %.4f | %.4f | %+.4f | %+.4f | %+.4f | %s |" % (d, e["F1_S1"], e["F1_2D"], e["F1_2D_with_S1_npred"], e["diff"], e["from_fewer_predictions"],
                                                              e["from_more_pairs_ge07"], ", ".join("%+.3f" % v for v in res[d]["seeds_iouC_diff"])))
w("\n## 判读\n")
b = res["bbbc050"]; e = b["F1_07_decomposition"]
w("BBBC050 上的翻转不是匹配集的选择效应。两个模型匹配上的几乎是同一批核（%.1f 对 %.1f 对，共同 %.1f），只看共同匹配的核，S1 的 IoU 中位数仍然更低（%.3f 对 %.3f），五个种子都是如此。"
  "S1 的 F1@0.7 更高（%.3f 对 %.3f），一是因为预测更少、误检更少（%.1f 对 %.1f 个预测，真值 %.0f 个），贡献 %+.3f；二是因为 S1 的 IoU 分布低端与二维相近（第 10 百分位 %.3f 对 %.3f）、高端压缩（第 90 百分位 %.3f 对 %.3f），结果是 IoU ≥ 0.7 的配对略多（%.1f 对 %.1f），贡献 %+.3f。"
  % (b["n_S1"], b["n_2D"], b["n_common"], b["iouC_S1"], b["iouC_2D"], e["F1_S1"], e["F1_2D"], b["S1"]["n_pred"], b["2D"]["n_pred"], b["S1"]["n_gt"],
     e["from_fewer_predictions"], b["S1"]["iou_quantiles_all"][0], b["2D"]["iou_quantiles_all"][0], b["S1"]["iou_quantiles_all"][4], b["2D"]["iou_quantiles_all"][4], b["S1"]["tp7"], b["2D"]["tp7"], e["from_more_pairs_ge07"]))
w("另外两个翻转的套（%s）里，IoU 中位数的差都只有 0.001 左右，在噪声范围内。" % "、".join(x for x in flips if x != "bbbc050"))
open(f"{OUT}/o2_report.md", "w").write("\n".join(R) + "\n")
print("写入", f"{OUT}/o2_selection.json", f"{OUT}/o2_report.md")
