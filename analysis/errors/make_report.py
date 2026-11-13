"""由 results.json 生成 report.md（数字部分全自动；判读文字在 interp.md，手写，重跑后需按新数字核对）。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import os

OUT = f"{_NSP_SCRATCH}/nc/errors"
R = json.load(open(f"{OUT}/results.json"))
M12 = "12 套（去 mesospim）"
L = []
w = L.append


def f(x, p=4, sign=False):
    if x is None:
        return "—"
    return f"{x:+.{p}f}" if sign else f"{x:.{p}f}"


def c1key(d, sub):
    return [k for k in d if "去 mesospim" in k][0] if sub != "all" else M12


w("# NucSeg3D 投稿分析：误差分解与 AI 分析（CPU 部分）\n")
w(f"生成时间 {R['generated']}。测试集只报告，不据此改任何设置。口径按 cpsam_finetune/投稿分析方案.html 第 4 节（A1、A2、S4）；"
  "C3、C4 在方案里写的是“口径与上一版相同”，上一版原文（主代理 11-05 提供）为——C3：“在同一批 45 个体积上比较二维估直径与真值直径，S1 和二维都算，并报告直径估计误差的分布”；C4：“每个体积事后选最优阈值作为上限，报告规则 T 与上限的差距，三个模型都算，只作参照”。下文的具体做法按此执行，在对应小节写明。复现核对注意：规则 T 改了阈值且 z/xy ≥ 2 的体积，存档只有原层位，最终掩膜由 dP、cp 沿 z 线性插值回各向同性网格后重新跟流，是近似（见第 0 节，宏平均差 ≤ 5.3e-4），未重跑网络。\n")
for proto in ("gt", "fp"):
    P = R[proto]
    w(f"- {'gt 协议' if proto == 'gt' else '完整流程'}：{P['n_vol']} 个体积；S1 与二维配对用的种子 {', '.join(P['seeds'])}"
      f"（已完成的种子：S1 {', '.join(P['seeds_complete']['S1'])}；二维 {', '.join(P['seeds_complete']['2D'])}）。"
      + ("**临时版**：五种子尚未齐。" if len(P["seeds"]) < 5 else ""))
w("")
if os.path.exists(f"{OUT}/interp.md"):
    w(open(f"{OUT}/interp.md").read())

# ---------------- 复现核对
w("\n## 0. 复现核对\n")
for proto in ("gt", "fp"):
    cv = R[proto]["consistency"]
    w(f"**{'gt 协议' if proto == 'gt' else '完整流程'}，逐体积**（我重算的计数对 rows.json 在规则 T 所选 (t, 0.15) 上的记录）：\n")
    w("| 运行 | 体积 | F1 计数相同 | 全部字段相同 | 重算掩膜（其中 z 插值） | F1 不同的体积 |\n|---|---|---|---|---|---|")
    for tg, v in sorted(cv["per_volume"].items()):
        w(f"| {tg} | {v['n']} | {v['f1_equal']} | {v['all_fields_equal']} | {v['recomputed']}（{v['recomputed_iso']}） | {', '.join(v['f1_diff_keys']) or '—'} |")
    w("")
    if cv["macro_vs_test_r6"]:
        w(f"宏平均对 test_r6_{proto}_basefixed.json（种子 0/1000/2000 逐种子）的最大绝对差：\n")
        w("| 指标 | S1 | 二维 | 原版 |\n|---|---|---|---|")
        for k, v in cv["macro_vs_test_r6"].items():
            w(f"| {k} | {v['S1']['max_abs_diff']:.2e} | {v['2D']['max_abs_diff']:.2e} | {v['base']['max_abs_diff']:.2e} |")
        w("")

# ---------------- A1
w("\n## 1. A1 误差分解与 z 向切碎\n")
w("规则与 eval_gt.taxonomy 相同（每个真值核归为 hit/miss/merge/split/boundary，每个未匹配预测归为 false/merge/split/boundary，"
  "1−F1@0.5 = (FN+FP)/(n_gt+n_pred) 分成五项）。“z 向切碎”：一个真值核被 ≥2 个预测核各覆盖 ≥15%（与 split 判据相同），"
  "且其中某两个预测核的 xy 投影 IoU > 0.5（两个三维上不相交的核 xy 投影大面积重叠，就是上下叠放）；下表的 z 向切碎只计未命中（最佳 IoU < 0.5）的真值核。"
  "“z 向合并”：真值核与另一个真值核被同一个预测核各覆盖 > 50%，且两者质心连线以 z 为主（|Δz| > xy 距离，µm）。"
  "所有比例先在每套里把体积合起来算，再对数据集求平均（宏平均），种子取均值；原版为固定 cp−2。\n")
for proto in ("gt", "fp"):
    for sub in ("all", "C1", "C2"):
        A = R[proto]["A1"][sub]; k = c1key(A, sub); e = A[k]
        w(f"**{'gt 协议' if proto == 'gt' else '完整流程'} · {sub}（{k}）**\n")
        w("| | S1 | 二维 | 原版 | S1−二维（种子 SD） | S1−原版 |\n|---|---|---|---|---|---|")
        w(f"| 1−F1@0.5 | {f(e['S1']['one_minus_f1'])} | {f(e['2D']['one_minus_f1'])} | {f(e['base']['one_minus_f1'])} | | |")
        for c in ("miss", "merge", "split", "boundary", "false"):
            d = e["diff"][c]
            w(f"| 分解·{c} | {f(e['S1']['decomp'][c])} | {f(e['2D']['decomp'][c])} | {f(e['base']['decomp'][c])} | {f(d['S1_minus_2D'], sign=True)}（{f(d['sd'])}） | {f(d['S1_minus_base'], sign=True)} |")
        for c, nm in (("zsplit", "z 向切碎 /真值核"), ("zsplit_any", "z 向切碎（含命中）/真值核"), ("split", "切碎 /真值核"), ("zmerge", "卷入 z 向合并 /真值核"),
                      ("xymerge", "卷入 xy 向合并 /真值核"), ("merge", "合并 /真值核"), ("miss", "漏检 /真值核"), ("boundary", "边界不准 /真值核")):
            d = e["diff"]["gt:" + c]
            w(f"| {nm} | {f(e['S1']['gt_rate'][c])} | {f(e['2D']['gt_rate'][c])} | {f(e['base']['gt_rate'][c])} | {f(d['S1_minus_2D'], sign=True)}（{f(d['sd'])}） | {f(d['S1_minus_base'], sign=True)} |")
        w(f"| z 向切碎占真值核错误的比例 | {f(e['S1']['zsplit_share_of_gt_errors'])} | {f(e['2D']['zsplit_share_of_gt_errors'])} | {f(e['base']['zsplit_share_of_gt_errors'])} | | |")
        w(f"| 合并对中 z 向的比例 | {f(e['S1']['zmerge_pair_share'])} | {f(e['2D']['zmerge_pair_share'])} | {f(e['base']['zmerge_pair_share'])} | | |")
        w("")
    if proto == "gt":
        e = R["gt"]["A1"]["all"][M12]
        w("逐套（gt 协议，全部体积；每 100 个真值核的 z 向切碎 / 卷入 z 向合并，S1 | 二维 | 原版）：\n")
        w("| 数据集 | z 向切碎 | z 向合并 | 1−F1 中 split 项 | 1−F1 中 merge 项 |\n|---|---|---|---|---|")
        for d in e["S1"]["per_ds"]:
            g = lambda nm, c: 100 * e[nm]["per_ds"][d][c]
            w(f"| {d} | {g('S1','zsplit'):.2f} / {g('2D','zsplit'):.2f} / {g('base','zsplit'):.2f} | {g('S1','zmerge'):.2f} / {g('2D','zmerge'):.2f} / {g('base','zmerge'):.2f} "
              f"| {g('S1','d_split'):.2f} / {g('2D','d_split'):.2f} / {g('base','d_split'):.2f} | {g('S1','d_merge'):.2f} / {g('2D','d_merge'):.2f} / {g('base','d_merge'):.2f} |")
        w("")

w("\n### 1.1 分层（gt 协议，去 mesospim 的 12 套，全部体积）\n")
w("分层变量：核体积用“本体积内真值核体积中位数的倍数”（各套核大小差几十倍，绝对体积会把数据集差别混进来）；深度用质心 z 除以 (层数−1)；"
  "局部拥挤度取到 10 个最近邻真值核质心的平均距离（µm）除以本体积真值核等效球直径中位数（越小越挤；核数 ≤ 10 的体积用其余全部核），"
  f"按 gt 协议全部真值核的五分位数分成 Q1（最挤）–Q5，边界 {', '.join(f'{x:.2f}' for x in R['gt']['A1_strata_crowd_edges'][1:-1])}；"
  "z/xy 为体积的体素比。表中数字为宏平均（每套在该层至少 20 个真值核才计入），单位 %；括号里是把所有核合在一起的值。"
  "注意：体积边缘的核近邻少，拥挤度偏大；深度分层受每套成像深度不同的影响，只能看同一模型内的趋势和模型间的差。\n")
S = R["gt"]["A1_strata"]["all"]
for var, nmv in (("relvol", "相对核体积"), ("depth", "相对深度"), ("crowd", "拥挤度五分位"), ("zxy", "z/xy")):
    ent = S[var]
    w(f"**{nmv}**（每层真值核数 {', '.join(map(str, ent['n']))}）\n")
    w("| 层 | 未命中 S1 / 二维 / 原版 | z 向切碎 S1 / 二维 / 原版 | 卷入 z 向合并 S1 / 二维 / 原版 |\n|---|---|---|---|")
    for b, lab in enumerate(ent["labels"]):
        cells = []
        for oc in ("err", "zsplit", "zmerge"):
            cells.append(" / ".join("—" if ent[nm][oc]["macro"][b] is None else f"{100 * ent[nm][oc]['macro'][b]:.1f}" for nm in ("S1", "2D", "base"))
                         + "（" + " / ".join(f"{100 * ent[nm][oc]['pooled'][b]:.1f}" for nm in ("S1", "2D", "base")) + "）")
        w(f"| {lab} | " + " | ".join(cells) + " |")
    w("")

# ---------------- S4
w("\n## 2. S4 F1–IoU 曲线（IoU 0.50–0.95，步长 0.05）\n")
w("匹配沿用评测脚本的匈牙利匹配（按重叠图的连通分量分别求解，最优值与稠密矩阵相同），一次匹配后数 IoU ≥ 阈值的配对；"
  "曲线下面积按梯形法除以 0.45（即 0.50–0.95 上的平均 F1）。gt 协议每套 pooled，完整流程逐体积平均，再对数据集宏平均。\n")
grid = [0.5 + 0.05 * i for i in range(10)]
for proto in ("gt", "fp"):
    for sub in ("all", "C1", "C2"):
        A = R[proto]["S4"][sub]; k = c1key(A, sub); e = A[k]
        w(f"**{'gt 协议' if proto == 'gt' else '完整流程'} · {sub}（{k}）**：面积 S1 {f(e['S1']['auc_norm'])}、二维 {f(e['2D']['auc_norm'])}、原版 {f(e['base']['auc_norm'])}；"
          f"S1−二维 {f(e['S1_minus_2D_auc']['mean'], sign=True)}（种子 SD {f(e['S1_minus_2D_auc']['sd'])}），S1−原版 {f(e['S1_minus_base_auc'], sign=True)}。\n")
        w("| IoU | " + " | ".join(f"{g:.2f}" for g in grid) + " |\n|---|" + "---|" * len(grid))
        for nm in ("S1", "2D", "base"):
            w(f"| {nm} | " + " | ".join(f"{v:.3f}" for v in e[nm]["curve"]) + " |")
        w("| S1−二维 | " + " | ".join(f"{v:+.3f}" for v in e["S1_minus_2D_curve"]) + " |\n")

# ---------------- C3
w("\n## 3. C3 免直径（二维估直径）的代价\n")
w("做法：完整流程的 45 个体积，同一模型、同一种子，比较完整流程（直径用二维 r2a2 估计的 d_used，规则 T）与 gt 协议（真值直径，规则 T）在同一体积上的 F1；"
  "代价 = 真值直径的 F1 − 估计直径的 F1，逐体积求差，按完整流程的口径（逐体积平均 → 每套 → 宏平均）汇总，种子取均值。"
  "直径误差用 d_used / d_gt（d_gt 为真值核 xy 等效直径中位数）。\n")
for nm in ("S1", "2D"):
    for sub in ("all", "C1", "C2"):
        A = R["C3"][nm][sub]; k = c1key({x: 1 for x in A if x != "diam_error"}, sub); e = A[k]; de = A["diam_error"]
        w(f"- {nm} · {sub}（{k}）：F1@0.5 真值直径 {f(e['F1_05_gtdiam'])}、估计直径 {f(e['F1_05_estdiam'])}，代价 {f(e['cost_05'], sign=True)}；"
          f"F1@0.7 {f(e['F1_07_gtdiam'])} 对 {f(e['F1_07_estdiam'])}，代价 {f(e['cost_07'], sign=True)}。"
          + (f"直径比中位数 {de['median_ratio']:.3f}，绝对误差中位数 {de['median_abs_pct']:.1f}%，±10% 内 {100 * de['within10']:.0f}%，±25% 内 {100 * de['within25']:.0f}%，"
             f"5–95% 分位 [{de['q05_q95_ratio'][0]:.2f}, {de['q05_q95_ratio'][1]:.2f}]；|log 误差| 与代价的斯皮尔曼 ρ = {de['spearman_abslog_vs_cost05'][0]:.2f}（p = {de['spearman_abslog_vs_cost05'][1]:.2g}）。"
             if sub == "all" else ""))
w("\n逐套直径比中位数（S1 所用的 d_used）：" + "，".join(f"{d} {v:.2f}" for d, v in R["C3"]["S1"]["all"]["diam_error"]["per_ds_median_ratio"].items()) + "。\n")

# ---------------- C4
w("\n## 4. C4 规则 T 与逐体积事后最优阈值\n")
w("做法：每个体积在阈值网格 {0, −1, −2, −3, −4, −6}（体积下限固定 0.15）里按该体积自己的 F1 取最好的阈值（oracle，F1@0.5 与 F1@0.7 各取一次）；"
  "另给阈值 × 体积下限 {0, 0.15, 0.25} 的 oracle 作上界。与规则 T、固定 cp−2 比，口径同主分析。原版这里同时给规则 T（主对照的原版是固定 cp−2）。\n")
for proto in ("gt", "fp"):
    for sub in ("all", "C1"):
        w(f"**{'gt 协议' if proto == 'gt' else '完整流程'} · {sub}**\n")
        w("| 模型 | 固定 cp−2 | 规则 T | oracle 阈值 | oracle 阈值×下限 | 差距 oracle−规则 T | 规则 T 选中 oracle 的体积比例 |\n|---|---|---|---|---|---|---|")
        for nm in ("S1", "2D", "base"):
            A = R[proto]["C4"][nm][sub]; k = c1key({x: 1 for x in A if isinstance(A[x], dict) and "rule_T F1@0.5" in A[x]}, sub); e = A[k]
            for ml in ("F1@0.5", "F1@0.7"):
                w(f"| {nm} {ml} | {f(e['fixed_cp-2 ' + ml])} | {f(e['rule_T ' + ml])} | {f(e['oracle_thr ' + ml])} | {f(e['oracle_thr_frac ' + ml])} | "
                  f"{f(e['gap_oracle_minus_ruleT ' + ml], sign=True)} | {100 * A['share_ruleT_equals_oracle_' + ml]:.0f}% |")
        w("")
    w(f"{'gt 协议' if proto == 'gt' else '完整流程'}全部体积上 oracle 阈值（F1@0.5）的分布：" + "；".join(
        f"{nm} " + "，".join(f"{t}:{n}" for t, n in R[proto]["C4"][nm]["all"]["oracle_thr_hist_F1@0.5"].items()) for nm in ("S1", "2D", "base"))
      + "。规则 T 选的阈值：" + "；".join(f"{nm} " + "，".join(f"{t}:{n}" for t, n in R[proto]["C4"][nm]["all"]["ruleT_thr_hist"].items()) for nm in ("S1", "2D", "base")) + "。\n")

# ---------------- A2
w("\n## 5. A2 种子一致度能否指出该复查的核\n")
w("做法：以种子 0 的最终预测核（规则 T 后）为基准，求它与其余每个种子最佳匹配的 IoU，取平均作一致度；“错核”= 与真值最佳 IoU < 0.5；"
  "用 1 − 一致度排序，报告 AUROC 与复查最不确定的 10%、20% 预测核时找回的错核比例（同分随机打破，种子 0）。去 mesospim，pooled 与按数据集宏平均都给。\n")
for proto in ("gt", "fp"):
    for nm in ("S1", "2D"):
        e = R[proto]["A2"].get(nm)
        if not e:
            continue
        w(f"**{'gt 协议' if proto == 'gt' else '完整流程'} · {nm}**（种子 {e['seeds'][0].replace('_', ', ')}{'；临时版' if e['provisional'] else ''}）\n")
        w("| 子集 | 预测核 | 错核比例 | AUROC pooled / 宏平均 | 前 10% 召回 pooled / 宏平均 | 前 20% 召回 pooled / 宏平均 |\n|---|---|---|---|---|---|")
        for sub in ("all", "C1", "C2"):
            x = e[sub]
            w(f"| {sub} | {x['n_pred']} | {x['err_rate']:.3f} | {f(x['auroc_pooled'], 3)} / {f(x['auroc_macro'], 3)} | {f(x['recall10_pooled'], 3)} / {f(x['recall10_macro'], 3)} | {f(x['recall20_pooled'], 3)} / {f(x['recall20_macro'], 3)} |")
        w("")

w("\n## 6. 文件\n")
w("脚本：common.py（规则 T、最终掩膜、稀疏版误差分类、z 向切碎、协变量）、vol_pass.py（逐运行逐体积，可续跑）、a2_pass.py、aggregate.py、figs.py、make_report.py、run_all.sh（一键重跑）。"
  "结果：results.json；逐体积 per/<协议_模型_种子>/<ds__id>.json；最终掩膜与逐核结果 cache/；真值核协变量 gtcov/；A2 逐核一致度 a2/。"
  "图：a1_decomposition.png、a1_strata.png、s4_f1_iou.png、c3_diameter.png、c4_oracle_gap.png、a2_review_curve.png。\n")
open(f"{OUT}/report.md", "w").write("\n".join(L))
print("写入", f"{OUT}/report.md")
