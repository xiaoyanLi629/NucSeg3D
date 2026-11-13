"""把 nc_stats_results.json 写成给主代理看的 Markdown 摘要。"""
MODE = {"gt": "gt 协议", "fp": "完整流程"}
SUB = {"all": "全部", "C1": "C1", "C2": "C2"}
CN = {"vs2D": "对二维", "vsBase": "对原版"}


def f(x, n=4):
    return "—" if x is None else f"{x:+.{n}f}"


def ci(c, n=4):
    return "—" if not c else f"[{c[0]:+.{n}f}, {c[1]:+.{n}f}]"


def write(res, path):
    L = [f"# NC 统计分析 S1–S3、S5–S7（{res['tag']}，种子 {'、'.join(res['seeds'])}）", "",
         "口径按 cpsam_finetune/投稿分析方案.html 第 4.1 节（预先登记）。测试集只报告。差值一律为“我们的模型（S1 + 规则 T）减对照”，"
         "体积级 F1 先按种子平均再统计。全部数字在 nc_stats_results.json，森林图在 fig/。", ""]
    c = res["consistency"]
    L += ["## 一致性核对", "",
          c.get("note") or (f"用本代码的读数复算 s5_stats.py / test_r6.py 已报告的逐种子宏平均与逐套点估计（gt、fp × 全部、C1、C2 × 两个阈值，"
                            f"共 {c['n_items']} 项），最大绝对差为 {c['worst_abs_diff']:.2g}，"
                            f"{'逐项完全一致' if c['identical'] else '存在差异，需要排查'}。"
                            + (f"核对对象为用同样 {len(res['seeds'])} 个种子独立运行的 s5_stats.py --base_fixed 输出（ref/）。" if len(res['seeds']) != 3 else "")
                            + (f"另外只取种子 0/1000/2000 时，与 r6 已报告的三种子数字核对 {c['three_seed_subset']['n_items']} 项，最大绝对差 {c['three_seed_subset']['worst_abs_diff']:.2g}。" if 'three_seed_subset' in c else "")), ""]
    mm = res.get("s7_mask_check")
    L += ["## 执行口径说明", "",
          "S1 的区间是 MixedLM 的 Wald 95% 区间；簇自助按数据集整簇有放回重抽并重拟合 REML 截距。S2 中 n < 2 的套只画在森林图上（空心点），"
          "森林图上每套的横线为 ±1.96 SE。S3 只对 F1@0.5 做（界值只登记了 F1@0.5），三档定义为：单侧 p < 0.05 为“非劣效成立”，"
          "配对 t 的 90% 区间上界 < −0.02 为“非劣效不成立”，其余（含单体积套）为“证据不足”；自助 90% 区间同时报告。"
          "S5 的族为同一协议、同一子集、同一对照、同一指标下的全部 n ≥ 2 的数据集。S6 中“核密度”方案没有给出定义，这里取真值前景体素占比；"
          "核直径用 xy 像素（部分数据集体素未标定）；信噪比偏离预先登记的定义：原定分母为核外强度 MAD，但 celegans、cellos、nis3d、nisnet_zf、parhyale 的部分或全部体积核外 MAD = 0，无法计算；按 AGENTS.md 提交 a099d3a 改为（核内中位数 − 核外中位数）/ 核外强度标准差，分子不变（偏离）。核密度 = 真值前景体素占比、核直径用 xy 像素，均为 a099d3a 登记的口径澄清。"
          "S7 的核体积与最近邻距离按清单体素尺寸换算，各数据集单位不统一，所以主口径用百分比差，核数另给原始差。"]
    if mm:
        L += ["", f"S7 用的最终掩膜共 {mm['n_masks']} 个，其中 {mm['n_recomputed']} 个规则 T 改了阈值、需在 CPU 上由存下的流场重算；"
                  f"重算后的核数与评测记录 n_pred 不一致的有 {mm['n_mismatch']} 个（都在重算的掩膜里，差几个核，来自 CPU 与评测时 GPU 跟流的数值差异，"
                  "test_r6.hits() 的 McNemar 也用同样的重算）。固定阈值的掩膜全部一致。"]
    L.append("")
    for mode in ("gt", "fp"):
        for sub in ("all", "C1", "C2"):
            R = res["results"][mode][sub]
            L += [f"## {MODE[mode]} · {SUB[sub]}（{R['n_vol']} 个体积，{R['n_ds']} 套）", ""]
            L += ["| 指标 | 比较 | S1 混合模型 β [95% CI] | 簇自助 95% CI | 去 mesospim β [CI] | S2 合并 μ [HK 95% CI] | I² | k | S5 Holm 显著（正/负）/检验套数 |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for lab in ("F1@0.5", "F1@0.7"):
                for cn in ("vs2D", "vsBase"):
                    e = R[lab][cn]; m = e["S1"]["main"]; nm = e["S1"]["no_mesospim"]
                    nm_txt = "同主分析" if isinstance(nm, str) else f"{f(nm['beta'])} {ci(nm['ci'])}"
                    s5 = e["S5"]
                    tested = [d for d, v in s5.items() if v.get("p") is not None]
                    pos = [d for d in tested if v_sig(s5[d]) and s5[d].get("mean", 0) > 0]
                    neg = [d for d in tested if v_sig(s5[d]) and s5[d].get("mean", 0) < 0]
                    L.append(f"| {lab} | {CN[cn]} | {f(m['beta'])} {ci(m['ci'])} | {ci(e['S1']['cluster_boot']['ci'])} | {nm_txt} | "
                             f"{f(e['S2']['mu'])} {ci(e['S2']['ci_hk'])} | {100 * e['S2']['I2']:.0f}% | {e['S2']['k']} | {len(pos)}/{len(neg)}/{len(tested)} |")
            L.append("")
            # S5 列出负向显著与正向不显著
            for lab in ("F1@0.5", "F1@0.7"):
                for cn in ("vs2D", "vsBase"):
                    s5 = R[lab][cn]["S5"]
                    neg = [f"{d}（p_Holm={v['p_holm']:.3g}）" for d, v in s5.items() if v_sig(v) and v.get("mean", 0) < 0]
                    ns = [d for d, v in s5.items() if v.get("p") is not None and not v_sig(v)]
                    L.append(f"- S5 {lab} {CN[cn]}：Holm 后不显著的套为 {('、'.join(ns)) or '无'}；显著变差的套为 {('、'.join(neg)) or '无'}。")
            # S3
            for cn in ("vs2D", "vsBase"):
                s3 = R["F1@0.5"][cn]["S3"]
                if not s3:
                    L.append(f"- S3 F1@0.5 {CN[cn]}：没有点估计为负的数据集。")
                else:
                    items = []
                    for d, v in s3.items():
                        if "p_one_sided" in v:
                            items.append(f"{d}（n={v['n']}，差 {v['effect']:+.3f}，单侧 p={v['p_one_sided']:.3g}，自助 90% CI {ci(v['boot_ci90'], 3)}，{v['verdict']}）")
                        else:
                            items.append(f"{d}（n=1，差 {v['effect']:+.3f}，{v['verdict']}）")
                    L.append(f"- S3 F1@0.5 {CN[cn]}：" + "；".join(items) + "。")
            # S6
            for lab in ("F1@0.5", "F1@0.7"):
                for cn in ("vs2D", "vsBase"):
                    s6 = R[lab][cn]["S6"]
                    if "note" in s6:
                        L.append(f"- S6 {lab} {CN[cn]}：{s6['note']}。"); continue
                    parts = [(f"{c} 无法计算（{s6[c]['note']}）" if s6[c].get("not_executable") else
                              f"{c}{'（偏离：分母为 SD）' if c == 'snr' else ''} 斜率 {s6[c]['slope']:+.3g} {ci(s6[c]['ci'], 3)}，p={s6[c]['p']:.2g}") for c in ("zxy", "density", "diam_px", "snr")]
                    L.append(f"- S6（探索性，单协变量元回归，斜率 [KH 95% CI]）{lab} {CN[cn]}，k={s6['zxy']['k']}：" + "；".join(parts) + "。")
            # S7
            s7 = R["S7"]
            rows = []
            for nm, lab in (("S1", "S1"), ("2D", "二维"), ("base", "原版")):
                cells = []
                for q in ("n", "vol_med", "nn_med"):
                    o = s7[nm][q]
                    cells.append("—" if "note" in o else f"{o['pct']['bias']:+.1f}% [{o['pct']['loa'][0]:+.1f}, {o['pct']['loa'][1]:+.1f}]")
                cn_raw = s7[nm]["n"]["raw"]
                rows.append(f"| {lab} | {cells[0]} | {cn_raw['bias']:+.2f} [{cn_raw['loa'][0]:+.1f}, {cn_raw['loa'][1]:+.1f}] | {cells[1]} | {cells[2]} |")
            L += ["", "S7 Bland–Altman（百分比差 = 100·(预测 − 真值)/均值；偏差 [95% 一致性界限]）：", "",
                  "| 模型 | 核数（%） | 核数（个） | 核体积中位数（%） | 最近邻距离中位数（%） |", "|---|---|---|---|---|"] + rows + [""]
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def v_sig(v):
    return bool(v.get("sig_holm_005"))
