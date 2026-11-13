"""汇总 CPUX：results.json、per_volume.csv、report.html（样式取自 $NUCSEG3D_SCRATCH/nc/gpu/vox2/vox2_report.html）。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import csv
import hashlib
import html
import json
import sys
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/cpu_extra/scripts'))
from cpux_common import INP, OUT  # noqa: E402

A = json.load(open(f"{OUT}/a_results.json")); Bn = json.load(open(f"{OUT}/b_results.json")); REP = json.load(open(f"{OUT}/a_repro.json"))
APV = json.load(open(f"{OUT}/a_per_volume.json")); BPV = json.load(open(f"{OUT}/b_per_volume.json")); MIS = json.load(open(f"{OUT}/b_mismatch.json"))
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
res = {"design": f"{_NSP_SCRATCH}/nc/cpu_extra/DESIGN_CPUX.md", "design_md5": "27bd232653ae0c1892e7622c7b568d4b", "registered": "AGENTS.md b8932d9",
       "post_hoc": True, "test_report_only": True, "inputs_manifest_md5": md5(f"{INP}/MANIFEST_md5.json"),
       "repro_S7": REP, "analysis_A": A, "analysis_B": Bn, "b_tp_mismatch_detail": MIS}
json.dump(res, open(f"{OUT}/results.json", "w"), ensure_ascii=False, indent=1)
keys = sorted(set(APV) | set(BPV))
cols = sorted({c for k in keys for c in list(APV.get(k, {})) + list(BPV.get(k, {}))})
with open(f"{OUT}/per_volume.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["ds", "id"] + cols)
    for k in keys:
        ds, vid = k.split("__", 1) if "__" in k else k.split("||", 1)
        e = {**APV.get(f"{ds}||{vid}", {}), **BPV.get(f"{ds}__{vid}", {})}
        w.writerow([ds, vid] + [("" if e.get(c) is None else e.get(c)) for c in cols])

STYLE = """:root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--bd:#ddd;--hd:#f4f4f6}@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#e8e8ea;--mut:#a0a0a6;--bd:#3a3a3e;--hd:#222226}}:root[data-theme="dark"]{--bg:#161618;--fg:#e8e8ea;--mut:#a0a0a6;--bd:#3a3a3e;--hd:#222226}
body{background:var(--bg);color:var(--fg);font-family:ui-sans-serif,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;max-width:980px;margin:2em auto;padding:0 1em;line-height:1.7}
.tw{overflow-x:auto}table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:.9em}td,th{border:1px solid var(--bd);padding:.3em .55em;text-align:right}th{background:var(--hd)}td:first-child,th:first-child{text-align:left}p.m{color:var(--mut)}"""
f4 = lambda x: f"{x:+.4f}"
ci = lambda c: f"[{c[0]:+.4f}, {c[1]:+.4f}]"
QN = {"n": "核数", "vol_med": "核体积中位数", "nn_med": "最近邻距离中位数"}


def tbl(head, rows):
    return ("<div class='tw'><table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table></div>")


H = [f"<!doctype html><html lang='zh'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>CPUX 形态学与深度信噪比</title><style>{STYLE}</style><body>",
     "<h1>CPUX：形态学偏差检验与误差随深度、信噪比的变化</h1>",
     f"<p class='m'>设计 {_NSP_SCRATCH}/nc/cpu_extra/DESIGN_CPUX.md（md5 27bd232653ae0c1892e7622c7b568d4b），登记于 AGENTS.md 提交 b8932d9。两项都是事后（post hoc）补充分析，测试集只报告，不据此改任何设置。测试集 gt 协议 227 个体积，S1 与二维对照各五个种子（同种子配对），在 hostC 上用 CPU 计算（≤ 40 个进程），只读已存的预测。</p>",
     "<nav><ol><li><a href='#s0'>核对</a></li><li><a href='#s1'>分析 A：形态学偏差检验</a></li><li><a href='#s2'>分析 B：按深度</a></li><li><a href='#s3'>分析 B：按局部信噪比</a></li><li><a href='#s4'>分析 B：深度 × 信噪比</a></li><li><a href='#s5'>局限</a></li></ol></nav>"]
H.append("<h2 id='s0'>0. 核对</h2>")
v = REP["values"]
H.append(f"<p>第一步用读数缓存重现论文 Morphometric readouts 的合并偏差（对均值的百分比差）：核数 S1 {v['S1|n']['bias_pct']:+.2f}%、二维 {v['2D|n']['bias_pct']:+.2f}%、原版 {v['base|n']['bias_pct']:+.2f}%；核体积中位数 {v['S1|vol_med']['bias_pct']:+.2f}%、{v['2D|vol_med']['bias_pct']:+.2f}%、{v['base|vol_med']['bias_pct']:+.2f}%；最近邻距离中位数 {v['S1|nn_med']['bias_pct']:+.2f}%、{v['2D|nn_med']['bias_pct']:+.2f}%、{v['base|nn_med']['bias_pct']:+.2f}%。四舍五入到一位小数后与正文九个数字全部一致，所以分析 A 与正文同源。</p>")
c = Bn["check"]
H.append(f"<p>分析 B 用误差分析缓存里的规则 T 最终掩膜与逐真值核最佳 IoU。由它推出的 tp5、tp7 与评测记录逐体积逐种子比对（227 × 10 × 2 项），有 {c['tp_mismatch']} 项不同，涉及 {len(MIS)} 个（体积 × 模型种子），合计 |Δtp5| = {sum(abs(x['d_tp5']) for x in MIS)} 个核、|Δtp7| = {sum(abs(x['d_tp7']) for x in MIS)} 个核（全部约 2.7 万个真值核 × 10）。这些全部是规则 T 改了阈值、需要在 CPU 上由存下的流场重算掩膜的体积（其中除 mesospim 一例外都是 z/xy ≥ 2 的 z 插值近似），与误差分析报告第 0 节已列出的情况相同。匹配对上重算的 IoU 与缓存的 g_best 最大差为 {c['iou_maxdiff']:.1e}。</p>")
H.append("<h2 id='s1'>1. 分析 A：形态学偏差检验</h2>")
H.append("<p>每个体积每个模型种子取相对误差 e = (预测 − 真值)/真值。主统计量是“|偏差|”：每套 e 的均值取绝对值，再对数据集宏平均；次统计量是每套 mean|e| 的宏平均。表中的差为 S1 − 二维，负值表示 S1 的偏差更小。判定沿用论文规则：五个种子差的均值绝对值大于 2 倍 SD（种子一致），并且套内分层体积自助 10000 次的 95% 区间不含 0。相对误差没有预先约定最小效应量，不设 0.01 线。<b>三项读数 × 12/13 套共六个判定，不做多重比较校正，结果不论通过与否全部列出。</b>原版只作描述。</p>")
rows = []
for q in ("n", "vol_med", "nn_med"):
    for tag in ("12", "13"):
        for stat, sl in (("abs_bias", "|偏差|（主）"), ("mean_abs_err", "mean|e|（次）")):
            e = A["readouts"][q]["sets"][tag][stat]
            rows.append([QN[q], tag, sl, f"{e['S1']:.4f}", f"{e['two_d']:.4f}", f"{e['base']:.4f}", f"{f4(e['diff_mean'])} ± {e['diff_sd']:.4f}", ci(e['boot_ci']),
                         "是" if e["seed_consistent"] else "否", "是" if e["ci_excludes_0"] else "否", "<b>确认</b>" if e["confirmed"] else "未确认"])
H.append(tbl(["读数", "套", "统计量", "S1", "二维", "原版", "S1 − 二维（种子均值 ± SD）", "自助 95% 区间", "种子一致", "区间不含 0", "判定"], rows))
ex = {q: A["readouts"][q]["excluded_by_ds"] for q in A["readouts"]}
H.append(f"<p class='m'>成对完整排除的体积：{html.escape(json.dumps(ex, ensure_ascii=False))}。</p>")
a = lambda q, t, s="abs_bias": A["readouts"][q]["sets"][t][s]
H.append(f"<p>按预先写定的规则，六个主判定中有两个通过：12 套口径下，S1 的核数 |偏差| 比二维小 {abs(a('n','12')['diff_mean']):.3f}（{a('n','12')['S1']:.3f} 对 {a('n','12')['two_d']:.3f}），核体积中位数 |偏差| 小 {abs(a('vol_med','12')['diff_mean']):.3f}（{a('vol_med','12')['S1']:.3f} 对 {a('vol_med','12')['two_d']:.3f}），两者都种子一致且区间不含 0。最近邻距离中位数的偏差本来就小（S1 {a('nn_med','12')['S1']:.3f}、二维 {a('nn_med','12')['two_d']:.3f}），差别没有通过。13 套口径（加入只有 1 个体积的 mesospim）下三项都没有通过：核数的区间上界刚到 0（{ci(a('n','13')['boot_ci'])}），核体积的种子差不一致。次统计量（逐体积 mean|e|）的结论与主统计量相同。由于六个判定没有校正多重比较，正文若引用，宜写成“12 套上核数与核体积的偏差差异通过了论文的确认规则（未校正多重比较），13 套上未通过”。</p>")
DL = ("浅", "中", "深"); SL = ("低", "中", "高")


def btab(sysn, labs, tag):
    rows = []
    for lab in labs:
        e = Bn["strata"].get(f"{sysn}|{lab}|{tag}")
        if not e or "hit5" not in e:
            rows.append([lab, e["n_nuclei"] if e else 0, 0] + ["—"] * 8); continue
        r = [lab, e["n_nuclei"], len(e["datasets_included"])]
        for met in ("hit5", "hit7", "biou", "biou_common"):
            m = e[met]
            r += [f"{m['S1']:.3f} / {m['two_d']:.3f}", f"{f4(m['diff_mean'])} {ci(m['boot_ci'])}"]
        rows.append(r)
    return tbl(["层", "真值核数", "纳入套数", "命中 IoU≥0.5（S1 / 二维）", "差 [95% 区间]", "命中 IoU≥0.7", "差 [95% 区间]", "边界 IoU（全部匹配）", "差 [95% 区间]", "边界 IoU（两者都匹配）", "差 [95% 区间]"], rows)


g = lambda s, l, m, t="12": Bn["strata"][f"{s}|{l}|{t}"][m]
H.append("<h2 id='s2'>2. 分析 B：按深度（只描述）</h2>")
H.append(f"<p>每个真值核按质心的相对深度 z/(层数−1) 等宽分三档。命中率为 IoU ≥ 0.5、≥ 0.7 的真值核比例，边界 IoU 按 Cheng 等（2021），带宽 d = max(1, 0.1 × 真值核 xy 等效直径中位数)，xy 按 1 像素、z 按 z/xy 换算。每层每套把核合起来算，只纳入该层至少 20 个真值核的套，再宏平均；S1、二维各取五个种子宏平均的均值；区间为套内分层体积自助 10000 次。只描述，不做确认判定。</p>")
H.append("<h3>12 套（主）</h3>" + btab("depth", DL, "12") + "<h3>13 套</h3>" + btab("depth", DL, "13"))
H.append(f"<p>三档深度上 IoU ≥ 0.5 的命中率两者几乎相同（差 {f4(g('depth','浅','hit5')['diff_mean'])}、{f4(g('depth','中','hit5')['diff_mean'])}、{f4(g('depth','深','hit5')['diff_mean'])}，区间都含 0），两个模型的命中率都随深度下降（S1 从 {g('depth','浅','hit5')['S1']:.3f} 降到 {g('depth','深','hit5')['S1']:.3f}）。S1 的优势体现在形状上：IoU ≥ 0.7 的命中率差从浅层的 {f4(g('depth','浅','hit7')['diff_mean'])} 增大到深层的 {f4(g('depth','深','hit7')['diff_mean'])}，边界 IoU 的差从 {f4(g('depth','浅','biou')['diff_mean'])} 增大到 {f4(g('depth','深','biou')['diff_mean'])}，方向一致，但各档的区间互相重叠，这只是描述性的趋势。层序是文件里的 z 顺序，各套不一定都是“离物镜越远越深”。</p>")
H.append("<h2 id='s3'>3. 分析 B：按局部信噪比（只描述）</h2>")
bins = Bn["bins"]
H.append(f"<p>局部信噪比 =（核内强度中位数 − 周围壳层强度中位数）/ 壳层强度 SD，壳层取核外 s = max(2, 0.25 × D) 以内、去掉其他真值核和未定义区域的体素。三档边界在看测试预测之前只用验证集 {bins['n_val_vol']} 个体积的真值核确定（bins.json md5 {Bn['bins_md5']}），主口径每套用自己的三分位数（13 套验证核都 ≥ 30 个，没有用到合并回退），敏感性口径用全部验证核合并的三分位数（{bins['snr_pooled'][0]:.2f}、{bins['snr_pooled'][1]:.2f}）。测试集信噪比无定义的核：{html.escape(json.dumps(Bn['snr_undefined_test'], ensure_ascii=False))}。</p>")
H.append("<h3>每套三分位（主，12 套）</h3>" + btab("snr_own", SL, "12") + "<h3>每套三分位（13 套）</h3>" + btab("snr_own", SL, "13")
         + "<h3>合并三分位（敏感性，12 套）</h3>" + btab("snr_pooled", SL, "12"))
H.append(f"<p>命中率随信噪比明显上升（S1 IoU ≥ 0.5：低 {g('snr_own','低','hit5')['S1']:.3f}、中 {g('snr_own','中','hit5')['S1']:.3f}、高 {g('snr_own','高','hit5')['S1']:.3f}），两模型在 IoU ≥ 0.5 上没有可分辨的差别。IoU ≥ 0.7 的差在低信噪比档最大（{f4(g('snr_own','低','hit7')['diff_mean'])}，高档 {f4(g('snr_own','高','hit7')['diff_mean'])}），边界 IoU 的差在三档都在 +0.016 到 +0.020 之间，没有明显趋势。合并三分位的结果方向相同。</p>")
H.append("<h2 id='s4'>4. 分析 B：深度 × 信噪比（每套三分位，只描述）</h2>")
H.append("<h3>12 套</h3>" + btab("depth_x_snr", [f"{d}×{s}" for d in DL for s in SL], "12") + "<h3>13 套</h3>" + btab("depth_x_snr", [f"{d}×{s}" for d in DL for s in SL], "13"))
H.append("<p class='m'>九层里每层的核数与纳入的套数都更少，区间更宽，只作参考。</p>")
H.append("<h2 id='s5'>5. 局限</h2>")
H.append("<p>两项都是事后分析，测试集只报告。分析 A 的六个判定没有校正多重比较；13 套口径受只有一个体积的 mesospim 影响很大。分析 B 的掩膜里，规则 T 改了阈值且 z/xy ≥ 2 的体积是 z 插值近似，逐核计数与评测记录有少量差异（见第 0 节）；深度用的是文件里的层序，未按成像方向统一；局部信噪比由真值标注界定核与壳层，标注边界的误差会混进信噪比；边界 IoU 只在 IoU ≥ 0.5 的匹配对上算，匹配多的模型会纳入更难的核，所以另给了“两者都匹配”的口径，两种口径结论一致。</p>")
H.append("</body></html>")
open(f"{OUT}/report.html", "w", encoding="utf-8").write("\n".join(H))
print("写入 results.json、per_volume.csv、report.html")
