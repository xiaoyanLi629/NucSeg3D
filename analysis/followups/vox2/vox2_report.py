"""VOX2 结果整理（只做汇总与排版，不改预先登记的分析；读 vox2_analysis.py 的四个输出）。
每个数据集（BBBC050、NISNet3D zebrafish、NIS3D、Purdue）与宏平均（13 套、12 套、C1，以及这 4 套），F1@0.5 与 F1@0.7：
原体素 → 源体素的 S1、二维、原版分数，S1 − 二维 的五种子同种子配对差（均值 ± SD、配对自助区间、三档判定），确认规则（均值 > 2×SD 且 ≥ 0.01）。
输出 vox2_result.json、vox2_report.html。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import numpy as np
O = f"{_NSP_SCRATCH}/nc/gpu/vox2"
R = {f"{t}{s}": json.load(open(f"{O}/vox2_{t}_gt{s}.json")) for t in ("old", "new") for s in ("", "_C1")}
DS = [("bbbc050", "BBBC050"), ("nisnet_zf", "NISNet3D zebrafish"), ("nis3d", "NIS3D"), ("purdue", "Purdue")]
res = {"per_ds": {}, "macro": {}}
for lab in ("F1@0.5", "F1@0.7"):
    for ds, _ in DS:
        e = {}
        for t in ("old", "new"):
            x = R[t]["per_ds"][lab][ds]; sd = np.std(x["d2d_seeds"], ddof=1)
            e[t] = dict(S1=x["S1"], two_d=x["two_d"], base=x["base"], d=x["d2d"], sd=float(sd), seeds=x["d2d_seeds"], ci=x.get("ci2d"), call=x.get("call2d"),
                        confirmed=bool(x["d2d"] > 2 * sd and x["d2d"] >= 0.01))
        res["per_ds"].setdefault(lab, {})[ds] = e
    for t in ("old", "new"):
        m = R[t]["macro"][lab]
        for tag, v in m.items():
            res["macro"].setdefault(lab, {}).setdefault(tag, {})[t] = dict(S1=v["S1"], two_d=v["two_d"], base=v["base"], d=v["diff_mean"], sd=v["diff_sd"], confirmed=v["confirmed"], seeds=v["diff_seeds"])
        mc = R[t + "_C1"]["macro"][lab]
        for tag, v in mc.items():
            res["macro"][lab].setdefault("C1 " + tag, {})[t] = dict(S1=v["S1"], two_d=v["two_d"], base=v["base"], d=v["diff_mean"], sd=v["diff_sd"], confirmed=v["confirmed"], seeds=v["diff_seeds"])
        seeds = np.mean([R[t]["per_ds"][lab][ds]["d2d_seeds"] for ds, _ in DS], 0)
        mu, sd = float(seeds.mean()), float(seeds.std(ddof=1))
        res["macro"][lab].setdefault("这 4 套", {})[t] = dict(d=mu, sd=sd, confirmed=bool(mu > 2 * sd and mu >= 0.01), seeds=seeds.tolist(),
                                                           S1=float(np.mean([R[t]["per_ds"][lab][ds]["S1"] for ds, _ in DS])),
                                                           two_d=float(np.mean([R[t]["per_ds"][lab][ds]["two_d"] for ds, _ in DS])),
                                                           base=float(np.mean([R[t]["per_ds"][lab][ds]["base"] for ds, _ in DS])))
json.dump(res, open(f"{O}/vox2_result.json", "w"), ensure_ascii=False, indent=1)
f = lambda e: f"{e['d']:+.3f} ± {e['sd']:.3f}{' ✓' if e['confirmed'] else ''}"
for lab in ("F1@0.5", "F1@0.7"):
    for ds, nm in DS:
        e = res["per_ds"][lab][ds]
        print(f"{lab} {nm}: 旧 {f(e['old'])} {e['old']['ci']} {e['old']['call']} → 新 {f(e['new'])} {e['new']['ci']} {e['new']['call']}")
    for tag, e in res["macro"][lab].items():
        print(f"{lab} 宏平均 {tag}: S1 {e['old']['S1']:.4f}→{e['new']['S1']:.4f}，二维 {e['old']['two_d']:.4f}→{e['new']['two_d']:.4f}；S1−二维 旧 {f(e['old'])} → 新 {f(e['new'])}")
CSS = """:root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--bd:#ddd;--hd:#f4f4f6}@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#e8e8ea;--mut:#a0a0a6;--bd:#3a3a3e;--hd:#222226}}:root[data-theme="dark"]{--bg:#161618;--fg:#e8e8ea;--mut:#a0a0a6;--bd:#3a3a3e;--hd:#222226}
body{background:var(--bg);color:var(--fg);font-family:ui-sans-serif,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;max-width:980px;margin:2em auto;padding:0 1em;line-height:1.7}
.tw{overflow-x:auto}table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:.9em}td,th{border:1px solid var(--bd);padding:.3em .55em;text-align:right}th{background:var(--hd)}td:first-child,th:first-child{text-align:left}p.m{color:var(--mut)}"""
H = ["<!doctype html><html lang='zh'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>VOX2 源体素敏感性</title><style>" + CSS + "</style><body>",
     "<h1>VOX2：源体素敏感性分析（第三轮预审 MF2）</h1><p class='m'>投稿前事后补加，只推理、只报告；预先登记于 DESIGN_VOX2.md（md5 7d33fc4f…）。gt 协议测试集，S1 与二维五种子同种子配对，规则 T；原版固定 cp−2。“原体素”列复现正式五种子表（r6/s5final），“源体素”列把 BBBC050、NISNet3D zebrafish 的全部 60 个体积与 NIS3D 4 个、Purdue 3 个体积换成源体素结果。差值为五种子 S1 − 二维的均值 ± SD，✓ 表示通过确认（均值 > 2×SD 且 ≥ 0.01）；方括号为五种子共用体积索引的配对自助 95% 区间。</p>"]
for lab in ("F1@0.5", "F1@0.7"):
    H.append(f"<h2>{lab}：逐套</h2><div class='tw'><table><tr><th>数据集</th><th>S1 旧→新</th><th>二维 旧→新</th><th>原版 旧→新</th><th>S1−二维 原体素</th><th>S1−二维 源体素</th></tr>")
    for ds, nm in DS:
        e = res["per_ds"][lab][ds]; o, n = e["old"], e["new"]
        ci = lambda x: f" [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] {x['call']}" if x["ci"] else f" {x['call']}"
        H.append(f"<tr><td>{nm}</td><td>{o['S1']:.3f}→{n['S1']:.3f}</td><td>{o['two_d']:.3f}→{n['two_d']:.3f}</td><td>{o['base']:.3f}→{n['base']:.3f}</td><td>{f(o)}{ci(o)}</td><td>{f(n)}{ci(n)}</td></tr>")
    H.append(f"</table></div><h2>{lab}：宏平均</h2><div class='tw'><table><tr><th>口径</th><th>S1 旧→新</th><th>二维 旧→新</th><th>S1−二维 原体素</th><th>S1−二维 源体素</th></tr>")
    for tag, e in res["macro"][lab].items():
        H.append(f"<tr><td>{tag}</td><td>{e['old']['S1']:.4f}→{e['new']['S1']:.4f}</td><td>{e['old']['two_d']:.4f}→{e['new']['two_d']:.4f}</td><td>{f(e['old'])}</td><td>{f(e['new'])}</td></tr>")
    H.append("</table></div>")
H.append("<p class='m'>分析脚本 vox2_analysis.py（md5 c350987a…，预先登记，未改动）；整理脚本 vox2_report.py；数据 vox2_result.json 与 vox2_{old,new}_gt[_C1].json。完整流程表未重跑，仍为估计体素。</p></body></html>")
open(f"{O}/vox2_report.html", "w").write("\n".join(H))
