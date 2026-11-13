"""把 test_r6 / s5_stats 的逐套结果转成报告里的表格行（与 9.2 节同一套标签）。用法：python html_perds.py 结果.json F1@0.5|F1@0.7 [--n]"""
import json
import sys

NAME = {"mesospim": "mesoSPIM（域外）"}


def f3(x):
    """带符号三位小数，负号用 U+2212，与报告其余数字一致。"""
    return f"{x:+.3f}".replace("-", "−")


def verdict(e, which):
    d, ci, call = (e["d2d"], e.get("ci2d"), e["call2d"]) if which == "2d" else (e["dbase"], e.get("cibase"), e["callbase"])
    if ci is None:
        return f'<td class="n">{f3(d)}</td><td class="muted">单体积，见下文</td>'
    cell = f'<td class="n">{f3(d)} [{f3(ci[0])}, {f3(ci[1])}]</td>'
    if call == "胜":
        return cell + '<td class="up">胜</td>'
    if call == "不劣":
        return cell + "<td>不劣</td>"
    if ci[1] < 0:
        return cell + '<td class="bad">落后</td>'
    if d < 0:
        return cell + '<td class="warn">未判定（点估计落后）</td>'
    return cell + '<td class="warn">未判定（区间宽）</td>'


r = json.load(open(sys.argv[1], encoding="utf-8"))
P = r["per_ds"][sys.argv[2]]
for ds, e in P.items():
    row = f"<tr><td>{NAME.get(ds, ds)}</td>"
    if "--n" in sys.argv:
        row += f'<td class="n">{e["n"]}</td>'
    row += f'<td class="n">{e["S1"]:.3f}</td><td class="n">{e["two_d"]:.3f}</td><td class="n">{e["base"]:.3f}</td>'
    row += verdict(e, "2d") + verdict(e, "base") + "</tr>"
    print(row)
