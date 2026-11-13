"""比较两次 eval_gt 的逐体积结果。用法：python t_ev_cmp.py <目录A> <目录B> [标签]
默认要求逐行相同（去掉耗时字段与 round 6 的 res 字段）；打印主格 cp-2/vol0.15 的逐体积 F1。"""
import json
import sys

A, B = sys.argv[1:3]
tag = sys.argv[3] if len(sys.argv) > 3 else ""


def rows(d):
    return sorted(json.load(open(f"{d}/rows.json")), key=lambda x: (x["ds"], x["id"], x["thr"], x["frac"]))


def strip(r):
    return {k: v for k, v in r.items() if k not in ("sec_net", "sec_total", "res")}


ra, rb = rows(A), rows(B)
same = [strip(x) for x in ra] == [strip(x) for x in rb]
print(f"T_ev {tag} {A.split('/')[-1]} vs {B.split('/')[-1]}：{len(ra)}/{len(rb)} 行，逐行相同（去掉耗时）= {same}")
for x, y in zip(ra, rb):
    if x["thr"] == -2.0 and x["frac"] == 0.15:
        print(f"    {x['ds']:9s} {x['id']:18s} F1@0.5 {x['f1']:.4f} / {y['f1']:.4f}   F1@0.7 {x['f1_7']:.4f} / {y['f1_7']:.4f}"
              + (f"   res {y.get('res')}" if "res" in y else ""))
