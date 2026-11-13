"""nc G3 选参（CPU）：按 G2 同一规则：体积下限 0.15，13 套（含 mesospim）宏平均 pooled F1@0.5 最高，并列取 F1@0.7 高者，再并列取编号小者。
第一轮（--stage 1）：只看 v0–v5，写 stage1 = 选中编号 + 6（下一步要跑的加流一致性过滤的那一组）。第二轮（--stage 2）：把第一轮目录与第二轮目录合起来，在全部至多 8 组里选 variant。
用法：python useg_select.py --stage 1 --out sel1.json base=<目录> d2=<目录>
      python useg_select.py --stage 2 --out selected.json base=<目录1>,<目录2> d2=<目录1>,<目录2>"""
import json, sys
import numpy as np
from collections import defaultdict
a = sys.argv[1:]; out = a[a.index("--out") + 1]; stage = int(a[a.index("--stage") + 1]); specs = [x for x in a if "=" in x]
res = {}
for s in specs:
    k, d = s.split("=", 1)
    rows = {}
    for dd in d.split(","):
        for x in json.load(open(dd + "/rows.json")):
            if x["frac"] == 0.15:
                rows[(x["ds"], x["id"], x["variant"])] = x
    rows = [x for x in rows.values() if stage == 2 or x["variant"] < 6]
    assert "--smoke" in a or len({(x["ds"], x["id"]) for x in rows}) == 88, d
    g = {}
    for v in sorted({x["variant"] for x in rows}):
        per = defaultdict(lambda: [0.0, 0.0, 0.0])
        for x in rows:
            if x["variant"] == v:
                p = per[x["ds"]]; p[0] += x["tp5"]; p[1] += x["tp7"]; p[2] += x["n_gt"] + x["n_pred"]
        g[v] = (float(np.mean([2 * p[0] / max(p[2], 1) for p in per.values()])), float(np.mean([2 * p[1] / max(p[2], 1) for p in per.values()])))
    best = sorted(g, key=lambda v: (-round(g[v][0], 6), -round(g[v][1], 6), v))[0]
    res[k] = dict(variant=best, stage1=best + 6 if best < 6 else best, f1_05=g[best][0], f1_07=g[best][1], grid={str(v): g[v] for v in g}, run=d)
    print(k, "选 v%d" % best, g[best], {v: round(g[v][0], 4) for v in g})
json.dump(res, open(out, "w"), indent=1)
