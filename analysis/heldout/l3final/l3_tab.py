import json, numpy as np
O = json.load(open("l3_final.json")); SS = [str(s) for s in O["seeds"]]
P = O["per_seed"]
def v(s, n, m, i, rule="T"): return P[s][f"{n}|{rule}"][m][i]
def verdict(d):
    m, sd = np.mean(d), np.std(d, ddof=1)
    return "确认领先" if m > 2 * sd and m >= 0.01 else ("确认落后" if -m > 2 * sd and -m >= 0.01 else "未确认")
fm = lambda d: f"{np.mean(d):+.4f}±{np.std(d, ddof=1):.4f}（" + "、".join(f"{x:+.4f}" for x in d) + f"）{verdict(d)}"
for m, lab in (("m12", "12 套"), ("m13", "13 套")):
    for i in (0, 1):
        a = [v(s, "P3", m, i) for s in SS]; b = [v(s, "S1_2Ddiam", m, i) for s in SS]
        print(f"{lab} F1@{0.5 if i == 0 else 0.7}：L3 {np.mean(a):.4f}±{np.std(a, ddof=1):.4f}，二维直径 {np.mean(b):.4f}±{np.std(b, ddof=1):.4f}；差 {fm([x - y for x, y in zip(a, b)])}")
print("三遍（固定 cp−2 / 规则 T，12 套 F1@0.5，五种子均值）：")
for n in ("P1", "P2", "P3", "S1_2Ddiam"):
    f = np.mean([v(s, n, "m12", 0, "fix") for s in SS]); t = np.mean([v(s, n, "m12", 0, "T" if n != "P1" else "fix") for s in SS])
    f7 = np.mean([v(s, n, "m12", 1, "fix") for s in SS]); t7 = np.mean([v(s, n, "m12", 1, "T" if n != "P1" else "fix") for s in SS])
    print(f"  {n:10s} 固定 {f:.4f}/{f7:.4f}  规则T {t:.4f}/{t7:.4f}")
dss = sorted(P[SS[0]]["P3|T"]["per"])
print("逐套（规则 T，F1@0.5/0.7 五种子均值；L3 − 二维直径；直径/真值之比的中位 [最小,最大]，五种子 × 体积）：")
for ds in dss:
    a = [P[s]["P3|T"]["per"][ds] for s in SS]; b = [P[s]["S1_2Ddiam|T"]["per"][ds] for s in SS]
    r3 = [O["ratio"][s][k]["P3"] for s in SS for k in O["ratio"][s] if k.startswith(ds + "__")]
    r2 = [O["ratio"][s][k]["S1_2Ddiam"] for s in SS for k in O["ratio"][s] if k.startswith(ds + "__")]
    print(f"  {ds:12s} L3 {np.mean([x[0] for x in a]):.3f}/{np.mean([x[1] for x in a]):.3f} 二维直径 {np.mean([x[0] for x in b]):.3f}/{np.mean([x[1] for x in b]):.3f} 差 {np.mean([x[0]-y[0] for x,y in zip(a,b)]):+.3f}/{np.mean([x[1]-y[1] for x,y in zip(a,b)]):+.3f} | 比 L3 {np.median(r3):.2f}[{min(r3):.2f},{max(r3):.2f}] 二维 {np.median(r2):.2f}[{min(r2):.2f},{max(r2):.2f}]")
allr = lambda n: np.array([O["ratio"][s][k][n] for s in SS for k in O["ratio"][s] if not k.startswith("_")])
for n in ("P1", "P2", "P3", "S1_2Ddiam"):
    r = allr(n); e = np.abs(np.log(r))
    print(f"  全部体积 {n:10s} |log 比| 中位 {np.median(e):.3f}，比在 [0.8,1.25] 内 {np.mean((r>=0.8)&(r<=1.25)):.0%}，<0.5 {np.mean(r<0.5):.0%}，>2 {np.mean(r>2):.0%}")
print("规则 T 改阈值的体积数（每种子）：", {s: O["ratio"][s]["_nthr"] for s in SS})
