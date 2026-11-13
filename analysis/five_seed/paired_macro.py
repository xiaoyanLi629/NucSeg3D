"""从 test_r6 / s5_stats 的结果 json 里算 S1−二维 的逐种子配对差、均值、SD（ddof=1）与确认判定（均值 > 2×SD 且 ≥ 0.01）。"""
import json
import statistics as st
import sys

for f in sys.argv[1:]:
    r = json.load(open(f, encoding="utf-8"))
    print("==", f)
    for lab, M in r["macro"].items():
        for tag, e in M.items():
            s1, s2 = e["S1_seeds"], e["two_d_seeds"]
            d = [a - b for a, b in zip(s1, s2)]
            mu, sd = st.mean(d), st.stdev(d)
            ok = mu > 2 * sd and mu >= 0.01
            m1 = e.get("S1", st.mean(s1)); m2 = e.get("2D", e.get("two_d", st.mean(s2))); mb = e["base"]
            print("  %s %s: S1 %.4f 二维 %.4f 原版 %.4f | 差 %s 均值 %+.4f SD %.4f %s；对原版 %+.4f"
                  % (lab, tag, m1, m2, mb, "、".join("%+.4f" % x for x in d), mu, sd, "通过" if ok else "未通过", m1 - mb))
