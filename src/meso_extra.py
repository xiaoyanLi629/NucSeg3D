"""mesoSPIM 额外域外评测（round 4 补充分析，不用于任何选择）。

mesoSPIM 是项目指定的域外数据集：清单训练部分的 c1、c2、c4、c5 四个体积按设计从不参与训练，验证集只有 c3 一个体积，
单体积分数随种子摆动很大。这里把这 4 个从未用过的体积当作额外的域外评测体积，用 gt 协议、单视图（视图规则下 mesospim
本来就用单视图）逐个模型评测，看三维模型在 mesospim 上是系统性不如二维，还是只是单体积噪声。
用法（orx script 模式）：python meso_extra.py --out <结果目录> --models <模型名或目录> ...
"""
import argparse
import json
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
sj = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mesospim_extra.json")
res = {}
for m in a.models:
    name = os.path.basename(m.rstrip("/"))
    o = f"{a.out}/{name}"
    subprocess.run([sys.executable, "eval_gt.py", "--model", m, "--out", o, "--split_json", sj, "--only", "mesospim",
                    "--workers", str(a.workers)], check=True)
    c = json.load(open(f"{o}/summary.json"))["cells"]["cp-2_vol0.15"]["per_ds"]["mesospim"]
    pv = {x["id"]: [round(x["f1"], 3), round(x["f1_7"], 3)] for x in json.load(open(f"{o}/rows.json")) if x["thr"] == -2.0 and x["frac"] == 0.15}
    res[name] = dict(f1_05=c["f1_05"], f1_07=c["f1_07"], volmean_f1_05=c["f1_05_volmean"], per_vol=pv)
    print(f"MESO {name:24s} pooled {c['f1_05']:.3f}/{c['f1_07']:.3f}  逐体积 {pv}", flush=True)
json.dump(res, open(f"{a.out}/meso_extra.json", "w"), indent=1)
print("FINAL_METRICS " + json.dumps({k: [round(v["f1_05"], 4), round(v["f1_07"], 4)] for k, v in res.items()}), flush=True)
