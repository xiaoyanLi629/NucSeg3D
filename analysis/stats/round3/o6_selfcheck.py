"""O6 的宏平均自检补做：o6.py 运行时 ref/ 还没生成，这里把登记格的逐种子宏平均与 s5_stats.py val gt（五种子）比对，写回 o6/results.json。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
R = f"{_NSP_SCRATCH}/nc/stats/round3/o6"
o = json.load(open(f"{R}/results.json", encoding="utf-8"))
ref = json.load(open(f"{R}/ref/s5_val_gt_5seeds_basefixed.json", encoding="utf-8"))["macro"]
worst = 0.0
for tag, rt in (("13", "13 套"), ("12", "去 mesospim 12 套")):
    g = o["grid"][f"{tag}|a=0.7|b=0.85|f=0.15"]
    for lab in ("F1@0.5", "F1@0.7"):
        worst = max([worst] + [abs(x - y) for x, y in zip(g[lab]["S1_seeds"], ref[lab][rt]["S1_seeds"])]
                    + [abs(x - y) for x, y in zip(g[lab]["two_d_seeds"], ref[lab][rt]["two_d_seeds"])])
o["selfcheck_macro_vs_s5_stats"] = worst
json.dump(o, open(f"{R}/results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("登记格逐种子宏平均与 s5_stats.py val（五种子）最大绝对差：", worst)
