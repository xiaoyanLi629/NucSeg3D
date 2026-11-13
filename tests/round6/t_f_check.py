"""测试 f 的核对：条件模型的 eval_gt 结果里，每个体积记录的 f（res 字段，Model3D.eval 实际交给网络的体积 f）
与按清单体素、所用直径和 eval_gt / Model3D.eval 取整规则推出的 f（r6common.expected_res）一致。
用法：python t_f_check.py <eval_gt 输出目录> ..."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r6common import expected_res  # noqa: E402

ok = True
for d in sys.argv[1:]:
    seen = set()
    for r in json.load(open(f"{d}/rows.json")):
        k = (r["ds"], r["id"])
        if k in seen:
            continue
        seen.add(k)
        f, net = expected_res(r["shape"], r["ratio"], r.get("d_used", r["d_gt"]))
        same = r.get("res") is not None and all(abs(a - b) < 1e-3 for a, b in zip(r["res"], f))
        ok &= same
        print(f"T_f {os.path.basename(d)} {r['ds']:9s} {r['id']:18s} z/xy {r['ratio']:.3f} 直径 {r.get('d_used', r['d_gt']):.2f}  "
              f"eval_gt 记录的 f {r.get('res')}  推算 {[round(v, 4) for v in f]}（网络网格 {net}）一致 = {same}")
print("T_f f 记录与推算全部一致 =", ok)
