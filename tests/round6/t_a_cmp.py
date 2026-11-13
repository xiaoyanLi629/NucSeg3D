"""测试 a-1 的比对：t_a_data.py 在旧代码、新代码上的输出逐块比较。用法：python t_a_cmp.py old.json new.json"""
import json
import sys

o, n = (json.load(open(p)) for p in sys.argv[1:3])
ok_all = True
for name in ("A1", "X1", "X3", "perm", "ALL"):
    a = o[name]
    for tag in (name, name + "+res"):
        b = n[tag]
        same = a["blocks"] == b["blocks"]
        first = next((i for i, (x, y) in enumerate(zip(a["blocks"], b["blocks"])) if x != y), None)
        gs = (a["gpu"] == b["gpu"]) if a["gpu"] else None
        line = f"T_a1 {tag:9s} 旧代码 vs 新代码：CPU 块（图像/标签/有效区/增强参数）{len(a['blocks'])} 块逐项一致 = {same}"
        if first is not None:
            line += f"（第一处不同：块 {first}）"
        if gs is not None:
            gfirst = next((i for i, (x, y) in enumerate(zip(a["gpu"], b["gpu"])) if x != y), None)
            line += f"；GPU to_patch 输出 {len(a['gpu'])} 块逐位一致 = {gs}" + (f"（第一处不同：块 {gfirst}）" if gfirst is not None else "")
        print(line)
        ok_all &= same and (gs is not False)
print("T_a1 全部一致 =", ok_all)
