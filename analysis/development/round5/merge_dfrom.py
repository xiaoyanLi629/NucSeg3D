"""完整流程的直径来源合并：旧的二维测试集 rows.json 去掉 mesospim，换成按源数据体素重跑的 mesospim 行，写到新目录供 eval_gt --diam_from 用。
用法：python merge_dfrom.py <旧目录> <新 mesospim 目录> <输出目录>"""
import json, os, sys
old, new, out = sys.argv[1:4]
A = [r for r in json.load(open(f"{old}/rows.json")) if r["ds"] != "mesospim"]
Bm = [r for r in json.load(open(f"{new}/rows.json")) if r["ds"] == "mesospim"]
assert Bm and all(r.get("d_used") for r in Bm), "新目录里没有 mesospim 的 d_used"
os.makedirs(out, exist_ok=True); json.dump(A + Bm, open(f"{out}/rows.json", "w"))
print(out, len(A), "+", len(Bm), "行；mesospim d_used", sorted({(r["id"], round(r["d_used"], 3)) for r in Bm}))
