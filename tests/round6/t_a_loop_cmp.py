"""测试 a-4 的比对：同一份数据流（--workers 0、固定 pid）下 train3d 前几步的损失/梯度范数。
用法：python t_a_loop_cmp.py <目录> [前缀]  —— 读 <前缀>old / old2 / new / newcond 的 train_log.json（缺的跳过）。"""
import json
import os
import sys

O = sys.argv[1]; pre = sys.argv[2] if len(sys.argv) > 2 else "loop_"
L = {t: json.load(open(os.path.join(O, f"{pre}{t}", "train_log.json"))) for t in ("old", "old2", "new", "newcond")
     if os.path.exists(os.path.join(O, f"{pre}{t}", "train_log.json"))}
print("步  " + "  ".join(f"{t:>22s}" for t in L) + "   （损失；末列为梯度范数 old/new）")
for i in range(len(L["old"])):
    print(f"{L['old'][i]['step']:2d}  " + "  ".join(f"{L[t][i]['loss']!r:>22}" for t in L)
          + f"   {L['old'][i]['gnorm']!r} / {L['new'][i]['gnorm']!r}")
k = lambda t, i: (L[t][i]["loss"], L[t][i]["flow"], L[t][i]["cp"], L[t][i]["gnorm"])
n = len(L["old"])
rel = lambda t: max(abs(L[t][i]["loss"] - L["old"][i]["loss"]) / abs(L["old"][i]["loss"]) for i in range(n))
if "old2" in L:
    print(f"T_a4{pre} 旧代码两次运行逐步相同（损失/流/概率/梯度范数）=", [k("old", i) == k("old2", i) for i in range(n)],
          f"；损失最大相对差 {rel('old2'):.1e}")
print(f"T_a4{pre} 新代码（--cond none）与旧代码逐步相同 =", [k("old", i) == k("new", i) for i in range(n)], f"；损失最大相对差 {rel('new'):.1e}")
print(f"T_a4{pre} 第 1 步前向损失：old/new 相同 = {L['old'][0]['loss'] == L['new'][0]['loss']}")
if "newcond" in L:
    print(f"T_a4{pre} 新代码 --cond res --cond_enc 2 第 1 步损失与旧代码相同（零初始化）=", L["newcond"][0]["loss"] == L["old"][0]["loss"],
          "；之后各步相同 =", [L["newcond"][i]["loss"] == L["old"][i]["loss"] for i in range(1, n)])
