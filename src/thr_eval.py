"""第六轮候选 T（尺度一致阈值）的评测入口（orx script 模式）：
先跑 eval_gt（阈值网格 0、−1、−2、−3、−4、−6，--save_masks 存 cellprob），再用 sizeThr_dir.py 按 AGENTS.md 预定的规则 R2 离线逐体积选阈值。
用法：python thr_eval.py --out <结果目录> --model <模型> [eval_gt 的其他参数，如 --views 3 --views1_ds cellos parhyale mesospim]"""
import subprocess
import sys

args = sys.argv[1:]
i = args.index("--out"); out = args[i + 1]; rest = args[:i] + args[i + 2:]
cmd = [sys.executable, "eval_gt.py", "--out", out, "--thrs", "0", "-1", "-2", "-3", "-4", "-6", "--save_masks"] + rest
print("RUN", " ".join(cmd[1:]), flush=True)
subprocess.run(cmd, check=True)
subprocess.run([sys.executable, "sizeThr_dir.py", out], check=True)
