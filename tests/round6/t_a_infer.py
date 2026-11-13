"""测试 a-3：关闭新功能时，验证体积的推理输出（float32 的 dP、cellprob）与改前逐位相同。
用法：python t_a_infer.py <src目录> <模型目录> [tag]
预处理照抄 eval_gt.run_volume，调 Model3D.eval（见 t_a_infer_lib），打印每种设置下 dP / cellprob 的哈希。
设置：bbbc050 单视图、三视图；cellos（走 z 插值）单视图；tapenade view_perm 102。"""
import os
import sys

src, mdir = sys.argv[1], sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(src.rstrip("/"))
sys.path.insert(0, src)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
from t_a_infer_lib import run_cases  # noqa: E402

CASES = [("bbbc050", "train_Emb11_t501", {}), ("bbbc050", "train_Emb11_t501", dict(views=3)),
         ("cellos", "EGFP_d4_flt32_28", {}), ("tapenade", "val_000027", dict(view_perm=(1, 0, 2)))]
for k, r in run_cases(mdir, CASES):
    print(f"T_a3 {tag:8s} {k:45s} 输入 {r['shape_in']} d={r['d']:.2f} an={r['an']:.3f} dP {r['dP']} cellprob {r['cp']} "
          f"|dP|均值 {r['mean_abs_dP']:.6f}", flush=True)
print("T_A_INFER_DONE")
