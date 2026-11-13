"""按数据集汇总逐核形状（读 eval_gt --save_masks 的 npz，复用 shape_diag.analyze/summarize）。
用法：python shape_by_ds.py <结果目录A> [<结果目录B> ...]   每个目录逐数据集打印一行；两个目录时再打印 B−A 的中位比差。"""
import glob, json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))  # release: shape_diag.py lives in src/
from shape_diag import analyze, summarize

res = {}
for d in sys.argv[1:]:
    by = defaultdict(list)
    for f in sorted(glob.glob(f"{d}/masks/*.npz")):
        by[os.path.basename(f).split("__")[0]] += analyze(f)
    res[d] = {ds: summarize(rs) for ds, rs in sorted(by.items())}
    json.dump(res[d], open(f"{d}/shape_by_ds.json", "w"), indent=1)
    print(f"== {os.path.basename(d.rstrip('/'))}")
    print(f"  {'数据集':13s} {'核数':>5s} {'IoU<.5':>7s} {'中位IoU':>7s} {'层数比':>6s} {'xy直径比':>8s} {'体积比':>6s} {'cp段z':>6s} {'cp段x':>6s}")
    for ds, S in res[d].items():
        print(f"  {ds:13s} {S['n_gt']:5d} {S['iou_lt05'] / max(S['matched'], 1):7.2f} {S['med_iou']:7.2f} {S['med_nz_ratio']:6.2f} {S['med_dxy_ratio']:8.2f}"
              f" {S['med_V_ratio']:6.2f} {S.get('med_cpz_ratio', float('nan')):6.2f} {S.get('med_cpx_ratio', float('nan')):6.2f}")
