"""五种子正式版的附加核对：只取种子 0/1000/2000 时，本代码的读数与 r6 已报告的三种子数字是否仍逐项相同。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json, sys
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/stats'))
import ncio, nc_stats
seeds = ["s0", "s1000", "s2000"]
M = {m: ncio.load_mode(m, seeds)[0] for m in ncio.MODES}
c = nc_stats.consistency(M, seeds)
json.dump(c, open(f"{_NSP_SCRATCH}/nc/stats/check3.json", "w"), ensure_ascii=False, indent=1)
print(f"三种子子集核对：{c['n_items']} 项，最大绝对差 {c['worst_abs_diff']:.3g}，{'完全一致' if c['identical'] else '有差异'}")
