"""五种子正式版的一致性核对（在 ref/ 的 s5_stats.py 五种子输出出来后补做），并把三种子子集核对一起写回 nc_stats_results.json，再重写摘要。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json, sys
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/stats'))
import ncio, nc_stats
P = f"{_NSP_SCRATCH}/nc/stats/nc_stats_results.json"
res = json.load(open(P, encoding="utf-8"))
seeds = res["seeds"]
M = {m: ncio.load_mode(m, seeds)[0] for m in ncio.MODES}
c = nc_stats.consistency(M, seeds)
c3 = json.load(open(f"{_NSP_SCRATCH}/nc/stats/check3.json", encoding="utf-8"))
c["three_seed_subset"] = {k: c3[k] for k in ("n_items", "worst_abs_diff", "identical")}
res["consistency"] = c
json.dump(res, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(f"五种子核对：{c.get('note') or str(c['n_items']) + ' 项，最大绝对差 ' + format(c['worst_abs_diff'], '.3g')}；三种子子集 {c3['n_items']} 项，最大绝对差 {c3['worst_abs_diff']:.3g}")
nc_stats.replot(P)
