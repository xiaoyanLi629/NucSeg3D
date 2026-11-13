"""并行生成 recs 缓存（每个运行目录一个进程）。用法：python precache.py s0 s1000 ..."""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import sys
from multiprocessing import Pool
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/stats'))
import ncio


def go(a):
    d, fx = a
    ncio.cached_recs(d, fixed=fx)
    return d


if __name__ == "__main__":
    seeds = sys.argv[1:]
    J = []
    for mode in ncio.MODES:
        D = ncio.s5_stats.dirs("test", mode, seeds)
        J += [(d, False) for d in D["S1"] + D["2D"]] + [(D["base"][0], True)]
    with Pool(len(J)) as p:
        for d in p.imap_unordered(go, J):
            print("done", d, flush=True)
