import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
s = open("eo_analysis_part2.py").read()
s = s.replace("（并行运行用：eo_analysis.py 第 2、3 部分的逐字拷贝，内容未改；主脚本仍在跑第 1 部分）",
              f"（本地读取版：eo_analysis_part2.py 的逐字拷贝，只改数据来源——已完整拷到本地 NVMe {_NSP_SCRATCH}/nc/eo_local 的目录从本地读（逐字节相同的副本），其余仍读共享盘；recs 按目录缓存；分析逻辑未改）", 1)
a = "from test_r6 import recs, score  # noqa: E402\n"
b = a + f'''import functools  # noqa: E402
LOCAL = "{_NSP_SCRATCH}/nc/eo_local"


def loc(d):
    """已完整拷到本地并核对过的目录改从本地读；其余照旧读共享盘。"""
    p = os.path.join(LOCAL, os.path.basename(d))
    return p if os.path.exists(os.path.join(LOCAL, ".complete_" + os.path.basename(d))) else d


recs = functools.lru_cache(maxsize=None)(recs)   # 纯函数：同一目录只读一次（errs 与分层统计会重复调用）
'''
assert s.count(a) == 1; s = s.replace(a, b)
a2 = "orig_dirs = S5.dirs\n"
b2 = '''_orig_dirs = S5.dirs


def orig_dirs(split, mode, seeds):
    D = _orig_dirs(split, mode, seeds)
    return {k: [loc(d) for d in v] for k, v in D.items()}
'''
assert s.count(a2) == 1; s = s.replace(a2, b2)
a3 = """    return [glob.glob(f"{RES}/orx_{ST[KEY[(mode, s)]]['run']}_thr_eval")[0] for s in SEEDS]"""
b3 = """    return [loc(glob.glob(f"{RES}/orx_{ST[KEY[(mode, s)]]['run']}_thr_eval")[0]) for s in SEEDS]"""
assert s.count(a3) == 1; s = s.replace(a3, b3)
s = s.replace("eo_strat_p2.json", "eo_strat_local.json").replace("eo_val_p2.json", "eo_val_local.json")
open("eo_analysis_part2_local.py", "w").write(s)
print("ok")
