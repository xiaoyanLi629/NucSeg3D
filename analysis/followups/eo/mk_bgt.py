import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
s = open("eo_analysis.py").read()
head = s[:s.index("for which, lab in")]
head = head.replace('"""EO 对照分析', '"""（补算版：eo_analysis.py 第 1 部分里 B × gt × C1 这一组的逐字调用，原进程在写出这一组之前被结束；只改数据来源为本地逐字节副本，s5_stats 的 recs 加缓存，分析逻辑未改）EO 对照分析', 1)
a = "from test_r6 import recs, score  # noqa: E402\n"
head = head.replace(a, a + f'''import functools  # noqa: E402
LOCAL = "{_NSP_SCRATCH}/nc/eo_local"


def loc(d):
    p = os.path.join(LOCAL, os.path.basename(d))
    return p if os.path.exists(os.path.join(LOCAL, ".complete_" + os.path.basename(d))) else d


S5.recs = functools.lru_cache(maxsize=None)(S5.recs)
''', 1)
head = head.replace("orig_dirs = S5.dirs\n", '''_orig_dirs = S5.dirs


def orig_dirs(split, mode, seeds):
    D = _orig_dirs(split, mode, seeds)
    return {k: [loc(d) for d in v] for k, v in D.items()}
''', 1)
head = head.replace("""    return [glob.glob(f"{RES}/orx_{ST[KEY[(mode, s)]]['run']}_thr_eval")[0] for s in SEEDS]""",
                    """    return [loc(glob.glob(f"{RES}/orx_{ST[KEY[(mode, s)]]['run']}_thr_eval")[0]) for s in SEEDS]""", 1)
body = f'''
which, lab, mode, sub = "B", "编码器再训练的贡献：EO − 二维（输出里“S1”= EO，“二维”= 二维）", "gt", "{_NSP_SHARED}/cellposeft_orx/audit/clean_test.json"
S5.dirs = make(which)
sys.argv = ["s5_stats.py", "test", mode, "--seeds", *SEEDS, "--base_fixed", "--out", f"{{O}}/eo_{{which}}_{{mode}}{{'_C1' if sub else ''}}.json"] + (["--subset", sub] if sub else [])
print(f"\\n######## {{lab}}｜{{'完整流程' if mode == 'fp' else 'gt 协议'}}{{'｜C1' if sub else ''}}", flush=True)
S5.main()
print("EO_BGTC1_DONE", flush=True)
'''
open("eo_B_gt_C1_local.py", "w").write(head + body)
print("ok")
