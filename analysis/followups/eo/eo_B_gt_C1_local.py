"""（补算版：eo_analysis.py 第 1 部分里 B × gt × C1 这一组的逐字调用，原进程在写出这一组之前被结束；只改数据来源为本地逐字节副本，s5_stats 的 recs 加缓存，分析逻辑未改）EO 对照分析（第三轮预审 SF2；口径按 DESIGN_EO.md 第 5 节预先写定，md5 fb9507e7…，AGENTS.md 331ec39 登记；只报告，CPU）。
1）复用 r6/s5_stats.main（不改它），参数与正式五种子表相同（test fp|gt，五种子，--base_fixed），把它读的两列换成：
    A 解码器的贡献：“S1”= S1，“二维”= EO（同种子）；B 编码器再训练的贡献：“S1”= EO，“二维”= 二维（同种子）。全部 13 套与 C1 子集。
2）按视图分层（单视图 3 套 cellos、parhyale、mesospim；三视图 10 套；另给 12 套与 13 套），S1−EO、EO−二维、S1−二维 的五种子配对均值 ± SD 与确认判定
   （均值 > 2×SD 且 ≥ 0.01），以及分层误差分解（切碎、合并、漏检、边界、误检、z 向合并占比；五种子平均）。同 m1/m1_strat.py 的写法。
3）验证集参照（不检验）：EO、S1、二维的五种子验证集 gt 协议 12/13 套宏平均（规则 T）。
种子 1000 的测试集 gt 用重跑 EO_tGT_s1000_r2（9d487c18）；原运行 d22bc567 在 18/227 被 SIGTERM 终止，不用。
输出 $NUCSEG3D_SCRATCH/nc/gpu/eo/eo_A_{fp,gt}[_C1].json、eo_B_*.json、eo_strat.json、eo_val.json、eo_analysis.log。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import glob, json, os, sys
import numpy as np
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/five_seed'))
import s5_stats as S5  # noqa: E402
from test_r6 import recs, score  # noqa: E402
import functools  # noqa: E402
LOCAL = f"{_NSP_SCRATCH}/nc/eo_local"


def loc(d):
    p = os.path.join(LOCAL, os.path.basename(d))
    return p if os.path.exists(os.path.join(LOCAL, ".complete_" + os.path.basename(d))) else d


S5.recs = functools.lru_cache(maxsize=None)(S5.recs)
RES = f"{_NSP_ROOT}/results"; O = f"{_NSP_SCRATCH}/nc/gpu/eo"
ST = json.load(open(f"{_NSP_SCRATCH}/r6/sched_state.json"))
SEEDS = ["s0", "s1000", "s2000", "s3000", "s4000"]; V1 = ("cellos", "parhyale", "mesospim")
_orig_dirs = S5.dirs


def orig_dirs(split, mode, seeds):
    D = _orig_dirs(split, mode, seeds)
    return {k: [loc(d) for d in v] for k, v in D.items()}
KEY = {("gt", s): ("EO_tGT_s1000_r2" if s == "s1000" else f"EO_tGT_{s}") for s in SEEDS}
KEY.update({("fp", s): f"EO_tFP_{s}" for s in SEEDS}); KEY.update({("val", s): f"EO_v_{s}" for s in SEEDS})


def eo_dirs(mode):
    return [loc(glob.glob(f"{RES}/orx_{ST[KEY[(mode, s)]]['run']}_thr_eval")[0]) for s in SEEDS]


def make(which):
    def dirs(split, mode, seeds):
        assert seeds == SEEDS
        D = orig_dirs(split, mode, seeds); E = eo_dirs("val" if split == "val" else mode)
        return {"S1": D["S1"], "2D": E, "base": D["base"]} if which == "A" else {"S1": E, "2D": D["2D"], "base": D["base"]}
    return dirs



which, lab, mode, sub = "B", "编码器再训练的贡献：EO − 二维（输出里“S1”= EO，“二维”= 二维）", "gt", f"{_NSP_SHARED}/cellposeft_orx/audit/clean_test.json"
S5.dirs = make(which)
sys.argv = ["s5_stats.py", "test", mode, "--seeds", *SEEDS, "--base_fixed", "--out", f"{O}/eo_{which}_{mode}{'_C1' if sub else ''}.json"] + (["--subset", sub] if sub else [])
print(f"\n######## {lab}｜{'完整流程' if mode == 'fp' else 'gt 协议'}{'｜C1' if sub else ''}", flush=True)
S5.main()
print("EO_BGTC1_DONE", flush=True)
