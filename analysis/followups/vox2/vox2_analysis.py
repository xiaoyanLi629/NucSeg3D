"""VOX2 分析（第三轮审稿 MF2；口径见 DESIGN_VOX2.md，只报告，CPU）。直接复用 r6/s5_stats.main（不改它），只换它读的逐体积记录：
  旧：原样调用（正式五种子测试集 gt 协议，与补充表 3 的来源 r6/s5final/test_gt.json 相同的参数 --base_fixed），先核对与已存结果逐项相同；
  新：每个种子的记录 = 原记录，但 BBBC050、NISNet3D zebrafish 全部体积与 NIS3D 4 个、Purdue 3 个体积换成源体素运行的记录
      （S1 五种子 VOX_S1_* + VOX2N_S1_*，二维五种子 VOX_2D_s0、VOX2_2D_* + VOX2N_2D_*，原版 VOX_base + VOX2N_base）。
两种口径各跑全部 13 套（含 12 套宏平均与确认规则、逐套配对自助 10000 次与三档判定）与 C1 干净子集。输出 $NUCSEG3D_SCRATCH/nc/gpu/vox2/。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import glob, json, os, sys
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/five_seed'))
import s5_stats as S5  # noqa: E402
import test_r6  # noqa: E402
RES = f"{_NSP_ROOT}/results"; O = f"{_NSP_SCRATCH}/nc/gpu/vox2"; os.makedirs(O, exist_ok=True)
ST = json.load(open(f"{_NSP_SCRATCH}/r6/sched_state.json"))
SEEDS = ["s0", "s1000", "s2000", "s3000", "s4000"]
orig_recs = S5.recs; orig_dirs = S5.dirs


def rd(key):
    return glob.glob(f"{RES}/orx_{ST[key]['run']}*_thr_eval")[0]


NEWD = {"S1": [[rd(f"VOX_S1_{s}"), rd(f"VOX2N_S1_{s}")] for s in SEEDS],
        "2D": [[rd("VOX_2D_s0") if s == "s0" else rd(f"VOX2_2D_{s}"), rd(f"VOX2N_2D_{s}")] for s in SEEDS],
        "base": [[rd("VOX_base"), rd("VOX2N_base")]]}
SUBST = {}


def dirs_new(split, mode, seeds):
    D = orig_dirs(split, mode, seeds)
    out = {}
    for nm in ("S1", "2D", "base"):
        out[nm] = []
        for i, d in enumerate(D[nm]):
            tok = f"MERGED::{nm}::{i}::{d}"; SUBST[tok] = (d, NEWD[nm][i]); out[nm].append(tok)
    return out


def recs_new(d, fixed=False):
    if not d.startswith("MERGED::"):
        return orig_recs(d, fixed)
    base, news = SUBST[d]
    R = dict(orig_recs(base, fixed)); n = 0
    for nd in news:
        for k, v in orig_recs(nd, fixed).items():
            assert k in R, k
            R[k] = v; n += 1
    print(f"  替换 {n} 个体积：{os.path.basename(base)[:12]} ← {[os.path.basename(x)[:12] for x in news]}", flush=True)
    return R


for tag in ("old", "new"):
    for sub in (None, f"{_NSP_SHARED}/cellposeft_orx/audit/clean_test.json"):
        if tag == "new":
            S5.dirs, S5.recs = dirs_new, recs_new
        else:
            S5.dirs, S5.recs = orig_dirs, orig_recs
        out = f"{O}/vox2_{tag}_gt{'_C1' if sub else ''}.json"
        sys.argv = ["s5_stats.py", "test", "gt", "--seeds", *SEEDS, "--base_fixed", "--out", out] + (["--subset", sub] if sub else [])   # 与 r6/s5final/run.sh 的正式表同参数
        print(f"\n######## {'原体素（正式表）' if tag == 'old' else '源体素替换后'}｜gt 协议{'｜C1' if sub else ''}", flush=True)
        S5.main()
for mine, ref in (("vox2_old_gt.json", "test_gt.json"), ("vox2_old_gt_C1.json", "test_gt_clean_test.json")):
    a, b = json.load(open(f"{_NSP_SCRATCH}/r6/s5final/" + ref)), json.load(open(f"{O}/{mine}"))
    same = a["macro"] == b["macro"] and a["per_ds"] == b["per_ds"]
    print("核对：", mine, "与正式表 r6/s5final/" + ref, "逐项相同" if same else "不同！", flush=True)
