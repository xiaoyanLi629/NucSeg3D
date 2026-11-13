"""M1_val_s0（cc8925e6）补跑合并：把单独补跑的 1 个体积（同一工作树、同一参数、--workers 1）的 vols/*.json 与 masks/*.npz 拷进原结果目录，
核对原有 87 个体积的文件 md5 不变、合并后体积集合与 a69ee4fa 的 88 个相同；再由全部 vols 重组 rows.json 与 summary.json（eval_gt.summarize），
然后跑 gx_thr_eval 的离线部分（sizeThr_dir.py 与 ruleT.json）。用法：python m1_merge.py"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_RUNS = _nsp_os.environ.get('NUCSEG3D_RUNS', '/path/to/orx_runs')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import glob, json, os, shutil, subprocess, sys
W = f"{_NSP_RUNS}/cc8925e6-ed99-4123-a5df-f1db5e3ff888/repo/src"
D = f"{_NSP_ROOT}/results/orx_cc8925e6-ed99-4123-a5df-f1db5e3ff888_gx_thr_eval"
F = f"{_NSP_SCRATCH}/nc/gpu/m1fix"
REF = glob.glob(f"{_NSP_ROOT}/results/orx_a69ee4fa*")[0]
sys.path.insert(0, W); os.chdir(W)
import eval_gt as E
before = dict(reversed(l.split()) for l in open(f"{F}/md5_before.txt"))
for f in os.listdir(f"{F}/one/vols"):
    assert not os.path.exists(f"{D}/vols/{f}"), f
    shutil.copy(f"{F}/one/vols/{f}", f"{D}/vols/{f}")
    k = f[:-5]; shutil.copy(f"{F}/one/masks/{k}.npz", f"{D}/masks/{k}.npz")
after = subprocess.run(f"cd {D} && md5sum " + " ".join(before), shell=True, capture_output=True, text=True).stdout
after = dict(reversed(l.split()) for l in after.strip().splitlines())
assert after == before, "原有文件的 md5 变了"
v = sorted(os.listdir(f"{D}/vols")); assert set(v) == set(os.listdir(f"{REF}/vols")) and len(v) == 88, len(v)
rows = [x for f in v for x in json.load(open(f"{D}/vols/{f}"))]
json.dump(rows, open(f"{D}/rows.json", "w"))
S = E.summarize(rows); S.update(model=f"{_NSP_ROOT}/models/models/orx_t3d_r5s1_s0", eval_opts=dict(views=3, ens2d=None, ens_alpha=0.5),
                                 protocol=dict(tag="gtval", split="val", max_per_ds=0, diam_from=None, views1_ds=["cellos", "parhyale", "mesospim"]),
                                 note="M1_val_s0：原运行在 87/88 挂死，缺的 blastospim__BlastoSPIM1_val_F38_109_image_0001 由同一工作树、同一参数、--workers 1 补跑后合并（m1_merge.py）")
json.dump(S, open(f"{D}/summary.json", "w"), indent=1)
print(f"md5 不变 {len(before)} 个文件；合并后 {len(v)} 个体积；", E.summary_line("S1_s0_zero_dec", S, "gtval"), flush=True)
subprocess.run([sys.executable, "sizeThr_dir.py", D], check=True)
sys.argv = ["gx_thr_eval.py"]
import importlib.util
src = open("gx_thr_eval.py").read(); ns = {}
exec(src[:src.index("args = sys.argv[1:]")], ns)
print("ruleT.json", ns["rule_t"](D), "个体积")
