"""把补跑的 3 个体积并入 L3v_P3_d17（17a33dac）的结果目录，生成 rows.json / summary.json（与 eval_gt 结尾同样的写法），再跑 sizeThr_dir（thr_eval 第二步）。
核对：合并后体积集合 = P2（63912bb6）的 88 个；已有 85 个体积的 vols/*.json 与 masks 不改动（合并前后 md5 相同）。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_RUNS = _nsp_os.environ.get('NUCSEG3D_RUNS', '/path/to/orx_runs')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import glob, hashlib, json, os, shutil, subprocess, sys
SRC = f"{_NSP_RUNS}/17a33dac-a81a-4a89-96c9-6e814266cc3f/repo/src"; sys.path.insert(0, SRC)
R = f"{_NSP_ROOT}/results"
P3 = f"{R}/orx_17a33dac-a81a-4a89-96c9-6e814266cc3f_thr_eval"; P2 = f"{R}/orx_63912bb6-841f-40c2-aaaa-527bdccf001d_thr_eval"
FX = f"{_NSP_SHARED}/nc_heldout/L3/fix_p3_d17"
IDS = ["BlastoSPIM1_val_M6_016_image_0001", "BlastoSPIM1_val_M6_018_image_0001", "BlastoSPIM1_val_T2_006_image_0001"]
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
assert not os.path.exists(P3 + "/rows.json"), "P3 已有 rows.json，不覆盖"
before = {f: md5(f) for f in glob.glob(P3 + "/vols/*.json") + glob.glob(P3 + "/masks/*.npz")}
assert len(before) == 170, len(before)
for i in IDS:
    k = f"blastospim__{i}"
    for sub, ext in (("vols", "json"), ("masks", "npz")):
        s, d = f"{FX}/{i}/{sub}/{k}.{ext}", f"{P3}/{sub}/{k}.{ext}"
        assert os.path.exists(s) and not os.path.exists(d), (s, d)
        shutil.copy2(s, d)
after = {f: md5(f) for f in glob.glob(P3 + "/vols/*.json") + glob.glob(P3 + "/masks/*.npz")}
assert all(after[f] == h for f, h in before.items()), "已有体积被改动"
v3 = sorted(os.path.basename(f) for f in glob.glob(P3 + "/vols/*.json")); v2 = sorted(os.path.basename(f) for f in glob.glob(P2 + "/vols/*.json"))
assert v3 == v2 and len(v3) == 88, (len(v3), set(v2) ^ set(v3))
from eval_gt import summarize, summary_line
rows = []
for f in sorted(glob.glob(P3 + "/vols/*.json")):
    rows += json.load(open(f))
json.dump(rows, open(P3 + "/rows.json", "w"))
S = summarize(rows); S.update(model=f"{_NSP_ROOT}/models/models/orx_t3d_r5s1_s0", minutes=None,
                               eval_opts=dict(views=3, ens2d=None, ens_alpha=0.5),
                               protocol=dict(tag="fpval", split="val", max_per_ds=0, diam_from=f"{_NSP_SHARED}/nc_heldout/L3/val_d17_r2",
                                             views1_ds=["cellos", "parhyale", "mesospim"]),
                               merged_note="执行偏离：85 个体积来自 17a33dac，3 个 blastospim 由 fix_p3_d17 以 --workers 1 补跑后合并")
json.dump(S, open(P3 + "/summary.json", "w"), indent=1)
print(summary_line(S["model"], S, "fpval"))
out = subprocess.run([sys.executable, f"{SRC}/sizeThr_dir.py", P3], capture_output=True, text=True, check=True).stdout
open(P3 + "/sizeThr.out", "w").write(out); open(FX + "/sizeThr.out", "w").write(out); print(out[-1500:])
print("MERGE_OK", len(rows), "行")
