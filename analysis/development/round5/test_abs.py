import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
# 测试集：各模型逐种子的宏平均绝对值与 mesospim、purdue 逐种子（完整流程逐体积口径；gt 协议 pooled），cp-2_vol0.15，去 bbbc032
import sys, json, statistics as st
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/development'))
from fp_cmp import load, agg, macros
FP = {"二维": ["orx_r5merged_2d_s0_test", "orx_r5merged_2d_s1000_test", "orx_r5merged_2d_s2000_test"],
      "A1": ["orx_2ca37361-de8e-4c92-8d8b-b01715027fc1_fptest", "orx_4649931a-e7e8-49b5-86fe-224319cc8c60_fptest", "orx_74ac442f-da13-48cf-9e78-84e31bfa18f1_fptest"],
      "S1": ["orx_12a21e03-3c34-4414-9182-3d657c95f67f_fptest", "orx_15b6437e-57c2-4258-ac8a-b78a8b34a0e9_fptest", "orx_1c26abce-3600-4bd3-b1a5-c62f997899e6_fptest"]}
base = load("orx_r5merged_base_test")
print("== 完整流程（45 个体积，逐体积再宏平均）")
for nm, ds in FP.items():
    rows = []
    for d in ds:
        A = load(d); ks = sorted(set(A) & set(base)); P = agg(A, ks)
        rows.append((macros(P), P, len(ks)))
    for s in ("全13套", "去mesospim"):
        v5 = [r[0][s]["vm5"] for r in rows]; v7 = [r[0][s]["vm7"] for r in rows]
        print(f"{nm} {s}: " + "  ".join(f"{a:.4f}/{b:.4f}" for a, b in zip(v5, v7)) + f"  均值 {st.mean(v5):.4f}/{st.mean(v7):.4f}  体积 {[r[2] for r in rows]}")
    for d in ("mesospim", "purdue", "tapenade", "nucverse3d"):
        print(f"  {nm} {d} 逐种子: " + "  ".join(f"{r[1][d]['vm5']:.3f}/{r[1][d]['vm7']:.3f}" for r in rows) + f" (n={rows[0][1][d]['n']})")
PB = agg(base, sorted(base)); MB = macros(PB)
for s in ("全13套", "去mesospim"):
    print(f"原版 {s}: {MB[s]['vm5']:.4f}/{MB[s]['vm7']:.4f}")
print("原版 mesospim", f"{PB['mesospim']['vm5']:.3f}/{PB['mesospim']['vm7']:.3f}")
print("每套体积数", {d: PB[d]["n"] for d in sorted(PB)})

print("== gt 协议（227 个体积，pooled）")
R = f"{_NSP_ROOT}/results/"; C = "cp-2_vol0.15"
MESO = R + "orx_4349ddbc-f2c2-4119-8162-fa0dd9f75fae_meso_fp/"
GT = {"二维": ["orx_26044a2f-1198-47ed-86e5-5d514a05d00f_gttest", "orx_440f0cf2-ee6b-436f-bf09-fb634a50db53_gttest", "orx_0021daf0-5778-4632-a567-4cec8bff207f_gttest"],
      "A1": ["orx_8026a9ee-b35c-4457-acb4-b78b42e6a2c1_gttest", "orx_66ec2595-e352-425b-9dce-3120a24f666a_gttest", "orx_941e12de-8474-4e29-9d11-cd99f56b752c_gttest"],
      "S1": ["orx_1c8ec641-e5b7-422a-a1fa-53276db305f2_gttest", "orx_85fa07a0-22d3-4110-a351-7f4881dddf70_gttest", "orx_ff250960-492a-4151-a8ed-be1e6b7b9587_gttest"]}
for nm, ds in GT.items():
    out = []
    for i, d in enumerate(ds):
        P = json.load(open(R + d + "/summary.json"))["cells"][C]["per_ds"]
        if nm == "二维":
            P["mesospim"] = json.load(open(MESO + f"orx_v5_2d_s{(0, 1000, 2000)[i]}_gt_test/summary.json"))["cells"][C]["per_ds"]["mesospim"]
        out.append(P)
    for d in ("mesospim", "purdue", "tapenade"):
        print(f"{nm} {d} 逐种子: " + "  ".join(f"{p[d]['f1_05']:.3f}/{p[d]['f1_07']:.3f}" for p in out))
b = json.load(open(MESO + "base_gt_test/summary.json"))["cells"][C]["per_ds"]["mesospim"]
print("原版 mesospim", {k: b[k] for k in b if k.startswith("f1") or k.startswith("n")})
p = json.load(open(R + GT["S1"][0] + "/summary.json"))["cells"][C]["per_ds"]
print("S1 种子0 每套字段示例", list(p["purdue"].keys()))
print("每套体积/核数", {k: (v.get("n_vol", v.get("n")), v.get("n_gt")) for k, v in p.items()})
