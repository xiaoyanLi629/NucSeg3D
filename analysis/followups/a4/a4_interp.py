"""nc A4 判读（CPU）：读 a4_resid.py 的逐体积直方图（运行 bc2dcc2b），出区域汇总与示意图。
有符号距离 = 到最近真值边界（核–背景、核–核交界）的欧氏距离，网络体素（核径约 30），核内为负、核外为正，取整后夹到 [−20, 20]。
流的单位是网络原始输出（5×单位向量，模长约 5）；Δ = 开 − 关（关 = 三维解码器末层置零，dz ≡ 0）。
区域：核心 d ≤ −4，边界带 −3 ≤ d ≤ 3，背景 d ≥ 4。每个区域先在每套内合并体素求均值，再对数据集平均（13 套）；
另给各区域占 Σ|Δ| 的份额（体素加权，说明“总量”落在哪里），并按 z/xy ≥ 2 与 < 2 分层。
示意图的体积按中位数原则选：z/xy ≥ 2 的体积里，单视图边界带 |Δdz| 最接近这些体积中位数的那个。
输出 $NUCSEG3D_SCRATCH/nc/gpu/a4/a4_interp.json、a4_fig.png。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False; plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
D = glob.glob(f"{_NSP_ROOT}/results/orx_bc2dcc2b*_a4_resid")[0]
O = f"{_NSP_SCRATCH}/nc/gpu/a4"
V = [json.load(open(f)) for f in sorted(glob.glob(D + "/vols/*.json"))]
NB = V[0]["nb"]; bins = np.arange(-NB, NB + 1)
REG = dict(core=bins <= -4, band=(bins >= -3) & (bins <= 3), bg=bins >= 4)
Q = ("dz", "dxy", "dcp", "on_z", "on_xy", "off_xy")


def agg(vs, tag):
    per = {}
    for ds in sorted({v["ds"] for v in vs}):
        L = [v[tag] for v in vs if v["ds"] == ds]
        tot = {k: np.sum([np.asarray(x[k], float) for x in L], 0) for k in ("n",) + Q}
        per[ds] = tot
    curve = {k: np.nanmean([np.where(t["n"] > 0, t[k] / np.maximum(t["n"], 1), np.nan) for t in per.values()], 0) for k in Q}
    reg = {r: {k: float(np.mean([t[k][m].sum() / max(t["n"][m].sum(), 1) for t in per.values()])) for k in Q} for r, m in REG.items()}
    share = {k: {r: float(np.mean([t[k][m].sum() / max(t[k].sum(), 1e-9) for t in per.values()])) for r, m in REG.items()} for k in ("dz", "dxy", "dcp")}
    vox = {r: float(np.mean([t["n"][m].sum() / t["n"].sum() for t in per.values()])) for r, m in REG.items()}
    return dict(curve={k: v.round(4).tolist() for k, v in curve.items()}, region=reg, mass_share=share, voxel_share=vox, n_ds=len(per), n_vol=len(vs))


res = {"bins": bins.tolist(), "sign": "核内为负、核外为正（网络体素）"}
for tag in ("v1", "vr"):
    res[tag] = agg(V, tag)
    res[tag + "_aniso"] = agg([v for v in V if v["ratio"] >= 2], tag)
    res[tag + "_fine"] = agg([v for v in V if v["ratio"] < 2], tag)
an = [v for v in V if v["ratio"] >= 2]
bz = [np.asarray(v["v1"]["dz"])[REG["band"]].sum() / max(np.asarray(v["v1"]["n"])[REG["band"]].sum(), 1) for v in an]
pick = an[int(np.argmin(np.abs(np.array(bz) - np.median(bz))))]
res["fig_volume"] = f"{pick['ds']}__{pick['id']}"
json.dump(res, open(f"{O}/a4_interp.json", "w"), indent=1)
for tag in ("v1", "vr", "v1_aniso", "v1_fine"):
    r = res[tag]
    print(f"{tag}（{r['n_vol']} 体积 {r['n_ds']} 套）区域均值 |Δdz|/|Δxy|/|Δcp|：" + "  ".join(
        f"{k} {r['region'][k]['dz']:.2f}/{r['region'][k]['dxy']:.2f}/{r['region'][k]['dcp']:.2f}" for k in REG) +
          f"  | 开的 |dz|/|dxy| 边界带 {r['region']['band']['on_z']:.2f}/{r['region']['band']['on_xy']:.2f}" +
          "  | Σ|Δdz| 份额 " + " ".join(f"{k} {r['mass_share']['dz'][k]:.2f}" for k in REG) +
          "  Σ|Δxy| 份额 " + " ".join(f"{k} {r['mass_share']['dxy'][k]:.2f}" for k in REG) +
          "  体素份额 " + " ".join(f"{k} {r['voxel_share'][k]:.2f}" for k in REG))
c = res["v1"]["curve"]; i0 = NB
print("v1 曲线 d=-12..12 |Δdz|:", " ".join(f"{x:.2f}" for x in c["dz"][i0 - 12:i0 + 13]))
print("v1 曲线 d=-12..12 |Δxy|:", " ".join(f"{x:.2f}" for x in c["dxy"][i0 - 12:i0 + 13]))
print("v1 曲线 d=-12..12 |Δcp|:", " ".join(f"{x:.2f}" for x in c["dcp"][i0 - 12:i0 + 13]))
c = res["vr"]["curve"]
print("vr 曲线 d=-12..12 |Δdz|:", " ".join(f"{x:.2f}" for x in c["dz"][i0 - 12:i0 + 13]))
print("vr 曲线 d=-12..12 |Δxy|:", " ".join(f"{x:.2f}" for x in c["dxy"][i0 - 12:i0 + 13]))
print("示意图体积", res["fig_volume"])
# ---- 图
F = np.load(f"{D}/fig/{res['fig_volume']}.npz")
img, lab, on, off = (F[k + "_xz"].astype(np.float32) for k in ("img", "lab", "on", "off"))
img, lab = img[0], lab[0]
bd = np.zeros(lab.shape, bool)
bd[:-1] |= lab[:-1] != lab[1:]; bd[1:] |= lab[:-1] != lab[1:]; bd[:, :-1] |= lab[:, :-1] != lab[:, 1:]; bd[:, 1:] |= lab[:, :-1] != lab[:, 1:]
dz = np.abs(on[0] - off[0]); dxy = np.abs(on[2] - off[2]); dcp = on[3] - off[3]
fig = plt.figure(figsize=(14, 8.5))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1])
ax = fig.add_subplot(gs[0, :2]); x = bins
for tag, ls in (("v1", "-"), ("vr", "--")):
    cc = res[tag]["curve"]
    ax.plot(x, cc["dz"], ls, color="C3", label=f"|Δdz| ({'单视图' if tag == 'v1' else '视图规则'})")
    ax.plot(x, cc["dxy"], ls, color="C0", label=f"|Δ(dy,dx)| ({'单视图' if tag == 'v1' else '视图规则'})")
ax.plot(x, res["v1"]["curve"]["on_z"], ":", color="0.5", label="开：|dz|（单视图，参照）")
ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("到最近真值边界的有符号距离（网络体素，核内为负；核径约 30）"); ax.set_ylabel("流的差（5×单位向量）")
ax.set_title("A4：三维残差开/关的流场差（13 套宏平均，25 个测试体积）"); ax.legend(fontsize=8)
ax2 = fig.add_subplot(gs[0, 2:])
for tag, ls in (("v1", "-"), ("vr", "--")):
    ax2.plot(x, res[tag]["curve"]["dcp"], ls, color="C2", label=f"|Δcellprob| ({'单视图' if tag == 'v1' else '视图规则'})")
ax2.axvline(0, color="k", lw=0.8); ax2.set_xlabel("有符号距离（网络体素）"); ax2.set_ylabel("|Δ logit|"); ax2.legend(fontsize=8); ax2.set_title("cellprob 的差")
for j, (arr, ttl, cm) in enumerate(((img, "图像（xz 切面）+ 真值边界", "gray"), (dz, "|Δdz|", "magma"), (dxy, "|Δdx|", "magma"), (dcp, "Δcellprob（开 − 关）", "RdBu_r"))):
    a = fig.add_subplot(gs[1, j])
    if cm == "RdBu_r":
        vmax = np.percentile(np.abs(arr), 99); im = a.imshow(arr, cmap=cm, vmin=-vmax, vmax=vmax, aspect="auto")
    else:
        im = a.imshow(arr, cmap=cm, aspect="auto", vmax=np.percentile(arr, 99.5))
    yy, xx = np.where(bd); a.scatter(xx, yy, s=0.2, c="cyan" if j == 0 else "w", alpha=0.6)
    a.set_title(ttl, fontsize=9); a.set_xticks([]); a.set_yticks([]); a.set_ylabel("z（网络网格）" if j == 0 else "")
    if j: plt.colorbar(im, ax=a, fraction=0.046)
fig.suptitle(f"示意体积：{res['fig_volume']}（z/xy {pick['ratio']:.2f}，单视图，按中位数原则选）", fontsize=10)
fig.tight_layout(); fig.savefig(f"{O}/a4_fig.png", dpi=150)
print("PNG", f"{O}/a4_fig.png")
