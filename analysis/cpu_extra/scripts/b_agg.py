"""分析 B 的汇总（DESIGN_CPUX 第 2 节，只描述）：深度三档、局部信噪比三档（主：每套验证集三分位；敏感性：全部验证核合并三分位）、深度 × 信噪比九层。
每层每套 pooled（命中率 IoU ≥ 0.5 / ≥ 0.7、匹配对边界 IoU 均值），只纳入该层真值核 ≥ 20 的套，宏平均；种子先各自宏平均再取均值；
S1 − 二维同种子配对差的均值、SD 与套内分层体积自助 10000 次区间；边界 IoU 另给“两者都匹配”口径。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import glob
import json
import os
import sys
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/cpu_extra/scripts'))
from cpux_common import B, OUT, RUNMAP, SEEDS, models, np  # noqa: E402

TAGS = {nm: [f"gt_{nm}_{s}" for s in SEEDS] for nm in ("S1", "2D")}
DLAB = ("浅", "中", "深"); SLAB = ("低", "中", "高")


def main():
    bins = json.load(open(f"{OUT}/bins.json"))
    M, _ = models()
    V = {}
    for f in sorted(glob.glob(f"{OUT}/bvol/*.npz")):
        V[os.path.basename(f)[:-4]] = dict(np.load(f))
    assert len(V) == 227, len(V)
    # 核对：g_best ≥ 0.5 / 0.7 的计数与评测记录 tp5 / tp7
    bad = 0; maxiou = 0.0
    for key, v in V.items():
        x = tuple(key.split("__", 1)); maxiou = max(maxiou, float(v["iou_check_maxdiff"]))
        for nm in ("S1", "2D"):
            for i, tg in enumerate(TAGS[nm]):
                r = M[nm][i][x]
                bad += int((v[f"best_{tg}"] >= 0.5).sum() != r["tp5"]) + int((v[f"best_{tg}"] >= 0.7).sum() != r["tp7"])
    print(f"核对：tp5/tp7 与评测记录不一致 {bad} 处（227 × 10 × 2 项）；匹配对 IoU 与 g_best 最大差 {maxiou:.2e}", flush=True)
    # 逐核分层
    for key, v in V.items():
        ds = key.split("__")[0]
        v["dbin"] = np.digitize(v["depth"], bins["depth_edges"])
        e_own = bins["snr_per_ds"][ds][0]; e_pool = bins["snr_pooled"]
        v["sbin"] = np.where(np.isnan(v["snr"]), -1, np.digitize(np.nan_to_num(v["snr"], nan=0), e_own))
        v["sbin_pool"] = np.where(np.isnan(v["snr"]), -1, np.digitize(np.nan_to_num(v["snr"], nan=0), e_pool))
    systems = {"depth": [(DLAB[i], lambda v, i=i: v["dbin"] == i) for i in range(3)],
               "snr_own": [(SLAB[i], lambda v, i=i: v["sbin"] == i) for i in range(3)],
               "snr_pooled": [(SLAB[i], lambda v, i=i: v["sbin_pool"] == i) for i in range(3)],
               "depth_x_snr": [(f"{DLAB[i]}×{SLAB[j]}", lambda v, i=i, j=j: (v["dbin"] == i) & (v["sbin"] == j)) for i in range(3) for j in range(3)],
               "all": [("全部", lambda v: np.ones(len(v["ids"]), bool))]}
    keys = sorted(V); dss_all = sorted({k.split("__")[0] for k in keys})
    kd = {ds: [k for k in keys if k.startswith(ds + "__")] for ds in dss_all}
    rng = np.random.default_rng(0)
    idx = {ds: rng.integers(0, len(kd[ds]), (B, len(kd[ds]))) for ds in dss_all}
    nseed = len(SEEDS)
    res = {"bins": bins, "bins_md5": __import__("hashlib").md5(open(f"{OUT}/bins.json", "rb").read()).hexdigest(),
           "check": dict(tp_mismatch=bad, iou_maxdiff=maxiou), "snr_undefined_test": {}, "strata": {}}
    for ds in dss_all:
        res["snr_undefined_test"][ds] = int(sum(np.isnan(V[k]["snr"]).sum() for k in kd[ds]))
    for sysn, strata in systems.items():
        for lab, sel in strata:
            # 每套每体积的计数：n [vol]；h5,h7,bs,bn [模型, 种子, vol]；common bsS1,bs2D,bnC [种子, vol]
            C = {}
            for ds in dss_all:
                n = np.zeros(len(kd[ds])); H5 = np.zeros((2, nseed, len(kd[ds]))); H7 = H5.copy(); BS = H5.copy(); BN = H5.copy()
                CS = np.zeros((2, nseed, len(kd[ds]))); CN = np.zeros((nseed, len(kd[ds])))
                for j, k in enumerate(kd[ds]):
                    v = V[k]; m = sel(v); n[j] = m.sum()
                    for a, nm in enumerate(("S1", "2D")):
                        for s, tg in enumerate(TAGS[nm]):
                            b = v[f"best_{tg}"][m]; bi = v[f"biou_{tg}"][m]
                            H5[a, s, j] = (b >= 0.5).sum(); H7[a, s, j] = (b >= 0.7).sum()
                            BS[a, s, j] = np.nansum(bi); BN[a, s, j] = (~np.isnan(bi)).sum()
                    for s in range(nseed):
                        b1 = v[f"biou_{TAGS['S1'][s]}"][m]; b2 = v[f"biou_{TAGS['2D'][s]}"][m]; both = ~np.isnan(b1) & ~np.isnan(b2)
                        CS[0, s, j] = b1[both].sum(); CS[1, s, j] = b2[both].sum(); CN[s, j] = both.sum()
                C[ds] = dict(n=n, H5=H5, H7=H7, BS=BS, BN=BN, CS=CS, CN=CN)
            for tag, dsel in (("12", [d for d in dss_all if d != "mesospim"]), ("13", dss_all)):
                inc = [ds for ds in dsel if C[ds]["n"].sum() >= 20]
                ent = {"n_nuclei": int(sum(C[ds]["n"].sum() for ds in dsel)), "datasets_included": inc}
                if not inc:
                    res["strata"][f"{sysn}|{lab}|{tag}"] = ent; continue
                for met in ("hit5", "hit7", "biou", "biou_common"):
                    def stat(ds, ix=None):
                        c = C[ds]
                        g = (lambda a: a.sum(-1)) if ix is None else (lambda a: a[..., ix].sum(-1))   # [.., 种子] 或 [.., 种子, B]
                        with np.errstate(invalid="ignore", divide="ignore"):
                            if met == "hit5":
                                return g(c["H5"]) / g(c["n"])
                            if met == "hit7":
                                return g(c["H7"]) / g(c["n"])
                            if met == "biou":
                                return g(c["BS"]) / g(c["BN"])
                            return g(c["CS"]) / g(c["CN"])[None]
                    per = np.nanmean([stat(ds) for ds in inc], 0)                   # [模型, 种子]
                    diff = per[0] - per[1]
                    bs = np.nanmean([np.nanmean(stat(ds, idx[ds]), 1) for ds in inc], 0)   # [模型, B]
                    bd = bs[0] - bs[1]
                    ent[met] = dict(S1=float(np.nanmean(per[0])), two_d=float(np.nanmean(per[1])), diff_seeds=diff.tolist(),
                                    diff_mean=float(np.nanmean(diff)), diff_sd=float(np.nanstd(diff, ddof=1)),
                                    boot_ci=[float(np.nanpercentile(bd, 2.5)), float(np.nanpercentile(bd, 97.5))])
                res["strata"][f"{sysn}|{lab}|{tag}"] = ent
                if tag == "12" and sysn != "depth_x_snr":
                    h, b7, bi = ent["hit5"], ent["hit7"], ent["biou"]
                    print(f"{sysn:10s} {lab:3s} 12套 核 {ent['n_nuclei']:6d} 套 {len(inc):2d} | 命中@0.5 S1 {h['S1']:.3f} 二维 {h['two_d']:.3f} 差 {h['diff_mean']:+.4f} [{h['boot_ci'][0]:+.4f},{h['boot_ci'][1]:+.4f}]"
                          f" | @0.7 差 {b7['diff_mean']:+.4f} [{b7['boot_ci'][0]:+.4f},{b7['boot_ci'][1]:+.4f}] | 边界IoU S1 {bi['S1']:.3f} 二维 {bi['two_d']:.3f} 差 {bi['diff_mean']:+.4f} [{bi['boot_ci'][0]:+.4f},{bi['boot_ci'][1]:+.4f}]", flush=True)
    json.dump(res, open(f"{OUT}/b_results.json", "w"), ensure_ascii=False, indent=1)
    # 逐体积摘要
    pv = {}
    for k, v in V.items():
        e = dict(n_gt=int(len(v["ids"])), D=float(v["D"]), d_band=float(v["d_band"]), snr_undefined=int(np.isnan(v["snr"]).sum()))
        for nm in ("S1", "2D"):
            e[f"hit5_{nm}"] = float(np.mean([(v[f"best_{t}"] >= 0.5).mean() for t in TAGS[nm]])) if len(v["ids"]) else None
            e[f"hit7_{nm}"] = float(np.mean([(v[f"best_{t}"] >= 0.7).mean() for t in TAGS[nm]])) if len(v["ids"]) else None
        for i in range(3):
            m = v["dbin"] == i; e[f"n_depth_{DLAB[i]}"] = int(m.sum())
            for nm in ("S1", "2D"):
                e[f"hit5_depth_{DLAB[i]}_{nm}"] = float(np.mean([(v[f"best_{t}"][m] >= 0.5).mean() for t in TAGS[nm]])) if m.any() else None
            m = v["sbin"] == i; e[f"n_snr_{SLAB[i]}"] = int(m.sum())
            for nm in ("S1", "2D"):
                e[f"hit5_snr_{SLAB[i]}_{nm}"] = float(np.mean([(v[f"best_{t}"][m] >= 0.5).mean() for t in TAGS[nm]])) if m.any() else None
        pv[k] = e
    json.dump(pv, open(f"{OUT}/b_per_volume.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
