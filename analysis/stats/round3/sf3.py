"""SF3（口径 round3/SF3_DESIGN.md，md5 314dc567…）：S1 − nnU-Net 的数据集随机效应分析，方法直接调用 nc_stats 的 s1（MixedLM REML + 整簇自助 10000）与 s2（DL + HK）。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import sys
from multiprocessing import Pool
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'analysis/stats/round3'))
from r3common import R3, SEEDS, dump, json, ncio, nc_stats, nnunet, np, vf1  # noqa: E402


def main():
    out = {"design": "round3/SF3_DESIGN.md", "seeds": SEEDS, "results": {}}
    with Pool(32) as pool:
        for mode in ("gt", "fp"):
            M, _ = ncio.load_mode(mode, SEEDS)
            Bv, bdir, sel = nnunet(mode)
            ref = json.load(open(f"{_NSP_SHARED}/baselines/results/eval/paired_stats_{mode}.json", encoding="utf-8"))
            common = sorted(set(Bv).intersection(*[set(R) for R in M["S1"]]))
            for sub in ("13", "12"):
                keys = [x for x in common if sub == "13" or x[0] != "mesospim"]
                for k, lab in ncio.METRICS:
                    rows = [{"ds": x[0], "id": x[1], f"S1_{k}": float(np.mean([vf1(R[x], k) for R in M["S1"]])), f"nn_{k}": vf1(Bv[x], k)} for x in keys]
                    eff = nc_stats.per_ds_effects(rows, "nn", k)
                    s1 = nc_stats.s1(rows, "nn", k, pool, 10000, seed=7)
                    s2 = nc_stats.s2(eff)
                    e = ref[f"nnU-Net|{sub}|{lab}"]
                    out["results"][f"{mode}|{sub}|{lab}"] = dict(n_vol=len(keys), n_ds=len(eff), mixed=s1["main"], cluster_boot=s1["cluster_boot"],
                                                                 meta=s2, per_ds=eff,
                                                                 existing=dict(seed_mean=e["mean"], seed_sd=e["sd"], confirm=e["confirm"], boot_d=e["boot_d"], boot_ci=e["boot_ci"]))
                    m = s1["main"]
                    print(f"{mode} {sub:2s} {lab}  现有 种子 {e['mean']:+.4f}±{e['sd']:.4f} {e['confirm']} 自助 {e['boot_d']:+.4f} [{e['boot_ci'][0]:+.4f},{e['boot_ci'][1]:+.4f}] | "
                          f"MixedLM {m['beta']:+.4f} [{m['ci'][0]:+.4f},{m['ci'][1]:+.4f}] 簇自助 [{s1['cluster_boot']['ci'][0]:+.4f},{s1['cluster_boot']['ci'][1]:+.4f}] | "
                          f"HK {s2['mu']:+.4f} [{s2['ci_hk'][0]:+.4f},{s2['ci_hk'][1]:+.4f}] I²={s2['I2']:.2f} k={s2['k']}", flush=True)
                effs = {f"S1 − nnU-Net {lab}": out["results"][f"{mode}|{sub}|{lab}"]["per_ds"] for _, lab in ncio.METRICS}
                s2s = {f"S1 − nnU-Net {lab}": out["results"][f"{mode}|{sub}|{lab}"]["meta"] for _, lab in ncio.METRICS}
                nc_stats.FIG = f"{R3}/sf3"
                nc_stats.forest(effs, s2s, f"SF3｜{'gt 协议' if mode == 'gt' else '完整流程'}｜{sub} 套｜五种子", f"{R3}/sf3/forest_{mode}_{sub}.png")
    dump(out, f"{R3}/sf3/results.json")


if __name__ == "__main__":
    main()
