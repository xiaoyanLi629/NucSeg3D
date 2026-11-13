"""nc G1 补调参：三个对比模型（Tapenade StarDist、AnyStar、Cellos StarDist）在放宽网格上重选倍数与体积下限。设计见 $NUCSEG3D_SCRATCH/nc/gpu/DESIGN_G3.md 第 1 节。
推理、裁块、未定义区域、匹配与旧 bench_others.py（提交 d3aca0f）完全相同（prob_thresh 0.5、nms_thresh 0.3、384 裁块、max_vox 3e8，超限的体积记空预测）。
  --stage val：验证集 36 个体积（eval_val.pick_cases(k=4)，清单当前体素，mesospim 为 0.475）跑 --mults × --fracs，逐体积写 <pub>/val/<key>.json；
               结束后按原选法选（逐体积 F1@0.5 先按套平均、再 13 套平均，取最大，先小倍数后小下限），写 <pub>/selected.json；
               同时核对新结果与旧运行重合的格子（倍数、下限都在旧网格里，去 mesospim）是否逐行相同。
  --stage test：读 selected.json；若 (倍数, 下限) 与旧结果相同，写 kept_old.json 后退出，不做任何推理（保留旧结果）；
               否则只在测试集 45 个体积上按选中的倍数推理一次，写 <pub>/test/rows.json 与 summary.json（字段同旧横评），只报告。
用法：python bench_others2.py --model cellos --pub <目录> --stage val --mults 0.7 1 1.4 2 2.8 --fracs 0 .15 .25 .35 .45 .55 .65 .75 [--max_cases n]"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ANYSTAR = _nsp_os.environ.get('NUCSEG3D_ANYSTAR', '/path/to/AnyStar')
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import argparse, json, os, sys, time
from collections import defaultdict
os.environ.setdefault("OMP_NUM_THREADS", "12")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NS = f"{_NSP_SHARED}"
MODELS = {   # 与 bench_others.py 相同
    "anystar": dict(basedir=f"{_NSP_ANYSTAR}/repo/models", name="anystar-mix", target=(12.0, 12.0, 12.0)),
    "tapenade": dict(basedir=f"{NS}/data/tapenade/stardist_tapenade_model", name="tapenade_stardist", target=(15.0, 15.0, 15.0)),
    "cellos": dict(basedir=f"{NS}/models/cellos", name="stardist", target=(2.0, 18.0, 18.0)),
}
R = f"{_NSP_ROOT}/results/"
OLD = dict(tapenade="orx_1020_004144_others_tapenade", anystar="orx_1020_004452_others_anystar", cellos="orx_1020_004452_others_cellos")


def normalize(x, lo=1.0, hi=99.8):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / (b - a + 1e-8), 0, 1).astype(np.float32)


def macro(rows):
    by = defaultdict(list)
    for x in rows:
        by[x["ds"]].append(x["f1"])
    return float(np.mean([np.mean(v) for v in by.values()])) if by else 0.0


def load_tf(cfg):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(12); tf.config.threading.set_inter_op_parallelism_threads(2)
    for d in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(d, True)
    from stardist.models import StarDist3D
    return StarDist3D(None, name=cfg["name"], basedir=cfg["basedir"])


def run_case(model, cfg, r, split, mults, fracs, max_vox, evaluate, load_case, eq_diam_xy_px, drop_ignored):
    img, gt, box, ign = load_case(r, 384)
    if ign is not None:
        gt = gt.copy(); gt[ign] = 0
    if gt.max() == 0:
        return [], []
    vox = r["voxel"] or (1.0, 1.0, 1.0); d = eq_diam_xy_px(gt, vox); dz = d * vox[1] / vox[0]
    base = (cfg["target"][0] / dz, cfg["target"][1] / d, cfg["target"][2] / d)
    x = normalize(img.astype(np.float32)); rows, bad = [], []
    for m in mults:
        sc = tuple(b * m for b in base)
        if np.prod([s * n for s, n in zip(sc, img.shape)]) > max_vox:
            lab = np.zeros_like(gt); bad.append(dict(mult=m, why="max_vox"))
        else:
            nt = tuple(max(1, int(np.ceil(s * n / 160))) for s, n in zip(sc, img.shape))
            try:
                lab, _ = model.predict_instances(x, prob_thresh=0.5, nms_thresh=0.3, scale=sc, n_tiles=nt, show_tile_progress=False)
            except Exception as e:
                print("  失败", r["ds"], r["id"], m, repr(e)[:160], flush=True); lab = np.zeros_like(gt); bad.append(dict(mult=m, why=repr(e)[:160]))
        lab = drop_ignored(lab.astype(np.int32), ign)
        cnt = np.bincount(lab.ravel()); exp_vol = np.pi / 6 * d ** 3 * vox[1] / vox[0]
        for f in fracs:
            p = lab
            if f > 0:
                small = np.where(cnt < f * exp_vol)[0]; small = small[small > 0]
                p = lab.copy(); p[np.isin(p, small)] = 0
            ev = evaluate(gt, p)
            rows.append(dict(split=split, ds=r["ds"], id=r["id"], mult=m, frac=f, f1=ev["f1@0.5"], f1_7=ev["f1@0.7"], n_gt=ev["n_gt"], n_pred=ev["n_pred"]))
    return rows, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS); ap.add_argument("--pub", required=True); ap.add_argument("--out", default=None)
    ap.add_argument("--stage", required=True, choices=["val", "test"])
    ap.add_argument("--mults", type=float, nargs="+"); ap.add_argument("--fracs", type=float, nargs="+")
    ap.add_argument("--max_vox", type=float, default=3e8); ap.add_argument("--max_cases", type=int, default=0, help="冒烟用")
    ap.add_argument("--no_select", action="store_true", help="冒烟用：不做选择")
    a = ap.parse_args()
    cfg = MODELS[a.model]; os.makedirs(a.pub, exist_ok=True)
    old = json.load(open(R + OLD[a.model] + "/summary.json"))
    sj = os.path.join(HERE, "..", "val_split.json")
    if a.stage == "test":
        sel = json.load(open(f"{a.pub}/selected.json"))
        if sel["mult"] == old["mult"] and sel["frac"] == old["frac"]:
            json.dump(dict(kept_old=OLD[a.model], mult=old["mult"], frac=old["frac"]), open(f"{a.pub}/kept_old.json", "w"))
            print(f"SUMMARY {a.model}：选出的 ({sel['mult']:g}, {sel['frac']:g}) 与旧结果相同，保留旧结果，不跑测试集", flush=True)
            return
    from metrics import evaluate
    from eval_cpsam import load_case, eq_diam_xy_px, drop_ignored
    from eval_val import pick_cases
    model = load_tf(cfg)
    if a.stage == "val":
        os.makedirs(f"{a.pub}/val", exist_ok=True)
        cases = pick_cases("val", sj, 4)
        assert len(cases) == 36 or a.max_cases, len(cases)
        cases = cases[:a.max_cases] if a.max_cases else cases
        rows, bad = [], {}; t0 = time.time()
        for i, r in enumerate(cases):
            k = f"{r['ds']}__{r['id']}"; p = f"{a.pub}/val/{k}.json"
            if os.path.exists(p):
                z = json.load(open(p))
            else:
                rs, bd = run_case(model, cfg, r, "val", a.mults, a.fracs, a.max_vox, evaluate, load_case, eq_diam_xy_px, drop_ignored)
                z = dict(rows=rs, bad=bd); json.dump(z, open(p + ".part", "w")); os.replace(p + ".part", p)
            rows += z["rows"]
            if z["bad"]:
                bad[k] = z["bad"]
            print(f"  {i + 1}/{len(cases)} {k[:50]}  [{(time.time() - t0) / 60:.1f} min] 空预测 {len(z['bad'])}", flush=True)
        json.dump(rows, open(f"{a.pub}/val_rows.json", "w"))
        od = {(x["ds"], x["id"], x["mult"], x["frac"]): x for x in json.load(open(R + OLD[a.model] + "/rows.json")) if x["split"] == "val"}
        ov, same = [], 0
        for x in rows:
            q = od.get((x["ds"], x["id"], x["mult"], x["frac"]))
            if q is not None and x["ds"] != "mesospim":
                ok = abs(q["f1"] - x["f1"]) < 1e-9 and q["n_pred"] == x["n_pred"]
                same += ok; ov += [] if ok else [(x["ds"], x["id"], x["mult"], x["frac"], q["f1"], x["f1"])]
        print(f"REPRO 与旧运行重合的格子：相同 {same}，不同 {len(ov)}；样例 {ov[:4]}", flush=True)
        if a.no_select:
            return
        grid = {(m, f): macro([x for x in rows if x["mult"] == m and x["frac"] == f]) for m in a.mults for f in a.fracs}
        grid12 = {(m, f): macro([x for x in rows if x["mult"] == m and x["frac"] == f and x["ds"] != "mesospim"]) for m in a.mults for f in a.fracs}
        bm, bf = max(grid, key=grid.get); bm12, bf12 = max(grid12, key=grid12.get)
        out = dict(model=a.model, mult=bm, frac=bf, val_f1_05=grid[(bm, bf)], sel_12sets=dict(mult=bm12, frac=bf12), old=dict(mult=old["mult"], frac=old["frac"]),
                   at_edge=dict(mult=bm in (min(a.mults), max(a.mults)), frac=bf == max(a.fracs)), grid={f"{m:g}/{f:g}": v for (m, f), v in grid.items()},
                   empty_preds=bad, repro=dict(same=same, diff=len(ov)))
        json.dump(out, open(f"{a.pub}/selected.json", "w"), indent=1)
        print(f"SUMMARY {a.model}：验证集选 倍数 {bm:g} / 体积下限 {bf:g}，val {grid[(bm, bf)]:.4f}（旧 {old['mult']:g}/{old['frac']:g}）；12 套选 {bm12:g}/{bf12:g}", flush=True)
    else:
        cases = [r for r in pick_cases("test", sj, 4) if r["ds"] != "bbbc032"]
        assert len(cases) == 45 or a.max_cases, len(cases)
        cases = cases[:a.max_cases] if a.max_cases else cases
        rows, bad = [], {}
        for i, r in enumerate(cases):
            rs, bd = run_case(model, cfg, r, "test", [sel["mult"]], [sel["frac"]], a.max_vox, evaluate, load_case, eq_diam_xy_px, drop_ignored)
            rows += rs
            if bd:
                bad[f"{r['ds']}__{r['id']}"] = bd
            print(f"  {i + 1}/{len(cases)} {r['ds']} {r['id'][:40]}", flush=True)
        os.makedirs(f"{a.pub}/test", exist_ok=True)
        json.dump(rows, open(f"{a.pub}/test/rows.json", "w"))
        json.dump(dict(model=a.model, mult=sel["mult"], frac=sel["frac"], empty_preds=bad, n=len(cases)), open(f"{a.pub}/test/summary.json", "w"), indent=1)
        print(f"SUMMARY {a.model}：测试集按 {sel['mult']:g}/{sel['frac']:g} 跑完 {len(cases)} 个体积（只报告）", flush=True)


if __name__ == "__main__":
    main()
