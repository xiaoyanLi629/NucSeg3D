"""G2 基线（nnU-Net、StarDist-3D）的评测入口：与 eval_gt.py 同一套样本、裁块、匹配与指标。

每个体积：eval_cpsam.load_case 读原始各向异性体积（xy 超过 --box 取前景最多的窗口，与 eval_gt 相同）→
  直径 d = 真值 xy 等效球直径中位数（gt 协议），或 --diam_from 目录 rows.json 的 d_used（完整流程，与我们模型用同一个直径）→
  按 iso30 的建库规则缩放到“核径 30 像素、各向同性”的网格：f_xy = 30/d，f_z = f_xy·vz/vxy，图像 1–99 百分位归一化后线性插值 →
  基线模型在该网格上出实例（每个“变体”一份，见各方法的 Predictor）→ 按最近邻取回原始网格 → drop_ignored →
  体积下限 {0, 0.15, 0.25}×π/6·d³·vxy/vz → eval_gt.taxonomy（匈牙利匹配，F1@0.5/0.7 与误差分类）。
逐体积结果的字段与 eval_gt 相同；thr 字段存“变体编号”（nnU-Net：1 = argmax 内部类连通域做种子，2/3/4 = P(内部) > 0.6/0.75/0.9 做种子；
StarDist：1 = 验证集上 optimize_thresholds 得到的阈值，2 = 默认阈值 0.5/0.4）。汇总用 eval_gt.summarize（按数据集 pooled 再宏平均）。
输出 <out>/vols/*.json、rows.json、summary.json；--save_masks 时 <out>/masks/<ds>__<id>.npz（pred_v<k>（体积下限前）、gt、vox、d）。
用法：python bl_eval.py --method nnunet --model <nnUNet 结果目录/fold> --out <dir> [--split val|test] [--max_per_ds 4] [--diam_from <dir>]
      python bl_eval.py --method stardist --model <stardist 模型目录> --out <dir> ..."""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import eval_gt as EG  # noqa: E402

TARGET_D = 30.0


def iso_factors(d, vox):
    fxy = TARGET_D / d
    return (fxy * vox[0] / vox[1], fxy, fxy)


def to_iso(img, f3):
    x = img.astype(np.float32)
    step = max(1, x.size // 20_000_000)
    p1, p99 = np.percentile(x.ravel()[::step], [1, 99])
    x = (x - p1) / max(p99 - p1, 1e-6)
    return ndi.zoom(x, f3, order=1)


def back_index(n_orig, n_iso):
    if n_orig == 1:
        return np.zeros(1, int)
    return np.clip(np.round(np.arange(n_orig) * (n_iso - 1) / (n_orig - 1)).astype(int), 0, n_iso - 1)


def make_predictor(a):
    if a["method"] == "nnunet":
        from nnunet_pred import NNUNetPredictor
        return NNUNetPredictor(a["model"], **a.get("popts", {}))
    from stardist_pred import StarDistPredictor
    return StarDistPredictor(a["model"], **a.get("popts", {}))


_P = None


def _init(a, gpu):
    global _P
    if gpu is not None:                                                  # 不给 --gpus 时沿用外部设的 CUDA_VISIBLE_DEVICES，绝不改写
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    _P = make_predictor(a)


def run_volume(job):
    r, a = job
    from eval_cpsam import load_case, eq_diam_xy_px, drop_ignored
    key = f"{r['ds']}__{r['id']}"
    vp = f"{a['out']}/vols/{key}.json"
    if os.path.exists(vp):
        return json.load(open(vp))
    t0 = time.time()
    img, gt, box, ign = load_case(r, a["box"] or 10**9)
    if ign is not None:
        gt = gt.copy(); gt[ign] = 0
    if gt.max() == 0:
        return []
    vox = r["voxel"] or (1.0, 1.0, 1.0)
    d = d_gt = eq_diam_xy_px(gt, vox)
    if a.get("dmap"):
        d = a["dmap"][key]
    f3 = iso_factors(d, vox)
    x = to_iso(img, f3)
    preds = _P.predict(x)                                                # {变体编号: 实例标签（iso 网格）}
    t_net = time.time() - t0
    ii = [back_index(n, m) for n, m in zip(img.shape, x.shape)]
    G = EG.GTInfo(gt, vox)
    exp_vol = np.pi / 6 * d ** 3 * vox[1] / vox[0]
    rows, keep = [], {}
    for v, lab in sorted(preds.items()):
        m = lab[np.ix_(*ii)].astype(np.int32)
        m = drop_ignored(m, ign)
        keep[f"pred_v{v}"] = m
        cnt = np.bincount(m.ravel())
        for f in EG.FRACS:
            p = m
            if f > 0:
                small = np.where(cnt < f * exp_vol)[0]; small = small[small > 0]
                p = m.copy(); p[np.isin(p, small)] = 0
            tx = EG.taxonomy(G, p)
            den = max(tx["n_gt"] + tx["n_pred"], 1)
            rows.append(dict(ds=r["ds"], id=r["id"], thr=float(v), frac=f, d_gt=d_gt, d_used=d, ratio=vox[0] / vox[1], iso=True,
                             shape=list(gt.shape), iso_shape=list(x.shape), f1=2 * tx["tp5"] / den, f1_7=2 * tx["tp7"] / den, **tx))
    if a.get("save_masks"):
        os.makedirs(f"{a['out']}/masks", exist_ok=True)
        np.savez_compressed(f"{a['out']}/masks/{key}.npz", gt=gt.astype(np.int32), vox=np.asarray(vox, np.float64), d=np.float64(d), **keep)
    tiles = getattr(_P, "last_tiles", None)                              # StarDist：本体积实际用的 n_tiles（显存不足时会加倍）
    for x_ in rows:
        x_["sec_net"] = round(t_net, 1); x_["sec_total"] = round(time.time() - t0, 1)
        if tiles is not None:
            x_["n_tiles"] = [int(t) for t in tiles]
    os.makedirs(os.path.dirname(vp), exist_ok=True)
    json.dump(rows, open(vp + ".part", "w")); os.replace(vp + ".part", vp)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["nnunet", "stardist"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--split_json", default=os.path.join(HERE, "val_split.json"))
    ap.add_argument("--max_per_ds", type=int, default=0)
    ap.add_argument("--diam_from", default=None)
    ap.add_argument("--only", nargs="*"); ap.add_argument("--ids", nargs="*")
    ap.add_argument("--box", type=int, default=384)
    ap.add_argument("--gpus", nargs="+", default=[None], help="每个元素起一个进程（可重复同一张卡）；不给时用外部的 CUDA_VISIBLE_DEVICES")
    ap.add_argument("--save_masks", action="store_true")
    ap.add_argument("--popts", default="{}", help="传给 Predictor 的 JSON 参数")
    ap.add_argument("--limit", type=int, default=0, help="冒烟测试：只跑前 N 个体积")
    ap.add_argument("--vox_override", nargs="*", default=None, help="ds=vz/vy/vx 覆盖清单体素（mesospim 一律用源数据体素 5/10.52/10.52）")
    a = ap.parse_args()
    cases = EG.pick_cases(a.split, a.split_json, a.only, a.ids, a.max_per_ds)
    if a.vox_override:                                                   # 与 eval_gt 相同
        ov = {k: tuple(float(x) for x in v.split("/")) for k, v in (t.split("=") for t in a.vox_override)}
        for r in cases:
            if r["ds"] in ov:
                r["voxel"] = ov[r["ds"]]
        print(f"体素覆盖 {ov}", flush=True)
    if a.limit:
        cases = cases[:: max(1, len(cases) // a.limit)][: a.limit]
    cfg = dict(method=a.method, model=a.model, out=a.out, box=a.box, save_masks=a.save_masks, popts=json.loads(a.popts))
    if a.diam_from:
        cfg["dmap"] = {f"{x['ds']}__{x['id']}": x["d_used"] for x in json.load(open(f"{a.diam_from}/rows.json"))}
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(vars(a), n_cases=len(cases)), open(f"{a.out}/args.json", "w"), indent=1)
    box = a.box or 10**9
    cases = sorted(cases, key=lambda r: -r["z"] * min(r["y"], box) * min(r["x"], box) * max(r["voxel"][0] / r["voxel"][1] if r["voxel"] else 1, 1))
    print(f"{len(cases)} volumes, method={a.method}, model={a.model}", flush=True)
    rows, t0 = [], time.time()
    if len(a.gpus) == 1:
        _init(cfg, a.gpus[0])
        for k, r in enumerate(cases):
            rr = run_volume((r, cfg)); rows += rr
            _log(k, len(cases), r, rr, t0)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        q = ctx.Queue(); res = ctx.Queue()
        for r in cases:
            q.put(r)
        ps = [ctx.Process(target=_worker, args=(cfg, g, q, res)) for g in a.gpus]
        for _ in ps:
            q.put(None)
        for p in ps:
            p.start()
        for k in range(len(cases)):
            r, rr = res.get(); rows += rr
            _log(k, len(cases), r, rr, t0)
        for p in ps:
            p.join()
    json.dump(rows, open(f"{a.out}/rows.json", "w"))
    S = EG.summarize(rows)
    json.dump(S, open(f"{a.out}/summary.json", "w"), indent=1)
    for k, c in S["cells"].items():
        print(f"  v{c['thr']:g} vol{c['frac']:g}: macro F1@0.5={c['macro_f1_05']:.4f} F1@0.7={c['macro_f1_07']:.4f}")
    b = S["cells"][S["best"]]
    print(f"SUMMARY method={a.method} model={a.model} {a.split} n={S['n_volumes']}/{S['n_datasets']}ds best {S['best']}: "
          f"F1@0.5={b['macro_f1_05']:.4f} F1@0.7={b['macro_f1_07']:.4f} [{(time.time() - t0) / 60:.1f} min]", flush=True)


def _worker(cfg, gpu, q, res):
    _init(cfg, gpu)
    while True:
        r = q.get()
        if r is None:
            break
        res.put((r, run_volume((r, cfg))))


def _log(k, n, r, rr, t0):
    m = [x for x in rr if x["frac"] == 0.15]
    s = "  ".join(f"v{x['thr']:g} {x['f1']:.3f}/{x['f1_7']:.3f}" for x in m)
    sec = rr[0]["sec_total"] if rr else 0
    print(f"[{k + 1}/{n}] {r['ds']}/{r['id']} vol0.15 {s}  {sec:.0f}s  [{(time.time() - t0) / 60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
