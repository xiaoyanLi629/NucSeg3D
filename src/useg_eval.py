"""nc G3 第一步（env_torch，GPU）+ 评测：二维 Cellpose-SAM（原版 / 二维种子 0）的逐切片输出 → u-Segment3D 共识重建 → 与 eval_gt 相同的匹配与汇总。
设计见 $NUCSEG3D_SCRATCH/nc/gpu/DESIGN_G3.md 第 2 节。选样、裁块、未定义区域、直径（真值 / --diam_from 的 d_used）、z/xy ≥ 2 的事先插值都照 eval_gt（eval_gt.main 不改，只换 run_volume）。
逐切片输出的取法：与 eval_gt 同样调用 model.eval(do_3D=True)，临时替换 cellpose.models.run_3D，抓住 core.run_3D 三个方向各自的 run_net 输出（网络网格：各向同性、核径约 30），
所以缩放、归一化、分块与 eval_gt 的二维推理逐字相同，只是不把三个方向相加。
共识：子进程调用 u-Segment3D 环境（useg_cons.py）。标签最近邻取回原网格（bl_eval.back_index，与 eval_gt 取回原层位同一公式），再 drop_ignored、体积下限 {0, 0.15, 0.25}、eval_gt.taxonomy。
thr 字段存变体：v0..v6 记为 0, −1, …, −6（所以 main 格 cp-2_vol0.15 = v2，不要按 cellprob 阈值理解，选参读 rows.json 的 variant 字段）。
--useg_resume <结果目录>[,<结果目录>]：先把这些目录 vols/ 里已完成的逐体积结果拷进本次的 vols/（同一模型、同一代码失败后续跑用），已有的不覆盖。
临时文件（每个体积的三方向输出与共识标签，单个体积可达十余 GB）写在共享盘 $NUCSEG3D_SHARED/nc_gpu/tmp/<运行 ID>/，体积做完即删，运行结束删整个目录。
--useg_sel <json> --useg_key base|d2 [--useg_field variant|stage1]：只跑一个变体（stage1 = 第一轮 v0–v5 选出的变体加流一致性过滤，即编号 +6），结果同时记成 thr=−2 与 thr=0 两份（只为满足 eval_gt.main 的打印，汇总读 cp-2_vol0.15）。
用法：python useg_eval.py --out <目录> --model base --workers 2 --box 384 --vox_override mesospim=5/10.52/10.52 [--split test] [--max_per_ds 4 --diam_from <目录>] [--useg_sel <json>] [--useg_key model]"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import json, os, shutil, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_gt as E  # noqa: E402

_M = None
UPY = f"{_NSP_SCRATCH}/bl_envs/env_useg/bin/python"
# 临时文件放共享盘（hostB 本地盘只剩十几 GB，2026-11-06 19:16 G3v1_d2 因此失败）；每个运行一个子目录，结束后整个删掉
TMP = os.environ.get("USEG_TMP") or f"{_NSP_SHARED}/nc_gpu/tmp/{os.environ.get('ORX_RUN_ID') or 'local_' + str(os.getpid())}"
HOLD = {}


class _Stop(Exception):
    pass


def capture_3D(net, imgs, batch_size=8, augment=False, tile_overlap=0.1, bsize=224, net_ortho=None, progress=None):
    """core.run_3D 的副本：不相加，把三个方向的输出存下来后中止 eval。view 顺序 xy、xz、yz。"""
    from cellpose.core import run_net
    pm = [(0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3)]; ipm = [(0, 1, 2), (1, 0, 2), (1, 2, 0)]
    shape = imgs.shape[:-1]
    probs = np.zeros((3, *shape), np.float32); flows = np.zeros((3, 2, *shape), np.float32)
    for p in range(3):
        y, _ = run_net(net, imgs.transpose(pm[p]), batch_size=batch_size, augment=augment, bsize=bsize, tile_overlap=tile_overlap, rsz=None)
        probs[p] = y[..., -1].transpose(ipm[p])
        for j in range(2):
            flows[p, j] = y[..., j].transpose(ipm[p])
        del y
    HOLD["probs"], HOLD["flows"] = probs, flows
    raise _Stop


def _init(spec, opts=None):
    global _M
    import logging
    logging.disable(logging.WARNING)
    _M = E.load_model(spec, opts)
    import cellpose.models as cm
    cm.run_3D = capture_3D


def back_index(n_orig, n_iso):
    if n_orig == 1:
        return np.zeros(1, int)
    return np.clip(np.round(np.arange(n_orig) * (n_iso - 1) / (n_orig - 1)).astype(int), 0, n_iso - 1)


def run_volume(job):
    r, a = job
    from eval_cpsam import load_case, eq_diam_xy_px, drop_ignored
    from eval_val import zinterp
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
    vox = r["voxel"] or (1.0, 1.0, 1.0); ratio = vox[0] / vox[1]; d = d_gt = eq_diam_xy_px(gt, vox)
    r4 = bool(a.get("dmap")) or a.get("views1_ds") is not None
    if a.get("dmap"):
        d = a["dmap"][key]
    if ratio >= 2:
        Zu = int(round(img.shape[0] * ratio))
        im = img.astype(np.float32); lo, hi = np.percentile(im, [1, 99.8]); im = (im - lo) / max(hi - lo, 1e-6)
        img_in, an = zinterp(im, Zu), 1.0
    else:
        img_in, an = img, ratio
    try:
        _M.eval(img_in, do_3D=True, z_axis=0, diameter=d, anisotropy=an, flow3D_smooth=1.0, batch_size=a["batch"], compute_masks=False)
        raise RuntimeError("capture_3D 没有被调用")
    except _Stop:
        pass
    t_net = time.time() - t0
    sel = json.loads(os.environ.get("USEG_VARS", "null"))
    variants = sel if sel is not None else list(range(7))
    os.makedirs(TMP, exist_ok=True)
    td = tempfile.mkdtemp(dir=TMP)
    try:
        np.save(f"{td}/probs.npy", HOLD.pop("probs")); np.save(f"{td}/flows.npy", HOLD.pop("flows"))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", MPLBACKEND="Agg")
        p = subprocess.run([UPY, os.path.join(HERE, "useg_cons.py"), "--in", td, "--variants", *map(str, variants)], env=env,
                           stdout=open(f"{td}/cons.log", "w"), stderr=subprocess.STDOUT)
        if p.returncode != 0:
            raise RuntimeError("useg_cons 失败 " + key + ": " + open(f"{td}/cons.log").read()[-1200:])
        st = json.load(open(f"{td}/stats.json"))
        G = E.GTInfo(gt, vox); exp_vol = np.pi / 6 * d ** 3 * vox[1] / vox[0]; rows = []
        for v in variants:
            L = np.load(f"{td}/v{v}.npy")
            iz, iy, ix = (back_index(gt.shape[i], L.shape[i]) for i in range(3))
            m = drop_ignored(L[np.ix_(iz, iy, ix)].astype(np.int32), ign)
            if a.get("save_masks"):
                os.makedirs(f"{a['out']}/masks", exist_ok=True)
                np.savez_compressed(f"{a['out']}/masks/{key}__v{v}.npz", pred=m, gt=gt.astype(np.int32), vox=np.asarray(vox, np.float64), d=np.float64(d))
            cnt = np.bincount(m.ravel())
            tags = (-2.0, 0.0) if sel is not None else (-float(v),)
            for f in E.FRACS:
                q = m
                if f > 0:
                    small = np.where(cnt < f * exp_vol)[0]; small = small[small > 0]
                    q = m.copy(); q[np.isin(q, small)] = 0
                tx = E.taxonomy(G, q); den = max(tx["n_gt"] + tx["n_pred"], 1)
                for t in tags:
                    rows.append(dict(ds=r["ds"], id=r["id"], thr=t, frac=f, variant=v, d_gt=d_gt, ratio=ratio, iso=ratio >= 2, shape=list(gt.shape),
                                     f1=2 * tx["tp5"] / den, f1_7=2 * tx["tp7"] / den, **tx, cons=st[str(v)], **(dict(d_used=d) if r4 else {})))
    finally:
        shutil.rmtree(td, ignore_errors=True)
    os.makedirs(os.path.dirname(vp), exist_ok=True)
    for x in rows:
        x["sec_net"] = round(t_net, 1); x["sec_total"] = round(time.time() - t0, 1)
    json.dump(rows, open(vp + ".part", "w")); os.replace(vp + ".part", vp)
    return rows


def main():
    argv = sys.argv[1:]
    if "--useg_sel" in argv:
        i = argv.index("--useg_sel"); sf = argv[i + 1]; j = argv.index("--useg_key"); k = argv[j + 1]
        fld = "variant"
        if "--useg_field" in argv:
            q = argv.index("--useg_field"); fld = argv[q + 1]; argv = argv[:q] + argv[q + 2:]
            i = argv.index("--useg_sel"); j = argv.index("--useg_key")
        v = json.load(open(sf))[k][fld]
        os.environ["USEG_VARS"] = json.dumps([v]); print("USEG 选中变体", k, v, flush=True)
        argv = [x for n, x in enumerate(argv) if n not in (i, i + 1, j, j + 1)]
    if "--useg_vars" in argv:                                                     # 冒烟用：--useg_vars 0,5
        i = argv.index("--useg_vars"); os.environ["USEG_VARS"] = json.dumps([int(x) for x in argv[i + 1].split(",")]); argv = argv[:i] + argv[i + 2:]
    if "--useg_resume" in argv:
        i = argv.index("--useg_resume"); src = argv[i + 1].split(","); argv = argv[:i] + argv[i + 2:]
        out = argv[argv.index("--out") + 1]; os.makedirs(f"{out}/vols", exist_ok=True); n = 0
        for s in src:
            for f in sorted(os.listdir(f"{s}/vols")):
                if f.endswith(".json") and not os.path.exists(f"{out}/vols/{f}"):
                    shutil.copy(f"{s}/vols/{f}", f"{out}/vols/{f}"); n += 1
        print("USEG 续用已完成的体积", n, "个，来自", src, flush=True)
    E.run_volume = run_volume; E._init = _init
    sys.argv = [sys.argv[0]] + argv
    os.environ["USEG_TMP"] = TMP                                                  # 让 spawn 出来的工作进程用同一个目录
    try:
        E.main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
