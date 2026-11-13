"""OpenResearch 的统一入口：读 config.json，按需训练，再在验证集和测试集上评测并打印指标。

每个实验节点只改 config.json（或 src/ 里的代码），运行命令始终是 `bash run.sh`。
数据、原版权重和训练好的权重都在共享盘 $NUCSEG3D_ROOT 上，本仓库只带代码。

config.json 字段：
  model        评测用的权重名（models/models/ 下），或 "base" 表示原版 cpsam_v2
  train        null 表示只评测；否则为 {"name": "orx_xxx", "args": [...train_cpsam2.py 的参数], "exclude_val": true,
               "extra_tar": "dimnuc_extra.tar"（可选，共享盘 slices/ 下的额外切片包）}
               name 必须以 orx_ 开头；exclude_val 为 true 时去掉 val_split.json 里验证体积的切片；
               "slices": "v5_2d" / "v5_3d"（可选）改用共享盘 slices/<名>.tar 里的切片集，并读其中预计算的 _flows.tif 作训练目标
  thrs         cellprob 阈值网格（默认 [0, -1, -2]）
  fracs        体积下限比例网格（默认 [0, 0.15, 0.25, 0.35]）
  val_per_ds   验证集每套数据至多几个样本（默认 4）
  test_per_ds  测试集每套数据至多几个样本（默认 4；0 表示全部）
  no_eval      r9：为真时训练完就结束，不跑 eval_val（交叉验证的二维训练；留出清单用 train.args 里的 --exclude_json 传，exclude_val 设 false）
选择规则（事先定好）：在验证集上取宏平均 F1@0.5 最高的（阈值, 体积下限）组合；测试集只报告这个组合和默认组合（cp0, vol0.25），不参与选择。

真三维模型线（约定见 $NUCSEG3D_SCRATCH/true3d_spec.md）用 "mode" 字段，旧配置没有 mode，行为不变：
  {"mode": "t0", "eval_models": ["orx_v5_2d_s0", ...]}
      基础节点：确保各向同性体积库 iso30 存在（本机 $NUCSEG3D_SCRATCH/cpft_iso30；没有就先从共享盘 iso30/iso30.tar 解包，
      共享盘也没有就用 src/make_iso_vols.py 现建再打包；flock 加锁、建完写 .done 标记，并发的 run 不会重复建），
      建库在后台进行，同时用 src/eval_gt.py（真值直径协议，88 个验证体积）逐个评测 eval_models，每个模型打印一行 SUMMARY。
  {"mode": "gt", "model": <权重名 | 含 model3d.json 的目录>, "train3d": null | {"name": "orx_t3d_xxx", "args": [...]}}
      三维节点通用入口：有 train3d 时先确保 iso30 存在，再运行
        src/train3d.py --iso <iso30 目录> --out <共享盘 models/models/<name>> <args...>
      （train3d.py 由三维节点提供；输出目录里须有 model3d.json，训练完成后写 <out>.orx_done，已存在则跳过训练），
      然后用 eval_gt.py 评测该模型，结果在 results/orx_<runid>_gtval/。
  这两种模式的可选字段：eval_workers（同卡并行进程数，默认 3）、box（xy 裁块，默认 384）、iso_workers（建库进程数，默认 6）。
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json, os, subprocess, sys, time

import numpy as np

ROOT = f"{_NSP_ROOT}"
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
SLICE_DIRS = [f"{_NSP_SCRATCH}/cpft_slices_v3/train", f"{_NSP_SCRATCH}/cpft/v3/train"]


def sh(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=SRC)


def slices_dir():
    for d in SLICE_DIRS:
        if os.path.isdir(d) and sum(f.endswith("_flows.tif") for f in os.listdir(d)) > 6000:
            return d
    d = SLICE_DIRS[1]
    os.makedirs(d, exist_ok=True)
    print("本地没有训练切片，从共享盘解包到", d, flush=True)
    for t in ("v3_train.tar", "v3_flows.tar"):
        subprocess.run(["tar", "xf", f"{ROOT}/slices/{t}", "-C", d], check=True)
    return d


def slices_set(nm):
    """共享盘 slices/<nm>.tar 里的切片集，第一次用时解包到本地盘（解包完成后写标记文件，避免读到半截）。"""
    base = f"{_NSP_SCRATCH}/cpft_v5"
    d, done = f"{base}/{nm}", f"{base}/{nm}.unpacked"
    if not os.path.exists(done):
        import fcntl
        os.makedirs(base, exist_ok=True)
        with open(f"{base}/{nm}.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if not os.path.exists(done):
                print("从共享盘解包切片集", nm, flush=True)
                subprocess.run(["rm", "-rf", d], check=True)
                subprocess.run(["tar", "xf", f"{ROOT}/slices/{nm}.tar", "-C", base], check=True)
                open(done, "w").write(time.ctime())
    return d


ISO_DIR = f"{_NSP_SCRATCH}/cpft_iso30"
ISO_TAR = f"{ROOT}/iso30/iso30.tar"


def iso30_store(workers=6):
    """各向同性体积库：本机有完成标记就直接用；否则加锁后从共享盘解包，共享盘也没有就现建并打包到共享盘。"""
    done = ISO_DIR + ".done"
    if os.path.exists(done):
        return ISO_DIR
    import fcntl
    with open(ISO_DIR + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not os.path.exists(done):
            subprocess.run(["rm", "-rf", ISO_DIR], check=True)
            if os.path.exists(ISO_TAR):
                print("从共享盘解包 iso30：", ISO_TAR, flush=True)
                subprocess.run(["tar", "xf", ISO_TAR, "-C", os.path.dirname(ISO_DIR)], check=True)
                open(done, "w").write("unpacked " + time.ctime())
            else:
                print("共享盘没有 iso30，现建", flush=True)
                sh([sys.executable, "make_iso_vols.py", "--out", ISO_DIR, "--workers", str(workers)])
                open(done, "w").write("built " + time.ctime())
                os.makedirs(os.path.dirname(ISO_TAR), exist_ok=True)
                subprocess.run(["tar", "cf", ISO_TAR + ".part", "-C", os.path.dirname(ISO_DIR), os.path.basename(ISO_DIR)], check=True)
                os.replace(ISO_TAR + ".part", ISO_TAR)
                print("iso30 已打包到", ISO_TAR, os.path.getsize(ISO_TAR) // 2**30, "GB", flush=True)
    return ISO_DIR


def eval_gt(model, out, cfg):
    sh([sys.executable, "eval_gt.py", "--model", model, "--out", out, "--workers", str(cfg.get("eval_workers", 3)),
        "--box", str(cfg.get("box", 384))]
       + (["--views", str(cfg["views"])] if cfg.get("views") else [])
       + (["--ens2d", cfg["ens2d"], "--ens_alpha", str(cfg.get("ens_alpha", 0.5))] if cfg.get("ens2d") else [])
       + (["--split", cfg["split"]] if cfg.get("split") else [])
       + (["--max_per_ds", str(cfg["max_per_ds"])] if cfg.get("max_per_ds") else [])
       + (["--diam_from", cfg["diam_from"]] if cfg.get("diam_from") else [])
       + (["--views1_ds", *cfg["views1_ds"]] if cfg.get("views1_ds") else [])
       + (["--split_json", cfg["split_json"]] if cfg.get("split_json") else [])
       + (["--only", *cfg["only"]] if cfg.get("only") else []))
    return json.load(open(f"{out}/summary.json"))


def main_t0(cfg, prefix, t0):
    """基础节点：建 iso30（后台）+ 用 gt-diam 协议评测若干现有模型。"""
    import threading
    err = []

    def build():
        try:
            iso30_store(cfg.get("iso_workers", 6))
        except Exception as e:                                          # 建库失败不影响评测，最后再报错
            err.append(e)
    th = threading.Thread(target=build); th.start()
    res = {}
    for m in cfg.get("eval_models", []):
        S = eval_gt(m, f"{ROOT}/results/{prefix}_gtval_{m}", cfg)
        c = S["cells"]
        res[m] = dict(main=round(c[S["main"]]["macro_f1_05"], 4), main_07=round(c[S["main"]]["macro_f1_07"], 4),
                      best=S["best"], best_f1_05=round(c[S["best"]]["macro_f1_05"], 4))
    th.join()
    if err:
        raise SystemExit(f"iso30 建库失败：{err[0]!r}")
    idx = json.load(open(f"{ISO_DIR}/index.json"))
    metrics = dict(models=res, iso30=dict(dir=ISO_DIR, tar=ISO_TAR, n=len(idx), n_train=sum(x["split"] == "train" for x in idx)),
                   results_prefix=f"{ROOT}/results/{prefix}_gtval_", minutes=round((time.time() - t0) / 60, 1))
    print("FINAL_METRICS " + json.dumps(metrics, ensure_ascii=False), flush=True)
    if res:
        v5 = [r["main"] for r in res.values()]; v7 = [r["main_07"] for r in res.values()]
        print(f"SUMMARY t0 gt-diam cp-2_vol0.15 {len(res)} 个模型均值 F1@0.5={np.mean(v5):.4f} (SD {np.std(v5, ddof=1) if len(v5) > 1 else 0:.4f}) "
              f"F1@0.7={np.mean(v7):.4f} | iso30 {len(idx)} 个体积", flush=True)


def main_gt(cfg, prefix, t0):
    """三维节点通用入口：（可选）训练三维模型，再按 gt-diam 协议评测。"""
    model = cfg.get("model")
    tr = cfg.get("train3d")
    if tr:
        name = tr["name"]
        assert name.startswith("orx_"), "训练出的权重名必须以 orx_ 开头"
        out = f"{ROOT}/models/models/{name}"
        if os.path.exists(out + ".orx_done"):
            print("三维模型已存在，跳过训练：", out, flush=True)
        else:
            iso = iso30_store(cfg.get("iso_workers", 6))
            # ---- 三维训练钩子：src/train3d.py 由三维节点实现，须在 out 下写出 model3d.json ----
            sh([sys.executable, "train3d.py", "--iso", iso, "--out", out] + [str(x) for x in tr.get("args", [])])
            open(out + ".orx_done", "w").write(time.ctime())
        model = out
    if cfg.get("no_eval"):                                             # round 5：只训练，评测另开节点
        print("FINAL_METRICS " + json.dumps(dict(model=model, trained=True, minutes=round((time.time() - t0) / 60, 1)), ensure_ascii=False), flush=True)
        return
    assert model, "gt 模式需要 model 或 train3d"
    # round 4：完整流程（diam_from）或测试集时结果目录改名，避免与开发期 gtval 混淆；config 里给了 tag 就用 _<tag>
    sfx = (f"_{cfg['tag']}" if cfg.get("tag") else
           f"_{'fp' if cfg.get('diam_from') else 'gt'}{cfg.get('split') or 'val'}"
           if (cfg.get("diam_from") or cfg.get("split") or cfg.get("max_per_ds")) else "_gtval")
    S = eval_gt(model, f"{ROOT}/results/{prefix}{sfx}", cfg)
    c = S["cells"]
    metrics = dict(model=model, main=S["main"], main_f1_05=round(c[S["main"]]["macro_f1_05"], 4), main_f1_07=round(c[S["main"]]["macro_f1_07"], 4),
                   main3_f1_05=round(c[S["main"]]["macro3_f1_05"], 4), best=S["best"], best_f1_05=round(c[S["best"]]["macro_f1_05"], 4),
                   best_f1_07=round(c[S["best"]]["macro_f1_07"], 4), results=f"{ROOT}/results/{prefix}{sfx}", minutes=round((time.time() - t0) / 60, 1))
    print("FINAL_METRICS " + json.dumps(metrics, ensure_ascii=False), flush=True)


def main():
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    print("CONFIG", json.dumps(cfg, ensure_ascii=False), flush=True)
    t0 = time.time()
    if cfg.get("mode") in ("t0", "gt"):
        prefix = "orx_" + (os.environ.get("ORX_RUN_ID") or time.strftime("%m%d_%H%M%S"))
        return (main_t0 if cfg["mode"] == "t0" else main_gt)(cfg, prefix, t0)
    if cfg.get("script"):                                              # 只跑一个分析脚本（不训练、不评测）
        prefix = "orx_" + (os.environ.get("ORX_RUN_ID") or time.strftime("%m%d_%H%M%S"))
        sh([sys.executable, cfg["script"], "--out", f"{ROOT}/results/{prefix}_{cfg['script'].split('.')[0]}"] + [str(x) for x in cfg.get("args", [])])
        return
    model = cfg.get("model", "base")
    tr = cfg.get("train")
    if tr:
        name = tr["name"]
        assert name.startswith("orx_"), "训练出的权重名必须以 orx_ 开头"
        path = f"{ROOT}/models/models/{name}"
        if os.path.exists(path + ".orx_done"):
            print("权重已存在，跳过训练：", path, flush=True)
        else:
            args = list(tr.get("args", []))
            if tr.get("extra_tar"):
                # 额外切片（如暗核增强副本）：共享盘上的 tar，第一次用时解包到本地盘
                d = f"{_NSP_SCRATCH}/cpft/extra_{os.path.basename(tr['extra_tar']).split('.')[0]}"
                if not os.path.isdir(d) or not os.listdir(d):
                    os.makedirs(d, exist_ok=True)
                    subprocess.run(["tar", "xf", f"{ROOT}/slices/{tr['extra_tar']}", "-C", d], check=True)
                sub = [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
                args += ["--extra", os.path.join(d, sub[0]) if len(sub) == 1 and not any(f.endswith(".tif") for f in os.listdir(d)) else d]
            if tr.get("exclude_val"):
                args += ["--exclude_json", os.path.join(HERE, "val_split.json")]
            if tr.get("slices"):
                sdir = slices_set(tr["slices"])
                args += ["--labels_suffix", "_flows.tif"]
            else:
                sdir = slices_dir()
            sh([sys.executable, "train_cpsam2.py", "--slices", sdir, "--name", name] + args)
            open(path + ".orx_done", "w").write(time.ctime())
        model = name
    if cfg.get("no_eval"):                                             # r9：二维也可以只训练，评测另开节点（交叉验证每折的留出体积）
        print("FINAL_METRICS " + json.dumps(dict(model=model, trained=bool(tr), minutes=round((time.time() - t0) / 60, 1)), ensure_ascii=False), flush=True)
        return
    mpath = f"{ROOT}/models/base/cpsam_v2" if model == "base" else f"{ROOT}/models/models/{model}"
    thrs = [str(t) for t in cfg.get("thrs", [0, -1, -2])]; fracs = [str(f) for f in cfg.get("fracs", [0, 0.15, 0.25, 0.35])]
    prefix = "orx_" + (os.environ.get("ORX_RUN_ID") or time.strftime("%m%d_%H%M%S"))
    grids = {}
    cmds = {split: [sys.executable, "eval_val.py", "--model", mpath, "--split", split, "--thrs", *thrs, "--fracs", *fracs,
                    "--max_per_ds", str(k), "--prefix", prefix, "--diam_mode", cfg.get("diam_mode", "gt")]
                   + (["--scales", *map(str, cfg["scales"])] if cfg.get("scales") else []) + (["--save_raw"] if cfg.get("save_raw") else [])
                   + (["--joint_rule", cfg["joint_rule"]] if cfg.get("joint_rule") else []) + (["--iso"] if cfg.get("iso") else []) + (["--iso_final"] if cfg.get("iso_final") else [])
            for split, k in (("val", cfg.get("val_per_ds", 4)), ("test", cfg.get("test_per_ds", 4))) if split in cfg.get("splits", ["val", "test"])}
    if cfg.get("parallel_splits"):
        # 验证集与测试集同时在同一张卡上跑，各自日志写到结果目录，结束后打印
        procs = {}
        for split, cmd in cmds.items():
            os.makedirs(f"{ROOT}/results/{prefix}_{split}", exist_ok=True)
            print("+", " ".join(cmd), flush=True)
            procs[split] = subprocess.Popen(cmd, cwd=SRC, stdout=open(f"{ROOT}/results/{prefix}_{split}/eval.log", "w"), stderr=subprocess.STDOUT)
        for split, pr in procs.items():
            rc = pr.wait()
            print(open(f"{ROOT}/results/{prefix}_{split}/eval.log").read(), flush=True)
            if rc:
                raise SystemExit(f"{split} 评测失败，退出码 {rc}")
    else:
        for split, cmd in cmds.items():
            sh(cmd)
    for split in cmds:
        grids[split] = json.load(open(f"{ROOT}/results/{prefix}_{split}/grid.json"))
    if len(cmds) == 1:
        # 只跑一个划分（记录阶段）：打印网格，规则比较在本地对 rows_all.json 做
        split = next(iter(cmds)); g = grids[split]
        metrics = dict(model=model, split=split, **{k: round(v["macro_f1_05"], 4) for k, v in g.items()},
                       results_prefix=f"{ROOT}/results/{prefix}", minutes=round((time.time() - t0) / 60, 1))
        print("FINAL_METRICS " + json.dumps(metrics, ensure_ascii=False), flush=True)
        print(f"SUMMARY model={model} {split} cp-2_vol0.15={metrics.get('cp-2_vol0.15')} cp0_vol0.25={metrics.get('cp0_vol0.25')}", flush=True)
        return
    best = max(grids["val"], key=lambda key: grids["val"][key]["macro_f1_05"])
    default = "cp0_vol0.25"
    pick = lambda split, key: round(grids[split][key]["macro_f1_05"], 4) if key in grids[split] else None
    pick7 = lambda split, key: round(grids[split][key]["macro_f1_07"], 4) if key in grids[split] else None
    print("\nPER_DATASET F1@0.5（验证集选出的设置 %s）" % best)
    for split in ("val", "test"):
        for d, v in grids[split][best]["per_ds"].items():
            print(f"  {split:4s} {d:14s} {v:.3f}")
    metrics = dict(model=model, val_selected=best,
                   val_f1_05_selected=pick("val", best), val_f1_05_default=pick("val", default),
                   test_f1_05_selected=pick("test", best), test_f1_07_selected=pick7("test", best),
                   test_f1_05_default=pick("test", default), results_prefix=f"{ROOT}/results/{prefix}",
                   minutes=round((time.time() - t0) / 60, 1))
    print("FINAL_METRICS " + json.dumps(metrics, ensure_ascii=False), flush=True)
    print(f"SUMMARY model={model} 验证集选出 {best}：val F1@0.5={metrics['val_f1_05_selected']} "
          f"test F1@0.5={metrics['test_f1_05_selected']} (默认 cp0/vol0.25：val {metrics['val_f1_05_default']} test {metrics['test_f1_05_default']})",
          flush=True)


if __name__ == "__main__":
    main()
