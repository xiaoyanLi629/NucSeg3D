"""在 2D 切片上微调 Cellpose-SAM（cpsam_v2），按数据集平衡采样。

参数沿用 Cellpose-SAM 论文与官方 Colab：AdamW，lr 1e-5，weight decay 0.1，batch 1，从内置 cpsam_v2 出发。
数据：make_slices.py 生成的 <数据集>__<样本>__<方向>.tif 与 *_masks.tif。
采样：每个数据集总概率相同（可用 --weights 调整，如 cellos=2,tapenade=2），数据集内均匀。
用法: python train_cpsam.py --name ft_all [--only ds ...] [--exclude ds ...] [--epochs 100] [--nimg 1000]
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import argparse
import glob
import json
import os
from collections import Counter

import numpy as np

ROOT = f"{_NSP_ROOT}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default=f"{_NSP_SCRATCH}/cpft_slices_v3/train")
    ap.add_argument("--name", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--exclude_samples", default="", help="排除样本清单文件")
    ap.add_argument("--weights", default="", help="数据集权重，如 cellos=2,tapenade=2")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--nimg", type=int, default=1000, help="每个 epoch 采样的切片数")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--base", default="cpsam_v2")
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--seed_offset", type=int, default=0, help="Cellpose 每轮用 np.random.seed(轮次)；加偏移得到不同的随机抽样与增强，用于估计重复训练的波动")
    a = ap.parse_args()

    files = sorted(f for f in glob.glob(f"{a.slices}/*.tif") if not f.endswith(("_masks.tif", "_flows.tif")))
    ds_of = lambda f: os.path.basename(f).split("__")[0]
    files = [f for f in files if (not a.only or ds_of(f) in a.only) and ds_of(f) not in a.exclude]
    if a.exclude_samples:                        # 排除指定样本（文件每行一个“数据集__样本编号”）
        bad = {l.strip() for l in open(a.exclude_samples) if l.strip()}
        files = [f for f in files if "__".join(os.path.basename(f).split("__")[:2]) not in bad]
        print("按样本排除:", len(bad), "个", flush=True)
    rng = np.random.default_rng(0)
    rng.shuffle(files)
    nval = max(20, int(len(files) * a.val_frac))
    val, train = files[:nval], files[nval:]
    cnt = Counter(ds_of(f) for f in train)
    w = {k: 1.0 for k in cnt}
    for kv in filter(None, a.weights.split(",")):
        k, v = kv.split("="); w[k] = float(v)
    probs = np.array([w[ds_of(f)] / cnt[ds_of(f)] for f in train]); probs /= probs.sum()
    print("训练切片:", dict(cnt), " 验证切片:", len(val), " 数据集权重:", w, flush=True)

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")   # 输出 Cellpose 的逐轮损失
    from cellpose import models, train as cptrain
    if a.seed_offset:
        _seed = np.random.seed
        np.random.seed = lambda s=None: _seed(None if s is None else int(s) + a.seed_offset)
        import torch; torch.manual_seed(a.seed_offset)
    model = models.CellposeModel(gpu=True, pretrained_model=a.base)
    save = f"{ROOT}/models"
    os.makedirs(save, exist_ok=True)
    json.dump(dict(args=vars(a), counts=dict(cnt), weights=w), open(f"{save}/{a.name}_config.json", "w"), indent=1)
    path, train_losses, test_losses = cptrain.train_seg(
        model.net, train_files=train, train_labels_files=[f[:-4] + "_masks.tif" for f in train], train_probs=probs,
        test_files=val, test_labels_files=[f[:-4] + "_masks.tif" for f in val],
        learning_rate=a.lr, weight_decay=0.1, n_epochs=a.epochs, batch_size=1, nimg_per_epoch=a.nimg,
        min_train_masks=3, save_path=save, model_name=a.name, save_every=20)
    json.dump(dict(model_path=str(path), train_losses=list(map(float, train_losses)),
                   test_losses=list(map(float, test_losses))), open(f"{save}/{a.name}_losses.json", "w"), indent=1)
    print("TRAIN_DONE", path, flush=True)


if __name__ == "__main__":
    main()
