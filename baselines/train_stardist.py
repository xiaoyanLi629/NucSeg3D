"""StarDist-3D 基线训练（iso30 训练体积，与我们三维模型完全相同的 910 个体积；mesospim 不参与）。

数据：共享盘 iso30.tar 零拷贝 memmap 读取。体积太大（约 37 G 体素），不能像 StarDist 示例那样整份读进内存，
  所以把训练集包装成“虚拟区域列表”：第 k 项是按固定种子 k 抽出的一块区域（边长 = 训练块 × 1.25），
  抽样规则与我们三维模型的 data3d.PatchSource 相同：数据集份额 ∝ 权重（cellos、tapenade 为 2，其余 1，blastospim ≤ 25%），
  数据集内按体积均匀，90% 的区域以随机前景体素为中心。StarDist 自己的数据生成器再在区域里取训练块、算星形距离、做增强。
验证：每个 iso30 验证体积取一块以前景为中心的 64×256×256 区域（每套至多 3 块，mesospim 除外），只用于监控损失、
  学习率衰减与 StarDist 自带的 optimize_thresholds（选概率与 NMS 阈值）；正式指标由 bl_eval.py 在原始网格上算。
配置：--cfg A = StarDist 3D 默认配置（resnet 主干、96 条金螺旋射线、grid 2,2,2、400 轮 × 100 步、batch 2），
  训练块从示例的 48×96×96 放大到 64×128×128（iso30 核径约 30 像素）；--cfg B = 同上但用 U-Net 主干、深度 3（感受野更大）。
增强：StarDist 3D 示例的做法（各轴随机翻转、xy 90° 旋转、亮度缩放平移、加性高斯噪声）。
用法：python train_stardist.py --cfg A --out <basedir> [--epochs N --steps M]（冒烟测试用小的 N、M）"""
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iso30tar import Iso30  # noqa: E402

DEFAULT_W = {"cellos": 2.0, "tapenade": 2.0}
CAPS = {"blastospim": 0.25}


def dataset_probs(entries):
    """与 data3d.dataset_probs 相同。"""
    dss = sorted({e["ds"] for e in entries})
    share = np.array([DEFAULT_W.get(d, 1.0) for d in dss], float); share /= share.sum()
    for d, c in CAPS.items():
        if d in dss:
            i = dss.index(d)
            if share[i] > c:
                share = share * (1 - c) / (1 - share[i]); share[i] = c
    cnt = {d: sum(e["ds"] == d for e in entries) for d in dss}
    p = np.array([share[dss.index(e["ds"])] / cnt[e["ds"]] for e in entries])
    return p / p.sum(), dict(zip(dss, share.round(4).tolist()))


class Regions:
    """虚拟区域列表的共享部分：第 k 项的 (体积序号, 切片) 由种子 k 决定，图像与标签两个视图按同一规则取。"""

    def __init__(self, entries, probs, size, n, seed=0, p_fg=0.9, patch=None):
        self.patch = patch
        self.S = Iso30(); self.e = entries; self.p = probs; self.size = np.array(size); self.n = n
        self.seed, self.p_fg = seed, p_fg
        self._fg = {}

    def _fgc(self, i):
        c = self._fg.get(i)
        if c is None:
            lab = self.S.npy(self.e[i]["lab"])
            sub = np.asarray(lab[::4, ::4, ::4]); del lab
            fg = np.argwhere(sub > 0) * 4
            if len(fg) > 20000:
                fg = fg[np.random.default_rng(i).choice(len(fg), 20000, replace=False)]
            c = self._fg[i] = (self.e[i]["shape"], fg)
        return c

    def where(self, k):
        rng = np.random.default_rng([self.seed, k])
        i = int(rng.choice(len(self.e), p=self.p))
        shp, fg = self._fgc(i)
        shp = np.array(shp); R = np.minimum(self.size, shp)
        if len(fg) and rng.random() < self.p_fg:
            c = fg[rng.integers(len(fg))] + rng.integers(0, 4, 3)
        else:
            c = np.array([rng.integers(0, n) for n in shp])
        st = np.clip(c - R // 2, 0, shp - R)
        return i, tuple(slice(int(a), int(a + r)) for a, r in zip(st, R))

    def get(self, k, what):
        i, sl = self.where(k)
        a = np.array(self.S.npy(self.e[i][what])[sl])
        if self.patch is not None:                                         # 少数体积某轴小于训练块（purdue、tapenade 等），补零
            a = np.pad(a, [(0, max(0, p - s)) for p, s in zip(self.patch, a.shape)])
        return a.astype(np.float32) if what == "img" else a.astype(np.int32)


class View:
    def __init__(self, R, what):
        self.R, self.what = R, what

    def __len__(self):
        return self.R.n

    def __getitem__(self, k):
        return self.R.get(int(k), self.what)


def val_crops(entries, size=(64, 256, 256), per_ds=3):
    S = Iso30(); X, Y, names = [], [], []
    by = {}
    for e in entries:
        by.setdefault(e["ds"], []).append(e)
    for ds, es in sorted(by.items()):
        if ds == "mesospim":
            continue
        for e in es[:: max(1, len(es) // per_ds)][:per_ds]:
            lab = S.npy(e["lab"]); shp = np.array(lab.shape); R = np.minimum(size, shp)
            sub = np.asarray(lab[::4, ::4, ::4]); fg = np.argwhere(sub > 0) * 4
            c = fg[len(fg) // 2] if len(fg) else shp // 2
            st = np.clip(c - R // 2, 0, shp - R); sl = tuple(slice(int(a), int(a + r)) for a, r in zip(st, R))
            y = np.array(lab[sl]).astype(np.int32)
            pad = [(0, max(0, p - s)) for p, s in zip((64, 128, 128), y.shape)]
            if y.max() == 0:
                continue
            X.append(np.pad(np.array(S.npy(e["img"])[sl]).astype(np.float32), pad)); Y.append(np.pad(y, pad)); names.append(f"{ds}/{e['name']}")
    return X, Y, names


def augmenter(x, y):
    """StarDist 3D 示例的增强（这里网格各向同性，三个轴都可翻转）。"""
    perm = (0, 2, 1) if np.random.rand() < 0.5 else (0, 1, 2)            # xy 转置 + 翻转 = 90° 旋转
    x, y = x.transpose(perm), y.transpose(perm)
    for ax in range(3):
        if np.random.rand() < 0.5:
            x, y = np.flip(x, ax), np.flip(y, ax)
    x = x * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    sig = 0.02 * np.random.uniform(0, 1)
    x = x + sig * np.random.normal(0, 1, x.shape)
    return np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="A", choices=["A", "B"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_opt", action="store_true", help="跳过 optimize_thresholds（冒烟测试）")
    a = ap.parse_args()
    import tensorflow as tf
    mem = int(os.environ.get("BL_TF_MEM_MB", "0"))                     # 冒烟测试时限制显存（与别的作业共卡）
    for g in tf.config.list_physical_devices("GPU"):
        if mem:
            tf.config.set_logical_device_configuration(g, [tf.config.LogicalDeviceConfiguration(memory_limit=mem)])
        else:
            tf.config.experimental.set_memory_growth(g, True)
    from stardist import Rays_GoldenSpiral
    from stardist.models import Config3D, StarDist3D
    np.random.seed(a.seed); tf.random.set_seed(a.seed)
    idx = Iso30().index()
    tr = [e for e in idx if e["split"] == "train" and e["ds"] != "mesospim"]
    va = [e for e in idx if e["split"] == "val"]
    probs, share = dataset_probs(tr)
    patch = (64, 128, 128)
    common = dict(rays=Rays_GoldenSpiral(96, anisotropy=(1, 1, 1)), grid=(2, 2, 2), anisotropy=(1, 1, 1), n_channel_in=1,
                  train_patch_size=patch, train_batch_size=2, train_sample_cache=False, train_epochs=a.epochs,
                  train_steps_per_epoch=a.steps, use_gpu=False)
    if a.cfg == "A":
        conf = Config3D(**common)                                          # 默认 resnet 主干
    else:
        conf = Config3D(backbone="unet", unet_n_depth=3, **common)
    name = a.name or f"stardist_{a.cfg}_s{a.seed}"
    model = StarDist3D(conf, name=name, basedir=a.out)
    n_virtual = a.epochs * a.steps * conf.train_batch_size + 1000
    R = Regions(tr, probs, tuple(int(math.ceil(p * 1.25)) for p in patch), n_virtual, seed=a.seed, patch=patch)
    Xv, Yv, vn = val_crops(va)
    json.dump(dict(args=vars(a), share=share, n_train=len(tr), val_crops=vn, config=vars(conf)),
              open(f"{a.out}/{name}/bl_meta.json", "w"), indent=1, default=str)
    print("dataset share", share, "| val crops", len(Xv), flush=True)
    t0 = time.time()
    model.train(View(R, "img"), View(R, "lab"), validation_data=(Xv, Yv), augmenter=augmenter, epochs=a.epochs,
                steps_per_epoch=a.steps, workers=a.workers)
    print(f"TRAIN_DONE {(time.time() - t0) / 60:.1f} min", flush=True)
    if not a.no_opt:
        model.optimize_thresholds(Xv, Yv, nms_threshs=[0.3, 0.4, 0.5])   # 缩小 NMS 网格以控制耗时（StarDist 默认 0.3/0.4/0.5 相同）
        print("THRESH", model.thresholds, flush=True)


if __name__ == "__main__":
    main()
