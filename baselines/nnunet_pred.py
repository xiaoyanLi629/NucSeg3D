"""nnU-Net（三类：背景/核内部/核边界）在 iso30 网格上的推理与实例化，供 bl_eval.py 调用。

推理：nnU-Net v2 官方 nnUNetPredictor，默认设置（滑窗步长 0.5、高斯加权、镜像测试时增强），体素间距 1,1,1（iso30 网格）。
实例化（文献里三类表示的常规做法：内部类做种子、连通域编号，再把边界体素分给相邻的核）：
  前景 = argmax ≠ 背景；种子 = 前景内的“内部”体素（变体 1：argmax = 内部；变体 2/3/4：P(内部) > 0.6/0.75/0.9），
  6 邻接连通域编号；再以 −P(内部) 为地形、种子为标记、前景为掩膜做标记分水岭，把边界体素分给最近的核。
  没有种子的前景块被丢弃（体积下限在 bl_eval 里统一处理）。"""
import os

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.segmentation import watershed


def instances(prob, seeds, fg):
    mk, n = ndi.label(seeds & fg)
    if n == 0:
        return np.zeros(fg.shape, np.int32)
    return watershed(-prob[1].astype(np.float32), mk, mask=fg).astype(np.int32)


class NNUNetPredictor:
    def __init__(self, model_dir, fold=0, checkpoint="checkpoint_final.pth", step=0.5, mirror=True, seed_thrs=(0.6, 0.75, 0.9), on_device=False):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        frac = float(os.environ.get("BL_GPU_FRAC", "0"))                  # 与别的作业共卡时限制显存；概率图累积放内存，不占显存
        if frac:
            torch.cuda.set_per_process_memory_fraction(frac)
        self.p = nnUNetPredictor(tile_step_size=step, use_gaussian=True, use_mirroring=mirror, perform_everything_on_device=on_device,
                                 device=torch.device("cuda"), verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        self.p.initialize_from_trained_model_folder(model_dir, use_folds=tuple(fold) if isinstance(fold, (list, tuple)) else (fold,),
                                                    checkpoint_name=checkpoint)   # 五折集成：popts {"fold": [0,1,2,3,4]}，nnU-Net 对各折 softmax 取平均
        self.seed_thrs = seed_thrs

    def net(self, x):
        """网络部分（nnU-Net 自己的预处理、滑窗、镜像 TTA），返回 (argmax 分割, 各类概率)。"""
        return self.p.predict_single_npy_array(x[None].astype(np.float32), {"spacing": [1.0, 1.0, 1.0]}, None, None, True)

    def post(self, raw, v):
        """实例化，变体 v（1 = argmax 内部类做种子；2/3/4 = P(内部) > seed_thrs[v−2]）。"""
        seg, prob = raw
        fg = seg != 0
        return instances(prob, seg == 1 if v == 1 else prob[1] > self.seed_thrs[v - 2], fg)

    def predict(self, x):
        seg, prob = self.net(x)
        fg = seg != 0
        out = {1: instances(prob, seg == 1, fg)}
        for k, t in enumerate(self.seed_thrs):
            out[k + 2] = instances(prob, prob[1] > t, fg)
        return out
