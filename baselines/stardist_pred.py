"""StarDist-3D 在 iso30 网格上的推理，供 bl_eval.py 调用（需在 env_stardist 下运行）。
网络只跑一次（model.predict，自动分块），再用两组阈值做 NMS 出实例：
  变体 1 = 训练结束时在验证块上 optimize_thresholds 得到的阈值（thresholds.json），变体 2 = StarDist 默认阈值（prob 0.5、nms 0.4）。"""
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np


class StarDistPredictor:
    def __init__(self, model_dir, prob_default=0.5, nms_default=0.4):
        import tensorflow as tf
        mem = int(os.environ.get("BL_TF_MEM_MB", "0"))                     # 与别的作业共卡时限制显存
        for g in tf.config.list_physical_devices("GPU"):
            if mem:
                tf.config.set_logical_device_configuration(g, [tf.config.LogicalDeviceConfiguration(memory_limit=mem)])
            else:
                tf.config.experimental.set_memory_growth(g, True)
        from stardist.models import StarDist3D
        model_dir = model_dir.rstrip("/")
        self.m = StarDist3D(None, name=os.path.basename(model_dir), basedir=os.path.dirname(model_dir))
        t = self.m.thresholds
        self.var = {1: (float(t.prob), float(t.nms)), 2: (prob_default, nms_default)}

    def predict(self, x):
        raw = self.net(x)
        return {v: self.post(raw, v, x.shape) for v in self.var}

    def post(self, raw, v, shape):
        prob, dist = raw
        lab, _ = self.m._instances_from_prediction(shape, prob, dist, prob_thresh=self.var[v][0], nms_thresh=self.var[v][1])
        return lab.astype(np.int32)

    def net(self, x):
        import tensorflow as tf
        nt = list(self.m._guess_n_tiles(x))
        for _ in range(6):                                                 # 显存限 7.5 GB 时大体积会让 cuDNN 申请不到工作区：加倍分块重试
            try:
                prob, dist = self.m.predict(x, n_tiles=tuple(nt), show_tile_progress=False)
                break
            except (tf.errors.UnknownError, tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
                ax = int(np.argmax([n / t for n, t in zip(x.shape, nt)]))
                nt[ax] *= 2
                print(f"    StarDist 显存不足，n_tiles 改为 {nt}（{type(e).__name__}）", flush=True)
        else:
            raise RuntimeError("StarDist 分块加到上限仍显存不足")
        self.last_tiles = nt
        return prob, dist
