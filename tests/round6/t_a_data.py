"""测试 a-1：关闭新功能时训练块序列与改前逐项相同（哈希比对）。
用法：python t_a_data.py <src目录> <输出 json> [n] [gpu]
五种增强配置（r6common.CONFIGS）各从同一种子抽 n 块（真实读 iso30 盘、单进程、os.getpid 固定为 4242，
与 r5/rngcheck.py 同法），逐块哈希 img/lab/valid 字节与增强参数；给 gpu 时对 A1、ALL 两种配置再做 to_patch
（GPU 上缩放、翻转旋转、各向异性模拟、三维流），哈希输出的 (x, target, valid)。
src 为新代码时，再用 res_cond=True 各跑一遍：去掉 res 后的序列应与关闭时相同，并记下每块的 f。"""
import hashlib
import json
import os
import sys

os.getpid = lambda: 4242
src, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
gpu = sys.argv[4] if len(sys.argv) > 4 else None
sys.path.insert(0, src)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import data3d  # noqa: E402
from r6common import CONFIGS, train_source, aug_key, h_arrays  # noqa: E402

new = "res_cond" in data3d.PatchSource.__init__.__code__.co_varnames
dev = torch.device(gpu) if gpu else None
if dev is not None:
    torch.cuda.set_device(dev)
R = {"src": src, "new_code": new, "n": n}
for name, cfg in CONFIGS.items():
    for rc in ([False, True] if new else [False]):
        kw = dict(res_cond=True) if rc else {}
        ps, idx = train_source(data3d, cfg, **kw)
        it = iter(ps)
        per, gper, res, dss = [], [], [], []
        for k in range(n):
            s = next(it)
            per.append(h_arrays(s["img"], s["lab"], s["valid"]) + "|" + aug_key(s["aug"]))
            dss.append(s["ds"])
            if rc:
                res.append(data3d.block_res(s["aug"]))
            if dev is not None and name in ("A1", "ALL"):
                o = data3d.to_patch(s, dev)
                gper.append(h_arrays(*(t.cpu().numpy() for t in o[:3])))
                if rc:
                    assert len(o) == 4 and torch.allclose(o[3].cpu(), torch.tensor(res[-1])), (len(o), o[-1], res[-1])
                else:
                    assert len(o) == 3, len(o)
        tag = name + ("+res" if rc else "")
        R[tag] = dict(blocks=per, gpu=gper, ds=dss, res=res)
        hh = hashlib.sha256("".join(per).encode()).hexdigest()[:16]
        gh = hashlib.sha256("".join(gper).encode()).hexdigest()[:16] if gper else "-"
        print(f"{tag:10s} {n} 块  CPU 序列哈希 {hh}  GPU to_patch 哈希 {gh}", flush=True)
json.dump(R, open(out, "w"))
print("T_A_DATA_DONE", out)
