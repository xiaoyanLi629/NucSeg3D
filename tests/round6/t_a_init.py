"""测试 a-2：关闭新功能时，按 train3d 的顺序（torch.manual_seed / np.random.seed → CPSAM3D(cfg, init)）建网络后，
torch 与 numpy 的随机数状态、全部参数、cfg（写进 model3d.json 的那份）、可训练参数名与分组，与改前逐项相同。
用法：python t_a_init.py <src目录>（CPU，打印哈希，与另一份代码的输出比对）"""
import hashlib
import json
import sys

src = sys.argv[1]
sys.path.insert(0, src)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from net3d import CPSAM3D  # noqa: E402


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


CFGS = {
    "A1(dec3d)": dict(variant="dec3d", freeze_encoder=False, lora=0, patch_z=32),
    "zadapt_zconv1": dict(variant="zadapt", freeze_encoder=False, lora=0, patch_z=32,
                          adapt_blocks=list(range(0, 24)), adapt_kind="zconv", adapt_dim=64),
    "dec3d_aux": dict(variant="dec3d", freeze_encoder=False, lora=0, patch_z=32, aux_ch=1),
}
for name, cfg in CFGS.items():
    torch.manual_seed(0); np.random.seed(0)
    m = CPSAM3D(cfg, init="orx_v5_2d_s0")
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    enc, new = m.param_groups()
    tr = sorted(n for n, p in m.named_parameters() if p.requires_grad)
    print(f"T_a2 {name:14s} 参数 {h.hexdigest()[:16]}  torch RNG {sha(torch.get_rng_state().numpy().tobytes())}  "
          f"numpy RNG {sha(np.random.get_state()[1].tobytes())}  cfg {sha(json.dumps(m.cfg, sort_keys=True).encode())}  "
          f"可训练名 {sha(repr(tr).encode())}  分组 {len(enc)}/{len(new)}", flush=True)
    del m
print("T_A_INIT_DONE")
