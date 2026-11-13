"""测试 c：一步优化后 FiLM 参数有非零梯度。
条件模型（cond=res、cond_enc=2、cond_nfreq=2）从 S1 种子 0 的权重出发（同 train3d --init3d），取一个真实训练块
（PatchSource res_cond=True，种子 0，带 f；为省显存用 16×256×256 的块），按 train3d 的方式（bf16 autocast、loss_fn、AdamW 两组学习率、梯度裁剪）
连做三步：第 1 步零初始化的头（dec.film、rcond_enc）应有非零梯度，MLP 主干因为头为零、梯度应为 0；
optimizer.step 之后第 2 步主干也应有梯度；第 3 步用 res_drop=True，“未知”嵌入 null 应有梯度。
最后确认更新后的网络输出随 f 变化、未知嵌入与给定 f 的输出不同。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import os
import sys

os.getpid = lambda: 4242
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import data3d  # noqa: E402
from net3d import CPSAM3D  # noqa: E402
from train3d import loss_fn, load_init3d  # noqa: E402
from r6common import S1, CONFIGS, train_source  # noqa: E402

dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
torch.manual_seed(0)
model = CPSAM3D(dict(variant="dec3d", freeze_encoder=False, lora=0, patch_z=16, cond="res", cond_enc=2, cond_nfreq=2),
                init="orx_v5_2d_s0").to(dev).train()
load_init3d(model, S1)
enc_p, new_p = model.param_groups()
opt = torch.optim.AdamW([dict(params=enc_p, lr=1e-5), dict(params=new_p, lr=2e-4)], weight_decay=0.05)
params = enc_p + new_p
PATCH = (16, 256, 256)                               # 16 层的块（省显存；机器与别的任务共用）
ps, _ = train_source(data3d, CONFIGS["A1"], res_cond=True, patch=PATCH)
it = iter(ps)
for _ in range(3):                                   # 跳过几块，取一个前景多的块
    s = next(it)
x, tgt, va, res = data3d.to_patch(s, dev, PATCH)
print(f"训练块 {s['ds']} {s['name']}，f = {[round(v, 3) for v in res.tolist()]}，前景比例 {float(tgt[3].mean()):.3f}", flush=True)
GROUPS = {"dec.film（解码器 FiLM 头）": "dec.film.", "rcond_enc（编码器 token 偏置头）": "rcond_enc.",
          "rcond.mlp（条件 MLP 主干）": "rcond.mlp.", "rcond.null（未知嵌入）": "rcond.null"}


def gnorms():
    out = {}
    for g, pre in GROUPS.items():
        gs = [p.grad for n, p in model.named_parameters() if n.startswith(pre) and p.grad is not None]
        out[g] = float(torch.sqrt(sum((t.float() ** 2).sum() for t in gs))) if gs else 0.0
    return out


def step(k, **kw):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = model(x[None], res=res[None], **kw)
    loss, lf, lc = loss_fn(y, tgt[None], va[None])
    loss.backward()
    g = gnorms()
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    print(f"第 {k} 步{'（res_drop=True）' if kw else ''} 损失 {float(loss):.5f}，总梯度范数 {float(gn):.3f}；" +
          "；".join(f"{a} {b:.3e}" for a, b in g.items()), flush=True)
    return g


g1 = step(1)
g2 = step(2)
g3 = step(3, res_drop=torch.tensor([True], device=dev))
print("T_c 第 1 步 FiLM 头梯度非零 =", g1["dec.film（解码器 FiLM 头）"] > 0, "；编码器偏置头梯度非零 =", g1["rcond_enc（编码器 token 偏置头）"] > 0,
      "；MLP 主干梯度为 0（头为零时的预期）=", g1["rcond.mlp（条件 MLP 主干）"] == 0)
print("T_c 一步优化后（第 2 步）FiLM 头与 MLP 主干梯度都非零 =", g2["dec.film（解码器 FiLM 头）"] > 0 and g2["rcond.mlp（条件 MLP 主干）"] > 0)
print("T_c 第 3 步（条件 dropout）未知嵌入梯度非零 =", g3["rcond.null（未知嵌入）"] > 0)
model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    ya = model(x[None], res=res[None]).float()
    yb = model(x[None], res=(res * 2)[None]).float()
    yn = model(x[None], res=None).float()
print(f"T_c 更新后输出随 f 变化：f 与 2f 的最大差 {float((ya - yb).abs().max()):.3e}；未知嵌入与 f 的最大差 {float((ya - yn).abs().max()):.3e}")
print("T_C_DONE")
