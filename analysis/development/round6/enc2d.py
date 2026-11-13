"""第六轮候选 W：三维模型的编码器一侧（net.*）向同种子二维起点回拉，解码器（dec.*）不变。
net.* = (1 − β)·三维 + β·二维，β 为回拉比例（W 固定 0.5）。用法: enc2d.py <三维目录> <输出目录> <β>"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import json, os, shutil, sys
import torch
src, out, beta = sys.argv[1], sys.argv[2], float(sys.argv[3])
M2 = f"{_NSP_ROOT}/models/models"
meta = json.load(open(os.path.join(src, "model3d.json")))
p2 = meta["init"] if os.path.exists(meta["init"]) else f"{M2}/{meta['init']}"
s3 = torch.load(os.path.join(src, meta["weights"]), map_location="cpu", weights_only=True)
s2 = torch.load(p2, map_location="cpu", weights_only=True)
sd = {}; n = 0
for k, v in s3.items():
    if k.startswith("net."):
        w2 = s2[k[4:]]; assert w2.shape == v.shape, k
        sd[k] = ((1 - beta) * v.double() + beta * w2.double()).to(v.dtype) if v.is_floating_point() else v; n += 1
    else:
        assert k.startswith("dec."), k; sd[k] = v
tmp = out + ".part"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
torch.save(sd, os.path.join(tmp, "weights.pt"))
json.dump(dict(meta, weights="weights.pt", enc2d_from=src, enc2d_beta=beta, final_loss=None), open(os.path.join(tmp, "model3d.json"), "w"), indent=1, ensure_ascii=False)
assert not os.path.exists(out), out
os.replace(tmp, out)
print(f"ENC2D_DONE {out} β={beta} 回拉张量 {n}，二维 {p2}")
