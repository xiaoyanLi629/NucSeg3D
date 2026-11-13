"""零训练“拼装”诊断（只用于理解，不参与选择）：编码器（net.encoder.*）与其余可训练参数（三维解码器 dec.* 与输出头 net.out.*）
分别取自模型 A、模型 B 或两者平均。A、B 须同一二维初始权重与网络配置（如同种子的 C 与 X1）。写法与 r3_aniso/avg_weights.py 相同。
用法: partmix.py <A 目录> <B 目录> <输出目录> --enc A|B|avg --dec A|B|avg
例：编码器取 X1、解码器取 C：partmix.py <X1> <C> <out> --enc A --dec B"""
import argparse
import json
import os
import shutil

import torch

ap = argparse.ArgumentParser()
ap.add_argument("A"); ap.add_argument("B"); ap.add_argument("out")
ap.add_argument("--enc", choices=["A", "B", "avg"], required=True); ap.add_argument("--dec", choices=["A", "B", "avg"], required=True)
a = ap.parse_args()
ma, mb = (json.load(open(os.path.join(p, "model3d.json"))) for p in (a.A, a.B))
assert ma["init"] == mb["init"] and ma["cfg"] == mb["cfg"] and ma["variant"] == mb["variant"], "初始权重或配置不同，不能拼装"
sa = torch.load(os.path.join(a.A, ma["weights"]), map_location="cpu", weights_only=True)
sb = torch.load(os.path.join(a.B, mb["weights"]), map_location="cpu", weights_only=True)
assert sa.keys() == sb.keys()
sd, cnt = {}, {"enc": 0, "dec": 0}
for k in sa:
    part = "enc" if k.startswith("net.encoder.") else "dec"
    how = a.enc if part == "enc" else a.dec
    if not sa[k].is_floating_point():
        assert torch.equal(sa[k], sb[k]), k; sd[k] = sa[k]; continue
    sd[k] = sa[k] if how == "A" else sb[k] if how == "B" else (0.5 * sa[k].double() + 0.5 * sb[k].double()).to(sa[k].dtype)
    cnt[part] += sa[k].numel()
tmp = a.out + ".part"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
torch.save(sd, os.path.join(tmp, "weights.pt"))
meta = dict(ma, weights="weights.pt", partmix=dict(A=a.A, B=a.B, enc=a.enc, dec=a.dec), final_loss=None)
json.dump(meta, open(os.path.join(tmp, "model3d.json"), "w"), indent=1, ensure_ascii=False)
assert not os.path.exists(a.out), f"{a.out} 已存在"
os.replace(tmp, a.out)
print(f"PARTMIX_DONE {a.out}：编码器 {cnt['enc']} 个参数取 {a.enc}，其余 {cnt['dec']} 个取 {a.dec}")
