"""零训练权重平均：w = α·A + (1−α)·B（α 事先固定 0.5，不在 val 上调）。两个模型须同一二维初始权重、同一网络配置；
只平均 weights.pt 里的可训练参数，其余参数载入时都来自同一个二维初始权重，所以平均后的网络是逐参数的 α 混合。
用法: avg_weights.py <A 目录> <B 目录> <输出目录> [α]"""
import json, os, shutil, sys
import torch
A, B, OUT = sys.argv[1:4]; al = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
ma, mb = (json.load(open(os.path.join(p, "model3d.json"))) for p in (A, B))
assert ma["init"] == mb["init"] and ma["cfg"] == mb["cfg"] and ma["variant"] == mb["variant"], "初始权重或配置不同，不能平均"
sa = torch.load(os.path.join(A, ma["weights"]), map_location="cpu", weights_only=True)
sb = torch.load(os.path.join(B, mb["weights"]), map_location="cpu", weights_only=True)
assert sa.keys() == sb.keys() and all(sa[k].shape == sb[k].shape and sa[k].dtype == sb[k].dtype for k in sa)
sd = {}
for k in sa:
    if sa[k].is_floating_point():
        sd[k] = (al * sa[k].double() + (1 - al) * sb[k].double()).to(sa[k].dtype)
    else:
        assert torch.equal(sa[k], sb[k]), k; sd[k] = sa[k]
tmp = OUT + ".part"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
torch.save(sd, os.path.join(tmp, "weights.pt"))
meta = dict(mb, weights="weights.pt", avg_of=[A, B], avg_alpha=al, final_loss=None)
json.dump(meta, open(os.path.join(tmp, "model3d.json"), "w"), indent=1, ensure_ascii=False)
assert not os.path.exists(OUT), f"{OUT} 已存在"
os.replace(tmp, OUT)
d = max(float((sd[k].double() - sa[k].double()).abs().max()) for k in sd if sd[k].is_floating_point())
print(f"AVG_DONE {OUT}：{len(sd)} 个张量，α={al}，与 A 的最大差 {d:.3e}")
