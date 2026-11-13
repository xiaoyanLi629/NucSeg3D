"""第六轮零训练诊断：编码器一侧（net.*，含二维输出头）与解码器一侧（dec.*）分别按各自系数混合。
w = (1 − a)·C + a·X，a_enc 作用于 net.*，a_dec 作用于 dec.*；a=0 取 C，a=1 取 X，a=0.5 为平均（两侧都 0.5 即 S1）。
只用于理解鲁棒性在哪一侧，不参与选择。用法: avg_part.py <C 目录> <X 目录> <输出目录> <a_enc> <a_dec>"""
import json, os, shutil, sys
import torch
C, X, OUT = sys.argv[1:4]; ae, ad = float(sys.argv[4]), float(sys.argv[5])
mc, mx = (json.load(open(os.path.join(p, "model3d.json"))) for p in (C, X))
assert mc["init"] == mx["init"] and mc["cfg"] == mx["cfg"] and mc["variant"] == mx["variant"]
sc = torch.load(os.path.join(C, mc["weights"]), map_location="cpu", weights_only=True)
sx = torch.load(os.path.join(X, mx["weights"]), map_location="cpu", weights_only=True)
assert sc.keys() == sx.keys()
sd = {}; cnt = {"net": 0, "dec": 0}
for k in sc:
    side = k.split(".")[0]; assert side in cnt, k; cnt[side] += 1
    a = ae if side == "net" else ad
    sd[k] = ((1 - a) * sc[k].double() + a * sx[k].double()).to(sc[k].dtype) if sc[k].is_floating_point() else sc[k]
tmp = OUT + ".part"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
torch.save(sd, os.path.join(tmp, "weights.pt"))
meta = dict(mc, weights="weights.pt", avg_of=[C, X], avg_alpha_enc=ae, avg_alpha_dec=ad, final_loss=None)
json.dump(meta, open(os.path.join(tmp, "model3d.json"), "w"), indent=1, ensure_ascii=False)
assert not os.path.exists(OUT), OUT
os.replace(tmp, OUT)
print(f"PART_DONE {OUT} a_enc={ae} a_dec={ad} 张量 {cnt}")
