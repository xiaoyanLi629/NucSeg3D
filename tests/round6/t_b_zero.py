"""测试 b：加条件但未训练的模型（FiLM 头、编码器偏置零初始化）与不加条件的同权重模型，输出差值为 0。
权重：S1 种子 0（orx_t3d_r5s1_s0，第五轮模型）。输入：验证集 iso30 体积 cellos Hoechst_d4_flt32_50 中心的 32×256×256 块。
三种前向：推理（no_grad + bf16 autocast，同 infer3d）、fp32、训练模式（梯度 + 检查点 + bf16，同 train3d）。
条件配置：cond_enc 0/1/2、cond_nfreq 0/2；f 取三组值、None（“未知”嵌入）和 res_drop。
再用 Model3D 走完整的 eval（cond 模型目录指向同一份权重），bbbc050 与 cellos（z 插值）两个验证体积的 dP/cellprob 逐位比较。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import os
import shutil
import sys

sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from net3d import CPSAM3D  # noqa: E402
from r6common import S1, ISO, T  # noqa: E402

dev = torch.device("cuda")
meta = json.load(open(f"{S1}/model3d.json"))
sd = torch.load(f"{S1}/{meta['weights']}", map_location="cpu", weights_only=True)
idx = json.load(open(f"{ISO}/index.json"))
e = next(e for e in idx if e["ds"] == "cellos" and e["name"] == "Hoechst_d4_flt32_50" and e["split"] == "val")
v = np.load(f"{ISO}/{e['img']}", mmap_mode="r")
z0, y0, x0 = (max(0, (n - w) // 2) for n, w in zip(v.shape, (32, 256, 256)))
blk = np.zeros((32, 256, 256), np.float32)
s = np.asarray(v[z0:z0 + 32, y0:y0 + 256, x0:x0 + 256], np.float32); blk[:s.shape[0], :s.shape[1], :s.shape[2]] = s
x = torch.from_numpy(blk)[None, None].to(dev)
print("输入块", e["ds"], e["name"], "体积", v.shape, "块", tuple(x.shape), flush=True)


def build(extra):
    torch.manual_seed(0)
    m = CPSAM3D({**meta["cfg"], **extra}, init=meta["init"])
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not unexp, unexp[:5]
    return m.to(dev), sorted({".".join(k.split(".")[:2]) for k in miss if k.startswith(("rcond", "dec.film"))})


def fwd(m, mode, **kw):
    if mode == "infer":
        m.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return m(x, **kw).float().cpu()
    if mode == "fp32":
        m.eval()
        with torch.no_grad():
            return m(x, **kw).float().cpu()
    m.train()                                              # 训练模式：要梯度、走检查点
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = m(x, **kw).float()
    y.sum().backward()                                     # 确认能反传
    m.zero_grad(set_to_none=True)
    return y.detach().cpu()


MODES = ("infer", "fp32", "train")
plain, _ = build({})
Y0 = {md: fwd(plain, md) for md in MODES}
del plain; torch.cuda.empty_cache()
print("不加条件的输出：|y| 均值 " + "  ".join(f"{md} {float(Y0[md].abs().mean()):.5f}" for md in MODES), flush=True)
RES = [(12.7, 1.5, 1.5), (3.17, 6.67, 6.67), (1.0, 1.0, 1.0)]
ok = True
for extra in (dict(cond="res"), dict(cond="res", cond_enc=1), dict(cond="res", cond_enc=2, cond_nfreq=2)):
    m, miss = build(extra)
    nnew = sum(p.numel() for n, p in m.named_parameters() if n.startswith(("rcond", "dec.film")))
    print(f"配置 {extra}：新增参数 {nnew / 1e6:.3f}M，未从权重载入（保持初始化）的模块 {miss}", flush=True)
    for md in MODES:
        kws = [dict(res=torch.tensor([r])) for r in RES] + [dict(res=None), dict(res=torch.tensor([RES[0]]), res_drop=torch.tensor([True]))]
        if md == "train":
            kws = kws[:1] + kws[3:4]                       # 训练模式只比两种（省时间）
        for kw in kws:
            y = fwd(m, md, **kw)
            d = float((y - Y0[md]).abs().max())
            ok &= d == 0.0
            desc = "None（未知嵌入）" if kw["res"] is None else (tuple(kw["res"][0].tolist()) if "res_drop" not in kw else "res_drop=True")
            print(f"  T_b {md:5s} f={str(desc):28s} 与不加条件的最大差 {d:.3e}", flush=True)
    del m; torch.cuda.empty_cache()

# ---- Model3D 完整 eval：cond 模型目录（cfg 加 cond=res、cond_enc=2，权重指向 S1）对 S1 本身
from t_a_infer_lib import run_cases  # noqa: E402
cd = f"{T}/out_b/s1_cond_untrained"
shutil.rmtree(cd, ignore_errors=True); os.makedirs(cd)
mc = dict(meta, cfg={**meta["cfg"], "cond": "res", "cond_enc": 2}, weights=f"{S1}/{meta['weights']}")
json.dump(mc, open(f"{cd}/model3d.json", "w"), indent=1, ensure_ascii=False)
ha = run_cases(S1, [("bbbc050", "train_Emb11_t501", {}), ("cellos", "EGFP_d4_flt32_28", {}), ("bbbc050", "train_Emb11_t501", dict(views=3))])
hb = run_cases(cd, [("bbbc050", "train_Emb11_t501", {}), ("cellos", "EGFP_d4_flt32_28", {}), ("bbbc050", "train_Emb11_t501", dict(views=3))])
for (k, a), (_, b) in zip(ha, hb):
    same = a["dP"] == b["dP"] and a["cp"] == b["cp"]
    ok &= same
    print(f"  T_b Model3D.eval {k}：S1 {a['dP']}/{a['cp']}  cond 未训练 {b['dP']}/{b['cp']}  逐位相同 = {same}；cond 模型的体积 f = {b['res']}", flush=True)
print("T_b 全部差值为 0 =", ok)
