"""训练真三维模型（true3d_spec §4）。

用法: python train3d.py --variant dec3d --init orx_v5_2d_s0 --seed 0 --steps 12000 --out t3d_dec3d_s0 [--freeze_encoder] [--lora 8]
保存到 $NUCSEG3D_ROOT/models/models3d/<out>/：model3d.json（变体、配置、二维初始权重、权重文件名）+ weights.pt
（只存可训练参数；载入时先读二维初始权重再覆盖）。验证体积始终不参与训练（iso30 库里 split=val 的体积不取）。
损失 = MSE(网络流, 5×三维单位流)/2 + BCE(cellprob)，与 Cellpose 相同，只算体积内部的体素。
round 6：--cond res [--cond_enc 0|1|2] [--cond_drop p] [--cond_nfreq n]，分辨率条件化（net3d.ResCond、data3d.block_res），
网络配置写进 model3d.json 的 cfg，infer3d / eval_gt 载入时自动使用；默认 --cond none 时训练与旧版逐位相同。
r9（交叉验证，默认关闭）：--exclude_ids <留出清单.json>（格式同 val_split.json）时，训练体积改为 iso30 里 split 为 train 或 val 的
全部体积（--exclude_ds 照旧）减去清单里的体积；并检查二维初始权重（及 --init3d 的三维模型）是用同一份清单训练的，防止跨折泄漏。
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data3d import ISO_DIR, make_loader, to_patch  # noqa: E402
from net3d import CPSAM3D  # noqa: E402

OUT_ROOT = f"{_NSP_ROOT}/models/models3d"
MODELS = f"{_NSP_ROOT}/models"


def load_exclude_ids(path, store, exclude_ds):
    """r9：读留出清单 {ds: [id, ...]}，返回 {(ds, id)} 与文件 md5；清单里每个体积都必须在 iso30 里（防止拼错编号而没排除掉）。"""
    import hashlib
    raw = open(path, "rb").read()
    ex = {(d, i) for d, ids in json.loads(raw).items() for i in ids}
    from data3d import load_index
    have = {(e["ds"], e["id"]) for e in load_index(store)}
    miss = sorted(x for x in ex if x not in have and x[0] not in exclude_ds)
    assert not miss, f"留出清单里有 {len(miss)} 个体积不在 iso30：{miss[:5]}"
    return ex, hashlib.md5(raw).hexdigest()


def fold_guard(a, md5, cur=None):
    """r9：二维初始权重（train_cpsam2 的 <name>_config.json 里记了 --exclude_json）与 --init3d 的三维模型（model3d.json 的
    train.exclude_ids）必须用内容相同的留出清单训练；否则起点模型见过本折的留出体积。"""
    import hashlib
    h = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest() if p and os.path.exists(p) else None
    if a.guard_parent:                                                 # nc L2：起点模型用“父清单”训练，本次只从父清单里放回少量体积
        par = {(d, i) for d, ids in json.load(open(a.guard_parent)).items() for i in ids}
        back = par - cur
        assert cur <= par, f"--exclude_ids 里有 {len(cur - par)} 个体积不在父清单 {a.guard_parent} 里"
        assert len({d for d, _ in back}) <= 1 and len(back) <= a.guard_max_back, f"放回的体积超出约定：{sorted(back)[:5]}（共 {len(back)} 个）"
        print(f"nc L2：父清单 {a.guard_parent}，放回 {len(back)} 个体积：{sorted(back)}", flush=True)
        md5 = h(a.guard_parent)
    c2 = os.path.join(MODELS, f"{a.init}_config.json")
    ex2 = json.load(open(c2))["args"].get("exclude_json") if os.path.exists(c2) else None
    assert h(ex2) == md5, f"二维初始权重 {a.init} 的留出清单（{ex2}）与 --exclude_ids 内容不同或已不存在"
    if a.init3d:
        ex3 = json.load(open(os.path.join(a.init3d, "model3d.json")))["train"].get("exclude_ids")
        assert h(ex3) == md5, f"--init3d {a.init3d} 的留出清单（{ex3}）与 --exclude_ids 内容不同或已不存在"


def parse_kv(s, typ=float):
    d = {}
    for kv in filter(None, (s or "").split(",")):
        k, v = kv.split("="); d[k] = typ(v)
    return d


def loss_fn(y, tgt, va, w=None, cp_pos_weight=1.0):
    """y: (B,4,Z,H,W) 网络输出；tgt: (B,4,Z,H,W) [单位流 dz,dy,dx, 是否细胞核]；va: (B,Z,H,W)。
    w: 可选逐体素权重 (B,Z,H,W)（界面加权），已归一到有效体素上均值为 1；None 时与旧版逐位相同。
    cp_pos_weight: cellprob BCE 的正类权重（1 = 关闭）。"""
    y = y.float()
    n = va.sum().clamp(min=1)
    vw = va if w is None else va * w                                  # 加权时分母仍是有效体素数 n（w 均值为 1）
    lf = (((y[:, :3] - 5.0 * tgt[:, :3]) ** 2).sum(1) * vw).sum() / (3 * n) / 2
    pw = None if cp_pos_weight == 1.0 else y.new_tensor(cp_pos_weight)
    lc = (F.binary_cross_entropy_with_logits(y[:, 3], tgt[:, 3], reduction="none", pos_weight=pw) * vw).sum() / n
    return lf + lc, lf.detach(), lc.detach()


def cp_round3_loss(ycp, sdt, va, a):
    """round 3：cellprob 的边界目标（只改 cp 项，流项不变）。ycp: (B,Z,H,W) logit；sdt: (B,Z,H,W) 带符号距离（体素，前景为正）。
    --cp_soft τ > 0：软目标 t = σ(s/τ + c₀)，c₀ = cp_soft_at/3（推理时 cellprob = 3×logit，所以评测阈值 cp_soft_at
        对应的等值面正好落在 s = 0 的真值分界面上）；BCE 对软目标的最优 logit 就是 s/τ + c₀，即网络学一个缩放的带符号距离。
    --cp_band_w W > 1：|s| ≤ cp_band_r 的边界带（前景、背景两侧）cp 损失权重 W，再按块归一到有效体素均值 1。
    两者可同时开。返回 cp 损失（分母 = 有效体素数）。"""
    n = va.sum().clamp(min=1)
    if a.cp_soft > 0:
        tgt = torch.sigmoid(sdt / a.cp_soft + a.cp_soft_at / 3.0)
    else:
        tgt = (sdt > 0).float()
    w = va
    if a.cp_band_w != 1.0:
        band = (sdt.abs() <= a.cp_band_r).float()
        wb = 1.0 + (a.cp_band_w - 1.0) * band
        w = va * wb * (n / (va * wb).sum().clamp(min=1e-6))
    return (F.binary_cross_entropy_with_logits(ycp, tgt, reduction="none") * w).sum() / n


def iface_weight(iface, va, W):
    """界面体素权重 W、其余 1，再按块归一：w ← w · n_valid / Σ(w·valid)，使有效体素上权重均值恰为 1（损失量级不变，
    只把梯度在块内往界面挪）。iface/va: (Z,H,W)。"""
    w = 1.0 + (W - 1.0) * iface.float()
    s = (w * va).sum()
    return w * (va.sum().clamp(min=1) / s.clamp(min=1e-6))


def gpu_prefetch(src, dev, patch, depth=3, iface_r=0, sdt_cap=0):
    """后台线程在单独的 CUDA 流上做 to_patch（缩放、增强、三维扩散），与训练步重叠。"""
    import queue, threading
    q = queue.Queue(maxsize=depth)
    stream = torch.cuda.Stream(device=dev)

    def run():
        torch.cuda.set_device(dev)
        try:
            while True:
                s = next(src)[0]
                with torch.cuda.stream(stream):
                    out = to_patch(s, dev, patch, iface_r=iface_r, sdt_cap=sdt_cap)
                stream.synchronize()
                q.put(out)
        except Exception as e:  # 让主线程看到错误
            q.put(e)
    threading.Thread(target=run, daemon=True).start()
    while True:
        out = q.get()
        if isinstance(out, Exception):
            raise out
        yield out


def load_init3d(model, path):
    """载入已训练三维模型（model3d.json 所在目录）的可训练权重。解码器输出层通道数不同（多了 aux 通道）时，
    把旧的前几行拷进去，多出来的行保持零初始化。"""
    meta = json.load(open(os.path.join(path, "model3d.json")))
    assert meta["variant"] == model.variant, (meta["variant"], model.variant)
    sd = torch.load(os.path.join(path, meta["weights"]), map_location="cpu", weights_only=True)
    own = model.state_dict()
    fixed = {}
    for k, v in sd.items():
        if k in own and own[k].shape != v.shape:
            assert k.startswith("dec.out.") and own[k].shape[1:] == v.shape[1:], (k, own[k].shape, v.shape)
            t = own[k].clone(); t[:v.shape[0]] = v; v = t
        fixed[k] = v
    miss, unexp = model.load_state_dict(fixed, strict=False)
    assert not unexp, unexp[:5]
    print(f"init3d：从 {path} 载入 {len(fixed)} 个张量", flush=True)


def lr_at(step, total, warm):
    if step < warm:
        return (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def save(model, cfg, a, out_dir, tag, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    wf = f"weights{tag}.pt"
    torch.save(model.trainable_state_dict(), os.path.join(out_dir, wf + ".part"))
    os.replace(os.path.join(out_dir, wf + ".part"), os.path.join(out_dir, wf))
    meta = dict(variant=cfg["variant"], cfg=cfg, init=a.init, weights=wf, patch_z=a.patch_z,
                infer=dict(patch_z=a.patch_z, overlap_z=0.25, overlap_xy=0.25),
                train=dict(vars(a)), **(extra or {}))
    if tag == "":
        json.dump(meta, open(os.path.join(out_dir, "model3d.json"), "w"), indent=1, ensure_ascii=False)
    return wf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["dec3d", "zadapt", "p25d"])
    ap.add_argument("--init", default="orx_v5_2d_s0", help="二维初始权重名（models/models 下）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--patch_z", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1, help="每步的块数（梯度累积）")
    ap.add_argument("--lr_enc", type=float, default=1e-5)
    ap.add_argument("--lr_dec", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--lora", type=int, default=0)
    ap.add_argument("--cfg", default="{}", help="额外网络配置 JSON，如 '{\"dec_ch\":128,\"adapt_kind\":\"attn\"}'")
    # ---- round 2（默认全部关闭，旧配置逐位复现）
    ap.add_argument("--iface_w", type=float, default=1.0, help="界面体素（窗内 ≥2 个不同核）的损失权重；1 = 关闭")
    ap.add_argument("--iface_r", type=int, default=2, help="界面半径（体素，切比雪夫距离，在训练网格上）")
    ap.add_argument("--cp_pos_weight", type=float, default=1.0, help="cellprob BCE 正类权重；1 = 关闭")
    ap.add_argument("--perm_aug", type=float, default=0.0, help="轴置换增强概率；0 = 关闭")
    ap.add_argument("--zadapt_every", type=int, default=0,
                    help="zadapt：每 N 个 block 后放一个适配器（1 = 每个 block，MA-SAM 式 zconv、瓶颈 64）；0 = 用 cfg 的 adapt_blocks")
    # ---- round 3（默认全部关闭，旧配置逐位复现）：cellprob 边界
    ap.add_argument("--cp_soft", type=float, default=0.0,
                    help="cellprob 软目标温度 τ（体素）：t = σ(s/τ + cp_soft_at/3)，s = 前景/背景带符号距离；0 = 关闭（二值目标）")
    ap.add_argument("--cp_soft_at", type=float, default=-2.0, help="软目标里真值分界面对应的评测 cellprob 阈值（推理单位，= 3×logit）")
    ap.add_argument("--cp_band_w", type=float, default=1.0, help="|s| ≤ cp_band_r 的边界带 cp 损失权重；1 = 关闭")
    ap.add_argument("--cp_band_r", type=float, default=2.0, help="边界带半径（体素，训练网格上，s 的取值为 ±0.5, ±1.5, ...）")
    ap.add_argument("--aux_dist", type=float, default=0.0,
                    help="辅助距离通道的损失权重 λ：解码器多一个输出，回归 clip(s, ±sdt_cap)/sdt_cap（MSE），推理不用；0 = 关闭")
    ap.add_argument("--sdt_cap", type=int, default=8, help="带符号距离的封顶层数")
    # ---- round 4（默认关闭，旧配置逐位复现）：各向异性模拟增强（见 data3d.PatchSource.aniso_params / aniso_sim）
    ap.add_argument("--aniso_aug", type=float, default=0.0,
                    help="各向异性模拟增强的概率：把块沿某轴模拟成原始采样间隔 F 体素、再线性插值回来（只改图像）；0 = 关闭")
    ap.add_argument("--aniso_lo", type=float, default=2.0, help="目标采样间隔 F 的下限（网络网格体素，对数均匀抽样）")
    ap.add_argument("--aniso_hi", type=float, default=13.0, help="F 的上限（cellos 约 12.8，mesospim xy 7.85，parhyale 7.0）")
    ap.add_argument("--aniso_xy", type=float, default=0.0, help="增强作用在 y、x 两轴（mesospim 型）而不是 z 轴的概率")
    ap.add_argument("--aniso_blur", type=float, default=0.0,
                    help="> 0 时取样核为 σ = aniso_blur × f 的高斯（模拟轴向 PSF / 层厚积分），0 = 点取样")
    ap.add_argument("--zstretch_p", type=float, default=0.0,
                    help="round 5：以此概率把训练块沿模型 z 轴单独拉伸/压缩（图像与标签一起），倍数对数均匀取 [1/zstretch_a, zstretch_a]；0 = 关闭")
    ap.add_argument("--zstretch_a", type=float, default=3.0, help="round 5：z 轴拉伸倍数上限")
    ap.add_argument("--aniso_blur_max", type=float, default=0.0,
                    help="round 5：> aniso_blur 时每块的模糊系数在 [aniso_blur, aniso_blur_max] 均匀抽样；0 = 固定为 aniso_blur")
    # ---- round 6（默认全部关闭，旧配置逐位复现）：分辨率条件化（见 net3d.ResCond、data3d.block_res）
    ap.add_argument("--cond", default="none", choices=["none", "res"],
                    help="res = 每块三个轴的源采样间隔 f（网络体素）经 MLP 给三维解码器各块做 FiLM（零初始化）；none = 关闭")
    ap.add_argument("--cond_enc", type=int, default=0, choices=[0, 1, 2],
                    help="配合 --cond res：0 = 编码器不加条件；1 = 条件经零初始化线性层加到输入层之后的 token 上；2 = 加到每个 block 之前")
    ap.add_argument("--cond_drop", type=float, default=0.0, help="配合 --cond res：每块以此概率把条件换成可学习的“未知”向量")
    ap.add_argument("--cond_nfreq", type=int, default=0, help="配合 --cond res：log f 之外再加几组傅里叶特征 sin/cos(2^j·log f)")
    ap.add_argument("--aniso_abs", type=int, default=0, choices=[0, 1],
                    help="round 6：1 = 各向异性模拟的 F 就是网络网格上的实际采样间隔（只在 F > 1.25·g 时生效）；0 = 旧行为（每 F/g 体素取样）")
    ap.add_argument("--init3d", default=None, help="从已训练的三维模型目录（含 model3d.json）载入可训练权重后微调；默认不载入")
    ap.add_argument("--store", "--iso", dest="store", default=ISO_DIR)
    ap.add_argument("--weights", default="cellos=2,tapenade=2", help="数据集抽样权重")
    ap.add_argument("--caps", default="blastospim=0.25", help="数据集份额上限")
    ap.add_argument("--exclude_ds", default="mesospim")
    ap.add_argument("--exclude_val", type=int, default=1, help="恒为 1：验证体积不参与训练")
    ap.add_argument("--exclude_ids", default=None,
                    help="r9 交叉验证：留出清单（JSON，格式同 val_split.json）。给出时训练体积 = iso30 里 split 为 train 或 val 的全部体积"
                         "减去清单里的体积（--exclude_ds 照旧）；不给时行为不变")
    ap.add_argument("--guard_parent", default=None,
                    help="nc L2：起点模型（--init 与 --init3d）所用的留出清单；本次 --exclude_ids 必须是它的子集，且放回的体积只来自一套数据")
    ap.add_argument("--guard_max_back", type=int, default=10, help="nc L2：最多放回几个体积")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--keep_every", type=int, default=0,
                    help="每 N 步另存一份保留的 weights_<步数>.pt（训练结束不删，便于看轨迹）；0 = 关闭")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out_root", default=OUT_ROOT)
    ap.add_argument("--gpu", default="cuda:0")
    ap.add_argument("--zero_dec", action="store_true",
                    help="nc EO 对照（第三轮审稿 SF2，事后）：三维解码器输出层置零并冻结整个解码器，只训练编码器与二维头；默认关闭，关闭时与原版逐位相同")
    a = ap.parse_args()
    assert a.exclude_val == 1, "验证体积必须排除"
    exd = tuple(filter(None, a.exclude_ds.split(",")))
    cv_ex, cv_md5 = (None, None)
    if a.exclude_ids:                                                  # r9：交叉验证，本折留出体积代替验证集
        cv_ex, cv_md5 = load_exclude_ids(a.exclude_ids, a.store, exd)
        fold_guard(a, cv_md5, cv_ex)
        print(f"r9 交叉验证：留出清单 {a.exclude_ids}（md5 {cv_md5}，{len(cv_ex)} 个体积）；split=val 的其余体积参与训练", flush=True)

    dev = torch.device(a.gpu); torch.cuda.set_device(dev)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    extra = json.loads(a.cfg)
    if a.zadapt_every:
        assert a.variant == "zadapt", "--zadapt_every 只用于 zadapt"
        extra = {"adapt_blocks": list(range(a.zadapt_every - 1, 24, a.zadapt_every)), "adapt_kind": "zconv", "adapt_dim": 64, **extra}
    if a.aux_dist > 0:
        extra = {"aux_ch": 1, **extra}
    use_cond = a.cond == "res"
    if use_cond:
        extra = {"cond": "res", "cond_enc": a.cond_enc, "cond_nfreq": a.cond_nfreq, "cond_drop": a.cond_drop, **extra}
    else:
        assert a.cond_enc == 0 and a.cond_drop == 0 and a.cond_nfreq == 0, "--cond_enc/--cond_drop/--cond_nfreq 要配合 --cond res"
    cfg = dict(variant=a.variant, freeze_encoder=a.freeze_encoder, lora=a.lora, patch_z=a.patch_z, **extra)
    print("训练目标：三维流（dz,dy,dx 单位向量 ×5）+ cellprob；变体", a.variant, "配置", cfg, flush=True)
    model = CPSAM3D(cfg, init=a.init).to(dev).train()
    cfg = model.cfg
    if a.init3d:
        load_init3d(model, a.init3d)
    if a.zero_dec:                                                     # nc EO（第三轮审稿 SF2，事后对照）：三维解码器输出层恒为零
        assert hasattr(model, "dec"), "--zero_dec 只用于带三维解码器的变体（dec3d / zadapt）"
        with torch.no_grad():
            model.dec.out.weight.zero_(); model.dec.out.bias.zero_()
        for p_ in model.dec.parameters():
            p_.requires_grad_(False)
        print("EO：三维解码器输出层置零并冻结（残差 r ≡ 0，dz ≡ 0），只训练编码器与二维头", flush=True)
    enc_p, new_p = model.param_groups()
    print(f"可训练参数：编码器侧 {sum(p.numel() for p in enc_p) / 1e6:.1f}M（lr {a.lr_enc}），"
          f"新增模块 {sum(p.numel() for p in new_p) / 1e6:.1f}M（lr {a.lr_dec}）", flush=True)
    groups = [g for g in (dict(params=enc_p, lr=a.lr_enc, base_lr=a.lr_enc),
                          dict(params=new_p, lr=a.lr_dec, base_lr=a.lr_dec)) if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=a.wd, betas=(0.9, 0.999))
    params = [p for g in groups for p in g["params"]]

    patch = (a.patch_z, 256, 256)
    zstretch = dict(p=a.zstretch_p, a=a.zstretch_a) if a.zstretch_p > 0 else None
    if zstretch:
        print(f"round-5 选项：z 轴单独拉伸 {zstretch}", flush=True)
    aniso = None
    if a.aniso_aug > 0:
        aniso = dict(p=a.aniso_aug, lo=a.aniso_lo, hi=a.aniso_hi, xy=a.aniso_xy, blur=a.aniso_blur)
        if a.aniso_blur_max > 0:
            aniso["blur_max"] = a.aniso_blur_max
        if a.aniso_abs:
            aniso["abs"] = 1
        print(f"round-4 选项：各向异性模拟增强 {aniso}", flush=True)
    if use_cond:
        print(f"round-6 选项：分辨率条件 cond=res cond_enc={a.cond_enc} cond_drop={a.cond_drop} cond_nfreq={a.cond_nfreq}", flush=True)
    cv_kw = dict(include_val=True, exclude_ids=cv_ex) if a.exclude_ids else {}
    dl, idx, share = make_loader(a.store, "train", parse_kv(a.weights), parse_kv(a.caps), patch, workers=a.workers,
                                 seed=a.seed, exclude_ds=exd, perm_aug=a.perm_aug,
                                 aniso=aniso, zstretch=zstretch, res_cond=use_cond, **cv_kw)
    if a.exclude_ids:
        assert not any((e["ds"], e["id"]) in cv_ex for e in idx)
        print(f"训练体积 {len(idx)} 个（本折留出体积已排除，其中原验证体积 {sum(e['split'] == 'val' for e in idx)} 个参与训练），"
              f"数据集份额 {share}", flush=True)
    else:
        print(f"训练体积 {len(idx)} 个（验证体积已排除），数据集份额 {share}", flush=True)
    out_dir = a.out if os.path.isabs(a.out) else os.path.join(a.out_root, a.out)
    os.makedirs(out_dir, exist_ok=True)
    use_iface = a.iface_w != 1.0
    use_r3 = a.cp_soft > 0 or a.cp_band_w != 1.0 or a.aux_dist > 0
    if use_r3:
        print(f"round-3 选项：cp_soft={a.cp_soft} cp_soft_at={a.cp_soft_at} cp_band_w={a.cp_band_w} cp_band_r={a.cp_band_r} "
              f"aux_dist={a.aux_dist} sdt_cap={a.sdt_cap} init3d={a.init3d}", flush=True)
        assert not use_iface and a.cp_pos_weight == 1.0, "round-3 cp 目标不与 iface_w / cp_pos_weight 混用"
    if use_iface or a.cp_pos_weight != 1.0 or a.perm_aug > 0:
        print(f"round-2 选项：iface_w={a.iface_w} iface_r={a.iface_r} cp_pos_weight={a.cp_pos_weight} perm_aug={a.perm_aug}", flush=True)
    it = gpu_prefetch(iter(dl), dev, patch, depth=3, iface_r=a.iface_r if use_iface else 0, sdt_cap=a.sdt_cap if use_r3 else 0)
    ifr = []; auxs = []
    log, hist = [], []
    ck = {}                                                           # round 6：传给网络的条件（关闭时为空，调用与旧版相同）
    crng = np.random.default_rng([a.seed, 6]) if a.cond_drop > 0 else None   # 条件 dropout 用单独的随机数流
    t0 = time.time(); tdata = 0.0
    torch.cuda.reset_peak_memory_stats(dev)
    for step in range(a.steps):
        f = lr_at(step, a.steps, a.warmup)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        opt.zero_grad(set_to_none=True)
        tot = lfs = lcs = 0.0
        for _ in range(a.batch):
            td = time.time()
            b = next(it)
            if use_cond:                                              # to_patch 把 res 放在最后
                ck = dict(res=b[-1][None])
                if crng is not None:
                    ck["res_drop"] = torch.tensor([crng.random() < a.cond_drop], device=dev)
                b = b[:-1]
            x, tgt, va = b[:3]
            w = None
            if use_iface:
                w = iface_weight(b[3], va, a.iface_w)[None]
                ifr.append(float((b[3].float() * va).sum() / va.sum().clamp(min=1)))
            tdata += time.time() - td
            if use_r3:
                sdt = b[-1][None]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x[None], return_aux=a.aux_dist > 0, **ck)
                y, ya = out if a.aux_dist > 0 else (out, None)
                y = y.float(); vb = va[None]; n = vb.sum().clamp(min=1)
                lfg = (((y[:, :3] - 5.0 * tgt[None][:, :3]) ** 2).sum(1) * vb).sum() / (3 * n) / 2   # 与 loss_fn 的流项相同
                lcg = cp_round3_loss(y[:, 3], sdt, vb, a)
                loss = lfg + lcg
                if ya is not None:
                    D = float(a.sdt_cap)
                    la = (((ya[:, 0].float() - sdt.clamp(-D, D) / D) ** 2) * vb).sum() / n
                    loss = loss + a.aux_dist * la
                    auxs.append(float(la.detach()))
                lf, lc = lfg.detach(), lcg.detach()
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = model(x[None], **ck)
                loss, lf, lc = loss_fn(y, tgt[None], va[None], w, a.cp_pos_weight)
            (loss / a.batch).backward()
            tot += loss.item() / a.batch; lfs += float(lf) / a.batch; lcs += float(lc) / a.batch
        gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
        opt.step()
        hist.append((tot, lfs, lcs, float(gn)))
        if (step + 1) % a.log_every == 0 or step == 0:
            h = np.array(hist[-a.log_every:])
            el = time.time() - t0
            rec = dict(step=step + 1, loss=h[:, 0].mean(), flow=h[:, 1].mean(), cp=h[:, 2].mean(), gnorm=h[:, 3].mean(),
                       lr=f, s_per_step=el / (step + 1), data_s=tdata / (step + 1),
                       mem_gb=torch.cuda.max_memory_allocated(dev) / 2**30)
            if not np.isfinite(rec["loss"]):
                print(f"警告：损失非有限值（步 {step + 1}）", flush=True)
            if use_iface:
                rec["iface_frac"] = float(np.mean(ifr[-a.log_every * a.batch:]))
            if auxs:
                rec["aux"] = float(np.mean(auxs[-a.log_every * a.batch:]))
            if use_cond:                                              # round 6：本步最后一块的 f（模型 z,y,x）
                rec["res"] = [round(v, 3) for v in ck["res"][0].tolist()]
            log.append(rec)
            print(f"步 {step + 1}/{a.steps} 损失 {rec['loss']:.4f}（流 {rec['flow']:.4f} 概率 {rec['cp']:.4f}）"
                  f" 梯度范数 {rec['gnorm']:.2f} lr×{f:.3f} {rec['s_per_step']:.2f}s/步（取数 {rec['data_s']:.2f}）"
                  f" 峰值显存 {rec['mem_gb']:.1f}GB" + (f" 界面体素 {rec['iface_frac']:.3f}" if use_iface else "")
                  + (f" f={rec['res']}" if use_cond else ""), flush=True)
        if a.save_every and (step + 1) % a.save_every == 0 and step + 1 < a.steps:
            save(model, cfg, a, out_dir, "_last")
            json.dump(log, open(os.path.join(out_dir, "train_log.json"), "w"), indent=1)
        if a.keep_every and (step + 1) % a.keep_every == 0 and step + 1 < a.steps:
            save(model, cfg, a, out_dir, f"_{step + 1}")             # 只写权重；评测时复制 model3d.json 并把 weights 改成此文件名
    wf = save(model, cfg, a, out_dir, "", extra=dict(final_loss=log[-1]["loss"] if log else None))
    for p in ("weights_last.pt",):
        if os.path.exists(os.path.join(out_dir, p)):
            os.remove(os.path.join(out_dir, p))
    json.dump(log, open(os.path.join(out_dir, "train_log.json"), "w"), indent=1)
    print("TRAIN3D_DONE", out_dir, wf, flush=True)


if __name__ == "__main__":
    main()
