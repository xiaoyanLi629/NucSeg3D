"""真三维网络：在二维 Cellpose-SAM（CPSAM）编码器上加三维部分，输出 (B,4,Z,256,256) = [dz, dy, dx, cellprob]。

三个变体（cfg["variant"]）：
  "dec3d"  (A)  每个 z 层用二维编码器 + neck 得到 (B,256,Z,32,32) 特征，送进三维残差解码器（GroupNorm、3x3x3 卷积，
               z 向可设膨胀）。解码器最后一层零初始化，输出加到原二维输出头（dy,dx,cp）的 token 上，dz 只来自三维支路；
               和二维头一样按每个 token 8x8 像素展开。cfg: freeze_encoder（A0 冻结 / A1 全量微调）、lora（qkv 上的秩）。
  "zadapt" (C)  A 再加 z 向适配器：在指定 block 之后，把 (B*Z,32,32,1024) token 降到 128 维，沿 z 混合
               （3x3x3 卷积或沿 z 的注意力），升回 1024 维加回去；升维层零初始化，初始时恒等。
               adapt_kind="zconv"（round 2，train3d --zadapt_every N）：MA-SAM 式 LN→down→Conv3d(3×1×1)→GELU→up，
               可放在每个 block 之后（adapt_blocks=[0..23]）。
  "p25d"   (B)  2.5D：z-1、z、z+1 三层放进二维网络的三个输入通道（z 在第 0 通道，与训练时的灰度图一致；
               另两通道的 patch-embed 权重零初始化，初始时与二维网络逐层输出完全相同），二维头加一个零初始化的 dz 输出，
               逐层计算，没有三维解码器。

数值约定：网络输出的流是 5×单位向量（与 Cellpose 训练目标一致），cellprob 是单个 logit。
注意力换成 F.scaled_dot_product_attention（相对位置偏置作为 attn_mask），与原实现数值一致（见 __main__ 自检）。

round 6（分辨率条件化，cfg cond="res"，默认 None = 关闭、不建任何新模块）：forward 多收一个 res = 块的 (f_z, f_y, f_x)，
即模型三个轴上“一个源采样间隔等于多少个网络体素”。ResCond 把 log f（可加傅里叶特征）经两层 MLP 变成嵌入，
三维解码器每个残差块（第二个 GroupNorm 之后）和输出前的 GroupNorm 由零初始化线性层得到按通道的 FiLM h·(1+γ)+β；
cond_enc=1/2 时嵌入再经零初始化线性层加到编码器输入层之后 / 每个 block 之前的 token 上。零初始化保证训练前输出
与不加条件的同权重网络逐位相同。res=None 或 res_drop 为 True 的样本用可学习的“未知”嵌入（条件 dropout）。
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

MODELS2D = f"{_NSP_ROOT}/models/models"

DEFAULT_CFG = dict(variant="dec3d", freeze_encoder=False, lora=0, lora_alpha=0,
                   dec_ch=256, dec_blocks=4, dec_zdil=[1, 2, 4, 8], dec_groups=16,
                   adapt_blocks=[5, 11, 17, 23], adapt_dim=128, adapt_kind="conv",
                   chunk=16, ckpt=True, patch_z=32, aux_ch=0)
# aux_ch（round 3）：三维解码器多出的辅助输出通道数（训练时的边界距离辅助目标，推理不用）；0 = 关闭，网络与旧版逐位相同。
COND_CFG = dict(cond=None, cond_enc=0, cond_nfreq=0, cond_hid=128, cond_drop=0.0)
# round 6：cfg 里 cond="res" 时才用 COND_CFG 补全这些键（关闭时 cfg 与旧版完全相同，model3d.json 也不多键，avg_weights 照常比对）。
# cond_enc：0 = 编码器不加条件，1 = 加在输入层之后，2 = 加在每个 block 之前；cond_nfreq：log f 之外的傅里叶特征组数；
# cond_hid：条件 MLP 宽度；cond_drop：训练时的条件 dropout 概率（只作记录，丢弃由 train3d 决定）。


# ---------------------------------------------------------------- SDPA attention
def sdpa_forward(self, x):
    """SAM Attention.forward 的 SDPA 版本：相对位置偏置 rel_h[...,None] + rel_w[...,None,:] 作为加性 mask。"""
    from segment_anything.modeling.image_encoder import get_rel_pos
    B, H, W, _ = x.shape
    nh = self.num_heads
    qkv = self.qkv(x).reshape(B, H * W, 3, nh, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)                                   # (B, nh, HW, hd)
    bias = None
    if self.use_rel_pos:
        Rh = get_rel_pos(H, H, self.rel_pos_h).to(q.dtype)
        Rw = get_rel_pos(W, W, self.rel_pos_w).to(q.dtype)
        rq = q.reshape(B * nh, H, W, -1)
        rel_h = torch.einsum("bhwc,hkc->bhwk", rq, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", rq, Rw)
        bias = (rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).reshape(B, nh, H * W, H * W)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    o = o.permute(0, 2, 1, 3).reshape(B, H, W, -1)
    return self.proj(o)


class LoRA(nn.Module):
    def __init__(self, base, r, alpha=0):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = (alpha or r) / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t().to(x.dtype) @ self.B.t().to(x.dtype)) * self.scale


# ---------------------------------------------------------------- 3D modules
def film(h, p):
    """round 6：h (B,C,...) 按通道调制 h·(1+γ)+β，p (B,2,C) = (γ, β)。γ=β=0 时逐位等于 h。"""
    s = (slice(None), slice(None)) + (None,) * (h.dim() - 2)
    return h * (1 + p[:, 0][s]) + p[:, 1][s]


class ResBlock3D(nn.Module):
    def __init__(self, ch, groups, zdil):
        super().__init__()
        self.n1 = nn.GroupNorm(groups, ch); self.c1 = nn.Conv3d(ch, ch, 3, padding=(zdil, 1, 1), dilation=(zdil, 1, 1))
        self.n2 = nn.GroupNorm(groups, ch); self.c2 = nn.Conv3d(ch, ch, 3, padding=1)

    def forward(self, x, fl=None):
        h = self.c1(F.gelu(self.n1(x)))
        if fl is None:
            h = self.c2(F.gelu(self.n2(h)))
        else:                                                   # round 6：第二个 GroupNorm 之后做 FiLM
            h = self.c2(F.gelu(film(self.n2(h), fl)))
        return x + h


class Dec3D(nn.Module):
    """neck 特征 (B,256,Z,32,32) -> token 空间残差 (B, nout*64, Z, 32, 32)，最后一层零初始化。
    film_dim > 0（round 6）：条件嵌入 (B, film_dim) 经零初始化线性层给每个残差块和输出前的 GroupNorm 各产生一组 (γ, β)。"""
    def __init__(self, cin=256, ch=256, nblocks=4, zdil=(1, 2, 4, 8), groups=16, nout=4, ps=8, film_dim=0):
        super().__init__()
        zdil = list(zdil) + [1] * max(0, nblocks - len(zdil))
        self.inp = nn.Conv3d(cin, ch, 1)
        self.blocks = nn.ModuleList([ResBlock3D(ch, groups, zdil[i]) for i in range(nblocks)])
        self.nout_ = nn.GroupNorm(groups, ch)
        self.out = nn.Conv3d(ch, nout * ps * ps, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        if film_dim:
            self.film = nn.Linear(film_dim, (nblocks + 1) * 2 * ch)
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, f, c=None):
        if c is None:
            x = self.inp(f)
            for b in self.blocks:
                x = b(x)
            return self.out(F.gelu(self.nout_(x)))
        with torch.autocast("cuda", enabled=False):             # FiLM 参数在 fp32 下算
            fl = self.film(c.float()).view(c.shape[0], len(self.blocks) + 1, 2, -1)    # (B, 块数+1, γ/β, ch)
        x = self.inp(f)
        for i, b in enumerate(self.blocks):
            x = b(x, fl[:, i])
        return self.out(F.gelu(film(self.nout_(x), fl[:, -1])))


class ResCond(nn.Module):
    """round 6：分辨率条件 f (B,3)（模型 z,y,x 三个轴一个源采样间隔等于多少网络体素）-> 嵌入 (B,hid)。
    输入 log f 与 nfreq 组 sin/cos(2^j·log f)，两层 MLP；全程 fp32。f=None 时整批、drop (B,) 为 True 的样本
    换成可学习的“未知”嵌入 null（条件 dropout 用）。"""
    def __init__(self, nfreq=0, hid=128):
        super().__init__()
        self.nfreq = int(nfreq)
        self.mlp = nn.Sequential(nn.Linear(3 * (1 + 2 * self.nfreq), hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU())
        self.null = nn.Parameter(torch.zeros(hid))

    def forward(self, f, B, drop=None):
        with torch.autocast("cuda", enabled=False):
            if f is None:
                return self.null.float()[None].expand(B, -1)
            lf = torch.log(f.float().clamp(min=1e-3))
            feats = [lf] + [g(lf * 2.0 ** j) for j in range(self.nfreq) for g in (torch.sin, torch.cos)]
            h = self.mlp(torch.cat(feats, -1))
            if drop is not None:
                drop = torch.as_tensor(drop, dtype=torch.bool, device=h.device).reshape(B, 1)
                h = torch.where(drop, self.null.float()[None].expand_as(h), h)
        return h


def sincos_z(Z, dim, device, dtype):
    pos = torch.arange(Z, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(dim // 2, device=device, dtype=torch.float32)[None]
    ang = pos / (10000 ** (2 * i / dim))
    return torch.cat([ang.sin(), ang.cos()], 1).to(dtype)     # (Z, dim)


class ZAdapter(nn.Module):
    """token (B*Z,H,W,C) -> 加上沿 z 混合后的瓶颈残差；up 零初始化，初始时恒等。"""
    def __init__(self, dim=1024, hid=128, kind="conv", heads=4):
        super().__init__()
        self.kind, self.heads = kind, heads
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hid)
        if kind == "conv":
            self.mix = nn.Conv3d(hid, hid, 3, padding=1)
        elif kind == "zconv":    # MA-SAM 式：LN → down → Conv3d(3×1×1) 沿 z → GELU → up（零初始化）
            self.mix = nn.Conv3d(hid, hid, (3, 1, 1), padding=(1, 0, 0))
        elif kind == "attn":
            self.qkv = nn.Linear(hid, 3 * hid); self.proj = nn.Linear(hid, hid)
        else:
            raise ValueError(kind)
        self.up = nn.Linear(hid, dim)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, x, B, Z):
        N, H, W, C = x.shape
        if self.kind == "zconv":
            h = self.down(self.norm(x))
            d = h.shape[-1]
            h = h.view(B, Z, H, W, d).permute(0, 4, 1, 2, 3)
            h = F.gelu(self.mix(h)).permute(0, 2, 3, 4, 1).reshape(N, H, W, d)
            return x + self.up(h)
        h = F.gelu(self.down(self.norm(x)))                   # (B*Z,H,W,hid)
        d = h.shape[-1]
        if self.kind == "conv":
            h = h.view(B, Z, H, W, d).permute(0, 4, 1, 2, 3)
            h = F.gelu(self.mix(h)).permute(0, 2, 3, 4, 1).reshape(N, H, W, d)
        else:
            t = h.view(B, Z, H * W, d).permute(0, 2, 1, 3).reshape(B * H * W, Z, d)
            t = t + sincos_z(Z, d, t.device, t.dtype)[None]
            q, k, v = self.qkv(t).view(B * H * W, Z, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
            o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B * H * W, Z, d)
            h = F.gelu(self.proj(o)).view(B, H * W, Z, d).permute(0, 2, 1, 3).reshape(N, H, W, d)
        return x + self.up(h)


# ---------------------------------------------------------------- main model
def load_cpsam2d(init):
    """init: 二维权重名（models/models 下）或路径。返回 fp32 的 cellpose.vit.CPSAM。"""
    import os
    from cellpose.vit import CPSAM
    path = init if os.path.exists(init) else f"{MODELS2D}/{init}"
    net = CPSAM(nout=3)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {(k[7:] if k.startswith("module.") else k): v.float() for k, v in sd.items()}
    miss, unexp = net.load_state_dict(sd, strict=False)
    if miss or unexp:
        raise RuntimeError(f"二维权重键不匹配: missing={miss[:5]} unexpected={unexp[:5]}")
    return net.float()


class CPSAM3D(nn.Module):
    def __init__(self, cfg, init=None):
        super().__init__()
        cfg = {**DEFAULT_CFG, **(cfg or {})}
        if cfg.get("cond"):                                   # round 6：开了条件才补全条件相关的键
            cfg = {**COND_CFG, **cfg}
        self.cfg = cfg
        self.variant = cfg["variant"]
        assert self.variant in ("dec3d", "zadapt", "p25d"), self.variant
        net = load_cpsam2d(init) if init else _empty_cpsam()
        self.net = net
        self.ps = net.ps
        for blk in net.encoder.blocks:
            blk.attn.forward = types.MethodType(sdpa_forward, blk.attn)
        if cfg["lora"]:
            for blk in net.encoder.blocks:
                blk.attn.qkv = LoRA(blk.attn.qkv, cfg["lora"], cfg["lora_alpha"])
        cond = cfg.get("cond")
        if cond:
            assert cond == "res" and self.variant in ("dec3d", "zadapt"), (cond, self.variant)
            assert int(cfg["cond_enc"]) in (0, 1, 2), cfg["cond_enc"]
        if self.variant in ("dec3d", "zadapt"):
            self.dec = Dec3D(256, cfg["dec_ch"], cfg["dec_blocks"], cfg["dec_zdil"], cfg["dec_groups"], 4 + int(cfg["aux_ch"]), self.ps,
                             film_dim=int(cfg["cond_hid"]) if cond else 0)
        else:
            assert not cfg["aux_ch"], "aux_ch 只用于 dec3d / zadapt"
        if self.variant == "zadapt":
            self.adapters = nn.ModuleDict({str(i): ZAdapter(1024, cfg["adapt_dim"], cfg["adapt_kind"])
                                           for i in cfg["adapt_blocks"]})
        if self.variant == "p25d":
            w = net.encoder.patch_embed.proj.weight
            with torch.no_grad():
                w[:, 1:].zero_()                            # 邻层通道零初始化 -> 初始时与二维完全一致
            self.out_dz = nn.Conv2d(net.out.in_channels, self.ps * self.ps, 1)
            nn.init.zeros_(self.out_dz.weight); nn.init.zeros_(self.out_dz.bias)
        if cond:                                              # round 6：条件 MLP（含“未知”嵌入）与可选的编码器 token 偏置
            self.rcond = ResCond(cfg["cond_nfreq"], int(cfg["cond_hid"]))
            n_enc = {0: 0, 1: 1, 2: len(net.encoder.blocks)}[int(cfg["cond_enc"])]
            if n_enc:
                self.rcond_enc = nn.Linear(int(cfg["cond_hid"]), n_enc * net.encoder.pos_embed.shape[-1])
                nn.init.zeros_(self.rcond_enc.weight); nn.init.zeros_(self.rcond_enc.bias)
        self._set_trainable()

    # ---- 哪些参数训练
    def _set_trainable(self):
        enc = self.net
        lora = self.cfg["lora"] > 0
        frozen = self.cfg["freeze_encoder"] or lora
        for n, p in enc.named_parameters():
            p.requires_grad = not frozen
            if n.endswith(("W2", "diam_mean", "diam_labels")):
                p.requires_grad = False
            if lora and (n.endswith(".A") or n.endswith(".B")):
                p.requires_grad = True
        pe = enc.encoder.patch_embed.proj
        # Cellpose 训练时 patch-embed 用 .data（不更新）；A/C 保持这一点，B 的邻层通道必须能学
        pe.weight.requires_grad = self.variant == "p25d"
        pe.bias.requires_grad = False
        if lora:   # LoRA 时 neck 和二维头也训练（与 train_cpsam2 --lora 一致）
            for p in list(enc.encoder.neck.parameters()) + list(enc.out.parameters()):
                p.requires_grad = True

    def param_groups(self):
        """(编码器侧参数, 新增模块参数)。"""
        new_names = ("dec.", "adapters.", "out_dz.", "rcond")
        enc, new = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_new = n.startswith(new_names) or (n.endswith((".A", ".B")) and ".qkv." in n)
            if self.variant == "p25d" and n == "net.encoder.patch_embed.proj.weight":
                is_new = True
            (new if is_new else enc).append(p)
        return enc, new

    def trainable_state_dict(self):
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}

    # ---- 编码器
    def _embed(self, x):
        pe = self.net.encoder.patch_embed.proj
        w = pe.weight if self.variant == "p25d" else pe.weight.detach()
        x = F.conv2d(x, w[:, :x.shape[1]], bias=pe.bias.detach(), stride=self.ps)
        return x.permute(0, 2, 3, 1) + self.net.encoder.pos_embed

    def _enc_needs_grad(self):
        return any(p.requires_grad for p in self.net.encoder.parameters())

    def encode(self, xs, B, Z, eoff=None):
        """xs: (B*Z, C, 256, 256) -> neck 特征 (B*Z, 256, 32, 32)。
        eoff（round 6）：None，或 (B*Z, n, 1024) 的条件 token 偏置；n = 1 加在输入层之后，n = block 数时加在每个 block 之前。"""
        enc = self.net.encoder
        grad = torch.is_grad_enabled()
        enc_grad = grad and self._enc_needs_grad()
        adapt = getattr(self, "adapters", None)
        first_ad = min(int(k) for k in adapt) if adapt else len(enc.blocks)
        cgrad = grad and eoff is not None and eoff.requires_grad        # round 6：编码器冻结时也要让梯度传到 token 偏置
        # 冻结编码器、没有适配器：整段 no_grad 分块算（省显存）
        if not enc_grad and not adapt and not cgrad:
            ck = self.cfg["chunk"]
            with torch.no_grad():
                outs = [self._encode_plain(xs[i:i + ck], None if eoff is None else eoff[i:i + ck]) for i in range(0, len(xs), ck)]
            return torch.cat(outs)
        use_ck = grad and self.cfg["ckpt"]
        with torch.set_grad_enabled(enc_grad):
            x = self._embed(xs)
        if eoff is not None and eoff.shape[1] == 1:
            x = x + eoff[:, 0, None, None, :]
        for i, blk in enumerate(enc.blocks):
            need = enc_grad or (grad and adapt is not None and i >= first_ad) or cgrad
            with torch.set_grad_enabled(need):
                if eoff is not None and eoff.shape[1] > 1:
                    x = x + eoff[:, i, None, None, :]
                x = checkpoint(blk, x, use_reentrant=False) if (use_ck and need) else blk(x)
                if adapt is not None and str(i) in adapt:
                    a = adapt[str(i)]
                    x = checkpoint(a, x, B, Z, use_reentrant=False) if use_ck else a(x, B, Z)
        return enc.neck(x.permute(0, 3, 1, 2))

    def _encode_plain(self, xs, eoff=None):
        x = self._embed(xs)
        if eoff is not None and eoff.shape[1] == 1:
            x = x + eoff[:, 0, None, None, :]
        for i, blk in enumerate(self.net.encoder.blocks):
            if eoff is not None and eoff.shape[1] > 1:
                x = x + eoff[:, i, None, None, :]
            x = blk(x)
        return self.net.encoder.neck(x.permute(0, 3, 1, 2))

    # ---- 前向
    def forward(self, x, return_aux=False, res=None, res_drop=None):
        """x: (B, 1, Z, 256, 256) 归一化后的灰度体积 -> (B, 4, Z, 256, 256) = [dz, dy, dx, cp]。
        return_aux=True（且 cfg aux_ch>0）时返回 (y, aux)，aux: (B, aux_ch, Z, 256, 256)；推理接口不变。
        res / res_drop（round 6，只在 cfg cond="res" 时用，否则忽略）：res 为 (B,3) 的 (f_z, f_y, f_x)（模型轴，网络体素/源采样
        间隔），None 时整批用“未知”嵌入；res_drop 为 (B,) bool，True 的样本用“未知”嵌入（训练时的条件 dropout）。"""
        B, C, Z, H, W = x.shape
        assert C == 1 and H == 256 and W == 256, x.shape
        c = eoff = None
        if self.cfg.get("cond"):
            if res is not None:
                res = torch.as_tensor(res, dtype=torch.float32, device=x.device).reshape(B, 3)
            c = self.rcond(res, B, res_drop)                                     # (B, cond_hid)，fp32
            if hasattr(self, "rcond_enc"):
                with torch.autocast("cuda", enabled=False):
                    eoff = self.rcond_enc(c).view(B, -1, self.net.encoder.pos_embed.shape[-1]).repeat_interleave(Z, 0)
        if self.variant == "p25d":
            xp = torch.cat([x[:, 0, :1], x[:, 0], x[:, 0, -1:]], 1)            # (B,Z+2,H,W)，两端复制
            xs =torch.stack([xp[:, 1:Z + 1], xp[:, 0:Z], xp[:, 2:Z + 2]], 2)   # (B,Z,3,H,W)：中心, z-1, z+1
            xs = xs.reshape(B * Z, 3, H, W)
            outs = []
            step = self.cfg["chunk"] if not torch.is_grad_enabled() else len(xs)
            for i in range(0, len(xs), step):
                f = self.encode(xs[i:i + step], B, Z)
                tok = torch.cat([self.out_dz(f), self.net.out(f)], 1)            # (n, 4*64, 32, 32)
                outs.append(F.pixel_shuffle(tok, self.ps))
            y = torch.cat(outs)
        else:
            xs = x.permute(0, 2, 1, 3, 4).reshape(B * Z, 1, H, W)
            f = self.encode(xs, B, Z, eoff)                                      # (B*Z,256,32,32)
            tok2d = self.net.out(f)                                              # (B*Z,192,32,32)
            f3 = f.view(B, Z, *f.shape[1:]).permute(0, 2, 1, 3, 4)               # (B,256,Z,32,32)
            r = self.dec(f3, c).permute(0, 2, 1, 3, 4).reshape(B * Z, -1, *f.shape[-2:])
            pp = self.ps * self.ps
            na = int(self.cfg["aux_ch"])
            if na:
                ra = r[:, 4 * pp:]; r = r[:, :4 * pp]
            tok = torch.cat([r[:, :pp], tok2d + r[:, pp:]], 1)
            y = F.pixel_shuffle(tok, self.ps)                                    # (B*Z,4,H,W)
            if na and return_aux:
                aux = F.pixel_shuffle(ra, self.ps).view(B, Z, na, H, W).permute(0, 2, 1, 3, 4)
                return y.view(B, Z, 4, H, W).permute(0, 2, 1, 3, 4), aux
        return y.view(B, Z, 4, H, W).permute(0, 2, 1, 3, 4)


def _empty_cpsam():
    from cellpose.vit import CPSAM
    return CPSAM(nout=3).float()


def build_from_dir(path, device="cuda"):
    """读 models3d/<name>/model3d.json，建网络并载入权重。"""
    import json, os
    meta = json.load(open(os.path.join(path, "model3d.json")))
    m = CPSAM3D(meta["cfg"], init=meta["init"])
    sd = torch.load(os.path.join(path, meta["weights"]), map_location="cpu", weights_only=True)
    miss, unexp = m.load_state_dict(sd, strict=False)
    if unexp:
        raise RuntimeError(f"多余的键: {unexp[:5]}")
    return m.to(device).eval(), meta


# ---------------------------------------------------------------- 自检
if __name__ == "__main__":
    import argparse, time
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="orx_v5_2d_s0")
    ap.add_argument("--gpu", default="cuda:1")
    a = ap.parse_args()
    dev = torch.device(a.gpu)
    torch.manual_seed(0)
    # 一块真实图像：iso_tiny 里的第一个体积（没有就用随机图）
    try:
        import json
        idx = json.load(open(f"{_NSP_SCRATCH}/true3d_dev/iso_tiny/index.json"))
        e = idx[0]
        v = np.load(e["img"] if e["img"].startswith("/") else f"{_NSP_SCRATCH}/true3d_dev/iso_tiny/" + e["img"], mmap_mode="r")
        z0 = max(0, v.shape[0] // 2 - 4); y0 = max(0, v.shape[1] // 2 - 128); x0 = max(0, v.shape[2] // 2 - 128)
        blk = np.zeros((8, 256, 256), np.float32)
        s = np.asarray(v[z0:z0 + 8, y0:y0 + 256, x0:x0 + 256], np.float32)
        blk[:s.shape[0], :s.shape[1], :s.shape[2]] = s
        x = torch.from_numpy(blk)[None, None].to(dev)
        print("测试图像", e["ds"], e["name"])
    except Exception as ex:
        print("没有 iso_tiny，用随机图", ex)
        x = torch.rand(1, 1, 8, 256, 256, device=dev)

    # 1) SDPA 与原注意力：fp32 下逐块比较
    ref = load_cpsam2d(a.init).to(dev).eval()
    xs = x[0].permute(1, 0, 2, 3)                                      # (Z,1,H,W)
    with torch.no_grad():
        y_ref32 = ref(xs)[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y_ref16 = ref(xs)[0].float()
    m = CPSAM3D(dict(variant="dec3d"), init=a.init).to(dev).eval()
    with torch.no_grad():
        y_sd32 = m.net(xs)[0]          # m.net 的注意力已换成 SDPA
    def rel(a_, b_):
        return float((a_ - b_).abs().max() / b_.abs().max())
    print(f"SDPA vs 原实现 fp32: 最大相对误差 {rel(y_sd32, y_ref32):.2e}；原实现 bf16 vs fp32: {rel(y_ref16, y_ref32):.2e}")

    # 2) 各变体初始时 (dy,dx,cp) == 二维逐层输出
    for cfg in (dict(variant="dec3d"), dict(variant="dec3d", freeze_encoder=True), dict(variant="dec3d", lora=8),
                dict(variant="zadapt"), dict(variant="zadapt", adapt_kind="attn"), dict(variant="p25d")):
        mm = CPSAM3D(cfg, init=a.init).to(dev).eval()
        with torch.no_grad():
            y = mm(x)                                                   # (1,4,Z,H,W)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y16 = mm(x).float()
        y2d = y[0, 1:].permute(1, 0, 2, 3)
        y2d16 = y16[0, 1:].permute(1, 0, 2, 3)
        ntr = sum(p.numel() for p in mm.parameters() if p.requires_grad) / 1e6
        print(f"{str(cfg):60s} fp32 相对误差 {rel(y2d, y_ref32):.2e}  bf16 {rel(y2d16, y_ref32):.2e}  "
              f"dz 最大 {float(y[0, 0].abs().max()):.1e}  可训练 {ntr:.1f}M")
        del mm
    print("NET3D_SELFTEST_DONE")
