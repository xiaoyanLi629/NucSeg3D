"""在 train_cpsam.py 的基础上，加入三种可选的训练改动（可单独或组合使用），其余设置与“我们的模型”完全相同：
  --lora R      方向 4b：只训练 LoRA 低秩增量（ViT 每个 block 的 qkv/proj/mlp 四个线性层）以及 neck 和输出头，
                编码器其余参数冻结，用来减少“学新忘旧”。训练结束把 LoRA 合并进权重，存成普通 Cellpose 权重文件。
  --style P     方向 7：标注风格条件化。每套训练数据一个可学习向量，加到图像 token 上；每张切片以概率 P 改用零向量
                （“通用风格”）。推理时不加向量，即通用风格，所以推理代码不用改。
  --phys P      方向 5：按成像过程模拟的深度退化，每张切片以概率 P 施加：沿深度方向指数衰减（xy 切片整体按某一深度衰减，
                xz/yz 切片按行衰减，深的一端随机取上或下）、随深度增强的模糊、与亮度相关的噪声。标注不变。
用法: python train_cpsam2.py --slices $NUCSEG3D_SCRATCH/cpft/v3/train --name ft_lora16 --weights cellos=2,tapenade=2 --lora 16 --lr 1e-4
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import argparse, glob, json, os, re
from collections import Counter
import numpy as np

ROOT = f"{_NSP_ROOT}"
_CUR = {"ids": None}


def add_lora(net, r, alpha):
    import torch
    from torch import nn

    class _LoRA(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / np.sqrt(base.in_features))
            self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
            self.scale = alpha / r

        def forward(self, x):
            return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale

    for p in net.encoder.parameters():
        p.requires_grad = False
    n = 0
    for blk in net.encoder.blocks:
        blk.attn.qkv = _LoRA(blk.attn.qkv); blk.attn.proj = _LoRA(blk.attn.proj)
        blk.mlp.lin1 = _LoRA(blk.mlp.lin1); blk.mlp.lin2 = _LoRA(blk.mlp.lin2); n += 4
    for p in net.encoder.neck.parameters():
        p.requires_grad = True
    for p in net.out.parameters():
        p.requires_grad = True
    ntr = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"LoRA r={r}: {n} 个线性层，可训练参数 {ntr/1e6:.2f}M", flush=True)


def merge_lora(net):
    import torch
    with torch.no_grad():
        for blk in net.encoder.blocks:
            for parent, name in [(blk.attn, "qkv"), (blk.attn, "proj"), (blk.mlp, "lin1"), (blk.mlp, "lin2")]:
                m = getattr(parent, name)
                m.base.weight += (m.B @ m.A) * m.scale
                setattr(parent, name, m.base)


def add_style(net, nds):
    import torch, types
    import torch.nn.functional as F
    from torch import nn
    net.style_emb = nn.Parameter(torch.zeros(nds, net.encoder.pos_embed.shape[-1], device=net.encoder.pos_embed.device))

    def forward(self, x):
        x = F.conv2d(x, self.encoder.patch_embed.proj.weight.data[:, :x.shape[1]], bias=self.encoder.patch_embed.proj.bias.data, stride=self.ps)
        x = x.permute(0, 2, 3, 1)
        if self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed
        ids = _CUR["ids"]
        if self.training and ids is not None:
            emb = torch.stack([self.style_emb[i] if i >= 0 else torch.zeros_like(self.style_emb[0]) for i in ids])
            x = x + emb[:, None, None, :].to(x.dtype)
        if self.training and self.rdrop > 0:
            nlay = len(self.encoder.blocks)
            rdrop = (torch.rand((len(x), nlay), device=x.device) < torch.linspace(0, self.rdrop, nlay, device=x.device)).to(x.dtype)
            for i, blk in enumerate(self.encoder.blocks):
                mask = rdrop[:, i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                x = x * mask + blk(x) * (1 - mask)
        else:
            for blk in self.encoder.blocks:
                x = blk(x)
        x = self.encoder.neck(x.permute(0, 3, 1, 2))
        x1 = F.conv_transpose2d(self.out(x), self.W2, stride=self.ps, padding=0)
        return x1, torch.zeros((x.shape[0], 256), device=x.device)
    net.forward = types.MethodType(forward, net)


def phys_degrade(img, orient, rng):
    """img: (C, Ly, Lx) 已归一化。xz/yz 切片的行方向是 z。"""
    from scipy import ndimage as ndi
    ch = [c for c in range(img.shape[0]) if np.any(img[c])]
    out = img.copy()
    if orient in ("xz", "yz"):
        L = img.shape[1]; z = np.arange(L, dtype=np.float32)
        if rng.random() < 0.5:
            z = z[::-1].copy()
        z0 = rng.uniform(0.0, 0.4) * L                       # 从这一深度开始衰减
        ell = rng.uniform(0.3, 1.2) * L                      # 衰减长度与切片深度相当，切片里能看到由亮到暗的梯度
        A = np.maximum(np.exp(-np.clip(z - z0, 0, None) / ell), 0.1)[:, None]
    else:
        A = np.full((1, 1), rng.uniform(0.15, 1.0), np.float32)   # xy 切片：整张处于某一深度
    smax = rng.uniform(0.5, 3.0)
    nsig = rng.uniform(0.01, 0.06)
    for c in ch:
        x = img[c]
        bl = ndi.gaussian_filter(x, smax)
        w = np.clip(1 - A, 0, 1)                            # 越深越模糊
        x = (1 - w) * x + w * bl
        x = x * A
        noise = rng.normal(0, 1, x.shape).astype(np.float32) * nsig * np.sqrt(np.clip(x, 0, None) + 0.05)
        out[c] = (x + noise).astype(img.dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default=f"{_NSP_SCRATCH}/cpft/v3/train")
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", default="cellos=2,tapenade=2")
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--nimg", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--base", default=f"{ROOT}/models/base/cpsam_v2")
    ap.add_argument("--val_frac", type=float, default=0.02); ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--lora", type=int, default=0); ap.add_argument("--lora_alpha", type=float, default=0)
    ap.add_argument("--style", type=float, default=-1, help="<0 不用；否则为改用通用风格的概率")
    ap.add_argument("--phys", type=float, default=0.0)
    ap.add_argument("--extra", default="", help="额外切片目录（如自训练伪标签），文件名同样以“数据集__”开头")
    ap.add_argument("--labels_suffix", default="_masks.tif",
                    help="训练目标文件后缀：_masks.tif 让 Cellpose 现算二维流；_flows.tif 读预计算的 4 通道目标（如三维流目标）")
    ap.add_argument("--exclude_json", default="", help="val_split.json：去掉这些体积（验证集）的切片，文件名形如 数据集__体积id__方向编号")
    a = ap.parse_args()

    files = sorted(f for f in glob.glob(f"{a.slices}/*.tif") if not f.endswith(("_masks.tif", "_flows.tif")))
    print(f"训练目标：{a.labels_suffix}", flush=True)
    if a.extra:
        files += sorted(f for f in glob.glob(f"{a.extra}/*.tif") if not f.endswith(("_masks.tif", "_flows.tif")))
    ds_of = lambda f: os.path.basename(f).split("__")[0]
    if a.exclude_json:
        ex = {(ds, i) for ds, ids in json.load(open(a.exclude_json)).items() for i in ids}
        n0 = len(files)
        files = [f for f in files if tuple(os.path.basename(f).split("__")[:2]) not in ex]
        print(f"去掉验证体积的切片 {n0 - len(files)} 张（验证体积 {len(ex)} 个），剩 {len(files)} 张", flush=True)
    rng = np.random.default_rng(0); rng.shuffle(files)
    nval = max(20, int(len(files) * a.val_frac))
    val, train = files[:nval], files[nval:]
    cnt = Counter(ds_of(f) for f in train)
    w = {k: 1.0 for k in cnt}
    for kv in filter(None, a.weights.split(",")):
        k, v = kv.split("="); w[k] = float(v)
    probs = np.array([w[ds_of(f)] / cnt[ds_of(f)] for f in train]); probs /= probs.sum()
    print("训练切片:", dict(cnt), " 验证切片:", len(val), " 数据集权重:", w, flush=True)
    dsl = sorted(cnt)
    def _orient(f):
        m = re.search(r"__(xy|xz|yz|2d|obl)", os.path.basename(f))
        return m.group(1) if m else "xy"

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import torch
    from cellpose import models, train as cptrain
    if a.seed_offset:
        _seed = np.random.seed
        np.random.seed = lambda s=None: _seed(None if s is None else int(s) + a.seed_offset)
        torch.manual_seed(a.seed_offset)
    model = models.CellposeModel(gpu=True, pretrained_model=a.base)
    net = model.net
    if a.lora:
        add_lora(net, a.lora, a.lora_alpha or a.lora)
    if a.style >= 0:
        add_style(net, len(dsl))
    prng = np.random.default_rng(1234 + a.seed_offset)
    _orig = cptrain._get_batch
    # Cellpose 会去掉掩码太少的切片，索引以它返回的训练列表为准
    M = {}
    _orig_ptt = cptrain._process_train_test

    def _ptt(*args, **kw):
        out = _orig_ptt(*args, **kw)
        tf = out[2]
        M["n"] = len(tf); M["ds"] = [dsl.index(ds_of(str(f))) for f in tf]; M["or"] = [_orient(str(f)) for f in tf]
        print(f"Cellpose 保留训练切片 {M['n']} 张", flush=True)
        return out
    cptrain._process_train_test = _ptt

    def _get_batch(inds, data=None, labels=None, files=None, labels_files=None, **kw):
        imgs, lbls = _orig(inds, data=data, labels=labels, files=files, labels_files=labels_files, **kw)
        n = len(data) if data is not None else len(files)
        is_train = n == M["n"]
        if is_train:
            if a.style >= 0:
                _CUR["ids"] = [M['ds'][i] if prng.random() >= a.style else -1 for i in inds]
            if a.phys > 0:
                imgs = [phys_degrade(np.asarray(im), M['or'][i], prng) if prng.random() < a.phys else im for im, i in zip(imgs, inds)]
        else:
            _CUR["ids"] = None
        return imgs, lbls
    cptrain._get_batch = _get_batch

    save = f"{ROOT}/models"
    json.dump(dict(args=vars(a), counts=dict(cnt), weights=w, style_datasets=dsl), open(f"{save}/{a.name}_config.json", "w"), indent=1)
    path, tl, vl = cptrain.train_seg(
        net, train_files=train, train_labels_files=[f[:-4] + a.labels_suffix for f in train], train_probs=probs,
        test_files=val, test_labels_files=[f[:-4] + a.labels_suffix for f in val],
        learning_rate=a.lr, weight_decay=0.1, n_epochs=a.epochs, batch_size=1, nimg_per_epoch=a.nimg,
        min_train_masks=3, save_path=save, model_name=a.name, save_every=20)
    if a.lora:
        merge_lora(net)
    sd = {k: v for k, v in net.state_dict().items() if k != "style_emb"}
    if a.style >= 0:
        torch.save(net.style_emb.detach().cpu(), f"{save}/models/{a.name}_style_emb.pt")
    torch.save(sd, str(path))
    json.dump(dict(model_path=str(path), train_losses=list(map(float, tl)), test_losses=list(map(float, vl))),
              open(f"{save}/{a.name}_losses.json", "w"), indent=1)
    print("TRAIN_DONE", path, flush=True)


if __name__ == "__main__":
    main()
