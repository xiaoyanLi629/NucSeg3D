"""第六轮诊断 D 的旁证（CPU、只读权重）：C 与 X1 都从同种子 A1 出发多训 3000 步，比较两者相对 A1 的权重改动在编码器各段与
三维解码器上的分布：|C−A1|、|X1−A1|、|X1−C| 的 L2 范数，及 (X1−A1) 与 (C−A1) 的余弦。余弦低、|X1−C|/|C−A1| 高的部分，
就是 xy 降采样模拟额外改动最多的部分（只是提示，功能上要靠“交换编码器/解码器”的评测确认）。另核对 S1 = ½(C+X1)。
输出 $NUCSEG3D_SCRATCH/r6/diag/wdiff.json。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import re

import torch

M = f"{_NSP_ROOT}/models/models/"


def load(name):
    sd = torch.load(M + name + "/weights.pt", map_location="cpu", weights_only=False)
    return {k: v.float() for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point()}


def group(k):
    m = re.match(r"net\.encoder\.blocks\.(\d+)\.", k)
    if m:
        b = int(m.group(1)); return f"enc.blocks{(b // 6) * 6:02d}-{(b // 6) * 6 + 5:02d}"
    if k.startswith("net.encoder."):
        return "enc.other(patch/pos/neck)"
    if k.startswith("dec.blocks"):
        return "dec.blocks"
    return "dec.io/heads"


out = {}
for s in ("0", "1000", "2000"):
    A, C, X, S = load(f"orx_t3d_a1_s{s}"), load(f"orx_t3d_r5c_s{s}"), load(f"orx_t3d_r5x1_s{s}"), load(f"orx_t3d_r5s1_s{s}")
    acc = {}
    smax = 0.0
    for k in A:
        g = group(k); a = acc.setdefault(g, dict(cc=0.0, xx=0.0, xc=0.0, dxc=0.0, aa=0.0, n=0))
        dc, dx = C[k] - A[k], X[k] - A[k]
        a["cc"] += float((dc * dc).sum()); a["xx"] += float((dx * dx).sum()); a["xc"] += float((dc * dx).sum())
        a["dxc"] += float(((X[k] - C[k]) ** 2).sum()); a["aa"] += float((A[k] ** 2).sum()); a["n"] += A[k].numel()
        smax = max(smax, float((S[k] - 0.5 * (C[k] + X[k])).abs().max()))
    print(f"== 种子 {s}（S1 与 ½(C+X1) 最大逐元素差 {smax:.2e}）")
    print(f"  {'部分':28s} {'参数量':>10s} {'|C−A1|/|A1|':>12s} {'|X1−A1|/|A1|':>13s} {'|X1−C|/|C−A1|':>14s} {'cos(X1−A1,C−A1)':>16s}")
    res = {}
    for g in sorted(acc):
        a = acc[g]
        rc, rx = (a["cc"] / a["aa"]) ** 0.5, (a["xx"] / a["aa"]) ** 0.5
        ratio = (a["dxc"] / a["cc"]) ** 0.5 if a["cc"] > 0 else float("nan")
        cos = a["xc"] / (a["cc"] * a["xx"]) ** 0.5 if a["cc"] > 0 and a["xx"] > 0 else float("nan")
        res[g] = dict(n=a["n"], rel_C=rc, rel_X1=rx, x1c_over_ca1=ratio, cos=cos)
        print(f"  {g:28s} {a['n']:10d} {rc:12.2e} {rx:13.2e} {ratio:14.2f} {cos:16.3f}")
    out[s] = res
    del A, C, X, S
json.dump(out, open(f"{_NSP_SCRATCH}/r6/diag/wdiff.json", "w"), indent=1, ensure_ascii=False)
print(f"写出 {_NSP_SCRATCH}/r6/diag/wdiff.json")
