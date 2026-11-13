"""独立抽查审计的三条关键证据（只读图像与清单，不看任何模型输出）：
mouse_embryo E7_Ab1（test）是否为 E5_Ab1（train）在 (7,21,14) 的原样裁块；tapenade 第 80 号块是否逐像素落在 annotator1_image1 里；nis3d Drosophila_1 左右两块拼回去是否等于原图。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import sys

import numpy as np
import tifffile
from skimage.feature import match_template

sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'audit'))
from leak_audit import RAW, STD, manifest  # noqa: E402

M = {(r["ds"], r["id"]): r for r in manifest()}
print("STD", STD, "RAW", RAW)
print("示例记录", json.dumps(M[("mouse_embryo", "E7_Ab1")], ensure_ascii=False)[:300])


def img(ds, vid):
    r = M[(ds, vid)]
    p = r.get("img") or r.get("image") or r.get("path")
    p = p if p.startswith("/") else f"{STD}/{p}"
    return tifffile.imread(p), r["part"] if "part" in r else r.get("split")


a, pa = img("mouse_embryo", "E7_Ab1")
b, pb = img("mouse_embryo", "E5_Ab1")
print("E7_Ab1", a.shape, pa, "E5_Ab1", b.shape, pb)
sub = b[7:7 + a.shape[0], 21:21 + a.shape[1], 14:14 + a.shape[2]]
print("  E5_Ab1[7:,21:,14:] 与 E7_Ab1 逐像素相同：", sub.shape == a.shape and bool(np.array_equal(sub, a)))

ts = sorted(k for k in M if k[0] == "tapenade" and M[k].get("part", M[k].get("split")) != "test")
print("tapenade 非测试体积", len(ts), ts[:3])
t80 = [k for k in ts if k[1].endswith("80") or "_80" in k[1] or k[1].split("_")[-1] in ("80", "080")]
print("  名字里带 80 的", t80[:3])
k = t80[0] if t80 else ts[len(ts) // 2]
c, pc = img(*k)
big, pbig = img("tapenade", "annotator1_image1")
print("  块", k, c.shape, pc, "测试图", big.shape, pbig)
r = match_template(big.astype(np.float32), c.astype(np.float32))
pos = np.unravel_index(np.argmax(r), r.shape)
print("  最佳位置", pos, "NCC", float(r[pos]))
sl = big[pos[0]:pos[0] + c.shape[0], pos[1]:pos[1] + c.shape[1], pos[2]:pos[2] + c.shape[2]]
print("  逐像素相同：", bool(np.array_equal(sl, c)))
