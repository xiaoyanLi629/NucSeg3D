import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), "..", ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
# 平均模型能被 build_from_dir 正常载入，输出有限，且与两端都不同
import sys, torch
sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'src'))
from net3d import build_from_dir
x = torch.rand(1, 1, 32, 256, 256, device="cuda")
ys = []
for p in sys.argv[1:]:
    m, _ = build_from_dir(p, "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ys.append(m(x).float())
    del m; torch.cuda.empty_cache()
print("输出有限", all(bool(torch.isfinite(y).all()) for y in ys), "；|avg−A|", float((ys[2] - ys[0]).abs().mean()), "|avg−B|", float((ys[2] - ys[1]).abs().mean()), "|A−B|", float((ys[0] - ys[1]).abs().mean()))
