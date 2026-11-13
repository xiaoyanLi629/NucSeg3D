"""以固定的 os.getpid 运行某份代码的 train3d.py（配合 --workers 0，数据块序列在两次运行之间可复现）。
用法：python run_fixedpid.py <src目录> <train3d 的参数...>
环境变量 R6_DET=1 时再打开确定性模式（cuDNN 确定性算法、注意力走 math 后端、cuBLAS 固定工作区），
让反向传播也逐位可复现，用来逐步比较旧/新代码的训练轨迹（只用于测试，数值与正常训练略有不同）。"""
import os
import runpy
import sys

os.getpid = lambda: 4242
if os.environ.get("R6_DET"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
src = sys.argv[1]
sys.argv = [os.path.join(src, "train3d.py")] + sys.argv[2:]
runpy.run_path(os.path.join(src, "train3d.py"), run_name="__main__")
