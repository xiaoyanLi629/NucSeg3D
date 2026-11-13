# source 后使用：StarDist（TensorFlow 2.21）在 RTX 5090（sm_120）上运行所需的环境。
# TF 的 wheel 没有 sm_120 的二进制核，要从 PTX 即时编译；系统 /usr/local/cuda 的 ptxas 是 12.1，太旧，
# 必须改用 pip 装的 nvidia-cuda-nvcc-cu12（12.9）里的 ptxas 与 libdevice。JIT 结果缓存在本地盘，第二次起不再编译。
E=${NUCSEG3D_SCRATCH}/bl_envs/env_stardist
N=$E/lib/python3.10/site-packages/nvidia/cuda_nvcc
export PATH=$N/bin:$E/bin:$PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$N"
export TF_USE_LEGACY_KERAS=1 TF_CPP_MIN_LOG_LEVEL=2
export CUDA_CACHE_PATH=${NUCSEG3D_SCRATCH}/cuda_cache CUDA_CACHE_MAXSIZE=4294967296
