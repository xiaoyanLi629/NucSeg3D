# source 后使用：nnU-Net v2 的三个目录与环境
B=${NUCSEG3D_SHARED}/baselines
export nnUNet_raw=$B/data/nnUNet_raw
export nnUNet_preprocessed=${NNUNET_PREP:-$B/data/nnUNet_preprocessed}   # 训练时指向 /dev/shm 的副本
export nnUNet_results=$B/results/nnUNet_results
export nnUNet_compile=${nnUNet_compile:-f}
export PATH=${NUCSEG3D_SCRATCH}/bl_envs/env_nnunet/bin:$PATH   # 环境在 hostC 本地数据盘（共享盘 FUSE 上 pip 安装会卡死）
