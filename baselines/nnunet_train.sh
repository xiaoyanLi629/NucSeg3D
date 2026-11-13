#!/bin/bash
# nnU-Net 基线：规划 + 预处理（只做一次）→ 第 0 折训练（官方默认 3d_fullres、nnUNetTrainer、1000 轮）。
# 用法：GPU=1 bash nnunet_train.sh prep|train [额外参数，如 -tr nnUNetTrainer_5epochs]
set -e
source $(dirname $0)/nnunet_env.sh
D=801
case $1 in
  prep)
    nnUNetv2_plan_and_preprocess -d $D -c 3d_fullres -np ${NP:-8} --verify_dataset_integrity
    cp $nnUNet_raw/Dataset801_NucIso30/splits_final.json $nnUNet_preprocessed/Dataset801_NucIso30/   # 第 0 折 = iso30 train / val
    ;;
  train)
    shift
    CUDA_VISIBLE_DEVICES=${GPU:-1} nnUNetv2_train $D 3d_fullres 0 --npz "$@"
    ;;
esac
