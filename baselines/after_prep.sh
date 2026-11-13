#!/bin/bash
# 等 nnU-Net 预处理结束（进程退出且 splits_final 已拷入），然后在 GPU 2 上开第 0 折训练。
B=${NUCSEG3D_SHARED}/baselines; P=$B/data/nnUNet_preprocessed/Dataset801_NucIso30
while pgrep -f nnUNetv2_plan_and_preprocess > /dev/null; do sleep 60; done
if [ ! -f $P/splits_final.json ] || [ $(ls $P/nnUNetPlans_3d_fullres/*_seg.b2nd 2>/dev/null | wc -l) -lt 998 ]; then
  echo "PREP_FAILED $(date)"; exit 1; fi
echo "PREP_OK $(date) $(du -sh $P | cut -f1)"
cd $B/scripts && GPU=2 bash run_full.sh nnunet_train
echo "NNUNET_EXIT $? $(date)"
