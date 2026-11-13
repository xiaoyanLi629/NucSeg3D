#!/bin/bash
# 取代 after_prep.sh：等预处理结束 → 检查 → 拷到 /dev/shm → 从内存盘开第 0 折训练（GPU 2）。
B=${NUCSEG3D_SHARED}/baselines; P=$B/data/nnUNet_preprocessed; SHM=/dev/shm/bl_nnUNet_preprocessed
while pgrep -f nnUNetv2_plan_and_preprocess > /dev/null; do sleep 60; done
D=$P/Dataset801_NucIso30
if [ ! -f $D/splits_final.json ] || [ $(ls $D/nnUNetPlans_3d_fullres/*_seg.b2nd 2>/dev/null | wc -l) -lt 998 ]; then
  echo "PREP_FAILED $(date)"; exit 1; fi
echo "PREP_OK $(date) $(du -sh $D | cut -f1)"
t0=$(date +%s); mkdir -p $SHM
cp -r $D $SHM/ || { echo "COPY_FAILED $(date)"; exit 1; }
n=$(ls $SHM/Dataset801_NucIso30/nnUNetPlans_3d_fullres/*_seg.b2nd | wc -l)
echo "COPY_OK $(( $(date +%s)-t0 ))s files_seg=$n $(du -sh $SHM | cut -f1) $(date)"
cd $B/scripts && NNUNET_PREP=$SHM GPU=2 bash run_full.sh nnunet_train
echo "NNUNET_EXIT $? $(date)"
