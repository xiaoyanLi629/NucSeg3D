#!/bin/bash
# G5 同卡计时（hostC GPU2 独占）：先等 G2 测试集评测全部结束，再依次测 S1 → 二维 → 原版（g5_bench.py）→ nnU-Net → StarDist B（bl_bench.py）。
B=${NUCSEG3D_SHARED}/baselines; S=$B/scripts; O=$B/results/g5; mkdir -p $O
while pgrep -f "[b]l_eval.py --method" > /dev/null; do sleep 60; done
# 独占检查：GPU2 上不能有任何计算进程（按 UUID 查），否则每分钟复查，直到空闲
U=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 2)
while nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -q "$U"; do
  echo "GPU2 非空闲 $(date): $(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep "$U" | tr '\n' ';')"; sleep 60; done
echo "GPU2 独占确认 $(date): $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 2)"
echo "START $(date) load $(cut -d' ' -f1-3 /proc/loadavg)"
export CUDA_VISIBLE_DEVICES=2
cd $S/g5_src
for m in s1=${NUCSEG3D_ROOT}/models/models/orx_t3d_r5s1_s0 d2=orx_v5_2d_s0 base=base; do
  ${NUCSEG3D_SHARED}/env_torch/bin/python g5_bench.py --out $O --models $m 2>&1 | grep -v -i warn
  cp $O/summary.json $O/summary_${m%%=*}.json
done
cd $S
(source nnunet_env.sh; python bl_bench.py --method nnunet --out $O --tag nnunet \
   --model $B/results/nnUNet_results/Dataset801_NucIso30/nnUNetTrainer__nnUNetPlans__3d_fullres 2>&1 | grep -v -i warn)
# StarDist：逐体积部分一个进程（TF 显存池只增不减，峰值为累计值，表里标“上界”）；
# 扩展部分每个立方块尺寸各起一个新进程（显存按需增长），峰值即该尺寸的真实峰值，正文只用这一组比较。
(source stardist_env.sh; python bl_bench.py --method stardist --out $O --tag stardistB --parts P --model $B/results/stardist/stardist_B_s0 2>&1 \
   | grep -v -E "W0000|I0000|absl|WARNING")
for s in 64 96 128 160 192 256; do
  (source stardist_env.sh; python bl_bench.py --method stardist --out $O --tag stardistB_x$s --parts X --sizes $s --model $B/results/stardist/stardist_B_s0 2>&1 \
     | grep -v -E "W0000|I0000|absl|WARNING")
done
echo "G5_DONE $(date)"
