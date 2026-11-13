#!/bin/bash
# L3v_P3_d17（17a33dac）缺的 3 个体积单独补跑：同一模型、同一参数，只改 --workers 1，用 --only/--ids 限定；每个体积一个进程，避免 Pool 丢任务挂死。
cd ${NUCSEG3D_RUNS}/17a33dac-a81a-4a89-96c9-6e814266cc3f/repo/src
export CUDA_VISIBLE_DEVICES=0
O=${NUCSEG3D_SHARED}/nc_heldout/L3/fix_p3_d17
PY=${NUCSEG3D_SHARED}/env_torch/bin/python
for ID in BlastoSPIM1_val_T2_006_image_0001 BlastoSPIM1_val_M6_016_image_0001 BlastoSPIM1_val_M6_018_image_0001; do
  echo "=== $ID $(date)"
  $PY eval_gt.py --out $O/$ID --thrs 0 -1 -2 -3 -4 -6 --save_masks --model ${NUCSEG3D_ROOT}/models/models/orx_t3d_r5s1_s0 --workers 1 --box 384 --views 3 --views1_ds cellos parhyale mesospim --diam_from ${NUCSEG3D_SHARED}/nc_heldout/L3/val_d17_r2 --vox_override mesospim=5/10.52/10.52 --only blastospim --ids $ID > $O/$ID.log 2>&1 &
  P=$!; PK=0
  while kill -0 $P 2>/dev/null; do
    R=$(ps --ppid $P -o rss= 2>/dev/null | sort -n | tail -1); R=${R:-0}; [ "$R" -gt "$PK" ] && PK=$R
    for c in $(pgrep -P $P); do kill -0 $c 2>/dev/null; done
    sleep 10
  done
  wait $P; echo "rc=$? 工作进程峰值 RSS ${PK} kB $(date)"
done
echo FIX_ALL_DONE
