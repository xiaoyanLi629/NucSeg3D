#!/bin/bash
# G2 完整训练与评测命令（在 hostC 上执行；只用 GPU 1、2）。主代理登记 AGENTS.md 之前不要运行。
# 用法：bash run_full.sh nnunet_train | sd_train_A | sd_train_B | eval_val <method> <model> <out> | eval_test_gt ... | eval_test_fp ...
set -e
S=${NUCSEG3D_SHARED}/baselines/scripts; B=${NUCSEG3D_SHARED}/baselines; cd $S
MESO="--vox_override mesospim=5/10.52/10.52"
case $1 in
  nnunet_train)   # 官方默认 3d_fullres、nnUNetTrainer（1000 轮 × 250 步）、只训第 0 折；预处理已由 nnunet_train.sh prep 完成
    GPU=${GPU:-1} bash nnunet_train.sh train > $B/logs/nnunet_fold0.log 2>&1 ;;
  sd_train_A|sd_train_B)
    source stardist_env.sh; c=${1#sd_train_}
    CUDA_VISIBLE_DEVICES=${GPU:-2} python train_stardist.py --cfg $c --out $B/results/stardist > $B/logs/stardist_$c.log 2>&1 ;;
  eval_val)       # 验证集 gt 协议（88 个体积），网格内所有变体一次推理全部算出
    m=$2; model=$3; out=$4
    if [ $m = stardist ]; then source stardist_env.sh; PY=python; else source nnunet_env.sh; PY=python; fi
    CUDA_VISIBLE_DEVICES=${GPU:-1} $PY bl_eval.py --method $m --model $model --out $out --save_masks $MESO ;;
  eval_test_gt)   # 测试集 gt 协议（真值直径），只报告；变体与体积下限用验证集选定的一格
    m=$2; model=$3; out=$4
    if [ $m = stardist ]; then source stardist_env.sh; else source nnunet_env.sh; fi
    CUDA_VISIBLE_DEVICES=${GPU:-1} python bl_eval.py --method $m --model $model --out $out --split test --save_masks $MESO ;;
  eval_test_fp)   # 测试集完整流程：每套至多 4 个体积，直径取我们二维模型 r2a2 的 d_used（$5 = 对应 eval_val 结果目录），只报告
    m=$2; model=$3; out=$4; dfrom=$5
    if [ $m = stardist ]; then source stardist_env.sh; else source nnunet_env.sh; fi
    CUDA_VISIBLE_DEVICES=${GPU:-1} python bl_eval.py --method $m --model $model --out $out --split test --max_per_ds 4 --diam_from $dfrom --save_masks $MESO ;;
esac
