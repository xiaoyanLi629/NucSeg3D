#!/usr/bin/env bash
# round 6 代码测试（分辨率条件化 + 翻转 TTA）。用法：bash run_tests_r6.sh a1|a1cmp|a2|a3|a4|a5|b|c|d|e|f
# 只用验证集体积（val_split）与 mesoSPIM 开发集，不碰测试集；GPU 测试单卡、显存 < 24 GB。
# 每个 GPU 测试先过资源闸门（gate.sh：等到某张卡空闲显存够、容器内存余量够），卡号由闸门选。
set -u
P=${NUCSEG3D_SHARED}/env_torch/bin/python
T=${NUCSEG3D_SCRATCH}/r6/tests
OLD=${NUCSEG3D_SCRATCH}/r6/src_before_cond; NEW=${NUCSEG3D_SCRATCH}/true3d_dev/src
MM=${NUCSEG3D_ROOT}/models/models; A1=$MM/orx_t3d_a1_s0; S1=$MM/orx_t3d_r5s1_s0
sec=${1:-}
gate() { export CUDA_VISIBLE_DEVICES=$(bash $T/gate.sh $1 $2); }      # gate <显存GB> <内存GB>
ev() { local src=$1 model=$2 out=$3; shift 3; ( cd $src && $P eval_gt.py --model $model --out $out --workers 1 "$@" > $out.log 2>&1 ); echo "$(basename $out) rc=$? $(grep SUMMARY $out.log | cut -c1-140)"; }
O=$T/out_a; mkdir -p $O

if [ "$sec" = a1 ]; then
  echo "== a-1 训练块序列（CPU 200 块 × 5 配置，GPU to_patch 200 块 × 2 配置）"
  gate 8 15; $P $T/t_a_data.py $OLD $O/data_old.json 200 cuda:0 > $O/data_old.log 2>&1; echo "old rc=$?"
  gate 8 15; $P $T/t_a_data.py $NEW $O/data_new.json 200 cuda:0 > $O/data_new.log 2>&1; echo "new rc=$?"
  $P $T/t_a_cmp.py $O/data_old.json $O/data_new.json
fi
if [ "$sec" = a1cmp ]; then $P $T/t_a_cmp.py $O/data_old.json $O/data_new.json; fi

if [ "$sec" = a2 ]; then
  echo "== a-2 建网络后的随机数状态、参数、cfg（CPU）"
  bash $T/gate.sh 0 15 > /dev/null
  CUDA_VISIBLE_DEVICES= $P $T/t_a_init.py $OLD > $O/init_old.log 2>&1
  CUDA_VISIBLE_DEVICES= $P $T/t_a_init.py $NEW > $O/init_new.log 2>&1
  grep T_a2 $O/init_old.log | sed 's/^/  旧 /'; grep T_a2 $O/init_new.log | sed 's/^/  新 /'
  if diff <(grep T_a2 $O/init_old.log) <(grep T_a2 $O/init_new.log) > /dev/null; then echo "T_a2 旧/新逐项相同 = True"; else echo "T_a2 旧/新逐项相同 = False"; fi
fi

if [ "$sec" = a3 ]; then
  echo "== a-3 验证体积推理输出（float32 dP / cellprob 哈希；旧代码跑两次看确定性）"
  gate 10 20; $P $T/t_a_infer.py $OLD $S1 old > $O/infer_old.log 2>&1
  gate 10 20; $P $T/t_a_infer.py $OLD $S1 old2 > $O/infer_old2.log 2>&1
  gate 10 20; $P $T/t_a_infer.py $NEW $S1 new > $O/infer_new.log 2>&1
  grep -h T_a3 $O/infer_old.log $O/infer_old2.log $O/infer_new.log
  for t in old2 new; do
    if diff <(grep T_a3 $O/infer_old.log | awk '{$2=""; print}') <(grep T_a3 $O/infer_$t.log | awk '{$2=""; print}') > /dev/null; then echo "T_a3 old vs $t 逐位相同 = True"; else echo "T_a3 old vs $t 逐位相同 = False"; fi
  done
fi

if [ "$sec" = a4 ]; then
  echo "== a-4 train3d 前 6 步（--workers 0、固定 pid，同一数据流；从 A1 种子 0 微调；为省显存 --patch_z 16）"
  TA="--variant dec3d --init orx_v5_2d_s0 --init3d $A1 --seed 0 --steps 6 --warmup 100 --workers 0 --log_every 1 --save_every 0 --patch_z 16 --gpu cuda:0"
  gate 12 20; $P $T/run_fixedpid.py $OLD $TA --out $O/loop_old > $O/loop_old.log 2>&1; echo "loop_old rc=$? GPU$CUDA_VISIBLE_DEVICES"
  gate 12 20; $P $T/run_fixedpid.py $OLD $TA --out $O/loop_old2 > $O/loop_old2.log 2>&1; echo "loop_old2 rc=$? GPU$CUDA_VISIBLE_DEVICES"
  gate 12 20; $P $T/run_fixedpid.py $NEW $TA --out $O/loop_new > $O/loop_new.log 2>&1; echo "loop_new rc=$? GPU$CUDA_VISIBLE_DEVICES"
  gate 12 20; $P $T/run_fixedpid.py $NEW $TA --cond res --cond_enc 2 --out $O/loop_newcond > $O/loop_newcond.log 2>&1; echo "loop_newcond rc=$? GPU$CUDA_VISIBLE_DEVICES"
  rm -f $O/loop_*/weights.pt
  $P $T/t_a_loop_cmp.py $O
fi

if [ "$sec" = a5 ]; then
  echo "== a-5 eval_gt 端到端（旧/新代码，S1 种子 0；bbbc050 单/三视图，cellos 走 z 插值）"
  SJ=$O/split_a.json; printf '{"bbbc050": ["train_Emb11_t501"], "cellos": ["EGFP_d4_flt32_28"]}\n' > $SJ
  rm -rf $O/ev_*
  gate 10 20; ev $OLD $S1 $O/ev_old --split_json $SJ
  gate 10 20; ev $NEW $S1 $O/ev_new --split_json $SJ
  gate 10 20; ev $OLD $S1 $O/ev_old_v3 --split_json $SJ --only bbbc050 --views 3
  gate 10 20; ev $NEW $S1 $O/ev_new_v3 --split_json $SJ --only bbbc050 --views 3
  $P $T/t_ev_cmp.py $O/ev_old $O/ev_new a5
  $P $T/t_ev_cmp.py $O/ev_old_v3 $O/ev_new_v3 a5-三视图
fi

if [ "$sec" = b ]; then mkdir -p $T/out_b; gate 14 20; $P -u $T/t_b_zero.py 2>&1 | grep --line-buffered -v Warning; fi
if [ "$sec" = c ]; then mkdir -p $T/out_c; gate 12 20; $P -u $T/t_c_grad.py 2>&1 | grep --line-buffered -v Warning; fi
if [ "$sec" = d ]; then CUDA_VISIBLE_DEVICES= $P -u $T/t_d_res.py 6000 2>&1 | grep --line-buffered -v Warning; fi

if [ "$sec" = e ]; then
  OE=$T/out_e; mkdir -p $OE
  gate 12 20; $P -u $T/t_e_tta.py 2>&1 | grep --line-buffered -v Warning
  SJ=$OE/split_e.json; printf '{"bbbc050": ["train_Emb11_t501"], "cellos": ["EGFP_d4_flt32_28"]}\n' > $SJ
  rm -rf $OE/ev_*
  gate 10 20; ev $NEW $S1 $OE/ev_notta --split_json $SJ
  gate 10 20; ev $NEW $S1 $OE/ev_tta_yx --split_json $SJ --tta_flips yx
  gate 10 20; ev $NEW $S1 $OE/ev_tta_yx_v3 --split_json $SJ --only bbbc050 --tta_flips yx --views 3
  $P $T/t_ev_cmp.py $OE/ev_notta $OE/ev_tta_yx "e 不翻转 vs yx"
fi

if [ "$sec" = f ]; then
  OF=$T/out_f; mkdir -p $OF
  # 机器上另有 2 万步训练与多个评测长期占卡，等不到 18 GB 空闲显存，所以默认 --patch_z 16（PZ=32 可改回训练时的块厚）
  PZ=${PZ:-16}; GF=$([ $PZ -ge 32 ] && echo 18 || echo 11)
  TA="--variant dec3d --init orx_v5_2d_s0 --init3d $A1 --seed 0 --steps 30 --warmup 10 --workers 4 --log_every 5 --save_every 0 --patch_z $PZ --gpu cuda:0"
  ALLAUG="--aniso_aug 0.33 --aniso_xy 1 --aniso_lo 2 --aniso_hi 10 --aniso_blur 0.3 --aniso_blur_max 0.6 --aniso_abs 1 --perm_aug 0.3 --zstretch_p 0.33"
  for run in nocond cond_res cond_all; do
    case $run in
      nocond) X="";; cond_res) X="--cond res";; cond_all) X="--cond res --cond_enc 2 --cond_drop 0.2 --cond_nfreq 2 $ALLAUG";;
    esac
    rm -rf $OF/$run
    gate $GF 25
    ( cd $NEW && exec $P train3d.py $TA $X --out $OF/$run > $OF/$run.log 2>&1 ) &
    pid=$!; sleep 2; tp=$pid
    grep -q train3d /proc/$pid/cmdline 2>/dev/null || tp=$(pgrep -P $pid -f train3d.py | head -1)
    bash $T/memwatch.sh ${tp:-$pid} $OF/$run.mem; wait $pid
    echo "== $run rc=$? GPU$CUDA_VISIBLE_DEVICES $(cat $OF/$run.mem)"
    grep -E "round-|^步 (1|5|10|15|20|25|30)/|TRAIN3D_DONE|Error|error" $OF/$run.log | cut -c1-220
  done
  $P - <<'PY'
import json
import os
O = os.path.join(os.environ["NUCSEG3D_SCRATCH"], "r6/tests/out_f")  # release: exported env var
for r in ("nocond", "cond_res", "cond_all"):
    L = json.load(open(f"{O}/{r}/train_log.json")); last = L[-1]
    fin = all(abs(x["loss"]) < 1e9 for x in L)
    print(f"T_f {r:9s} 损失有限 = {fin}  第 30 步损失 {last['loss']:.4f}  {last['s_per_step']:.2f} s/步（30 步平均，含第 1 步；取数 {last['data_s']:.2f}）  峰值显存 {last['mem_gb']:.1f} GB")
    m = json.load(open(f"{O}/{r}/model3d.json"))
    print(f"      model3d.json cfg 中的条件键：{ {k: v for k, v in m['cfg'].items() if k.startswith('cond')} }")
PY
  SJ=$OF/split_f.json; printf '{"bbbc050": ["train_Emb11_t501"], "cellos": ["EGFP_d4_flt32_28"], "tapenade": ["val_000027"]}\n' > $SJ
  rm -rf $OF/ev_*
  gate 10 20; ev $NEW $OF/cond_res $OF/ev_cond_res --split_json $SJ
  gate 10 20; ev $NEW $OF/cond_all $OF/ev_cond_all_v3tta --split_json $SJ --only cellos tapenade --views 3 --views1_ds cellos --tta_flips yx
  gate 10 20; ev $NEW $OF/cond_res $OF/ev_cond_res_null --split_json $SJ --only bbbc050 --cond_null
  $P $T/t_f_check.py $OF/ev_cond_res $OF/ev_cond_all_v3tta
fi

if [ "$sec" = a4det ]; then
  echo "== a-4（确定性模式 R6_DET=1）：cuDNN 确定性 + math 注意力，反向传播也可复现时逐步比较旧/新代码的前 6 步"
  TA="--variant dec3d --init orx_v5_2d_s0 --init3d $A1 --seed 0 --steps 6 --warmup 100 --workers 0 --log_every 1 --save_every 0 --patch_z 16 --gpu cuda:0"
  for t in old old2 new newcond; do
    src=$NEW; [ "${t#old}" != "$t" ] && src=$OLD
    X=""; [ $t = newcond ] && X="--cond res --cond_enc 2"
    gate 11 20; R6_DET=1 $P $T/run_fixedpid.py $src $TA $X --out $O/det_$t > $O/det_$t.log 2>&1; echo "det_$t rc=$? GPU$CUDA_VISIBLE_DEVICES"
  done
  rm -f $O/det_*/weights.pt
  $P $T/t_a_loop_cmp.py $O det_
fi
