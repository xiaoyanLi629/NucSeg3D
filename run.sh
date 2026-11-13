#!/usr/bin/env bash
# OpenResearch 固定运行命令：bash run.sh
# 只能在挂了 AutoDL 共享盘（${NUCSEG3D_FS}）的服务器上跑：hostA、hostD、hostE。
set -euo pipefail
cd "$(dirname "$0")"
# DataLoader 进程会缓存所有体积的 memmap（每个占一个文件句柄），默认 1024 不够
ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn)
ROOT=${NUCSEG3D_ROOT}
[ -d "$ROOT" ] || { echo "这台机器没有 $ROOT（共享盘），无法运行"; exit 2; }

# 1. Python 环境：共享盘 env_torch（cu130）在驱动较旧的机器（如 hostE）上用不了 GPU，改用本地 env128
PY=${NUCSEG3D_SHARED}/env_torch/bin/python
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  PY=${NUCSEG3D_SCRATCH}/env128/bin/python
fi
"$PY" -c "import torch, cellpose; assert torch.cuda.is_available()" || { echo "没有可用的 GPU 环境"; exit 2; }

# 2. 内存保护：容器上限约 190 GB 且与其他项目共用，超了会被无声杀掉。
#    训练约需 45 GB、评测约 25 GB；可用内存不够就等（最多 6 小时）
#    t0 模式（建 iso30 与评测同时进行）约需 70 GB；train3d 与 train 一样按 45 GB 算
NEED=$(python3 -c "import json;c=json.load(open('config.json'));print(70 if c.get('mode')=='t0' else (45 if c.get('train') or c.get('train3d') else 25))")
avail() { python3 - <<'EOF'
import re
mx = open('/sys/fs/cgroup/memory.max').read().strip()
mx = int(mx) if mx != 'max' else 10**13
st = dict(l.split() for l in open('/sys/fs/cgroup/memory.stat'))
print((mx - int(st['anon']) - int(st.get('shmem', 0))) // 10**9)
EOF
}
for i in $(seq 1 72); do
  A=$(avail); [ "$A" -ge "$NEED" ] && break
  echo "可用内存 ${A} GB < 需要 ${NEED} GB，等 5 分钟（第 $i 次）"; sleep 300
done
echo "可用内存 $(avail) GB"

# 3. 选 GPU：加锁后跳过已被本项目其他 run 认领的卡（认领文件里记 PID），再在剩下的卡里选显存最空的
mkdir -p /tmp/orx_gpu
SHARE=$(python3 -c "import json;print(1 if json.load(open('config.json')).get('share_gpu') else 0)")
exec 9>/tmp/orx_gpu/lock; flock 9
GPU=$(python3 -c "import json;print(json.load(open('config.json')).get('force_gpu',''))")
[ -n "$GPU" ] || for g in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | cut -d, -f1); do
  c=/tmp/orx_gpu/claim_$g
  # config.json 里 share_gpu 为真时（如 96 GB 的单卡机器）不检查认领，允许与本项目其他 run 共用一张卡
  if [ "$SHARE" != "1" ] && [ -f "$c" ] && kill -0 "$(cat "$c")" 2>/dev/null; then continue; fi
  GPU=$g; echo $$ > "$c"; break
done
flock -u 9
[ -n "$GPU" ] || { echo "两张卡都已被本项目的其他 run 占用"; exit 3; }
export CUDA_VISIBLE_DEVICES=$GPU
echo "HOST $(hostname) GPU $GPU PY $PY"

set +e
"$PY" -u orx_run.py 2>&1 | grep --line-buffered -v -E "it/s\]|%\||Warn|coo = |sparse"
exit "${PIPESTATUS[0]}"
