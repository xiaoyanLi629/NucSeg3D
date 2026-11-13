#!/bin/sh
# 一条命令重跑全部（五种子齐后不带参数即可，自动取 S1 与二维在 gt、fp 两种协议下都齐全的种子）。
# 用法：sh run_all.sh [s0 s1000 s2000 s3000 s4000]
cd ${NUCSEG3D_SCRATCH}/nc/stats
PY=${NUCSEG3D_SHARED}/env_torch/bin/python
SEEDS="$@"
[ -z "$SEEDS" ] && SEEDS=$($PY -c "import sys; sys.path.insert(0,'.'); import nc_stats; print(' '.join(nc_stats.complete_seeds()))")
echo "种子：$SEEDS"
[ -f cache/covariates.json ] || $PY covariates.py --workers 32 > covariates.log 2>&1
$PY precache.py $SEEDS > precache.log 2>&1 && \
$PY readouts.py --seeds $SEEDS --workers 48 > readouts.log 2>&1 && \
$PY nc_stats.py --seeds $SEEDS --workers 64 > nc_stats.log 2>&1 && \
tail -3 nc_stats.log
