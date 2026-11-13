#!/bin/sh
# 生成缓存：recs → 读数。用法：sh run_pre.sh s0 s1000 s2000
cd ${NUCSEG3D_SCRATCH}/nc/stats
PY=${NUCSEG3D_SHARED}/env_torch/bin/python
$PY precache.py "$@" > precache.log 2>&1 && $PY readouts.py --seeds "$@" --workers 48 > readouts.log 2>&1
