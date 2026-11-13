#!/usr/bin/env bash
# r9 泄漏审计的后续各步，按顺序单进程跑（每步结束才开始下一步）
cd ${NUCSEG3D_SCRATCH}/r9
PY="nice -n 10 ${NUCSEG3D_SHARED}/env_torch/bin/python -u leak_audit.py"
for step in verify contain coords seq nis3d; do
  echo "== $(date +%T) 开始 $step"
  $PY $step > leak/$step.log 2>&1 < /dev/null
  echo "== $(date +%T) 结束 $step 退出码 $?"
done
