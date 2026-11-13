#!/usr/bin/env bash
# 重排后的剩余测试：c（重跑）→ a4det → f（patch_z 16）。每节先过资源闸门。
T=${NUCSEG3D_SCRATCH}/r6/tests
for s in c a4det f; do
  echo "$(date +%T) 开始 $s" >> $T/driver.log
  bash $T/run_tests_r6.sh $s > $T/out_$s.log 2>&1
  echo "$(date +%T) 结束 $s rc=$?" >> $T/driver.log
done
echo "$(date +%T) DRIVER4_DONE" >> $T/driver.log
