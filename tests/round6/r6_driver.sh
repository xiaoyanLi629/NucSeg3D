#!/usr/bin/env bash
# 依次跑 round 6 的各节测试（每节日志 out_<节>.log）；先等 a-1 的新代码数据测试结束。
T=${NUCSEG3D_SCRATCH}/r6/tests
while pgrep -f "t_a_data.py" > /dev/null; do sleep 20; done
for s in "$@"; do
  echo "$(date +%T) 开始 $s" >> $T/driver.log
  bash $T/run_tests_r6.sh $s > $T/out_$s.log 2>&1
  echo "$(date +%T) 结束 $s rc=$?" >> $T/driver.log
done
echo "$(date +%T) DRIVER_DONE" >> $T/driver.log
