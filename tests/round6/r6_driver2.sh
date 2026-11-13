#!/usr/bin/env bash
# 等主 driver（a4..f）跑完，再跑确定性模式的训练轨迹比较 a4det。
T=${NUCSEG3D_SCRATCH}/r6/tests
while pgrep -f "r6_driver.sh a4 a5 b c e f" > /dev/null; do sleep 30; done
echo "$(date +%T) 开始 a4det" >> $T/driver.log
bash $T/run_tests_r6.sh a4det > $T/out_a4det.log 2>&1
echo "$(date +%T) 结束 a4det rc=$?" >> $T/driver.log
echo "$(date +%T) DRIVER2_DONE" >> $T/driver.log
