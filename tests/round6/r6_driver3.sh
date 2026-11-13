#!/usr/bin/env bash
# 等前两个 driver 跑完，再重跑测试 c（第一次因 torch.device("cuda") 没写卡号而失败）。
T=${NUCSEG3D_SCRATCH}/r6/tests
while pgrep -f "r6_driver.sh a4 a5 b c e f" > /dev/null || pgrep -f "r6_driver2.sh" > /dev/null; do sleep 30; done
echo "$(date +%T) 开始 c（重跑）" >> $T/driver.log
bash $T/run_tests_r6.sh c > $T/out_c.log 2>&1
echo "$(date +%T) 结束 c（重跑）rc=$?" >> $T/driver.log
echo "$(date +%T) DRIVER3_DONE" >> $T/driver.log
