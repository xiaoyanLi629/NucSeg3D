#!/bin/sh
# 用 r6/s5_stats.py（只读调用）按五种子生成核对用的参考输出，--out 写到本目录
cd ${NUCSEG3D_SCRATCH}/r6
PY=${NUCSEG3D_SHARED}/env_torch/bin/python
O=${NUCSEG3D_SCRATCH}/nc/stats/ref
S="s0 s1000 s2000 s3000 s4000"
for m in gt fp; do
  $PY s5_stats.py test $m --base_fixed --out $O/s5_test_${m}_5seeds_basefixed.json --seeds $S > $O/log_${m}_all.txt 2>&1 &
  $PY s5_stats.py test $m --base_fixed --subset ${NUCSEG3D_SCRATCH}/r9/clean_test.json --out $O/s5_test_${m}_5seeds_basefixed_clean_test.json --seeds $S > $O/log_${m}_C1.txt 2>&1 &
  $PY s5_stats.py test $m --base_fixed --subset ${NUCSEG3D_SCRATCH}/r9/clean_test_plus_b.json --out $O/s5_test_${m}_5seeds_basefixed_clean_test_plus_b.json --seeds $S > $O/log_${m}_C2.txt 2>&1 &
done
wait
