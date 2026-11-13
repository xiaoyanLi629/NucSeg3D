cd ${NUCSEG3D_SCRATCH}/r6
${NUCSEG3D_SHARED}/env_torch/bin/python s5_stats.py val gt --seeds s0 s1000 s2000 s3000 s4000 --base_fixed --out s5final/val_gt.json > s5final/val_gt.log 2>&1 &
${NUCSEG3D_SHARED}/env_torch/bin/python s5_stats.py test fp --seeds s0 s1000 s2000 s3000 s4000 --base_fixed --out s5final/test_fp.json > s5final/test_fp.log 2>&1 &
${NUCSEG3D_SHARED}/env_torch/bin/python s5_stats.py test gt --seeds s0 s1000 s2000 s3000 s4000 --base_fixed --out s5final/test_gt.json > s5final/test_gt.log 2>&1 &
${NUCSEG3D_SHARED}/env_torch/bin/python s5_stats.py test fp --seeds s0 s1000 s2000 s3000 s4000 --base_fixed --twod_fixed --out s5final/test_fp_2dfixed.json > s5final/test_fp_2dfixed.log 2>&1 &
${NUCSEG3D_SHARED}/env_torch/bin/python s5_stats.py test gt --seeds s0 s1000 s2000 s3000 s4000 --base_fixed --twod_fixed --out s5final/test_gt_2dfixed.json > s5final/test_gt_2dfixed.log 2>&1 &
for sub in clean_test clean_test_plus_b; do for m in fp gt; do ${NUCSEG3D_SHARED}/env_torch/bin/python s5_stats.py test $m --seeds s0 s1000 s2000 s3000 s4000 --base_fixed --subset ${NUCSEG3D_SHARED}/cellposeft_orx/audit/$sub.json --out s5final/test_${m}_$sub.json > s5final/test_${m}_$sub.log 2>&1 & done; done
wait; echo ALLDONE > s5final/done
