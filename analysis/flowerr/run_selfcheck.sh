#!/bin/bash
cd ${NUCSEG3D_SCRATCH}/nc/flowerr
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
P=${NUCSEG3D_SHARED}/env_torch/bin/python
t0=$(date +%s)
nice -n 10 $P flowerr_target.py --workers 1 --threads 2 --only mouse_embryo__E15_Ab2 nis3d__Zebrafish_1_xR25 blastospim__BlastoSPIM1_test_modSNR_F33_067_image_0001 gonuclear__1170_noisy > selfcheck_target.log 2>&1
rc=$?; echo "TARGET_EXIT $rc $(( $(date +%s) - t0 ))s" >> selfcheck_target.log
[ $rc -ne 0 ] && { echo "EXIT $rc" > selfcheck_done; exit; }
t1=$(date +%s)
nice -n 10 $P flowerr_angles.py --selfcheck > selfcheck_angles.log 2>&1; rc=$?
echo "EXIT $rc angles $(( $(date +%s) - t1 ))s" > selfcheck_done
