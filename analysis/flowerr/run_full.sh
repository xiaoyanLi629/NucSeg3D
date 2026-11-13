#!/bin/bash
cd ${NUCSEG3D_SCRATCH}/nc/flowerr
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
P=${NUCSEG3D_SHARED}/env_torch/bin/python
echo "$(date -Is) start targets" >> RUNLOG.txt
nice -n 10 $P flowerr_target.py --workers 4 --threads 2 > full_target.log 2>&1 || { echo "EXIT target $?" > full_done; exit; }
echo "$(date -Is) targets done; start angles" >> RUNLOG.txt
nice -n 10 $P flowerr_angles.py --workers 4 > full_angles.log 2>&1 || { echo "EXIT angles $?" > full_done; exit; }
echo "$(date -Is) angles done; start stats" >> RUNLOG.txt
nice -n 10 $P flowerr_stats.py > full_stats.log 2>&1 || { echo "EXIT stats $?" > full_done; exit; }
echo "$(date -Is) stats done" >> RUNLOG.txt
echo "EXIT 0" > full_done
