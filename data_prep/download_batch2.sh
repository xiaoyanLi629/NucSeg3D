#!/bin/bash
# 第二批：BlastoSPIM 1.0 + 2.0（约 60 GB），等第一批结束后开始
set -u
ROOT=${NUCSEG3D_ROOT}
DL=${NUCSEG3D_SCRATCH}/cpft_dl/blastospim
RAW=$ROOT/data/raw/blastospim
LOG=$ROOT/logs/download.log
mkdir -p $DL $RAW
unset http_proxy https_proxy
PX=${NUCSEG3D_PROXY:-}   # release: optional HTTP proxy
log() { echo "$(date +%m-%d_%H:%M:%S) $*" | tee -a $LOG; }
until grep -q BATCH1_DONE $LOG; do sleep 60; done
log "开始第二批：BlastoSPIM"
for v in 1 2; do for s in val test_lowSNR test_modSNR train; do
  fn=BlastoSPIM${v}_${s}.zip
  aria2c -q -c -x16 -s16 -k4M --max-tries=20 --retry-wait=10 ${PX:+--all-proxy=$PX} -d $DL -o $fn \
    "https://blastospim.flatironinstitute.org/sample/$fn" \
    && log "完成下载 blastospim/$fn ($(du -h $DL/$fn | cut -f1))" || { log "下载失败 blastospim/$fn"; continue; }
  mkdir -p $RAW/BlastoSPIM${v}_${s} && unzip -q -o $DL/$fn -d $RAW/BlastoSPIM${v}_${s} && log "已解压 $fn" || log "解压失败 $fn"
done; done
log "第二批全部结束"
echo BATCH2_DONE >> $LOG
