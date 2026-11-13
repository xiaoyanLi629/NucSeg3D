#!/bin/bash
# Cellpose-SAM 微调数据：第一批下载（原始运行经本机 HTTP 代理；发布版：代理由 NUCSEG3D_PROXY 给出，不设则直连）
set -u
ROOT=${NUCSEG3D_ROOT}
DL=${NUCSEG3D_SCRATCH}/cpft_dl
RAW=$ROOT/data/raw
LOG=$ROOT/logs/download.log
mkdir -p $DL $RAW $ROOT/logs
unset http_proxy https_proxy
PX=${NUCSEG3D_PROXY:-}   # release: optional HTTP proxy
log() { echo "$(date +%m-%d_%H:%M:%S) $*" | tee -a $LOG; }
get() {  # 名称 文件名 URL
  local name=$1 fn=$2 url=$3
  mkdir -p $DL/$name $RAW/$name
  log "开始下载 $name/$fn"
  aria2c -q -c -x16 -s16 -k1M --max-tries=10 --retry-wait=5 ${PX:+--all-proxy=$PX} -d $DL/$name -o "$fn" "$url" \
    && log "完成下载 $name/$fn ($(du -h "$DL/$name/$fn" | cut -f1))" || { log "下载失败 $name/$fn"; return 1; }
  if [[ "$fn" == *.zip ]]; then
    unzip -q -o "$DL/$name/$fn" -d $RAW/$name && log "已解压 $name" || log "解压失败 $name"
  fi
}
Z=https://zenodo.org/records
get cellos     Cellos-1.0.0.zip                $Z/10065287/files/Cellos-1.0.0.zip?download=1 &
get nis3d      NIS3D.zip                       $Z/11456029/files/NIS3D.zip?download=1 &
get nucverse3d raw.zip                         $Z/18517324/files/raw.zip?download=1 &
get purdue     ground_truth_and_synthetic.zip  $Z/7065147/files/ground_truth_and_synthetic.zip?download=1 &
get celegans   c_elegans_nuclei.zip            $Z/5942575/files/c_elegans_nuclei.zip?download=1 &
get crypt      training_data.zip               $Z/17565548/files/training_data.zip?download=1 &
wait
# Go-Nuclear：只从 14 GB 的包里远程抽取 5 个完整 3D 体积（3d_all_in_one/*.h5）
log "开始抽取 gonuclear/3d_all_in_one"
HTTPS_PROXY=$PX HTTP_PROXY=$PX ${NUCSEG3D_SHARED}/anystar/env/bin/python - <<'PY' >> $LOG 2>&1
from remotezip import RemoteZip
import os
url = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/026/S-BIAD1026/Files/Nuclei_training_segmentation/Training%20data%20and%20Models.zip"
out = os.path.join(os.environ["NUCSEG3D_ROOT"], "data/raw/gonuclear")  # release: exported env var
os.makedirs(out, exist_ok=True)
z = RemoteZip(url)
for i in z.infolist():
    if "/3d_all_in_one/" in i.filename and i.filename.endswith(".h5"):
        z.extract(i, out); print("extracted", i.filename, i.file_size)
PY
log "Go-Nuclear 抽取结束"
# Tapenade：复用已有的人工标注训练数据（stardist_training_datasets）
mkdir -p $RAW/tapenade && cp -r ${NUCSEG3D_SHARED}/data/tapenade/stardist_training_datasets $RAW/tapenade/ && log "已复制 Tapenade 标注数据"
log "第一批全部结束"
echo BATCH1_DONE >> $LOG
