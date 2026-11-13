# 第三批：调研里列出但前两批没下的 3D 细胞核标注数据（加 Kromp 2D）
set -u
ROOT=${NUCSEG3D_ROOT}
DL=${NUCSEG3D_SCRATCH}/cpft_dl/batch3
LOG=$ROOT/logs/download_batch3.log
PX=${NUCSEG3D_PROXY:-}   # release: optional HTTP proxy
mkdir -p $DL
log() { echo "$(date +%m-%d_%H:%M:%S) $*" | tee -a $LOG; }
get() {  # get 名称 文件名 URL [代理]
  mkdir -p $DL/$1
  if [ "${4:-px}" = px ]; then P="${PX:+--all-proxy=$PX}"; else P=""; fi
  aria2c -q -c -x16 -s16 -k4M --max-tries=20 --retry-wait=10 $P -d $DL/$1 -o "$2" "$3" \
    && log "完成 $1/$2 ($(du -h "$DL/$1/$2" | cut -f1))" || log "失败 $1/$2"
}
Z=https://zenodo.org/api/records
for e in 1 2 3 4; do get nisnet_zf Embryo$e.zip $Z/7671626/files/Embryo$e.zip/content; done
get mouse_embryo MouseEmbryos.zip $Z/6546550/files/MouseEmbryos.zip/content
for f in README.pdf Parhyale_H2B-EGFP_images_tp01-50.tif; do get parhyale $f $Z/8252039/files/$f/content; done
for t in 01 11 21 31 41 50; do f=Parhyale_H2B-EGFP_instance-segmentation-labels_tp$t.tif; get parhyale $f $Z/8252039/files/$f/content; done
for f in annotations_BIOP.zip crops_BIOP_v1.zip val_rdm42.zip; do get ratbrain $f $Z/6645978/files/$f/content; done
get mesospim DATASET_WITH_GT.zip $Z/11095111/files/DATASET_WITH_GT.zip/content
B=https://data.broadinstitute.org/bbbc
get bbbc050 Images.zip $B/BBBC050/Images.zip; get bbbc050 GroundTruth.zip $B/BBBC050/GroundTruth.zip
get bbbc032 BBBC032_v1_dataset.zip $B/BBBC032/BBBC032_v1_dataset.zip; get bbbc032 BBBC032_v1_DatasetGroundTruth.tif $B/BBBC032/BBBC032_v1_DatasetGroundTruth.tif
mkdir -p $DL/kromp && (cd $DL/kromp && wget -q -r -np -nH --cut-dirs=6 -R "index.html*" https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/634/S-BIAD634/Files/) && log "完成 kromp ($(du -sh $DL/kromp | cut -f1))" || log "失败 kromp"
for d in nisnet_zf mouse_embryo parhyale ratbrain mesospim bbbc050 bbbc032 kromp; do
  mkdir -p $ROOT/data/raw/$d
  for z in $DL/$d/*.zip; do [ -f "$z" ] && unzip -q -o "$z" -d $ROOT/data/raw/$d/ && log "已解压 $(basename $z)"; done
  find $DL/$d -maxdepth 1 -type f ! -name "*.zip" -exec cp {} $ROOT/data/raw/$d/ \;
  [ $d = kromp ] && cp -r $DL/kromp/* $ROOT/data/raw/kromp/
done
log BATCH3_DONE
