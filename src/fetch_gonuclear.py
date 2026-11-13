"""从 Go-Nuclear 的 14 GB 包里远程抽取 3d_all_in_one/*.h5，校验大小，不完整则重试。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
import os, time
from remotezip import RemoteZip
URL = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/026/S-BIAD1026/Files/Nuclei_training_segmentation/Training%20data%20and%20Models.zip"
OUT = f"{_NSP_ROOT}/data/raw/gonuclear"
for attempt in range(8):
    try:
        z = RemoteZip(URL)
        todo = [i for i in z.infolist() if "/3d_all_in_one/" in i.filename and i.filename.endswith(".h5")
                and not (os.path.exists(os.path.join(OUT, i.filename)) and os.path.getsize(os.path.join(OUT, i.filename)) == i.file_size)]
        if not todo:
            print("ALL_OK", flush=True); break
        for i in todo:
            p = os.path.join(OUT, i.filename)
            if os.path.exists(p): os.remove(p)
            z.extract(i, OUT); print("extracted", i.filename, os.path.getsize(p), "/", i.file_size, flush=True)
    except Exception as e:
        print("retry", attempt, repr(e)[:120], flush=True); time.sleep(20)
