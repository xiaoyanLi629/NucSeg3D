"""零拷贝读取共享盘 iso30.tar 里的 .npy（tar 未压缩，成员数据连续存放，直接按偏移 memmap）。
用法：from iso30tar import Iso30; S = Iso30(); idx = S.index(); img = S.npy(e["img"])  # np.memmap，只读"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ROOT = _nsp_os.environ.get('NUCSEG3D_ROOT', '/path/to/CellposeFT')
_NSP_SHARED = _nsp_os.environ.get('NUCSEG3D_SHARED', '/path/to/NucSeg3D_shared')
import json
import os
import numpy as np

TAR = f"{_NSP_ROOT}/iso30/iso30.tar"
MEMBERS = f"{_NSP_SHARED}/baselines/data/iso30_tar_members.json"


class Iso30:
    def __init__(self, tar=TAR, members=MEMBERS):
        self.tar = tar
        self.m = json.load(open(members))

    def _off(self, rel):
        return self.m["cpft_iso30/" + rel]

    def index(self):
        off, size = self._off("index.json")
        with open(self.tar, "rb") as f:
            f.seek(off)
            return json.loads(f.read(size))

    def npy(self, rel):
        """返回 np.memmap（只读）。rel 形如 'bbbc050/train_Emb01_t001_img.npy'。"""
        off, size = self._off(rel)
        with open(self.tar, "rb") as f:
            f.seek(off)
            ver = np.lib.format.read_magic(f)
            shape, fortran, dtype = (np.lib.format.read_array_header_1_0(f) if ver == (1, 0)
                                     else np.lib.format.read_array_header_2_0(f))
            hdr = f.tell() - off
        return np.memmap(self.tar, dtype=dtype, mode="r", offset=off + hdr, shape=shape, order="F" if fortran else "C")

    def load(self, rel):
        return np.array(self.npy(rel))
