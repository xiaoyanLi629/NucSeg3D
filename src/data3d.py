"""从 iso30 库取三维训练块（true3d_spec §4）。

流程：DataLoader 的 CPU 进程只从内存映射的 .npy 里读出源区域（含边距，按随机缩放换算到源尺寸），
主进程在 GPU 上做：缩放到 (Z+2M, 256+2M, 256+2M)（图像三线性、标签 nearest-exact）、翻转 / xy 旋转 90°、
批量三维扩散算流（所有细胞核一次扩散，邻居只在同一编号内，相当于逐核独立扩散）、裁掉边距、亮度增强。
目标 = [dz, dy, dx] 单位向量 + 是否细胞核；valid 标出在体积内部的体素（块伸出体积外的部分不计损失）。

flows3d_gpu 与 make_slices_flow3d.flows3d（逐核调用 Cellpose masks_to_flows_gpu_3d）的区别只在迭代步数：
这里所有核共用块内最大核的步数（Cellpose 的规则 6×外接框边长和），小核多迭代几步，方向几乎不变（见 __main__ 的对照）。
同一编号由几块不相连部分组成时（裁块切开、或个别标注如此），第一轮扩散到不了的部分再按“该编号剩余部分”重新取中心扩散，
与按连通域拆开的做法一致。

round 6（分辨率条件，res_cond=True 时，默认关闭）：每块多带一个 f = (f_z, f_y, f_x)，即模型三个轴上“一个源采样间隔等于
多少个网络体素”，见 block_res；只是记录，不多抽随机数，图像、标签与增强参数的序列与关闭时逐位相同。
"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_REPO = _nsp_os.path.abspath(_nsp_os.path.join(_nsp_os.path.dirname(_nsp_os.path.abspath(__file__)), ".."))  # repository root
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

ISO_DIR = f"{_NSP_SCRATCH}/cpft_iso30"
DEFAULT_W = {"cellos": 2.0, "tapenade": 2.0}
# 轴置换增强（round 2）：非恒等的 5 种置换。perm[k] = 模型第 k 轴对应的体积轴（np.transpose 的约定）。
PERMS = [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


# ---------------------------------------------------------------- 批量三维流
@torch.no_grad()
def flows3d_gpu(lab, max_pass=3, max_iter=4000, dtype=torch.float64):
    """lab: (Z,Y,X) 整数张量（GPU）。返回 (3,Z,Y,X) float32 单位向量 (dz,dy,dx)，背景为 0。"""
    dev = lab.device
    Z, Y, X = lab.shape
    lp = F.pad(lab.long(), (1, 1, 1, 1, 1, 1))
    Yp, Xp = Y + 2, X + 2
    flat = lp.reshape(-1)
    offs = torch.tensor([0, Yp * Xp, -Yp * Xp, Xp, -Xp, 1, -1], device=dev)
    mu_all = torch.zeros(3, flat.numel(), dtype=torch.float32, device=dev)
    todo = flat > 0
    for _ in range(max_pass):
        fidx = torch.nonzero(todo).squeeze(1)
        n = fidx.numel()
        if n == 0:
            break
        cur = torch.where(todo, flat, torch.zeros_like(flat))
        labs = cur[fidx]
        uniq, inv = torch.unique(labs, return_inverse=True)
        L = uniq.numel()
        zz = fidx // (Yp * Xp); yy = (fidx // Xp) % Yp; xx = fidx % Xp
        co = torch.stack([zz, yy, xx], 1).to(torch.float64)
        cnt = torch.bincount(inv, minlength=L).to(torch.float64)
        mean = torch.zeros(L, 3, dtype=torch.float64, device=dev).index_add_(0, inv, co) / cnt[:, None]
        d2 = ((co - mean[inv]) ** 2).sum(1)
        dmin = torch.full((L,), float("inf"), dtype=torch.float64, device=dev).scatter_reduce(0, inv, d2, "amin")
        ar = torch.arange(n, device=dev)
        key = torch.where(d2 == dmin[inv], ar, torch.full_like(ar, n))
        first = torch.full((L,), n, device=dev, dtype=torch.long).scatter_reduce(0, inv, key, "amin")
        centers = fidx[first]
        # 步数：Cellpose 规则 6 × (外接框各边长 + 1 之和) 的最大值
        ci = torch.stack([zz, yy, xx], 1)
        lo = torch.full((L, 3), 1 << 40, device=dev, dtype=torch.long).scatter_reduce(0, inv[:, None].expand(-1, 3), ci, "amin")
        hi = torch.full((L, 3), -1, device=dev, dtype=torch.long).scatter_reduce(0, inv[:, None].expand(-1, 3), ci, "amax")
        niter = int(min(6 * int(((hi - lo) + 2).sum(1).max()), max_iter))
        nb = fidx[None] + offs[:, None]                              # (7, n)
        isn = (cur[nb] == labs[None]).to(dtype)                       # 同一编号的邻居
        # 在压缩下标上迭代：把 nb 映射到 fg 序号（不在本轮前景的邻居 -> 指向哑元 n）
        pos = torch.full((flat.numel(),), n, device=dev, dtype=torch.long)
        pos[fidx] = ar
        nbc = pos[nb]                                                 # (7, n)，哑元 = n
        cen = pos[centers]
        T = torch.zeros(n + 1, dtype=dtype, device=dev)
        w = isn / 7.0
        for _ in range(niter):
            T[cen] += 1
            T[:n] = (T[nbc] * w).sum(0)
        Tn = T[nbc] * isn                                             # 只看同一编号的邻居（逐核扩散时别的核为 0）
        mu = torch.stack([Tn[1] - Tn[2], Tn[3] - Tn[4], Tn[5] - Tn[6]])
        mu = mu / (1e-60 + (mu ** 2).sum(0).sqrt())
        reached = T[:n] > 0
        mu_all[:, fidx[reached]] = mu[:, reached].float()
        todo[fidx[reached]] = False
    return mu_all.view(3, Z + 2, Yp, Xp)[:, 1:-1, 1:-1, 1:-1].contiguous()


# ---------------------------------------------------------------- 数据集
def load_index(store=ISO_DIR):
    idx = json.load(open(os.path.join(store, "index.json")))
    for e in idx:
        for k in ("img", "lab"):
            if not os.path.isabs(e[k]):
                e[k] = os.path.join(store, e[k])
    return idx


def dataset_probs(entries, weights=None, caps=None):
    """数据集份额 ∝ 权重（与二维配方 w/计数 相同），再把 caps 里的数据集份额压到上限；数据集内按体积均匀。"""
    weights = {**DEFAULT_W, **(weights or {})}
    caps = caps if caps is not None else {"blastospim": 0.25}
    dss = sorted({e["ds"] for e in entries})
    share = np.array([weights.get(d, 1.0) for d in dss], float)
    share /= share.sum()
    for d, c in caps.items():
        if d in dss:
            i = dss.index(d)
            if share[i] > c:
                rest = 1 - share[i]
                share = share * (1 - c) / rest; share[i] = c
    cnt = {d: sum(e["ds"] == d for e in entries) for d in dss}
    p = np.array([share[dss.index(e["ds"])] / cnt[e["ds"]] for e in entries])
    return p / p.sum(), dict(zip(dss, share.round(4).tolist()))


def _drop_cache(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _read(path, src):
    mm = np.load(path, mmap_mode="r")
    out = np.array(mm[tuple(src)])
    del mm
    _drop_cache(path)
    return out


class PatchSource(torch.utils.data.IterableDataset):
    """无限流：每个样本是源区域 (img float16, lab uint32) 及其增强参数。"""
    def __init__(self, entries, probs, patch=(32, 256, 256), margin=32, scale=(0.8, 1.25), p_fg=0.9, seed=0, perm_aug=0.0,
                 aniso=None, zstretch=None, res_cond=False):
        super().__init__()
        self.entries, self.probs = entries, probs
        self.patch, self.margin, self.scale, self.p_fg, self.seed = patch, margin, scale, p_fg, seed
        # res_cond（round 6）：在 aug["res"] 里记下置换、缩放后模型各轴的源采样间隔（网络体素），不抽随机数；False 时与旧版逐位相同。
        self.res_cond = bool(res_cond)
        # perm_aug：以此概率把体积的三个轴随机置换后再当作 (Z,Y,X)（体积各向同性，模型的 z 可以是任一体积轴）。
        # 0 = 关闭，此时不多抽随机数，样本流与旧版逐位相同。
        self.perm_aug = float(perm_aug)
        # aniso（round 4，各向异性模拟增强）：dict(p, lo, hi, xy, blur)，见 aniso_params；None 或 p = 0 时不多抽随机数。
        self.aniso = aniso if aniso and aniso.get("p", 0) > 0 else None
        # zstretch（round 5）：dict(p, a)，以概率 p 把模型 z 轴的缩放再乘 e ~ 对数均匀 [1/a, a]（图像、标签一起拉伸）；None 时不多抽随机数。
        self.zstretch = zstretch if zstretch and zstretch.get("p", 0) > 0 else None

    def aniso_params(self, rng, i, s, perm, sz=None):
        """以概率 p 抽一个“目标采样间隔” F ~ 对数均匀 [lo, hi]（网络网格体素），作用轴以概率 xy 取模型的 y、x 两轴
        （像 mesospim 那样 xy 被插值放大），否则取模型 z 轴（像 cellos / parhyale）。该轴在 iso30 库里已有的插值放大倍数
        g = factors[轴] × 缩放增强 s（< 1 时按 1）；只在 F / g > 1.25 时生效，额外抽样间隔 f = F / g，
        即把块模拟成“该轴原始采样间隔为 F 体素、再线性插值回来”的体积。返回 None 或 (f, 相位, "z"|"xy", blur)。
        注（round 6 核对）：aniso_sim 是在网络网格上每 f 体素取一层，所以块在该轴的实际采样间隔是 max(g, f) = max(g, F/g)，
        只有 g ≈ 1 时才等于 F（例如 tapenade g≈2.2 时 F=10 只模拟出约 4.5）。A["abs"]（round 6，默认没有）为真时改为
        f = F（只在 F > 1.25·g 时生效），实际采样间隔就是 F；随机数的抽法与旧版相同。"""
        A = self.aniso
        if rng.random() >= A["p"]:
            return None
        F_t = float(math.exp(rng.uniform(math.log(A["lo"]), math.log(A["hi"]))))
        xy = A["xy"] > 0 and rng.random() < A["xy"]
        fac = self.entries[i]["factors"]
        fm = [fac[k] for k in perm] if perm is not None else list(fac)   # 模型各轴对应的体积轴放大倍数
        g = max(max(fm[1], fm[2]) * s if xy else fm[0] * (s if sz is None else sz), 1.0)   # sz：z 轴单独拉伸后的缩放（round 5）
        if A.get("abs"):                                                # round 6：F 就是网络网格上的模拟采样间隔
            f = F_t
            if f <= 1.25 * g:
                return None
        else:
            f = F_t / g
            if f <= 1.25:
                return None
        ph = float(rng.uniform(0, f))
        b = float(A["blur"])
        if A.get("blur_max", 0) > b:                                     # round 5：模糊系数在 [blur, blur_max] 均匀抽样
            b = float(rng.uniform(b, A["blur_max"]))
        return (f, ph, "xy" if xy else "z", b)

    def _vol(self, i):
        # 只缓存形状与前景坐标；体积本身每次按需读一块后立即释放映射并丢掉页缓存。
        # 一直挂着 memmap 会让每个 DataLoader 进程的 RSS 涨到 10 GB 以上（页缓存算进容器内存），
        # 也会占满文件句柄，平台会因此杀进程。
        c = self._cache.get(i)
        if c is None:
            e = self.entries[i]
            lab = np.load(e["lab"], mmap_mode="r")
            sub = np.asarray(lab[::4, ::4, ::4]); shp = lab.shape
            del lab; _drop_cache(e["lab"])
            fg = np.argwhere(sub > 0) * 4
            if len(fg) > 20000:
                fg = fg[np.random.default_rng(i).choice(len(fg), 20000, replace=False)]
            c = self._cache[i] = (shp, fg)
        return c

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi else 0
        rng = np.random.default_rng([self.seed, wid, os.getpid() % 100000])
        self._cache = {}
        Pz, Py, Px = self.patch; M = self.margin
        out = np.array([Pz + 2 * M, Py + 2 * M, Px + 2 * M])
        while True:
            i = int(rng.choice(len(self.entries), p=self.probs))
            shp0, fg = self._vol(i)
            s = float(math.exp(rng.uniform(math.log(self.scale[0]), math.log(self.scale[1]))))
            perm = None
            if self.perm_aug > 0 and rng.random() < self.perm_aug:
                perm = PERMS[int(rng.integers(len(PERMS)))]
            sv = np.array([s, s, s])                                    # 模型 (z, y, x) 各轴的缩放
            if self.zstretch is not None and rng.random() < self.zstretch["p"]:   # round 5：z 轴单独拉伸/压缩
                sv[0] *= float(math.exp(rng.uniform(-math.log(self.zstretch["a"]), math.log(self.zstretch["a"]))))
            # 源区域在体积坐标里的大小：模型第 k 轴 = 体积第 perm[k] 轴，所以体积轴 perm[k] 取模型轴 k 的长度
            out_v, pat_v, sv_v = out.copy(), np.array([Pz, Py, Px]), sv.copy()
            if perm is not None:
                out_v[list(perm)] = out; pat_v[list(perm)] = np.array([Pz, Py, Px]); sv_v[list(perm)] = sv
            R = np.ceil(out_v / sv_v).astype(int) + 1                 # 源区域大小（体积坐标）
            shp = np.array(shp0)
            if len(fg) and rng.random() < self.p_fg:
                c = fg[rng.integers(len(fg))] + rng.integers(0, 4, 3)
                jit = (pat_v / sv_v / 2 * rng.uniform(-0.8, 0.8, 3)).astype(int)
                c = c + jit
            else:
                c = np.array([rng.integers(0, n) for n in shp])
            st = c - R // 2
            src = [slice(max(a, 0), min(a + r, n)) for a, r, n in zip(st, R, shp)]
            dst = [slice(s_.start - a, s_.stop - a) for s_, a in zip(src, st)]
            im = np.zeros(R, np.float16); lb = np.zeros(R, np.uint32); va = np.zeros(R, np.uint8)
            if all(s_.stop > s_.start for s_ in src):
                e = self.entries[i]
                im[tuple(dst)] = _read(e["img"], src); lb[tuple(dst)] = _read(e["lab"], src); va[tuple(dst)] = 1
            if perm is not None:   # 转成模型的 (Z,Y,X) 布局；流目标在 to_patch 里由置换后的标签现算，所以自动一致
                im, lb, va = (np.ascontiguousarray(t.transpose(perm)) for t in (im, lb, va))
            aug = dict(scale=s, flip=rng.random(3) < 0.5, rot=int(rng.integers(4)),
                       gain=float(rng.uniform(0.75, 1.25)) if rng.random() < 0.5 else 1.0,
                       gamma=float(math.exp(rng.uniform(math.log(0.7), math.log(1.4)))) if rng.random() < 0.5 else 1.0)
            if perm is not None:
                aug["perm"] = perm
            if sv[0] != s:
                aug["ez"] = float(sv[0] / s)
            if self.res_cond:   # round 6：模型轴 k = 体积轴 pv[k]；iso30 放大倍数 × 本块的实际缩放 out/R（≈ 缩放增强 × z 拉伸）
                pv = perm if perm is not None else (0, 1, 2)
                fac = self.entries[i]["factors"]
                aug["res"] = [float(fac[pv[k]]) * float(out[k]) / float(R[pv[k]]) for k in range(3)]
            if self.aniso is not None:
                an = self.aniso_params(rng, i, s, perm, sz=float(sv[0]))
                if an is not None:
                    aug["aniso"] = an
            yield dict(img=im, lab=lb, valid=va, aug=aug, ds=self.entries[i]["ds"], name=self.entries[i]["name"])


def _collate(b):
    return b


@torch.no_grad()
def interface_mask(lb, r):
    """lb: (Z,Y,X) 整数标签（GPU）。返回 bool (Z,Y,X)：以该体素为中心、边长 2r+1 的立方窗里有 ≥2 个不同的非零编号
    （即离另一个核 ≤ r 体素（切比雪夫距离）的前景体素，以及夹在两个核之间的背景缝隙）。用两次 max_pool3d 实现。"""
    k = 2 * r + 1
    lf = lb.float()[None, None]                                       # 编号已在 to_patch 里压成 0..n，float32 精确
    mx = F.max_pool3d(lf, k, 1, r)[0, 0]
    big = 1e9
    mn = -F.max_pool3d(-torch.where(lf > 0, lf, torch.full_like(lf, big)), k, 1, r)[0, 0]
    return (mx > 0) & (mn < mx)


def _erode(x, cube):
    """x: (Z,Y,X) float 0/1。cube=True 用 3×3×3（26 邻域）腐蚀，否则 6 邻域十字腐蚀。块外按“前景”处理
    （只影响边距内的体素，边距 32 远大于距离上限）。"""
    if cube:
        return -F.max_pool3d(-x[None, None], 3, 1, 1)[0, 0]
    p = F.pad(x[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]
    Z, Y, X = x.shape
    m = torch.minimum(p[:-2, 1:-1, 1:-1], p[2:, 1:-1, 1:-1])
    m = torch.minimum(m, torch.minimum(p[1:-1, :-2, 1:-1], p[1:-1, 2:, 1:-1]))
    m = torch.minimum(m, torch.minimum(p[1:-1, 1:-1, :-2], p[1:-1, 1:-1, 2:]))
    return torch.minimum(m, x)


@torch.no_grad()
def signed_dist(fg, cap):
    """fg: (Z,Y,X) bool（前景 = 任一细胞核，核与核接触处不算边界）。返回 float32 带符号距离（体素）：
    前景第 k 层（k=1 为贴着背景的那层）= +(k − 0.5)，背景第 k 层 = −(k − 0.5)，|s| 封顶 cap + 0.5；
    s = 0 正好是前景/背景的分界面。用交替的 6 邻域 / 26 邻域腐蚀计层数（八边形距离，近似欧氏距离），全在 GPU 上。"""
    out = []
    for m in (fg.float(), (~fg).float()):
        cur, d = m, torch.zeros_like(m)
        for k in range(int(cap)):
            cur = _erode(cur, cube=(k % 2 == 1))
            d = d + cur
        out.append(d)
    return torch.where(fg, out[0] + 0.5, -(out[1] + 0.5))


def _resample_matrix(L, f, phase, blur, device):
    """(L, L) 矩阵 R：沿长 L 的轴每隔 f 体素取一层（位置 phase + k·f；blur > 0 时取样核是 σ = blur·f 的高斯，
    模拟轴向 PSF 的层厚积分，否则线性插值取值），再线性插值回 0..L−1 的每一层。y = R @ x（沿该轴）。
    取样位置超出两端时按边缘复制（只影响边距内的体素）。"""
    k0 = math.floor(-phase / f); k1 = math.ceil((L - 1 - phase) / f)
    p = phase + f * torch.arange(k0, k1 + 1, dtype=torch.float64, device=device)    # (K,) 覆盖 [0, L−1] 两端之外各一层
    j = torch.arange(L, dtype=torch.float64, device=device)
    if blur > 0:
        S = torch.exp(-0.5 * ((j[None] - p.clamp(0, L - 1)[:, None]) / (blur * f)) ** 2)
        S = S / S.sum(1, keepdim=True)
    else:
        S = (1 - (j[None] - p.clamp(0, L - 1)[:, None]).abs()).clamp(min=0)          # 线性插值取值 (K, L)
    # 插值回去：第 j 层落在取样位置 p[m] 与 p[m+1] 之间
    m = ((j - p[0]) / f).floor().long().clamp(0, len(p) - 2)
    t = (j - p[m]) / f
    B = torch.zeros(L, len(p), dtype=torch.float64, device=device)
    B[torch.arange(L, device=device), m] = 1 - t
    B[torch.arange(L, device=device), m + 1] = t
    return (B @ S).float()


def block_res(a):
    """round 6：块的分辨率条件 f = [f_z, f_y, f_x]（模型轴上“一个源采样间隔 = 多少网络体素”）。PatchSource 记下的
    a["res"] 已含轴置换与缩放（out/R）；xy 旋转 90°/270° 时 y、x 对调；各向异性模拟（在网络网格上每 f_sim 体素取一层
    再线性插值回来）把被模拟轴改成 max(原间隔, f_sim)——两次取样里较粗的那次决定该轴的有效采样间隔。翻转不改 f。"""
    r = [float(v) for v in a["res"]]
    if a["rot"] % 2:
        r[1], r[2] = r[2], r[1]
    if "aniso" in a:
        f_sim, _, axes, _ = a["aniso"]
        for k in ((0,) if axes == "z" else (1, 2)):
            r[k] = max(r[k], float(f_sim))
    return r


def aniso_sim(im, f, phase, axes, blur):
    """im: (Z,Y,X)。axes = "z"：沿模型 z 轴模拟低采样再插值回来；"xy"：y、x 两轴各做一次（同一 f、相位）。"""
    for ax in ((0,) if axes == "z" else (1, 2)):
        R = _resample_matrix(im.shape[ax], f, phase, blur, im.device)
        im = torch.movedim(torch.tensordot(R, torch.movedim(im, ax, 0), dims=1), 0, ax)
    return im.contiguous()


@torch.no_grad()
def to_patch(sample, device, patch=(32, 256, 256), margin=32, iface_r=0, sdt_cap=0):
    """源区域 -> GPU 上的 (img (1,Z,Y,X), target (4,Z,Y,X) = [dz,dy,dx,cp], valid (Z,Y,X))。
    iface_r > 0 时多返回一项 iface (Z,Y,X) bool（见 interface_mask，在含边距的块上算，再裁掉边距）。
    sdt_cap > 0（round 3）时再多返回一项 sdt (Z,Y,X) float32：前景/背景带符号距离（见 signed_dist），在含边距的块上算再裁掉。
    样本带 aug["res"]（round 6，PatchSource res_cond=True）时最后再多返回一项 res (3,) float32（见 block_res）。
    都关时返回值与旧版逐位相同。"""
    Pz, Py, Px = patch; M = margin
    out = (Pz + 2 * M, Py + 2 * M, Px + 2 * M)
    im = torch.from_numpy(sample["img"].astype(np.float32)).to(device)[None, None]
    lb = torch.from_numpy(sample["lab"].astype(np.int64)).to(device)
    va = torch.from_numpy(sample["valid"]).to(device)[None, None].float()
    u, inv = torch.unique(lb, return_inverse=True)                    # 先重新编号，nearest 插值时 float32 精确
    if u[0] != 0:
        inv = inv + 1
    lbf = inv.float()[None, None]
    im = F.interpolate(im, size=out, mode="trilinear", align_corners=False)[0, 0]
    lbf = F.interpolate(lbf, size=out, mode="nearest-exact")[0, 0]
    va = F.interpolate(va, size=out, mode="nearest-exact")[0, 0]
    lb = lbf.round().long()
    a = sample["aug"]
    dims = [d for d, f in zip((0, 1, 2), a["flip"]) if f]
    if dims:
        im, lb, va = im.flip(dims), lb.flip(dims), va.flip(dims)
    if a["rot"]:
        im, lb, va = (torch.rot90(t, a["rot"], (1, 2)) for t in (im, lb, va))
    if "aniso" in a:                                                  # round 4：只改图像，标签与流目标仍是真三维的
        im = aniso_sim(im, *a["aniso"])
    mu = flows3d_gpu(lb)
    c = (slice(M, M + Pz), slice(M, M + Py), slice(M, M + Px))
    iface = interface_mask(lb, iface_r)[c] if iface_r > 0 else None
    sdt = signed_dist(lb > 0, sdt_cap)[c] if sdt_cap > 0 else None
    im, lb, va, mu = im[c], lb[c], va[c], mu[(slice(None),) + c]
    if a["gamma"] != 1.0:
        im = torch.where(im > 0, im.clamp(min=0) ** a["gamma"], im)
    im = im * a["gain"]
    tgt = torch.cat([mu, (lb > 0).float()[None]], 0)
    out = [im[None].contiguous(), tgt.contiguous(), va.contiguous()]
    if iface is not None:
        out.append(iface.contiguous())
    if sdt is not None:
        out.append(sdt.contiguous())
    if "res" in a:                                                    # round 6：放在最后，train3d 先取走
        out.append(torch.tensor(block_res(a), dtype=torch.float32, device=device))
    return tuple(out)


def make_loader(store=ISO_DIR, split="train", weights=None, caps=None, patch=(32, 256, 256), margin=32,
                workers=4, seed=0, exclude_ds=("mesospim",), perm_aug=0.0, aniso=None, zstretch=None, res_cond=False,
                include_val=False, exclude_ids=None):
    # r9 交叉验证（默认关闭，体积列表与顺序同旧版）：include_val 把 split=val 的体积也纳入，exclude_ids 是要去掉的 {(ds, id)}
    splits = (split, "val") if include_val else (split,)
    ex = exclude_ids or set()
    idx = [e for e in load_index(store) if e["split"] in splits and e["ds"] not in exclude_ds and (e["ds"], e["id"]) not in ex]
    probs, share = dataset_probs(idx, weights, caps)
    src = PatchSource(idx, probs, patch, margin, seed=seed, perm_aug=perm_aug, aniso=aniso, zstretch=zstretch, res_cond=res_cond)
    dl = torch.utils.data.DataLoader(src, batch_size=1, num_workers=workers, collate_fn=_collate,
                                     prefetch_factor=4 if workers else None, persistent_workers=bool(workers))
    return dl, idx, share


# ---------------------------------------------------------------- 自检：与逐核 flows3d 对照、测吞吐
if __name__ == "__main__":
    import argparse, sys, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=f"{_NSP_SCRATCH}/true3d_dev/iso_tiny")
    ap.add_argument("--gpu", default="cuda:1")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patch_z", type=int, default=32)
    a = ap.parse_args()
    dev = torch.device(a.gpu)
    torch.cuda.set_device(dev)
    patch = (a.patch_z, 256, 256)
    dl, idx, share = make_loader(a.store, weights=None, patch=patch, workers=a.workers)
    print("体积", len(idx), "数据集份额", share)
    sys.path.insert(0, _nsp_os.path.join(_NSP_REPO, 'src'))
    from make_slices_flow3d import flows3d
    it = iter(dl)
    # 1) 与逐核 flows3d 对照（3 个块）
    for k in range(3):
        s = next(it)[0]
        Pz, Py, Px = patch; M = 32
        out = (Pz + 2 * M, Py + 2 * M, Px + 2 * M)
        lb = torch.from_numpy(s["lab"].astype(np.int64)).to(dev)
        u, inv = torch.unique(lb, return_inverse=True)
        lbf = F.interpolate(inv.float()[None, None], size=out, mode="nearest-exact")[0, 0].round().long()
        torch.cuda.synchronize(); t0 = time.time()
        mu = flows3d_gpu(lbf); torch.cuda.synchronize(); t1 = time.time()
        ref = flows3d(lbf.cpu().numpy().astype(np.int32), dev); t2 = time.time()
        fg = (lbf > 0).cpu().numpy()
        m = mu.cpu().numpy()[:, fg]; r = ref[:, fg]
        nm, nr = np.linalg.norm(m, axis=0), np.linalg.norm(r, axis=0)
        ok = (nm > 1e-6) & (nr > 1e-6)
        cos = (m[:, ok] * r[:, ok]).sum(0) / (nm[ok] * nr[ok])
        print(f"块{k} {s['ds']}: 核 {int(lbf.max())}，前景 {fg.sum()}，批量 {t1 - t0:.2f}s / 逐核 {t2 - t1:.2f}s；"
              f"余弦 均值 {cos.mean():.4f} 1%分位 {np.percentile(cos, 1):.3f}；零向量 批量 {(nm < 1e-6).mean():.4f} 逐核 {(nr < 1e-6).mean():.4f}")
    # 2) 吞吐
    torch.cuda.synchronize(); t0 = time.time(); tg = 0
    for k in range(a.n):
        s = next(it)[0]
        t1 = time.time()
        x, tgt, va = to_patch(s, dev, patch)
        torch.cuda.synchronize(); tg += time.time() - t1
    dt = time.time() - t0
    print(f"吞吐 {a.n / dt:.2f} 块/秒（GPU 处理 {tg / a.n:.3f} s/块）；x {tuple(x.shape)} tgt {tuple(tgt.shape)} "
          f"前景比例 {float(tgt[3].mean()):.3f} 有效 {float(va.mean()):.3f}")
    print("DATA3D_SELFTEST_DONE")
