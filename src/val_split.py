"""从训练体积里划出固定的三维验证集（用于选模型、选阈值、选体积下限；测试集不参与选择）。

规则（不用随机数，结果固定）：
- 数据集自带验证子集的（id 含 "val_"：BlastoSPIM、Tapenade），从中均匀取，至多 max(1, round(0.12 n)) 个；
- BBBC050 整个留出最后一个胚胎（同一胚胎不同时间点高度相似，按胚胎划分避免泄漏）；
- Go-Nuclear 留出最后一个编号（clear/noisy 两个版本一起留出）；
- 其他数据集训练体积 ≥ 4 个的，按 id 排序后均匀取 max(1, round(0.12 n)) 个；不足 4 个的不留出（ratbrain 不在测试集里，也不留出）。
输出 val_split.json：{ds: [id, ...]}。训练时用 --exclude_json 去掉这些体积的切片。
用法：python val_split.py --out ../val_split.json
"""
import argparse, json, re
from collections import defaultdict
import numpy as np
from stddata import load_manifest


def make_split():
    by = defaultdict(list)
    for r in load_manifest(split="train", dim="3d"):
        by[r["ds"]].append(r["id"])
    out = {}
    for ds, ids in sorted(by.items()):
        ids = sorted(set(ids))
        native = [i for i in ids if "val_" in i]
        k = max(1, round(0.12 * len(ids)))
        if native:
            out[ds] = native if len(native) <= k else [native[int(j)] for j in np.linspace(0, len(native) - 1, k).round()]
        elif ds == "bbbc050":
            embs = sorted({re.search(r"Emb\d+", i).group(0) for i in ids})
            out[ds] = [i for i in ids if embs[-1] in i]
        elif ds == "gonuclear":
            last = sorted({i.split("_")[0] for i in ids})[-1]
            out[ds] = [i for i in ids if i.startswith(last + "_")]
        elif ds != "ratbrain" and len(ids) >= 4:
            out[ds] = [ids[int(j)] for j in np.linspace(0, len(ids) - 1, k + 2)[1:-1].round()]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../val_split.json")
    a = ap.parse_args()
    s = make_split()
    json.dump(s, open(a.out, "w"), indent=1, ensure_ascii=False)
    tot = {ds: len(v) for ds, v in s.items()}
    print("验证体积数:", tot, "合计", sum(tot.values()))
