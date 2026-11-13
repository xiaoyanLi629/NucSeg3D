"""3D 实例分割评测：按 IoU 做匈牙利匹配，输出各 IoU 阈值下的 F1 等指标（论文用 F1@IoU0.5）。"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def iou_matrix(gt, pred):
    """返回 (n_gt, n_pred) 的 IoU 矩阵，以及两边的实例编号。"""
    gt = np.asarray(gt).ravel()
    pred = np.asarray(pred).ravel()
    gt_ids, gt_inv = np.unique(gt, return_inverse=True)
    pr_ids, pr_inv = np.unique(pred, return_inverse=True)
    # 交集体素数的列联表，第 0 行/列是背景
    inter = coo_matrix((np.ones_like(gt_inv, dtype=np.int64), (gt_inv, pr_inv)),
                       shape=(len(gt_ids), len(pr_ids))).toarray()
    gt_fg, pr_fg = gt_ids != 0, pr_ids != 0
    inter = inter[gt_fg][:, pr_fg]
    gt_size = np.bincount(gt_inv, minlength=len(gt_ids))[gt_fg]
    pr_size = np.bincount(pr_inv, minlength=len(pr_ids))[pr_fg]
    union = gt_size[:, None] + pr_size[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return iou, gt_ids[gt_fg], pr_ids[pr_fg]


def evaluate(gt, pred, thresholds=THRESHOLDS):
    """对一对标签图评测。匹配在全部 IoU 上做一次最优分配，再按阈值计 TP。"""
    iou, _, _ = iou_matrix(gt, pred)
    n_gt, n_pred = iou.shape
    res = {"n_gt": int(n_gt), "n_pred": int(n_pred),
           "count_err": float((n_pred - n_gt) / max(n_gt, 1))}
    if n_gt and n_pred:
        r, c = linear_sum_assignment(-iou)
        matched = iou[r, c]
    else:
        matched = np.zeros(0)
    for t in thresholds:
        tp = int((matched >= t).sum())
        fp, fn = n_pred - tp, n_gt - tp
        prec = tp / max(n_pred, 1)
        rec = tp / max(n_gt, 1)
        res[f"f1@{t}"] = 2 * tp / max(n_gt + n_pred, 1)
        res[f"prec@{t}"] = prec
        res[f"rec@{t}"] = rec
        res[f"acc@{t}"] = tp / max(tp + fp + fn, 1)
    hit = matched[matched >= 0.5]
    res["mean_iou_matched"] = float(hit.mean()) if hit.size else 0.0
    return res


if __name__ == "__main__":
    # 自检：完全相同得 1，整体平移后 F1 下降
    rng = np.random.default_rng(0)
    lab = np.zeros((20, 64, 64), np.uint16)
    for k in range(1, 11):
        z, y, x = rng.integers(4, 16), rng.integers(8, 56), rng.integers(8, 56)
        lab[z - 3:z + 3, y - 4:y + 4, x - 4:x + 4] = k
    assert evaluate(lab, lab)["f1@0.5"] == 1.0
    shifted = np.roll(lab, 3, axis=2)
    print("identical:", evaluate(lab, lab)["f1@0.5"], " shifted by 3 px:", round(evaluate(lab, shifted)["f1@0.5"], 3))
    print("empty pred:", evaluate(lab, np.zeros_like(lab))["f1@0.5"])
