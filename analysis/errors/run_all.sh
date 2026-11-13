#!/bin/bash
# NC 误差与 AI 分析（只用 CPU）一键重跑。已算过的体积会跳过；新种子（sched_state.json 里新完成的 S5_tGT_* / S5_tFP_* 运行）自动纳入。
# 五种子齐后直接再跑一次即为正式版（A2 会按新的种子组合重算；aggregate 用 S1 与二维都完整的种子）。
set -e
cd ${NUCSEG3D_SCRATCH}/nc/errors
PY=${NUCSEG3D_SHARED}/env_torch/bin/python
export NC_TORCH_THREADS=2 OMP_NUM_THREADS=2
W=${W:-16}
$PY vol_pass.py --workers $W 2>&1 | tee -a vol_pass.log | grep -v " ok " || true
grep -c ERR vol_pass.log && echo "有出错的作业，见 vol_pass.log" || true
$PY a2_pass.py --workers $W
rm -f _base_ruleT.json.stale
$PY aggregate.py
$PY figs.py
$PY make_report.py
