#!/bin/bash
# Partial-volume control: block-downsample every reference dataset in xy so that its median GT xy diameter matches
# mesospim's (from results_main/summary.json), downsample labels by majority (neutral) or `any` (inclusive) rule,
# and rerun the identical measurement. CPU only, nice 19, read-only on data.
set -e
cd ${NUCSEG3D_SCRATCH}/r5/agent_gt_conv
PY=${NUCSEG3D_SHARED}/env_torch/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PY - <<'EOF'
import json
S = json.load(open("results_main/summary.json"))
target = S["mesospim"]["metrics"]["run_xy"]["med"]
fac = {d: max(1, round(S[d]["metrics"]["run_xy"]["med"] / target)) for d in S if not d.startswith("_") and d != "mesospim"}
json.dump(dict(target_run_xy=target, factors=fac), open("control_factors.json", "w"), indent=1)
print("target", target, fac)
EOF
DOWN=$($PY -c "import json; f=json.load(open('control_factors.json'))['factors']; print(' '.join(f'{k}={v}' for k,v in f.items() if v>1))")
DS=$($PY -c "import json; f=json.load(open('control_factors.json'))['factors']; print(' '.join(k for k,v in f.items() if v>1))")
# order: interleave large and small volumes
ORDER="blastospim purdue gonuclear tapenade nis3d bbbc050 mouse_embryo ratbrain celegans parhyale nucverse3d cellos nisnet_zf"
SEL=""
for d in $ORDER; do case " $DS " in *" $d "*) SEL="$SEL $d";; esac; done
for RULE in majority any; do
  nice -n 19 $PY gt_conv.py --out results_ctrl_$RULE --workers 3 --gt_rule $RULE --down $DOWN --ds $SEL
  nice -n 19 $PY summarize.py results_ctrl_$RULE --focus none > results_ctrl_$RULE/summary_print.txt
done
echo CONTROL_DONE
