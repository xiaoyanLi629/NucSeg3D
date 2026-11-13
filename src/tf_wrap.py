"""orx script 入口：orx_run 用 env_torch 启动脚本，而 StarDist 系模型要 TF 环境，这里转交给 $NUCSEG3D_ANYSTAR/env/bin/python（旧横评用的同一环境）。"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_ANYSTAR = _nsp_os.environ.get('NUCSEG3D_ANYSTAR', '/path/to/AnyStar')
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2")
raise SystemExit(subprocess.run([f"{_NSP_ANYSTAR}/env/bin/python", "-u", os.path.join(here, "bench_others2.py")] + sys.argv[1:], env=env, cwd=here).returncode)
