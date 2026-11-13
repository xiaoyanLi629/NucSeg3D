"""nc G1 补调参后的表（CPU）：与 g1/g1_table.py 相同口径，只把对方方法换成补调参结果（有 <reg>/<model>/test/rows.json 时用新的，有 kept_old.json 时保留旧结果）。
输出 g1/g1_table2.json、g1_table2.md。用法：python g1_table2.py <reg 目录>"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import json, os, re, sys
reg = sys.argv[1]
s = open(f"{_NSP_SCRATCH}/nc/gpu/g1/g1_table.py").read()
s = s.replace('g1_table.json', 'g1_table2.json').replace('g1_table.md', 'g1_table2.md')
s = s.replace('''    d = R + dn; S = json.load(open(d + "/summary.json")); out = {}
    for x in json.load(open(d + "/rows.json")):''', '''    m = dict(orx_1020_004144_others_tapenade="tapenade", orx_1020_004452_others_anystar="anystar", orx_1020_004452_others_cellos="cellos")[dn]
    d = R + dn
    if os.path.exists(f"{REG}/{m}/test/rows.json"):
        d = f"{REG}/{m}/test"
    else:
        assert os.path.exists(f"{REG}/{m}/kept_old.json"), m
    S = json.load(open(d + "/summary.json")); out = {}
    for x in json.load(open(d + "/rows.json")):''')
exec(compile("REG = %r\n" % reg + s, "g1_table2", "exec"), dict(__name__="__main__"))
