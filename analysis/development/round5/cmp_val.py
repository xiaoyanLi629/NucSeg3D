"""第五轮：某个微调模型的验证集（视图规则）与 A1、二维同种子比较，逐数据集。用法：python cmp_val.py <run前缀> <种子> [C的run前缀]"""
import os as _nsp_os  # release: machine-specific paths come from environment variables (see README, Paths)
_NSP_SCRATCH = _nsp_os.environ.get('NUCSEG3D_SCRATCH', '/path/to/scratch')
import sys
run, seed = sys.argv[1], sys.argv[2]
src = open(f"{_NSP_SCRATCH}/r5/r5_screen.py").read().split("if __name__")[0]
exec(src)
A1R = {"0": ("orx_f9a4da66-77b6-440c-adcf-f62ac279a144_gtval", "orx_c7a77c61-bb5f-436c-942c-678787765d71_gtval"),
       "1000": ("orx_b8dbb6aa-201f-4ea1-ac91-4e645e793e47_gtval", "orx_15078456-6f8c-442b-a2f9-22e7c655313a_gtval"),
       "2000": ("orx_f76e7145-a114-40b9-b8e3-2c234cdc44a0_gtval", "orx_34192d05-fc09-4d1a-ad04-46bf9863e5ba_gtval")}
X = per(full(run)); A = rule(*A1R[seed]); D = per(R + "orx_2e0880e4-2e2c-408b-9d7e-397c60cd882e_gtval_orx_v5_2d_s" + seed)
C = per(full(sys.argv[3])) if len(sys.argv) > 3 else None
def line(nm, Y):
    return "%s %.4f/%.4f" % (nm, mac(Y, "f1_05"), mac(Y, "f1_07"))
print("12 套（去 mesospim）宏平均：" + "  ".join(line(n, Y) for n, Y in (("X", X), ("A1", A), ("二维", D)) + ((("C", C),) if C else ())))
for nm, Y in (("A1", A), ("二维", D)) + ((("C", C),) if C else ()):
    print("  X−%s %+.4f/%+.4f" % (nm, mac(X, "f1_05") - mac(Y, "f1_05"), mac(X, "f1_07") - mac(Y, "f1_07")))
for d in sorted(X):
    r = "  %-13s X %.3f/%.3f  A1 %.3f/%.3f  二维 %.3f/%.3f  X−A1 %+.3f/%+.3f" % (d, X[d]["f1_05"], X[d]["f1_07"], A[d]["f1_05"], A[d]["f1_07"], D[d]["f1_05"], D[d]["f1_07"],
                                                                         X[d]["f1_05"] - A[d]["f1_05"], X[d]["f1_07"] - A[d]["f1_07"])
    if C:
        r += "  X−C %+.3f/%+.3f" % (X[d]["f1_05"] - C[d]["f1_05"], X[d]["f1_07"] - C[d]["f1_07"])
    print(r)
