import os, sys, json
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
sys.path.insert(0, "src")
import realdata as rd, metrics_cost as mc
from detectors import CUSUM, calibrate_arl0, average_run_length
P = []
def chk(n, c, d=""): P.append(c); print(("PASS" if c else "FAIL"), "|", n, d)
SPW = 15.0
for ds in ["fashionmnist", "mnist", "svhn"]:
    rd.set_dataset(ds); res = rd.available_residuals()
    print(f"\n===== {ds}  (residuals: {res}) =====")
    bc = bs = bd = 0
    for m in ("isi", "cv"):
        for r in res:
            H = rd.pool(m, 0, r, "h"); F = rd.pool(m, 0, r, "f")  # spot neuron 0
            for c in range(10):
                H = rd.pool(m, c, r, "h"); F = rd.pool(m, c, r, "f")
                if abs(H.mean()) > 0.06: bc += 1
                if not (0.7 < H.std() < 1.3): bs += 1
                if abs(F.mean() - rd.neuron_shift(m, c, r)) > 0.03: bd += 1
    chk(f"{ds}: healthy centered", bc == 0, f"({bc} viol)")
    chk(f"{ds}: healthy unit-scaled", bs == 0, f"({bs} viol)")
    chk(f"{ds}: faulty mean==delta", bd == 0, f"({bd} viol)")
    # ARL0 calibration on raw isi
    cal = rd.sample_streams("isi", "raw", "healthy", 3500, 450, seed=3)
    det = calibrate_arl0(lambda h: CUSUM(0.5, h, two_sided=True), 200.0, cal, 0.5, 14.0)
    arl0 = average_run_length(det, rd.sample_streams("isi", "raw", "healthy", 5000, 900, seed=555))
    chk(f"{ds}: ARL0~200", 150 < arl0 < 300, f"(ARL0={arl0:.0f})")
    # e3 savings closed-form
    for m in ("isi", "cv"):
        e = json.load(open(f"results/e3_real_{ds}_{m}.json"))
        ok = abs(e["savings_once"] - mc.savings_factor(SPW, e["flag_rate_once"])) < 0.05 and abs(e["savings_vs_vp"] - mc.savings_factor(SPW, e["flag_rate"])) < 0.05
        chk(f"{ds} {m}: E3 savings==closed form", ok, f"(once {e['savings_once']}x recall {e['stage1_recall']}/{e['cascade_recall']})")
# golden MAE fidelity from master summary
M = json.load(open("results/real_summary_all.json"))["golden"]
for ds in ("fashionmnist", "mnist", "svhn"):
    for m in ("isi", "cv"):
        e = M[ds][m]["mean_rel_err"]; chk(f"{ds} {m}: golden MAE<10% vs shipped", e < 0.10, f"({e*100:.1f}%)")
print("\nSUMMARY:", sum(P), "/", len(P), "checks passed")
