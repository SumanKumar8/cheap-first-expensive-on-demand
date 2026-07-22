"""Real-data E1/E2/E3 for any of the 3 datasets (fashionmnist/mnist/svhn).

Usage:
  python3 real_experiments.py e1  <dataset> <isi|cv>
  python3 real_experiments.py e3  <dataset> <isi|cv>
  python3 real_experiments.py e13 <dataset> <isi|cv>   # E1 then E3
  python3 real_experiments.py e2  <dataset>            # conditioning (needs ml)
Detectors run TWO-SIDED, matched to in-control ARL0 on real healthy streams.
Raw (model-free) residual is available for all datasets; ml/lin only where
test-input spikes exist (fashionmnist, mnist).
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from detectors import (PerWindowThreshold, CUSUM, EWMA, calibrate_arl0, average_run_length)
from eval_utils import roc_from_thresholds, auroc, tpr_at_fpr
import metrics_cost as mc
import realdata as rd

RESULTS = os.path.join(os.path.dirname(__file__), "results")
ARL0 = 200.0
L_EP, CP, N_EP = 100, 50, 2000
K = 0.5
SPW = 15.0

def cal_streams(metric, residual):
    return rd.sample_streams(metric, residual, "healthy", 3500, 450, seed=3)

def eval_row(make_det, lo, hi, metric, residual):
    pos = rd.sample_streams(metric, residual, "faulty", N_EP, L_EP, cp=CP, seed=1)
    neg = rd.sample_streams(metric, residual, "healthy", N_EP, L_EP, seed=2)
    thr = np.linspace(lo, hi, 40)
    fpr, tpr, _ = roc_from_thresholds(make_det, thr, pos, neg)
    au, t01 = auroc(fpr, tpr), tpr_at_fpr(fpr, tpr, 0.01)
    det = calibrate_arl0(make_det, ARL0, cal_streams(metric, residual), lo, hi)
    faulty0 = rd.sample_streams(metric, residual, "faulty0", 3500, 280, seed=99)
    lat = average_run_length(det, faulty0)
    return round(au, 4), round(t01, 4), round(lat, 2)

def e1(ds, metric):
    rd.set_dataset(ds); has_ml = "ml" in rd.available_residuals()
    pw = lambda h: PerWindowThreshold(h, two_sided=True)
    cu = lambda h: CUSUM(K, h, two_sided=True)
    ew = lambda h: EWMA(0.2, h, two_sided=True)
    rows = {}
    rows["per-window threshold, raw residual (Paper 1)"] = eval_row(pw, 1.5, 5.0, metric, "raw")
    rows["CUSUM, raw residual"] = eval_row(cu, 0.5, 14.0, metric, "raw")
    rows["EWMA, raw residual"] = eval_row(ew, 1.0, 5.0, metric, "raw")
    if has_ml:
        rows["per-window threshold, ML residual (Paper 1 deployed)"] = eval_row(pw, 1.5, 5.0, metric, "ml")
        rows["residual-CUSUM, linear conditioner"] = eval_row(cu, 0.5, 14.0, metric, "lin")
        rows["CUSUM, ML residual"] = eval_row(cu, 0.5, 14.0, metric, "ml")
    js = {k: dict(auroc=a, tpr_at_1pct_fpr=t, latency_windows=l) for k, (a, t, l) in rows.items()}
    json.dump(js, open(f"{RESULTS}/e1_real_{ds}_{metric}.json", "w"), indent=2)
    pwl = rows["per-window threshold, raw residual (Paper 1)"][2]
    cul = rows["CUSUM, raw residual"][2]
    print(f"[{ds} {metric}] per-window raw AUROC={rows['per-window threshold, raw residual (Paper 1)'][0]:.3f} lat={pwl:.1f}"
          f" | CUSUM raw AUROC={rows['CUSUM, raw residual'][0]:.3f} lat={cul:.1f}"
          f" | speedup {pwl/cul:.2f}x"
          + (f" | cond gain {rows['CUSUM, ML residual'][0]-rows['CUSUM, raw residual'][0]:+.3f}" if has_ml else " | (raw-only)"))

def e2(ds):
    rd.set_dataset(ds)
    if "ml" not in rd.available_residuals():
        print(f"[{ds}] no ml residual (no test inputs) -> conditioning study N/A"); return
    out = {}
    cu = lambda h: CUSUM(K, h, two_sided=True)
    for metric in ("isi", "cv"):
        r = {res: eval_row(cu, 0.5, 14.0, metric, res)[0] for res in ("raw", "lin", "ml")}
        out[metric] = dict(auroc_raw=r["raw"], auroc_lin=r["lin"], auroc_ml=r["ml"],
                           gain_ml_minus_raw=round(r["ml"]-r["raw"], 4))
        print(f"[{ds} {metric}] AUROC raw={r['raw']:.3f} lin={r['lin']:.3f} ml={r['ml']:.3f} gain={r['ml']-r['raw']:+.3f}")
    json.dump(out, open(f"{RESULTS}/e2_real_{ds}.json", "w"), indent=2)

def _cusum2s_events(x, k, h):
    Sp = Sn = 0.0; ev = []
    for t in range(len(x)):
        Sp = max(0.0, Sp + x[t] - k); Sn = max(0.0, Sn - x[t] - k)
        if Sp > h or Sn > h:
            ev.append(t); Sp = Sn = 0.0
    return ev

def e3(ds, metric, residual="raw"):
    rd.set_dataset(ds)
    dsobj = rd.neuron_dataset_real(metric=metric, residual=residual, n_per_neuron=80,
                                   L=L_EP, fault_fraction=0.5, seed=7)
    h_det = calibrate_arl0(lambda h: CUSUM(K, h, two_sided=True), ARL0,
                           cal_streams(metric, residual), 0.5, 14.0)
    k, h = K, h_det.threshold
    N, L = dsobj.M.shape
    faulty = dsobj.change_point >= 0
    rng = np.random.default_rng(123)
    tot = heal_esc = heal_win = alarmed = 0
    true_flag = np.zeros(N, bool); isolated = np.zeros(N, bool); lat = []
    for i in range(N):
        ev = _cusum2s_events(dsobj.M[i], k, h); tot += len(ev); alarmed += 1 if ev else 0
        cp = int(dsobj.change_point[i])
        if not faulty[i]:
            heal_esc += len(ev); heal_win += L
        else:
            heal_win += cp; heal_esc += sum(1 for t in ev if t < cp)
            post = [t for t in ev if t >= cp]
            if post:
                true_flag[i] = True
                if rng.random() > np.exp(-3.0):
                    isolated[i] = True; lat.append(post[0]-cp)
    nf = int(faulty.sum()); flag_rate = tot/(N*L); p_once = alarmed/(N*L)
    res = dict(dataset=ds, metric=metric, residual=residual, N=N, n_faulty=nf,
               flag_rate=round(flag_rate, 4), flag_rate_once=round(p_once, 4),
               stage1_recall=round(true_flag.sum()/max(1, nf), 4),
               cascade_recall=round(isolated.sum()/max(1, nf), 4),
               mean_latency=round(float(np.mean(lat)), 3) if lat else None,
               false_escalation_rate=round(heal_esc/max(1, heal_win), 5),
               savings_vs_vp=round(mc.savings_factor(SPW, flag_rate), 2),
               savings_once=round(mc.savings_factor(SPW, p_once), 2),
               crossover_pstar=round(mc.crossover_flag_rate(SPW), 4))
    json.dump(res, open(f"{RESULTS}/e3_real_{ds}_{metric}.json", "w"), indent=2)
    print(f"[{ds} {metric}] E3 recall {res['stage1_recall']}/{res['cascade_recall']} "
          f"lat {res['mean_latency']} false-esc {res['false_escalation_rate']} "
          f"savings once {res['savings_once']}x reset {res['savings_vs_vp']}x")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "e1": e1(sys.argv[2], sys.argv[3])
    elif cmd == "e3": e3(sys.argv[2], sys.argv[3])
    elif cmd == "e13": e1(sys.argv[2], sys.argv[3]); e3(sys.argv[2], sys.argv[3])
    elif cmd == "e2": e2(sys.argv[2])
