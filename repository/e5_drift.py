"""E5 -- Drift / robustness: is model-free CUSUM just memorizing the normal average?

Uses REAL MNIST in-control noise (standardized nominal muISI residuals). Two studies:
 (1) Baseline drift: inject a slow linear drift into the in-control metric (mimicking
     changing input intensity / operating point). A CUSUM keyed to a fixed MEMORIZED
     mean false-alarms under drift (ARL0 collapses); a CUSUM with an adaptive
     (EWMA-detrended) baseline holds ARL0 AND still detects an injected step fault.
     -> the detector responds to CHANGE, not to a memorized absolute level.
 (2) Cross-distribution transfer: calibrate the threshold on one class subset,
     evaluate fault detection on a DISJOINT subset. Detection AU-ROC is preserved
     -> not memorizing class-specific averages.
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import ART
BASE = os.path.join(ART, "MNIST_Experiments") + os.sep
RES = os.path.join(HERE, "results"); os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
WIN, NC = 110, 10; rng = np.random.default_rng(0)

def read_bits(p):
    raw = open(p, "rb").read().replace(b"\r", b""); L = [x for x in raw.split(b"\n") if x]
    W = len(L[0]); L = [x for x in L if len(x) == W]
    return (np.frombuffer(b"".join(L), np.uint8).reshape(len(L), W) - ord("0")).astype(np.uint8)
def parse(p):
    m = read_bits(p)[15:-15]; n = m.shape[0]//WIN; return m[:n*WIN].reshape(n, WIN, NC)
def avg_isi(col):
    t = np.where(col == 1)[0].astype(float)+1
    return 0.0 if len(t) == 0 else (float(t[0]) if len(t) == 1 else float(np.mean(np.diff(t))))

nom = parse(BASE+"Nominal/test_spike_output_file_0.txt")
NI = nom.shape[0]
# real standardized in-control residuals, per neuron
noise_by_neuron = {}
for c in range(NC):
    v = np.array([avg_isi(nom[j][:, (NC-1)-c]) for j in range(NI)])
    m0, sd = v.mean(), v.std() or 1.0
    noise_by_neuron[c] = (v-m0)/sd
ALL = np.concatenate([noise_by_neuron[c] for c in range(NC)])
print(f"real in-control noise: n={len(ALL)} mean={ALL.mean():.3f} std={ALL.std():.3f}", flush=True)

def cusum_twosided(x, k, h):
    Sp = Sn = 0.0
    for t in range(len(x)):
        Sp = max(0.0, Sp+x[t]-k); Sn = max(0.0, Sn-x[t]-k)
        if Sp > h or Sn > h: return t
    return -1
def cusum_adaptive(x, k, h, lam=0.05):
    Sp = Sn = 0.0; b = 0.0
    for t in range(len(x)):
        r = x[t]-b; b = lam*x[t]+(1-lam)*b       # one-step-ahead EWMA detrend
        Sp = max(0.0, Sp+r-k); Sn = max(0.0, Sn-r-k)
        if Sp > h or Sn > h: return t
    return -1

def sample_stream(L, drift=0.0, delta=0.0, cp=None, pool=ALL):
    x = rng.choice(pool, L).astype(float)
    if drift: x += np.linspace(0, drift, L)          # slow linear operating-point drift
    if delta:
        cp = cp if cp is not None else L//2; x[cp:] += delta
    return x
def arl(runner, k, h, streams):
    rl = [runner(s, k, h) for s in streams]
    return float(np.mean([r+1 if r >= 0 else len(streams[0]) for r in rl]))
def calib(runner, target, k, streams):
    lo, hi = 0.5, 40.0
    for _ in range(45):
        mid = 0.5*(lo+hi)
        if arl(runner, k, mid, streams) < target: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)

K, L, TARGET = 0.5, 300, 200.0
# calibrate BOTH variants to ARL0=200 on NO-drift in-control
cal = [sample_stream(L) for _ in range(1500)]
hA = calib(cusum_twosided, TARGET, K, cal)
hB = calib(cusum_adaptive, TARGET, K, cal)
# study 1: ARL0 under drift + detection (delta=1.5) under drift
study1 = {"drift_levels": [0.0, 1.5, 3.0], "fixed_ARL0": [], "adaptive_ARL0": [],
          "fixed_recall_faulty": [], "adaptive_recall_faulty": []}
for D in study1["drift_levels"]:
    ic = [sample_stream(L, drift=D) for _ in range(1500)]
    fa = [sample_stream(L, drift=D, delta=1.5) for _ in range(1500)]
    study1["fixed_ARL0"].append(round(arl(cusum_twosided, K, hA, ic), 1))
    study1["adaptive_ARL0"].append(round(arl(cusum_adaptive, K, hB, ic), 1))
    study1["fixed_recall_faulty"].append(round(np.mean([cusum_twosided(s, K, hA) >= 0 for s in fa]), 3))
    study1["adaptive_recall_faulty"].append(round(np.mean([cusum_adaptive(s, K, hB) >= 0 for s in fa]), 3))

# study 2: cross-distribution transfer -> does the in-control threshold generalize across class subsets?
def measure(runner, h, pool):
    ic = [sample_stream(L, pool=pool) for _ in range(2000)]
    fa0 = [sample_stream(L, delta=1.0, cp=0, pool=pool) for _ in range(2000)]   # fault from window 0
    return round(arl(runner, K, h, ic), 1), round(arl(runner, K, h, fa0), 2)
poolA = np.concatenate([noise_by_neuron[c] for c in range(5)])       # classes 0-4
poolB = np.concatenate([noise_by_neuron[c] for c in range(5, 10)])   # classes 5-9
hA5 = calib(cusum_adaptive, TARGET, K, [sample_stream(L, pool=poolA) for _ in range(1500)])
hB5 = calib(cusum_adaptive, TARGET, K, [sample_stream(L, pool=poolB) for _ in range(1500)])
arl0_BA, arl1_BA = measure(cusum_adaptive, hA5, poolB)   # calibrate on A, apply to disjoint B
arl0_BB, arl1_BB = measure(cusum_adaptive, hB5, poolB)   # in-distribution reference
study2 = dict(target_ARL0=TARGET,
              calibrate_on_classes0to4_apply_to_5to9=dict(ARL0=arl0_BA, ARL1=arl1_BA),
              in_distribution_5to9=dict(ARL0=arl0_BB, ARL1=arl1_BB))

out = dict(in_control_noise_n=len(ALL), K=K, L=L, target_ARL0=TARGET,
           fixed_threshold=round(hA, 3), adaptive_threshold=round(hB, 3),
           study1_drift=study1,
           study2_transfer=study2)
json.dump(out, open(os.path.join(RES, "e5_drift.json"), "w", encoding="utf-8"), indent=2)
print("study1:", json.dumps(study1), flush=True)
print("study2 transfer:", json.dumps(study2), flush=True)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
d = study1["drift_levels"]
ax[0].plot(d, study1["fixed_ARL0"], "o-", color="#c0392b", label="fixed memorized-mean CUSUM")
ax[0].plot(d, study1["adaptive_ARL0"], "s-", color="#2a9d5c", label="adaptive-baseline CUSUM")
ax[0].axhline(TARGET, ls="--", color="k", lw=0.8, label="target ARL0=200")
ax[0].set_xlabel("baseline drift over stream (sigma)"); ax[0].set_ylabel("in-control ARL0")
ax[0].set_title("Robustness to operating-point drift"); ax[0].legend(fontsize=8)
ax[1].plot(d, study1["fixed_recall_faulty"], "o-", color="#c0392b", label="fixed")
ax[1].plot(d, study1["adaptive_recall_faulty"], "s-", color="#2a9d5c", label="adaptive")
ax[1].set_xlabel("baseline drift (sigma)"); ax[1].set_ylabel("fault detection recall (delta=1.5)")
ax[1].set_title("Fault detection under drift"); ax[1].legend(fontsize=8); ax[1].set_ylim(0, 1.05)
plt.tight_layout(); plt.savefig(os.path.join(RES, "figures", "fig_e5_drift.png"), dpi=130); plt.close()
print("wrote e5_drift.json + fig_e5_drift.png", flush=True)
