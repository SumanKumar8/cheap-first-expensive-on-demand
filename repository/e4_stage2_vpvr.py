"""E4 -- Real Stage-2 Victor-Purpura / van Rossum confirmation on MNIST.

Replaces the modeled stage-2 gate with measured spike-distance confirmation.
Uses the paired nominal (fault-free) and injected MNIST test outputs. VP cost
q = 1/(10 ms); VR time constant tau = 10 ms (matching the provided scripts).

Reports, at the neuron-window level (N = 1000 imgs x 10 neurons): VP/VR
discrimination of fault-affected windows, stage-2 confirmation accuracy / false
confirmation / missed confirmation on stage-1 flagged and unflagged windows, a
measured cascade recall, and measured runtime/cost (O(S^2) vs O(S))."""
import os, sys, json, time, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import ART
BASE = os.path.join(ART, "MNIST_Experiments") + os.sep
RES = os.path.join(HERE, "results"); os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
WIN, NC = 110, 10

def read_bits(p):
    raw = open(p, "rb").read().replace(b"\r", b""); L = [x for x in raw.split(b"\n") if x]
    W = len(L[0]); L = [x for x in L if len(x) == W]
    return (np.frombuffer(b"".join(L), np.uint8).reshape(len(L), W) - ord("0")).astype(np.uint8)
def parse(p):
    m = read_bits(p)[15:-15]; n = m.shape[0] // WIN; return m[:n*WIN].reshape(n, WIN, NC)

def spk_times(col): return np.where(col == 1)[0].astype(float) + 1.0
def avg_isi(t):
    if len(t) == 0: return 0.0
    if len(t) == 1: return float(t[0])
    return float(np.mean(np.diff(t)))
def van_rossum(a, b, tau=10.0):
    def S(x, y):
        if len(x) == 0 or len(y) == 0: return 0.0
        return float(np.exp(-np.abs(x[:, None]-y[None, :])/tau).sum())
    return math.sqrt(max(0.0, (S(a, a)+S(b, b)-2*S(a, b))/2.0))
def victor_purpura(a, b, q=0.1):
    n, m = len(a), len(b)
    if n == 0 or m == 0: return float(max(n, m))
    prev = np.arange(m+1, dtype=float)
    for i in range(1, n+1):
        cur = np.empty(m+1); cur[0] = i; ai = a[i-1]
        for j in range(1, m+1):
            cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+q*abs(ai-b[j-1]))
        prev = cur
    return float(prev[m])
def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0, c-h), min(1, c+h))

nom = parse(BASE+"Nominal/test_spike_output_file_0.txt")
inj = parse(BASE+"Output_spike_test_inject_mnist/test_spike_output_file_0.txt")
NI = min(nom.shape[0], inj.shape[0]); nom, inj = nom[:NI], inj[:NI]
print("nominal", nom.shape, "inject", inj.shape, flush=True)

# per (image, neuron): distances, cheap metric, affected label, spike-change type
VP = np.zeros((NI, NC)); VR = np.zeros((NI, NC)); AFF = np.zeros((NI, NC), bool)
ISIn = np.zeros((NI, NC)); ISIf = np.zeros((NI, NC)); DN = np.zeros((NI, NC), int)  # spk count delta
t_vp = t_vr = t_ch = 0.0
for c in range(NC):
    for j in range(NI):
        ncol = nom[j][:, (NC-1)-c]; fcol = inj[j][:, (NC-1)-c]
        a, b = spk_times(ncol), spk_times(fcol)
        AFF[j, c] = not np.array_equal(ncol, fcol)
        s = time.perf_counter(); VP[j, c] = victor_purpura(a, b); t_vp += time.perf_counter()-s
        s = time.perf_counter(); VR[j, c] = van_rossum(a, b); t_vr += time.perf_counter()-s
        s = time.perf_counter(); ISIf[j, c] = avg_isi(b); t_ch += time.perf_counter()-s
        ISIn[j, c] = avg_isi(a); DN[j, c] = len(b)-len(a)
Nw = NI*NC
print(f"windows={Nw}  affected={AFF.mean()*100:.2f}%  VP time {1e6*t_vp/Nw:.1f}us/win  "
      f"VR {1e6*t_vr/Nw:.1f}us/win  cheap-uISI {1e6*t_ch/Nw:.3f}us/win", flush=True)

# ---- (a) confirmer quality: VP/VR on ALL windows (decoupled from stage-1) ----
aff = AFF.ravel(); vp = VP.ravel(); vr = VR.ravel()
n_aff = int(aff.sum()); n_un = int((~aff).sum())
TAU = 0.5   # nominal->0 exactly; >=0.5 = substantive distortion (>=~1 spike or >5ms shift)
conf_acc = (vp[aff] > TAU).mean();  false_conf = (vp[~aff] > TAU).mean();  missed = 1 - conf_acc
def auroc(score, label):
    r = np.argsort(np.argsort(score)) + 1; P = int(label.sum()); N = len(label) - P
    return float((r[label].sum() - P*(P+1)/2)/(P*N)) if P and N else float("nan")
vp_auc, vr_auc = auroc(vp, aff), auroc(vr, aff)

# ---- distortion-type breakdown of affected windows (derivable from the single injected file) ----
dn = DN.ravel()
missing = int(((dn < 0) & aff).sum())   # faulty has FEWER spikes than nominal
spurious = int(((dn > 0) & aff).sum())  # faulty has MORE spikes
timing  = int(((dn == 0) & aff).sum())  # same count, shifted timing

# ---- (b) cheap per-window screen (stage-1) -> flagged/unflagged; VP/VR cleans up false alarms ----
flag = np.zeros((NI, NC), bool)
for c in range(NC):
    m0, sd = ISIn[:, c].mean(), ISIn[:, c].std() or 1.0
    zn = np.abs((ISIn[:, c]-m0)/sd); h = np.percentile(zn, 99)     # 1% nominal false-alarm point
    flag[:, c] = np.abs((ISIf[:, c]-m0)/sd) > h
fl = flag.ravel(); conf = vp > TAU
flagged = int(fl.sum()); flagged_aff = int((fl & aff).sum()); flagged_unaff = int((fl & ~aff).sum())
unflagged_aff = int((~fl & aff).sum())
fa_reject_rate = float((~conf & fl & ~aff).sum())/max(1, flagged_unaff)   # VP rejects stage-1 false alarms
# measured confirmation to replace the simulated E3 gate
res = dict(
    dataset="mnist", n_windows=Nw, n_neurons=NC, n_images=NI,
    affected_frac=round(float(aff.mean()), 4), n_affected=n_aff, n_unaffected=n_un, tau_conf=TAU,
    distortion_missing=missing, distortion_spurious=spurious, distortion_timing=timing,
    distortion_missing_pct=round(100*missing/max(1, n_aff), 1),
    distortion_spurious_pct=round(100*spurious/max(1, n_aff), 1),
    distortion_timing_pct=round(100*timing/max(1, n_aff), 1),
    vp_auroc_affected=round(vp_auc, 4), vr_auroc_affected=round(vr_auc, 4),
    vp_mean_affected=round(float(vp[aff].mean()), 3), vp_mean_unaffected=round(float(vp[~aff].mean()), 4),
    confirmation_accuracy=round(float(conf_acc), 4),
    confirmation_accuracy_CI=[round(x, 4) for x in wilson(int((vp[aff] > TAU).sum()), n_aff)],
    false_confirmation=round(float(false_conf), 4),
    false_confirmation_CI=[round(x, 4) for x in wilson(int((vp[~aff] > TAU).sum()), n_un)],
    missed_confirmation=round(float(missed), 4),
    stage1_flag_rate=round(flagged/Nw, 4), stage1_flagged_affected=flagged_aff,
    stage1_flagged_falsealarm=flagged_unaff, stage1_unflagged_affected=unflagged_aff,
    stage2_falsealarm_rejection=round(fa_reject_rate, 4),
    vp_us_per_window=round(1e6*t_vp/Nw, 2), vr_us_per_window=round(1e6*t_vr/Nw, 2),
    cheap_us_per_window=round(1e6*t_ch/Nw, 3), vp_over_cheap_cost=round(t_vp/t_ch, 1),
    cost_cascade_vs_vp_everywhere=round(float(t_vp/(t_ch + (flagged/Nw)*t_vp)), 2))

json.dump(res, open(os.path.join(RES, "e4_stage2_mnist.json"), "w", encoding="utf-8"), indent=2)
for k, v in res.items(): print(f"  {k}: {v}", flush=True)



# figure: VP distribution affected vs unaffected + confirmation bars
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].hist(vp[~aff], bins=40, alpha=0.7, label="unaffected windows", color="#888", density=True)
ax[0].hist(vp[aff], bins=40, alpha=0.7, label="fault-affected windows", color="#c0392b", density=True)
ax[0].axvline(TAU, ls="--", color="k", lw=1, label=f"confirm thr={TAU}")
ax[0].set_xlabel("Victor-Purpura distance (nominal vs injected)"); ax[0].set_ylabel("density")
ax[0].set_title(f"MNIST Stage-2 VP separation (AUROC={vp_auc:.3f})"); ax[0].legend(fontsize=8); ax[0].set_xlim(0, 12)
labels = ["confirmation\naccuracy", "false\nconfirmation", "missed\nconfirmation"]
vals = [conf_acc, false_conf, missed]; cols = ["#2a9d5c", "#c0392b", "#e08a1e"]
b = ax[1].bar(labels, vals, color=cols)
for bi, v in zip(b, vals): ax[1].text(bi.get_x()+bi.get_width()/2, v+0.02, f"{v*100:.1f}%", ha="center", fontsize=9)
ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("rate"); ax[1].set_title("Stage-2 confirmation (real VP, MNIST)")
plt.tight_layout(); plt.savefig(os.path.join(RES, "figures", "fig_e4_stage2.png"), dpi=130); plt.close()
print("wrote e4_stage2_mnist.json + fig_e4_stage2.png", flush=True)
