"""Fault-effect breakdown + per-fault analysis framework.

Two modes:
  python3 fault_breakdown.py derivable        # what IS derivable from the single
      injected MNIST file: per-output-neuron effects, missing/spurious/timing
      split, distortion magnitude (VP/VR). Writes results/fault_breakdown_mnist.json + figure.
  python3 fault_breakdown.py perfault <root>  # FS/bit/type breakdown when the
      per-fault injected folders are provided (structure below). Ready for the
      vmem/qmul x bit x S@0/S@1 data your VP/VR scripts use.

Per-fault folder structure expected by 'perfault' (matches the uploaded scripts):
  <root>/nominal/*.txt                      # fault-free reference spike file(s)
  <root>/<site>/<bit>/*.txt                 # site in {vmem,qmul,adder}, bit in 0..7
  fault type (S@0/S@1/bitflip) encoded in the path or filename.
Each spike file: one 0/1 per line (per timestep). VP q=1/10ms, VR tau=10ms.
"""
import os, sys, json, glob, re
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
def van_rossum(a, b, tau=10.0):
    def S(x, y):
        if len(x) == 0 or len(y) == 0: return 0.0
        return float(np.exp(-np.abs(x[:, None]-y[None, :])/tau).sum())
    return float(np.sqrt(max(0.0, (S(a, a)+S(b, b)-2*S(a, b))/2.0)))
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

def derivable():
    nom = parse(BASE+"Nominal/test_spike_output_file_0.txt")
    inj = parse(BASE+"Output_spike_test_inject_mnist/test_spike_output_file_0.txt")
    NI = min(nom.shape[0], inj.shape[0]); nom, inj = nom[:NI], inj[:NI]
    per_neuron = {}
    for c in range(NC):
        aff = miss = spur = tim = 0; vps = []
        for j in range(NI):
            ncol = nom[j][:, (NC-1)-c]; fcol = inj[j][:, (NC-1)-c]
            if np.array_equal(ncol, fcol): continue
            aff += 1; d = int(fcol.sum()) - int(ncol.sum())
            miss += d < 0; spur += d > 0; tim += d == 0
            vps.append(victor_purpura(spk_times(ncol), spk_times(fcol)))
        per_neuron[c] = dict(affected=aff, affected_pct=round(100*aff/NI, 1),
                             missing=miss, spurious=spur, timing=tim,
                             mean_VP=round(float(np.mean(vps)), 3) if vps else 0.0)
    tot_aff = sum(v["affected"] for v in per_neuron.values())
    out = dict(mode="derivable_single_injected_config", dataset="mnist", n_images=NI, n_neurons=NC,
               total_affected_windows=tot_aff,
               missing_pct=round(100*sum(v["missing"] for v in per_neuron.values())/max(1, tot_aff), 1),
               spurious_pct=round(100*sum(v["spurious"] for v in per_neuron.values())/max(1, tot_aff), 1),
               timing_pct=round(100*sum(v["timing"] for v in per_neuron.values())/max(1, tot_aff), 1),
               per_neuron=per_neuron,
               note="FS1/FS2/FS3, stuck-at type and bit position are injection parameters not recoverable "
                    "from a single aggregate output file; run 'perfault' mode on the per-fault dataset for that breakdown.")
    json.dump(out, open(os.path.join(RES, "fault_breakdown_mnist.json"), "w", encoding="utf-8"), indent=2)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    cs = list(range(NC)); ax[0].bar(cs, [per_neuron[c]["affected_pct"] for c in cs], color="#4472c4")
    ax[0].set_xlabel("output neuron (class)"); ax[0].set_ylabel("% test windows fault-affected")
    ax[0].set_title("MNIST: per-neuron fault manifestation"); ax[0].set_xticks(cs)
    typ = [out["missing_pct"], out["spurious_pct"], out["timing_pct"]]
    ax[1].bar(["missing", "spurious", "timing-only"], typ, color=["#c0392b", "#e08a1e", "#2a9d5c"])
    for i, v in enumerate(typ): ax[1].text(i, v+1, f"{v}%", ha="center")
    ax[1].set_ylabel("% of affected windows"); ax[1].set_title("Distortion type (single injected config)")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "figures", "fig_fault_breakdown.png"), dpi=130); plt.close()
    print("total affected windows:", tot_aff, "| missing/spurious/timing %:",
          out["missing_pct"], out["spurious_pct"], out["timing_pct"])
    print("wrote fault_breakdown_mnist.json + fig_fault_breakdown.png")

def _read_train(path):  # single fault/nominal spike file -> spike-time list
    logs = [int(x.strip()) for x in open(path).read().replace("\r", "").split("\n") if x.strip() in ("0", "1")]
    return np.array([i+1 for i, v in enumerate(logs) if v == 1], float)

def perfault(root):
    """Compute VP/VR breakdown per (fault-site, bit, type). Expects the structure in the header."""
    nomfiles = glob.glob(os.path.join(root, "nominal", "*.txt")) + glob.glob(os.path.join(root, "Nominal", "*.txt"))
    if not nomfiles:
        print("No per-fault dataset found at", root, "\nProvide nominal/ + <site>/<bit>/ folders (see header)."); return
    nominal = _read_train(sorted(nomfiles)[0])
    rows = []
    for site in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d.lower() not in ("nominal",)):
        for bitdir in sorted(glob.glob(os.path.join(root, site, "*"))):
            if not os.path.isdir(bitdir): continue
            bit = re.findall(r"\d+", os.path.basename(bitdir))
            bit = int(bit[0]) if bit else -1
            for fp in sorted(glob.glob(os.path.join(bitdir, "*.txt"))):
                ft = "S@0" if "s@0" in fp.lower() or "s0" in os.path.basename(fp).lower() else \
                     ("S@1" if "s@1" in fp.lower() or "s1" in os.path.basename(fp).lower() else "bitflip")
                fault = _read_train(fp)
                rows.append(dict(site=site, bit=bit, type=ft,
                                 VP=round(victor_purpura(nominal, fault), 3),
                                 VR=round(van_rossum(nominal, fault), 3),
                                 bitclass="high" if bit >= 4 else "low"))
    out = dict(mode="perfault", root=root, n=len(rows), rows=rows)
    # aggregate by site/type/bitclass
    def agg(key):
        g = {}
        for r in rows: g.setdefault(r[key], []).append(r["VP"])
        return {k: dict(mean_VP=round(float(np.mean(v)), 3), n=len(v)) for k, v in g.items()}
    out["by_site"] = agg("site"); out["by_type"] = agg("type"); out["by_bitclass"] = agg("bitclass")
    json.dump(out, open(os.path.join(RES, "fault_breakdown_perfault.json"), "w", encoding="utf-8"), indent=2)
    print("per-fault rows:", len(rows), "| by site:", out["by_site"], "| by type:", out["by_type"],
          "| low vs high bit:", out["by_bitclass"])

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "derivable"
    if mode == "derivable": derivable()
    elif mode == "perfault": perfault(sys.argv[2])
