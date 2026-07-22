"""Generalized residual-pool builder for the 3 Paper-1 datasets.

Usage:
  python3 build_pools.py <dataset> <c0> <c1>   # process classes c0..c1
  python3 build_pools.py <dataset> merge       # combine -> results/real_pools_<dataset>.npz
dataset in {fashionmnist, mnist, svhn}.

Raw (model-free) residual is built for every dataset from output spikes only.
ml/raw/lin are all built when test-input spikes exist (fashionmnist, mnist);
svhn has no test inputs, so only the raw residual is produced.
"""
import os, glob, re, time, json, sys
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

REPO = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ART
PAD, T, NC = 10, 100, 10
WIN = PAD + T
N_TRAIN = 2500

DATASETS = {
    "fashionmnist": dict(folder="FashionMNIST_Experiments", prefix="Fashion_mnist",
                         inject="Output_spike_test_inject_Fashion_mnist",
                         test_input="SpikeGenData_test_Fashion_mnist"),
    "mnist": dict(folder="MNIST_Experiments", prefix="mnist",
                  inject="Output_spike_test_inject_mnist",
                  test_input="SpikeGenData_test_mnist"),
    "svhn": dict(folder="SVHN_Experiments", prefix="mnist",
                 inject="Output_spike_test_inject_svhn", test_input="SpikeGenData_test_SVHN"),
}
def log(*a): print(*a, flush=True)

def getAvgIsi(x):
    if len(x) == 0: return 0
    if len(x) == 1: return x[0]
    x = list(x)
    if x[0] == 0: x[0] = 1
    return sum(x[i+1]-x[i] for i in range(len(x)-1))/(len(x)-1)
def coefficientOfVariation(x):
    if len(x) == 0: return 0
    if len(x) == 1: return x[0]
    x = list(x)
    if x[0] == 0: x[0] = 1
    imd = np.array(x, float); mu = imd.mean()
    return np.sqrt(np.sum((imd-mu)**2)/(len(imd)-1))/mu
def out_metric(spk, kind):
    pos = [i+1 for i, s in enumerate(spk) if s == 1]
    return getAvgIsi(pos) if kind == "isi" else coefficientOfVariation(pos)
def iqr_keep(d, f=1.5):
    Q1, Q3 = np.percentile(d, 25), np.percentile(d, 75); IQR = Q3-Q1
    return (d >= Q1-f*IQR) & (d <= Q3+f*IQR)

def read_bits(path):
    with open(path, "rb") as fh:
        raw = fh.read().replace(b"\r", b"")
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"": lines = lines[:-1]
    W = len(lines[0]); lines = [ln for ln in lines if len(ln) == W]
    return (np.frombuffer(b"".join(lines), np.uint8).reshape(len(lines), W) - ord("0")).astype(np.uint8)
def _natkey(p): return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', p)]
def parse_output_dir(folder):
    files = sorted(glob.glob(folder+"/*.txt"), key=_natkey)
    ch = []
    for f in files:
        m = read_bits(f)[15:-15]; n = m.shape[0]//WIN
        ch.append(m[:n*WIN].reshape(n, WIN, NC))
    return np.concatenate(ch, 0)
def parse_inputs(folder, max_n=None):
    files = sorted(glob.glob(folder+"/*.txt"), key=_natkey)
    ch = []; tot = 0
    for f in files:
        m = read_bits(f)[:-10]; n = m.shape[0]//WIN
        ch.append(m[:n*WIN].reshape(n, WIN, -1)); tot += n
        if max_n and tot >= max_n: break
    return np.concatenate(ch, 0)

def _feat_chunk(A):
    A = A.astype(np.float32)
    col = np.arange(1, A.shape[-1]+1, dtype=np.float32)
    nspk = A.sum(2); sum_pos = (A*col).sum(2); sum_pos2 = (A*col*col).sum(2)
    max_pos = (A*col).max(2); min_pos = np.where(A == 1, col, col[-1]+1).min(2)
    z = np.zeros_like(nspk)
    mean_pos = np.divide(sum_pos, nspk, out=z.copy(), where=nspk > 0)
    m1, m2 = nspk == 1, nspk >= 2
    avg = z.copy(); avg[m2] = (max_pos[m2]-min_pos[m2])/(nspk[m2]-1); avg[m1] = mean_pos[m1]
    var = z.copy(); var[m2] = (sum_pos2[m2]-nspk[m2]*mean_pos[m2]**2)/(nspk[m2]-1)
    cv = z.copy(); cv[m2] = np.sqrt(np.clip(var[m2], 0, None))/mean_pos[m2]; cv[m1] = mean_pos[m1]
    counts = np.stack([A[:, :, b:b+64].sum(2) for b in range(0, A.shape[-1], 64)], -1)
    cmean, cvar = counts.mean(-1), counts.var(-1)
    fano = np.divide(cvar, cmean, out=z.copy(), where=cmean > 0)
    for a in (avg, cv, fano): a[nspk == 0] = 0
    return np.nan_to_num(np.stack([avg, cv, np.zeros_like(avg), fano, nspk], -1).reshape(A.shape[0], -1))
def features(arr, chunk=250):
    return np.concatenate([_feat_chunk(arr[i:i+chunk]) for i in range(0, len(arr), chunk)], 0)

def paths(dsname):
    cfg = DATASETS[dsname]; base = ART + cfg["folder"] + "/"
    cache = REPO + "results/_cache/" + dsname + "/"; os.makedirs(cache, exist_ok=True)
    return cfg, base, cache

def get_test(dsname):
    cfg, base, cache = paths(dsname)
    inj = parse_output_dir(base + cfg["inject"])
    if cfg["test_input"] is None:
        return None, inj
    ftx = cache + "test_feats.npy"
    if os.path.exists(ftx):
        Xte = np.load(ftx)
    else:
        tin = parse_inputs(base + cfg["test_input"], max_n=inj.shape[0])
        Ni = min(inj.shape[0], tin.shape[0]); tin = tin[:Ni]
        Xte = features(tin); np.save(ftx, Xte)
    Ni = min(inj.shape[0], Xte.shape[0])
    return Xte[:Ni], inj[:Ni]

def process_class(dsname, c, Xte, inj):
    cfg, base, cache = paths(dsname)
    Ni = inj.shape[0]
    tgt = np.load(base + f"Correct_output_spk_per_cls/target_{cfg['prefix']}_cls_{c}.npy")
    rng = np.random.default_rng(c)
    idx = rng.choice(tgt.shape[0], min(N_TRAIN, tgt.shape[0]), replace=False)
    tgt_s = tgt[idx]
    feats = None
    if Xte is not None:
        dat = np.load(base + f"Data/data_{cfg['prefix']}_cls_{c}.npy")[idx]
        feats = features(dat)
    obs_full = inj[:, :, (NC-1)-c]
    store, summ = {}, {}
    for kind in ("isi", "cv"):
        y = np.array([out_metric(o, kind) for o in tgt_s])
        nz = np.where(y != 0)[0]
        keep = nz[iqr_keep(y[nz], 1.5)]
        tr, ho = train_test_split(keep, test_size=0.45, random_state=42)
        m0 = float(np.mean(y[tr]))
        obs_f = np.array([out_metric(obs_full[j], kind) for j in range(Ni)])
        pools = {"raw": (y[ho]-m0, obs_f-m0)}
        mae_ml = None
        if feats is not None:
            sc = RobustScaler().fit(feats[tr])
            Ztr = np.nan_to_num(sc.transform(feats[tr])); Zho = np.nan_to_num(sc.transform(feats[ho]))
            Zte = np.nan_to_num(sc.transform(Xte))
            rf = RandomForestRegressor(n_estimators=250, max_features='sqrt', max_depth=6,
                                       min_samples_split=2, random_state=42, n_jobs=-1).fit(Ztr, y[tr])
            lin = LinearRegression().fit(Ztr, y[tr])
            mae_ml = float(mean_absolute_error(y[ho], rf.predict(Zho)))
            pools["ml"] = (y[ho]-rf.predict(Zho), obs_f-rf.predict(Zte))
            pools["lin"] = (y[ho]-lin.predict(Zho), obs_f-lin.predict(Zte))
        rec = {"mae_ml": mae_ml, "n_h": int(len(ho)), "n_f": int(Ni), "has_ml": feats is not None}
        for name in pools:
            h, f = pools[name]
            sd = float(np.std(h)) or 1.0
            store[f"{kind}_{c}_{name}_h"] = (h/sd).astype(np.float32)
            store[f"{kind}_{c}_{name}_f"] = (f/sd).astype(np.float32)
            rec[name] = dict(sd=sd, h_mean=float(np.mean(h/sd)),
                             f_mean=float(np.mean(f/sd)), f_std=float(np.std(f/sd)))
        summ[f"{kind}_{c}"] = rec
        extra = f"MAE {mae_ml:.4f} " if mae_ml is not None else "raw-only "
        log(f"  {dsname} cls{c} {kind}: {extra}raw_delta {rec['raw']['f_mean']-rec['raw']['h_mean']:+.2f}s")
    np.savez_compressed(cache + f"pool_{c}.npz", **store)
    json.dump(summ, open(cache + f"pool_{c}.json", "w"))

def merge(dsname):
    cfg, base, cache = paths(dsname)
    store, summ = {}, {}
    for c in range(NC):
        d = np.load(cache + f"pool_{c}.npz")
        for k in d.files: store[k] = d[k]
        summ.update(json.load(open(cache + f"pool_{c}.json")))
    out = REPO + f"results/real_pools_{dsname}.npz"
    np.savez_compressed(out, **store)
    json.dump(summ, open(out.replace(".npz", ".summary.json"), "w"), indent=1)
    log(f"MERGED {dsname} -> {out} ({len(store)} arrays)")

if __name__ == "__main__":
    dsname = sys.argv[1]; t0 = time.time()
    if sys.argv[2] == "merge":
        merge(dsname)
    else:
        c0, c1 = int(sys.argv[2]), int(sys.argv[3])
        Xte, inj = get_test(dsname)
        log(f"{dsname}: test={None if Xte is None else Xte.shape} inj {inj.shape} ({round(time.time()-t0)}s)")
        for c in range(c0, c1+1):
            process_class(dsname, c, Xte, inj)
            log(f"  class {c} done ({round(time.time()-t0)}s)")
