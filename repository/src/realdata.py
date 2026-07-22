"""realdata.py -- Real Paper-1 fault-injection residual streams (multi-dataset).

Drop-in replacement for synth.py. Serves standardized residual streams derived
from Paper-1's QUANTISENC fault-injection data for fashionmnist / mnist / svhn.
Select the dataset with set_dataset(name). See docs/REAL_DATA.md and build_pools.py.

Residual = observed cheap metric (muISI/CV) - golden-model prediction, centered
on the healthy mean and standardized by the healthy std, so in-control ~ N(0,1)
and a fault adds a persistent (signed) shift. Detectors run TWO-SIDED. Residual
types: 'raw' (per-neuron mean baseline, model-free; available for ALL datasets),
'ml' (RF), 'lin' (linear) -- ml/lin need test-input spikes (fashionmnist, mnist).
"""
from __future__ import annotations
import os, json
import numpy as np
from dataclasses import dataclass

@dataclass
class NeuronDataset:
    U: np.ndarray; M: np.ndarray; fault: np.ndarray; beta: np.ndarray
    m0: np.ndarray; delta: np.ndarray; sigma_e: float; change_point: np.ndarray

_HERE = os.path.dirname(__file__)
N_NEURONS = 10
T_WINDOW = 100
DATASET = "fashionmnist"
_cache = {}

def set_dataset(name):
    global DATASET
    DATASET = name
    return name

def _load(ds=None):
    ds = ds or DATASET
    if ds not in _cache:
        npz = os.path.join(_HERE, "..", "results", f"real_pools_{ds}.npz")
        d = np.load(npz)
        store = {k: d[k] for k in d.files}
        summ = json.load(open(npz.replace(".npz", ".summary.json")))
        _cache[ds] = (store, summ)
    return _cache[ds]

def available_residuals(ds=None):
    store, _ = _load(ds)
    res = []
    for r in ("raw", "ml", "lin"):
        if f"isi_0_{r}_h" in store:
            res.append(r)
    return res

def pool(metric, c, residual, which, ds=None):
    store, summ = _load(ds)
    hm = summ[f"{metric}_{c}"][residual]["h_mean"]
    return store[f"{metric}_{c}_{residual}_{which}"] - hm

def neuron_shift(metric, c, residual, ds=None):
    _, summ = _load(ds)
    r = summ[f"{metric}_{c}"][residual]
    return r["f_mean"] - r["h_mean"]

def sample_streams(metric, residual, kind, n, L, cp=0, neurons=range(N_NEURONS), seed=0, ds=None):
    rng = np.random.default_rng(seed)
    neurons = list(neurons)
    X = np.empty((n, L), np.float32)
    for i in range(n):
        c = neurons[i % len(neurons)]
        H = pool(metric, c, residual, "h", ds); F = pool(metric, c, residual, "f", ds)
        if kind == "healthy":
            X[i] = rng.choice(H, L)
        elif kind == "faulty0":
            X[i] = rng.choice(F, L)
        elif kind == "faulty":
            if cp > 0: X[i, :cp] = rng.choice(H, cp)
            X[i, cp:] = rng.choice(F, L - cp)
        else:
            raise ValueError(kind)
    return X

def neuron_dataset_real(metric="isi", residual="raw", n_per_neuron=80, L=100,
                        fault_fraction=0.5, neurons=range(N_NEURONS), seed=0, ds=None):
    rng = np.random.default_rng(seed)
    neurons = list(neurons)
    N = n_per_neuron * len(neurons)
    M = np.empty((N, L), np.float32); fault = np.zeros((N, L), bool)
    change_point = np.full(N, -1); delta = np.zeros(N)
    for i in range(N):
        c = neurons[i % len(neurons)]
        H = pool(metric, c, residual, "h", ds); F = pool(metric, c, residual, "f", ds)
        if rng.random() < fault_fraction:
            cp = int(rng.integers(L // 4, L // 2))
            M[i, :cp] = rng.choice(H, cp); M[i, cp:] = rng.choice(F, L - cp)
            fault[i, cp:] = True; change_point[i] = cp
            delta[i] = neuron_shift(metric, c, residual, ds)
        else:
            M[i] = rng.choice(H, L)
    return NeuronDataset(U=np.zeros((N, L)), M=M, fault=fault, beta=np.zeros(N),
                         m0=np.zeros(N), delta=delta, sigma_e=1.0, change_point=change_point)

if __name__ == "__main__":
    for ds in ("fashionmnist", "mnist", "svhn"):
        set_dataset(ds)
        print(ds, "residuals:", available_residuals(), "| isi_0 mae:",
              _load()[1]["isi_0"]["mae_ml"])
