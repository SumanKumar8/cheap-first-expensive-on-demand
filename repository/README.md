# Cheap-First / Expensive-on-Demand — minimal reproducible pipeline

Runs the paper's experiments (E1 detectors, E2 conditioning, E3 cascade cost,
Stage-2 Victor-Purpura/van Rossum, fault breakdown, drift/robustness) on the real
QUANTISENC fault-injection data.

The large **`Hierarchical-Model-SNN-main`** artifacts are **not** bundled here — this
repo only *references* them by path.

## Setup
1. `pip install -r requirements.txt` (pins matter: numpy 1.23.5 / scipy 1.11.3 /
   scikit-learn 1.2.2 — the golden model is validated against Paper-1's shipped margins).
2. Have `Hierarchical-Model-SNN-main/artifacts/` locally (the folder containing
   `FashionMNIST_Experiments/`, `MNIST_Experiments/`, `SVHN_Experiments/`).

## Run
Open **`run_pipeline.ipynb`** in Jupyter, set `SNN_ARTIFACTS` in the first code cell to
your artifacts path, and run all cells. It builds the residual pools, runs every
experiment, generates tables + figures, verifies (24/24 checks), and displays the
figures inline.

Prefer the shell? Set the env var and call the scripts directly:
```
export SNN_ARTIFACTS=/path/to/Hierarchical-Model-SNN-main/artifacts
python build_pools.py fashionmnist 0 9 && python build_pools.py fashionmnist merge   # (repeat per dataset)
python real_experiments.py e13 fashionmnist isi     # E1+E3 ; also e2 <ds>
python e4_stage2_vpvr.py && python fault_breakdown.py derivable && python e5_drift.py
python make_report.py && python stats_support.py && python verify_all.py
```

## Files
- `config.py` — resolves the artifacts path (`SNN_ARTIFACTS`) and the `results/` output dir.
- `src/` — library: `detectors.py` (CUSUM/EWMA + ARL calibration), `eval_utils.py` (ROC/AUROC),
  `metrics_cost.py` (S/(1+pS) cost model), `realdata.py` (residual-stream loader).
- `build_pools.py` — raw spikes -> per-neuron residual pools (`results/real_pools_<ds>.npz`).
- `real_experiments.py` — E1/E2/E3.  `e4_stage2_vpvr.py` — measured VP/VR Stage-2 (MNIST).
- `e5_drift.py` — drift/robustness.  `fault_breakdown.py` — fault-effect breakdown (+ per-fault framework).
- `make_report.py` — tables + figures.  `stats_support.py` — CIs.  `verify_all.py` — sanity checks.

Outputs land in `results/` (pools cache, `figures/`, `tables/`, `real_summary_all.json`).
