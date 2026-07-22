# Cheap First, Expensive on Demand

Code and cached results for **"Cheap First, Expensive on Demand: Model-Free
Screening and On-Demand Spike-Distance Confirmation for Concurrent Testing of Spiking
Neuromorphic Hardware."**

It builds an adaptive metric-escalation layer on top of the hierarchical model-based
error-detection-and-isolation (EDI) scheme of Kumar *et al.* (DSN 2025) on the
open-source QUANTISENC LIF core.

## Key results (all reproduced by the code in this repo)

| Result | Measured |
|---|---|
| Golden-model fidelity, muISI (FMNIST / MNIST / SVHN) | 4.6 / 4.0 / 5.4 % |
| Stage-1 CUSUM-vs-per-window speedup, muISI | 7.1 / 5.7 / 2.05x |
| Conditioning gain (ML - model-free) | +0.000 / -0.001 AU-ROC |
| Cascade savings (escalate-once) | 13.6x |
| Cascade recall (stage-1 / end-to-end) | 1.00 / 0.95 |
| Stage-2 VP/VR confirmation (MNIST / FMNIST / SVHN) | 96.6 / 99.3 / 99.2 % |
| Stage-2 false-alarm rejection | 100 % |

MNIST Stage-2 is measured on real paired nominal/injected outputs; FashionMNIST and SVHN
are **MNIST-derived** projections (MNIST's measured fault model applied to each dataset's
real injected outputs at its own error rate). See the Stage-2 note in
`repository/run_stage2_datasets.py`.

## Reproduce

```bash
cd repository
pip install -r requirements.txt          # numpy, scipy, scikit-learn, matplotlib
python run_pipeline.py                    # runs E1-E5 + report end-to-end
```

Or run the stages individually (all read the cached residual pools in `repository/results/`):

```bash
cd repository
for ds in mnist fashionmnist svhn; do
  for m in isi cv; do python real_experiments.py e13 $ds $m; done   # E1 detectors + E3 cascade
  python real_experiments.py e2 $ds                                  # E2 conditioning
done
python run_stage2_datasets.py     # Stage-2 VP/VR (MNIST + MNIST-derived FMNIST/SVHN)
python e5_drift.py                # drift / cross-distribution robustness
python fault_breakdown.py derivable
python stats_support.py
python make_report.py             # aggregates -> results/tables/*.md + results/figures/*.png
```

Outputs land in `repository/results/` (`e*_*.json`, `tables/*.md`, `figures/*.png`).

## Layout

```
repository/
  src/                 detectors.py, eval_utils.py, metrics_cost.py, realdata.py
  build_pools.py       (re-)derive per-neuron residual pools from raw artifacts
  real_experiments.py  E1 detector comparison, E2 conditioning, E3 cascade cost
  run_stage2_datasets.py  E4 Victor-Purpura / van Rossum stage-2 confirmation
  e5_drift.py          E5 drift & cross-distribution robustness
  fault_breakdown.py   distortion-type breakdown of the injected faults
  stats_support.py     Wilson CIs + rule-of-three
  make_report.py       aggregate -> tables + figures
  results/             cached residual pools (real_pools_*.npz) + all result JSONs/tables/figures
Hierarchical-Model-SNN-main/artifacts/<DS>_Experiments/
  Nominal/                        fault-free output (MNIST real; FMNIST/SVHN MNIST-derived)
  Output_spike_test_inject_*/     released injected (faulty) test outputs
  epsilon_thrs.npz                shipped per-neuron margins from the prior work (MNIST)
```

## Data

The heavy raw QUANTISENC artifacts (per-class `Data/`, ~37 GB) are needed **only** to
re-derive the residual pools from scratch (`build_pools.py`); they come from the prior
work's release (Hierarchical-Model-SNN, DSN 2025). This repo ships the **cached residual
pools** (`repository/results/real_pools_*.npz`) plus the small nominal/injected spike
files, so every table and figure reproduces **without** the raw data.

## Hardware-level simulation (QUANTISENC RTL)

The `quantisenc-main_neuronid/` folder holds the QUANTISENC neuromorphic-hardware
sources. The tables and figures above reproduce from the cached residual pools; to
instead run the neuron model **at the hardware / RTL level** and generate the spike
outputs yourself, run the files in that folder.

## Citation

If you use this code, please cite this repository (see `CITATION.cff`) and the underlying
work:
S. Kumar, A. Mishra, A. Das, N. Kandasamy, "Hierarchical Model-Based Approach for
Concurrent Testing of Neuromorphic Architecture," *Proc. DSN*, 2025, pp. 511-523.
