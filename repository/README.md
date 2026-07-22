# Cheap-First / Expensive-on-Demand -- reproducible pipeline

Runs the escalation experiments on the QUANTISENC fault-injection data: E1 detector
comparison, E2 conditioning, E3 cascade cost, Stage-2 Victor-Purpura / van Rossum
confirmation, fault-effect breakdown, and drift / cross-distribution robustness.

## Reproduce from the shipped cache (no raw data needed)

The cached residual pools are already in `results/` (`real_pools_*.npz`), so every
table and figure regenerates without the ~37 GB of raw artifacts:

```bash
pip install -r requirements.txt      # numpy 1.23.5 / scipy 1.11.3 / scikit-learn 1.2.2 / matplotlib
for ds in mnist fashionmnist svhn; do
  for m in isi cv; do python real_experiments.py e13 $ds $m; done   # E1 detectors + E3 cascade
  python real_experiments.py e2 $ds                                  # E2 conditioning
done
python run_stage2_datasets.py    # Stage-2 VP/VR: MNIST measured; FMNIST/SVHN MNIST-derived
python e5_drift.py               # drift / cross-distribution robustness
python fault_breakdown.py derivable
python stats_support.py
python make_report.py            # -> results/tables/*.md + results/figures/*.png
```

Outputs land in `results/`.

## Rebuild the pools from raw data (optional)

Only needed to re-derive `results/real_pools_*.npz` from scratch. Set
`SNN_ARTIFACTS` to the prior work's `Hierarchical-Model-SNN-main/artifacts/` release
(DSN 2025) and run `python build_pools.py <dataset> 0 9 && python build_pools.py
<dataset> merge` per dataset. The small Nominal/injected spike files that Stage-2 needs
are already bundled under `../Hierarchical-Model-SNN-main/artifacts/`.

## Files

- `config.py` -- resolves the artifacts path (`SNN_ARTIFACTS`) and the `results/` dir.
- `src/` -- `detectors.py` (CUSUM/EWMA + ARL calibration), `eval_utils.py` (AUROC),
  `metrics_cost.py` (S/(1+pS) cost model), `realdata.py` (residual-stream loader).
- `real_experiments.py` -- E1 / E2 / E3.
- `run_stage2_datasets.py` -- Stage-2 VP/VR for all three datasets.
- `e4_stage2_vpvr.py` -- MNIST-only Stage-2 (full-schema, used by `stats_support.py`).
- `e5_drift.py`, `fault_breakdown.py`, `stats_support.py`, `make_report.py`.
- `build_pools.py` -- raw spikes -> per-neuron residual pools.
- `results/` -- cached pools, result JSONs, `tables/`, `figures/`.
