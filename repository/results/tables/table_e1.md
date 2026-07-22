# E1 -- Stage-1 detector comparison (real, matched ARL0=200)

| dataset | metric | per-window AUROC | per-window lat | CUSUM AUROC | CUSUM lat | latency speedup | cond. gain (ML-raw) |
|---|---|---|---|---|---|---|---|
| FMNIST | isi | 0.928 | 24.2 | 1.000 | 3.4 | 7.06x | +0.000 |
| FMNIST | cv | 1.000 | 1.1 | 1.000 | 1.1 | 0.98x | +0.000 |
| MNIST | isi | 0.959 | 14.2 | 1.000 | 2.5 | 5.66x | -0.000 |
| MNIST | cv | 1.000 | 1.1 | 1.000 | 1.1 | 0.98x | +0.000 |
| SVHN | isi | 0.986 | 7.6 | 1.000 | 3.7 | 2.05x | +0.000 |
| SVHN | cv | 1.000 | 1.6 | 1.000 | 1.9 | 0.82x | +0.000 |
