# E3 -- Cheap-first cascade cost (real, model-free stage-1, S=15)

| dataset | metric | stage-1 recall | cascade recall | latency (win) | false-esc | flag (once) | savings (once) | flag (reset) | savings (reset) |
|---|---|---|---|---|---|---|---|---|---|
| FMNIST | isi | 1.00 | 0.95 | 2.249 | 0.40% | 0.66% | 13.7x | 12.8% | 5.1x |
| FMNIST | cv | 1.00 | 0.95 | 0.145 | 0.45% | 0.69% | 13.6x | 28.1% | 2.9x |
| MNIST | isi | 1.00 | 0.95 | 1.421 | 0.34% | 0.66% | 13.7x | 13.1% | 5.0x |
| MNIST | cv | 1.00 | 0.95 | 0.156 | 0.38% | 0.66% | 13.6x | 28.3% | 2.9x |
| SVHN | isi | 1.00 | 0.95 | 2.392 | 0.42% | 0.68% | 13.6x | 9.0% | 6.4x |
| SVHN | cv | 1.00 | 0.95 | 0.696 | 0.42% | 0.68% | 13.6x | 21.3% | 3.6x |
