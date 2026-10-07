# Phase 2: squash a table onto its top k directions, measure loss (Pythia)

WikiText-103 test, first 16 x 1024 tokens. k* = smallest k within 0.01 bits/byte of the unmodified model.

| Model | d | layers | baseline bits/byte | input k* | input k*/d | output k* | output k*/d |
|---|---|---|---|---|---|---|---|
| pythia-70m | 512 | 6 | 1.3449 | 512 | 1.00 | 512 | 1.00 |
| pythia-160m | 768 | 12 | 1.1519 | 768 | 1.00 | 768 | 1.00 |
| pythia-410m | 1024 | 24 | 0.9731 | 1024 | 1.00 | 1024 | 1.00 |
| pythia-1b | 2048 | 16 | 0.9089 | 1946 | 0.95 | 2048 | 1.00 |
| pythia-1.4b | 2048 | 24 | 0.8635 | 1843 | 0.90 | 2048 | 1.00 |
| pythia-2.8b | 2560 | 32 | 0.8060 | 1792 | 0.70 | 2560 | 1.00 |
